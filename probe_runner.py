"""Portable autotune probe — backend-agnostic.

Runs one short ms-swift training probe on the local GPU, samples GPU/CPU utilization, parses
ms-swift's own log output, and returns a structured result. The same code runs on every backend:

  - `modal_probe.py` imports `run_probe` and runs it inside a Modal GPU function.
  - `vm_probe.py` uploads this file to an SSH GPU VM and runs it as a script:
        python probe_runner.py config.json result.json
    (swift output streams to stdout; the JSON result is written to result.json)

Metrics come from swift's logs plus nvidia-smi/psutil — nothing here is tied to a provider.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time

# --- log parsing: metrics + memory from swift's per-step log lines, e.g.: -------
#   {'global_step/max_steps': '3/7', 'elapsed_time': '4s',
#    'num_input_tokens_seen': 5518, 'memory(GiB)': '4.68', ...}
# global_step is the optimizer step; elapsed_time is humanized ('2s', '1m 30s');
# num_input_tokens_seen is cumulative.
_STEP_RE = re.compile(r"'global_step/max_steps':\s*'(\d+)/\d+'")
_ELAPSED_RE = re.compile(r"'elapsed_time':\s*'([^']+)'")
_TOK_RE = re.compile(r"'num_input_tokens_seen':\s*(\d+)")
_MEM_RE = re.compile(r"'memory\(GiB\)':\s*'([\d.]+)'")
_SUMMARY_SPS_RE = re.compile(r"'train_samples_per_second':\s*'?([\d.]+)")
_RUNTIME_RE = re.compile(r"'train_runtime':\s*'?([\d.]+)")


def _to_seconds(t):
    t = t.strip()
    if ":" in t:                                   # H:MM:SS / MM:SS fallback
        parts = [float(p) for p in t.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0.0)
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    total = 0.0                                    # humanized '1h 2m 3s'
    for val, unit in re.findall(r"([\d.]+)\s*([hms])", t):
        total += float(val) * {"h": 3600, "m": 60, "s": 1}[unit]
    return total


def parse_metrics(text, effective_batch, warmup=2):
    """Steady-state tokens/s and samples/s over post-warmup steps, plus peak memory.
    Falls back to swift's end-of-run figures when per-step timing is too coarse to measure."""
    per_step = {}  # global_step -> (elapsed_s, tokens_seen)
    for line in text.splitlines():
        ms, me, mt = _STEP_RE.search(line), _ELAPSED_RE.search(line), _TOK_RE.search(line)
        if ms and me and mt:
            per_step[int(ms.group(1))] = (_to_seconds(me.group(1)), int(mt.group(1)))

    summary = _SUMMARY_SPS_RE.search(text)
    runtime = _RUNTIME_RE.search(text)
    mems = [float(m) for m in _MEM_RE.findall(text)]
    toks = [int(t) for t in _TOK_RE.findall(text)]
    out = {
        "effective_batch": effective_batch,
        "peak_mem_gb": max(mems) if mems else None,
        "summary_samples_per_s": float(summary.group(1)) if summary else None,
        "samples_per_s": None,
        "tokens_per_s": None,
        "steps_measured": 0,
    }
    post = sorted((gs, el, tok) for gs, (el, tok) in per_step.items() if gs > warmup)
    if len(post) >= 2 and post[-1][1] - post[0][1] > 0:           # steady-state (preferred)
        dt = post[-1][1] - post[0][1]
        out["tokens_per_s"] = round((post[-1][2] - post[0][2]) / dt, 1)
        out["samples_per_s"] = round(effective_batch * (post[-1][0] - post[0][0]) / dt, 4)
        out["steps_measured"] = post[-1][0] - post[0][0]
    elif runtime and float(runtime.group(1)) > 0:                # fallback: swift's own figures
        out["samples_per_s"] = out["summary_samples_per_s"]
        if toks:
            out["tokens_per_s"] = round(max(toks) / float(runtime.group(1)), 1)
    return out
# ------------------------------------------------------------------------------


def _synthetic_dataset(path="/tmp/probe_data.jsonl"):
    """Tiny synthetic image dataset for quick local runs. Resolution varies per sample so vision-token
    count — the main driver of memory and throughput for this OCR workload — varies too."""
    from PIL import Image
    os.makedirs("/tmp/imgs", exist_ok=True)
    with open(path, "w") as f:
        for i in range(128):
            side = 224 + (i % 6) * 48            # 224-464 px
            p = f"/tmp/imgs/{i}.png"
            Image.new("RGB", (side, side), (i % 255, 90, 160)).save(p)
            f.write(json.dumps({
                "messages": [{"role": "user", "content": "<image>What text is in this image?"},
                             {"role": "assistant", "content": "Sample OCR transcription."}],
                "images": [p],
            }) + "\n")
    return path


def run_probe(config: dict) -> dict:
    """Run one probe and return the result dict. Identical on every backend."""
    # Levers: arbitrary `swift sft` CLI args, passed through unmodified. Harness-managed args
    # (model/dataset/max_length/ga/plumbing) stay protected because `fixed` below wins on overlap.
    levers = dict(config.get("levers", {}))

    bs = int(levers.get("per_device_train_batch_size", 1))   # used to derive ga
    eff = config["effective_batch"]                          # fixed input
    n_gpu = int(config.get("hardware_number", 1))            # DDP world_size
    attn = config.get("attn_impl", "flash_attn")             # fixed input (also selects the Modal image)
    denom = bs * n_gpu
    if denom == 0 or eff % denom:                            # eff = bs * ga * world_size → ga must be a +int
        return {"status": "error",
                "config": {"model": config["model"], "per_device_train_batch_size": bs, "attn_impl": attn,
                           "world_size": n_gpu, "effective_batch": eff, "gradient_accumulation_steps": None},
                "error_detail": f"effective_batch {eff} not divisible by bs*hardware_number "
                                f"({bs}*{n_gpu}={denom}) — pick a bs that divides it"}
    ga = eff // denom

    # Dataset resolution: an inlined local file (dataset_file, from probe.py) is materialized here;
    # 'synthetic' generates a local dev file; anything else is an HF/MS id swift resolves on the box.
    ds = config["dataset"]
    if config.get("dataset_file") is not None:
        ds = f"/tmp/{os.path.basename(ds)}"
        with open(ds, "w") as f:
            f.write(config["dataset_file"])
    elif ds == "synthetic":
        ds = _synthetic_dataset()

    levers.setdefault("per_device_train_batch_size", bs)    # default, overridable via levers

    # Fixed plumbing always set here; wins over any lever. Required for correct measurement, or
    # defaults that are always a win. ga is derived; bs stays in levers; attn is an input.
    fixed = {
        "model": config["model"], "dataset": ds,
        "max_length": int(config.get("max_length", 8192)),  # task's token cap
        "attn_impl": attn,                                  # fixed input, not a lever
        "gradient_accumulation_steps": ga,
        "max_steps": int(config.get("max_steps", 7)),       # probe mechanic
        "lazy_tokenize": "true",
        "truncation_strategy": "delete",                    # drop over-length samples, don't crash
        "logging_steps": "1",                               # required for per-step parsing
        "include_num_input_tokens_seen": "true",            # emits num_input_tokens_seen → tokens/s
        "dataloader_pin_memory": "true",                    # always a win
        "dataloader_persistent_workers": "true",            # always a win
        "output_dir": "/tmp/out", "report_to": "none",
    }
    args = {**levers, **fixed}                              # fixed wins on any overlap
    cmd = ["swift", "sft"]
    for k, v in args.items():
        cmd += [f"--{k}", str(v).lower() if isinstance(v, bool) else str(v)]
    # Multi-GPU: expose N GPUs and launch N DDP processes (NPROC_PER_NODE). USE_HF=1 resolves the
    # model on HF rather than ModelScope.
    # num_input_tokens_seen is the global (all-reduced) token count across ranks, so the parsed
    # tokens/s is already a whole-job figure and needs no ×N correction.
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES") or ",".join(str(i) for i in range(n_gpu))
    env = {**os.environ,
           "CUDA_VISIBLE_DEVICES": cvd,
           "NPROC_PER_NODE": str(n_gpu),               # swift → torchrun with N processes (DDP)
           "USE_HF": os.environ.get("USE_HF", "1")}

    # Launch swift first so the sampler can watch its process tree by pid.
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)

    # Background sampler: GPU util (nvidia-smi) + CPU util of the SWIFT PROCESS TREE, normalized to
    # the cores allocated to this container/VM. Watching the tree (not the whole system) is what makes
    # cpu_util meaningful in a container, where system-wide % is diluted by the host's many cores.
    # Distinguishes compute-bound (GPU ~100%) from data-bound (GPU low; CPU headroom → workers help,
    # CPU pinned → preprocessing wall).
    gpu, cpu, stop = [], [], threading.Event()
    try:
        import psutil
    except Exception:
        psutil = None
    ncores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)

    def _sample():
        seen = {}                                       # pid -> primed Process (cpu_percent needs a prior reading)
        while not stop.is_set():
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2)
                vals = [int(x.strip()) for x in r.stdout.strip().splitlines() if x.strip().isdigit()]
                if vals:
                    gpu.append(round(sum(vals) / len(vals)))    # mean util across all N GPUs
            except Exception:
                pass
            if psutil is not None:
                try:
                    root = psutil.Process(proc.pid)
                    total = 0.0                                  # sum of per-proc % (each up to 100×cores)
                    for c in [root, *root.children(recursive=True)]:
                        o = seen.get(c.pid)
                        if o is None:
                            c.cpu_percent(); seen[c.pid] = c     # prime; contributes 0 this round
                        else:
                            total += o.cpu_percent()
                    cpu.append(round(total / ncores, 1))         # % of allocated cores (0–100)
                except Exception:
                    pass
            time.sleep(0.5)

    sampler = threading.Thread(target=_sample, daemon=True)
    sampler.start()

    # stream swift output live while collecting it
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    stop.set()
    sampler.join(timeout=2)

    log = "".join(lines)
    rc = proc.returncode
    status = "ok" if rc == 0 else ("oom" if "out of memory" in log.lower() else "error")

    error_detail = None                               # the relevant failure message
    if status != "ok":
        if status == "oom":
            error_detail = next((ln.strip() for ln in log.splitlines()
                                 if "out of memory" in ln.lower()), None)
        if error_detail is None:                      # fall back to the last non-empty line (the exception)
            tail = [ln.strip() for ln in log.splitlines() if ln.strip()]
            error_detail = tail[-1] if tail else None

    gpu_mem_gb = None                                 # total GPU memory — informs the OOM ceiling
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        gpu_mem_gb = round(int(r.stdout.strip().splitlines()[0]) / 1024, 1)
    except Exception:
        pass

    return {
        "status": status,
        "returncode": rc,
        "config": {**levers, "model": config["model"], "world_size": n_gpu, "effective_batch": eff,
                   "attn_impl": attn, "gradient_accumulation_steps": ga},   # harness values win (match the cmd)
        **parse_metrics(log, eff),
        "gpu_mem_gb": gpu_mem_gb,
        "gpu_util_max": max(gpu) if gpu else None,
        "gpu_util_mean": round(sum(gpu) / len(gpu), 1) if gpu else None,
        "cpu_util_max": max(cpu) if cpu else None,
        "cpu_util_mean": round(sum(cpu) / len(cpu), 1) if cpu else None,
        "error_detail": error_detail,
        "step_lines": [ln for ln in log.splitlines() if "global_step/max_steps" in ln],
        "log_tail": log[-20000:],     # tail of the raw swift output (for debugging; not persisted)
    }


if __name__ == "__main__":
    # SSH/VM entrypoint: read a config json, run the probe, write the result json.
    cfg = json.load(open(sys.argv[1]))
    result = run_probe(cfg)
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    if out_path:
        with open(out_path, "w") as f:
            json.dump(result, f)
    else:
        print("\n=== PROBE RESULT ===")
        print(json.dumps(result))
