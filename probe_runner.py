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


# --- utilization windowing: align the util time-series with the measured steps -----
IDLE_GPU_PCT = 20          # a GPU sample below this counts as "idle" (GPU waiting, not computing)


def _pct(xs, q):
    """qth percentile (0-100) of a SORTED list, linear interpolation."""
    if not xs:
        return None
    i = (len(xs) - 1) * q / 100.0
    lo, hi = int(i), min(int(i) + 1, len(xs) - 1)
    return round(xs[lo] + (xs[hi] - xs[lo]) * (i - lo), 1)


def _util_stats(samples, idle=False):
    """p10/p50/p90/max over the samples; idle_frac (share < IDLE_GPU_PCT) only when idle=True (GPU)."""
    vals = sorted(samples)
    s = {"p10": _pct(vals, 10), "p50": _pct(vals, 50), "p90": _pct(vals, 90),
         "max": max(vals) if vals else None}
    if idle:                              # GPU only: how often it sat waiting; meaningless for CPU
        s["idle_frac"] = round(sum(v < IDLE_GPU_PCT for v in vals) / len(vals), 3) if vals else None
    return s


def _compute_util(gpu, cpu, step_times, warmup=2):
    """Window the (timestamp, value) util series to the measured steps (gs > warmup — the SAME window
    as tokens/s) and summarize. Both series live on our wall clock: util samples carry time.time(),
    step boundaries are stamped when each step's log line arrives. Returns None when there's no valid
    step window (OOM/crash/too-fast) or no in-window samples."""
    steps = sorted((gs, ts) for gs, ts in step_times if gs > warmup)
    if len(steps) < 2:
        return None
    t0, t1 = steps[0][1], steps[-1][1]
    g = [v for ts, v in gpu if t0 <= ts <= t1]
    c = [v for ts, v in cpu if t0 <= ts <= t1]
    if not g and not c:
        return None
    return {"gpu": _util_stats(g, idle=True), "cpu": _util_stats(c),
            "window_steps": [steps[0][0], steps[-1][0]], "n_samples": len(g)}
# ------------------------------------------------------------------------------


def scan_env() -> dict:
    """No-training capability scan: versions, GPU, and which accel libs are importable. Run once per
    session before the seed so the agent can skip env-unsupported kernel/compile levers.
    LIMIT: 'importable' != 'has a kernel for this arch' (liger imported yet had no Qwen3.5 kernels) —
    this kills missing-lib dead-ends, not present-but-no-kernel ones."""
    import importlib.util

    def have(mod):
        try:
            return importlib.util.find_spec(mod) is not None
        except Exception:
            return False

    env = {"libs": {m: have(m) for m in           # fla = flash-linear-attention (the linear-attn fast path)
                    ("causal_conv1d", "liger_kernel", "flash_attn", "fla")}}
    try:
        import torch
        env["torch"] = torch.__version__
        env["torch_cuda"] = torch.version.cuda    # CUDA toolkit torch was built against
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
            env["gpu_name"] = torch.cuda.get_device_name(0)
            env["compute_capability"] = f"{cc[0]}.{cc[1]}"
            env["gpu_count"] = torch.cuda.device_count()
    except Exception as e:
        env["torch_error"] = str(e)
    try:                                          # system driver (separate from torch's toolkit)
        r = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=5)
        out = r.stdout.strip().splitlines()
        env["driver_version"] = out[0].strip() if out else None
    except Exception:
        pass
    return env


# --- data-wait plugin: measures the fraction of step time the loop blocks on next(batch) ----------
# The authoritative data-starvation signal. Written to a temp file on the backend and loaded via
# swift's `--external_plugins` when config['wait'] is set. It times EVERY DataLoader.__next__ (with
# grad-accum the ga fetches are interleaved inside a step, so timing between steps would miss them)
# and prints per-optimizer-step totals; probe_runner windows them. General across model/dataset/ga/
# backend; rank-0 only; fully guarded so it can never crash the training run.
_WAIT_RE = re.compile(r"AUTOTUNE_WAIT step=(\d+) wait_s=([\d.]+) step_s=([\d.]+)")
_WAIT_PLUGIN = r'''
import time
try:
    import torch.utils.data as _tud
    _wait = {"blocked": 0.0}                       # cumulative seconds blocked on next(batch)
    _orig = _tud.DataLoader.__iter__
    def _timed(self):
        it = _orig(self)
        class _T:
            def __iter__(self_): return self_
            def __next__(self_):
                t = time.perf_counter()
                try:
                    return it.__next__()
                finally:
                    _wait["blocked"] += time.perf_counter() - t
        return _T()
    _tud.DataLoader.__iter__ = _timed              # time every batch fetch, any dataloader

    from swift.callbacks import callbacks_map, TrainerCallback

    class DataWaitCallback(TrainerCallback):
        def __init__(self, args, trainer):
            try: super().__init__(args, trainer)
            except Exception: pass
            self._t0, self._w0 = None, 0.0
        def on_step_begin(self, args, state, control, **kw):
            self._t0, self._w0 = time.perf_counter(), _wait["blocked"]
        def on_step_end(self, args, state, control, **kw):
            if self._t0 is None or not getattr(state, "is_world_process_zero", True):
                return
            print("AUTOTUNE_WAIT step=%d wait_s=%.4f step_s=%.4f"
                  % (int(state.global_step), _wait["blocked"] - self._w0, time.perf_counter() - self._t0),
                  flush=True)

    callbacks_map["data_wait"] = DataWaitCallback
except Exception as _e:                            # never break training if the hook shifts
    print("AUTOTUNE_WAIT_ERROR %r" % (_e,), flush=True)
'''


def parse_wait(text, warmup=2):
    """Windowed data-wait fraction from the plugin's per-step lines (steps > warmup — same window as
    tokens/s). None if the plugin emitted nothing usable (not requested, unsupported, or errored)."""
    rows = [(int(s), float(w), float(st)) for s, w, st in _WAIT_RE.findall(text)]
    post = [(w, st) for s, w, st in rows if s > warmup and st > 0]
    if not post:
        return None
    tw, ts = sum(w for w, _ in post), sum(st for _, st in post)
    return round(tw / ts, 3) if ts > 0 else None
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
        "max_steps": int(config.get("max_steps", 7)),       # probe mechanic (data levers want more)
        # lazy_tokenize is the default win, but it's incompatible with upfront preprocessing: packing
        # needs eager tokenization, and dataset_num_proc only parallelizes an eager .map(). Either
        # one flips it off so the lever is actually exercised.
        "lazy_tokenize": "false" if (levers.get("packing") or levers.get("dataset_num_proc")) else "true",
        "truncation_strategy": "delete",                    # drop over-length samples, don't crash
        "logging_steps": "1",                               # required for per-step parsing
        "include_num_input_tokens_seen": "true",            # emits num_input_tokens_seen → tokens/s
        "dataloader_pin_memory": "true",                    # always a win
        "dataloader_persistent_workers": "true",            # always a win
        "output_dir": "/tmp/out", "report_to": "none",
    }
    # Optional: measure dataloader wait-time via our external plugin (config['wait']). Written to the
    # backend's /tmp and loaded by swift; the callback prints per-step wait we parse back. Opt-in so a
    # plugin/version issue can't affect ordinary probes.
    if config.get("wait"):
        try:
            plugin_path = "/tmp/autotune_wait_plugin.py"
            with open(plugin_path, "w") as f:
                f.write(_WAIT_PLUGIN)
            fixed["external_plugins"] = plugin_path         # swift imports it at startup
            fixed["callbacks"] = "data_wait"                # activates the registered callback
        except Exception:
            pass                                            # fall back to no wait-time; never block the probe

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
                    gpu.append((time.time(), round(sum(vals) / len(vals))))   # (ts, mean util across N GPUs)
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
                    cpu.append((time.time(), round(total / ncores, 1)))   # (ts, % of allocated cores)
                except Exception:
                    pass
            # NOTE: spawns nvidia-smi per tick; switch to pynvml for cheaper/finer sampling later.
            time.sleep(0.25)

    sampler = threading.Thread(target=_sample, daemon=True)
    sampler.start()

    # stream swift output live while collecting it; stamp each step-line arrival on our wall clock
    # (the other half of the alignment — util samples carry time.time() too).
    lines, step_times = [], []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
        ms = _STEP_RE.search(line)
        if ms:
            step_times.append((int(ms.group(1)), time.time()))
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

    warmup = int(config.get("warmup", 2))                # skip leading steps in BOTH windows (throughput + util)
    m = parse_metrics(log, eff, warmup=warmup)
    util = _compute_util(gpu, cpu, step_times, warmup=warmup)   # None when no measurable window
    if config.get("wait"):                               # authoritative data-starvation signal (opt-in)
        util = {**(util or {}), "data_wait_frac": parse_wait(log, warmup)}
    return {
        "status": status,
        "returncode": rc,
        "config": {**levers, "model": config["model"], "world_size": n_gpu, "effective_batch": eff,
                   "attn_impl": attn, "gradient_accumulation_steps": ga},   # harness values win (match the cmd)
        # grouped metric contract (throughput + memory + utilization)
        "throughput": {"tokens_per_s": m["tokens_per_s"], "samples_per_s": m["samples_per_s"],
                       "steps_measured": m["steps_measured"]},
        "memory": {"peak_mem_gb": m["peak_mem_gb"], "gpu_mem_gb": gpu_mem_gb},
        "utilization": util,
        "error_detail": error_detail,
        "step_lines": [ln for ln in log.splitlines() if "global_step/max_steps" in ln],
        "log_tail": log[-20000:],     # tail of the raw swift output (for debugging; not persisted)
    }


if __name__ == "__main__":
    # SSH/VM entrypoint. `--scan result.json` runs the capability scan; otherwise read a config json,
    # run the probe, and write the result json.
    if sys.argv[1] == "--scan":
        result = scan_env()
        out_path = sys.argv[2] if len(sys.argv) > 2 else None
    else:
        cfg = json.load(open(sys.argv[1]))
        result = run_probe(cfg)
        out_path = sys.argv[2] if len(sys.argv) > 2 else None
    if out_path:
        with open(out_path, "w") as f:
            json.dump(result, f)
    else:
        print("\n=== PROBE RESULT ===")
        print(json.dumps(result))
