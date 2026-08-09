"""Portable autotune probe — the single on-box pipeline, backend-agnostic.

Runs one short ms-swift training probe on whatever GPU box it executes on, samples GPU/CPU
utilization, parses ms-swift's OWN output, and returns the structured Layer-1 result. No Modal,
no SSH here — just the probe itself, so both backends run identical code:

  - `modal_probe.py` imports `run_probe` and runs it inside a Modal GPU function.
  - `vm_probe.py` uploads this file to an SSH GPU VM and runs it as a script:
        python probe_runner.py config.json result.json
    (swift output streams to stdout; the JSON result is written to result.json)

The metrics come from swift's logs + nvidia-smi/psutil — nothing here is tied to a provider.
"""
import json
import os
import re
import subprocess
import sys
import threading
import time

# --- log parsing: corrected metric + memory (locked to swift's real format) -----
# Confirmed live: {'global_step/max_steps': '3/7', 'elapsed_time': '4s',
#                  'num_input_tokens_seen': 5518, 'memory(GiB)': '4.68', ...}
# global_step = OPTIMIZER step → samples/step = effective_batch (bs*ga). elapsed_time is
# humanized ('2s', '1m 30s'); num_input_tokens_seen is cumulative (unquoted int).
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
    """Steady-state tokens/s & samples/s (effective_batch numerator) over post-warmup steps,
    plus peak memory. ALWAYS returns samples_per_s/tokens_per_s keys — falling back to swift's
    end-of-run figures when per-step timing is too coarse (very fast steps log 0s deltas)."""
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


def _is_vision_model(model: str) -> bool:
    """Vision (image-in) model? Read the model's HF config on-box (transformers is present here).
    VLMs carry a nested vision_config; fall back to a name heuristic if the config can't be fetched."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model, trust_remote_code=True)
        return hasattr(cfg, "vision_config") or "vl" in getattr(cfg, "model_type", "").lower()
    except Exception:
        return any(t in model.lower() for t in ("-vl", "vl-", "vision", "llava"))


def _synthetic_image_dataset(path="/tmp/probe_data.jsonl"):
    """Tiny synthetic IMAGE dataset (dev, vision models) — vision tokens (image resolution) drive
    memory + throughput. Vary resolution per sample to exercise that (and padding_free)."""
    from PIL import Image
    os.makedirs("/tmp/imgs", exist_ok=True)
    with open(path, "w") as f:
        for i in range(128):
            side = 224 + (i % 6) * 48            # 224-464 px, varied (small enough to fit max_length)
            p = f"/tmp/imgs/{i}.png"
            Image.new("RGB", (side, side), (i % 255, 90, 160)).save(p)
            f.write(json.dumps({
                "messages": [{"role": "user", "content": "<image>What text is in this image?"},
                             {"role": "assistant", "content": "Sample OCR transcription."}],
                "images": [p],
            }) + "\n")
    return path


def _synthetic_text_dataset(path="/tmp/probe_data.jsonl"):
    """Tiny synthetic TEXT dataset (dev, text models) — varied-length messages to exercise the
    length distribution that packing / padding_free act on."""
    with open(path, "w") as f:
        for i in range(128):
            n = 20 + (i % 6) * 60                # varied length → uneven steps, like real data
            f.write(json.dumps({
                "messages": [{"role": "user", "content": "Summarize this: " + ("lorem ipsum " * n)},
                             {"role": "assistant", "content": "A concise summary of the text."}],
            }) + "\n")
    return path


def run_probe(config: dict) -> dict:
    """Run one probe and return the Layer-1 result dict. Identical on every backend.
    Reads knobs with defaults so any config version runs (only model + dataset are required)."""
    bs = config.get("per_device_train_batch_size", 1)
    ga = config.get("gradient_accumulation_steps", 1)
    eff = bs * ga
    attn = config.get("attn_impl", "flash_attn")

    # dev track: a tiny synthetic dataset avoids all hub/dataset-id flakiness — text or image to
    # match the model's modality. Real runs pass a hub id (any modality) straight through.
    ds = config["dataset"]
    if ds == "synthetic":
        ds = _synthetic_image_dataset() if _is_vision_model(config["model"]) else _synthetic_text_dataset()

    cmd = [
        "swift", "sft",
        "--model", config["model"],
        "--dataset", ds,
        "--lazy_tokenize", "true",                   # tokenize on the fly — don't pre-tokenize the whole set
        "--per_device_train_batch_size", str(bs),
        "--gradient_accumulation_steps", str(ga),
        "--padding_free", str(config.get("padding_free", False)).lower(),
        "--gradient_checkpointing", str(config.get("gradient_checkpointing", False)).lower(),
        "--dataloader_num_workers", str(config.get("dataloader_num_workers", 1)),
        "--max_length", str(config.get("max_length", 8192)),   # universal cap; leaves room for vision tokens
        "--truncation_strategy", "delete",                      # drop any over-length sample, don't crash
        "--max_steps", str(config.get("max_steps", 7)),
        "--logging_steps", "1",                      # required for per-step parsing
        "--attn_impl", attn,
        "--include_num_input_tokens_seen", "true",   # emits num_input_tokens_seen → tokens/s
        "--output_dir", "/tmp/out",
        "--report_to", "none",
    ]
    # v1 throughput knobs — only passed when set (avoid handing swift unrecognized flags on
    # non-VLM models / older versions). packing = sample concatenation; vit_gc = ViT checkpointing.
    for flag in ("packing", "vit_gradient_checkpointing"):
        if config.get(flag):
            cmd += [f"--{flag}", "true"]
    # USE_HF=1 resolves the model on HF (ModelScope's slow download was the timeout). Respect any
    # CUDA_VISIBLE_DEVICES already set by the backend (e.g. a multi-GPU VM); default to GPU 0.
    env = {**os.environ,
           "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
           "USE_HF": os.environ.get("USE_HF", "1")}

    # background GPU + CPU utilization sampler — feeds regime detection: gpu ~100% => compute-bound;
    # gpu low + cpu pinned ~100% => preprocessing wall (workers won't help) vs cpu headroom (they will).
    gpu, cpu, stop = [], [], threading.Event()
    try:
        import psutil
        psutil.cpu_percent()                         # prime the first (throwaway) reading
    except Exception:
        psutil = None

    def _sample():
        while not stop.is_set():
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2)
                head = r.stdout.strip().splitlines()
                if head and head[0].strip().isdigit():
                    gpu.append(int(head[0].strip()))
            except Exception:
                pass
            if psutil is not None:
                try:
                    cpu.append(psutil.cpu_percent())
                except Exception:
                    pass
            time.sleep(0.5)

    sampler = threading.Thread(target=_sample, daemon=True)
    sampler.start()

    # stream swift output live (visible in the backend's logs) while collecting it
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
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

    error_detail = None                               # the relevant failure message for the LLM
    if status != "ok":
        if status == "oom":
            error_detail = next((ln.strip() for ln in log.splitlines()
                                 if "out of memory" in ln.lower()), None)
        if error_detail is None:                      # fall back to the last non-empty line (the exception)
            tail = [ln.strip() for ln in log.splitlines() if ln.strip()]
            error_detail = tail[-1] if tail else None

    gpu_mem_gb, gpu_name = None, None                 # total GPU memory + SKU — informs the OOM ceiling
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.total,name", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        mem, _, name = r.stdout.strip().splitlines()[0].partition(",")
        gpu_mem_gb = round(int(mem.strip()) / 1024, 1)
        gpu_name = name.strip() or None
    except Exception:
        pass

    return {
        "status": status,
        "returncode": rc,
        "config": {"model": config["model"], "bs": bs, "ga": ga,
                   "effective_batch": eff, "attn_impl": attn},
        **parse_metrics(log, eff),
        "gpu_mem_gb": gpu_mem_gb,
        "gpu_name": gpu_name,
        "gpu_util_max": max(gpu) if gpu else None,
        "gpu_util_mean": round(sum(gpu) / len(gpu), 1) if gpu else None,
        "cpu_util_max": max(cpu) if cpu else None,
        "cpu_util_mean": round(sum(cpu) / len(cpu), 1) if cpu else None,
        "error_detail": error_detail,
        "step_lines": [ln for ln in log.splitlines() if "global_step/max_steps" in ln],
        "log_tail": log[-3000:],
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
