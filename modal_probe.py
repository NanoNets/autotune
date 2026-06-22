"""Dev-track autotune probe on Modal.

Runs a short ms-swift training probe and returns a structured Layer-1 result
(corrected metric + peak memory + gpu util) the agent can read without parsing logs.

Built/validated on a small PUBLIC model + dataset — no work access needed. To move to the
real probe, swap `model`/`dataset` in CONFIG and set attn_impl='flash_attn' (which also
needs flash-attn added to the image — see image note). Run:  modal run modal_probe.py
"""
import modal

app = modal.App("autotune-probe")

# Public defaults so this runs with a personal HF token only. Swap model/dataset for the
# real run; the rest of the machinery is identical. attn_impl='flash_attn' needs flash-attn
# in the image (the next build step).
CONFIG = {
    "model": "Qwen/Qwen2.5-VL-3B-Instruct",   # mature vision dev model (flash-attn works, no fla).
                                              # Real Qwen3.5-0.8B needs the prod env's fla/triton pins.
    "dataset": "synthetic",              # dev track: local JSONL (no hub dependency); real runs pass a hub id
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,   # ga>1: each global_step is one optimizer step (bs*ga samples)
    "max_steps": 7,
    "attn_impl": "flash_attn",           # required for padding_free on this model
}

# Default image: sdpa attention, no flash-attn (fast to build; fine for the dev model + the loop).
image = (
    modal.Image.debian_slim(python_version="3.11")
    # Qwen3.5-0.8B is a vision + linear-attention model -> VL stack + flash-linear-attention
    # (its padding_free / single-sequence path needs fla).
    .pip_install("torch", "torchvision", "ms-swift", "modelscope", "qwen_vl_utils", "decord",
                 "flash-linear-attention", "psutil")
)

# Flash-attn image — UNVERIFIED (needs a deploy to validate the build; ~10-20min first compile).
# Only needed for the real model and padding_free runs. To use: set image=flash_image in the
# @app.function decorator below and CONFIG["attn_impl"]="flash_attn".
flash_image = (
    modal.Image.debian_slim(python_version="3.11")
    # Pinned torch (cu124) + a PREBUILT flash-attn wheel matched to it (cu12, torch2.5, cp311,
    # cxx11abiFALSE = the ABI torch pip wheels use). No compilation -> no ABI/compiler mismatch.
    # torch 2.6 (ms-swift needs FSDP2 from it) + the matching prebuilt flash-attn wheel.
    .pip_install("torch==2.6.0", "torchvision==0.21.0",
                 index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("ms-swift", "modelscope", "qwen_vl_utils", "decord", "flash-linear-attention", "psutil")
    .pip_install("https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
                 "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")
)

hf_cache = modal.Volume.from_name("autotune-hf-cache", create_if_missing=True)


# --- log parsing: corrected metric + memory (locked to swift's real format) -----
# Confirmed live: {'global_step/max_steps': '3/7', 'elapsed_time': '4s',
#                  'num_input_tokens_seen': 5518, 'memory(GiB)': '4.68', ...}
# global_step = OPTIMIZER step → samples/step = effective_batch (bs*ga). elapsed_time is
# humanized ('2s', '1m 30s'); num_input_tokens_seen is cumulative (unquoted int).
import re

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


@app.function(
    image=flash_image,                # padding_free on Qwen3.5 requires flash-attn
    gpu="A10G",
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("hf-token")],
    timeout=1800,                     # probe RUN ceiling (not the image build, which has its own budget)
)
def probe(config: dict):
    import os, subprocess, threading, time, json

    bs = config["per_device_train_batch_size"]
    ga = config["gradient_accumulation_steps"]
    eff = bs * ga

    # dev track: a tiny synthetic dataset avoids all hub/dataset-id flakiness. Real runs pass a hub id.
    ds = config["dataset"]
    if ds == "synthetic":
        # Synthetic IMAGE data — production is OCR, so vision tokens (image resolution) drive
        # memory + throughput. Vary resolution per sample to exercise that (and padding_free).
        from PIL import Image
        ds = "/tmp/probe_data.jsonl"
        os.makedirs("/tmp/imgs", exist_ok=True)
        with open(ds, "w") as f:
            for i in range(128):
                side = 224 + (i % 6) * 48        # 224-464 px, varied (small enough to fit max_length)
                path = f"/tmp/imgs/{i}.png"
                Image.new("RGB", (side, side), (i % 255, 90, 160)).save(path)
                f.write(json.dumps({
                    "messages": [{"role": "user", "content": "<image>What text is in this image?"},
                                 {"role": "assistant", "content": "Sample OCR transcription."}],
                    "images": [path],
                }) + "\n")

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
        "--max_length", str(config.get("max_length", 8192)),   # room for vision tokens (OCR images)
        "--truncation_strategy", "delete",                      # drop any over-length sample, don't crash
        "--max_steps", str(config["max_steps"]),
        "--logging_steps", "1",                      # required for per-step parsing
        "--attn_impl", config["attn_impl"],
        "--include_num_input_tokens_seen", "true",   # emits num_input_tokens_seen → tokens/s
        "--output_dir", "/tmp/out",
        "--report_to", "none",
    ]
    # v1 throughput knobs — only passed when set (avoid handing swift unrecognized flags on
    # non-VLM models / older versions). packing = sample concatenation; vit_gc = ViT checkpointing.
    for flag in ("packing", "vit_gradient_checkpointing"):
        if config.get(flag):
            cmd += [f"--{flag}", "true"]
    # Model from HuggingFace (fast, and caches to the mounted volume); dataset is local JSONL.
    # USE_HF=1 resolves the model on HF — ModelScope's 250kB/s model download was the timeout.
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "USE_HF": "1"}

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

    # stream swift output live (visible in `modal run` + Modal logs) while collecting it
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

    gpu_mem_gb = None                                 # total GPU memory — PredictivePolicy needs it
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=5)
        gpu_mem_gb = round(int(r.stdout.strip().splitlines()[0]) / 1024, 1)
    except Exception:
        pass

    return {
        "status": status,
        "returncode": rc,
        "config": {"model": config["model"], "bs": bs, "ga": ga,
                   "effective_batch": eff, "attn_impl": config["attn_impl"]},
        **parse_metrics(log, eff),
        "gpu_mem_gb": gpu_mem_gb,
        "gpu_util_max": max(gpu) if gpu else None,
        "gpu_util_mean": round(sum(gpu) / len(gpu), 1) if gpu else None,
        "cpu_util_max": max(cpu) if cpu else None,
        "cpu_util_mean": round(sum(cpu) / len(cpu), 1) if cpu else None,
        "error_detail": error_detail,
        "step_lines": [ln for ln in log.splitlines() if "global_step/max_steps" in ln],
        "log_tail": log[-3000:],
    }


@app.local_entrypoint()
def main():
    out = probe.remote(CONFIG)
    print("status:", out["status"], "| returncode:", out["returncode"])
    print("config:", out["config"])
    print("\n--- Layer-1 metrics ---")
    for k in ("samples_per_s", "tokens_per_s", "summary_samples_per_s",
              "peak_mem_gb", "gpu_util_max", "gpu_util_mean",
              "cpu_util_max", "cpu_util_mean", "steps_measured"):
        if k in out:
            print(f"  {k}: {out[k]}")
    if out.get("error_detail"):
        print("  error_detail:", out["error_detail"])
    if out.get("samples_per_s") is None:
        print("\n(no steady-state metrics parsed — log tail:)")
        print(out["log_tail"])
