"""Modal backend for the autotune probe.

Ships `probe_runner.py` to a Modal serverless GPU and runs the shared `run_probe()` there — the
same pipeline the SSH backend runs, so both backends measure identically.

Usually invoked via the dispatcher (`python probe.py` with backend=modal); can also run
standalone:  modal run modal_probe.py
"""
import json
import os

import modal

app = modal.App("autotune-probe")

# The decorator below needs the GPU + image at IMPORT time, so they're resolved here from config —
# the SAME `config.json` inputs a user edits. `python probe.py` exports them as env vars first (so a
# custom config path works); standalone `modal run modal_probe.py` falls back to reading config.json.
try:
    _CONFIG = json.load(open("config.json"))
except Exception:
    _CONFIG = {}

def _setting(key, env, default):
    return os.environ.get(env) or _CONFIG.get(key, default)

GPU = _setting("hardware", "AUTOTUNE_GPU", "A10G")
GPU_COUNT = int(_setting("hardware_number", "AUTOTUNE_GPU_COUNT", 1))
GPU_SPEC = f"{GPU}:{GPU_COUNT}" if GPU_COUNT > 1 else GPU        # Modal spec, "TYPE:N" for N>1
ATTN = _setting("attn_impl", "AUTOTUNE_ATTN", "flash_attn")

# Default image: sdpa attention, no flash-attn (fast to build; fine for quick iteration).
# `add_local_python_source` bakes probe_runner.py into the image so the function can import it.
image = (
    modal.Image.debian_slim(python_version="3.11")
    # Qwen3.5-0.8B is a vision + linear-attention model -> VL stack + flash-linear-attention
    # (its padding_free / single-sequence path needs fla).
    .pip_install("torch", "torchvision", "ms-swift", "modelscope", "qwen_vl_utils", "decord",
                 "flash-linear-attention", "psutil")
    .add_local_python_source("probe_runner")
)

# Flash-attn image — only needed for the real model and padding_free runs. First build just
# downloads the large CUDA wheels (prebuilt flash-attn, no compilation) — slow once, then cached.
flash_image = (
    modal.Image.debian_slim(python_version="3.11")
    # Pinned torch (cu124) + a PREBUILT flash-attn wheel matched to it (cu12, torch2.6, cp311,
    # cxx11abiFALSE = the ABI torch pip wheels use). No compilation -> no ABI/compiler mismatch.
    .pip_install("torch==2.6.0", "torchvision==0.21.0",
                 index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("ms-swift", "modelscope", "qwen_vl_utils", "decord", "flash-linear-attention", "psutil")
    .pip_install("https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
                 "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")
    .add_local_python_source("probe_runner")
)

hf_cache = modal.Volume.from_name("autotune-hf-cache", create_if_missing=True)

# Image follows config `attn_impl`: flash_attn → the prebuilt flash-attn image, anything else → the
# fast sdpa image. Note: padding_free also needs flash-attn, so only use a non-flash attn_impl
# for an all-sdpa session.
IMAGE = flash_image if ATTN == "flash_attn" else image


@app.function(
    image=IMAGE,                      # from config `attn_impl` (AUTOTUNE_ATTN)
    gpu=GPU_SPEC,                      # from config `hardware`/`hardware_number`, e.g. "A100:4"
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("hf-token")],
    timeout=1800,                     # probe run ceiling (not the image build)
)
def probe(config: dict):
    from probe_runner import run_probe
    return run_probe(config)


@app.function(
    image=IMAGE,                      # SAME image as the probe, so the scan reports the probe's libs
    gpu=GPU_SPEC,                      # GPU present so the scan can read its name + compute capability
    timeout=300,                      # no training — just imports + nvidia-smi
)
def scan(check_libs: list = None):
    from probe_runner import scan_env
    return scan_env(check_libs)


@app.local_entrypoint()
def main():
    # Standalone check: run config.json once and print the metrics.
    config = json.load(open("config.json"))
    out = probe.remote(config)
    print("status:", out["status"], "| returncode:", out["returncode"])
    print("config:", out["config"])
    print("\n--- metrics ---")
    for grp in ("throughput", "memory", "utilization"):
        if out.get(grp):
            print(f"  {grp}: {out[grp]}")
    if out.get("error_detail"):
        print("  error_detail:", out["error_detail"])
    if (out.get("throughput") or {}).get("samples_per_s") is None:
        print("\n(no steady-state metrics parsed — log tail:)")
        print(out["log_tail"])
