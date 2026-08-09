"""Modal backend for the autotune probe.

Thin wrapper: it ships `probe_runner.py` to a Modal serverless GPU and runs the shared
`run_probe()` there. The probe pipeline itself lives in `probe_runner.py` (same code the SSH
backend runs), so Modal and an SSH VM measure identically.

Usually invoked via the backend dispatcher (`AUTOTUNE_BACKEND=modal python probe.py`); can also be
run standalone for a quick Modal-only check:  modal run modal_probe.py
"""
import json
import os

import modal

app = modal.App("autotune-probe")

# Infra (GPU SKU + image) follows config.json / env so any hardware + attention impl works. Modal
# fixes these on the function decorator, so read them at import. GPU: config["gpu"] > MODAL_GPU env
# > A10G default. Image: flash-attn build when attn_impl=flash_attn, else the fast base build.
_cfg = {}
try:
    _cfg = json.load(open("config.json"))
except Exception:
    pass
GPU = _cfg.get("gpu") or os.environ.get("MODAL_GPU", "A10G")
_ATTN = _cfg.get("attn_impl", "flash_attn")

# Base image: sdpa/default attention, fast to build. Deps are a superset that covers text, VLM
# (qwen_vl_utils/decord), and linear-attention (flash-linear-attention) models — unused deps are
# harmless, so one image serves any model. `add_local_python_source` bakes probe_runner in.
base_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "torchvision", "ms-swift", "modelscope", "qwen_vl_utils", "decord",
                 "flash-linear-attention", "psutil", "pillow")
    .add_local_python_source("probe_runner")
)

# Flash-attn image: base deps + a PREBUILT flash-attn wheel matched to a pinned torch (cu12, torch2.6,
# cp311, cxx11abiFALSE — the ABI torch's pip wheels use). No compile -> no ABI mismatch. First build
# ~10-20min. Used when attn_impl=flash_attn.
flash_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch==2.6.0", "torchvision==0.21.0",
                 index_url="https://download.pytorch.org/whl/cu124")
    .pip_install("ms-swift", "modelscope", "qwen_vl_utils", "decord", "flash-linear-attention",
                 "psutil", "pillow")
    .pip_install("https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/"
                 "flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp311-cp311-linux_x86_64.whl")
    .add_local_python_source("probe_runner")
)

image = flash_image if _ATTN == "flash_attn" else base_image
hf_cache = modal.Volume.from_name("autotune-hf-cache", create_if_missing=True)


@app.function(
    image=image,
    gpu=GPU,
    volumes={"/root/.cache/huggingface": hf_cache},
    secrets=[modal.Secret.from_name("hf-token")],
    timeout=1800,                     # probe RUN ceiling (not the image build, which has its own budget)
)
def probe(config: dict):
    from probe_runner import run_probe
    return run_probe(config)


@app.local_entrypoint()
def main():
    # Standalone Modal-only check: runs config.json once and prints the Layer-1 metrics.
    config = json.load(open("config.json"))
    out = probe.remote(config)
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
