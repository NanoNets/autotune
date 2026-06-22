"""Modal backend for the autotune probe.

Thin wrapper: it ships `probe_runner.py` to a Modal serverless GPU and runs the shared
`run_probe()` there. The probe pipeline itself lives in `probe_runner.py` (same code the SSH
backend runs), so Modal and an SSH VM measure identically.

Usually invoked via the backend dispatcher (`AUTOTUNE_BACKEND=modal python probe.py`); can also be
run standalone for a quick Modal-only check:  modal run modal_probe.py
"""
import json

import modal

app = modal.App("autotune-probe")

# Default image: sdpa attention, no flash-attn (fast to build; fine for the dev model + the loop).
# `add_local_python_source` bakes probe_runner.py into the image so the function can import it.
image = (
    modal.Image.debian_slim(python_version="3.11")
    # Qwen3.5-0.8B is a vision + linear-attention model -> VL stack + flash-linear-attention
    # (its padding_free / single-sequence path needs fla).
    .pip_install("torch", "torchvision", "ms-swift", "modelscope", "qwen_vl_utils", "decord",
                 "flash-linear-attention", "psutil")
    .add_local_python_source("probe_runner")
)

# Flash-attn image — UNVERIFIED (needs a deploy to validate the build; ~10-20min first compile).
# Only needed for the real model and padding_free runs.
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


@app.function(
    image=flash_image,                # padding_free on Qwen3.5 requires flash-attn
    gpu="A10G",
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
