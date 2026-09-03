"""Modal backend — run the probe on a serverless cloud GPU.

Ships this repo to Modal, runs the local flow (capture -> diagnose) on the requested GPU, and
returns the Observation dict. `config["gpu"]` doubles as the Modal GPU request (A10G / A100 / H100
/ L4 / T4). Needs `modal` + auth (`modal setup`); `modal` is imported lazily so this module imports
without it. The probe on Modal always runs on `cuda`.

Written against Modal 1.5. It can't run in a bare/local env — validate it live on Modal (billed to
the configured account), like the profiler's first A10G run.
"""
import os

# GPUs Modal exposes; config["gpu"] is passed straight through as the request.
_MODAL_GPUS = {"T4", "L4", "A10G", "A100", "A100-40GB", "A100-80GB", "H100", "H200", "L40S"}


def run(config):
    import modal

    gpu = config.get("gpu", "A10G")
    if gpu not in _MODAL_GPUS:
        raise ValueError(f"gpu {gpu!r} isn't a Modal GPU; known: {sorted(_MODAL_GPUS)}")

    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # the autotune/ root
    image = (modal.Image.debian_slim(python_version="3.12")
             .pip_install("torch", "psutil", "transformers", "datasets")
             .add_local_dir(repo, remote_path="/root/autotune"))
    app = modal.App("autotune-probe")

    @app.function(gpu=gpu, image=image, timeout=config.get("timeout", 600))
    def _probe(cfg):
        import sys
        sys.path.insert(0, "/root/autotune")
        from backends.local import run as local_run
        return local_run(cfg)                          # capture -> diagnose, on the GPU

    with app.run():
        return _probe.remote({**config, "device": "cuda"})   # the cloud probe always runs on cuda
