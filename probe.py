"""Backend-agnostic probe entrypoint — the loop's single command.

Pick where the probe runs with the AUTOTUNE_BACKEND env var; the pipeline (and the JSON it
returns) is identical either way:

  AUTOTUNE_BACKEND=modal  python probe.py     # default — Modal serverless GPU
  AUTOTUNE_BACKEND=ssh    python probe.py     # an SSH GPU VM (paperspace / lambda are aliases)

Reads the config from config.json (or a path given as argv[1]), runs one probe, prints the
Layer-1 metrics, and writes the full result to probe_result.json. All secrets/connection details
come from env — see .env.example.
"""
import json
import os
import sys

SSH_ALIASES = {"ssh", "vm", "paperspace", "lambda", "lambda_labs", "lambdalabs"}


def run_probe(config: dict) -> dict:
    """Dispatch one probe to the selected backend and return its Layer-1 result."""
    backend = os.environ.get("AUTOTUNE_BACKEND", "modal").strip().lower()
    if backend in SSH_ALIASES:
        import vm_probe
        return vm_probe.run_probe(config)
    if backend == "modal":
        import modal_probe
        with modal_probe.app.run():               # ephemeral app run — no separate `modal run` needed
            return modal_probe.probe.remote(config)
    raise SystemExit(f"unknown AUTOTUNE_BACKEND={backend!r} — use 'modal' or 'ssh' (paperspace/lambda)")


def _print(out):
    backend = os.environ.get("AUTOTUNE_BACKEND", "modal")
    print(f"status: {out.get('status')} | backend: {backend}")
    print(f"config: {out.get('config')}")
    for k in ("tokens_per_s", "samples_per_s", "peak_mem_gb", "gpu_util_max", "gpu_util_mean",
              "cpu_util_max", "cpu_util_mean", "gpu_mem_gb", "steps_measured"):
        if out.get(k) is not None:
            print(f"  {k}: {out[k]}")
    if out.get("error_detail"):
        print(f"  error_detail: {out['error_detail']}")


if __name__ == "__main__":
    cfg_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    config = json.load(open(cfg_path))
    out = run_probe(config)
    _print(out)
    with open("probe_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nfull result -> probe_result.json")
