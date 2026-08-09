"""Backend-agnostic probe entrypoint — the loop's single command.

Everything is configured in config.json (or a path given as argv[1]); only secrets live in env
(see .env.example). `backend` picks where the probe runs — the pipeline and the JSON it returns are
identical either way:

  "backend": "modal"   # default — Modal serverless GPU
  "backend": "ssh"     # an SSH GPU VM (paperspace / lambda are aliases)

Runs one probe, prints the Layer-1 metrics, and writes the full result to probe_result.json.
"""
import json
import sys

SSH_ALIASES = {"ssh", "vm", "paperspace", "lambda", "lambda_labs", "lambdalabs"}


def _backend(config):
    return config.get("backend", "modal").strip().lower()


def run_probe(config: dict) -> dict:
    """Dispatch one probe to the backend named in config.json and return its Layer-1 result."""
    backend = _backend(config)
    if backend in SSH_ALIASES:
        import vm_probe
        return vm_probe.run_probe(config)
    if backend == "modal":
        import modal_probe
        with modal_probe.app.run():               # ephemeral app run — no separate `modal run` needed
            return modal_probe.probe.remote(config)
    raise SystemExit(f"unknown backend={backend!r} in config.json — use 'modal' or 'ssh' (paperspace/lambda)")


def _print(out, config):
    print(f"status: {out.get('status')} | backend: {_backend(config)}")
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
    _print(out, config)
    with open("probe_result.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nfull result -> probe_result.json")
