"""Backend-agnostic probe entrypoint.

Reads inputs from config.json (backend, model, dataset, effective_batch, hardware) and secrets
from .env, runs one probe, prints its metrics, and appends the result to search_state.json. The
pipeline and the JSON it returns are identical on either backend.

  python probe.py             # backend chosen by config.json
  python probe.py other.json  # alternate config path
"""
import json
import os
import sys

try:
    from dotenv import load_dotenv
    load_dotenv()                                     # secrets (HF_TOKEN, SSH creds) from .env → env
except ImportError:
    pass                                              # fine if already in the environment

SSH_ALIASES = {"ssh", "vm", "paperspace", "lambda", "lambda_labs", "lambdalabs"}

STATE_PATH = "search_state.json"
# identifies a session; a probe with a different setup starts a new one
SETUP_KEYS = ("backend", "model", "dataset", "max_length", "attn_impl",
              "effective_batch", "hardware", "hardware_number")
# result fields persisted to history (raw logs excluded)
RESULT_KEYS = ("status", "tokens_per_s", "samples_per_s", "peak_mem_gb",
               "gpu_util_mean", "cpu_util_mean", "steps_measured", "error_detail")


def _inline_local_dataset(config: dict) -> dict:
    """If `dataset` is a local file, inline its contents so the backend can materialize it —
    neither Modal nor the SSH box has the local path. HF/MS ids and 'synthetic' pass through.
    (For image datasets the referenced images must still resolve on the backend.)"""
    ds = config.get("dataset")
    if isinstance(ds, str) and ds != "synthetic" and os.path.isfile(ds):
        return {**config, "dataset": os.path.basename(ds), "dataset_file": open(ds).read()}
    return config


def run_probe(config: dict) -> dict:
    """Dispatch one probe to the configured backend and return its result."""
    config = _inline_local_dataset(config)
    backend = str(config.get("backend", "modal")).strip().lower()
    if backend in SSH_ALIASES:
        import vm_probe
        return vm_probe.run_probe(config)
    if backend == "modal":
        os.environ["AUTOTUNE_GPU"] = str(config.get("hardware", "A10G"))          # GPU type
        os.environ["AUTOTUNE_GPU_COUNT"] = str(config.get("hardware_number", 1))   # GPU count
        os.environ["AUTOTUNE_ATTN"] = str(config.get("attn_impl", "flash_attn"))   # selects the image
        import modal_probe
        with modal_probe.app.run():               # ephemeral app run — no separate `modal run` needed
            return modal_probe.probe.remote(config)
    raise SystemExit(f"unknown backend={backend!r} in config — use 'modal' or 'ssh' (paperspace/lambda)")


def update_search_state(config: dict, out: dict, path: str = STATE_PATH):
    """Append this probe to search_state.json (history + recomputed best).
    Caller-managed fields (each entry's `note`, plus top-level `beliefs`/`active_extensions`) are preserved."""
    setup = {k: config.get(k) for k in SETUP_KEYS}
    state = None
    if os.path.exists(path):
        state = json.load(open(path))
        if state.get("setup") != setup:               # different setup → new session
            print("  (search_state.json was for a different setup — starting a fresh session)")
            state = None
    if state is None:
        state = {"setup": setup, "history": [], "best": None,
                 "active_extensions": [], "beliefs": ""}

    state["history"].append({
        "iter": len(state["history"]) + 1,
        "config": out.get("config", {}),             # exact levers that ran
        "result": {k: out.get(k) for k in RESULT_KEYS},
        "note": "",                                  # annotated later by the caller
    })

    ok = [h for h in state["history"]
          if h["result"].get("status") == "ok" and h["result"].get("tokens_per_s") is not None]
    if ok:
        win = max(ok, key=lambda h: h["result"]["tokens_per_s"])
        state["best"] = {"config": win["config"], "tokens_per_s": win["result"]["tokens_per_s"]}

    with open(path, "w") as f:
        json.dump(state, f, indent=2)
    return state


def _flags(cfg):
    """Notable levers in a config, excluding bs/ga and the always-present fields."""
    skip = {"model", "world_size", "effective_batch",
            "per_device_train_batch_size", "gradient_accumulation_steps"}
    parts = []
    for k, v in cfg.items():
        if k in skip or v in (False, None):
            continue
        if k == "attn_impl" and v == "flash_attn":        # default; omitted from the table
            continue
        parts.append(k if v is True else f"{k}={v}")
    return ", ".join(parts) or "-"


def render_table(state):
    """Markdown table of the whole session."""
    s = state["setup"]
    cell = lambda x: "-" if x is None else x
    lines = [f"# Probes — {s.get('model')} @ eff_batch {s.get('effective_batch')}, "
             f"max_len {s.get('max_length')}, {s.get('hardware')}×{s.get('hardware_number')}", ""]
    best = state.get("best")
    if best:
        bc = best.get("config", {})
        lines += [f"**best:** {best.get('tokens_per_s')} tok/s — bs "
                  f"{bc.get('per_device_train_batch_size')}, {_flags(bc)}", ""]
    if state.get("beliefs"):
        lines += [f"**beliefs:** {state['beliefs']}", ""]
    lines += ["| # | bs | ga | flags | tok/s | mem(G) | gpu% | cpu% | status | note |",
              "|---|----|----|-------|-------|--------|------|------|--------|------|"]
    for h in state["history"]:
        c, r = h["config"], h["result"]
        lines.append(
            f"| {h['iter']} | {cell(c.get('per_device_train_batch_size'))} | "
            f"{cell(c.get('gradient_accumulation_steps'))} | {_flags(c)} | "
            f"{cell(r.get('tokens_per_s'))} | {cell(r.get('peak_mem_gb'))} | "
            f"{cell(r.get('gpu_util_mean'))} | {cell(r.get('cpu_util_mean'))} | "
            f"{cell(r.get('status'))} | {h.get('note', '')} |")
    return "\n".join(lines) + "\n"


def _print(out, backend):
    print(f"status: {out.get('status')} | backend: {backend}")
    print(f"config: {out.get('config')}")
    for k in ("tokens_per_s", "samples_per_s", "peak_mem_gb", "gpu_util_max", "gpu_util_mean",
              "cpu_util_max", "cpu_util_mean", "gpu_mem_gb", "steps_measured"):
        if out.get(k) is not None:
            print(f"  {k}: {out[k]}")
    if out.get("error_detail"):
        print(f"  error_detail: {out['error_detail']}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?", default="config.json", help="config file with fixed inputs")
    ap.add_argument("--levers", default=None, help="JSON dict of swift sft args for this probe")
    a = ap.parse_args()
    config = json.load(open(a.config))
    if a.levers:                                       # per-probe levers, not persisted to config.json
        config["levers"] = json.loads(a.levers)
    out = run_probe(config)
    _print(out, config.get("backend", "modal"))
    state = update_search_state(config, out)
    with open("probes.md", "w") as f:                 # session table
        f.write(render_table(state))
    print(f"\nappended probe #{len(state['history'])} -> {STATE_PATH}  |  table -> probes.md")
