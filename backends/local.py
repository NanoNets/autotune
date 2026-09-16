"""Local backend — run the probe in-process on this machine's device.

capture (clean timing + trace) -> diagnose (roofline + trace -> Observation) — the same flow as
`python -m profiler`, but as a callable the loop drives. Returns the Observation as a dict. The
frozen ceilings the loop measured at Frame ride in on the config (`peak_flops` / `peak_mem_bw_gb_s`)
so every probe is scored against the same denominator. Heavy imports (torch, the runner) are lazy.
"""
import dataclasses
import sys

# framework name -> (module, runner class). Add a line per adapter.
_RUNNERS = {
    "torch": ("frameworks.torch", "TorchRunner"),
    "hf": ("frameworks.hf", "HFRunner"),
}


def _resolve_device(config):
    """Which device this probe runs on. Defaults to cuda — a GPU is the only case that matters.

    An unset `device` falls through to the runner's own "cpu" default, which returns a perfectly
    well-formed Observation full of CPU numbers and a meaningless MFU: indistinguishable downstream
    from a real probe, and worth a whole night of experiments. So cpu is something you ask for, and
    a fallback announces itself (here, and in the Observation's `device_type`).
    """
    device = config.get("device")
    if device:                                         # explicit wins, always — including "cpu"
        if device.startswith("cuda:"):                 # capture compares `device == "cuda"` exactly,
            raise ValueError(                          # so an indexed device skips sync + memory
                f"device {device!r}: use \"cuda\" (single-GPU only). An indexed device silently "
                "skips synchronisation and memory stats, producing a well-formed probe with "
                "garbage timings. Pick the GPU with CUDA_VISIBLE_DEVICES instead.")
        return device
    import torch
    if torch.cuda.is_available():
        return "cuda"
    print("autotune: no CUDA device found — running this probe on CPU. MFU and every GPU signal "
          "will be meaningless; treat it as a smoke test, not a result. Set `device` in run.toml "
          "to choose explicitly.", file=sys.stderr)
    return "cpu"


def _make_runner(framework):
    """Instantiate the named framework's runner (raises before importing anything on a bad name)."""
    if framework not in _RUNNERS:
        raise ValueError(f"unknown framework {framework!r}; known: {sorted(_RUNNERS)}")
    import importlib
    mod, cls = _RUNNERS[framework]
    return getattr(importlib.import_module(mod), cls)()


def run(config):
    """Run one probe here and return its Observation (a dict). Needs torch + the chosen runner."""
    from profiler.capture import capture
    from profiler.diagnostic import diagnose

    config = {**config, "device": _resolve_device(config)}   # resolve ONCE — capture and the
    runner = _make_runner(config.get("framework", "torch"))  #   runner must agree on the device
    cap = capture(runner, config,
                  peak_flops=config.get("peak_flops"),
                  peak_mem_bw_gb_s=config.get("peak_mem_bw_gb_s"))
    params = getattr(runner, "params", 0)              # 0 if setup failed before params were set
    tokens = runner.tokens_per_step() if params else 0
    obs = diagnose(cap, params=params, tokens=tokens, gpu=config.get("gpu", "local"))
    return dataclasses.asdict(obs)
