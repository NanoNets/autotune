"""Local backend — run the probe in-process on this machine's device.

capture (clean timing + trace) -> diagnose (roofline + trace -> Observation) — the same flow as
`python -m profiler`, but as a callable the loop drives. Returns the Observation as a dict. The
frozen ceilings the loop measured at Frame ride in on the config (`peak_flops` / `peak_mem_bw_gb_s`)
so every probe is scored against the same denominator. Heavy imports (torch, the runner) are lazy.
"""
import dataclasses

# framework name -> (module, runner class). Add a line per adapter.
_RUNNERS = {
    "torch": ("frameworks.torch", "TorchRunner"),
    "hf": ("frameworks.hf", "HFRunner"),
}


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

    runner = _make_runner(config.get("framework", "torch"))
    cap = capture(runner, config,
                  peak_flops=config.get("peak_flops"),
                  peak_mem_bw_gb_s=config.get("peak_mem_bw_gb_s"))
    params = getattr(runner, "params", 0)              # 0 if setup failed before params were set
    tokens = runner.tokens_per_step() if params else 0
    obs = diagnose(cap, params=params, tokens=tokens, gpu=config.get("gpu", "local"))
    return dataclasses.asdict(obs)
