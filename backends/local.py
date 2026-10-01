"""Local backend — run the probe in-process on this machine's device.

capture (clean timing + trace) -> diagnose (roofline + trace -> Observation) — the same flow as
`python -m profiler`, but as a callable the loop drives. Returns the Observation as a dict. The
frozen ceilings the loop measured at Frame ride in on the config (`peak_flops` / `peak_mem_bw_gb_s`)
so every probe is scored against the same denominator. Heavy imports (torch, the runner) are lazy.
"""
import dataclasses
import json
import os
import subprocess
import sys
import tempfile

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


_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _script_path(script):
    """The user's training script: absolute, or relative to the repo root (the form ssh needs)."""
    path = script if os.path.isabs(script) else os.path.join(_REPO, script)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"script {script!r} not found (relative paths resolve from {_REPO})")
    return path


def _launch_argv(config, out):
    """`python -m profiler.launch ... -- <script> <script_args>` for this probe."""
    argv = [config.get("python") or sys.executable, "-m", "profiler.launch", "--out", out,
            "--device", config["device"],
            "--warmup", str(config.get("warmup", 3)), "--clean", str(config.get("clean_steps", 10))]
    for flag, key in (("--peak-flops", "peak_flops"), ("--peak-bw", "peak_mem_bw_gb_s"),
                      ("--dtype", "dtype")):
        if config.get(key) is not None:
            argv += [flag, str(config[key])]
    return argv + ["--", _script_path(config["script"]), *map(str, config.get("script_args") or [])]


def run_script(config):
    """Bring-your-own-script probe: run the user's unchanged training script under
    `profiler.launch` in its own process, read back the Capture it wrote, and diagnose it.

    `script_files` (optional): other repo-relative files the script reads, e.g. the per-probe
    config copies under `runs/<run>/probes/`. Here they must exist; the ssh backend also ships them.

    The script must build its optimizer and loop the usual way, run on one GPU, fetch batches
    through a torch DataLoader, and stop by itself: the agent sets its step budget (`max_steps`
    or equivalent) to at least warmup + clean_steps + 1 via `script_args`. Knobs reach the script
    the same way (its own CLI flags / config file), never through this backend.
    A process killed outright writes no result: that is an `error` Observation, not a crash here.
    """
    from profiler.capture import Capture
    from profiler.diagnostic import diagnose

    for f in config.get("script_files") or []:          # fail before launching, not mid-probe
        _script_path(f)
    out = os.path.join(tempfile.mkdtemp(prefix="autotune_script_"), "cap.json")
    argv = _launch_argv(config, out)
    env = {**os.environ, **{str(k): str(v) for k, v in (config.get("env") or {}).items()}}
    env["PYTHONPATH"] = _REPO + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    p = subprocess.run(argv, cwd=config.get("script_cwd") or _REPO, env=env,
                       capture_output=True, text=True, timeout=config.get("timeout", 1800))
    if os.path.exists(out):
        with open(out) as f:
            rec = json.load(f)
        cap = Capture(**rec["capture"])
    else:                                              # killed (OOM killer, SIGKILL, os._exit)
        tail = (p.stderr or p.stdout or "").strip().splitlines()[-1:] or ["(no output)"]
        cap = Capture(status="error", device=config["device"],
                      error=f"launcher wrote no result (exit {p.returncode}): {tail[0]}")
        rec = {}
    obs = dataclasses.asdict(diagnose(cap, params=rec.get("params") or 0,
                                      tokens=rec.get("tokens"), gpu=config.get("gpu", "local")))
    obs["population"] = {**(obs.get("population") or {}), "script": config["script"],
                         "script_args": list(map(str, config.get("script_args") or [])),
                         "script_files": list(config.get("script_files") or []),
                         "micro_batches": rec.get("micro_batches"),
                         "script_error": rec.get("script_error")}
    return obs


def run(config):
    """Run one probe here and return its Observation (a dict). Needs torch + the chosen runner,
    or, when `config["script"]` is set, the user's training script (`run_script`)."""
    from profiler.capture import capture
    from profiler.diagnostic import diagnose

    if config.get("script"):
        return run_script({**config, "device": _resolve_device(config)})
    config = {**config, "device": _resolve_device(config)}   # resolve ONCE — capture and the
    runner = _make_runner(config.get("framework", "torch"))  #   runner must agree on the device
    cap = capture(runner, config,
                  # loop/loop.md Test 3 prescribes an "extended probe" for the champion; capture
                  # already takes the dials, they just were not reachable from a config. Defaults
                  # unchanged, so an ordinary probe is byte-for-byte what it was.
                  warmup=config.get("warmup", 3),
                  clean_steps=config.get("clean_steps", 10),
                  peak_flops=config.get("peak_flops"),
                  peak_mem_bw_gb_s=config.get("peak_mem_bw_gb_s"))
    params = getattr(runner, "params", 0)              # 0 if setup failed before params were set
    tokens = runner.tokens_per_step() if params else 0
    obs = diagnose(cap, params=params, tokens=tokens, gpu=config.get("gpu", "local"))
    # step_time_mean_s / clean_steps_measured used to be bolted on here. They are Observation
    # fields now, next to the population the window timed (profiler/diagnostic.py).
    return dataclasses.asdict(obs)
