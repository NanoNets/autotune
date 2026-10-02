"""Local backend — run the probe in-process on this machine's device.

capture (clean timing + trace) -> diagnose (roofline + trace -> Observation) — the same flow as
`python -m profiler`, but as a callable the loop drives. Returns the Observation as a dict. The
frozen ceilings the loop measured at Frame ride in on the config (`peak_flops` / `peak_mem_bw_gb_s`)
so every probe is scored against the same denominator. Heavy imports (torch, the runner) are lazy.
"""
import dataclasses
import json
import os
import signal
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
    timeout = config.get("timeout", 1800)
    stdout, stderr, timed_out, returncode = _launch(argv, config)
    if timed_out:
        cap = Capture(status="error", device=config["device"],
                      error=f"script exceeded timeout of {timeout}s (killed)")
        rec = {}
    elif os.path.exists(out):
        with open(out) as f:
            rec = json.load(f)
        cap = Capture(**rec["capture"])
    else:                                              # killed (OOM killer, SIGKILL, os._exit)
        tail = (stderr or stdout or "").strip().splitlines()[-1:] or ["(no output)"]
        cap = Capture(status="error", device=config["device"],
                      error=f"launcher wrote no result (exit {returncode}): {tail[0]}")
        rec = {}
    obs = dataclasses.asdict(diagnose(cap, params=rec.get("params") or 0,
                                      tokens=rec.get("tokens"), gpu=config.get("gpu", "local")))
    obs["population"] = {**(obs.get("population") or {}), "script": config["script"],
                         "script_args": list(map(str, config.get("script_args") or [])),
                         "script_files": list(config.get("script_files") or []),
                         "micro_batches": rec.get("micro_batches"),
                         "script_error": rec.get("script_error")}
    trained, declared = obs["population"].get("model_trained"), config.get("model")
    # Frame profiled `model`; the script may train something else. Say so rather than mix the two.
    obs["population"]["model_matches_config"] = (None if not (trained and declared)
                                                 else trained.rstrip("/") == declared.rstrip("/"))
    return obs


def ceiling_trial(config):
    """One memory-only ceiling trial (`probe_kind: "ceiling_trial"`, built by `frame.ceiling`):
    `trial_steps` optimizer steps at `micro_batch`, then `{"status": ok|oom|error, "error",
    "peak_gb", "reserved_gb", "total_gb", ...}`. No timing, no profiler, no Observation.

    Our runner: worst-case synthetic batches, in this process. A user's script: its own launch
    under `profiler.launch --memory-only`, reading Frame's ceiling rows through its own data setting
    (`script_args` / `script_files` already point at them), and every batch must be
    `[micro_batch, max_len]`."""
    steps = int(config.get("trial_steps", 3))
    if not config.get("script"):
        from frame.ceiling import runner_trial
        runner = _make_runner(config.get("framework", "torch"))
        return runner_trial(runner, config, config["max_len"], config.get("vocab"), steps,
                            config.get("seed", 0))(config["micro_batch"])
    for f in config.get("script_files") or []:
        _script_path(f)
    out = os.path.join(tempfile.mkdtemp(prefix="autotune_trial_"), "trial.json")
    argv = [config.get("python") or sys.executable, "-m", "profiler.launch", "--out", out,
            "--device", config["device"], "--memory-only", "--steps", str(steps),
            "--expect-shape", f"{config['micro_batch']},{config['max_len']}",
            "--", _script_path(config["script"]), *map(str, config.get("script_args") or [])]
    stdout, stderr, timed_out, code = _launch(argv, config)
    if timed_out:
        return {"status": "error", "error": f"trial exceeded timeout of {config.get('timeout', 1800)}s (killed)"}
    if not os.path.exists(out):                        # killed outright: no memory answer
        tail = (stderr or stdout or "").strip().splitlines()[-1:] or ["(no output)"]
        return {"status": "error", "error": f"launcher wrote no result (exit {code}): {tail[0]}"}
    with open(out) as f:
        return json.load(f)["trial"]


def _launch(argv, config):
    """Run the launcher in its own process group; (stdout, stderr, timed_out, returncode)."""
    env = {**os.environ, **{str(k): str(v) for k, v in (config.get("env") or {}).items()}}
    env["PYTHONPATH"] = _REPO + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    # Its own process group, so a timeout kills the script AND its children (DataLoader workers
    # would otherwise outlive it and keep holding the GPU).
    proc = subprocess.Popen(argv, cwd=config.get("script_cwd") or _REPO, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=config.get("timeout", 1800))
        return stdout, stderr, False, proc.returncode
    except subprocess.TimeoutExpired:                  # a hang (e.g. a worker deadlock) or simply
        os.killpg(proc.pid, signal.SIGKILL)            # too long: an error result, never a
        stdout, stderr = proc.communicate()            # crashed loop
        return stdout, stderr, True, proc.returncode


def run(config):
    """Run one probe here and return its Observation (a dict). Needs torch + the chosen runner,
    or, when `config["script"]` is set, the user's training script (`run_script`). A
    `probe_kind: "ceiling_trial"` config is a memory-only ceiling trial instead (`ceiling_trial`)."""
    from profiler.capture import capture
    from profiler.diagnostic import diagnose

    if config.get("probe_kind") == "ceiling_trial":
        return ceiling_trial({**config, "device": _resolve_device(config)})
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
