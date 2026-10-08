"""Run a user's training script with Attach's hooks installed first, in the same process.

  python -m profiler.launch --out cap.json [--device cuda --warmup 3 --clean 10 ...] -- train.py <its args>

The bring-your-own-script entry point. `backends/local.py` runs it in a subprocess when a probe's
config names a `script`, so every backend (local, ssh, modal) reaches it.

The script runs as if started directly (`python train.py <its args>`): its `__name__` is
"__main__", its `sys.argv` is its own, and its directory is on `sys.path`. Nothing in it changes.
Hooks go in before its first line runs, so they're in place before it builds an optimizer.

The Capture is written to `--out` the moment the profiled step ends. The script then keeps
running to its own end, so give it a small step budget (e.g. `max_steps`) for a probe.

How the run ended decides what `--out` says, with the same statuses `capture()` uses, so
`frame/ceiling.py` can read a user-script trial the way it reads its own:

  | ended                               | `--out` status | notes                                        |
  |-------------------------------------|----------------|----------------------------------------------|
  | normally, after the profiled step   | ok             | the Capture                                  |
  | CUDA OOM before the profiled step   | oom            | clean steps that ran + memory at failure     |
  | CUDA OOM after the profiled step    | oom            | the Capture, status overridden: it doesn't fit|
  | any other exception, before         | error          | `<ExceptionType>: <first line of message>`   |
  | any other exception, after          | ok             | the Capture + `script_error` alongside it     |
  | normally, but too few steps         | error          | how many optimizer steps it reached          |
  | torch left patched (nopatch guard)  | error          | overrides any status above, `ok` included    |
  | invalid warmup / clean settings     | error          | before the script starts                     |

`--memory-only --steps N [--expect-shape B,S]` runs a ceiling trial instead (`profiler/memtrial.py`):
no timing, no profiler. The script is stopped after N optimizer steps, and `--out` holds
`{"trial": {"status", "error", "steps", "shapes_seen", "peak_gb", "reserved_gb", "total_gb"}}`:
`ok` (fits), `oom` (doesn't), or `error` (a crash, too few steps, or a batch that wasn't
`[B, S]`, which means the trial measured some other workload).

The exception is re-raised after `--out` is written, so the process still exits non-zero and
the full traceback still reaches stderr. A process killed outright (OOM killer, SIGKILL,
`os._exit`) writes nothing; treat a missing `--out` as an error.
Single process only: multi-GPU launch is out of scope for now.
"""
import argparse
import dataclasses
import json
import os
import runpy
import sys

import torch

from profiler import nopatch
from profiler.attach import Attach


def _write(path, rec):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, default=str)
    os.replace(tmp, path)                                        # never a half-written file


def _first_line(e):
    msg = str(e).strip()
    return f"{type(e).__name__}: {msg.splitlines()[0] if msg else '(no message)'}"


def _classify(exc):
    """(status, error) for how the script ended. (None, None) = a normal end."""
    if exc is None:
        return None, None
    if isinstance(exc, SystemExit):                              # sys.exit(0) is a normal end
        return (None, None) if exc.code in (None, 0) else ("error", f"SystemExit: {exc.code}")
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return "oom", _first_line(exc)
    return "error", _first_line(exc)


def main():
    if "--" not in sys.argv:
        raise SystemExit("usage: python -m profiler.launch [options] -- train.py [script args]")
    split = sys.argv.index("--")
    ours, theirs = sys.argv[1:split], sys.argv[split + 1:]
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="where the Capture JSON goes")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default=None, help="dtype the ceiling is measured at")
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--clean", type=int, default=10)
    ap.add_argument("--peak-flops", type=float, default=None, help="frozen ceiling (else measured)")
    ap.add_argument("--peak-bw", type=float, default=None, help="frozen ceiling (else measured)")
    ap.add_argument("--memory-only", action="store_true", help="a ceiling trial: memory, no timing")
    ap.add_argument("--steps", type=int, default=3, help="memory-only: optimizer steps to run")
    ap.add_argument("--expect-shape", default=None, help="memory-only: B,S every batch must have")
    ap.add_argument("--expect-image-tokens", type=int, default=None,
                    help="memory-only, VLM: image tokens every row must carry")
    ap.add_argument("--image-token-id", type=int, default=None, help="memory-only, VLM: their id")
    a = ap.parse_args(ours)
    if not theirs:
        raise SystemExit("no script given after --")
    script = os.path.abspath(theirs[0])
    if a.memory_only:
        return _memory_trial(a, script, theirs)

    try:                                                         # bad settings: say so in --out,
        att = Attach(device=a.device, warmup=a.warmup, clean_steps=a.clean, dtype=a.dtype,
                     peak_flops=a.peak_flops, peak_mem_bw_gb_s=a.peak_bw)
    except ValueError as e:                                      # not a crash with no result
        _write(a.out, {"capture": {"status": "error", "error": f"invalid probe settings: {e}"}})
        raise SystemExit(2)
    record = {}

    def on_done(cap):                                            # called from inside their step
        record.update({"capture": dataclasses.asdict(cap), "params": att.params,
                       "params_total": att.params_total, "tokens": att.tokens, "step_tokens": att.step_tokens,
                       "micro_batches": att.micro_batches})
        _write(a.out, record)
    att.on_done = on_done
    before = nopatch.snapshot()
    att.install()

    sys.argv = theirs                                            # the script sees its own argv
    sys.path[0] = os.path.dirname(script)                        # ...and its own directory
    exc = None
    try:
        runpy.run_path(script, run_name="__main__")
    except BaseException as e:                                   # recorded, then re-raised below
        exc = e
        raise
    finally:
        status, error = _classify(exc)
        if not att.done:                                         # ended before the profiled step
            if status is None:                                   # a normal end, but too early
                status, error = "error", (f"script ended after {att._n} optimizer steps; "
                                          f"needs {a.warmup + a.clean + 1}")
            cap = att.failed(status, error)
            _write(a.out, {"capture": dataclasses.asdict(cap), "params": att.params,
                           "params_total": att.params_total, "tokens": None, "step_tokens": att.step_tokens, "micro_batches": None})
        elif status == "oom":                                    # fit for one step, not for the run
            record["capture"]["status"] = "oom"
            record["capture"]["error"] = f"{error} (after the profiled step)"
            _write(a.out, record)
        elif status == "error":                                  # the measurement stands
            record["script_error"] = error
            _write(a.out, record)
        try:
            nopatch.verify(before)                               # fails the run if torch was patched
        except SystemExit as guard:                              # ...and then it is not a measurement:
            with open(a.out) as f:                               # overwrite whatever was written,
                rec = json.load(f)                               # an `ok` included
            rec["capture"]["status"] = "error"
            rec["capture"]["error"] = str(guard)
            _write(a.out, rec)
            raise


def _run_script(script, theirs):
    sys.argv = theirs                                            # the script sees its own argv
    sys.path[0] = os.path.dirname(script)                        # ...and its own directory
    runpy.run_path(script, run_name="__main__")


def _memory_trial(a, script, theirs):
    """A ceiling trial: run the script for `--steps` optimizer steps, record memory, stop it."""
    from profiler.memtrial import MemoryTrial, TrialDone
    try:
        expect = tuple(int(x) for x in a.expect_shape.split(",")) if a.expect_shape else None
        trial = MemoryTrial(device=a.device, steps=a.steps, expect_shape=expect,
                            expect_image_tokens=a.expect_image_tokens, image_token_id=a.image_token_id)
    except ValueError as e:
        _write(a.out, {"trial": {"status": "error", "error": f"invalid trial settings: {e}"}})
        raise SystemExit(2)
    before = nopatch.snapshot()
    trial.install()
    exc = None
    try:
        _run_script(script, theirs)
    except TrialDone:                                            # the answer is in: stop here
        pass
    except BaseException as e:
        exc = e
    finally:
        trial.uninstall()
    status, error = _classify(exc)
    if status is None:
        if trial.n >= trial.steps:
            status = "ok"
        else:                                                    # ended by itself, too early
            status, error = "error", (f"script ended after {trial.n} optimizer steps; "
                                      f"needs {trial.steps}")
    rec = {"trial": trial.result(status, error)}
    try:
        nopatch.verify(before)
    except SystemExit as guard:
        rec["trial"]["status"], rec["trial"]["error"] = "error", str(guard)
        _write(a.out, rec)
        raise
    _write(a.out, rec)
    if exc is not None and not isinstance(exc, SystemExit):
        raise exc                                                # the traceback reaches stderr


if __name__ == "__main__":
    main()
