"""Run a user's training script with Attach's hooks installed first, in the same process.

  python launcher.py --out cap.json [--device cuda --warmup 3 --clean 10 ...] -- train.py <its args>

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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch                                                     # noqa: E402

import nopatch                                                   # noqa: E402
from attach import Attach                                        # noqa: E402


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
        raise SystemExit("usage: launcher.py [options] -- train.py [script args]")
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
    a = ap.parse_args(ours)
    if not theirs:
        raise SystemExit("no script given after --")
    script = os.path.abspath(theirs[0])

    att = Attach(device=a.device, warmup=a.warmup, clean_steps=a.clean, dtype=a.dtype,
                 peak_flops=a.peak_flops, peak_mem_bw_gb_s=a.peak_bw)
    record = {}

    def on_done(cap):                                            # called from inside their step
        record.update({"capture": dataclasses.asdict(cap), "params": att.params,
                       "tokens": att.tokens, "micro_batches": att.micro_batches})
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
                           "tokens": None, "micro_batches": None})
        elif status == "oom":                                    # fit for one step, not for the run
            record["capture"]["status"] = "oom"
            record["capture"]["error"] = f"{error} (after the profiled step)"
            _write(a.out, record)
        elif status == "error":                                  # the measurement stands
            record["script_error"] = error
            _write(a.out, record)
        nopatch.verify(before)                                   # fails the run if torch was patched


if __name__ == "__main__":
    main()
