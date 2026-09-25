"""Run a user's training script with Attach's hooks installed first, in the same process.

  python launcher.py --out cap.json [--device cuda --warmup 3 --clean 10 ...] -- train.py <its args>

The script runs as if started directly (`python train.py <its args>`): its `__name__` is
"__main__", its `sys.argv` is its own, and its directory is on `sys.path`. Nothing in it changes.
Hooks go in before its first line runs, so they're in place before it builds an optimizer.

The Capture is written to `--out` the moment the profiled step ends. The script then keeps
running to its own end, so give it a small step budget (e.g. `max_steps`) for a probe. If the
script ends before the profiled step, `--out` gets `{"status": "error", ...}` instead.
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

from attach import Attach                                        # noqa: E402


def _write(path, rec):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rec, f, default=str)
    os.replace(tmp, path)                                        # never a half-written file


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

    def on_done(cap):                                            # called from inside their step
        _write(a.out, {"capture": dataclasses.asdict(cap), "params": att.params,
                       "tokens": att.tokens, "micro_batches": att.micro_batches})
    att.on_done = on_done
    att.install()

    sys.argv = theirs                                            # the script sees its own argv
    sys.path[0] = os.path.dirname(script)                        # ...and its own directory
    try:
        runpy.run_path(script, run_name="__main__")
    finally:
        if not att.done:                                         # ended (or crashed) too early
            att.uninstall()
            _write(a.out, {"capture": {"status": "error", "error":
                           f"script ended after {att._n} optimizer steps; needs "
                           f"{a.warmup + a.clean + 1}"}})


if __name__ == "__main__":
    main()
