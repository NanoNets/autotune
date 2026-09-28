"""Run ONE experiment-4 arm in a fresh process; print a JSON record.

  R  capture() drives raw_runner.RawHFRunner (our labelled loop, same model + data)
  L  launcher.py -- train_hf.py         (HF Trainer, hooks installed by the launcher)
  T  train_hf.py --time-out             (HF Trainer, no hooks, TrainerCallback step timing)
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import tempfile

HF = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.dirname(HF)
sys.path[:0] = [HF, SANDBOX]                                      # the sandbox, and nothing else

import versions                                                   # noqa: E402
import workload                                                   # noqa: E402
from run_arm import fingerprint                                   # noqa: E402
from vendor.capture import Capture                                # noqa: E402
from vendor.diagnostic import diagnose                            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["R", "L", "T"], required=True)
    ap.add_argument("--time-mode", choices=["end", "opt", "two"], default="end",
                    help="arm T only: where the TrainerCallback syncs (see train_hf.StepTimer)")
    ap.add_argument("--config", default="{}")
    ap.add_argument("--peak-flops", type=float, required=True)
    ap.add_argument("--peak-bw", type=float, required=True)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--clean", type=int, default=10)
    a = ap.parse_args()
    got_versions = versions.check()
    cfg = workload.resolve(json.loads(a.config))
    steps = a.warmup + a.clean + 3                                # the probe's max_steps
    tmp = tempfile.mkdtemp()
    micro = None

    if a.arm == "R":
        from raw_runner import RawHFRunner
        from vendor.capture import capture
        r = RawHFRunner()
        cap = capture(r, {"device": "cuda", **cfg}, warmup=a.warmup, clean_steps=a.clean,
                      peak_flops=a.peak_flops, peak_mem_bw_gb_s=a.peak_bw)
        if cap.status != "ok":
            raise SystemExit(f"capture: {cap.status}: {cap.error}")
        params, tokens = r.params, r.tokens_per_step()
        truth = workload.truth(cfg, r.model)
        loaded = workload.describe(r.model)
    else:
        train = os.path.join(HF, "train_hf.py")
        tail = ["--config", json.dumps(cfg), "--max-steps", str(steps)]
        if a.arm == "L":
            out = os.path.join(tmp, "cap.json")
            cmd = [sys.executable, os.path.join(SANDBOX, "launcher.py"), "--out", out,
                   "--device", "cuda", "--dtype", "bfloat16", "--warmup", str(a.warmup),
                   "--clean", str(a.clean), "--peak-flops", str(a.peak_flops),
                   "--peak-bw", str(a.peak_bw), "--", train] + tail
        else:
            out = os.path.join(tmp, "steps.json")
            cmd = [sys.executable, train] + tail + ["--time-out", out, "--warmup", str(a.warmup),
                                                    "--clean", str(a.clean),
                                                    "--time-mode", a.time_mode]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=HF)
        if p.returncode != 0 or not os.path.exists(out):
            raise SystemExit(f"arm {a.arm} failed ({p.returncode}):\n{p.stderr[-4000:]}")
        got = json.load(open(out))
        ref = workload.build_model(cfg)                   # same call as the script, on CPU
        truth, loaded = workload.truth(cfg, ref), workload.describe(ref)
        del ref
        if a.arm == "T":                                          # timing only: no trace
            t = got["times"]
            wall = statistics.median(t)
            print(json.dumps({"arm": "T", "versions": got["versions"], "truth": truth,
                              "loaded": got.get("loaded"), "mode": got.get("mode"),
                              "host_after_sync": got.get("host_after_sync"),
                              "step_times_s": t,
                              "obs": {"step_time_s": wall,
                                      "step_time_cv": round(statistics.pstdev(t) / wall, 4)}}))
            return
        cap = Capture(**got["capture"])
        if cap.status != "ok":
            raise SystemExit(f"launcher capture: {cap.error}")
        params, tokens, micro = got["params"], got["tokens"], got["micro_batches"]

    obs = diagnose(cap, params=params, tokens=tokens, gpu="A10")
    leaked = sorted(m for m in sys.modules if m.split(".")[0] in ("profiler", "frameworks", "frame"))
    if leaked:
        raise SystemExit(f"live repo modules imported: {leaked}")
    print(json.dumps({"arm": a.arm, "versions": got_versions, "loaded": loaded, "obs": obs.__dict__,
                      "recovered": {"params": params, "tokens": tokens, "micro_batches": micro},
                      "truth": truth, "step_times_s": cap.step_times_s,
                      "fingerprint": fingerprint(cap.trace)}, default=str))


if __name__ == "__main__":
    main()
