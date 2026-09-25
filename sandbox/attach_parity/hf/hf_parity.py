"""Experiment 4: an HF Trainer script through the launcher (single GPU).

  ~/autotune_tools/venv_hf/bin/python hf/hf_parity.py --rounds 5 [--config '{...}']

Pass/fail:
  1. Versions are exactly the pins (hf/versions.py), in every arm.
  2. L (Trainer + hooks) records a complete Observation: every phase label present, nothing
     empty or wrongly 0 that R fills, grad accum / tokens / params recovered exactly.
  3. L's step time agrees with T's (the same Trainer run, timed by a TrainerCallback, no hooks)
     within the noise floor. Tolerance = median step_time_cv of L and T.
Informational:
  4. L vs R (our labelled raw loop, same model + data): what Trainer adds (kernels, CPU ops,
     step time). Differences here are Trainer's own work, not measurement error.
"""
import argparse
import json
import os
import subprocess
import sys

HF = os.path.dirname(os.path.abspath(__file__))
SANDBOX = os.path.dirname(HF)
sys.path[:0] = [HF, SANDBOX]

import parity                                                     # noqa: E402
import versions                                                   # noqa: E402

LABELS = ["step", "forward", "backward", "dataloader", "Optimizer.step#AdamW.step"]


def run(arm, cfg, pf, bw, warmup, clean):
    p = subprocess.run([sys.executable, os.path.join(HF, "run_hf_arm.py"), "--arm", arm,
                        "--config", json.dumps(cfg), "--peak-flops", str(pf), "--peak-bw", str(bw),
                        "--warmup", str(warmup), "--clean", str(clean)],
                       capture_output=True, text=True, cwd=HF,
                       env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    if p.returncode != 0:
        raise RuntimeError(f"arm {arm} failed:\n{p.stderr[-4000:]}")
    return json.loads(p.stdout.strip().splitlines()[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="{}")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--clean", type=int, default=10)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pinned = versions.check()
    cfg = json.loads(a.config)
    pf, bw = parity.ceilings("cuda", "bfloat16")
    print(f"versions: {pinned}")
    print(f"frozen ceilings (bf16): peak_flops {pf:.4g}  peak_bw {bw:.4g} GB/s   config {cfg}")

    recs = {"R": [], "L": [], "T": []}
    for i in range(a.rounds):
        for arm in ("R", "L", "T"):
            recs[arm].append(run(arm, cfg, pf, bw, a.warmup, a.clean))
        print(f"round {i + 1}: " + "  ".join(f"{k} {recs[k][-1]['obs']['step_time_s']:.4f}s"
                                              for k in recs))

    print("\n# 1. Versions")
    ok = all(r["versions"] == pinned for rs in recs.values() for r in rs)
    print(f"  every arm, every round, exactly the pins: {ok}")

    print("\n# 2. L completeness")
    L0 = recs["L"][0]
    ann = L0["fingerprint"]["annotations"]
    print(f"  labels in L's trace (count): { {k: ann.get(k, 0) for k in LABELS} }")
    print(f"  missing: {[k for k in LABELS if not ann.get(k)] or 'none'}")
    for r in recs["L"]:
        mb, t = r["recovered"]["micro_batches"], r["truth"]
        assert mb["forwards"] == mb["backwards"] == t["grad_accum"], (mb, t)
        assert r["recovered"]["tokens"] == t["tokens"] and r["recovered"]["params"] == t["params"]
    print(f"  recovered (every round identical): {L0['recovered']} vs truth {L0['truth']}")

    print("\n# 3. Timing: L (hooks) vs T (TrainerCallback, no hooks), same Trainer run")
    noise = parity.med([r["obs"]["step_time_cv"] for r in recs["L"] + recs["T"]])
    mL = parity.med([r["obs"]["step_time_s"] for r in recs["L"]])
    mT = parity.med([r["obs"]["step_time_s"] for r in recs["T"]])
    d = parity.rel(mT, mL)
    print(f"  tolerance = noise floor (median step_time_cv, L and T) = {noise:.2%}")
    sT = parity.spread([r["obs"]["step_time_s"] for r in recs["T"]])
    sL = parity.spread([r["obs"]["step_time_s"] for r in recs["L"]])
    print(f"  step_time_s: T {mT:.4f}  L {mL:.4f}  Δ {parity.fmt(d, True)}   "
          f"T spread {parity.fmt(sT, True)}  L spread {parity.fmt(sL, True)}   "
          f"within noise: {'yes' if abs(d) <= noise else 'NO'}")

    print("\n# 4. Informational: L (Trainer) vs R (raw labelled loop), what Trainer adds")
    parity.report({"A": recs["R"], "L": recs["L"]}, "L", a.rounds)
    fa, fl = recs["R"][0]["fingerprint"], recs["L"][0]["fingerprint"]
    for kind in ("kernels", "cpu_ops"):
        only_l = sorted(set(fl[kind]) - set(fa[kind]))
        only_r = sorted(set(fa[kind]) - set(fl[kind]))
        print(f"  {kind} only under Trainer ({len(only_l)}): {[n[:70] for n in only_l]}")
        print(f"  {kind} only in raw loop   ({len(only_r)}): {[n[:70] for n in only_r]}")

    if a.out:
        with open(a.out, "w") as f:
            json.dump({"config": cfg, "versions": pinned, "ceilings": [pf, bw], "records": recs},
                      f, default=str)
        print(f"\nraw records -> {a.out}")


if __name__ == "__main__":
    main()
