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
                       env={**{k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
                            "HF_HUB_OFFLINE": "1"})           # prefetched: no hub calls mid-study
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
    if cfg.get("model"):                                          # download once, before any round
        import workload
        from huggingface_hub import snapshot_download
        rev = workload.resolve(cfg)["revision"]
        path = snapshot_download(cfg["model"], revision=rev,
                                 allow_patterns=["*.json", "*.safetensors", "*.txt"])
        print(f"prefetched {cfg['model']}@{rev} -> {path}")
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
    if L0["truth"].get("total_params") and L0["truth"]["total_params"] != L0["truth"]["params"]:
        tr, tot, tok = L0["truth"]["params"], L0["truth"]["total_params"], L0["truth"]["tokens"]
        rep = 6 * tr * tok                                   # what the capture feeds 6ND
        cor = (4 * tot + 2 * tr) * tok                       # fwd + bwd-through-activations on all
        m = [r["obs"]["mfu"] for r in recs["L"] if r["obs"]["mfu"] is not None]   # + wgrad on trainable
        print(f"  LoRA: trainable {tr:,} of {tot:,} params ({tr / tot:.2%})")
        print(f"  FLOPs/step reported (6*N_trainable*D) {rep:.3e} vs LoRA-correct ((4*N_total + 2*N_trainable)*D) {cor:.3e}: x{cor / rep:.1f}")
        if m:
            mm = sorted(m)[len(m) // 2]
            print(f"  MFU as reported {mm:.2%} -> corrected {mm * cor / rep:.2%}")
    print(f"  recovered (every round identical): {L0['recovered']} vs truth {L0['truth']}")
    loads = {json.dumps(r["loaded"], sort_keys=True) for rs in recs.values() for r in rs}
    print(f"  model as loaded (identical in every arm and round: {len(loads) == 1}): {L0['loaded']}")

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
