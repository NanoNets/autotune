"""Experiment 6: is the hooks-vs-callback step-time gap caused by WHERE the CUDA sync happens?

  ~/autotune_tools/venv_hf/bin/python hf/sync_placement.py --rounds 8 [--model <hub id>]

Arms, all the same Trainer script, order shuffled every round:
  L      hooks (attach.py): sync right after the inner optimizer.step()
  T_opt  TrainerCallback syncing at on_optimizer_step: the SAME point as L, no hooks
  T_end  TrainerCallback syncing at on_step_end: after lr_scheduler / zero_grad
  GT     TrainerCallback syncing only twice, `clean` steps apart: the least perturbed reference

If sync placement explains the gap: T_opt ≈ L, T_end < L, and L − T_end ≈ the host time from the
sync to on_step_end (measured in T_opt).
"""
import argparse
import json
import os
import random
import statistics as st
import subprocess
import sys

HF = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HF, os.path.dirname(HF)]

import parity                                                     # noqa: E402
import versions                                                   # noqa: E402

ARMS = {"L": ["--arm", "L"], "T_opt": ["--arm", "T", "--time-mode", "opt"],
        "T_end": ["--arm", "T", "--time-mode", "end"], "GT": ["--arm", "T", "--time-mode", "two"]}


def run(extra, cfg, pf, bw):
    p = subprocess.run([sys.executable, os.path.join(HF, "run_hf_arm.py"), *extra,
                        "--config", json.dumps(cfg), "--peak-flops", str(pf), "--peak-bw", str(bw)],
                       capture_output=True, text=True, cwd=HF,
                       env={**{k: v for k, v in os.environ.items() if k != "PYTHONPATH"},
                            "HF_HUB_OFFLINE": "1"})
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-3000:])
    return json.loads(p.stdout.strip().splitlines()[-1])


def fit(rows, arms, ref):
    """Least squares: step time ~ round + position + arm. Returns {arm: (effect_ms, se_ms)} vs ref."""
    import numpy as np
    k, others = len(rows), [a for a in arms if a != ref]
    X, y = [], []
    for ri, r in enumerate(rows):
        for p, a in enumerate(r["order"]):
            x = np.zeros(k + (len(arms) - 1) + len(others))
            x[ri] = 1
            if p > 0:
                x[k + p - 1] = 1
            if a != ref:
                x[k + len(arms) - 1 + others.index(a)] = 1
            X.append(x)
            y.append(r[a]["mean_s"] * 1e3)
    X, y = np.array(X), np.array(y)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    s2 = ((y - X @ b) ** 2).sum() / (len(y) - X.shape[1])
    se = np.sqrt(np.diag(s2 * np.linalg.pinv(X.T @ X)))
    base = k + len(arms) - 1
    return {a: (b[base + i], se[base + i]) for i, a in enumerate(others)}, float(np.sqrt(s2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    pinned = versions.check()
    cfg = {"model": a.model}
    import workload
    from huggingface_hub import snapshot_download
    snapshot_download(a.model, revision=workload.resolve(cfg)["revision"],
                      allow_patterns=["*.json", "*.safetensors", "*.txt"])
    pf, bw = parity.ceilings("cuda", "bfloat16")
    print(f"versions {pinned}\nmodel {a.model}")

    rng, rows = random.Random(0), []
    for i in range(a.rounds):
        order = list(ARMS)
        rng.shuffle(order)
        row = {"order": order}
        for arm in order:
            r = run(ARMS[arm], cfg, pf, bw)
            t = r["step_times_s"]
            row[arm] = {"mean_s": st.mean(t), "median_s": st.median(t),
                        "host_after_sync_s": r.get("host_after_sync")}
        rows.append(row)
        print(f"round {i + 1} {order}: " + "  ".join(f"{k} {row[k]['mean_s']:.4f}" for k in ARMS),
              flush=True)

    eff, sd = fit(rows, list(ARMS), ref="GT")
    print(f"\nmean step time vs GT (least perturbed), adjusted for round + position; resid sd {sd:.2f} ms")
    for arm, (m, se) in eff.items():
        print(f"  {arm:6s} {m:+.2f} ± {se:.2f} ms")
    host = [x * 1e3 for r in rows for x in (r["T_opt"]["host_after_sync_s"] or [])]
    print(f"\nhost time from the sync at on_optimizer_step to on_step_end (T_opt): "
          f"median {st.median(host):.2f} ms, mean {st.mean(host):.2f} ms over {len(host)} steps")
    lt = eff["L"][0] - eff["T_end"][0]
    to = eff["T_opt"][0] - eff["T_end"][0]
    print(f"L − T_end = {lt:+.2f} ms;  T_opt − T_end = {to:+.2f} ms;  L − T_opt = "
          f"{eff['L'][0] - eff['T_opt'][0]:+.2f} ms")
    if a.out:
        json.dump({"model": a.model, "versions": pinned, "rows": rows}, open(a.out, "w"))


if __name__ == "__main__":
    main()
