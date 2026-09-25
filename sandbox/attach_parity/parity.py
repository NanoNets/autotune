"""Parity experiment: record_function labels (arm A) vs Attach hooks (arm C).

  python sandbox/attach_parity/parity.py --device cpu --rounds 5
  python sandbox/attach_parity/parity.py --device cuda --rounds 5 --config '{"hidden":2048,"layers":8,"micro_batch":32,"seq":512}'

Two questions:
  1. Information loss: does C record everything A records? (annotations, CPU ops, kernels,
     which Observation fields are populated, recovered params/tokens)
  2. Measurement: are C's numbers within tolerance of A's? Tolerance = the noise floor, the
     loop's own threshold: a difference smaller than step_time_cv is one the loop can't act on.

Ceilings are measured ONCE here and frozen into both arms (profiler/README.md). Arms alternate
A C A C ... each in a fresh process, so drift over time hits both equally.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                          # the sandbox, and nothing else

# Label A emits -> the label C carries the same work under. `optimizer` in A wraps step +
# zero_grad; C has only torch's own optimizer annotations (A has them too, nested inside).
LABEL_MAP = {"forward": "forward", "backward": "backward",
             "Optimizer.step#AdamW.step": "Optimizer.step#AdamW.step",
             "optimizer": "Optimizer.step#AdamW.step", "step": "step"}


def torch_dataloader(r):
    """Host time in torch's own `enumerate(DataLoader)#...__next__` annotations."""
    ph = r["obs"]["phase_times_s"]
    hits = [v for k, v in ph.items() if k.startswith("enumerate(DataLoader)")]
    return sum(hits) if hits else None


def unlabeled_gpu(r):
    """GPU time inside `step` but outside forward/backward/optimizer: work no phase claims.
    Optimizer is read from torch's own annotation in both arms so the two are comparable."""
    g = r["obs"]["gpu_phase_times_s"]
    if "step" not in g:
        return None
    return g["step"] - sum(g.get(k, 0.0) for k in ("forward", "backward", "Optimizer.step#AdamW.step"))


def ceilings(device, dtype):
    import torch
    from vendor.capture import measure_peak_bandwidth_gb_s, measure_peak_flops
    dt = getattr(torch, dtype) if dtype else torch.get_default_dtype()
    return measure_peak_flops(device, dt), measure_peak_bandwidth_gb_s(device)


def run(arm, cfg, pf, bw, warmup, clean, runner):
    out = subprocess.run([sys.executable, os.path.join(HERE, "run_arm.py"), "--arm", arm,
                          "--runner", runner,
                          "--config", json.dumps(cfg), "--peak-flops", str(pf),
                          "--peak-bw", str(bw), "--warmup", str(warmup), "--clean", str(clean)],
                         capture_output=True, text=True, cwd=HERE,
                         env={k: v for k, v in os.environ.items() if k != "PYTHONPATH"})
    if out.returncode != 0:
        raise RuntimeError(f"arm {arm} failed:\n{out.stderr[-3000:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def rel(a, c):
    return None if a in (None, 0) or c is None else (c - a) / a


def spread(xs):
    """Round-to-round relative spread within one arm: (max - min) / median."""
    xs = [x for x in xs if x is not None]
    return (max(xs) - min(xs)) / statistics.median(xs) if len(xs) > 1 and statistics.median(xs) else None


def fmt(x, pct=False):
    if x is None:
        return "—"
    return f"{x:+.1%}" if pct else (f"{x:.4g}" if isinstance(x, float) else str(x))


def info_loss(A, C, x="C"):
    """Per round pair: what A recorded that arm `x` didn't (and vice versa)."""
    lines = []
    fa, fc = A["fingerprint"], C["fingerprint"]
    aa, ac = set(fa["annotations"]), set(fc["annotations"])
    lines.append(f"annotations only in A: {sorted(aa - ac)}")
    lines.append(f"annotations only in {x}: {sorted(ac - aa)}")
    for kind in ("cpu_ops", "kernels"):
        a, c = fa[kind], fc[kind]
        diff = {k: (a.get(k, 0), c.get(k, 0)) for k in set(a) | set(c) if a.get(k, 0) != c.get(k, 0)}
        lines.append(f"{kind}: A {sum(a.values())} events / {len(a)} names, "
                     f"{x} {sum(c.values())} / {len(c)}; count mismatches (A, {x}): {diff or 'none'}")
    oa, oc = A["obs"], C["obs"]
    lost = [k for k in oa if oa[k] not in (None, {}, []) and oc.get(k) in (None, {}, [])]
    gained = [k for k in oc if oc[k] not in (None, {}, []) and oa.get(k) in (None, {}, [])]
    lines.append(f"Observation fields populated in A but empty in {x}: {lost or 'none'}")
    lines.append(f"Observation fields populated in {x} but empty in A: {gained or 'none'}")
    # populated is not enough: a field can be filled with a wrong 0.0 (it happened: data_wait_s)
    zeroed = [k for k in oa if isinstance(oa[k], (int, float)) and not isinstance(oa[k], bool)
              and oa[k] > 0 and oc.get(k) == 0]
    lines.append(f"Observation numbers > 0 in A but exactly 0 in {x}: {zeroed or 'none'}")
    lines.append(f"recovered by hooks: {x} {C['recovered']} vs truth {C['truth']}")
    return lines


def rows_for():
    rows = [("step_time_s", lambda r: r["obs"]["step_time_s"]),
            ("profiled_step_s", lambda r: r["obs"]["profiled_step_s"]),
            ("gpu_busy_s", lambda r: r["obs"]["gpu_busy_s"]),
            ("mfu", lambda r: r["obs"]["mfu"]),
            ("cpu_cores_busy", lambda r: r["obs"]["cpu_cores_busy"]),
            ("data_wait_s (the Observation field)", lambda r: r["obs"]["data_wait_s"]),
            ("data_wait_frac (the Observation field)", lambda r: r["obs"]["data_wait_frac"])]
    for la, lc in LABEL_MAP.items():
        rows.append((f"host {la} ~ {lc}",
                     (lambda la: lambda r: r["obs"]["phase_times_s"].get(la))(la),
                     (lambda lc: lambda r: r["obs"]["phase_times_s"].get(lc))(lc)))
        rows.append((f"gpu  {la} ~ {lc}",
                     (lambda la: lambda r: r["obs"]["gpu_phase_times_s"].get(la))(la),
                     (lambda lc: lambda r: r["obs"]["gpu_phase_times_s"].get(lc))(lc)))
    rows.append(("unlabeled GPU s (step − fwd − bwd − opt)", unlabeled_gpu))
    # data-wait: A's own `dataloader` label vs the label torch emits by itself
    rows.append(("host data-wait: A dataloader ~ enumerate(DataLoader)",
                 lambda r: r["obs"]["phase_times_s"].get("dataloader"), torch_dataloader))
    rows.append(("host enumerate(DataLoader), both arms", torch_dataloader))
    return rows


def report(recs, x, rounds):
    A, X = recs["A"], recs[x]
    print(f"\n# A vs {x}")
    print("\n## 1. Information loss (round 1)")
    for line in info_loss(A[0], X[0], x):
        print("  " + line)
    stable = all(info_loss(A[i], X[i], x)[:4] == info_loss(A[0], X[0], x)[:4] for i in range(rounds))
    print(f"  same set/count differences in every round: {stable}")

    print("\n## 2. Measurements (median over rounds)")
    noise = med([r["obs"]["step_time_cv"] for r in A + X])
    print(f"  tolerance = noise floor (median step_time_cv, both arms) = {noise:.2%}\n")
    print(f"  {'field':44} {'A':>10} {x:>10} {'Δ vs A':>9} {'A spread':>9} {x + ' spread':>9}  within noise?")
    for row in rows_for():
        name, fa = row[0], row[1]
        fx = row[2] if len(row) > 2 else fa
        va, vx = [fa(r) for r in A], [fx(r) for r in X]
        ma, mx = med(va), med(vx)
        d = rel(ma, mx)
        ok = "—" if d is None else ("yes" if abs(d) <= noise else "NO")
        print(f"  {name:44} {fmt(ma):>10} {fmt(mx):>10} {fmt(d, True):>9} "
              f"{fmt(spread(va), True):>9} {fmt(spread(vx), True):>9}  {ok}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--runner", choices=["mlp", "loader"], default="mlp")
    ap.add_argument("--arms", default="C", help="arms to compare against A, e.g. C or C,L")
    ap.add_argument("--config", default="{}", help="extra runner config JSON")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--clean", type=int, default=10)
    ap.add_argument("--out", default=None, help="write raw records here (JSON)")
    a = ap.parse_args()
    cfg = {"device": a.device, **json.loads(a.config)}
    others = a.arms.split(",")
    arms = ["A"] + others

    pf, bw = ceilings(a.device, cfg.get("dtype"))
    print(f"frozen ceilings: peak_flops {pf:.4g}  peak_bw {bw:.4g} GB/s   config {cfg}")
    recs = {k: [] for k in arms}
    for i in range(a.rounds):
        for arm in arms:
            recs[arm].append(run(arm, cfg, pf, bw, a.warmup, a.clean, a.runner))
        print(f"round {i + 1}: " + "  ".join(f"{k} {recs[k][-1]['obs']['step_time_s']:.5f}s" for k in arms))

    for x in others:
        report(recs, x, a.rounds)

    if a.out:
        with open(a.out, "w") as f:
            json.dump({"config": cfg, "ceilings": [pf, bw], "records": recs}, f, default=str)
        print(f"\nraw records -> {a.out}")


if __name__ == "__main__":
    main()
