"""Head-to-head: SweepPolicy vs PredictivePolicy across scenarios, scored vs the brute-force optimum."""
import fake_probe
from search import search, SweepPolicy, PredictivePolicy

BASE = fake_probe.SIM
def sim(**over):
    return {**BASE, **over}

# (name, sim, eff)
SCENARIOS = [
    ("small ceiling (pf helps throughput)", sim(), 16),
    ("large ceiling (pf memory-only)",
     sim(gpu_mem_gb=80.0, mem_per_sample_gb=0.5, pf_compute=1.0), 128),
    ("large ceiling (pf helps throughput, no mem pressure)",
     sim(gpu_mem_gb=80.0, mem_per_sample_gb=0.5), 128),
]
FLAGS = [(False, False), (True, False), (True, True)]


def brute_optimum(s, eff):
    best = None
    for pf, gc in FLAGS:
        for bs in range(1, eff + 1):
            r = fake_probe.probe({"per_device_train_batch_size": bs,
                                  "gradient_accumulation_steps": max(1, eff // bs),
                                  "padding_free": pf, "gradient_checkpointing": gc}, sim=s)
            if r["status"] == "ok" and (best is None or r["tokens_per_s"] > best[1]):
                best = ((bs, pf, gc), r["tokens_per_s"])
    return best


for name, s, eff in SCENARIOS:
    probe_fn = lambda cfg, s=s: fake_probe.probe(cfg, sim=s)
    opt, opt_tk = brute_optimum(s, eff)
    print(f"\n=== {name} | eff={eff} ===")
    print(f"  TRUE optimum (brute force, {eff * len(FLAGS)} probes): "
          f"bs={opt[0]} pf={int(opt[1])} gc={int(opt[2])}  tok/s={opt_tk}")
    for label, pol in [("Sweep (baseline)", SweepPolicy(eff)), ("Predictive", PredictivePolicy(eff))]:
        out = search(probe_fn, pol)
        b = out["best"]["config"]
        tk = out["best"]["result"]["tokens_per_s"]
        got = (b["per_device_train_batch_size"], b["padding_free"], b["gradient_checkpointing"])
        tag = "OPTIMAL" if got == opt else f"{100 * (opt_tk - tk) / opt_tk:.1f}% below optimum"
        print(f"  {label:<18} {out['n_probes']:>2} probes  ->  "
              f"bs={got[0]:<3} pf={int(got[1])} gc={int(got[2])}  tok/s={tk:<8.1f} {tag}")
