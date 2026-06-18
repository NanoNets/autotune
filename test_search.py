"""search() + policies against the fake probe — asserts they find the known optimum offline."""
import fake_probe
from search import search, SweepPolicy, PredictivePolicy

out = search(fake_probe.probe, SweepPolicy(effective_batch=16))

print(f"probes run: {out['n_probes']}")
for h in out["history"]:
    c, r = h["config"], h["result"]
    flags = ("pf " if c["padding_free"] else "") + ("gc" if c["gradient_checkpointing"] else "")
    print(f"  bs={c['per_device_train_batch_size']:<2} {flags:<6} "
          f"{r['status']:<4} sps={r['samples_per_s']}")

best = out["best"]["config"]
sps = out["best"]["result"]["samples_per_s"]
print(f"\nbest: bs={best['per_device_train_batch_size']} "
      f"pf={best['padding_free']} gc={best['gradient_checkpointing']} sps={sps}")

# known optimum of the default SIM: bs=8, padding_free, NO gc (gc fits bs=16 but its tax is slower)
assert best["per_device_train_batch_size"] == 8, best
assert best["padding_free"] is True, best
assert best["gradient_checkpointing"] is False, "should NOT pick the deeper bs=16+gc trap"
assert abs(sps - 1.6667) < 1e-3, sps
print("\n✓ baseline finds the known optimum (bs=8, padding_free, no gc)")

# PredictivePolicy: bisects to the real ceiling (bs=11), which the sweep's powers-of-2 miss
pred = search(fake_probe.probe, PredictivePolicy(effective_batch=16))["best"]["config"]
assert pred["per_device_train_batch_size"] == 11, pred
assert pred["padding_free"] is True and pred["gradient_checkpointing"] is False, pred
print("✓ predictive finds the finer optimum (bs=11, padding_free, no gc)")

# PredictivePolicy: tries padding_free as a throughput lever even with no memory pressure;
# the baseline (pf only on OOM) misses it on a high-ceiling scenario.
big = dict(fake_probe.SIM, gpu_mem_gb=80.0, mem_per_sample_gb=0.5)
probe = lambda cfg: fake_probe.probe(cfg, sim=big)
pred_big = search(probe, PredictivePolicy(128))["best"]
sweep_big = search(probe, SweepPolicy(128))["best"]
assert pred_big["config"]["padding_free"] is True, "predictive must try pf even with no mem pressure"
assert pred_big["result"]["tokens_per_s"] > sweep_big["result"]["tokens_per_s"], "predictive should beat baseline"
print("✓ predictive uses padding_free as a throughput win the baseline misses")
