"""search() + SweepPolicy against the fake probe — asserts it finds the known optimum offline."""
import fake_probe
from search import search, SweepPolicy

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
