# Autotune Results: Qwen/Qwen3.5-0.8B

**Machine:** 1x 80GB GPU, 16 vCPUs
**Extension active on optimal runs:** `padding_free=true` (gradient_checkpointing **off**)

> **Corrected against the numerator bug.** Throughput is now `effective_batch / mean_s_it`
> (was `bs / mean_s_it`). The recorded `results.tsv` numbers were the buggy reported values;
> the table below is `reported × ga` = true samples/s. The earlier "optimal = bs=16 + gc"
> configs were the **slowest** measured (the bug rewarded large `bs`); the true optimum is
> small `bs`, gc off. See `notes.md` Findings A/B.

## Optimal Configurations (corrected)

| eff_batch | bs | ga | gc  | true samples/s | old "optimal" (bs=16+gc) | speedup |
|-----------|----|----|-----|----------------|--------------------------|---------|
| 16        | 2  | 8  | off | ~0.588         | 0.169                    | ~3.5×   |
| 32        | 2  | 16 | off | ~0.544         | 0.167                    | ~3.3×   |
| 64        | 2  | 32 | off | ~0.579         | 0.182                    | ~3.2×   |

True samples/s = `reported × ga` from `results.tsv` Phase-1 rows. Full per-`bs` breakdown:

| bs | flags  | true samp/s (≈, ga-invariant) |
|----|--------|-------------------------------|
| 1  | pf     | ~0.55                         |
| 2  | pf     | **~0.57 ← best**              |
| 4  | pf+gc  | ~0.43                         |
| 8  | pf+gc  | ~0.29                         |
| 16 | pf+gc  | ~0.17 ← old "optimal" (worst) |

## Key Findings (corrected)

- **The GPU saturates by `bs=2`.** bs=1→~0.55, bs=2→~0.57; pushing `bs` higher buys nothing.
- **`gradient_checkpointing` only hurts here.** It was forced on at bs≥4 solely to reach the
  pointless high batch sizes the buggy metric rewarded, adding ~30% recompute for no gain.
  bs=2 fits with `padding_free` alone — no gc needed.
- **True throughput is flat across effective batches at fixed `bs`** (the ga-invariant), as
  physics requires. The old "throughput halves as eff_batch doubles" was the numerator artifact.
- **Dataloader tuning is irrelevant** (Phase 2 deltas <2%) — GPU-bound. Note: Phase 2 was swept
  at the wrong bs=16; at bs=2 the dataloader knobs are unmeasured but expected equally irrelevant.

## Caveats

- Absolute values are single, short (7-step) probes — uncorroborated by repeats. The **ranking**
  is robust (the ~3.4× flip dwarfs any plausible noise); the absolute samp/s are indicative only.
- bs=2 vs bs=1 is a small (~4–8%) edge, consistent across all three effective batches — likely
  real but near the noise floor. Tie-breaks to bs=2.

## swift sft Overrides (corrected)

For production training, add these overrides on top of `sft_base.yaml`:

### eff_batch=16
```yaml
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
gradient_checkpointing: false
padding_free: true
dataloader_persistent_workers: true
```

### eff_batch=32
```yaml
per_device_train_batch_size: 2
gradient_accumulation_steps: 16
gradient_checkpointing: false
padding_free: true
dataloader_persistent_workers: true
```

### eff_batch=64
```yaml
per_device_train_batch_size: 2
gradient_accumulation_steps: 32
gradient_checkpointing: false
padding_free: true
dataloader_persistent_workers: true
```
