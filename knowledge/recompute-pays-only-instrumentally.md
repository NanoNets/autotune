---
claim: Gradient checkpointing is never a throughput lever on its own — it is only ever worth taking
  when the ceiling it buys crosses a factorization boundary that pays more than its recompute costs.
  Re-check that arithmetic whenever the workload's real token count changes.
type: invariant
context:
  arch_family: any
  hardware_class: any CUDA where the ceiling is activation-bound
  dataset_shape: any, but the arithmetic is most volatile under dynamic padding, where real
    tokens/step depends on the rows drawn
evidence:
  probes: 6 (smollm2_1.7b_a100_wikitext experiments 9, 10, 12, 15, 33, 37/40) + 3 ceiling searches
  confidence: high — the tradeoff was measured on both sides of the flip and the winner confirmed
    twice to 0.04%
links: [[ce-does-not-dominate-activation-memory-modest-vocab]],
  [[micro-batch-interacts-with-dynamic-padding]],
  [[median-step-time-overstates-skewed-throughput]]
---

Recompute costs throughput by construction — it re-runs the forward. Measured standalone here:
**-35%** (52.1 vs 80.7 samples/s). It can still be the right call, but only indirectly: by raising the
OOM ceiling to a `micro_batch` that divides `effective_batch` better, cutting `grad_accum` and with it
the fixed per-step costs.

**Both sides of that trade, measured on one run:**

| | ceiling | micro_batch | grad_accum | samples/s |
|---|---|---|---|---|
| no recompute, compile | 12 | 8 | 2 | **102.7** |
| recompute, compile | >=16 | 16 | **1** | 83.8 |

At `sample=256` the second row won by **+49%**. At `sample=1000` the first wins by **+22.6%**. Same
hardware, same knobs, opposite verdict.

**Why it flips.** The recompute cost is *per token*; the grad_accum saving is *per step*. Widen the
real batch and the per-step saving is amortized over more work while the per-token cost is not. So the
break-even moves with the workload's real token count — which, under dynamic padding, is a property
of the probe's row sample ([[probe-sample-must-match-length-distribution]]), not of the config.

**The sharp edge: divisibility, not memory.** `micro_batch` must divide `effective_batch`. Compile
alone lifted the ceiling 8 → 12, and **12 bought nothing** because it does not divide 16 — four
samples of measured headroom, unusable. Recompute's entire value was getting from 12 to 16. A ceiling
only pays at a legal factorization, so the question is never "how much did the ceiling move" but
"did it cross the next divisor, and is that divisor worth the recompute".

**What to do:** treat recompute as a *memory* lever whose throughput value is derived, and recompute
the arithmetic — do not inherit the verdict — whenever `effective_batch`, `max_len`, the ceiling, or
the measured token count changes. In this run the trigger was neither a knob nor the hardware: it was
correcting the measurement.
