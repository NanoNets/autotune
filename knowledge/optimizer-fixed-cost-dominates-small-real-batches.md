---
claim: The optimizer's cost is per-PARAMETER, so it does not shrink with the batch — once dynamic
  padding cuts real tokens/step to a few thousand, AdamW can become the majority of the step, and
  AdamW(fused=True) is the fix.
type: regime
context:
  arch_family: any model large enough that param count dwarfs tokens/step (1.7B params vs ~1.2k
    real tokens/step here); most acute at grad_accum=1
  hardware_class: any CUDA
  dataset_shape: short rows under dynamic padding — i.e. downstream of
    [[static-padding-is-usually-the-bottleneck]]
evidence:
  probes: 4 (smollm2_1.7b_a100_wikitext experiments 10, 13, 15, 36) + the gpu_phase_times_s
    decomposition that predicted it
  confidence: high — diagnosed from the trace before probing, then measured twice at two samples
links: [[static-padding-is-usually-the-bottleneck]],
  [[micro-batch-interacts-with-dynamic-padding]]
---

Every lever in `loop/throughput.md` that touches the *model* scales with tokens. The optimizer does
not: AdamW touches each parameter once per step whatever the batch contains. Fix the padding problem
well enough and you invert which one is the bottleneck.

**Diagnosed from the Observation, not guessed.** At the point this was found the run had reached 155
samples/s with idle at 1% — no launch overhead left to recover, so the next move had to come from the
work itself. `gpu_phase_times_s` said where it was:

| phase | seconds | share |
|---|---|---|
| forward | 0.0121 | 12% |
| backward | 0.0333 | 33% |
| **optimizer (AdamW.step)** | **0.0565** | **55%** |

and 5 of the top 8 kernels were `multi_tensor_apply` (the foreach AdamW). 1.7B parameters were being
updated to consume ~1.2k real tokens.

**The fix is a kernel swap, not an algorithm change.** `torch.optim.AdamW(fused=True)` replaces the
foreach kernel chain with one fused kernel — same AdamW, so it is a speed knob, and
`loop/throughput.md` already lists "fused optimizer" under Kernels. It was simply not exposed by
`frameworks/hf.py`; adding it was three lines. Optimizer time **0.0565 → 0.0203s**, and peak memory
also fell 15.96 → 13.34 GB (fewer intermediates).

**Effect size depends on how little work surrounds it** — which is the whole point of the claim:

| measured at | gain from fusing |
|---|---|
| `sample=256` (tokens/step 1088) | **+55%** |
| `sample=1000` (tokens/step 2462) | **+19%** |

A fixed cost matters in proportion to how little else is in the step.

**What to do:** when a dynamic-padding run reaches high GPU-active and low idle, read
`gpu_phase_times_s` before reaching for another model-side lever. If `Optimizer.step#...` is a large
share, fuse it first — it is one flag and it is free. Note also that fused AdamW cuts the *fixed*
memory term only: re-measuring the ceiling with it changed nothing from mb=4 upward (still 12 under
compile), because that ceiling is activation-bound. It is a throughput lever, not a ceiling lever.
