---
claim: When real tokens/step is small, the step is bound by the NUMBER of kernel launches, not by
  GPU work. Kernel gaps track grad_accum and wall-clock tracks gaps, so only levers that cut launch
  count move throughput — levers that cut GPU work return nothing.
type: regime
context:
  arch_family: any, but acute on models with many small ops per layer — hybrid linear-attention
    (Mamba-family gated delta rule) is the extreme case, especially on its reference PyTorch path
  hardware_class: any CUDA fast enough to outrun a single dispatching CPU core; a big card makes it
    worse, not better
  dataset_shape: short sequences relative to max_len, i.e. exactly the regime dynamic padding
    creates once it succeeds
evidence:
  probes: 12 (qwen3.5_2b_a100_wikitext experiments 1, 3-6, 9, 10, 13, 15, 16, 19, 22)
  confidence: high — the ladder was walked in four rungs, three null levers confirmed the negative,
    and two predictions made in advance about gap_count held
links: [[optimizer-fixed-cost-dominates-small-real-batches]],
  [[recompute-pays-only-instrumentally]],
  [[tf32-inert-under-bf16]],
  [[linear-attention-kernel-library-is-the-lever]]
---

The signature: **`cpu_cores_busy` pinned at 1.00 of 30 vCPUs**, GPU-active 40-60%, `data_wait` 0%.
One core saturated issuing kernels while the GPU waits.

**The ladder.** Kernel gaps per step against `grad_accum`, all other knobs held:

| grad_accum | 8 (pre-fla) | 8 | 4 | 2 | 1 |
|---|---|---|---|---|---|
| gaps | 84,585 | 43,113 | 21,353 | 12,883 | 6,487 |
| step (s) | 1.383 | 1.146 | 0.623 | 0.560 | 0.444 |

Gaps halve with `grad_accum` almost exactly, and wall-clock follows. Every large win in that run
came from cutting launches: the fla kernel swap (fewer kernels per layer, +20.7%) and each
`micro_batch` doubling (fewer micro-steps, +83.8% at one rung).

**The negative half is what makes it a regime claim rather than a correlation.** Three levers cut
GPU-side cost substantially and returned nothing:

- `tf32` — GPU phase time **-8.8%**, throughput -3.6% (inside noise)
- `causal_conv1d` kernel — GPU busy **-5.5%**, throughput -0.5% (inside noise)
- `pad_multiple` 8 -> 256 — **2.5x more work per launch**, GPU-active 42% -> 82%, throughput -10.7%

That last one is the cleanest: it bought occupancy by wasting tokens and still lost. **We were not
short of GPU; we were short of work per launch.**

**Two predictions this model made in advance, both held.** (1) Recompute would not halve gaps when
it halved micro-steps, because its recomputed forward re-issues kernels — predicted ~0.65x,
measured 0.63x. (2) Turning off `length_grouped` would cost throughput with `gap_count` UNCHANGED,
because grouping changes batch width and not batch count — measured -16.9% with gaps 12,883 ->
12,932 (+0.4%).

**Diagnostic recipe.** Read `cpu_cores_busy` and `gap_count` before reading MFU. If one core is
pegged and GPU-active is well under 100%, rank candidate levers by how many launches they remove,
not by how much GPU time they save — and distrust any estimate of a lever's worth that is derived
from its share of GPU phase time (it understated `fused_optimizer` by 3.2 points, 4.9% predicted
vs 8.1% measured, precisely because fused AdamW removes *launches* as well as work).
