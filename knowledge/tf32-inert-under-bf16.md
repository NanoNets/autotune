---
claim: tf32 has no measurable effect on a pure-bf16 training config — there is no fp32 matmul in the
  graph for it to accelerate.
type: dead-lever
context:
  arch_family: any transformer trained in bf16
  hardware_class: Ampere+ (tf32 requires SM80+ to exist at all)
  dataset_shape: any
evidence:
  probes: 1 (experiment 5)
  confidence: high — mechanism is structural, not empirical luck
links: [[ce-dominates-activation-memory-large-vocab]]
---

Measured -0.67% (12940 → 12853 tokens/s), inside the noise floor (cv ≈ 0.005), with peak memory
**byte-identical** at 18.706 GB — which is the tell that nothing changed at all.

TF32 only accelerates fp32 **matmul**. In a bf16 run every GEMM in the trace is already
`ampere_bf16_s16816gemm` / `s1688gemm`, and the only fp32 work is the cross-entropy softmax and the
AdamW update — both elementwise. So the knob has nothing to act on.

**Generalisation:** expect this for any pure-bf16 training job, not just this one. `tf32` is only a
live lever where fp32 matmuls actually remain (mixed-precision setups that keep some ops in fp32).
Not worth a probe unless the trace shows fp32 GEMMs.
