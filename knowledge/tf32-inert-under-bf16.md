---
claim: tf32 has no measurable effect on a pure-bf16 training config — there is no fp32 matmul in the
  graph for it to accelerate.
type: dead-lever
context:
  arch_family: any transformer trained in bf16
  hardware_class: Ampere+ (tf32 requires SM80+ to exist at all)
  dataset_shape: any
evidence:
  probes: 2 (test_a10G experiment 5; qwen3.5_2b_a100_wikitext experiment 4)
  confidence: high for the VERDICT; the stated mechanism is NOT universal — see the 2026-09-21 note
links: [[ce-dominates-activation-memory-large-vocab]]
---

> **2026-09-21 (qwen3.5_2b_a100_wikitext) — the precondition failed, and the guard caught it.**
> This belief says tf32 is inert because "there is no fp32 matmul in the graph". On a **hybrid
> linear-attention** model that premise is false: `mamba_ssm_dtype: float32` makes the reference
> gated-delta-rule run its recurrence in fp32, and `ampere_sgemm_128x128_{nn,tn,nt}` were **three of
> the top six kernels** in a bf16 run. The belief's own closing guard — *"not worth a probe unless
> the trace shows fp32 GEMMs"* — is exactly what made this visible, and it fired correctly.
>
> The probe was spent and tf32 **did** engage: all three sgemm kernels vanished and GPU phase time
> fell **8.8%** (0.8013 -> 0.7306s). Throughput did not move (-3.6%, inside a ±6% band), because
> that step was launch-bound rather than compute-bound
> ([[dispatch-count-bounds-small-real-batches]]).
>
> So the VERDICT held for a different reason than the claim gives, on a model where the claim's
> mechanism did not apply. **Then it was restored:** once `flash-linear-attention` was installed,
> the fp32 GEMMs disappeared entirely and the original mechanism holds again for any post-fla
> config. Read this belief as: *inert whenever the trace has no fp32 GEMMs — and on hybrid
> linear-attention models, check, because the reference kernels put them there.*

Measured -0.67% (12940 → 12853 tokens/s), inside the noise floor (cv ≈ 0.005), with peak memory
**byte-identical** at 18.706 GB — which is the tell that nothing changed at all.

TF32 only accelerates fp32 **matmul**. In a bf16 run every GEMM in the trace is already
`ampere_bf16_s16816gemm` / `s1688gemm`, and the only fp32 work is the cross-entropy softmax and the
AdamW update — both elementwise. So the knob has nothing to act on.

**Generalisation:** expect this for any pure-bf16 training job, not just this one. `tf32` is only a
live lever where fp32 matmuls actually remain (mixed-precision setups that keep some ops in fp32).
Not worth a probe unless the trace shows fp32 GEMMs.
