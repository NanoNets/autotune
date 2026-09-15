# Beliefs — index

One line per belief file. Always loaded, so keep each line to the claim itself — the evidence lives
in the file. Retrieval is by `context` match (`{arch_family, hardware_class, dataset_shape}`); see
the `Belief` contract in `loop/contracts.md`.

## Regime — what kind of problem this is

- [static-padding-is-usually-the-bottleneck](static-padding-is-usually-the-bottleneck.md) — when
  `max_len` sits far above the data's real lengths, padding is the dominant cost and fixing it at the
  **loader** outweighs every kernel lever combined. Check p50/p99 vs `max_len` before any kernel work.
- [padding-regime-flips-attention-verdict](padding-regime-flips-attention-verdict.md) —
  `flash_attention_2` beats `sdpa` only while batches carry heavy padding; remove the padding and its
  unpad/repad overhead makes it a net loss.

## Ceiling — what bounds the memory

- [ce-dominates-activation-memory-large-vocab](ce-dominates-activation-memory-large-vocab.md) — on a
  small model with a large vocab, the logits/cross-entropy term (~70%), not attention, dominates
  activation memory at long `max_len`, and recompute cannot touch it.
- [compile-cuts-memory-not-just-time](compile-cuts-memory-not-just-time.md) — `torch.compile` fuses
  the CE and avoids the fp32 logits copies, cutting peak memory and **raising the OOM ceiling**. It
  stales the ceiling like any memory knob.

## Invariant — how knobs interact

- [micro-batch-interacts-with-dynamic-padding](micro-batch-interacts-with-dynamic-padding.md) — under
  pad-to-longest a larger `micro_batch` is cheaper per token but wastier per sample; length grouping
  is what makes it pay.
- [probe-sample-must-match-length-distribution](probe-sample-must-match-length-distribution.md) —
  under dynamic padding, batch width IS the rows drawn, so an unrepresentative probe sample measures
  a workload the run won't train on. Static padding hides this. Mechanism argued, effect size not
  yet measured.

## Dead levers & safety

- [tf32-inert-under-bf16](tf32-inert-under-bf16.md) — no effect on a pure-bf16 graph; there is no fp32
  matmul to accelerate. Don't spend a probe unless the trace shows fp32 GEMMs.
- [cudagraphs-incompatible-with-grad-accum](cudagraphs-incompatible-with-grad-accum.md) — any
  cudagraph-enabling compile mode breaks a runner that invokes the model several times per optimizer
  step, and `cudagraph_mark_step_begin()` does not fix it.
