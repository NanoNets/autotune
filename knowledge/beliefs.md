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
- [optimizer-fixed-cost-dominates-small-real-batches](optimizer-fixed-cost-dominates-small-real-batches.md)
  — the optimizer's cost is per-PARAMETER, so it does not shrink with the batch; once padding is
  gone it can be the majority of the step. Read `gpu_phase_times_s`, then `AdamW(fused=True)`.

## Ceiling — what bounds the memory

- [ce-dominates-activation-memory-large-vocab](ce-dominates-activation-memory-large-vocab.md) — on a
  small model with a **large** vocab, the logits/cross-entropy term (~70%), not attention, dominates
  activation memory at long `max_len`, and recompute cannot touch it.
- [ce-does-not-dominate-activation-memory-modest-vocab](ce-does-not-dominate-activation-memory-modest-vocab.md)
  — its sibling and counter-case: at a **modest** `vocab x seq`, CE is ~9% and the layer term rules,
  so recompute moves the ceiling hard (8 → >=16). Discriminate on `vocab*seq*4B` vs the model state.
- [compile-cuts-memory-not-just-time](compile-cuts-memory-not-just-time.md) — `torch.compile` fuses
  the CE and avoids the fp32 logits copies, cutting peak memory and **raising the OOM ceiling**. It
  stales the ceiling like any memory knob. Holds even where CE does not dominate.

## Invariant — how knobs interact

- [micro-batch-interacts-with-dynamic-padding](micro-batch-interacts-with-dynamic-padding.md) — under
  pad-to-longest a larger `micro_batch` is cheaper per token but wastier per sample; length grouping
  is what makes it pay, and the optimum is interior and moves with the probe's sample.
- [recompute-pays-only-instrumentally](recompute-pays-only-instrumentally.md) — gradient checkpointing
  is never a throughput lever on its own; it pays only when the ceiling it buys crosses a
  factorization boundary of `effective_batch`. Re-do that arithmetic when the token count changes.

## Measurement — when to distrust the number

- [probe-sample-must-match-length-distribution](probe-sample-must-match-length-distribution.md) —
  under dynamic padding, batch width IS the rows drawn, so an unrepresentative probe sample measures a
  workload the run won't train on. **Measured at 1.9x**; a bigger head slice does not converge.
- [median-step-time-overstates-skewed-throughput](median-step-time-overstates-skewed-throughput.md) —
  a rate built on a MEDIAN step time is biased up whenever step times are right-skewed, and length
  grouping makes them so. MFU > 100% is the tell. Report the mean; more steps does not fix it.

## Dead levers & safety

- [tf32-inert-under-bf16](tf32-inert-under-bf16.md) — no effect on a pure-bf16 graph; there is no fp32
  matmul to accelerate. Don't spend a probe unless the trace shows fp32 GEMMs.
- [cudagraphs-incompatible-with-grad-accum](cudagraphs-incompatible-with-grad-accum.md) — any
  cudagraph-enabling compile mode **crashes** a runner that invokes the model several times per
  optimizer step, and `cudagraph_mark_step_begin()` does not fix it.
- [cudagraphs-incompatible-with-dynamic-shapes](cudagraphs-incompatible-with-dynamic-shapes.md) — the
  second blocker behind it: even at `grad_accum=1` cudagraphs are unusable under `padding="longest"`
  (39x slower, re-capture per shape). Reaching grad_accum=1 does NOT unlock them.
