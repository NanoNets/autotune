# Beliefs — index

One line per belief file. Always loaded, so keep each line to the claim itself — the evidence lives
in the file. Retrieval is by `context` match (`{arch_family, hardware_class, dataset_shape}`); see
the `Belief` contract in `loop/contracts.md`. One heading per `type`.

## Regime — what kind of problem this is

- [static-padding-is-the-first-lever](static-padding-is-the-first-lever.md) — max_len far above real lengths →
  padding=longest first; samples/s multiplies while tokens/s and MFU fall.
- [micro-step-count-bounds-small-real-batches](micro-step-count-bounds-small-real-batches.md) — after padding, cost
  scales with micro-step count (bf16 grad-accum adds + launches), not tokens; buy micro_batch before kernel levers.

## Ceiling — what bounds the memory

## Calibration — how far a prediction or a reading is off, and when to distrust the number

- [synthetic-ceiling-misses-padded-mask](synthetic-ceiling-misses-padded-mask.md) — all-ones synthetic rows miss the
  padded-mask SDPA path (+~0.5 GB at [1, 2048]); a padded config near the card edge is unproven.
- [recompiles-show-as-mean-over-median](recompiles-show-as-mean-over-median.md) — mean step >> median with an
  unchanged GPU step = recompiles in the window; a multi-pass probe separates them from length variation.
- [full-gc-after-compile-stalls-one-step](full-gc-after-compile-stalls-one-step.md) — a one-step ~1.3s stall with 0
  recompiles is a gen-2 GC over compile-time objects; gc.freeze() after the compiling step removes it.

## Dead lever — what doesn't move throughput here

- [liger-flce-dead-when-vocab-dwarfs-hidden](liger-flce-dead-when-vocab-dwarfs-hidden.md) — Liger FLCE cuts a
  micro-batch into ~BT·H/V-token chunks; at V/H ~170 that's -72%.
- [fused-adamw-slower-on-bf16-params](fused-adamw-slower-on-bf16-params.md) — fused AdamW was slower than foreach on
  bf16 params (A10, torch 2.7).
- [max-autotune-dead-on-a10-gemms](max-autotune-dead-on-a10-gemms.md) — max-autotune-no-cudagraphs tied or lost to
  cuBLAS on A10 training GEMMs (2 models), at minutes of autotune wall.
- [hf-fa2-padded-batches-break-compile](hf-fa2-padded-batches-break-compile.md) — HF FA2 + padded mask under compile
  graph-breaks per layer (-27%); padding-free FA2 is clean but buys only the residual pad.
- [cudagraphs-crash-on-peft-graph-breaks](cudagraphs-crash-on-peft-graph-breaks.md) — reduce-overhead on a peft HF
  model crashes (cudagraph outputs overwritten across graph-break segments); mark_step_begin doesn't fix it.

## Safety — whether a knob changes the model

- [eval-mode-base-disables-gradient-checkpointing](eval-mode-base-disables-gradient-checkpointing.md) — a script that
  never calls .train() leaves HF layers in eval; gradient checkpointing is then a silent no-op.

## Invariant — how knobs interact

- [chunked-ce-costs-a-head-grad-add-per-chunk](chunked-ce-costs-a-head-grad-add-per-chunk.md) — chunked CE adds
  the head's weight grad once per chunk; use the fewest chunks that fit, and prefer compile's fused CE.
- [recompute-pays-when-it-cuts-micro-steps](recompute-pays-when-it-cuts-micro-steps.md) — recompute can win when the
  micro_batch it unlocks halves grad_accum on a micro-step-bound step.
- [length-grouping-value-scales-with-micro-batch](length-grouping-value-scales-with-micro-batch.md) — grouping is
  minor at mb2 and load-bearing at mb8; re-test after every micro_batch increase.
- [compile-the-optimizer-step](compile-the-optimizer-step.md) — torch.compile(opt.step) takes foreach AdamW to its
  bandwidth floor (+22% at accum 1 on 1.7B); pays with the optimizer's share of the step.
- [padded-mask-costs-flash-attention](padded-mask-costs-flash-attention.md) — any pad in the mask takes HF SDPA off
  flash (7.5x slower attn bwd on ~700-tok rows); right-padded causal LM doesn't need the mask, so drop it.
- [compiled-hf-sdpa-never-picks-flash](compiled-hf-sdpa-never-picks-flash.md) — under compile HF materializes a causal
  mask for SDPA; use FA2 with no mask instead (+7.8%).
- [peft-fp32-adapters-skip-tensor-cores](peft-fp32-adapters-skip-tensor-cores.md) — peft's fp32 LoRA adapters run
  as SIMT sgemm on a bf16 base; allow_tf32 puts them on tensor cores, loss inside run-to-run noise.
