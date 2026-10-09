---
claim: Gradient checkpointing is a throughput loss on its own, but it pays when the micro_batch it unlocks
  halves grad_accum on a micro-step-bound step. The saved grad-accum adds and launches, plus fatter GEMMs, can exceed
  the re-run forward. Do the arithmetic on the trace, not on "recompute = +33% compute".
type: invariant
context:
  arch_family: dense decoder (Qwen2.5-0.5B, SmolLM2-1.7B, Qwen2.5-1.5B), under torch.compile
  hardware_class: A10
  dataset_shape: real rows under padding=longest + length grouping (short wikitext rows; ~700-token cosmopedia rows)
evidence:
  probes: 4 (a10_qwen2.5-0.5b_wikitext exp 9 -> 13: mb4 -> mb8 via recompute, +11.5%, same-rows; exp 15 extended
    probe reproduced it within 0.4% over 3 passes; a10_smollm2-1.7b_wikitext exp 6 -> 8: mb4 -> mb16 / accum 1 via
    recompute, +8.9%, same-rows; exp 12: partial recompute 21/24 layers +2.2%; a10_qwen2.5-1.5b_cosmopedia exp 8 -> 10:
    mb1 -> mb4 via full recompute + grouping, +18.4%, same-rows; exp 11 partial 25/28 +1.8%, repeat pair 13/14 +1.6%)
  confidence: high (three models; it held even on long, compute-bound rows, where I predicted -5% to 0)
links: [[micro-step-count-bounds-small-real-batches]], [[compile-the-optimizer-step]]
---

GPU busy went 0.633 -> 0.601s despite the recompute: the grad-accum add fell 0.125 -> 0.060s and the top GEMMs moved
to cutlass 256x128 tiles. Host side: gap_count 24.5k -> 15.2k (width-invariant), wall step 0.811 -> 0.726s.

Once recompute has bought the micro_batch, spend leftover headroom un-checkpointing layers (hf runner
`recompute_layers`): SmolLM2-1.7B at [16, 1024], ~1.2 GB per un-checkpointed layer, 3 of 24 fit = +2.2% (backward
0.244 -> 0.233s, same reference batch). The bound is the [mb, seq, vocab] bf16 logits at the loss.

Qwen2.5-1.5B, ~700-token rows at max_len 2048 (2026-10-08): the micro-step costs were still ~20% of the GPU step at accum
16 (bf16 grad-accum add 0.358s + 7% idle). Full recompute bought mb4 (accum 4): add 0.358 -> 0.070s, idle 7% -> 0%,
gap_count 72k -> 8k. Un-checkpointed layer at [4, 2048]: ~0.54 GB. Grouping was required: without it mb4 lost -8.6% to pad.
