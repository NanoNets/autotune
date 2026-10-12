---
claim: Once padding is gone and real tokens/step are small (~7-9k), the cost scales with the NUMBER of
  micro-steps, not the tokens. The bf16 grad-accumulation add (one full-parameter read-modify-write per micro-step)
  becomes the top kernel, and launches leave the GPU idle. Buying a larger micro_batch (via any memory lever) beats
  any kernel lever, and pad trimming pays only once the step is GPU-bound again.
type: regime
context:
  arch_family: dense decoder, full fine-tune (Qwen2.5-0.5B 494M; SmolLM2-1.7B)
  hardware_class: A10 (Ampere, 22GB, 447 GB/s)
  dataset_shape: short real rows (mean ~106 tok) under padding=longest, effective_batch 64 or 16
evidence:
  probes: 9 (a10_qwen2.5-0.5b_wikitext exps 2->3 +81%, 4->5 +16%, 9->13 +11.5%, 6 vs 9 pad_multiple dead at 77%
    active / +8% at 95%; a10_smollm2-1.7b_wikitext exps 2->3 mb1->2 +23.6%, 3->4 mb2->4 +19.8%, 6->7 mb4->8 +1.7%,
    7->8 mb8->16 / accum 1 +7.0%), throughput basis same-rows. Kernel attribution same-reference-batch where batch_key
    matched (exp 5 vs 7, 7 vs 8); otherwise an observation.
  confidence: high (two models, one card; the mechanism is visible in the trace)
links: [[static-padding-is-the-first-lever]], [[chunked-ce-costs-a-head-grad-add-per-chunk]],
  [[recompute-pays-when-it-cuts-micro-steps]]
---

Top kernel after dynamic padding: `vectorized_elementwise_kernel<CUDAFunctor_add<BFloat16>>`. It took 0.534s/step at
accum 64, 0.271 at 32, 0.125 at 16 (compiled), and 0.060 at 8, so it halves exactly with the micro-step count:
~8 ms per 494M-param bf16 add, which is the A10's bandwidth. Tell: GPU-active well under 90%, top kernel
elementwise, and the step barely moves when tokens/step drops (exp 6: -25% tokens, +1.4% samples/s).

SmolLM2-1.7B (2026-10-08): the add is ~23 ms per 1.71B-param bf16 add (0.34s at accum 16, 0.16 at 8), the same
bandwidth rate. At accum 1 the add is gone and the step turns GPU-bound. The per-step optimizer (30% there) is then the
next fixed cost: see [[compile-the-optimizer-step]].
