---
claim: Liger's fused linear cross-entropy (0.8.4) is a large throughput LOSS when vocab/hidden is large, even
  though it does remove the logits memory. Its chunk size is next_pow2(BT / ceil(V/H)), so V/H ~170 cuts a micro-batch
  into ~8-token chunks, each running a full [V x H] grad-weight GEMM. There is no chunk-size override.
type: dead-lever
context:
  arch_family: small-hidden / large-vocab decoders (Qwen2.5-0.5B: H 896, V 151936)
  hardware_class: any (the cause is the chunking heuristic)
  dataset_shape: any
evidence:
  probes: 1 (a10_qwen2.5-0.5b_wikitext exp 14: 24.5 vs 88.1 samples/s, -72%, same-rows; forward 0.13 -> 1.86s;
    cause read from liger_kernel/ops/fused_linear_cross_entropy.py)
  confidence: high for this shape. Untested where V/H is small (~8 for Llama-8B), where the heuristic was designed
    to work.
links: [[chunked-ce-costs-a-head-grad-add-per-chunk]]
---

Numerically it was fine: identical loss, and gradient deviation in the same class as chunked_ce (bf16
reassociation). The memory claim held too: mb8 fit with no recompute (19.47 GB). Speed is the only problem.
