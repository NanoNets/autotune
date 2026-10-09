---
claim: Checkpointed chunked cross-entropy accumulates the lm_head weight grad once PER CHUNK. With a large tied
  head that cost scales with chunks x micro-steps, so use the fewest chunks that fit. Under torch.compile it is
  usually beaten outright by compile's own fused CE, which also frees the logits memory.
type: invariant
context:
  arch_family: dense decoder with a large (tied) vocab head (Qwen2.5-0.5B, head 136M = 28% of params)
  hardware_class: A10
  dataset_shape: any; it matters most when micro-steps are many
evidence:
  probes: 3 (a10_qwen2.5-0.5b_wikitext exp 5 vs 7: chunks 8 -> 4 = +15.8%, add kernel 0.321 -> 0.215s,
    same-reference-batch; exp 7 -> 8: compile without chunking +75%, same-rows; ceiling: compile alone fits mb4
    where plain fits 2)
  confidence: medium
links: [[micro-step-count-bounds-small-real-batches]], [[liger-flce-dead-when-vocab-dwarfs-hidden]]
---

Before 2026-10-08, the hf runner's compile + chunked_ce ran fully eager: the chunked loss called the backbone
directly, bypassing the compiled wrapper. It now compiles the backbone in that case (branch
run/a10_qwen2.5-0.5b_wikitext). Even so, chunked_ce=8 + compile fit only mb4, the same as compile alone, because
the fp32 chunk logits OOM at mb8.
