---
claim: Under padding="longest", length-grouped batching is worth little at micro_batch 2 and is load-bearing at 8,
  because pad-to-longest of k random rows grows with k. Re-test it after every micro_batch increase, and never read
  the higher tokens/s or MFU of ungrouped batches as a win.
type: invariant
context:
  arch_family: any
  hardware_class: any
  dataset_shape: wide length spread (wikitext-2: p50 91 / p99 386; cosmopedia-100k: p50 649 / p99 1453)
evidence:
  probes: 3 (a10_qwen2.5-0.5b_wikitext exp 3 -> 4 at mb2: +3.3%; exp 13 vs 17 at mb8: off = -47% while tokens/s
    rose 9799 -> 12600 and MFU 33% -> 43%; a10_smollm2-1.7b_wikitext exp 4 -> 5 at mb4: +39.5%, tokens/step
    3890 -> 2345; a10_qwen2.5-1.5b_cosmopedia exp 9 -> 10 at mb4: +29.5%, tokens/step 14763 -> 10971), basis same-rows
  confidence: high
links: [[static-padding-is-the-first-lever]]
---
