---
claim: When max_len sits far above the data's real lengths, padding="longest" at the loader is the first and
  largest lever. Removing the padding moves samples/s by a multiple while tokens/s and MFU FALL. Check p50/p99
  against max_len before any kernel work.
type: regime
context:
  arch_family: dense decoder (Qwen2, Llama)
  hardware_class: A10 (Ampere, 22GB)
  dataset_shape: max_len well above real lengths (wikitext-2 p50 91 / p99 386; cosmopedia-100k p50 649 / p99 1453, max_len 2048)
evidence:
  probes: 3 pairs (a10_qwen2.5-0.5b_wikitext exps 1 -> 2: 3.97x at max_len 2048; a10_smollm2-1.7b_wikitext exps
    1 -> 2: 3.21x at max_len 1024 = 2.6x p99; a10_qwen2.5-1.5b_cosmopedia exps 2 -> 3: 3.00x at ~67% pad), basis same-rows
  confidence: high for the direction; the size follows the pad fraction (~95% -> 3.97x, ~90% -> 3.21x, ~67% -> 3.00x)
links: [[micro-step-count-bounds-small-real-batches]], [[length-grouping-value-scales-with-micro-batch]]
---

4.158 -> 16.513 samples/s (3.97x) from `padding=longest` alone. tokens/s fell 8515 -> 2390 and MFU fell
33% -> 8%, because both count pad as work. Afterwards the step left the GPU-bound regime (active 98% -> 52%), so
the next lever is not a kernel one ([[micro-step-count-bounds-small-real-batches]]).

Qwen2.5-1.5B on cosmopedia (2026-10-08): tokens/s ROSE 3057 -> 3199 instead of falling, because the padded mask also
forced the slow masked attention kernel, so pad cost more than its token share. And rounding to pad_multiple 64 still
left a mask: removing the last pad was another +17.4% ([[padded-mask-costs-flash-attention]]).
