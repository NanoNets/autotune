---
claim: The synthetic ceiling (every row max_len, all-ones mask) is not the worst case for a config that passes a padded
  mask to HF SDPA. The real padded batch takes the repeat_kv + masked mem-efficient path, which synthetic rows never
  exercise: ~+0.5 GB at [1, 2048] on Qwen2.5-1.5B. Treat a padded-mask config within ~1 GB of the card as unproven.
type: calibration
context:
  arch_family: HF GQA decoders, attn_implementation sdpa, padded batches (padding=max_length, or longest with pad)
  hardware_class: any
  dataset_shape: any with padding
evidence:
  probes: 1 (a10_qwen2.5-1.5b_cosmopedia exp 1 OOM; on-box diagnostic ~/autotune_tools/memdiag.py at mb1 accum 2, step-1
    peak: synthetic 14.84 GB, real padded 15.33, real ids with an all-ones mask 14.84; step 2: synthetic 20.60, real OOM)
  confidence: medium
links: [[padded-mask-costs-flash-attention]]
---

The larger part of exp 1's OOM (+2.8 GB) was a different instrument bug: ceiling trials ran at accumulation 1, so the
grads were never live through a forward. That one is fixed in code (frame.ceiling, 8d5fa6d), not a belief.
