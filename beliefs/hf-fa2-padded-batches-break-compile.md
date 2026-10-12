---
claim: Under torch.compile, HF attn_implementation="flash_attention_2" on padded batches (an attention_mask with pad)
  is a large LOSS. HF unpads q/k/v with data-dependent ops at every layer, which splits the compiled graph, and that
  outweighs the pad attention skips. Padding-free FA2 with all four varlen kwargs precomputed host-side and no mask is
  compile-clean, but it only buys the residual pad.
type: dead-lever
context:
  arch_family: dense decoder, HF transformers 5.15 (SmolLM2-1.7B / Llama)
  hardware_class: A10 (flash-attn 2.8.3), torch 2.7.0
  dataset_shape: short rows under padding=longest + length grouping (~8% pad left)
evidence:
  probes: 1 (a10_smollm2-1.7b_wikitext exp 15: -27%, same-rows; GPU step 0.365 -> 0.451s same-reference-batch:
    forward 0.082 -> 0.104, backward 0.233 -> 0.297). Padding-free microbenchmark (not a probe, observation only):
    +1.5% fwd+bwd vs padded SDPA, losses match to ~1e-3.
  confidence: medium for the padded-mask loss; the padding-free number is an observation
links: [[length-grouping-value-scales-with-micro-batch]], [[compiled-hf-sdpa-never-picks-flash]],
  [[padded-mask-costs-flash-attention]]
---

The padding-free branch in `transformers/modeling_flash_attention_utils.py` needs cu_seq_lens_q/k AND max_length_q/k
(int, bucketed so dynamo doesn't specialize per value) and attention_mask=None. Position ids must restart per row for
RoPE, and labels need -100 at each row's first token so there is no cross-row prediction.

2026-10-08, Qwen2.5-1.5B on cosmopedia: the simplest padding-free form is no mask at all. For right-padded rows of a causal
LM the mask is redundant, and HF FA2 with attention_mask=None calls flash_attn_func(causal=True), which compiles clean.
With the hf runner's `drop_pad_mask` it was +7.8% over eager SDPA-flash at mb1 (exp 4 -> 7, which also turned compile on).
It also carried to grouped mb4 batches (exps 10-11). See [[compiled-hf-sdpa-never-picks-flash]].

2026-10-09, SmolLM2-360M LoRA user script (a10_smollm2-360m_lora_script exp 6): padding-free FA2 compiled clean again
(0 recompiles in 70 steps, dynamic=True; width-invariant) with a simpler bound: max_length_q/k = the constant max_len
(1024) instead of bucketing, so dynamo has nothing to specialize on; cu_seq_lens int32 built in the collate. "Only buys
the residual pad" is about how much pad is left: under per-batch padding with no length grouping, mb8 of wikitext rows
was ~61% pad and padding-free was +124% (script probe, an observation, not same-rows evidence).
