## a10_qwen2.5-1.5b_cosmopedia — 2026-10-08
Context: `Qwen/Qwen2.5-1.5B` · `HuggingFaceTB/cosmopedia-100k` · 1x NVIDIA A10 22.06GB (CC 8.6) · bf16 (checkpoint's, none set) · effective_batch 16 · max_len 2048
Baseline: 1.493 samples/s  →  Best: 7.052 samples/s  (+372%, 4.72x)   # hf runner, ssh backend
(The shipped config OOMs on this card, exp 1. The baseline is exp 2: the shipped config plus
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, which changes the allocator only.)
Winning config (hf runner, branch run/a10_qwen2.5-1.5b_cosmopedia): `micro_batch=4` (grad_accum 4),
`gradient_checkpointing=True`, `recompute_layers=25`, `attn_implementation="flash_attention_2"`, `torch_compile=True`
(inductor default), `compile_optimizer=True`, `drop_pad_mask=True`, `padding="longest"`, `pad_multiple=1`,
`length_grouped=True`, `tf32=False`, `fused_optimizer=False`, `chunked_ce=0`,
`env={"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}`. Repeat 7.019 (exp 14); 3-pass extended probe 6.901 (exp 12).
`drop_pad_mask` is a runner knob this run added (commit 1baf2b5).

Model: Qwen2ForCausalLM 1543.7M, 28 layers, hidden 1536, 12q/2kv GQA (head_dim 128), gated MLP 8960, vocab 151936, tied
embeddings. V/H = 99, so the head is 233M (15% of params) and one 2048-token row's logits are 0.62 GB bf16 / 1.24 GB fp32.
Model state 11.5 GB. Data: cosmopedia-100k, mean 683 / p50 649 / p99 1453 tokens: about 6x longer rows than wikitext,
and max_len 2048 left ~67% pad.

Ceilings @2048 (synthetic, accumulation 2 since 8d5fa6d): plain 1 (20.60 GB, card_frac 0.98) · + expandable segments 1 ·
compile 1 (17.46) · compile+FA2 1 (17.18) · + recompute_layers 4: 1 · + chunked_ce=4 (no recompute): 1 · compile+FA2+full
recompute 4 (19.15; mb8 OOM on the 4.64 GiB bf16 logits) · recompute_layers 25: 4 (20.76, reserved 21.01).

Findings (all same-rows: one sample_hash, whole passes; noise floor 0.3-0.5% between launches, exps 10/13 and 11/14):
- **The memory ceiling was wrong, and the shipped config didn't fit** (exp 1 OOM where Frame said mb1 fits). Frame's
  trials ran at accumulation 1, where zero_grad(set_to_none) frees the grads before every forward. From micro-step 2 of
  an accumulating step they are live: +2.8 GB here (17.80 -> 20.60 GB at mb1). Fixed in frame.ceiling (8d5fa6d). A padded
  mask adds ~0.5 GB more that synthetic all-ones rows never show (see below).
- **Dynamic padding: 3.00x** (exp 3), smaller than wikitext's 3.2-4.0x because the pad share was smaller (~67%).
  tokens/s ROSE slightly (3057 -> 3199): the pad cost more than its token share, through masked attention.
- **Any pad at all costs flash attention.** With a zero in the mask, HF SDPA repeat_kv's K/V and SDPA runs the masked
  mem-efficient kernel. Its backward on ~700-token rows was 0.414 s/step against 0.055 s for flash (7.5x). pad_multiple
  64 -> 1 at mb1 (no pad left): +17.4% (exp 4).
- **Compiled HF SDPA never picks flash** (exps 5, 6: -1.9% each). transformers 5.15 disables its is_causal skip while
  tracing, so a causal mask is materialized even with attention_mask=None. HF `flash_attention_2` with no mask calls
  flash_attn_func(causal=True) and is compile-clean: compile + FA2 + drop_pad_mask, +7.8% (exp 7).
- **Compiling the optimizer step: +3.25%** (exp 8), optimizer GPU 0.133 -> 0.045 s.
- **Full recompute + mb4 lost -8.6% on random batches (pad +35%) and won +18.4% with length grouping** (exps 9 -> 10). On
  these long, compute-bound rows the re-run forward was paid for by the micro-step costs it removed: grad-accum add
  0.358 -> 0.070 s, idle 7% -> 0%, gap_count 72k -> 8k. Partial recompute 25/28: +1.8% (exp 11), +1.6% on the repeat pair.
- Not probed, by arithmetic: mb8 needs chunked CE for the logits (head forward re-run ~4.5% + per-chunk head-grad adds) to
  save the last ~0.035 s of adds; tf32 (no fp32 matmuls); CUDA graphs (idle 0%); dataloader workers (data-wait 0%).

Surprises / contradictions:
- Exp 1: the measured ceiling said mb1 fits, and it OOMed. Cause above; fixed.
- My first compile diagnosis (exp 5: "compile materializes the all-ones padding mask") was incomplete. Dropping the mask
  (exp 6) changed nothing, because HF builds a causal mask itself while tracing.
- pad_multiple 1 under compile did NOT recompile in the window here (mean/median <= 1.035 in exps 5-8, 10, 11), against
  SmolLM2's exp 13 where it did.
- The 3-pass extended probe read 2.1% below the 1-pass probe of the same config. The 1-pass repeat was within 0.5%, so
  this is probably sustained-load clocks on a 150 W card (not measured per pass).

Open hypotheses for next run:
- Logits are still the memory bound at mb4. A fused linear-CE with a sane fixed chunk (1-4k tokens, head grad accumulated
  once per micro-step) would free ~2-4 GB for 4-7 more un-checkpointed layers (~0.6% each) or for mb8.
- drop_pad_mask + FA2 should also fix any padded mb>1 config. On short-row data (wikitext) it is untested and should beat
  SmolLM2's padded-FA2 loss (belief hf-fa2-padded-batches-break-compile).
- Synthetic ceiling rows have no pad, so they understate a config that passes a padded mask to SDPA (~0.5 GB here). A
  padded-mask trial (or drop_pad_mask) would close it. Not built.
