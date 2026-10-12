## a10_qwen2.5-0.5b_wikitext — 2026-10-08
Context: `Qwen/Qwen2.5-0.5B` · `wikitext-2-raw-v1` · 1x NVIDIA A10 22.06GB (CC 8.6) · bf16 (checkpoint's, none set) · effective_batch 64 · max_len 2048
Baseline: 4.158 samples/s  →  Best: 88.120 samples/s  (+2019%, 21.2x)   # hf runner, ssh backend
Winning config (hf runner): `micro_batch=8` (grad_accum 8), `gradient_checkpointing=True`, `torch_compile=True`
(inductor default), `padding="longest"`, `pad_multiple=8`, `length_grouped=True`, `attn_implementation="sdpa"`,
`tf32=False`, `fused_optimizer=False`, `chunked_ce=0`. Extended probe (3 passes): 87.81 (-0.4%).

Model: Qwen2ForCausalLM 494.0M, 24 layers, hidden 896, 14q/2kv GQA (head_dim 64), MLP 4864, vocab 151936,
tied embeddings. The vocab/hidden ratio (~170) is the shape fact that decided most of this run: the
[tokens, vocab] logits cap the ceiling, and the lm_head (136M = 28% of params) dominates the per-micro-step
weight-grad traffic.

Ceilings @2048 (synthetic): plain 2 · chunked_ce=8 4 · chunked_ce=4 4 · chunked_ce=2 4 (20.6 GB, too close)
· compile 4 · compile+max-autotune 4 · compile+recompute 8 (mb16 OOM on the 9.27 GiB bf16 logits) ·
compile+recompute+eager 8 · Liger FLCE+compile 8 · compile(backbone)+chunked_ce=8 4.

Findings (all same-rows unless noted; run-to-run noise 0.2%, exps 9/10):
- **Dynamic padding first: 3.97x** (exp 2). max_len 2048 vs p99 386. tokens/s fell 8515 -> 2390 while samples/s rose.
- **After that the step is micro-step-bound, not token-bound.** The top kernel became the bf16 grad-accum add
  (`CUDAFunctor_add<BFloat16>`, 0.53s/step at accum 64), which scales with the micro-step count, not with tokens.
  mb 1 -> 2: +81% (exp 3); mb 2 -> 4 via chunked_ce: +16% (exp 5); mb 4 -> 8 via recompute: +11.5% (exp 13).
- **chunked_ce carries a hidden per-CHUNK cost:** each checkpointed slice accumulates its own lm_head
  (= tied embedding, 136M) weight grad. chunks 8 -> 4 at mb4: +15.8%, add kernel 0.321 -> 0.215s
  (same-reference-batch, exp 5 vs 7).
- **torch.compile: +75%** at mb4 (exp 7 -> 8: 41.7 -> 73.1), replacing chunked_ce. Inductor's fused CE alone lifts
  the ceiling 2 -> 4, removes the per-chunk adds, and cuts idle 19% -> 5% (same-reference-batch). Dynamic widths
  recompiled only in warmup (median ~ mean in every compiled probe).
- **pad_multiple 64 -> 8 is regime-dependent:** +1.4% (noise) at 77% GPU-active (exp 6), +8.0% at 95% (exp 9).
- **Recompute paid instrumentally (+11.5%, exp 13), against my wash-to-negative prediction:** GPU busy barely moved
  (0.633 -> 0.601s) because the halved grad-accum adds and fatter GEMMs paid for the re-run forward, and gap_count fell
  24.5k -> 15.2k.
- **length_grouped is load-bearing at mb8:** turning it off cost -47% (exp 17), while tokens/s ROSE to 12600 and MFU to
  43%. At mb2 it was only +3.3% (exp 4). Its value scales with micro_batch.
- Dead levers here: fused AdamW (-1.4%: optimizer phase 0.042 -> 0.054s on bf16 params, exp 11);
  compile `max-autotune-no-cudagraphs` (-0.7%, 351s wall, exp 12); eager attention under compile (-2.5%, exp 16);
  Liger FLCE (-72%, exp 14, below).

Surprises / contradictions:
- **Liger fused-linear-CE is catastrophic on small-hidden / huge-vocab models** (exp 14: 24.5 vs 88.1 samples/s,
  forward 0.13 -> 1.86s). liger 0.8.4 sets chunk_size = next_pow2(BT / ceil(V/H)); V/H = 170 cuts a ~900-token
  micro-batch into ~8-token chunks, each running a full [151936 x 896] grad-weight GEMM. Its memory claim held
  (mb8 without recompute). The knob was added to the runner for this probe and then reverted.
- **compile + chunked_ce was silently eager**, because the chunked loss calls `self.model.model`, which bypasses the
  compiled wrapper. Fixed on branch run/a10_qwen2.5-0.5b_wikitext (c8bcf46: compile the backbone in that case).
  No ceiling benefit here (mb8 OOMs on the fp32 chunk logits).
- Ceiling-stamp quirk: Frame stamps the resolved dtype ('bfloat16'), while a probe config with no dtype stamps null,
  so the first `frame.ceiling` check spent 2 trials on a false "stale".

Open hypotheses for next run:
- **mb16 is the next step** (accum 8 -> 4, about half the remaining 0.06s of adds plus launches), blocked only by the
  logits. A fused linear-CE with a sane chunk size (fixed ~1-4k tokens/chunk, grad-weight accumulated once per
  micro-step) would get there. Liger's heuristic doesn't, for this shape.
- flash_attention_2 is not installed on these images. Padding-free FA2 (flattened rows) would remove the last ~5% of pad
  and the masked mem-efficient attention, but attention is a small share at ~110-token rows. Low priority.
