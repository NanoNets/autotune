## a10_smollm2-1.7b_wikitext — 2026-10-08
Context: `HuggingFaceTB/SmolLM2-1.7B` · `wikitext-2-raw-v1` · 1x NVIDIA A10 22.06GB (CC 8.6) · bf16 (checkpoint's, none set) · effective_batch 16 · max_len 1024
Baseline: 3.677 samples/s  →  Best: 35.991 samples/s  (+879%, 9.79x)   # hf runner, ssh backend
Winning config (hf runner, branch run/a10_smollm2-1.7b_wikitext): `micro_batch=16` (grad_accum 1),
`gradient_checkpointing=True`, `recompute_layers=21`, `torch_compile=True` (inductor default), `compile_optimizer=True`,
`padding="longest"`, `pad_multiple=8`, `length_grouped=True`, `attn_implementation="sdpa"`, `tf32=False`,
`fused_optimizer=False`, `chunked_ce=0`. Extended probe (3 passes): 35.84 (-0.4%). Noise floor 0.03% (exps 10/11).
`recompute_layers` and `compile_optimizer` are knobs this run added to the runner.

Model: LlamaForCausalLM 1711.4M, 24 layers, hidden 2048, 32/32 heads (MHA, head_dim 64), gated MLP 8192, vocab 49152,
tied embeddings. V/H = 24 (Qwen2.5-0.5B: 170), so the head is 5.9% of params: logits bound memory but not compute.
Model state 12.75 GB of 22.06, so at small micro-batches the peak is the optimizer step, not activations
(ceiling mb1 15.95 GB vs mb2 15.99).

Ceilings @1024 (synthetic): plain 2 (mb4 OOM on the 768 MiB fp32 logits) · compile 4 · compile+recompute 16 (capped;
reserved 21.0) · + compile_optimizer 16 (reserved 17.6) · recompute_layers 12 → 8, 18 → 8 (mb16 OOM on the 1.5 GiB bf16
logits), 21 → 16 (19.21 GB, card_frac 0.89) · max-autotune 16 · FA2 16 · no recompute + compile_optimizer 4.

Findings (all same-rows: one sample_hash, whole passes):
- **Dynamic padding first: 3.21x** (exp 2). max_len 1024 = 2.6x p99 396; mean row 107. tokens/s fell 3765 -> 1722.
- **Micro-step count then bounds the step**, as on Qwen: mb 1 -> 2 +23.6% (exp 3, bf16 grad-accum add 0.34 -> 0.16s),
  compile + mb4 +19.8% (exp 4), recompute + mb8 +1.7% (exp 7), mb16 / accum 1 +7.0% (exp 8, the add kernel gone).
- **Length grouping at mb4: +39.5%** (exp 5), tokens/step 3890 -> 2345. pad_multiple 64 -> 8: +8.7% (exp 6).
- **Compiling the optimizer step: +22%** (exp 10). torch.compile(opt.step) fuses foreach AdamW: optimizer phase
  0.146 -> 0.050s (same reference batch), about the A10's bandwidth floor for one pass over params, grads, m and v (~24 GB).
  It also cut the mb16 ceiling trial's reserved memory 21.0 -> 17.6 GB. Equivalence: <= 1 bf16 ulp on 0.008% of elements.
  Once accum is 1 and the step is GPU-bound, the optimizer was 30% of it.
- **Partial recompute: +2.2%** (exp 12). It checkpoints 21 of 24 layers, using the headroom full recompute left at mb16.
  An un-checkpointed layer costs ~1.2 GB at [16, 1024] here.
- Dead levers here: fused AdamW -3.9% (exp 9, optimizer 0.146 -> 0.178s); max-autotune-no-cudagraphs -0.5% with a 509s
  wall (exp 16); flash_attention_2 on padded batches -27% (exp 15); pad_multiple 1 -82% (exp 13, recompiles).
- Padding-free FA2 (microbenchmark, not a probe, research.md): +1.5% fwd+bwd once grouping leaves ~8% pad. Not built.

Surprises / contradictions:
- **HF flash_attention_2 under compile is a large loss with padded batches** (exp 15: GPU step 0.365 -> 0.451s, less of
  the model compiled). HF unpads q/k/v with data-dependent ops at every layer. The flattened path with all four precomputed
  varlen kwargs and no mask is compile-clean (microbenchmark), but it only buys the pad.
- **pad_multiple 1 recompiles inside the clean window** (exp 13): median step unchanged, mean 6x, wall doubled.
  pad_multiple 8 does not. A 3-pass extended probe (exp 14) confirmed the incumbent's mean/median of 1.14 is length
  variation (one length group per step at accum 1), not recompiles.
- The ceiling-stamp quirk (Frame stamps the resolved dtype, a probe config stamps null) cost 2 trials again.

Open hypotheses for next run:
- The remaining memory lever is the 1.5 GiB bf16 logits at mb16. Chunked CE or a fused linear-CE with a sane chunk size
  would free 1-3 more layers from recompute (~0.7% each), minus their own overhead. Liger's heuristic gives 128-token
  chunks at V/H 24 (vs 8 on Qwen). That's untested and would refine the Liger belief's context.
- 8-bit optimizer states would free ~3 GB and cut optimizer traffic. They are lossy, so that is the human's call, not tested.
- compile_optimizer should matter on any full fine-tune where accum is small and the step is GPU-bound. Re-test it on the
  Qwen run's champion (accum 8 there).
