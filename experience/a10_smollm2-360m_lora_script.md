## a10_smollm2-360m_lora_script — 2026-10-09
Context: `HuggingFaceTB/SmolLM2-360M` LoRA r16 (7 projections, plain-PyTorch user script `examples/byo_script/torch_lora.py`) · `wikitext-2-raw-v1` · 1x NVIDIA A10 22.06GB (CC 8.6) · bf16 (script sets it; adapters fp32 via peft) · effective_batch 16 · max_len 1024
Baseline: 27.74 samples/s  →  Best: 116.43 samples/s  (+320%, 4.20x)   # script mode, ssh backend; extended 300-step probe 116.53
(The baseline, exp 1, is the script's own micro-batch 8, which is ABOVE the measured ceiling of 4 at 1024 eager: a
denominator, never a recommendation. Exps 1-11 ran on one A10 box, exps 12-23 on a second;
the incumbent re-anchored within 1% on the second box, exp 13.)
Winning config (flags on the verified copy `runs/a10_smollm2-360m_lora_script/train.py` + `_data.py`):
`--micro-batch 8 --grad-accum 2 --seq-len 1024 --compile --compile-dynamic --padding-free --attn flash_attention_2
--compile-opt --cpp-wrapper --gc-freeze --tf32` (+ `--data-path <rows.jsonl>`). `--cpp-wrapper` needs pybind11 headers:
this image's system torch has none (`pip install pybind11==2.13.6`, `CPLUS_INCLUDE_PATH=<site-packages>/pybind11/include`).
Model: LlamaForCausalLM 361.8M, 32 layers, hidden 960, 15q/5kv GQA (head_dim 64), MLP 2560, vocab 49152, ctx 8192.
LoRA trains 8.68M (2.3%), model state 0.8 GB. wikitext-2 rows: p50 92, p99 396, max 959 tokens.

**Script-mode run.** Every timing is `fixed-position`, not same-rows: the numbers below are observations, kept by
interleaved re-probes (6/9 vs 8/10, 13/15 vs 14/16, 18/20 vs 19/21) and a noise floor of 0.3-0.9% between launches.
Only kernel identities, recompile/GC counts and memory-trial peaks went to beliefs.

Ceilings @1024 (ceiling rows, accumulation 1): eager sdpa 4 (mb8 OOM on the fp32 logits) · compile 8 (19.08 GB) ·
compile dynamic + pad 8: 8 (19.87) · + recompute: 16+ (7.80; only after the eval-mode fix below) · compile + FA2
padding-free: 8 (18.09; trials expect `[1, mb*1024]`, backends `ceiling_expect_flat`) · + recompute 16+ (6.30).
Identical on both boxes.

Findings:
- **Compile is the first lever, but only once it stops recompiling** (exp 2 timed out at 600s): random per-batch widths
  recompiled ~95s each, first on width, then on HF SDPA's alignment guards (`width % 8`). `dynamic=True` from the start
  plus widths padded to a multiple of 8: 0 recompiles, **+74%** (exp 3). Eager it was launch-bound on peft's fp32 adapter
  path (cast, dropout, scale, add per module x 224, top kernels fp32 elementwise); compile fused those.
- **Padding-free batching: +124%** (exp 6, 48.3 -> 108.1). Per-batch dynamic padding still left ~61% pad at mb8 (16 rows x
  107 mean tokens = 1710 real of 4388). Flatten each micro-batch into one row: position_ids restart per row, labels -100 at
  each row's first token, FA2 varlen bounds built host-side with max_length = the constant 1024, no mask. Compile-clean,
  samples/step 16 counted from position_ids, loss inside run-to-run noise (2.95e-3 vs 2.94e-3). Not packing: same rows,
  same steps.
- After padding-free the step is mixed: short batches launch-bound, long ones GPU-bound (traced step 1109 tok: wall 0.202
  vs GPU 0.091s). Small host levers then add up: compiled optimizer step +0.8-0.9%, Inductor C++ wrapper +1.8-2.0%
  (traced step wall 0.21 -> 0.154s at unchanged GPU), TF32 for the fp32 adapter GEMMs +3.4-4.6% (sgemm -> TF32
  tensor-core kernels; loss inside run-to-run noise).
- **Recompute to reach one micro-step lost twice:** -35% padded (exp 4: +21% pad at mb16 plus the re-run forward), -16.6%
  padding-free (exp 7: wins on short launch-bound steps, loses on long GPU-bound ones; the mean decides).
- Dead: dataloader workers (data-wait 1.6% -> 0.1%, step unchanged), compiling the inner HF model to skip peft's two graph
  breaks (+0.1-0.4%, noise), CUDA graphs (crash, below).

Surprises / contradictions:
- **The user's script trains with its base model in eval mode.** `from_pretrained` returns eval and the script never calls
  `.train()`; peft's new dropout modules are in train mode, the decoder layers are not. Harmless for SmolLM2
  (attention_dropout 0), but transformers gates checkpointing on `self.training`, so `gradient_checkpointing_enable()`
  was a silent no-op (ceiling 19.92 vs 19.87 GB). The copy's `--grad-ckpt` calls `model.train()`.
- **I misread exps 3-5 for a while:** I compared the traced step's GPU time (2624 tok) with the MEAN step (4388 tok), two
  populations, and called a GPU-bound step 40% idle. Same-step, exp 3 was 93% GPU-busy. Read `profiled_step_s` against
  `gpu_phase_times_s`, never `step_time_mean_s`.
- **A one-step ~1.3s stall in 3 of 6 probes on the second box** (12, 15, 17; none of 4 on the first): generation-2 Python
  GC over ~2-3M compile-time objects (0.87-1.43s per pass, gc.callbacks). `gc.freeze()` after step 1 moves them to the
  permanent generation: no slow GC in 199 steps after it. Without it steady state is clean too until one 0.83s pass at
  step 199; the probe stalls were the tail of the compile-time passes landing in the clean window.
- CUDA graphs (`mode="reduce-overhead"`) crash on this stack: "accessing tensor output of CUDAGraphs that has been
  overwritten", at a resume function after a graph break; `cudagraph_mark_step_begin()` per micro-step did not help.
- The C++ wrapper reports an allocation failure as a generic `aoti_torch_empty_strided` RuntimeError, so `frame.ceiling`
  can't read its OOM (exp 11). It changes launch code only: real-row peak 4.383 vs 4.375 GB.

Open hypotheses for next run:
- Recompute at mb16 was measured before the C++ wrapper and TF32 changed the host/GPU balance; the arithmetic says it
  still loses (less host time to save), unprobed.
- Fused LoRA kernels (q/k/v adapters sharing their input, one GEMM) would cut the launch count that bounds short steps;
  needs a peft change, not a script knob.
- How often a full GC recurs in a long training run without `gc.freeze` (one 0.83s pass seen by step 199): measure
  over a few thousand steps.
- A noise-level progress.py "record" (exp 22, 116.57) sits above the kept champion: the staircase steps on noise.
