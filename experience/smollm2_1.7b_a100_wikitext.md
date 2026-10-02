## smollm2_1.7b_a100_wikitext — 2026-09-21
Context: `HuggingFaceTB/SmolLM2-1.7B-Instruct` · `wikitext-2-raw-v1` · 1x NVIDIA A100-SXM4-40GB
(CC 8.0, 39.5GB), bf16, effective_batch 16, max_len 1024, ssh backend
Baseline: 9.38 samples/s → Best: **102.77 samples/s (+995%, 10.96x)**, MFU 43% → 60%
40 experiments, 39 ok / 1 timeout; 3 readings discarded as measurement artifacts (MFU > 100%).
Plot: `runs/smollm2_1.7b_a100_wikitext/progress.png` — note its header says "30.4x", which reads
exp28, one of the discarded artifacts; the honest figure is 10.96x at matched methodology.

**Winning config:** `micro_batch=8`, `gradient_checkpointing=False`, `torch_compile=True`,
`fused_optimizer=True`, `attn_implementation=sdpa`, `padding=longest`, `length_grouped=True`,
`pad_multiple=8`, `tf32=False`. Confirmed twice (102.731 / 102.77, 0.04% apart).
**Measured at `sample=1000`, `clean_steps=300`, scored on MEAN step time** — the methodology is part
of the config here, see the measurement findings below.

### Findings

- **Not computing padding was again the dominant single lever**, worth **6.26x** in clean isolation
  on the final config (exp38: 16.4 vs 102.8 samples/s, only `padding` moved). Second dataset shape,
  second model, second card — the belief holds.
- **The optimizer, not the model, was the bottleneck at grad_accum=1.** Read off
  `gpu_phase_times_s`, not guessed: AdamW was 0.0565s of a 0.1019s step (55%), and 5 of the top 8
  kernels were `multi_tensor_apply`. The optimizer's cost is **per-parameter**, so it does not shrink
  with a small batch — with only ~1.2k real tokens/step, updating 1.7B params *was* the step.
  `AdamW(fused=True)` cut it to 0.0203s. Worth +55% as first measured, +19% once measured honestly.
- **`torch.compile` was worth +42%** and again cut peak memory (20.8→18.6GB at mb=8; 16.6→14.5GB at
  mb=16), raising the ceiling 8 → 12. Third independent confirmation of the memory effect.
- **Recompute's value here was purely instrumental.** Standalone it costs 35%. It was worth taking
  only while it bought `micro_batch=16` → `grad_accum=1`, and it stopped being worth taking the
  moment the batches got wider. See the flip below.
- **Length grouping was worth +96%**, the second-largest lever, and it is also what makes the
  measurement hard (below).
- **`pad_multiple=8` is an interior optimum**, bracketed on both sides: 2/4/8/16/32/64 →
  249.5 / 252.1 / **264.4** / 241.0 / 231.7 / 162.6. This **closes** the prior run's open hypothesis
  that (8, 64) was unexplored: it is monotone in that interval, with the turn just below 8.
- **`micro_batch=8` is an interior optimum too** (mb4 76.9 < mb8 102.7 > mb16+recompute 83.8).
- **`max-autotune-no-cudagraphs` timed out at 601s** without reaching a timed step — re-confirms on a
  second model and card that its search can exceed the entire probe budget.
- **`sdpa` beat `eager` by only 5.4%.** At ~68-token batches attention is a small share of the work.
  `flash_attention_2` was **not testable** — `flash_attn` is not installed on this box.
- **Dead lever:** `checkpoint_use_reentrant`. Both settings landed inside the noise band of the
  default (262.6 / 264.6 vs 264.4), so the knob was reverted back out of the adapter rather than
  kept — recompute's cost here is the recomputed forward, not the checkpoint plumbing.

### Surprises / contradictions

- **The run's pivotal decision flipped at experiment 37, and both causes were measurement
  corrections rather than knob moves.** `micro_batch=16` + recompute (grad_accum=1) beat
  `micro_batch=8` by 49% under the original methodology; measured at a representative sample with a
  mean-scored rate, `micro_batch=8` **without** recompute wins by 22.6%. Wider real batches make the
  fixed per-step costs that grad_accum=1 was bought to eliminate a smaller share of the step, while
  recompute's extra forward is a per-token cost that does not shrink. The discipline the prior run
  wrote down — re-test winners after a regime change — paid for itself, and the regime change was
  the instrument.
- **The probe's row sample moved samples/s by 1.9x** (264.4 → 138.1 for the same config at
  `sample` 256 → 1000; tokens/step 1088 → 2057). The shipped 256-row head slice of wikitext is far
  shorter than the split. This is the first **measured** effect size for
  [[probe-sample-must-match-length-distribution]], which previously had none.
- **`tokens/s` is sample-robust while `samples/s` is sample-sensitive** (17994 → 17758, 1.3%, across
  the same 1.9x change). That is the exact mirror of the cross-padding-mode problem, where tokens/s
  inflates static padding and samples/s is the honest one. **Neither metric is safe alone**; the pair
  has to be read against what changed.
- **A median-based rate overstated throughput by up to 2.2x.** Three probes returned MFU > 100% and
  one reported GPU-active 560% — physically impossible, so the numbers did not apply. Cause:
  `samples/s` divided a mean token count by a **median** step time, and length-grouped dynamic
  padding makes step times violently right-skewed (`median/mean` reached **0.45**). Raising
  `clean_steps` 10 → 300 did not fix it; reporting the mean did.
  **The lever that wins is the same lever that biases its own measurement** — `median/mean` is 1.00
  with static padding and 1.03 with grouping off, versus 0.45–0.93 with it on.
- **Cudagraphs did not fail the way the prior belief predicted.** At `grad_accum=1` the documented
  precondition is gone and they did **not** crash — they ran and collapsed to 4.01 samples/s
  (GPU-active 3%), because under `padding=longest` every batch is a new shape and the graph is
  re-captured constantly. Different mechanism, same verdict. Belief narrowed and split.
- **Cross-entropy did *not* dominate activation memory here**, unlike the A10 run: at vocab 49152 /
  seq 1024 the logits term is ~9% of the ~3.2GB/sample, not 70%. Recompute correspondingly moved the
  ceiling much more (8 → ≥16). Belief narrowed by context rather than overwritten.

### Open hypotheses for next run

- **A stratified probe sample, not a bigger head slice.** `sample` 1000/2000/4000 never converged
  (tokens/step 2057 / 1719 / 1810, non-monotone), so no head slice of an ordered file is
  representative at any size. The fix is to draw to match the distribution (since built:
  `frame/dataset.py:draw_sample`). Until then every dynamic-padding samples/s in this repo is sample-conditional.
- **Throughput should be reported as total tokens / total time.** `step_time_mean_s` was added at the
  backend as a stopgap; the profiler's own `mfu`, `gpu_active_frac` and `roofline_floor_s` are still
  computed from the median and still print impossible values under skew.
- **Packing / `padding_free`** remains the unbuilt lever that would end the whole padding/width/skew
  problem at once: fill every position with real tokens and batch width stops being a random
  variable. Still not in `frameworks/hf.py`. Now doubly motivated — it fixes the measurement too.
- **`flash_attention_2` was never testable** (lib absent). Installing it would also finally close the
  prior run's open question about the attention verdict on a no-padding config.
- **8-bit optimizer states** were never tried. They attack the one term that is now provably
  load-bearing (the optimizer was 55% of the step, and its memory is what caps the ceiling at 12
  under compile). Would need `bitsandbytes` on the box.
- **The ceiling under compile is 12, which does not divide `effective_batch=16`** — so ~4 samples of
  measured headroom are unusable purely because of a divisibility constraint. Anything that lifts it
  to 16 without recompute unlocks grad_accum=1 at no per-token cost.
