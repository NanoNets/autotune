## qwen3.5_2b_a100_wikitext — 2026-09-21
Context: `Qwen/Qwen3.5-2B` · `wikitext-2-raw-v1` · 1x NVIDIA A100-SXM4-40GB (CC 8.0, 39.5GB), bf16,
effective_batch 16, max_len 1024, ssh backend
Baseline: 2.154 samples/s → Best: **36.006 samples/s (+1571%, 16.72x)**, MFU 11% → 33%
25 experiments, 19 ok / 6 failed (2 compile timeouts, 3 OOM, 1 measurement rejected).
Plot: `runs/qwen3.5_2b_a100_wikitext/progress.png` — **its header says 17.4x, which reads exp 24
(`chunked_ce=4`), a config discarded as inside noise and never safety-proven.** The honest figure is
16.72x. Second run running where the plot's headline disagrees with the shipped config; see below.

**Winning config:** `micro_batch=16` (**grad_accum=1**), `gradient_checkpointing=True`,
`chunked_ce=8`, `fused_optimizer=True`, `padding=longest`, `length_grouped=True`, `pad_multiple=8`,
`attn_implementation=sdpa`, `tf32=False`, `torch_compile=False`.
**Environment is part of the config here:** `flash-linear-attention` 0.5.2 + `causal-conv1d` 1.7.0
installed, and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
Confirmed twice (36.006 / 35.894, 0.31% apart). Worst case **proven** at 16.805GB of 39.494.

### The architecture, because it drives everything below

Not a dense transformer. 24 layers, `full_attention_interval=4` → **18 linear-attention (gated
delta rule, Mamba-family) + 6 full-attention (GQA, `attn_output_gate`)**, vocab **248,320**, and a
vision tower + MTP head in the checkpoint that `AutoModelForCausalLM` never instantiates (text
stack 1.882B of 2.274B). Config nests dims under `text_config`; `frame/model.py` crashed on it and
needed `_flatten()` (committed).

### Findings

- **The run was a dispatch ladder, and nothing else mattered as much.** Kernel gaps per step track
  `grad_accum` almost exactly and wall-clock tracks gaps: **43113 / 21353 / 12883 / 6487** for
  grad_accum 8 / 4 / 2 / 1. Every large win came from cutting launches; every lever that cut GPU
  *work* bought nothing. `cpu_cores_busy` sat at **1.00 of 30 vCPUs** in every probe.
- **Three separate levers reduced GPU work and returned zero throughput**: `tf32` (−8.8% GPU time),
  `causal_conv1d` (−5.5%), coarse bucketing (+2.5x work per launch, −10.7% throughput). On a
  launch-bound step, GPU-time share is the wrong denominator for judging a lever.
- **`flash-linear-attention` was worth +20.7%** and is the single biggest kernel lever: gaps halved
  84585 → 43113, all `ampere_sgemm_*` fp32 GEMMs disappeared, peak fell 2.4GB. Without it,
  transformers silently runs a *reference PyTorch* gated-delta-rule in 18 of 24 layers (its own
  source comment: ">1 order of magnitude" slower on an H100).
- **`causal-conv1d` engaged and was worth ~0** (−0.5%, inside noise; gaps −3.0%). Kept because it
  is free in code and trims memory, but it costs a 196MB compiled wheel and an nvcc toolchain for
  no measured throughput at this sequence shape.
- **`torch.compile` is unusable on this architecture.** Two 601s timeouts, never reaching a timed
  step — first with ~128 distinct batch widths, then with **4**. Shape count is exonerated; the
  cause is the delta rule's data-dependent chunking plus fla's opaque Triton kernels. Cost: 20 min.
- **Chunked cross-entropy, built this run** (`frameworks/hf.py`, `chunked_ce` knob), **is what won
  it.** At vocab 248,320 the `[mb,1024,248320]` logits and their fp32 CE copy were **9.35GB at
  mb=4** (36.449 → 27.095GB). Slicing the token axis with a checkpoint per slice made memory nearly
  batch-independent — peak is *flat at 17.542GB for mb=1/2/4/8* — which lifted the ceiling to
  **≥16 (capped by `effective_batch`, not memory)** and delivered grad_accum=1. Loss verified
  identical to the unchunked path to 7.8e-08 relative at every chunk count.
- **`fused_optimizer` is worth +8.1%** here, *more* than its 4.9% GPU-time share predicts, because
  it removes *launches* as well as work.
- **`length_grouped` is worth +20.3%**, and its damage when removed is entirely compute:
  tokens/step +150% with **gap_count unchanged** (12883 → 12932).
- **`attn_implementation` is a near-dead lever on a hybrid model** (−1.7%, inside noise): it
  reaches 6 of 24 layers. A quarter of the prior run's dense 5.4% is ~1.4%, almost exactly measured.

### Surprises / contradictions

- **THE SYNTHETIC CEILING IS NOT A SAFE BOUND NEAR THE CARD LIMIT.** `frame/README.md` states that
  a config fitting the synthetic worst case "fits every batch the dataset can produce". **Two
  measured counter-examples at matched shapes**: ceiling said mb=8+recompute OK at 34.174GB → real
  run OOM'd; ceiling said mb=4 OK at 36.449GB → real run OOM'd at 35.697GB. A verified-safe config
  measured real peak **25.862GB against a 22.353GB ceiling (+3.51GB, +15.7%)**. The ceiling reuses
  one device-resident batch and omits per-micro-step host→device staging and allocator churn, and
  the gap appears to grow with `micro_batch`. Every failure sat at `card_frac ≥ 0.878`; everything
  verified sits at `≤ 0.60`. **This invalidated two champions mid-run** (13.27x and 11.98x both
  OOM on a full-width batch) and is why the shipped config is worst-case-proven, not ceiling-argued.
- **Fragmentation, not capacity, held the first ceiling.** mb=4 OOM'd with peak 30.8GB but
  reserved 35.3GB — 3.1GB of fragmentation. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
  collapsed it to 0.203GB and lifted the ceiling 3 → 4. Needed generic `config["env"]` forwarding
  in `backends/ssh.py` (committed), since the allocator reads it before any runner code runs.
- **Recompute is a DISPATCH cost, not only a compute cost.** At mb=8 it won (+10.8%, crossing a
  factorization boundary); at mb=4 it **lost 40%** — gaps 20610 → 25974 — because its recomputed
  forward re-issues every kernel with no micro-step saving to offset it. I predicted 26-28
  samples/s and measured 15.4. On a launch-bound step recompute is strictly negative unless the
  boundary it crosses cuts micro-steps by more than its own added launches.
- **`tf32`'s precondition failed, then came back.** Pre-fla the trace was full of fp32 GEMMs —
  `mamba_ssm_dtype: float32` runs the reference recurrence in fp32 — so the belief's own guard
  ("don't spend a probe unless the trace shows fp32 GEMMs") correctly *fired*. tf32 engaged and cut
  GPU time 8.8% for zero throughput. Post-fla the fp32 GEMMs are gone and the belief is restored.
- **The divisibility trap cost measured headroom three times**: ceilings of 3 and of 6 both left
  usable sets of `{1,2}` and `{1,2,4}` because `micro_batch` must divide `effective_batch=16`.
  Third run in a row this has appeared. It is structural, not incidental.
- **`gpu_active_frac` printed 126%** on the champion — the median/mean bias again (median/mean
  0.781). Third sighting across two runs; `mfu`, `gpu_active_frac` and `roofline_floor_s` are still
  median-based. `samples/s` is safe (mean-based).
- **Padding's isolated value shrinks as the dispatch wall lifts**: 2.01x at mb=4+recompute, 3.58x
  on the final config, vs the prior run's 6.26x on a dense model. Under static padding this config
  hits MFU 49% / GPU-active 100% — it is *efficient* per token and merely computes 6.8x too many.

### Open hypotheses for next run

- **`packing` / `padding_free`** — now the clear top lever. It would end batch-width variance,
  which is what caps this run's precision (cv 1.34 at grad_accum=1, sem ±6.7%, untightenable by
  more steps), and would fill the 6.8x of wasted positions that static padding shows the GPU can
  process efficiently. Still unbuilt.
- **`effective_batch` is now the binding constraint, not memory.** The ceiling search terminated
  `capped: true` at 16 — the card has room for more. Raising `effective_batch` is the human's call,
  but it is the only remaining way to grow `micro_batch`, and grad_accum is already 1.
- **The champion's heavy tail is unexplained.** cv 1.34; I disconfirmed per-shape Triton autotuning
  (pad_multiple 8 → 4 left cv unchanged at 0.60). Worth a trace of the slowest steps.
- **`chunked_ce` chunk count is unresolved** — 4 measured +4.1% over 8 but inside a ±7.2% band.
  Needs either a lower-variance workload (packing) or a much longer probe to settle.
- **`probe-sample-must-match-length-distribution` was NOT addressed** and the confirmation run does
  not test it: `frameworks/hf.py` builds the loader deterministically (first 1000 rows in file
  order, `random.Random(0)` chunk shuffle, `shuffle=False`), so exps 22 and 25 saw identical rows.
  Every dynamic-padding number here remains conditional on a 1000-row head slice.
- **`progress.py` headlines the max `samples/s` row, not the shipped config** — it reported 17.4x
  from a discarded, never-safety-proven probe. Second run in a row. It should read the kept config.
- **A `frame`-level fix for the ceiling gap**: either measure the ceiling through the real
  dataloader path, or have `frame` report a margin-adjusted bound. Two champions died on this.

### Addendum — 2026-09-21 (experiments 26-28 + a packing investigation)

Three of the open hypotheses above are now **closed**, and the run's binding constraint has moved.

- **`flash_attn` is NOT hard to install, and three runs recorded it as untestable.** The prebuilt
  wheel `flash_attn-2.8.0.post2+cu12torch2.7cxx11abiTRUE-cp310` installs in **5 seconds**. Read
  `torch.version.cuda` and `torch._C._GLIBCXX_USE_CXX11_ABI`, fetch the matching wheel, don't build
  from source. With it: **FA2 is -1.0% vs sdpa** on the final config (inside noise), and `gap_count`
  rose 5.6% — it *adds* launches for an unpad/repad that has nothing left to recover. Together with
  eager at -1.7%, all three attention kernels sit within ~2% here. **CLOSED.**
- **The champion's heavy tail is explained, and is not a defect.** Twice disconfirmed the
  Triton-autotune-variant theory (`pad_multiple` 8→4 at ga=2: cv unchanged at 0.60; 8→32 at ga=1:
  cv 1.340→1.287). The real cause is structural: at grad_accum=1 a step *is* one batch, so step
  time tracks batch width, and `length_grouped` **sorts by length**, spanning ~64 to ~8300 tokens.
  Corroborated by exp 26 — grouping OFF gives cv 0.550, because unsorted batches all pad to the max
  of 16 random draws and are therefore more *uniform*. Grouping buys throughput by making widths
  unequal; the variance is the mechanism working. **CLOSED.**
- **Packing would have silently changed the model** — see
  [[packing-leaks-across-documents-on-hybrid-models]]. Logit-equivalence tests against standalone
  forwards, with a determinism control (0.0000), show a packed document reading its neighbour by
  12.94 / 15.17 / 16.97 max|Δ| under naive packing, position_id resets, and explicit `cu_seq_lens`
  respectively — against 0.2891 of shape-dependent numerical variation. **Position resets made it
  worse.** The linear-attention half is the hard part: a gated-delta *recurrence* needs state reset
  at boundaries, which no attention mask expresses. Both prior runs listed this as the top unbuilt
  lever; it is not the free win it looked like. **Re-scoped, not closed.**

**The binding constraint is now `effective_batch`, and it is the human's.** Peak memory on the
champion is 15.735GB of 39.494 (worst case 16.805GB) — memory is no longer binding at all. The
ceiling search terminates `capped: true`. `grad_accum` is 1, its floor. Every safe lever in
`loop/throughput.md` has been probed on this stack: batch knobs are at the cap, kernels are
installed and measured, compile is dead, tf32 is inert, recompute and chunked CE are kept,
packing is blocked on correctness, and 8-bit optimizer states would buy nothing because memory is
abundant. **Raising `effective_batch` is the only remaining direction, and it is a quality decision,
not a knob** — at `effective_batch=32` the card has room for `micro_batch=32` at roughly 21GB by
the flat-peak evidence, which would keep grad_accum=1 while doubling the work per launch.
