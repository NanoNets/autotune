# Design improvements

> **Developer notes. Not part of the agent's prompt hierarchy.**
> `program.md` does not route here and the loop never reads or writes this file. It's a scratchpad
> for us to decide what to build next — the doctrine the agent follows lives in `program.md`,
> `loop/`, and the per-directory READMEs.
>
> Paths into `sandbox/attach_parity/` below refer to git tag **`sandbox-archive`**: the harness was
> removed from the tree on 2026-10-02 (its capture now lives in `profiler/attach.py` / `launch.py`).
> "BYO experiment N", "axiom audit B*N*" and "metric-coverage audit §N" cite the bring-your-own-script
> experiments, their Measurement-axiom audit and their metric-coverage audit, also kept in that tag.

Open questions and proposed changes that are **not yet settled by evidence**. Each entry states the
claim, why it might be wrong, and **what experiment would settle it** — so a future eval can close
it rather than re-litigate it.

Nothing here is a decision. Once an experiment settles an entry, make the change in the code and
the owning md, then delete the entry from here.

---

## 1. Replace the bisection with divisors + a memory curve

**Status: open. Proposed, not implemented — `search()` still bisects.**

`find_ceiling` brackets by doubling and bisects to the exact ceiling (`1,2,4,8,16,12,10,11` = 8
trials). Two problems:

- **The exact ceiling isn't actionable.** `micro_batch` must divide `effective_batch`, so a ceiling
  of 11 against `effective_batch=16` means you run **8**. Three trials were spent resolving 8→11 to
  learn a number no config can use.
- **The trials are already a memory curve and we throw it away.** Each trial records
  `(micro_batch, peak_gb)`, and peak is linear in batch at fixed `seq_len`:
  `peak_gb(mb) = model_state + activation_per_sample * mb`. Two points give the slope.

**Proposal:** test the divisors of `effective_batch` ascending, stop at the first OOM, keep the fit.

| | Now | Proposed |
|---|---|---|
| Trials (eb=16) | 8 | 5 |
| Answer | `11` — not runnable | `8` — runnable |
| Output | one integer | largest fitting divisor **+** fitted slope/intercept |
| Code | `search()` ladder, ~40 lines | a loop over divisors, ~10 |

Fewer probes, an actionable answer, more information. Per `program.md`'s Simplicity rule
(*"removing something and getting equal or better results is a great outcome"*), that's a keep.

**Experiment to settle it:** fit `peak_gb` vs `micro_batch` from a real ceiling run and check
linearity (R²). Three real ceiling runs are already on disk (`runs/test_a10G`,
`runs/smollm2_1.7b_a100_wikitext`, `runs/qwen3.5_2b_a100_wikitext`, each `ceiling.tsv`), so this
needs no GPU time. The fit predicts `allocated`; OOM fires on `reserved` + fragmentation, so also
record how far the predicted ceiling sits from the measured one. If the residual is small and
stable, drop `search()`.

---

## 2. Flag a low-composite `effective_batch` to the human

**Status: open, cheap, advisory only.**

`micro_batch` must divide `effective_batch`, so the legal set is
`{divisors of effective_batch} ∩ [1, ceiling]`. A near-prime `effective_batch` collapses it: `17`
has divisors `{1, 17}`, so a ceiling of 11 leaves **only** `micro_batch=1` — no search space at all.

`effective_batch` is the human's and **off-limits** (`loop/throughput.md`), so the loop must never
change it. But Frame can *tell* them at `program.md` Step 3 that 16/24/32/64 buys a real search
space where 17 buys none.

**Proposal:** add a Profile `note` when the legal set has fewer than ~3 values. Sits alongside the
existing padding flag in `frame/profile.py:_notes`.

---

## 3. Divisibility wastes headroom

**Status: open, low priority.**

At ceiling 11 with `effective_batch=16` you run `micro_batch=8` and leave room for 3 samples unused
— the price of holding `effective_batch` exactly.

Uneven accumulation (a smaller final micro-batch) would recover it, but `HFRunner.step` divides the
loss by `grad_accum` uniformly (`frameworks/hf.py`), so uneven batches would be mis-weighted — a
**quality** change, not a speed one. It would need token-count weighting to be safe.

**Not proposed.** Recorded so the waste is a known cost, not an oversight.

---

## 4. Second ceiling at the dataset p99 — the packing signal

**Status: open. One extra call, not wired.**

`find_ceiling(..., seq_len=p99)` alongside the `max_len` one. The **gap between the two ceilings
estimates what `packing` is worth** on this job, before spending a probe on it.

Deferred at implementation time as a nice-to-have. Only meaningful under `padding="longest"`: under
`padding="max_length"` a p99 ceiling describes a regime the adapter never runs in.

---

## 5. Fragmentation is not reproduced by fixed-shape probing

**Status: known limitation, recorded not fixed.**

Fixed-shape trials avoid the allocator fragmentation that varying real lengths cause, so `frag_gb`
understates a long run's drift. `card_frac` is reported so a margin can be kept below the ceiling;
how much margin is the agent's call.

**Experiment:** run at the ceiling for a few hundred steps with varying real lengths and see whether
it survives. This is what `loop/loop.md`'s extended probe is for.

---

## 6. `AutoModelForCausalLM` + a flat config are an unstated single-modality assumption

**Status: open, partly fixed.** Defect 1 (nested configs) is fixed on `main` (`9227a26`,
`frame/model.py:_flatten`), and Qwen3.5-2B has since been run end to end
(`experience/qwen3.5_2b_a100_wikitext.md`). Defects 2 and 3 are open. Raised by the A100/H100 PoC
model selection. The table below is the state **before** that fix.

Two places assume, without saying so, that a checkpoint is a text-only model with a flat config:

- `frame/model.py:internals` reads `num_hidden_layers` / `hidden_size` off the **top level** of the
  config mapping.
- `frameworks/hf.py:135` loads through `AutoModelForCausalLM`.

Modern checkpoints are increasingly natively multimodal and nest their dims under `text_config` /
`vision_config`. Run against the six models the PoC wants (`transformers` 5.15.1):

| Checkpoint | Config shape | `internals()` | `AutoModelForCausalLM` → | Failure |
|---|---|---|---|---|
| `SmolLM2-1.7B-Instruct` | flat | ✅ 1.711 B | `LlamaForCausalLM` | — |
| `Llama-3.2-3B-Instruct` | flat | ✅ | `LlamaForCausalLM` | — |
| `Qwen/Qwen3.5-2B` | nested | ❌ `TypeError` | `Qwen3_5ForCausalLM` | **Frame dies at Step 2** |
| `Qwen/Qwen3.5-9B` | nested | ❌ `TypeError` | `Qwen3_5ForCausalLM` | Frame dies at Step 2 |
| `SmolVLM2-2.2B-Instruct` | nested | ❌ `TypeError` | **`None`** → `ValueError` | dies twice |
| `gemma-3-4b-it` | nested (gated, unverified) | ❌ expected | `Gemma3ForConditionalGeneration` | **silent** |

`internals()` fails as `head_dim = hidden // heads` with both operands `None` — `_get` finds neither
alias at the top level and its `default=None` propagates. The control case is clean: on the flat
SmolLM2 config the analytical count is 1,711,276,032 against a real 1,711,376,384, 0.006% out.

**Three distinct defects, in increasing order of how bad they are:**

1. ~~**Frame crashes on a nested config.**~~ **Fixed** (`frame/model.py:_flatten` reads
   `text_config` as the base and overlays the top level).
2. **No class to load.** `smolvlm` maps to `None` in `MODEL_FOR_CAUSAL_LM_MAPPING_NAMES`, and no
   `SmolVLMForCausalLM` exists anywhere in `transformers` — the module exports only
   `SmolVLMForConditionalGeneration`, `SmolVLMModel` and `SmolVLMVisionTransformer`. Loud, but
   expensive to fix; that's §6b.
3. **The quiet one.** `gemma3` maps to `Gemma3ForConditionalGeneration` in the **CausalLM** table —
   it is the one model type where the causal and image-text-to-text entries are the *same class*. So
   the auto-class hands back the full multimodal model, vision tower included, and **nothing raises**.
   On a text-only batch that buys: AdamW state for ~0.4 B params that receive zero gradient every
   step; an inflated `N` in `roofline.training_flops`'s `6·N·D`, so **MFU reads high** and isn't
   comparable to the flat-config models; and a depressed measured ceiling for a reason unrelated to
   the model under test.

Defect 3 is the one worth the entry. `test_a10G`'s ceiling lesson was that *the probe that never
ran the real shape is the dangerous one*; this is the same failure in a different place: **a wrong
number that looks right beats a crash for cost every time.**

The gap underneath all three: the README's contract is "any model, GPU, and framework," but the
code's contract is "any dense text model with a flat config." That's an undeclared precondition, not
a bug list. **Frame is where preconditions belong** — it should establish or refuse, loudly, rather
than let the runner discover it at loop step 1, and certainly rather than let the number come out
wrong.

**Proposal, cheapest first:**

| Fix | Cost | Buys |
|---|---|---|
| ~~`internals` descends into `text_config` when present~~ | **done** | Qwen3.5 and Gemma frame correctly |
| Flag modality in the Profile notes. `internals()` already reports `modality` (`vision_config` present → `multimodal`); nothing reads it yet | ~5 lines | `frame/profile.py:_notes` tells the agent before it tunes |
| Honour an explicit `model_class` in the run config, falling back to the auto-class | ~10 lines | `Gemma3ForCausalLM` on Gemma; any `*ForConditionalGeneration` with a text-only sibling |
| Refuse at Frame when modality isn't text and no text-only class exists | ~10 lines | SmolVLM fails at Step 2 **with a reason**, not at loop step 1 |

**Experiment to settle it:** run Frame only (Step 2, no loop) against all six PoC checkpoints and
record three things — (a) does `internals()` return; (b) does the analytical param count agree with
`sum(p.numel())` from a real load; (c) on Gemma, how far apart are the measured ceiling and the MFU
between the auto-class load and an explicit `Gemma3ForCausalLM` load. (c) is the one that matters:
it puts a number on what the vision tower costs, and decides whether defect 3 is worth code or just
a Profile note.

---

## 6b. Multimodal is a missing axis, not a missing model

**Status: open. Scoping, not a claim. Proposed as a milestone — deliberately *not* a PoC row.**

The tempting read of §6 defect 2 is "add a second load path and SmolVLM works." It doesn't. The whole
measurement chain assumes a batch is `(input_ids, attention, labels)`:

| Component | What it assumes | What a VLM does instead |
|---|---|---|
| `frame/dataset.py:scan` | one text column; p50/p90/p99 are token counts | most of the sequence is image tokens from a processor — the reported distribution describes the wrong thing |
| `_build_loader` / `_pad_batch` | emits ids + mask + labels | also needs `pixel_values` |
| `_synthetic_batch` | worst case = every row exactly `seq_len` | worst case is *also* image count and resolution — a second dimension the ceiling never probes |
| the knobs | `padding`, `length_grouped`, `packing` — text-shape levers | resolution, tiling, vision-tower freezing; and **freezing is a quality boundary**, so it's off-limits under `loop/throughput.md`, not a knob |
| `roofline.training_flops` | `6·N·D`, N = params, D = tokens | the vision tower runs per *image*, not per token — 6ND is the wrong estimate before the vision tower is even counted |

So a VLM is a new **modality axis** cutting through Frame, the loader, the ceiling and the roofline —
comparable in size to the multi-GPU axis the README already defers, and it should be scheduled like
one. Worth noting that `frame/model.py` already reports a `modality` field: it is the only part of
the chain that anticipated this, and right now nothing reads it.

**Nothing to settle by experiment** — this is scoping. The cheap precursor is §6's fourth row: once
Frame declines non-text checkpoints with a reason, the axis gets built deliberately instead of
discovered by a `ValueError` halfway through a booked GPU hour.

---

## 7. The ceiling for a user's own training script

**Context:** bring your own script (`profiler/launch.py`). When the loop tunes a user's `train.py` through the
launcher, `find_ceiling` (`frame/ceiling.py`, its `fits()` trial) can't run as it does today. It drives OUR
runner with `data="synthetic"`, and a user's script has no such switch. Without a ceiling,
`micro_batch` is unbounded again, and every probe measures memory on whatever rows it happened to
draw. That is exactly the failure the ceiling exists to prevent (`frame/README.md`). This is the most
important gap in the user-script path.

**The claim that it can't be measured was too strong.** With a torch `DataLoader`, the worst
case is reachable through public attributes (torch 2.13 `dataloader.py`):
- `loader.dataset` (`:300`)
- `loader.collate_fn` (`:421`)
- `loader.batch_size` (`:409`)

**A dataset-aware worst case** follows from them:
1. Measure every row's length (`len(input_ids)`, cheap on an Arrow-backed HF dataset).
2. Take the `micro_batch` longest rows, truncated at `max_len`.
3. Pass them through the script's own `collate_fn`.

That gives the widest batch *this* dataset can produce, in exactly the format the model sees. When
fewer than `micro_batch` rows reach `max_len`, it is narrower than `[micro_batch, max_len]`, so
the real ceiling is higher than the synthetic one.

**A synthetic `max_len` probe is also a data change, not a new mechanism.** Either:
- **Config:** the agent writes a small file of rows that tokenize past `max_len` and points the
  probe's YAML at it (swift / axolotl / LLaMA-Factory take dataset paths). No patching;
  framework-specific file format.
- **Batch substitution:** during a ceiling probe, the loader yields the worst-case batch instead
  of real ones. Framework-agnostic, but it alters data. Acceptable only for a memory probe the
  contract already marks `probe_kind: synthetic_ceiling` and never scores for throughput.

**What makes it hard:**

| Situation | Problem |
|---|---|
| Streaming / `IterableDataset` | no `len`, no random access. Finding the longest rows means one full pass (cacheable, possibly unbounded) |
| Tokenization in `__getitem__` / `collate_fn` | length is only known after the transform, so the scan runs it on every row |
| Length isn't what sets memory | images per row (multimodal), routing (MoE) |
| Getting the loader object | public for HF (`train_dataloader` is passed to `TrainerCallback`s). In a raw loop, only via the iterator's private `_dataset` / `_collate_fn` or a patched `DataLoader.__iter__` |
| **Cost: one launch per trial** | the bisection needs ~2·log₂(n) trials. Our runner survives an OOM in-process; a user's script doesn't (the exception escapes `trainer.train()`). So each trial is a separate launch, and each launch reloads the checkpoint with `from_pretrained`. That's minutes per trial for a 7B model |

**Options, cheapest measurement first:**
1. Separate launch per trial, bisection seeded by the dataset-aware worst case (fewer trials).
2. **In-process trials.** After the script has built its model and optimizer (both reachable
   from the hooks), run the fwd/bwd/step trials ourselves on worst-case batches, in one launch.
   That keeps the script's model setup (autocast wrapper, checkpointing, attention
   implementation) and bypasses its loop and loss. Fine for memory, but it needs checking.
3. Analytical estimate only (`frame/memory.py`). Weaker: `beliefs/synthetic-ceiling-
   underestimates-real-peak.md` says even the measured ceiling runs low.

**Decided (2026-09-25): the synthetic-dataset config route.** A torch `DataLoader` is assumed, so
the probe config points the script's own data setting at a small synthetic file of rows that
tokenize past `max_len`. The framework's own template, collator and loader then build
worst-case batches in exactly the real format. No patching.
- One launch per bisection trial: an OOM ends a user's run.
- When the ceiling goes stale, re-measure **from the previous ceiling** (step up if it still
  fits, down if it OOMs; typically 1–3 trials), not from scratch.
- Every trial checks its batch shape from the trace (`Input Dims` of the embedding) and fails
  loudly unless it's `[micro_batch, max_len]`. This catches frameworks that drop or split
  over-length rows.

**Prerequisite, done:** the launcher reports a CUDA OOM as `status: "oom"`, with memory at
failure. `find_ceiling` refuses `error` as a memory answer, and reads only `oom` as "doesn't
fit". Tested on CPU (`tests/test_launch.py`) and with a real OOM on the A10 (SmolLM2-360M,
micro-batch 64; and `docs/verification/byo_script.md` row 14).

**Experiment that would settle it:** on one model and config, through `profiler/launch.py`,
compare three ceilings:
- (a) today's synthetic `find_ceiling`;
- (b) the dataset-aware worst case, one launch per trial;
- (c) the same, in-process.

It settles it if (b) and (c) agree with each other and are ≥ (a) wherever the dataset has fewer
than `micro_batch` rows at `max_len`. If (c) matches (b), option 2 wins on cost.

---

## 8. Define the noise floor between launches, not within one

**The claim:** the loop decides "within the noise floor" (`loop/loop.md:83`, `:106`), but the only
noise floor it measures is the **within-run** coefficient of variation (`step_time_cv`, from one
probe's clean window). Configs are compared **across launches**: a fresh process each, with its
own allocator state, clocks and model load. So the between-launch spread is the noise that
actually separates two probes, and it is not defined anywhere the loop reads.

**What already exists (practice and observation, not a threshold):**
- Champions are re-probed and "confirmed twice": 0.6% apart (`experience/test_a10G.md:11`), 0.04% (`experience/smollm2_1.7b_a100_wikitext.md:11`),
  0.31% (`:220`).
- Between-run agreement was measured once, after the fact: "repeats of identical configs agreed
  to 0.4-0.9%" while within-run cv was 0.29-0.37 (`experience/test_a10G.md:49-50`).
- A "noise band" is used informally to call levers dead (`experience/smollm2_1.7b_a100_wikitext.md:40`, `experience/qwen3.5_2b_a100_wikitext.md:38, 54, 120`).
- The contract's `reliability` field (`loop/contracts.md:70`) is within-probe only.

**Why it matters: the two floors disagree, in both directions.**

| Evidence | Within-run floor | Between-launch spread |
|---|---|---|
| `experience/test_a10G.md:49-50` (dynamic padding) | cv 0.29-0.37 | repeats 0.4-0.9%. **Within-run is far too loose** |
| BYO experiment 4b, SmolLM2-360M, fixed shapes | 0.15-0.16% | 0.5-0.9% round to round. **Within-run is ~4× too tight** |
| exp. 4b, Qwen2.5-0.5B | 0.17-0.18% | 0.2-0.4% |

Too tight means a 0.4% "win" gets believed when it's launch-to-launch noise. The old-vs-new
capture comparison (BYO experiment 5) nearly made exactly that mistake:
5 rounds, one fixed order. Too loose means a real 2% gain gets thrown away as "inside noise".

**Proposed:**
1. **Measure the between-launch floor at Frame,** once per (model, hardware, framework): an A/A
   of the baseline config, k fresh launches in shuffled order. Record the spread on the Profile
   next to the ceiling.
2. **Use it as the decision threshold.** A candidate that beats the incumbent by less than the
   between-launch floor is "not yet a win": re-probe both, interleaved, before promoting it.
   This formalizes the existing "confirmed twice" habit and applies it before promotion, not
   just to the final champion.
3. **Rank on the mean as well as the median.** Throughput is a mean;
   `beliefs/median-step-time-overstates-skewed-throughput.md` already shows the median can
   overstate it under skew.
4. **Report both floors in the Observation's `reliability`:** within-run (steady-state inside
   one probe) and between-launch (whether two probes differ).

**Why it might be wrong:** k extra launches at Frame cost real time, especially with
`from_pretrained` reloads, and the between-launch spread may itself drift over a long run
(thermals, other tenants). It may need re-measuring, not a one-off.

**First data point (2026-09-25):** the A/A/B control found:
- a **within-round position effect** of +0.7 / +1.6 ms for the 2nd / 3rd launch;
- **session drift** of about 1.6% (1.270 → 1.291 s);
- a 1.9% max − min spread across 30 launches of three near-identical arms, against a
  within-run cv of about 0.15%.

So between-launch comparisons also need **shuffled or interleaved order**, not just repeats. A
fixed order turned a ~1.5 ms effect into an apparent ~4 ms one (BYO
experiment 5).

**Experiment that would settle it:** the A/A/B run on the A10 (`new` vs `new` vs
`old` capture, 10 shuffled rounds, SmolLM2-360M) is the first controlled between-launch
measurement. Repeat it for 2-3 (model, card) pairs; if the between-launch spread is consistently
more than 2× the within-run cv, adopt (1)-(2).

---

## 9. Give the agent a metric reference: what each probe field measures, and what it doesn't

**The claim:** the agent's decisions are only as good as its reading of the Observation, and today
nothing it reads says what each field **does not** measure. `program.md:56` tells it to "skim"
`profiler/README.md`, whose 41-row table gives one-line meanings and a few traps. The
**limitations** we've established are all in developer notes the agent never sees
(the BYO experiment notes, the metric-coverage audit, this file). So the agent can make
the mistakes those notes record.

**Mistakes the notes document that an uninformed agent would repeat:**

| Trap | Where it's recorded |
|---|---|
| Comparing host phase times across loops (Trainer syncs every micro-batch, which moves where the host waits) | BYO experiment 4 |
| Believing a win smaller than the between-launch spread (the within-run cv is ~4× too tight on long fixed-shape steps) | §8 |
| Median step time overstating throughput under skew | `beliefs/median-step-time-overstates-skewed-throughput.md` |
| Reading data-wait 0.0 as "not data-bound" when there's no torch `DataLoader` | metric-coverage audit §0 |
| MFU counting pad tokens as work; MFU wrong under LoRA (trainable-only N) | `experience/test_a10G.md`, metric-coverage audit §5 |
| A step time that reads ~0.25% high from a per-step sync placed before Trainer's bookkeeping (attached capture) | BYO experiment 6 |

**Proposed: `profiler/metrics/`, the single source of truth for field semantics.**

```
profiler/metrics/
  README.md        table of contents: every field, one-line meaning, what to trust it for,
                   link to its section
  timing.md        step_time_s, step_time_cv, profiled_step_s
  throughput.md    flops, mfu, roofline_floor_s, peak_flops, peak_dtype, peak_mem_bw_gb_s
  gpu.md           gpu_busy_s, gpu_active_frac, gap_s, gap_frac, gap_count, max_gap_s
  phases.md        phase_times_s, gpu_phase_times_s, gpu_phase_coverage
  data.md          data_wait_s, data_wait_frac
  cpu.md           cpu_util_mean, cpu_util_max, nproc, cpu_cores_busy
  memory.md        peak_mem_gb, reserved_mem_gb, mem_total_gb, headroom_gb
  kernels.md       top_kernels, kernel_count, top8_coverage, provenance,
                   achieved_occupancy_pct, limiting_factor
  run.md           status, error, gpu, device, device_type, trace_path, summary
```

- **Grouped by family, not one file per field.** Related fields share caveats: every timing field
  inherits the sync bias, every phase field inherits host-vs-GPU. Nine docs are maintainable;
  41 would drift.
- **One template per field:** *Measures* (what, unit, how computed, from which source) ·
  *Doesn't measure* (what it's easy to assume it means) · *Known biases / traps* · *Null when* ·
  *Evidence* (the experiment or `beliefs/` belief behind each claim, linked, so a claim can be
  re-checked or retired).
- **Single source of truth:** `profiler/README.md`'s 41-row field table shrinks to a pointer, so
  the two can't drift.
- **Routed where it's used:** a pointer at the Observe step of `loop/loop.md` (where the agent
  interprets an Observation), not only at `program.md` step 1's "skim".
- **Capture modes:** both are live: the runner capture (`capture()`) and the attached capture
  for user scripts (`profiler/attach.py`). Document the attached capture's differences per field
  (padded tokens, trainable-only params, trace-derived phases, DataLoader-only data-wait, the
  timing method, unknown tokens when the model input has no token tensor). Today they're only
  summarized in `profiler/README.md`, "Attached capture".

**Why it might be wrong:**
- More reading for the agent at the point of decision. The index must stay short enough to scan,
  with detail only on demand.
- Semantics docs rot when code changes. Mitigation: each claim cites its evidence, and the
  field list in the index is checked against `Observation` in `profiler/diagnostic.py` (a test
  can assert every dataclass field has an index row).

**What would settle it:** after building it, replay a past run's decisions (e.g. the
`experience/` retrospectives) with the reference available. Count how many recorded
misreadings the doc would have prevented. At minimum: the host-phase comparison, the
inside-noise calls, and the MFU-with-padding reading.

---

## 10. MFU is wrong whenever weights are frozen (LoRA / PEFT); keep "model FLOPs", fix the count

**Scope and owner:** a **separate branch**, now unblocked: the bring-your-own-script work it waited
on has merged (#17, #18). Handed to a colleague; this entry is written to be read cold. Until it
lands, MFU under LoRA is a documented known limitation of the BYO-script path.

**The problem, measured.** BYO experiment 7, 1× A10, peft 0.20.0, LoRA
r=16 on every attention / MLP projection:

| Model | Trainable / total params | MFU reported | MFU with the corrected count |
|---|---|---|---|
| SmolLM2-360M | 8.68M / 370.5M (2.34%) | 0.53% | ≈15.3% (28.8× low) |
| Qwen2.5-0.5B | 8.80M / 502.8M (1.75%) | 0.51% | ≈19.7% (38.4× low) |

A 0.5% MFU tells the agent the GPU is idle, and sends it after problems that don't exist.

**Where the error is.** The pipeline is: probe measures N, D, step time and peak → 6·N·D
(`profiler/roofline.py:17`) → MFU = FLOPs / time / peak (`:25-29`). The reporting arithmetic is
correct. Two inputs are wrong, pulling in opposite directions:
1. **N, on the probe side (the big error).** The attached capture takes N from the
   optimizer's `param_groups` (`profiler/attach.py`, `self.params`), i.e. trainable params only.
   OG runners count every parameter (`frameworks/hf.py`, `self.params`), so this is new with the
   BYO-script path.
2. **The 6 in 6ND (the smaller error).** 6 = 2 (forward) + 2 (gradients back through
   activations) + 2 (weight gradients). Frozen weights skip the last 2. With N_total plugged into
   6ND, SmolLM2 + LoRA reads ≈22.5%: ~1.5× high.

**What MFU is for** (why the definition matters, not just the number):
- **Scoring against the possible:** "Primary metric: tokens/s, reported as MFU (% of the
  hardware roofline)" (`loop/throughput.md:6`); `objective` = tokens/s + MFU
  (`loop/contracts.md:65`).
- **Stopping:** stop when near the roofline (`loop/loop.md:106`); report the "% of roofline
  reached" (`:112`).
- **Diagnosis:** "the 'is the GPU utilized well' signal" (`profiler/README.md:44`), with measured
  bandwidth for compute- vs memory-bound.
- **Cross-run comparison:** every retrospective in `experience/` reports it.
- **Not** within-run ranking. That's tokens/s or samples/s, and MFU already misled there under
  static padding (`beliefs/static-padding-is-usually-the-bottleneck.md:39`).

**Keep the abstraction: "model FLOPs", not "executed FLOPs".** MFU counts the math the model
*requires*, by structure, whatever the implementation does. Counting FLOPs from the trace's matmul
shapes would turn it into **HFU (hardware FLOPs utilization)** and break the comparison MFU exists
for:
- Gradient checkpointing's recomputed forwards would count as work, so the metric would **rise**
  while training got **slower**.
- Other redundant work would also inflate it.

**Proposed:**
1. **Generalize the formula; don't replace it.**
   `model FLOPs / step = (2·N_total + 2·N_total + 2·N_trainable) · D = (4·N_total + 2·N_trainable) · D`.
   With every weight trainable, this is **exactly 6ND**, so every past MFU in `experience/` stays
   comparable. Under LoRA it gives the corrected column above. `training_flops(params, tokens)`
   becomes `training_flops(total, trainable, tokens)`.
2. **Report N_total and N_trainable separately** (Observation `provenance`, or new fields; check
   `tests/test_diagnostic.py:179-189`, which pins the README's field count).
   - **N_trainable:** exact today, from the optimizer (public API).
   - **N_total, OG runners:** they already know it (`frameworks/hf.py`, `self.params`); pass it through as a
     declared input.
   - **N_total, user scripts, patch-free:** sum the weight shapes of the matmul ops
     (`aten::linear` / `aten::addmm` / `aten::mm`, `Input Dims`) in **one forward pass** of the
     profiled step (`record_shapes=True` is already on). That's the parameters doing math per
     token, which is the right N for FLOPs. Embedding lookups do no FLOPs, and a tied `lm_head`
     shows up as its matmul. Use the forward span only (`derive.py`), so checkpointing's
     recomputes, which run inside backward, don't count. **Untested.**
3. **Optional, separate diagnostic: HFU**, from trace-counted matmul FLOPs over the whole step.
   HFU − MFU is the recompute / redundant work, useful for judging gradient checkpointing. It is
   never the objective.
4. **Known, separate, not fixed here:** D counts **padded** tokens (`frameworks/hf.py:264`,
   `derive.py`), so MFU still counts pad as work under static padding (`experience/test_a10G.md:47-48`).
   Worth its own entry if the objective moves to real tokens.

**Why it might be wrong:**
- The trace-derived N_total may not equal a model's true parameter count. Embeddings and norms
  are excluded, and untied embeddings drop out entirely. That changes absolute MFU vs historical
  6ND runs on large-vocabulary untied models. Decide whether continuity (count all params) or
  correctness (count matmul params) wins, and record which.
- `4·N_total` for backward-through-activations is itself approximate. No gradient flows into
  a frozen first layer's input or into the embedding.
- Attention's own FLOPs (~12·layers·d_model·seq per token) are ignored, as today
  (`profiler/roofline.py:5`). That's a separate refinement.

**What would settle it:** on a GPU, with the experiment 4b / 7 Trainer scripts
(`sandbox/attach_parity/hf/` in tag `sandbox-archive`):
1. **Full fine-tune** (experiment 4b configs): the new formula must equal 6ND within a few
   percent, using the trace-derived N_total. That validates the N_total extraction.
2. **LoRA** (experiment 7 configs): MFU lands at ≈15% / ≈20%, not 0.5%.
3. **Gradient checkpointing on vs off:** MFU must not rise when checkpointing slows training, while
   HFU does. That demonstrates the model-vs-hardware distinction.

Suggested branch: `mfu-frozen-weights`, cut from `main`.

---

## 11. Compare the same examples without timing a whole pass

**Status:** a proposal to change PR #14's clean-window policy. **Not adopted**: PR #14's rule
(the clean window covers whole passes over the sample, starting at a pass boundary;
`loop/contracts.md`, Measurement axiom corollary 4) stays in force until this is settled. Raised
2026-09-27 while auditing the bring-your-own-script path (axiom audit B2).

**The claim:** the axiom's goal is that two configs are compared **on the same examples**. Timing
every row of the sample is one way to guarantee that, and an expensive one. Under a fixed seed it
is usually unnecessary, and where it *is* necessary (length grouping) there may be a cheaper way to
guarantee the same examples.

**What a whole pass costs:** `steps_per_pass = sample rows ÷ effective_batch`. With PR #14's
example (992 rows, `effective_batch=16`) that's 62 steps, against 10 requested, per probe. For a
user's script, which can't be restarted at a pass boundary the way OG's `start_pass()` does, it
also means warming up for a full pass (frameworks reshuffle each epoch). That's ~125 steps vs ~14.
On SmolLM2 at 1.3 s/step, it's ~2.7 min vs ~18 s per probe.

**Why a short window already times the same examples for most comparisons:** with a fixed seed, the
sampler shuffles row **indices**. `micro_batch` / `grad_accum` only decide how many consecutive
rows form a step, and `effective_batch` is fixed. So the first K steps are the same K × 16 rows, in
the same order, for every config that doesn't change the sampler: micro-batch, grad accum, padding
mode, attention implementation, compile, checkpointing, chunked CE. Two conditions:
- **the whole probe fits inside epoch 1**, so a per-epoch reshuffle never happens (14 steps < 62);
- **the sampler is seeded and doesn't depend on batch size.** HF Trainer's `seed` is fixed by
  default (42); the agent checks for a custom sampler; a repeat probe of the same config must show
  an identical per-step token sequence (the B1 counts) as a fingerprint of the rows timed.

**The case that isn't covered: length grouping on vs off.** Grouping changes the sampler's order,
so the first K steps hold **different examples**. The requirement stands: **a length-grouped vs
ungrouped comparison must be on the same examples.** Options to guarantee that without a whole pass
over the full sample:

| Option | How | Catch |
|---|---|---|
| (a) **Window = whole megabatches** | HF's `LengthGroupedSampler` shuffles indices, splits them into megabatches (50 × batch size), and sorts **within** each megabatch by length. If the same seeded permutation feeds both samplers, the **set** of rows in megabatch 1 is the same whether or not it's then sorted; only the order inside it changes. A window covering whole megabatches then times the same examples | **To verify:** whether `LengthGroupedSampler` and the default `RandomSampler` draw the same permutation from the same seed (implementation-specific, and version-specific). Megabatch size scales with batch size, so it differs between configs that change `micro_batch` |
| (b) **A small comparison sample, one whole pass over it** | Frame draws a subset of the probe sample, sized to the window (e.g. 160 rows = 10 steps at `effective_batch=16`). Both configs time one whole pass over just that subset, so they see the same examples in whatever order | a small subset represents the length distribution less faithfully; it needs its own stratified draw |
| (c) **Only length-grouping comparisons pay for a whole pass** | keep a short window for everything else; when the two configs differ in a sampler-changing knob, time both over whole passes | two window policies to track; the population must say which one each probe used |

**Recording, whatever is chosen:** the population must say how the timed examples were selected
(sample hash, window start and length, sampler signature: seed and grouping). "Same rows" is
claimed only when they match, or when the option above makes them provably the same set.

**Why it might be wrong:**
- A short window is a subset of the sample, so its length mix is a random draw from it, not the
  stratified whole. Throughput from 160 rows may be noisier than from 992.
- Fixed-seed determinism is framework behaviour, not a guarantee; a framework upgrade could change
  the sampler.
- (a) relies on HF implementation details.

**What would settle it:** on one model with `padding="longest"`:
1. Short-window vs whole-pass ranking over several non-sampler knobs: does the order of configs
   agree?
2. Length-grouped vs ungrouped under (a): log the example IDs in the window for both, and check
   they're the same set.
3. The between-launch noise of a short window vs a whole pass (§8).

Adopt the short window if rankings agree and noise is within the floor.

**Interaction:** OG and user-script probes should share one clean-window rule, or their throughput
isn't comparable (axiom audit B5).

