# Design improvements

> **Developer notes. Not part of the agent's prompt hierarchy.**
> `program.md` does not route here and the loop never reads or writes this file. It's a scratchpad
> for us to decide what to build next — the doctrine the agent follows lives in `program.md`,
> `loop/`, and the per-directory READMEs.
>
> Paths into `sandbox/attach_parity/` below refer to git tag **`sandbox-archive`**: the harness was
> removed from the tree on 2026-10-02 (its capture now lives in `profiler/attach.py` / `launch.py`).

Open questions and proposed changes that are **not yet settled by evidence**. Each entry states the
claim, why it might be wrong, and **what experiment would settle it** — so a future eval can close
it rather than re-litigate it.

Nothing here is a decision. Once an experiment settles an entry, make the change in the code and
the owning md, then delete the entry from here.

---

## 1. ~~The synthetic ceiling may be redundant~~ — SETTLED: it is load-bearing

**Status: closed by `test_a10G`.** Kept here until the finding is folded into a belief.

The doubt was: the ceiling was argued for on the grounds that real batches rarely hold a `max_len`
row, but `frameworks/hf.py` shipped `padding="max_length"`, so every real batch was *already*
worst-case shaped. On that loader the ceiling was a determinism-and-cost win, not a safety one.

**The run removed that loader.** Static padding turned out to be the dominant bottleneck, and the
winning config uses `padding="longest"` — which is exactly the condition this entry named as making
the ceiling load-bearing. The evidence arrived in the same run that raised the doubt.

The sharper form of the argument, which the original entry missed: it is not only that shapes vary,
it is that **a probe sees a sample and the sample's maximum is not the population's maximum.**

| | |
|---|---|
| Rows scanned | 256 (`sample`), giving p50 10 · p99 374 · **max 520** |
| Declared `max_len` | **2048** |
| Widest batch any of 23 experiments ran | ~520 tok |
| Widest batch the full dataset can produce | **2048** — one long row pads its whole batch |

So `[8, 2048]` — the batch the shipped config *will* meet in training — was never run by any probe.
Only the ceiling ran it (`ceiling.tsv`, final block): 17.386 GB allocated / 20.721 GB reserved of
22.1 GB, fits at `micro_batch=8`, **OOMs at 9**. The shipped config is safe, and safe by one.

**Doctrine that came out of it** (now in `loop/throughput.md`, `frame/README.md`, `program.md`):
`padding="longest"` is safe **iff `micro_batch <= ceiling(max_len)`**, and `micro_batch` must NEVER
be raised on a probe's reported `headroom_gb` — under dynamic padding that headroom is an artifact
of which rows were drawn.

**Still to do:** promote this to a belief in `knowledge/` and delete the entry.

---

## 1b. The probe sample should be drawn to match the length distribution

**Status: open. Raised by `test_a10G`; the stopgap (raise `sample` until the tail agrees) is
doctrine, this is the real fix.**

`HFRunner._build_loader` takes the **first** `sample` usable rows (`_usable_texts` breaks at the
cap). Under `padding="max_length"` that was harmless — every batch was `seq_len` wide regardless.
Under `padding="longest"` batch width *is* the rows drawn, so a head-of-file slice measures a
workload the run won't train on. `test_a10G` measured its 25x on 256 rows whose max was ~520, while
real training meets rows up to the 2048 truncation cap.

Raising `sample` helps but is the blunt instrument: it costs memory and setup time linearly, and a
head slice can still miss the tail entirely if the file is ordered.

**Options, cheapest first:**

| Approach | Cost | Catch |
|---|---|---|
| Raise `sample` until the loader's p50/p90/p99/max match frame's scan | linear memory/time | still a head slice; ordered files defeat it |
| Reservoir-sample the stream instead of taking the head | ~free | needs a pass, but `scan()` already streams |
| Stratify by length: keep the observed p50/p90/p99 proportions | cheap | the honest fix — reproduces the distribution, not just its size |

**Experiment to settle it:** measure the winning config's samples/s at `sample` = 256, 1k, 4k, and
against a stratified draw. If throughput falls monotonically and then flattens, the flat point is
the sample size that represents the dataset — and the gap from 256 quantifies how overstated the
25x is.

**Related:** `frame/dataset.py scan()` already computes exactly the distribution the loader should
be matched against, and `frame/profile.py` already flags `max_len` >> p99. The check is one
comparison away from free.

---

## 2. Replace the bisection with divisors + a memory curve

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
linearity (R²). The fit predicts `allocated`; OOM fires on `reserved` + fragmentation, so also
record how far the predicted ceiling sits from the measured one. If the residual is small and
stable, drop `search()`.

---

## 3. Does `micro_batch_max` ever bind?

**Status: open.**

Two bounds constrain `micro_batch`, and only the smaller one matters:

- `micro_batch <= effective_batch` — **arithmetic**, free, already enforced twice
  (`frameworks/hf.py:80`, `find_ceiling(cap=...)`).
- `micro_batch <= ceiling` — **physical**, needs measurement.

The measured ceiling only earns its cost when it comes in **below** `effective_batch`. When it
doesn't, `capped: true` is already reporting exactly that, and the measurement bought nothing.

**Experiment to settle it:** across a few realistic (model, card, max_len) triples, record how often
the ceiling lands below `effective_batch`. If it rarely does at the sizes this repo targets, demote
the ceiling from a Frame-time step to something the agent calls only when a config OOMs.

---

## 4. Flag a low-composite `effective_batch` to the human

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

## 5. Divisibility wastes headroom

**Status: open, low priority.**

At ceiling 11 with `effective_batch=16` you run `micro_batch=8` and leave room for 3 samples unused
— the price of holding `effective_batch` exactly.

Uneven accumulation (a smaller final micro-batch) would recover it, but `HFRunner.step` divides the
loss by `grad_accum` uniformly (`frameworks/hf.py`), so uneven batches would be mis-weighted — a
**quality** change, not a speed one. It would need token-count weighting to be safe.

**Not proposed.** Recorded so the waste is a known cost, not an oversight.

---

## 6. Second ceiling at the dataset p99 — the packing signal

**Status: open. One extra call, not wired.**

`find_ceiling(..., seq_len=p99)` alongside the `max_len` one. The **gap between the two ceilings
estimates what `packing` is worth** on this job, before spending a probe on it.

Deferred at implementation time as a nice-to-have. Note that entry 1 weakens it too: under static
`padding="max_length"` a p99 ceiling describes a regime the adapter never runs in.

---

## 7. Fragmentation is not reproduced by fixed-shape probing

**Status: known limitation, recorded not fixed.**

Fixed-shape trials avoid the allocator fragmentation that varying real lengths cause, so `frag_gb`
understates a long run's drift. `card_frac` is reported so a margin can be kept below the ceiling;
how much margin is the agent's call.

**Experiment:** run at the ceiling for a few hundred steps with varying real lengths and see whether
it survives. This is what `loop/loop.md`'s extended probe is for.

---

## 8. The ceiling has never run on a GPU

**Status: blocking. Everything above is unvalidated until this is done.**

`search()` is unit-tested pure, and the live path is walked on CPU against a *simulated*
`torch.cuda.OutOfMemoryError`. Off-GPU every trial's memory field is `None`, so the bisection has
never seen a real OOM, a real `peak_gb`, or real fragmentation.

**This is the first experiment to run.** See "Initial experiments" below.

---

## 10. The ceiling for a user's own training script

**Context:** `bring_your_own_script.md`. When the loop tunes a user's `train.py` through the
launcher, `find_ceiling` (`frame/ceiling.py:109-116`) can't run as it does today. It drives OUR
runner with `data="synthetic"`, and a user's script has no such switch. Without a ceiling,
`micro_batch` is unbounded again, and every probe measures memory on whatever rows it happened to
draw. That is exactly the failure §1 settled the ceiling exists to prevent. This is the most
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
3. Analytical estimate only (`frame/memory.py`). Weaker: `knowledge/synthetic-ceiling-
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
failure. `frame/ceiling.py:126-133` refuses `error` as a memory answer, and reads only `oom`
as "doesn't fit". Tested on CPU (`sandbox/attach_parity/tests/test_launcher.py`, 8 cases) and
with a real OOM on the A10 (SmolLM2-360M, micro-batch 64).

**Experiment that would settle it:** in `sandbox/attach_parity/`, on one model and config,
compare three ceilings:
- (a) today's synthetic `find_ceiling`;
- (b) the dataset-aware worst case, one launch per trial;
- (c) the same, in-process.

It settles it if (b) and (c) agree with each other and are ≥ (a) wherever the dataset has fewer
than `micro_batch` rows at `max_len`. If (c) matches (b), option 2 wins on cost.

---

## 11. Define the noise floor between launches, not within one

**The claim:** the loop decides "within the noise floor" (`loop/loop.md:83`, `:106`), but the only
noise floor it measures is the **within-run** coefficient of variation (`step_time_cv`, from one
probe's clean window). Configs are compared **across launches**: a fresh process each, with its
own allocator state, clocks and model load. So the between-launch spread is the noise that
actually separates two probes, and it is not defined anywhere the loop reads.

**What already exists (practice and observation, not a threshold):**
- Champions are re-probed and "confirmed twice": 0.6% apart (`knowledge.md:21`), 0.04% (`:117`),
  0.31% (`:220`).
- Between-run agreement was measured once, after the fact: "repeats of identical configs agreed
  to 0.4-0.9%" while within-run cv was 0.29-0.37 (`knowledge.md:59-60`).
- A "noise band" is used informally to call levers dead (`knowledge.md:146, 243, 259, 325`).
- The contract's `reliability` field (`loop/contracts.md:70`) is within-probe only.

**Why it matters: the two floors disagree, in both directions.**

| Evidence | Within-run floor | Between-launch spread |
|---|---|---|
| `knowledge.md:59-60` (dynamic padding) | cv 0.29-0.37 | repeats 0.4-0.9%. **Within-run is far too loose** |
| `bring_your_own_script.md` exp. 4b, SmolLM2-360M, fixed shapes | 0.15-0.16% | 0.5-0.9% round to round. **Within-run is ~4× too tight** |
| exp. 4b, Qwen2.5-0.5B | 0.17-0.18% | 0.2-0.4% |

Too tight means a 0.4% "win" gets believed when it's launch-to-launch noise. The old-vs-new
capture comparison (`bring_your_own_script.md` experiment 5) nearly made exactly that mistake:
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
   `knowledge/median-step-time-overstates-skewed-throughput.md` already shows the median can
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
fixed order turned a ~1.5 ms effect into an apparent ~4 ms one (`bring_your_own_script.md`,
experiment 5).

**Experiment that would settle it:** the A/A/B run on the A10 (`new` vs `new` vs
`old` capture, 10 shuffled rounds, SmolLM2-360M) is the first controlled between-launch
measurement. Repeat it for 2-3 (model, card) pairs; if the between-launch spread is consistently
more than 2× the within-run cv, adopt (1)-(2).

---

## 12. Give the agent a metric reference: what each probe field measures, and what it doesn't

**The claim:** the agent's decisions are only as good as its reading of the Observation, and today
nothing it reads says what each field **does not** measure. `program.md:56` tells it to "skim"
`profiler/README.md`, whose 41-row table gives one-line meanings and a few traps. The
**limitations** we've established are all in developer notes the agent never sees
(`bring_your_own_script.md`, `profiler_metric_coverage.md`, this file). So the agent can make
the mistakes those notes record.

**Mistakes the notes document that an uninformed agent would repeat:**

| Trap | Where it's recorded |
|---|---|
| Comparing host phase times across loops (Trainer syncs every micro-batch, which moves where the host waits) | `bring_your_own_script.md`, experiment 4 |
| Believing a win smaller than the between-launch spread (the within-run cv is ~4× too tight on long fixed-shape steps) | §11 |
| Median step time overstating throughput under skew | `knowledge/median-step-time-overstates-skewed-throughput.md` |
| Reading data-wait 0.0 as "not data-bound" when there's no torch `DataLoader` | `profiler_metric_coverage.md` §0 |
| MFU counting pad tokens as work; MFU wrong under LoRA (trainable-only N) | `knowledge.md`, `profiler_metric_coverage.md` §5 |
| A step time that reads ~0.25% high from a per-step sync placed before Trainer's bookkeeping (attached capture) | `bring_your_own_script.md`, experiment 6 |

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
  *Evidence* (the experiment or `knowledge/` belief behind each claim, linked, so a claim can be
  re-checked or retired).
- **Single source of truth:** `profiler/README.md`'s 41-row field table shrinks to a pointer, so
  the two can't drift.
- **Routed where it's used:** a pointer at the Observe step of `loop/loop.md` (where the agent
  interprets an Observation), not only at `program.md` step 1's "skim".
- **Capture modes:** document the runner capture (`capture()`) that exists today. When the
  attached capture (user scripts) is promoted out of the sandbox, add its differences per field
  (padded tokens, trainable-only params, trace-derived phases, DataLoader-only data-wait, the
  timing method). Until then they stay in developer notes, so the agent never reasons about a
  mode it can't use.

**Why it might be wrong:**
- More reading for the agent at the point of decision. The index must stay short enough to scan,
  with detail only on demand.
- Semantics docs rot when code changes. Mitigation: each claim cites its evidence, and the
  field list in the index is checked against `Observation` in `profiler/diagnostic.py` (a test
  can assert every dataclass field has an index row).

**What would settle it:** after building it, replay a past run's decisions (e.g. the
`knowledge.md` retrospectives) with the reference available. Count how many recorded
misreadings the doc would have prevented. At minimum: the host-phase comparison, the
inside-noise calls, and the MFU-with-padding reading.

---

## 13. MFU is wrong whenever weights are frozen (LoRA / PEFT); keep "model FLOPs", fix the count

**Scope and owner:** a **separate branch, after the bring-your-own-script work lands**
(`integration_plan.md`). Handed to a colleague; this entry is written to be read cold. Until it
lands, MFU under LoRA is a documented known limitation of the BYO-script path.

**The problem, measured.** `bring_your_own_script.md` experiment 7, 1× A10, peft 0.20.0, LoRA
r=16 on every attention / MLP projection:

| Model | Trainable / total params | MFU reported | MFU with the corrected count |
|---|---|---|---|
| SmolLM2-360M | 8.68M / 370.5M (2.34%) | 0.53% | ≈15.3% (28.8× low) |
| Qwen2.5-0.5B | 8.80M / 502.8M (1.75%) | 0.51% | ≈19.7% (38.4× low) |

A 0.5% MFU tells the agent the GPU is idle, and sends it after problems that don't exist.

**Where the error is.** The pipeline is: probe measures N, D, step time and peak → 6·N·D
(`profiler/roofline.py:17`) → MFU = FLOPs / time / peak (`:25-29`). The reporting arithmetic is
correct. Two inputs are wrong, pulling in opposite directions:
1. **N, on the probe side (the big error).** The sandbox measurement takes N from the
   optimizer's `param_groups` (`sandbox/attach_parity/attach.py:115`), i.e. trainable params only.
   OG runners count every parameter (`frameworks/hf.py:156`), so this is new with the BYO-script
   path.
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
- **Cross-run comparison:** every retrospective in `knowledge.md` reports it.
- **Not** within-run ranking. That's tokens/s or samples/s, and MFU already misled there under
  static padding (`knowledge/static-padding-is-usually-the-bottleneck.md:39`).

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
   With every weight trainable, this is **exactly 6ND**, so every past MFU in `knowledge.md` stays
   comparable. Under LoRA it gives the corrected column above. `training_flops(params, tokens)`
   becomes `training_flops(total, trainable, tokens)`.
2. **Report N_total and N_trainable separately** (Observation `provenance`, or new fields; check
   `tests/test_diagnostic.py:179-189`, which pins the README's field count).
   - **N_trainable:** exact today, from the optimizer (public API).
   - **N_total, OG runners:** they already know it (`frameworks/hf.py:156`); pass it through as a
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
   `derive.py`), so MFU still counts pad as work under static padding (`knowledge.md:57-58`).
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

**What would settle it:** in `sandbox/attach_parity/hf/`:
1. **Full fine-tune** (experiment 4b configs): the new formula must equal 6ND within a few
   percent, using the trace-derived N_total. That validates the N_total extraction.
2. **LoRA** (experiment 7 configs): MFU lands at ≈15% / ≈20%, not 0.5%.
3. **Gradient checkpointing on vs off:** MFU must not rise when checkpointing slows training, while
   HFU does. That demonstrates the model-vs-hardware distinction.

Suggested branch: `mfu-frozen-weights`, cut from `main` after the BYO-script branch merges.

---

## 14. Compare the same examples without timing a whole pass

**Status:** a proposal to change PR #14's clean-window policy. **Not adopted**: PR #14's rule
(the clean window covers whole passes over the sample, starting at a pass boundary;
`loop/contracts.md`, Measurement axiom corollary 4) stays in force until this is settled. Raised
2026-09-27 while auditing the bring-your-own-script path (`integration_plan.md`, Appendix A, B2).

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

**What would settle it:** in the sandbox, on one model with `padding="longest"`:
1. Short-window vs whole-pass ranking over several non-sampler knobs: does the order of configs
   agree?
2. Length-grouped vs ungrouped under (a): log the example IDs in the window for both, and check
   they're the same set.
3. The between-launch noise of a short window vs a whole pass (§11).

Adopt the short window if rankings agree and noise is within the floor.

**Interaction:** OG and user-script probes should share one clean-window rule, or their throughput
isn't comparable (`integration_plan.md` Appendix A, B5).

---

# Initial experiments — where do we currently stand?

The goal is **not** to tune anything. It's to find out which of the claims above survive contact
with a real card. Run them in order; each one gates the next.

Setup: `backend = "modal"`, `gpu = "A10G"`, `Qwen/Qwen2.5-0.5B`, `wikitext-2-raw-v1`,
`effective_batch = 16`, `max_len = 2048`. Log to `runs/<name>/ceiling.tsv`, never `results.tsv`.

| # | Experiment | Answers | Settles |
|---|---|---|---|
| **E0** | Run `find_ceiling` once on a real A10G with `HFRunner` | Does it run at all? Do `peak_gb` / `oom_at` / `frag_gb` populate? | **8** (blocking) |
| **E1** | Same run — fit `peak_gb` vs `micro_batch` | Is the memory curve linear? How far is the predicted ceiling from the measured one? | **2** |
| **E2** | Ceiling on synthetic vs ceiling on real rows, same config | Do they agree? (They should, under `padding="max_length"`) | **1** |
| **E3** | E2 again with pad-to-longest instead of `padding="max_length"` | Does the real ceiling become unstable once padding is dynamic? | **1** |
| **E4** | Ceiling with `gradient_checkpointing` on vs off | How far does one memory lever move it? Does the staleness rule earn its cost? | **3** |
| **E5** | Sweep (model, max_len) — 0.5B/1.5B × 1024/2048/4096 | How often does the ceiling land *below* `effective_batch`? | **3** |

**E0 is the gate.** Nothing else is interpretable until the search has seen a real OOM.

**E2/E3 are the ones that decide whether this feature was worth building.** If E2 shows agreement
and E3 shows no instability, the honest conclusion is that the ceiling is a determinism-and-cost
win rather than a safety one, and `frame/README.md` should stop claiming otherwise.

These are dev experiments, run by hand to size up the current state — not a tuning session, so
they don't follow `program.md`'s loop. Delete an entry above once its experiment has settled it.
