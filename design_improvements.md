# Design improvements

> **Developer notes. Not part of the agent's prompt hierarchy.**
> `program.md` does not route here and the loop never reads or writes this file. It's a scratchpad
> for us to decide what to build next — the doctrine the agent follows lives in `program.md`,
> `loop/`, and the per-directory READMEs.

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
