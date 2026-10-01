# Integration plan: sandbox → live

> **Developer notes. Not part of the agent's prompt hierarchy.** The plan for moving the sandbox
> workflow out of `sandbox/attach_parity/` and into the live code.
> **Revision 2, 2026-09-27: rebased onto PR #14** ("align every metric's inputs, and make data
> parity an axiom of the loop", merged 2026-09-24). Revision 1 (2026-09-25) predated PR #14 and
> is in git history (`git show 0faddb7:integration_plan.md`).
> Background and evidence: `bring_your_own_script.md`. Open design items: `design_improvements.md`
> §10–§13.

>
> **`sandbox/attach_parity/` is a local, untracked harness** (gitignored). The experiments below ran
> there; paths into it describe where the code and raw results live on the machine that ran them.
---

## Terms

Each workflow is two things: something that **trains** and something that **measures**.

| | What trains | What measures |
|---|---|---|
| **OG** | **OG runners**: `frameworks/hf.py`, `frameworks/torch.py`. They build model + data, apply `knobs()`, make synthetic worst-case batches, survive an OOM in-process, and (since PR #14) offer the optional hooks `steps_per_pass` / `start_pass()` / `last_step_tokens()` / `reference_step()` / `population()` | **OG measurement**: `profiler/capture.py: capture(runner, config)`. It rounds the clean window up to whole passes over Frame's drawn sample, counts tokens every clean step, and replays a fixed **reference batch** twice (profiler off, then traced) so trace fractions divide by the same batch's wall. Phases come from `record_function` labels in the OG runners |
| **sandbox** | the user's own `train.py`, started by the **sandbox launcher** (`sandbox/attach_parity/launcher.py`) | **sandbox measurement**: `attach.py` (one public optimizer hook + timing + one profiled step), `derive.py` (phases, tokens, micro-batch count read from torch's own trace), `nopatch.py` (guard) |

**The Measurement axiom** (`loop/contracts.md`, from PR #14) governs everything below: *a number
may only be compared with, or divided by, another number measured on the same population of work
under the same instrument.* Instruments: the **clean window** (profiler off) and the **trace**
(profiler on). Observations record their population (`sample_hash`, `batch_key`). Impossible
readings (a fraction > 1 + 2%) are `invalid` and never scored.

---

## What changed with PR #14, and why this plan changed

The sandbox experiments (1–7) were measured against **pre-PR #14 OG** (`vendor/` = `bdc33d6`).
Against PR #14's axiom, sandbox measurement breaks it in three places:

| Axiom requirement | OG since PR #14 | sandbox measurement today |
|---|---|---|
| **Rates from totals over the same steps** (corollary 3) | tokens counted every clean step (`last_step_tokens()`); MFU and tokens/s from clean-window means | tokens from **one** profiled step ÷ clean-window time ❌ |
| **Whole passes over identical rows** (corollary 4) | window rounded to whole passes over Frame's sample, started at a pass boundary (`start_pass()`) | window = 10 steps after 3 warmup steps, wherever the loader happens to be ❌ |
| **Trace fractions over the same batch's wall** (corollary 2) | reference batch replayed: profiler off (`reference_step_s`), then traced | traced step's busy ÷ the clean median of **other** steps ❌ |
| **Population recorded** (corollary 5) | `sample_hash`, `batch_key` | none ❌ |

So the plan gains a phase **before** integration (Phase S): realign sandbox measurement to the
axiom inside the sandbox, and re-prove parity against PR #14's OG, not the old one.

**The target changes too:** one measurement engine, **two trace protocols.**
- **OG runners** can replay a batch, so they keep PR #14's reference-batch protocol.
- **User scripts** can't be made to replay a batch, so they trace one step and divide by that
  step's own wall.
- The Observation records which protocol each trace number rests on.
- Step boundaries, phase derivation, token counting and the clean-window rules are shared.

---

## The target

1. **Sandbox measurement is the only step-boundary / phase / token engine.** It replaces OG's
   `record_function` labels *(decided 2026-09-25: OG runners mirror sandbox measurement's trace
   reading)*. Phases always come from `derive.py`.
2. **One clean-window protocol for both:** whole passes over Frame's drawn sample, starting at a
   pass boundary, with tokens counted over exactly the timed steps. That's corollaries 3–4, for
   OG runners and user scripts alike.
3. **Two trace protocols, recorded:**
   - `reference-replay`: OG runners with `reference_step()`. PR #14's protocol, unchanged.
   - `traced-step`: user scripts. One step, traced, with fractions over that step's own
     profiler-on wall.
4. **`capture(runner, config)` keeps its name, signature and PR #14 structure** (whole passes,
   reference replay, OOM recovery). It drops the `record_function` dependency and takes phases,
   tokens and the step boundary from sandbox measurement.
5. **User scripts run through the launcher**, promoted to `python -m profiler.launch`.

```
OG runner:    capture(runner)  → whole passes → reference replay (off, then traced) ─┐
                                                                                       ├→ derive.py → diagnose() → Observation
user script:  profiler.launch  → whole passes → one traced step (own wall)          ─┘            (population + trace basis recorded)
```

**Not changing:** `profiler/diagnostic.py: diagnose()`'s pairing rules and `invalid` gate,
`profiler/trace.py`, `profiler/roofline.py` (until `design_improvements.md` §13), the
Measurement axiom, and `frame/ceiling.py: search()`.

---

## Decisions

### Already made

| Decision | Date | Where recorded |
|---|---|---|
| No monkeypatching; public torch APIs only; enforced at runtime | 2026-09-25 | `profiler_metric_coverage.md` §0 |
| A torch `DataLoader` is assumed; scripts without one are future work | 2026-09-25 | `bring_your_own_script.md`, experiment 5 |
| Single GPU for now | 2026-09-24 | `bring_your_own_script.md`, experiment 3 |
| Probe length is set by `max_steps` in the probe config; the launcher never stops a script | 2026-09-24 | same |
| The agent handles framework-specific knob names; no name-mapping layer | 2026-09-24 | `bring_your_own_script.md` §1 |
| Ceiling for user scripts: synthetic-dataset config route, one launch per trial, re-measure from the previous ceiling, batch-shape check | 2026-09-25 | `design_improvements.md` §10 |
| Launcher reports `ok` / `oom` / `error` in `capture()`'s statuses | 2026-09-25 | `sandbox/attach_parity/launcher.py` |
| The ~0.25% step-time bias is acceptable if disclosed | 2026-09-25 | `bring_your_own_script.md`, experiment 6 |
| **OG runners mirror sandbox measurement's trace reading (no labels)** | 2026-09-25 | this plan |
| Agent-facing metric reference (§12) is written after integration, against the one semantics | 2026-09-25 | `design_improvements.md` §12 |
| **MFU under LoRA / frozen weights (D2) is deferred** to a separate branch after this work (`mfu-frozen-weights`) | 2026-09-27 | `design_improvements.md` §13 |
| **B1 / D1: tokens are counted on every clean step**, padded `input_ids.numel()` (the same definition as OG), via a per-module `register_forward_pre_hook(with_kwargs=True)` on the user's model. The model is found by a one-shot global pre-hook during warmup, removed before any timed step. The traced step's trace-derived count is kept only as a cross-check. Reason: tokens/s and MFU must divide the timed steps' own tokens by their time (axiom corollaries 2–3); one traced step is off by (its width ÷ the mean width), silently and per config. **Implemented in the sandbox** (`attach.py`), with guard tests in `tests/test_attach.py` (`test_b1_*`) | 2026-09-27 | this plan, Appendix A B1 |

### Open: needed before Phase S / Phase 3

| # | Decision | Options | Recommendation |
|---|---|---|---|
| ~~D1~~ | ~~Tokens for user scripts~~ | **Decided 2026-09-27**, see "Already made" (B1) | per-clean-step hook |
| D3 | CUDA-event timing | adopt in Phase S, or defer | **Adopt in Phase S.** One re-baselining, and experiment 6's design validates it |
| D4 | Gradient checkpointing / `torch.compile` | blockers for Phase 8 deletion, or documented limitations | **Must pass before Phase 8 deletes anything; not needed to start** |
| D5 | Data-wait with no DataLoader | `null`, or keep 0.0 | **`null`.** Note that PR #14's `diagnose()` still reads `ph_s.get("dataloader", 0.0)` |
| D6 | `sandbox/attach_parity/vendor/` | keep, or delete | **Re-freeze it from `2c4c135` in Phase S** (the new baseline); delete in Phase 8 |
| **D7** *(new)* | **Trace basis for user scripts** (can't replay a batch) | (a) fractions over the traced step's **own profiler-on wall**: same batch, same instrument, reads slightly low from profiler overhead, never impossible; (b) no GPU-active / gap for user scripts at all (`null`) | **(a)**, recorded as `trace_basis: "traced-step"`. `batch_key` = a hash of the traced step's input shapes (weaker than OG's row identity), so cross-probe trace durations compare only at matching shapes |
| **D8** *(new)* | **Whole passes for user scripts** | the probe config points the script's data setting at Frame's `sample.jsonl`; a pass = one epoch; sandbox measurement counts optimizer steps and aligns the clean window to epoch boundaries (e.g. warm up for exactly one pass) | Needs the sample size to be a multiple of samples per step (Frame already rounds, e.g. 1000 → 992). **Check:** frameworks reshuffle per epoch (HF Trainer seeds by epoch). That's fine for throughput (the axiom allows re-batching the same rows), but it means no two passes share a traced batch. **The audit adds:** `sample.jsonl` isn't framework-readable (a meta header line, `{"row","text","tokens"}` rows), so Frame must also write the drawn rows in the user dataset's own schema (Appendix A, S3); and the rows actually trained must be cross-checked against the file (S5) |

---

## Milestone P (plumbing first): done 2026-09-27

**Decided 2026-09-27:** build the plumbing that lets a user's own script run through the repo
**before** any measurement-policy work (Phase S, B2–B4, S3–S7, §14), so the work can extend beyond
the sandbox. **PR #14's policy is unchanged.** Script probes record that they don't meet it
(`population.driver: "script"`, `clean_window: "fixed-position"`, `trace_basis: "traced-step"`),
and `program.md` treats them as provisional, non-promotable evidence.

| Piece | Where |
|---|---|
| Measurement engine, promoted from the sandbox onto the live `profiler.capture` helpers and PR #14's `Capture` (incl. `step_tokens`, B1) | `profiler/attach.py`, `profiler/derive.py`, `profiler/nopatch.py` |
| Entry point | `python -m profiler.launch --out cap.json -- <script> <args>` (`profiler/launch.py`) |
| Loop reachability | `backends/local.py: run_script` (ssh and modal reach it via `backends.local.run` on the box); `backends/ssh.py` checks the script is repo-relative and pushes one kept under `runs/` |
| User workflow | `README.md`, "Bring your own training script"; `run.toml.example` (`script`, `script_args`) |
| Agent flow | `program.md`, "Script mode"; `backends/README.md`, "Script probes"; `profiler/README.md`, "Attached capture" |
| Tests | `tests/test_{derive,trace_format,attach,nopatch,launch}.py` (ported) + `tests/test_backends_script.py` (end to end through `backends.run`) |

After this milestone, the phases below re-sequence as "policy and data-side": Phase S first, then
Phases 3–8. Phases 1–2 are partly done by the promotion (the helpers are imported from
`profiler/capture.py` rather than split into `ceilings.py` / `window.py`; that split is still
optional cleanup).

## Phase S: realign sandbox measurement to the axiom (in the sandbox, before any live change)

**Why first:** everything proven so far was proven against pre-PR #14 OG. The engine being
promoted must satisfy PR #14's rules, and be proven against PR #14's OG, before it touches
`profiler/`.

- **Re-freeze the baseline:** `vendor/` ← `profiler/` + `frameworks/` + `frame/synthetic.py` at
  `2c4c135` (D6). Keep the old copy only in git history.
- **Clean window:** whole passes over the probe sample (D8), with tokens counted over exactly the
  timed steps (D1).
- **Trace:** `trace_basis` recorded (D7). Fractions over the traced step's own wall for scripts.
- **Population:** `sample_hash` (from the sample file) and `batch_key` (shape hash) on every sandbox
  Capture.
- **Timing:** CUDA events if D3 is adopted.
- **Launcher / derive:** unchanged, except derive must skip phase names already present (a user
  script may carry its own `forward` label; `trace.phase_times_us` sums by name).
- **Tests:** extend the 28 sandbox tests (whole-pass alignment on a CPU fixture; tokens-per-step
  totals; `trace_basis`; population).

**GPU parity (A10, pins in `hf/requirements-hf.txt`), against PR #14's OG:**
- the dynamic-padding config (experiment 2 config b's shape with `padding="longest"`);
- experiment 4b (Trainer, `from_pretrained`);
- experiment 7 (LoRA);
- order shuffled every round, fitted for round + position (the experiment 5 / 6 method).

**Pass criteria:**
- identical kernel sets;
- tokens per step equal to OG's under D1's chosen definition;
- step time and tokens/s within the between-launch floor (§11);
- no `invalid` readings;
- trace fractions ≤ 1.

**Exit:** a short write-up in `bring_your_own_script.md` (experiment 8) with the numbers.

## Phase 1: split OG measurement's shared parts (no behaviour change)

- `profiler/capture.py` (353 lines since PR #14) → keep `Capture` + `capture()`.
- New `profiler/ceilings.py`: `measure_peak_flops`, `measure_peak_bandwidth_gb_s`, `device_facts`.
- New `profiler/window.py`: `_sync`, `_read_memory`, `_tree_cpu_seconds`, the CPU sampler, `_cpu_stats`.

**Guard:** all of `tests/` passes unchanged. That's 149 tests: 148 pass, and one fails on `main`
itself (see "Known on main" below). **Exit:** only imports and moved code change.

## Phase 2: promote sandbox measurement into `profiler/`

- `profiler/attach.py`, `profiler/derive.py`, and `profiler/launch.py` (`python -m profiler.launch`).
- `tests/support/nopatch.py`.
- Port the sandbox tests into `tests/`.

**Guard:** the ported tests pass, and Phase S's parity still holds with the promoted modules.
**Exit:** OG measurement is still what `backends/*` call; nothing user-visible has changed.

## Phase 3: close the remaining gaps

| Change | Where |
|---|---|
| Declared inputs win: OG runners' `last_step_tokens()` and total params override the derived values | `profiler/attach.py` |
| `provenance` / Observation gains `trace_basis`, `token_source`, `micro_batches`, `phase_source`. Keep `tests/test_diagnostic.py::test_readme_documents_every_observation_field` passing: the README table is 49 fields and must grow with it | `profiler/diagnostic.py`, `profiler/README.md`, `loop/contracts.md` |
| D5 (data-wait `null`) | `profiler/diagnostic.py` |

## Phase 4: OG runners measured through the engine

- `capture(runner, config)` keeps PR #14's structure (whole passes, `start_pass()`, reference
  replay, OOM recovery). Its step boundary, token count and phases now come from the engine.
- **Remove the `record_function` labels** from `frameworks/torch.py:44-57` and
  `frameworks/hf.py` (the `_rf` wrapper, and `reference_step()`'s labels too). Keep them behind a
  temporary `labels=True` option, used only by the legacy arm.
- Keep the old implementation as `capture_legacy()` until Phase 8.

**GPU parity:** `capture_legacy` on the labelled OG runner vs the new `capture` on the unlabelled
one, same `HFRunner` config, shuffled order, 5+ rounds.

**Configs:**
- defaults;
- `padding="longest"` + `length_grouped`;
- `gradient_checkpointing=True`;
- `torch_compile="default"`;
- `chunked_ce=4`.

**Pass criteria:**
- identical kernel sets;
- identical `population` (same `sample_hash`, same `batch_key`);
- tokens and params equal;
- derived GPU forward / backward within ±1% of the legacy labels;
- step time within the between-launch floor;
- the same ceiling on one model / card;
- no new `invalid`.

## Phase 5: one ceiling search, two kinds of trial

As in revision 1 (`design_improvements.md` §10):
- `find_ceiling(trial)`, with runner trials (in-process) and script trials (one launch each,
  synthetic-dataset config route);
- the `Input Dims` batch-shape check on every trial;
- re-measure from the previous ceiling;
- stamp the full probe-config diff on the record for script trials (`_MEMORY_KNOBS` can't see
  user flags).

**Also reconcile with PR #14's out-of-scope note:** a follow-up PR is planned to tie the ceiling to
the full config it was measured under, including env vars such as `PYTORCH_CUDA_ALLOC_CONF`. Do
this phase with, or after, that PR, not in parallel with it.

**Guard:** fake-trial unit tests, plus one GPU run where the runner-trial and script-trial ceilings
agree.

## Phase 6: user scripts reachable from the loop

- `backends/local.py` and `backends/ssh.py` (both changed in PR #14): a `script` branch that runs
  `python -m profiler.launch` and reads `cap.json`. A missing file → `error`.
- `program.md` Step 2 already has the agent draw Frame's sample (`--sample-out`) and pass it as
  `sample_file` to every probe. For user scripts, the agent **also points the script's own data
  setting at that file** (D8), and writes the probe config (knobs + `max_steps` = warmup pass +
  whole clean passes + 1, with logging / eval / save off).

**Guard:** an end-to-end CPU test (a tiny `train.py` through `backends.local.run`), and one GPU
probe of `sandbox/attach_parity/hf/train_hf.py` through the backend.

## Phase 7: docs (one semantics)

- `profiler/README.md`: one engine, two drivers, two trace protocols.
- `frameworks/README.md`: OG runners are reference loops, with no labels.
- `frame/README.md`: the ceiling for scripts.
- `loop/contracts.md`: `trace_basis`, and the new population fields for scripts.
- **Then §12**, the agent-facing metric reference. It must state, per field, which instrument and
  population it comes from, and how that differs between the `reference-replay` and
  `traced-step` bases.

## Phase 8: validate end to end, then delete

**Before deleting anything:**
- Phase 4's GPU parity passes.
- A short full loop on the new path reproduces a PR #14-era run within the between-launch
  floor: baseline + ~5 experiments + one ceiling re-measure. The runs recorded in
  `docs/verification/pr14.md` are the natural reference.

**Then delete:**
- `capture_legacy()`;
- the `labels=True` option;
- `vendor/` (D6).

**Keep:**
- the sandbox parity harnesses, as a nightly GPU check.

---

## Known on `main` (not caused by this branch)

- `tests/test_capture.py::test_cpu_signal_survives_a_window_too_short_to_sample` fails **every
  time** on this Mac (5/5 on the rebased branch, and 3/3 on a clean checkout of `2c4c135`):
  `cap.cpu["mean"]` reads `0.0` where the test expects `None`. PR #14 lists it as "fails
  intermittently on `main` too".

## Still out of scope after integration

These are known limitations, to be disclosed in §12:
- multi-GPU;
- fp16 GradScaler skipped steps;
- scripts without a torch `DataLoader`;
- installed commands such as `swift sft`;
- killed processes (no `cap.json` → `error`);
- non-text models;
- **MFU under LoRA / frozen weights**, until §13;
- anything D4 marks as a limitation.

User scripts additionally get the weaker `traced-step` trace basis (D7): their trace fractions
read slightly low, and their `batch_key` is shape-based.

## Risks

| Risk | Mitigation |
|---|---|
| torch renames a trace event `derive.py` reads | `tests/test_trace_format.py` in CI, run per supported torch version |
| Two trace protocols confuse the agent | `trace_basis` on every Observation; §12 documents both; cross-probe trace durations compare only within one basis and a matching `batch_key` (axiom corollary 5) |
| Whole-pass alignment fails when the user's framework doesn't make a pass = an epoch (e.g. `max_steps`-driven infinite samplers, drop_last differences) | Phase S test: count optimizer steps per epoch on the real Trainer path before trusting the alignment; fall back to recording `clean_passes` as fractional, so throughput isn't compared across probes |
| Dropping OG labels changes OG runners' phase numbers vs past runs | Phase 4 parity quantifies it; note in `knowledge/README.md` where beliefs cite host phase times |
| The per-step sync bias (~0.25%) | shared by both drivers, so it cancels in rankings; D3 removes it |

---

## Appendix A: Measurement-axiom compatibility audit of sandbox measurement (2026-09-27)

Every rule of the axiom (`loop/contracts.md`, corollaries 1–7), and every PR #14 mechanism the
Observation depends on, checked against sandbox measurement's code (`sandbox/attach_parity/`:
`attach.py`, `derive.py`, `launcher.py`). Line refs are to the rebased tree.

**Severity:**
- **breaks:** produces a number the axiom forbids, or loses one silently.
- **design:** needs a decision or new mechanism for user scripts.
- **minor:** small drift from OG.
- **ok:** already compatible.

### A.1 Breaks: must be fixed in Phase S

| # | Rule | What sandbox measurement does | Why it breaks | Fix | Verify with |
|---|---|---|---|---|---|
| B1 ✅ *decided + implemented in the sandbox 2026-09-27; GPU re-verification in Phase S* | **Rates from totals over the timed steps** (corollaries 2–3) | tokens from ONE traced step (`derive.augment` → `Attach.tokens`), passed to `diagnose()` as the caller's `tokens` | PR #14's `diagnose()` uses the caller's `tokens` whenever `capture.step_tokens` is empty (`profiler/diagnostic.py:153`), so tokens/s and MFU silently become one step's tokens ÷ the clean window's mean time. **The `invalid` gate won't catch it** unless MFU exceeds 1 + 2% | Fill `Capture.step_tokens` with per-clean-step counts (D1: option (b), a per-module pre-hook on the model counting padded `input_ids.numel()`), or pass `tokens=None` so the numbers read as unknown, not wrong | CPU test: dynamic-width fixture, where sandbox `tokens_per_step` equals the per-step mean; GPU: `padding="longest"` vs OG `last_step_tokens()` |
| B2 *(PR #14 policy kept as-is, 2026-09-27; the lighter "same examples, short window" alternative is `design_improvements.md` §14, not adopted)* | **Whole passes, starting at a pass boundary** (corollary 4) | clean window = 10 steps after 3 warmup steps, wherever the loader is | which rows get timed depends on warmup length and loader position; `clean_passes` is `None`, so no two probes are `same-rows` | D8: count optimizer steps per pass; warm up to a pass boundary; round the window to whole passes; fill `Capture.clean_passes` / `clean_steps_requested` | CPU test on a fixture whose steps-per-pass is known; GPU: Trainer path, steps per epoch counted |
| B3 | **Record the population** (corollaries 5, 7) | `Capture.population` unset → `{}` | throughput and trace durations can't be checked for comparability; beliefs can't be promoted; `program.md`'s "never time a probe on the head fallback" can't be checked (no `sample_method`) | populate `sample_file`, `sample_method`, `sample_hash`, `steps_per_pass`, and a `batch_key` for the traced step (D7: a shape hash) | unit test on the populated Capture; Observation carries `population` |
| B4 | **Trace fractions over the same batch's wall** (corollary 2) | no `reference_step_s` | PR #14's `diagnose()` computes `gpu_active_frac`, `gap_s` and `gap_frac` only from `reference_step_s`, so for sandbox Captures they come back **null**. They were *mixed* under the pre-PR #14 baseline; now they're *lost* | D7: a separate traced-step basis (busy ÷ the traced step's own wall, `trace_basis: "traced-step"`), **not** by writing the profiler-on wall into `reference_step_s`, which PR #14 defines as profiler-OFF | unit test: `trace_basis` set; fraction ≤ 1 |
| B5 | **Same instrument** (the axiom's premise, corollary 5) | sync inside the optimizer post-hook (experiment 6: reads ~0.25% high vs a `TrainerCallback`, ~half from sync placement) | OG-driven and sandbox-driven probes time steps with different sync placement, i.e. different instruments. Comparing their throughput is outside the axiom | record the driver / timing method in `population` (e.g. `timing: "post-hook-sync"` or `"cuda-events"`), and compare throughput only within one; D3 (CUDA events) shrinks the gap | parity: the same OG runner under both drivers, shuffled order (Phase 4) |

### A.2 Design: user scripts can't meet a rule the way OG does

| # | Rule | The obstacle | Proposed handling |
|---|---|---|---|
| S1 | **Reference batch fixed by rule** (corollary 4) | a user's script can't be made to replay a chosen batch | D7: the `traced-step` basis. **Consequence for corollary 7:** script probes have no `same-reference-batch` evidence, so beliefs about trace **durations** (busy, phase and kernel times) can't be promoted from them. Only `width-invariant` counts and `same-rows` throughput can. §12 must say so |
| S2 | **`invalid` as the population-error alarm** (corollary 6) | under the traced-step basis, busy ÷ its own profiler-on wall is ≤ 1 by construction, so the GPU-active alarm can never fire. A population error would go unnoticed on that field | Keep the MFU > 1 alarm (it still applies). Add a cross-instrument sanity check: the traced step's profiler-on wall vs the clean-window mean. A ratio far from PR #14's recorded profiler overhead flags the traced step as unrepresentative |
| S3 | **The probe sample as the population** (PR #14: Frame draws it; every probe loads it) | **Frame's sample file isn't a training dataset.** Its first line is a `{"meta": …}` header, rows are `{"row", "text", "tokens"}` (`frame/dataset.py:126-175`), and it's drawn from a single `text_field`. A user's framework (HF `datasets` JSON loader, swift, TRL) would ingest the header as a row, and expects its own schema (`messages`, `instruction` / `output`, …) | Frame writes a **framework-readable sample** alongside `sample.jsonl`: the drawn rows in the user dataset's **own schema**, with no header. `meta` stays in the existing file. Needs Frame to sample from the user's dataset in its native columns |
| S4 | **Token lengths behind the sample** (stratification; D1 option (a)) | `tokens` is counted by the raw tokenizer on `text` with truncation at `max_len`. User frameworks apply **chat templates / prompt formats** and may mask or drop rows, so the lengths they train on differ from Frame's | Stratify on lengths **as the framework tokenizes them** (the agent can compute them with the script's own template), or accept a weaker stratification. **Rules out D1 option (a)** (tokens known in advance from the file) unless the counts come from the framework's own preprocessing |
| S5 | **`sample_hash` must name the rows actually trained** | frameworks may drop rows during preprocessing (over-length deletion, empty-after-template, filtering), so the rows hashed from the file ≠ the rows trained | cross-check samples-per-pass observed (micro-batches × batch rows, from `derive`) against rows in the file. A mismatch → record it, and don't claim `same-rows` |
| S6 | **A pass = an epoch** (D8) | HF Trainer: optimizer steps per epoch = ⌈batches / grad_accum⌉. A partial last step if not divisible; `dataloader_drop_last` defaults to False, so the last micro-batch is smaller; `DataLoaderShard` fetches one batch ahead | require sample size to be a multiple of micro_batch × grad_accum (Frame's `whole_steps()` already rounds to `effective_batch`, `frame/dataset.py:120`); verify steps per epoch on the real Trainer path before trusting alignment. Otherwise record `clean_passes` as fractional and don't compare throughput |
| S7 | **Warmup to a pass boundary** | aligning the window to an epoch start means warming up for one full pass (PR #14 already proposes this for Qwen + `fla`) | accept the cost: one extra pass per probe. Size the sample so a pass is short (PR #14's example: 992 rows = 62 steps at `effective_batch=16`) |

### A.3 Minor

| # | Item | Fix |
|---|---|---|
| M1 | memory is read after the profiled step (`attach.py: _stop_profiled`); PR #14 reads it at the end of the clean window, before the reference (`profiler/capture.py`) | read memory when the clean window closes |
| M2 | `clean_steps_requested` isn't recorded | set it (B2) |
| M3 | data-wait with no DataLoader reads 0.0 (`diagnostic.py`: `ph_s.get("dataloader", 0.0)`) | D5: null (the DataLoader is assumed anyway) |

### A.4 Already compatible

- **Host fractions:** PR #14 pairs data-wait as annotation ÷ `profiled_step_s` of the same traced
  step (`diagnostic.py`, `wait = frac(data_wait, profiled)`). `derive`'s `dataloader` span sits in
  the traced step, and `Attach` sets `profiled_step_s` for that step: same step, same instrument.
- **Width-invariant evidence** (kernel count, gap count, provenance, dtypes): valid across any
  batch (corollary 5), so it's promotable from script probes.
- **Clean window with the profiler off, CPU sampled over the same window, peak memory reset after
  warmup:** the same instrument as OG, once B2 aligns the window.
- **The MFU > 1 `invalid` alarm:** applies unchanged.
- **Step boundary via the optimizer:** the axiom is indifferent to how steps are delimited, as long
  as the window and the token count cover the same steps (B1, B2).
- **OG runners under `reference-replay`:** `derive`'s phases come from the traced reference batch,
  so trace durations keep `same-reference-batch` (Phase 4 must confirm `batch_key` is unchanged).

### A.5 Verification: redo the experiments against PR #14

All sandbox evidence (experiments 1–7) predates the axiom and must be re-run after Phase S. Order:
1. **CPU unit tests** for B1–B4, S5, S6 and M1 (new), plus the 28 existing ones.
2. **GPU parity vs PR #14 OG** on an OG runner through both drivers (B5), using Frame's drawn sample,
   `padding="longest"`, 5+ shuffled rounds. That covers B1 / B2 / B5.
3. **User-script path**: the Trainer script (experiment 4b) and LoRA (experiment 7) with a
   framework-readable sample (S3), steps per epoch counted (S6), and the rows-trained cross-check
   (S5).
4. **The sync / CUDA-event question** (D3), with experiment 6's design, on the realigned engine.
