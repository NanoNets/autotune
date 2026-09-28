# Integration plan: sandbox → live

> **Developer notes. Not part of the agent's prompt hierarchy.** The plan for moving the sandbox
> workflow out of `sandbox/attach_parity/` and into the live code. Written 2026-09-25.
> **Starts after experiment 7 (LoRA) is complete** (Phase 0). Background and evidence:
> `bring_your_own_script.md`. Open design items: `design_improvements.md` §10–§12.

---

## Terms

Each workflow is two things: something that **trains** and something that **measures**.

| | What trains | What measures |
|---|---|---|
| **OG** | **OG runners**: `frameworks/hf.py`, `frameworks/torch.py`. They build model + data, apply `knobs()`, make synthetic worst-case batches, and survive an OOM in-process | **OG measurement**: `profiler/capture.py: capture(runner, config)` calls `runner.step()` and times it. Phases come from the `record_function` labels inside the OG runners |
| **sandbox** | the user's own `train.py`, started by the **sandbox launcher** (`sandbox/attach_parity/launcher.py`) | **sandbox measurement**: `attach.py` (one public optimizer hook + timing + one profiled step), `derive.py` (phases, tokens, micro-batch count read from torch's own trace), `nopatch.py` (guard) |

## The target

1. **Sandbox measurement is the only measurement.** OG measurement is retired.
2. **OG runners stay, as training loops, measured by sandbox measurement.** `capture(runner,
   config)` keeps its name and signature, but becomes a thin driver: install sandbox measurement,
   call `runner.step()` until it's done, catch OOM in-process.
3. **OG runners drop their `record_function` labels** *(decided 2026-09-25)*. Every probe's
   phases, whether from an OG runner or a user script, come from sandbox measurement's trace
   reading. One meaning per field.
4. **User scripts run through the sandbox launcher**, promoted into `profiler/`.

```
OG runner:    capture(runner)    → OG runner step()      ─┐
                                                           ├→ sandbox measurement → diagnose() → Observation
user script:  profiler.launch    → user's train.py        ─┘
```

**Not changing:** `profiler/diagnostic.py: diagnose()`, `profiler/trace.py`, `profiler/roofline.py`,
the Observation contract (fields may be *added* to `provenance`), and `frame/ceiling.py: search()`.

---

## Decisions already made

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

---

## Phase 0: gate (before any code moves)

**Must be true:**
- ~~Experiment 7 (LoRA, SmolLM2-360M + Qwen2.5-0.5B, 5 rounds) is complete and written up.~~
  **Done 2026-09-25** (`bring_your_own_script.md`, experiment 7). Measurement is complete under
  PEFT; MFU reads 29–38× low. D2 is confirmed necessary.
- The decisions below are made. Each changes Phase 3.

| # | Decision needed | Options | Why it matters |
|---|---|---|---|
| D1 | **Tokens under variable batch width, for user scripts** | (a) record input shapes over the whole clean window (profiler overhead lands in timing); (b) count tokens on a few extra profiled steps and average; (c) rank user scripts on samples/s and report tokens as "profiled-step only" | Sandbox measurement counts tokens from ONE profiled step. OG measurement uses the mean over all steps, because under `padding="longest"` one step gave 7490 vs 9449 tok/s for identical configs (`frameworks/hf.py:324-340`). OG runners are covered by declared tokens (Phase 3); user scripts are not |
| D2 | **Parameter count under LoRA** *(experiment 7: MFU 0.53% vs ~15.3% corrected on SmolLM2, 0.51% vs ~19.7% on Qwen2.5)* | **Deferred (2026-09-27): a separate branch after this work,** `design_improvements.md` §13. Keep MFU as *model* FLOPs and generalize 6ND to (4·N_total + 2·N_trainable)·D, rather than counting trace FLOPs, which would make it HFU | Until §13 lands, MFU under LoRA is a documented known limitation of the BYO-script path |
| D3 | **CUDA-event timing** | adopt in Phase 3, or defer | Removes the per-step sync bias (experiment 6), but shifts every step time vs the `knowledge.md` baselines, which then need re-baselining |
| D4 | **Gradient checkpointing and `torch.compile`** | blockers for Phase 8 deletion, or documented limitations | Both untested under sandbox measurement. Compile may fuse away `aten::embedding` (tokens); checkpointing recomputes forwards inside backward (phase split). Both are knobs the loop turns on OG runners |
| D5 | **Data-wait with no DataLoader** | report `null` (unknown), or keep 0.0 | Only `frameworks/torch.py`'s in-memory fixture hits it; `diagnose()` defaults to 0.0 today |
| D6 | **`sandbox/attach_parity/vendor/` after promotion** | keep as the frozen oracle, or delete once the parity report is committed | It's the baseline every sandbox experiment was compared against |

---

## Phase 1: split OG measurement's shared parts (no behaviour change)

**Why first:** both measurements already use the same helpers. Sandbox measurement imports them from
`vendor/capture.py`. Splitting them out gives both one home before anything is swapped.

- `profiler/capture.py` → keep the `Capture` dataclass and `capture()` here.
- New `profiler/ceilings.py`: `measure_peak_flops`, `measure_peak_bandwidth_gb_s`, `device_facts`.
- New `profiler/window.py`: `_sync`, `_read_memory`, `_tree_cpu_seconds`, the CPU sampler, `_cpu_stats`.
  Also dedupe the copy of the CPU sampler in `sandbox/attach_parity/attach.py`.

**Guard:** every test in `tests/` passes unchanged. **Exit:** `git diff` touches only imports and
moved code.

## Phase 2: promote sandbox measurement into `profiler/`

- `profiler/attach.py`, `profiler/derive.py` (from the sandbox), importing the live
  `profiler.capture` / `window` / `ceilings` instead of `vendor/`.
- `profiler/launch.py`, the sandbox launcher, runnable as `python -m profiler.launch --out cap.json -- train.py …`.
- `tests/support/nopatch.py`, the guard as test support.
- Port the 28 sandbox tests into `tests/`: `test_derive.py`, `test_trace_format.py`,
  `test_attach.py`, `test_nopatch.py`, `test_launcher.py`.
- The sandbox keeps its `vendor/` copy and harnesses, so parity against the frozen baseline stays
  runnable (see D6).

**Guard:** the ported tests pass. `sandbox/attach_parity/parity.py` experiment 2 config b still
passes on a GPU when pointed at the promoted modules. **Exit:** OG measurement is still what
`backends/*` call; nothing user-visible has changed.

## Phase 3: close sandbox measurement's gaps (so OG runners don't regress)

| Change | Where | Why |
|---|---|---|
| **Declared inputs override derived ones:** `Attach(params=…, tokens_fn=…)`. When an OG runner supplies its parameter count and `tokens_per_step()` (the mean over steps run), those win over the optimizer's params and the profiled-step token count | `profiler/attach.py` | keeps OG runners' exact semantics (`frameworks/hf.py:156`, `:324-340`) |
| **Derived spans never duplicate an existing name.** `derive.augment` skips a phase name already present in the trace, and records `phase_source: "derived"` or `"labelled"` | `profiler/derive.py` | a user script with its own `forward` label would otherwise be double-counted by `trace.phase_times_us` (`profiler/trace.py:28-42`). OG runners will have none after Phase 4 |
| **D1** (tokens for user scripts) | `profiler/attach.py`, `derive.py` | see Phase 0 |
| **D2** (params under LoRA) | `profiler/attach.py`, FLOP formula in `profiler/roofline.py` if (b) | see Phase 0 |
| **D5** (data-wait null) | `profiler/diagnostic.py` | see Phase 0 |
| **D3** if adopted: CUDA-event timing | `profiler/attach.py` | see Phase 0 |
| `provenance` gains `token_source`, `micro_batches` (forwards / backwards), `phase_source`, and any declared-vs-derived disagreement | `profiler/diagnostic.py`, `loop/contracts.md` | the agent can see how each number was obtained |

**Guard:** new CPU tests on hand-built traces with labels **and** derived spans (no double count),
declared tokens / params winning, and each D-option chosen. **Exit:** all of `tests/` passes.

## Phase 4: OG runners measured by sandbox measurement

- **`capture(runner, config)` becomes the driver:**
  1. measure or accept the ceilings;
  2. `runner.setup(config)`;
  3. install `Attach(params=runner.params, tokens_fn=runner.tokens_per_step)`;
  4. loop `runner.step()` until `att.done`;
  5. catch `torch.cuda.OutOfMemoryError` → `att.failed("oom", …)`, as `capture()` does today;
  6. return the Capture.

  Same signature, same statuses, same `backends/local.py:59` call site.
- **Remove the `record_function` labels from the OG runners:** `frameworks/torch.py:44-57`,
  `frameworks/hf.py:137-142` / `:262-277`. Keep them behind a temporary `labels=True` runner
  option, used **only** by the legacy arm below. It's deleted in Phase 8.
- Keep the old implementation as `capture_legacy()` until Phase 8.

**GPU parity** (A10; pins in `sandbox/attach_parity/hf/requirements-hf.txt`): `capture_legacy`
on the labelled OG runner vs the new `capture` on the unlabelled OG runner, same `HFRunner`
config, order shuffled every round, 5+ rounds, fitted for round + position (the experiment 5 / 6
method).

**Configs:**
- defaults (`padding="max_length"`);
- `padding="longest"` + `length_grouped` (exercises D1 / declared tokens);
- `gradient_checkpointing=True`;
- `torch_compile="default"`;
- `chunked_ce=4`.

**Pass criteria:**
- identical kernel sets;
- tokens and params equal (declared);
- derived GPU forward / backward within ±1% of the legacy labels;
- step time within the between-launch floor (§11) of legacy;
- ceiling equal on one model / card.

Anything failing under checkpointing or compile is resolved or documented per D4.

## Phase 5: one ceiling search, two kinds of trial

- `frame/ceiling.py: find_ceiling(trial, …)`, where `trial(n) -> Capture`. `search()` is unchanged.
  - **Runner trial:** in-process through the new `capture(runner, …)` on `data="synthetic"`, as
    today.
  - **Script trial:** one launch of `profiler.launch` per trial, with the probe config pointing
    the script's data setting at a synthetic dataset of rows that tokenize past `max_len` (§10).
- **Every trial checks its batch shape** from the trace (`Input Dims` of `aten::embedding`) and
  returns `error` unless it's `[micro_batch, max_len]`.
- **Stale ceiling:** re-measure starting from the previous ceiling (step up if it fits, down on
  `oom`) instead of from scratch.
- `_MEMORY_KNOBS` (`frame/ceiling.py:20-35`) can't see a user framework's flag names. For script
  trials, stamp the whole probe-config diff on the ceiling record, and treat any memory-relevant
  change the agent declares as stale.

**Guard:** fake-trial unit tests (a `trial` that OOMs above n). One GPU run where the runner-trial
and script-trial ceilings agree on the same model and settings (§10's experiment, arms (a) vs (b)).

## Phase 6: user scripts reachable from the loop

- `backends/local.py: run(config)`. If `config["script"]` is set, launch
  `python -m profiler.launch --out … -- <script> <args>` as a subprocess and read `cap.json`.
  Otherwise use the OG runner path. A missing `cap.json` → `status: "error"` (the process was killed).
- `backends/ssh.py` already runs one process per probe (`:111`); add the same script branch.
- Carry `backends/local.py`'s mean-alongside-median addition for both paths.
- `program.md` Step 2 / `loop/loop.md` Test: a user-script branch. The agent reads the script,
  writes the probe config (knobs + `max_steps`, logging / eval / save off), and launches.

**Guard:** an end-to-end CPU test (a tiny `train.py` through `backends.local.run`). One GPU probe of
`sandbox/attach_parity/hf/train_hf.py` through the backend matches its experiment 4b numbers.

## Phase 7: docs (one semantics)

- `profiler/README.md`: one capture engine, two drivers. The Observation table stays as the
  contract until §12 replaces it. `tests/test_diagnostic.py:179-189` checks that every
  `Observation` field appears in `profiler/README.md` with the stated count, so either keep the
  table or move that check to §12's index.
- `frameworks/README.md`: OG runners are reference training loops, no labels. Phases come from
  the trace.
- `frame/README.md`: the ceiling for user scripts (Phase 5).
- `loop/contracts.md`: the new `provenance` fields.
- **Then §12**, the agent-facing metric reference (`profiler/metrics/`), written once against the
  single semantics.

## Phase 8: validate end to end, then delete

**Before deleting anything:**
- Phase 4's GPU parity passes.
- A short full loop on the new path reproduces a prior run's numbers within the between-launch
  floor: baseline + ~5 experiments + one ceiling re-measure. Suggested target: the Qwen2.5-0.5B
  wikitext A10 run.

**Then delete:**
- `capture_legacy()`;
- the OG runners' `labels=True` option;
- `vendor/`, per D6.

**Keep:**
- the sandbox parity harnesses, as a nightly GPU check of the promoted code.

---

## Still out of scope after integration

These are known limitations, to be disclosed in §12:
- multi-GPU;
- fp16 GradScaler skipped steps (invisible at the torch level);
- scripts without a torch `DataLoader`;
- installed commands such as `swift sft` (not a `.py` file);
- killed processes (no `cap.json`; reported as `error`);
- non-text models (token counting reads `aten::embedding`);
- **MFU under LoRA / frozen weights** (reads 29–38× low) until `design_improvements.md` §13 lands
  on its own branch;
- anything D4 marks as a limitation.

## Risks

| Risk | Mitigation |
|---|---|
| torch renames a trace event `derive.py` reads | `tests/test_trace_format.py` in CI, run per supported torch version; a rename shows as a missing phase, not a wrong number |
| Dropping OG labels changes an OG runner's phase numbers relative to past runs in `knowledge.md` | Phase 4 parity quantifies it (±1% GPU in experiments 1–5); note it in `knowledge/README.md` when past beliefs cite host phase times |
| The per-step sync bias (~0.25%) differs between the two drivers | both drivers use the same `Attach` timing, so it cancels in rankings; D3 removes it |
| Script trials make the ceiling slow for large checkpoints | seed from the previous ceiling; `frame/memory.py`'s estimate as the first guess |
