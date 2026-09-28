# Profiler metric coverage for a user's script

> **Developer notes. Not part of the agent's prompt hierarchy.** Companion to
> `bring_your_own_script.md`. Status as of 2026-09-25, against `sandbox/attach_parity/attach.py`.

**The question:** the original profiler (`profiler/`) produces a 41-field **Observation** from a
runner we wrote. When the script is the user's own, run through the launcher, how is each of
those fields obtained, and at what cost?

- The reference is **what the original profiler provides**, not what the loop might want.
- The two runner inputs the Observation depends on (parameter count, tokens per step) are
  included, because several fields are computed from them.

---

## 0. Update 2026-09-25: the patch-free rewrite

`sandbox/attach_parity/attach.py` was rewritten to use **no monkeypatches and no module hooks**.
Sections 1–6 below describe the **patched** version (commit `51c8c4e`) and are kept for the
record.

| Old mechanism | Bucket | Replaced by | Bucket now |
|---|---|---|---|
| M11 one-shot global module hook (find the model) | 🟠 | not needed: tokens come from the trace | removed |
| M10 per-module pre-hook (tokens, micro-batches) | 🟢 | `record_shapes=True` → `Input Dims` of the first `aten::embedding` (else first linear) per forward. The **padded** count, same as `frameworks/hf.py:264` | 🟢 |
| M12 global module hooks (`forward` label) | 🟠 | derived from the trace: main-thread work from the first non-copy op after the batch arrives up to that micro-batch's backward (`derive.py`) | 🟢 |
| M13 patched `torch.autograd.backward` (`backward` label) | 🔴 | derived from the trace: clusters of torch's own `autograd::engine::evaluate_function` events | 🟢 |
| M14 patched `_BaseDataLoaderIter.__next__` (`dataloader` name) | 🔴 | torch's own `enumerate(DataLoader)#…` span, re-emitted under the name `dataloader` in the exported trace | 🟢 |
| M15 rewrapped optimizer class `step` (profiler window) | 🔴 | the public post-hook opens and closes the window directly, inside torch's `Optimizer.step` label. On CPU, that label for the profiled step is still present in the trace (whether its duration is complete is not yet checked) | 🟢 |

**Tally now:** of the 38 measured fields, **38 🟢, 0 🟠, 0 🔴**. The derived spans are added to our
own exported trace, marked `args.derived = "attach.derive"`, so the frozen `diagnose()` reads
them unchanged.

**Enforced:** `nopatch.py` snapshots the torch functions the old version patched, plus the
global module-hook tables. Every C / L run fails if anything was replaced or registered.

**Verified so far: CPU only** (the A10 was down). On the experiment 1–3 workloads:
- identical CPU-op sets vs the labelled baseline;
- 4 / 4 micro-batches recovered, tokens exact through both the embedding and first-linear paths;
- nothing empty or wrongly zero, **except** data-wait on a runner with no DataLoader (our
  in-memory smoke-test fixture). There, the frozen `diagnose()` reports 0.0 where the truth is
  "unknown". The patched version had the same gap.

**Scope decision (2026-09-25): a torch `DataLoader` is assumed.** It's the standard in every
framework we target. Scripts without one are a future improvement, not a gap in the supported
case.

**GPU-verified 2026-09-25** (A10, experiments 1–4b rerun; `bring_your_own_script.md`,
experiment 5):
- all 38 fields populated and correct;
- trace-derived GPU forward/backward within about ±1% of the labels on the raw-loop workloads;
- grad accum, tokens and params exact, including through HF Trainer and both `from_pretrained`
  checkpoints.

**Timing:** an order-shuffled A/A/B control puts the patch-free version at about +1.5 ms/step
(≈0.12%) vs the patched one on long HF steps. The first estimate of ~4 ms was inflated by a
fixed run order and GPU warm-up. The effect is far below the between-launch spread, and its
cause is parked (`bring_your_own_script.md`, experiment 5).

---

## 1. The four buckets

| Bucket | Meaning |
|---|---|
| 🟢 **Existing PyTorch tooling** | Public APIs used as designed. Nothing in torch is replaced, and nothing slows the model down. Includes `torch.profiler` / CUPTI, `torch.cuda` memory stats, torch's **own** built-in labels, the public global optimizer hook, and plain Python (psutil, exceptions). |
| 🟠 **Public API, but invasive** | A supported registration API, but one torch documents as "only intended for debugging/profiling purposes". Registering it switches off `Module.__call__`'s no-hook fast path for **every** module, and it is expected to interfere with `torch.compile` (not tested). No torch code is replaced. |
| 🔴 **Monkeypatched** | We assign our own function over a torch function or method. It is undone after use, but while active it affects the whole process, can collide with other tools patching the same thing, and can break across torch releases. |
| ⚫ **Can't be measured** | Not obtainable from a user's script with this approach, whatever tooling is used. |

---

## 2. The mechanisms, and which bucket each is in

Everything the Observation needs from a user's script comes from these. Line numbers are in
`sandbox/attach_parity/attach.py`.

| # | Mechanism | Provides | Bucket | Where |
|---|---|---|---|---|
| M1 | `torch.profiler.profile` (CUPTI) around one step | the Chrome trace: every kernel, memcpy, CPU op, runtime call, occupancy args, device props | 🟢 | `:232` |
| M2 | torch's **built-in** labels, emitted with no help from us | `Optimizer.step#<Cls>.step`, `Optimizer.zero_grad#…`, `enumerate(DataLoader)#<Iter>.__next__`, backward ops as `autograd::engine::evaluate_function: …` | 🟢 | torch itself |
| M3 | `register_optimizer_step_post_hook` (global, public) | "an optimizer step just finished", plus the optimizer object (its `param_groups`) | 🟢 | `:157` |
| M4 | `torch.cuda.synchronize` + `time.perf_counter` at each boundary | clean per-step wall time | 🟢 | `_on_step` |
| M5 | `torch.cuda` memory API (`reset_peak_memory_stats`, `max_memory_allocated`, `mem_get_info`) | peak / reserved / total / headroom | 🟢 | via `vendor/capture.py` helpers |
| M6 | psutil sampler thread + process-tree CPU counters | CPU util, cores busy | 🟢 | `_start_cpu` / `_stop_cpu` |
| M7 | our own `record_function("step")` around the profiled step | the enclosing `step` label (and so phase coverage) | 🟢 | `_start_profiled` |
| M8 | GEMM and device-copy micro-benchmarks, run before the script | peak FLOP/s, peak bandwidth | 🟢 | `vendor/capture.py` |
| M9 | Python exception from `runpy.run_path` | the script's failure cause | 🟢 | `launcher.py` (`finally`) |
| M10 | per-module `register_forward_pre_hook(..., with_kwargs=True)` on the top-level model | input tensors → **tokens**, and the **micro-batch count** | 🟢 (one module only; does not touch the fast path of the others) | `:91` |
| M11 | global `register_module_forward_pre_hook`, one-shot during warmup | which module is the top-level model (for M10) | 🟠 (removed after its first call, before any timed step) | `:158` |
| M12 | global `register_module_forward_pre_hook` / `register_module_forward_hook`, profiled step only | the **`forward`** label | 🟠 | `:88-89` |
| M13 | replace `torch.autograd.backward`, profiled step only | the **`backward`** label | 🔴 | `:99` (undone `:126`) |
| M14 | replace `_BaseDataLoaderIter.__next__` (a **private** class), profiled step only | a label literally named **`dataloader`** | 🔴 | `:109` (undone `:129`) |
| M15 | replace the first optimizer's class `step`, and set `step.hooked = True` (torch's internal marker) | a step boundary that fires **after** torch's `Optimizer.step` label closes, so starting or stopping the profiler doesn't cut that label in half | 🔴 | `:200` (undone `:173`) |

M13–M15 are the only monkeypatches. For each one, the information it produces is either
already in the trace or obtainable some other way. **Those alternatives are untested:**

| Patch | What it adds | Non-invasive alternative (untested) |
|---|---|---|
| M13 `backward` | a `backward` label | the trace already marks backward work (M2: `autograd::engine::evaluate_function: …`, run on autograd's own thread) |
| M14 `dataloader` | the **name** `dataloader` only | the same span is already labelled by torch (M2). `diagnose()` could accept `enumerate(DataLoader)#…` as an alias: a one-line change in the diagnostic, no patch |
| M15 step rewrap | profiler start/stop outside torch's `Optimizer.step` label | the public post-hook (M3) plus `torch.profiler.schedule` and `prof.step()`. Whether that cuts the label is not yet known |

---

## 3. Field by field

**Columns:**
- **Original profiler:** where `profiler/` gets the field today, from our runner.
- **User's script:** the mechanism(s) from §2.
- **Bucket:** the most invasive bucket among the mechanisms the field needs, as implemented today.
- **Verified:** the experiment that compared it against the original (§5 of `bring_your_own_script.md`).

| # | Field | Original profiler | User's script | Bucket | Verified |
|---|---|---|---|---|---|
| 1 | `status` | `capture()` catches `OutOfMemoryError` → `oom`, any other exception → `error` | M9. `ok` works. **`oom` is not implemented yet:** an OOM currently reports as `error` (catching it in the launcher is plain Python) | 🟢 (OOM: not built) | `ok` path, exp. 3–4b |
| 2 | `gpu` | CLI label | CLI label | n/a (input) | — |
| 3 | `peak_flops` | GEMM micro-benchmark | M8 | 🟢 | frozen, identical |
| 4 | `peak_dtype` | the run's dtype | M8, dtype passed to the launcher | 🟢 | — |
| 5 | `peak_mem_bw_gb_s` | copy micro-benchmark | M8 | 🟢 | frozen, identical |
| 6 | `step_time_s` | `capture()` times `runner.step()` | M3 + M4 (M15 today) | 🟢 possible · 🔴 today | exp. 1–4b; ~2 ms/step overhead in 4b |
| 7 | `step_time_cv` | same | same | 🟢 possible · 🔴 today | exp. 1–4b |
| 8 | `flops` | 6 × `runner.params` × `runner.tokens_per_step()` | params: M3's `param_groups`. tokens: M10 (needs M11) | 🟠 | exact, exp. 1–4b |
| 9 | `roofline_floor_s` | from 3 + 8 | same | 🟠 | derived |
| 10 | `mfu` | from 3 + 6 + 8 | same | 🟠 (+ 🔴 via 6 today) | exp. 1–4b |
| 11 | `gpu_busy_s` | trace kernels | M1 | 🟢 | exp. 1–4b |
| 12 | `gpu_active_frac` | 11 / 6 | same | 🟢 (+ 🔴 via 6 today) | derived |
| 13 | `profiled_step_s` | wall of the one profiled step | M1 + M7, window placed by M15 | 🟢 possible (untested) · 🔴 today | exp. 1–4b |
| 14 | `phase_times_s` | our runner's `record_function` labels | `step` M7 🟢, `Optimizer.step#…` M2 🟢, `enumerate(DataLoader)#…` M2 🟢, **`forward` M12 🟠, `backward` M13 🔴, `dataloader` M14 🔴** | 🔴 | exp. 1–4b (host times differ by loop: Trainer syncs every micro-batch) |
| 15 | `gpu_phase_times_s` | trace, kernels linked to labels by correlation id | same labels as 14 | 🔴 | exp. 1–4b |
| 16 | `gpu_phase_coverage` | from 15 | from 15 (the `step` label, M7, covers everything) | 🟢 | exp. 1–4b |
| 17 | `data_wait_s` | our runner's `dataloader` label | M14 today. **Alias to M2 possible** | 🔴 today · 🟢 possible | exp. 2–4b |
| 18 | `data_wait_frac` | 17 / 6 | same | 🔴 today · 🟢 possible | exp. 2–4b |
| 19 | `gap_s` | 6 − 11 | same | 🟢 (+ 🔴 via 6 today) | derived |
| 20 | `gap_frac` | 19 / 6 | same | same | derived |
| 21 | `cpu_util_mean` | psutil over the clean window | M6, window from M3 | 🟢 | populated, not value-compared |
| 22 | `cpu_util_max` | same | same | 🟢 | same |
| 23 | `nproc` | `os.cpu_count()` | same | 🟢 | same |
| 24 | `cpu_cores_busy` | process-tree CPU counters | M6 | 🟢 | exp. 1–3 |
| 25 | `peak_mem_gb` | `torch.cuda` API, reset after warmup | M5, reset from M3 | 🟢 | populated, not value-compared |
| 26 | `reserved_mem_gb` | same | same | 🟢 | same |
| 27 | `mem_total_gb` | same | same | 🟢 | same |
| 28 | `headroom_gb` | same | same | 🟢 | same |
| 29 | `provenance` | kernel names in the trace | M1 | 🟢 | kernel set identical, exp. 1–3 |
| 30 | `top_kernels` | trace | M1 | 🟢 | same |
| 31 | `kernel_count` | trace | M1 | 🟢 | same |
| 32 | `top8_coverage` | trace | M1 | 🟢 | same |
| 33 | `gap_count` | trace kernel timeline | M1 | 🟢 | populated, not value-compared |
| 34 | `max_gap_s` | same | M1 | 🟢 | same |
| 35 | `achieved_occupancy_pct` | trace kernel args | M1 | 🟢 | same |
| 36 | `limiting_factor` | same | M1 | 🟢 | same |
| 37 | `device` | trace header | M1 | 🟢 | same |
| 38 | `device_type` | config | launcher `--device` | n/a (input) | — |
| 39 | `trace_path` | our file | our file | 🟢 | — |
| 40 | `error` | caught exception, first line | M9 | 🟢 | — |
| 41 | `summary` | derived | derived | follows its inputs | — |

**Runner inputs** (not Observation fields, but 8–10 depend on them):

| Input | Original profiler | User's script | Bucket |
|---|---|---|---|
| `params` (N) | `runner.params` | M3: `sum(numel)` over the optimizer's `param_groups` (trainable only) | 🟢 |
| `tokens_per_step` (D) | `runner.tokens_per_step()` | M10 (on the model found by M11): `attention_mask.sum()`, else `input_ids.numel()` | 🟠 |

---

## 4. Tally, as implemented today

The 41 fields, excluding the three inputs (`gpu`, `device_type`, `summary`), leaves 38:

| Bucket | Fields | Count |
|---|---|---|
| 🟢 no invasive mechanism at all | 1, 3, 4, 5, 11, 16, 21–37, 39, 40 | **25** |
| 🟠 needs the global module hook (via tokens) | 8, 9 | **2** |
| 🔴 needs a monkeypatch today | 6, 7, 10, 12, 13, 14, 15, 17, 18, 19, 20 | **11** |
| ⚫ can't be measured | none of the 41 in the supported case (see §5) | **0** |

The 🔴 row overstates the real dependence. Seven of those eleven (6, 7, 10, 12, 13, 19, 20) are
🔴 only through the step-boundary rewrap (M15), and step **timing** itself needs only the public
post-hook (M3). Two (17, 18) are 🔴 only because of the **name** `dataloader`. **Only the
`forward` / `backward` split in 14–15 needs information the public tools don't hand over
directly.** Whether the trace structure alone can recover it is the open test.

---

## 5. What can't be measured

These are out of reach from a user's script with torch-level tooling. Some are reachable through
a framework's own extension points (HF `TrainerCallback`), which are not torch-level.

| What | Why | Effect on the Observation | Reachable another way? |
|---|---|---|---|
| **A killed process** (OOM killer, SIGKILL, `os._exit`) | the launcher's `finally` never runs | no `cap.json` at all; the loop must treat a missing file as `error` | no |
| **Data-wait from a non-torch loader** (custom iterator, DALI, nanoGPT-style `get_batch()`). *Out of scope: a torch `DataLoader` is assumed (2026-09-25); future improvement* | nothing torch-level runs while it fetches, so there's no label (M2 or M14) | `data_wait_s` = 0.0, which reads as "not data-bound". **Misleading, not null** | only by patching that loader |
| **Skipped fp16 steps** (GradScaler overflow) | the skipped `optimizer.step()` never runs, so no boundary fires, and two steps merge into one window | `step_time_s` inflated on overflow steps | yes: Trainer's `optimizer_step_was_skipped` |
| **Which steps were eval / checkpoint** | torch sees only more work | an eval or save step inside the clean window shows up only as a timing outlier | yes: Trainer knows; a probe config that disables eval/save avoids it |
| **The configured grad accum** (as opposed to the observed one) | torch has no notion of it | the observed count (M10) is available; the configured value lives in the user's YAML | yes: the YAML, or Trainer's `args` |
| **Phase split under `torch.compile`** | untested; M12 is expected to cause graph breaks or recompiles | `forward` / `backward` may be missing or distort the run | unknown |

### Obtainable but currently wrong (a definition problem, not a tooling gap)

| What | Problem | Fix |
|---|---|---|
| `flops` / `mfu` under **LoRA / PEFT** | N comes from the optimizer, which holds trainable params only, so 6ND counts a sliver of the model | the top-level model's `parameters()` (public) gives total N; the formula needs total and trainable separately |
| `phase_times_s` / `gpu_phase_times_s` under **gradient checkpointing** | recomputed forwards run inside backward and would be labelled `forward` by M12 | untested |
| `status = "oom"` | not implemented in the launcher yet | catch `torch.cuda.OutOfMemoryError` from `runpy`, read memory stats (🟢) |

---

## 6. One-paragraph summary

Of the original profiler's 38 measured fields, 25 come entirely from public PyTorch tooling:
every kernel, memory, CPU, occupancy and device field, plus the ceilings. Two more need a public
but invasive global module hook, because of token counting. Eleven depend on a monkeypatch **as
implemented today**, but only one piece of information behind them, the `forward`/`backward`
phase split, lacks a known public source. The rest are patched either to place the profiler
window (step rewrap) or to rename a label torch already emits (`dataloader`). Nothing in the
supported case (single GPU, torch `DataLoader`, bf16/fp32, no compile) is unmeasurable. Outside
it, a killed process, a non-torch loader, skipped fp16 steps and eval-step placement can't be
seen at the torch level.
