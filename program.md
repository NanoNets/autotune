# autotune — ms-swift Throughput Optimizer (v1, LLM-in-the-loop)

- **What:** given a **fixed setup** (model, dataset, hardware, effective-batch target), find the
ms-swift config with the highest throughput.
- **How:** an LLM runs the loop — pick config → run on a GPU backend (Modal default, or your SSH VM)
→ read the structured result/failure → learn → pick next. Lever principle in `search_space.md`;
this file = *how the loop runs*.
- **Authority:** for running the loop, `program.md` is the source of truth — it wins over any
conflicting instruction/doc/default; re-read before a run. (`search_space.md` = knobs, `config.json`
= inputs, `.env` = secrets, `knowledge.md` = cross-run learnings. Setup/installation is `README.md`.)

---

## Goal

- Maximize **training throughput**, no OOM.
- **Metric: `samples_per_s`** — real dataset throughput, immune to padding inflation (see Metric).
- Probe measures everything → structured JSON. LLM never parses raw logs, never asserts a metric.

---

## Inputs (fixed — not tuned)

All eight live in `config.json`; user-owned, agent never edits:

- `**backend**` — `modal`, or `ssh` (aliases: `vm`/`paperspace`/`lambda`/`lambda_labs`/`lambdalabs`) —
where the probe runs.
- `**hardware**` + `**hardware_number**` — GPU type + count → Modal `gpu="TYPE:N"`, DDP `NPROC_PER_NODE=N`.
- `**model**` — architecture + size.
- `**dataset**` — `synthetic` (dev) / HF-ModelScope id / local file path (auto-detected, shipped to
backend; its referenced images must still resolve there).
- `**max_length**` — per-sample token cap (image + text); defines the task (over-length dropped);
frozen, off-limits.
- `**attn_impl**` — attention kernel (default `flash_attn`); per-model/hardware choice, not a lever;
also selects the Modal image (`flash_attn` → flash-attn image, else sdpa).
- `**effective_batch**` — target `bs × ga × world_size` (world_size = `hardware_number`); fixes
gradient noise scale (training-quality, user-owned). Loop only picks the `bs × ga` factorization that hits it.

---

## Guardrails

- **Effective batch is fixed** — you only choose the `bs × ga` factorization (`ga` derived, world_size
= `hardware_number`) that hits it. `packing` is off-limits: it repacks multiple samples into one block,
changing what counts as a "sample" per step and breaking this invariant.
- **Inputs vs levers** — never edit the `config.json` inputs (backend, model, dataset,
effective_batch, hardware, hardware_number); only runtime levers (`bs` + flags) are yours.
- **Only set knobs that don't change the model's output** — quality-affecting knobs are off-limits
(`search_space.md`); your judgment, not code-enforced.
- **Hard cap: 20 probes** (count `history` entries) so the loop can't run away.
- **Hard cap: 5 accel-lib probes** (Phase 0 Stage-B confirmation + categorical A/B) — separate from,
and additive to, the 20-probe tuning cap.
- **Every throughput number comes from a real probe** — never assert a metric.

---

## The loop (LLM-in-the-loop)

```
INPUTS  : model, dataset, hardware, effective_batch target (all fixed)
BUDGET  : 20 probes max (count history entries) — hard cap so the loop can't run away
PHASES  : PHASE 0 (once) → PRIORS (once) → ITER 1 (open) → LOOP (repeat until plateau) → PROMOTE → OUTPUT


PHASE 0 — CAPABILITY SCAN + ACCEL-LIB RESOLUTION (once, no training)
─────────────────────────────────────────────────────────────────────

  SCAN
    `python probe.py --scan` → records env facts to search_state.json `env` (torch/cuda versions,
    GPU name + compute capability, and which accel libs are importable — default list
    causal_conv1d/liger_kernel/flash_attn/fla; pass more via `--check-libs` once you've identified
    candidates below). The scan is code — it can only check importability of names you give it, never
    decide which libraries are worth checking.

  CANDIDATES (your judgment, not the scan's)
    Given `model` architecture + `env` (torch/cuda/compute capability), name any acceleration library
    that plausibly applies beyond the default list, with a JUSTIFICATION (what it accelerates, why
    this architecture benefits, why the torch/cuda/compute-capability combo supports it). No
    justification → not worth a Stage-A check. Also read knowledge.md: an INVARIANT entry that a
    library helps this architecture → skip straight to INSTALL below (install eagerly, skip
    A/B-testing the direction) rather than treating it as a fresh candidate.

  GATE (two stages — a scan only ever proves Stage A)
    STAGE A (cheap, pre-install) — does a build/wheel exist for this torch/cuda/compute-capability?
      No → reject, record why, don't attempt install.
    STAGE B (one probe, post-install) — does it actually get EXERCISED by this model, not just
      importable? (liger_kernel imported clean but had no Qwen3.5 kernels — Stage A can't catch
      this, only running it can.)

  INSTALL
    For anything passing Stage A — install → re-scan to confirm importable → resolve version/wheel/
    ABI errors the way `flash_image` already does (pin the wheel to `env.torch` / `env.torch_cuda` /
    compute capability). ADDITIVE by default — accel libs mostly coexist in one env; fork a separate
    environment (second Modal image / SSH venv) only for a genuine dependency conflict — never
    uninstall in place just to compare two libraries.

  CATEGORICAL CHOICES
    Libraries competing on the SAME problem (e.g. alternate attention / linear-attn kernels) are NOT
    independent adopt/reject candidates: install every Stage-A-compatible alternative (they coexist
    on disk) but ACTIVATE only one per probe — monkey-patching libraries (liger-style) conflict at
    runtime if two patch the same forward call at once. Resolve the choice HERE with a small A/B
    (PRIMARY signal: throughput + memory, same as any probe); the winner FREEZES for the session,
    same status as `attn_impl` — not re-opened as a lever in the main loop.

  BUDGET (this phase)
    Up to 5 probes (Stage-B confirmation + categorical A/B) — separate from, and additive to, the
    20-probe tuning cap (see Guardrails).

  DOCUMENT
    Every candidate considered (adopted or rejected) in search_state.json `accel_libs` (agent-owned,
    like `beliefs`): name, justification, decision, install_status, verified, note. Promotable to
    knowledge.md at PROMOTE time, same gate as any confirmed finding.

  CONSULT
    `env` + `accel_libs` before proposing any kernel/compile lever in the main loop: a missing or
    rejected lib means that fast path is unavailable (e.g. no flash_attn → can't select it).


PRIORS — once at start, before ITER 1
──────────────────────────────────────

  READ knowledge.md (the cross-run KB); filter each entry against THIS setup + env:
    invariant            → apply as a prior; don't waste a probe re-confirming the direction.
    contextual, matches  → strong prior; probe only if cheap or you doubt it.
    contextual, differs  → do NOT assume; carry it in as a HYPOTHESIS and test it (learned elsewhere).


ITER 1 — OPEN (choose the first probe from PRIORS + reasoning — don't waste it)
─────────────────────────────────────────────────────────────────────────────────

  COLD (no/weak matching priors)
    Open SAFE to characterize the regime — small bs (1 or 2), ga = eff/(bs×world_size), no
    extensions (flash_attn default).

  PRIORS MATCH this setup
    Skip re-confirming the known baseline — open with the most-informative SAFE probe, applying
    matching invariant/strong priors (e.g. a mandatory GC) so you don't seed a known OOM or burn a
    probe on what you already know.

  EITHER WAY
    Probe it → read throughput (samples/s), memory.peak_mem_gb; consult utilization for the regime
    (compute / memory / data-bound).

  RECORD
    The opener's `note` (what it showed) + your first `beliefs` (the regime).


LOOP — repeat until throughput plateaus or budget is hit
───────────────────────────────────────────────────────────

  1. READ search_state.json — the WHOLE `history` + `best` + your last `beliefs` (NOT just the last probe).

  2. REASON over the history before choosing (this is the judgment we're paying for):
       REVIEW    — did the last probe match what you predicted? what does the gap teach?
       INVARIANT — sanity-check the result against physics. If a number looks impossible (e.g. peak_mem
                   above the GPU's memory with no OOM, or samples/s that can't match the step times),
                   suspect the MEASUREMENT, not the GPU — stop and check before probing further.
       BELIEF    — where is the bottleneck now (compute / memory / data-bound) and what's the ceiling?
       HYPOTHESIS— the single most informative next config, the throughput you predict, and what
                   result would confirm vs refute it. You pass this to the probe (step 4) — it is
                   persisted at launch, BEFORE the outcome is known.

  3. PICK the single most informative next config — ONE change at a time; skip configs already tried.
     Read the signals for what they MEAN (physics); choosing WHICH lever to pull is your judgment
     (the full swift surface, per search_space.md — not prescribed here).

       PRIMARY (drives the decision) — throughput Δ + memory/OOM:
         status == oom       → at the MEMORY ceiling: the config needs less memory (per sample, or fewer).
         throughput flat     → SATURATED (plateau = last 3 lever attempts each <5% over best): try another regime, else stop.

       SECONDARY (diagnostic only — consult utilization AFTER throughput plateaus, to decide if
       dataloader tuning is even worth a probe; do NOT let a util scalar pick a lever before then).
       Empirical, not predicted — every branch is a measured delta; `cpu.p90` is reported context,
       never the decider:
         D0 (free, no probe) — read `idle_frac` off probes already run: <~0.05 → COMPUTE-BOUND, the
           GPU isn't waiting on anything → skip this phase entirely.
         D1 (one probe) — current best config + `--wait` at baseline dataloader settings → read
           `data_wait_frac` (fraction of step time blocked on next(batch); the authoritative signal,
           a direct measurement, not a proxy): ~0 → NOT DATA, idle is structure (GC/optimizer/sync/
           launch) — leave the dataloader alone. Meaningfully high → proceed to D2.
         D2 (one probe) — same config, `dataloader_num_workers` up to `nproc // 2` (your judgment on
           where in that range; on multi-GPU each rank spawns its own worker pool, so divide across
           `world_size` with care) + `--wait` → compare `data_wait_frac`/throughput AGAINST D1,
           directly:
           drops, throughput up   → keep it, done.
           drops partially       → optionally one more probe on `dataloader_prefetch_factor`.
           barely moves near cap → PREPROCESSING WALL, measured not guessed → stop, out of scope.

       Until throughput plateaus the GPU isn't waiting on data — don't tune the dataloader. Memory is
       often the real wall; don't over-trust one util number.

  4. PROBE it (respecting the fixed effective_batch). Pass your HYPOTHESIS with `--hypothesis` so it's
     persisted at launch. probe.py appends {config, hypothesis, result} to history.

  5. RECORD what you learned in search_state.json:
       - write this probe's `note` = the REVIEW: your hypothesis's prediction vs the actual result,
         and what the gap taught (the trail — append-only, never revised).
       - UPDATE `beliefs` — carry forward what still holds, revise what this probe changed
         (the cumulative head, not a per-probe overwrite).


PROMOTE — once at run end
───────────────────────────

  Curate knowledge.md (the gate that grows cross-run memory):
    - promote durable, CONTEXT-TAGGED, confirmed beliefs into knowledge.md (schema there).
    - contradicted an existing entry? amend it (add the caveat) or retire it (mark superseded).
    - gate it: only confirmed findings, never per-probe churn — that's what `beliefs` is for.


OUTPUT — `best` from search_state.json (highest samples/s)
```

```
LLM picks config → run probe on backend → probe returns JSON → LLM reads result/failure
       ↑________________________ learn, pick next ________________________|
```

---

## Probe mechanics

- **Probe mechanics win over any lever** (`probe_runner.py`): `logging_steps=1`,
`truncation_strategy=delete`, `include_num_input_tokens_seen=true`,
`dataloader_pin_memory/persistent_workers=true`. `lazy_tokenize=true` UNLESS `dataset_num_proc`
is set (needs eager preprocessing). `max_steps`/`warmup` per probe via `--max-steps`/`--warmup`
(default 7/2; raise for data levers — see Metric).
- **Invocation** — pass this probe's levers (any `swift sft` arg except off-limits; `search_space.md`
- ms-swift docs) on the command line. `config.json` is untouched; every probe's exact levers are
recorded in `search_state.json` history (exactly what ran).

```bash
python probe.py --scan                                                        # phase-0 capability scan (once)
python probe.py                                                               # seed probe (no levers, bs=1)
python probe.py --levers '{"per_device_train_batch_size": 2, "padding_free": true}' \
                --hypothesis "bs2 halves steps; predict ~+20% samples/s unless mem-bound (OOM refutes)"  # compute lever
python probe.py --levers '{"dataloader_num_workers": 8}' --max-steps 20 --warmup 5 --wait \
                --hypothesis "data_wait_frac was high; more workers should drop it & raise samples/s"      # data lever (--wait)
```

Probe parses its OWN output on the GPU and returns the structured result below; `probe.py` appends
it to `search_state.json` history. Agent reads the structured result, never the raw training log.

---

## Probe output (the contract the LLM reads)

```json
{
  "status": "ok | oom | error",
  "config": {                       // the exact levers that ran (ms-swift CLI args) — feeds search state
    "model": "...", "world_size": 1, "effective_batch": 8,
    "per_device_train_batch_size": 2, "gradient_accumulation_steps": 4, "attn_impl": "flash_attn",
    "max_length": 8192, "padding_free": true, "gradient_checkpointing": false,
    "vit_gradient_checkpointing": false, "dataloader_num_workers": 1
  },
  "throughput": {                   // PRIMARY — real dataset throughput, immune to padding inflation
    "samples_per_s": 0.42, "steps_measured": 4
  },
  "memory": { "peak_mem_gb": 70.0, "gpu_mem_gb": 79.6 },   // OOM margin + total GPU memory
  "utilization": {                  // SECONDARY/diagnostic — null when no measurable step window (OOM/crash/too-fast)
    "gpu": { "idle_frac": 0.05, "p10": 82, "p50": 96, "p90": 100, "max": 100 },  // idle_frac = share of window < 20% (GPU waiting)
    "cpu": { "p10": 20, "p50": 41, "p90": 68, "max": 90 },                       // p90 = preprocessing ceiling
    "window_steps": [3, 7], "n_samples": 18,          // util is windowed to the SAME steps as samples/s
    "data_wait_frac": 0.03            // ONLY with --wait: fraction of step time blocked on next(batch) — direct data signal
  },
  "error_detail": null              // on failure: the relevant message (OOM shape, traceback, …)
}
```

- `status`: `oom` when log contains "out of memory", else `error` on any non-zero exit.
- `utilization`: windowed to the measured steps (load/save phases excluded); **diagnostic only** —
consult after throughput plateaus, never to pick a lever before then.
  - `gpu.idle_frac` = how often the GPU sat waiting (any cause — not necessarily data); the free D0
  pre-check (loop step 3).
  - `data_wait_frac` (with `--wait`) = the authoritative data-starvation signal — a direct measurement of
  time blocked on next(batch), not a proxy. Drives D1/D2 (loop step 3).
  - `cpu.p90` = swift process-tree CPU ceiling (% of allocated cores) — reported context only; no longer
  picks a lever, `data_wait_frac` deltas do that.
- `error_detail`: the actual failure message (OOM + offending shape, recompile thrash, CUDA error) —
reason about the cause instead of guessing.

---

## Metric

- `**samples_per_s` is the metric** (steady-state) — real dataset throughput, immune to padding
inflation (padding still costs compute but never counts as a sample, unlike `tokens_per_s`).
- `samples_per_s = effective_batch × Δsteps / Δelapsed` — numerator is `bs × ga`, **not** `bs`:
ms-swift logs one entry per **optimizer step** (= `bs × ga` samples). Using `bs` alone divides true
throughput by `ga` — a measurement artifact, not physics.
- **Sanity check:** two runs of the same config should give ~the same samples/s; a wild swing →
suspect the metric, not the GPU.

**Probe length** (per probe, via `--max-steps`/`--warmup`; both windows skip the warmup):

- **Compute levers** (bs, GC, padding_free, …) stabilize fast → default `7/2`.
- **Data levers** (workers, prefetch, dataset_num_proc) need warm caches + spun-up workers → `20/5`
(a short probe under-counts them; our one 7-step `workers=8` probe read −20% mostly from unamortized spin-up).
- Higher throughput = better.

**OOM detection:** log contains "out of memory". Crash: non-zero exit → `status: error` + `error_detail`.

---

## Levers

When you hit a ceiling, these are the *physics* of the levers — what they do, not a sequence. Read  
the signals for what they MEAN; choosing WHICH lever to pull is your judgment (the full swift  
surface, per `search_space.md` — not prescribed here).

**Saturation / plateau** = the last 3 lever attempts each improved <5% over the current best →
ceiling reached, stop. This is the loop's "throughput flat" branch.

---

## Search state — the loop's memory (`search_state.json`)

- `probe.py` appends every probe to `search_state.json` (gitignored, local) — the durable record the
loop reasons over. One file per **session**: `setup` = the fixed inputs; a different setup starts a fresh session.
- **Setup change = fresh session:** the code detects the mismatch and **overwrites** the file on the
first probe (old history is lost) — archive `search_state.json` first if you want to keep the prior run.
- **Three memory tiers:** `history[].note` (per probe) → `beliefs` (per session) → `**knowledge.md`**
(across runs — the curated KB, read at start, promoted to at end; see PRIORS/PROMOTE).

**Who writes what:**

- **Code owns `history` + `best` + `env`** — auto-written (`env` once by `--scan`, history per probe), can't be forgotten.
- **Agent owns each entry's `note`, plus `beliefs`, plus `accel_libs`** — it edits these; code preserves them (only appends).
- `**hypothesis` write-once at launch** — passed via `--hypothesis`, persisted with the probe before
the result is known; not revised (`note` holds the review).

**The trail — `hypothesis` (at launch) → `note` (after) → `beliefs` (whole session):**

- `history[].hypothesis` = the **prediction** — why this lever, predicted throughput, what confirms/refutes. Before the outcome.
- `history[].note` = the **REVIEW** — prediction vs actual + what the gap taught (append-only, never revised). Pair makes each probe self-documenting.
- `beliefs` = the **head** — single cumulative *current* understanding across all probes, revised each iteration. Read it to decide the next probe.
- `env` = the **capability scan** (top-level, NOT in `setup` so it can't break session matching). Consult before kernel/compile levers.
- `accel_libs` = the **acceleration-library trail** (top-level, agent-owned, NOT in `setup`) — every
candidate considered in Phase 0, adopted or rejected, with its justification + install/verify status.
Written during Phase 0; promoted to knowledge.md at PROMOTE time.

```json
{
  "setup": { "backend": "modal", "model": "...", "dataset": "...", "max_length": 8192,
             "effective_batch": 8, "hardware": "A10G", "hardware_number": 1 },
  "env": { "torch": "2.6.0", "torch_cuda": "12.4", "driver_version": "550.x",
           "gpu_name": "NVIDIA H100", "compute_capability": "9.0", "gpu_count": 1,
           "libs": { "causal_conv1d": false, "liger_kernel": true, "flash_attn": true, "fla": true } },
  "accel_libs": [
    { "name": "causal_conv1d",
      "justification": "model uses linear-attn; without it the fast path falls back to slow PyTorch",
      "decision": "adopt", "install_status": "installed",
      "verified": "re-scan: importable; load log shows fast path selected (no 'fallback' warning)",
      "note": "Stage A passed (wheel exists for torch 2.6.0/cu12.4); no Stage-B probe needed — no
               competing implementation for this op, so not a categorical choice" },
    { "name": "liger_kernel",
      "justification": "knowledge.md flags no kernels for this arch — kept for re-check only",
      "decision": "reject", "install_status": "already installed", "verified": "no-op confirmed prior run",
      "note": "importable but no fused kernel fires for this architecture (Stage-B trap)" }
  ],
  "history": [
    { "iter": 1,
      "config": { "...": "the exact levers that ran (identical to the probe-output config)" },
      "hypothesis": "padding_free at bs=2: predict ~+200% (long uniform samples, little padding to lose); OOM or <seed refutes",
      "result": { "status": "ok",
                  "throughput": { "samples_per_s": 0.59, "steps_measured": 4 },
                  "memory": { "peak_mem_gb": 12.5, "gpu_mem_gb": 24.0 },
                  "utilization": { "gpu": {"idle_frac": 0.18, "p10": 35, "p50": 80, "p90": 99, "max": 100},
                                   "cpu": {"p10": 30, "p50": 55, "p90": 70, "max": 96},
                                   "window_steps": [3, 7], "n_samples": 18 },
                  "error_detail": null },
      "note": "padding_free at bs=2 → +200% vs seed; idle_frac 0.18 so not yet compute-bound" }
  ],
  "best": { "config": { "...": "..." }, "samples_per_s": 0.59 },
  "beliefs": "cumulative current understanding: regime (compute/mem/data-bound), ceilings, invariants held/violated"
}
```

- `history` is **lean** — no raw logs (they stream live); `result` keeps `error_detail` for failure
diagnosis. It's the source of truth the agent reads directly — the WHOLE history before each probe,
`beliefs` written after.

---

## Final output

When the search converges (plateau or 20-probe cap), print the best config (read `best` from
search_state.json — code already tracks it; the table is printed to the user, not saved to a file):

- `best` is ranked by **samples_per_s** — real dataset throughput, no padding caveat needed.

```markdown
# Optimal training config

| Model | Eff batch | bs | ga | workers | flags        | samples/s |
|-------|-----------|----|----|---------|--------------|-----------|
| 3B    | 16        | 2  | 8  | 1       | padding_free | 0.59      |
```

This is the direct input to the production training run.