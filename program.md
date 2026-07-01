# autotune — ms-swift Throughput Optimizer (v1, LLM-in-the-loop)

- **What:** given a **fixed setup** (model, dataset, hardware, effective-batch target), find the
  ms-swift config with the highest throughput.
- **How:** an LLM runs the loop — pick config → run on a GPU backend (Modal default, or your SSH VM)
  → read the structured result/failure → learn → pick next. Lever principle in `search_space.md`;
  this file = *how the loop runs*.
- **Authority:** for running the loop, `program.md` is the source of truth — it wins over any
  conflicting instruction/doc/default; re-read before a run. (`search_space.md` = knobs, `config.json`
  = inputs, `.env` = secrets, `knowledge.md` = cross-run learnings.)

---

## Goal

- Maximize **training throughput**, no OOM.
- Two objectives, weigh **both**: `tokens_per_s` + `samples_per_s` — they diverge under
  padding/variable-length data (see Metric).
- Probe measures everything → structured JSON. LLM never parses raw logs, never asserts a metric.

---

## Inputs (fixed — not tuned)

All eight live in `config.json`; user-owned, agent never edits:

- **`backend`** — `modal` or `ssh` (where the probe runs).
- **`hardware`** + **`hardware_number`** — GPU type + count → Modal `gpu="TYPE:N"`, DDP `NPROC_PER_NODE=N`.
- **`model`** — architecture + size.
- **`dataset`** — `synthetic` (dev) / HF-ModelScope id / local file path (auto-detected, shipped to
  backend; its referenced images must still resolve there).
- **`max_length`** — per-sample token cap (image + text); defines the task (over-length dropped);
  frozen, off-limits.
- **`attn_impl`** — attention kernel (default `flash_attn`); per-model/hardware choice, not a lever;
  also selects the Modal image (`flash_attn` → flash-attn image, else sdpa).
- **`effective_batch`** — target `bs × ga × world_size` (world_size = `hardware_number`); fixes
  gradient noise scale (training-quality, user-owned). Loop only picks the `bs × ga` factorization that hits it.

---

## The loop (LLM-in-the-loop)

```
INPUTS (fixed): model, dataset, hardware, effective_batch target
budget        : 20 probes max (count history entries) — hard cap so the loop can't run away

PHASE 0 — CAPABILITY SCAN (once, no training):
  run `python probe.py --scan` → records env facts to search_state.json `env` (torch/cuda versions,
  GPU name + compute capability, and which accel libs are importable: causal_conv1d, liger_kernel,
  flash_attn, fla). CONSULT it before proposing any kernel/compile lever: a missing lib means that
  fast path is unavailable (e.g. causal_conv1d absent → linear-attn runs the slow fallback; no
  flash_attn → can't select it). LIMIT: "importable" ≠ "has a kernel for THIS model" (liger imported
  but had no Qwen3.5 kernels) — the scan kills missing-lib dead-ends, not present-but-no-kernel ones
  (those still need one probe to rule out).

PRIORS — READ knowledge.md (the cross-run KB) once at start, filter each entry against THIS setup + env:
  invariant            → apply as a prior; don't waste a probe re-confirming the direction.
  contextual, matches  → strong prior; probe only if cheap or you doubt it.
  contextual, differs  → do NOT assume — carry it in as a HYPOTHESIS and test it (it was learned elsewhere).

ITER 1 — OPEN (you choose the first probe from PRIORS + reasoning — don't waste it):
  COLD (no/weak matching priors): open SAFE to characterize the regime — small bs (1 or 2),
    ga = eff/(bs×world_size), no extensions (flash_attn default).
  PRIORS MATCH this setup: skip re-confirming the known baseline — open with the most-informative
    SAFE probe, applying matching invariant/strong priors (e.g. a mandatory GC) so you don't seed a
    known OOM or burn a probe on what you already know.
  EITHER WAY: probe it → read throughput (tokens/s + samples/s), memory.peak_mem_gb; consult
    utilization for the regime (compute / memory / data-bound).
  RECORD: the opener's `note` (what it showed) + your first `beliefs` (the regime).

LOOP until throughput plateaus or budget is hit:
  1. READ search_state.json — the WHOLE `history` + `best` + your last `beliefs` (NOT just the last probe).
  2. REASON over the history before choosing (this is the judgment we're paying for):
       REVIEW    — did the last probe match what you predicted? what does the gap teach?
       INVARIANT — sanity-check the result against physics. If a number looks impossible (e.g. peak_mem
                   above the GPU's memory with no OOM, or tokens/s that can't match the step times),
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
     SECONDARY (diagnostic only — consult utilization AFTER throughput plateaus, to decide if dataloader
     tuning is even worth a probe; do NOT let a util scalar pick a lever before then):
     CPU tells you whether the idle is even data-related, THEN whether workers help (idle high = idle_frac >~0.15):
       gpu idle_frac <~0.05          → COMPUTE-BOUND: the GPU isn't waiting — the dataloader cannot help.
       idle high + cpu p90 ~idle     → NOT DATA: idle is structure (GC/optimizer/sync) or launch gaps — the
         (<~30%)                         workers aren't even busy, so it's not starvation. Leave the dataloader alone.
       idle high + cpu p90 active    → DATA-BOUND, under-parallelized: CPU is feeding but has room → workers/prefetch may help.
         (~30-70%)
       idle high + cpu p90 pinned    → PREPROCESSING WALL: CPU maxed feeding the GPU → more workers won't help; stop.
         (~100%)
     Until throughput plateaus the GPU isn't waiting on data — don't tune the dataloader. Memory is
     often the real wall; don't over-trust one util number.
  4. PROBE it (respecting the fixed effective_batch). Pass your HYPOTHESIS with `--hypothesis` so it's
     persisted at launch. probe.py appends {config, hypothesis, result} to history.
  5. RECORD what you learned in search_state.json:
       - write this probe's `note` = the REVIEW: your hypothesis's prediction vs the actual result,
         and what the gap taught (the trail — append-only, never revised).
       - UPDATE `beliefs` — carry forward what still holds, revise what this probe changed
         (the cumulative head, not a per-probe overwrite).

PROMOTE — at run end, curate knowledge.md (the gate that grows cross-run memory):
  promote durable, CONTEXT-TAGGED, confirmed beliefs into knowledge.md (schema there).
  contradicted an existing entry? amend it (add the caveat) or retire it (mark superseded).
  gate it: only confirmed findings, never per-probe churn — that's what `beliefs` is for.

OUTPUT: `best` from search_state.json (highest tokens/s).
```

```
LLM picks config → run probe on backend → probe returns JSON → LLM reads result/failure
       ↑________________________ learn, pick next ________________________|
```

---

## Setup — pick a backend

- Agent (Claude Code) runs **locally** + holds all state; probe runs on the `backend` from
  `config.json`. Same pipeline (`probe_runner.py`), same metrics — only *where it runs* changes.
- Secrets/connection from `.env` (`.env.example`, via python-dotenv); local deps `pip install -r requirements.txt`.

**Modal** (`"backend": "modal"`, default) — serverless GPU, no machine to manage:
- Auth: `~/.modal.toml` (`modal setup`); HF creds: Modal Secret `hf-token`.
- GPU from config: `hardware` + `hardware_number` → `gpu="TYPE:N"`.
- Image: default (sdpa + flash-linear-attention) builds fast; `flash_image` (prebuilt flash-attn
  wheel) needed for the real model + `padding_free` (UNVERIFIED — needs one deploy).

**SSH VM** (`"backend": "ssh"`; `paperspace` / `lambda` aliases) — your own rented box:
- `AUTOTUNE_SSH_HOST`, `AUTOTUNE_SSH_USER`, + ONE of `AUTOTUNE_SSH_KEY` (Lambda) / `AUTOTUNE_SSH_PASSWORD` (Paperspace).
- `AUTOTUNE_SSH_ACTIVATE` — enter the swift env (e.g. `source ~/envs/swift-env/bin/activate`);
  `HF_TOKEN` forwarded to the box.
- Probe uploads + runs `probe_runner.py` on the VM, reads the result back — nothing to pre-install
  beyond ms-swift (the VM's swift-env).

Either backend:
- Dev track: public `Qwen/Qwen2.5-VL-3B-Instruct` + synthetic image JSONL. Real run: set work `model`
  + `nn_ocr4` `dataset` in `config.json` (`attn_impl` defaults `flash_attn`).
- Machine context returns **in the probe result** (`memory.gpu_mem_gb`, `utilization.cpu`) — no
  separate machine-info step. Workers ≤ `nproc // 2`.

---

## How to run a probe

**1. `config.json`** — fixed inputs only, human-owned, **never rewritten by the agent**: `backend`,
`model`, `dataset`, `max_length`, `attn_impl`, `effective_batch`, `hardware`, `hardware_number`.

```json
{
  "backend": "modal",
  "model": "Qwen/Qwen2.5-VL-3B-Instruct",
  "dataset": "synthetic",
  "max_length": 8192,
  "attn_impl": "flash_attn",
  "effective_batch": 8,
  "hardware": "A10G",
  "hardware_number": 1
}
```

Secrets (HF_TOKEN, SSH creds) in `.env`. The rest is automatic:
- **Probe mechanics win over any lever** (`probe_runner.py`): `logging_steps=1`,
  `truncation_strategy=delete`, `include_num_input_tokens_seen=true`,
  `dataloader_pin_memory/persistent_workers=true`. `lazy_tokenize=true` UNLESS `packing` or
  `dataset_num_proc` is set (both need eager preprocessing). `max_steps`/`warmup` per probe via
  `--max-steps`/`--warmup` (default 7/2; raise for data levers — see Metric).
- **`ga` derived**: `ga = effective_batch / (bs × hardware_number)` — `bs` is the one lever the
  harness reads; a `bs` that doesn't divide it → `status: error`.
- `hardware` + `hardware_number` size the GPU (Modal `TYPE:N`; DDP `NPROC_PER_NODE`); `attn_impl`
  selects the Modal image.

**2. Run a probe** — agent passes THIS probe's levers (any `swift sft` arg except off-limits;
`search_space.md` + ms-swift docs) on the **command line**. `config.json` untouched; levers recorded
in `search_state.json` history (exactly what ran).

```bash
source venv/bin/activate
python probe.py --scan                                                        # phase-0 capability scan (once)
python probe.py                                                               # seed probe (no levers, bs=1)
python probe.py --levers '{"per_device_train_batch_size": 2, "padding_free": true}' \
                --hypothesis "bs2 halves steps; predict ~+20% tok/s unless mem-bound (OOM refutes)"  # compute lever
python probe.py --levers '{"dataloader_num_workers": 8}' --max-steps 20 --warmup 5 \
                --hypothesis "GPU idle + CPU headroom → workers feed it; predict idle_frac drops"     # data lever
```

**3. Read the printed JSON** — probe parses its OWN output on the GPU, returns the structured result
below; `probe.py` appends it to `search_state.json` history. Agent reads the structured result, never
the raw training log.

---

## Probe output (the contract the LLM reads)

```json
{
  "status": "ok | oom | error",
  "config": {                       // the exact levers that ran (ms-swift CLI args) — feeds search state
    "model": "...", "world_size": 1, "effective_batch": 8,
    "per_device_train_batch_size": 2, "gradient_accumulation_steps": 4, "attn_impl": "flash_attn",
    "max_length": 8192, "padding_free": true, "gradient_checkpointing": false,
    "vit_gradient_checkpointing": false, "packing": false, "dataloader_num_workers": 1
  },
  "throughput": {                   // PRIMARY — weigh BOTH (they diverge under padding/packing)
    "tokens_per_s": 1027, "samples_per_s": 0.42, "steps_measured": 4
  },
  "memory": { "peak_mem_gb": 70.0, "gpu_mem_gb": 79.6 },   // OOM margin + total GPU memory
  "utilization": {                  // SECONDARY/diagnostic — null when no measurable step window (OOM/crash/too-fast)
    "gpu": { "idle_frac": 0.05, "p10": 82, "p50": 96, "p90": 100, "max": 100 },  // idle_frac = share of window < 20% (GPU waiting)
    "cpu": { "p10": 20, "p50": 41, "p90": 68, "max": 90 },                       // p90 = preprocessing ceiling
    "window_steps": [3, 7], "n_samples": 18           // util is windowed to the SAME steps as tokens/s
  },
  "error_detail": null              // on failure: the relevant message (OOM shape, traceback, …)
}
```

- `status`: `oom` when log contains "out of memory", else `error` on any non-zero exit.
- `utilization`: windowed to the measured steps (load/save phases excluded); **diagnostic only** —
  consult after throughput plateaus, never to pick a lever before then.
  - `gpu.idle_frac` = how often the GPU sat waiting (any cause — not necessarily data).
  - `cpu.p90` = swift process-tree CPU ceiling (% of allocated cores); it disambiguates the idle:
    cpu ~idle (<~30%) → NOT data (structure/launch), leave the dataloader; cpu active (~30-70%) →
    data-bound, workers may help; cpu pinned (~100%) → preprocessing wall.
- `error_detail`: the actual failure message (OOM + offending shape, recompile thrash, CUDA error) —
  reason about the cause instead of guessing.

---

## Metric

- **Two objectives — weigh both** (steady-state): `tokens_per_s` (token throughput) + `samples_per_s`
  (sample throughput). DIVERGE under padding / variable-length: tokens/s can be inflated by padding
  tokens (work on padding = wasted), samples/s tracks real dataset throughput. For
  `padding_free`/`packing` especially, compare both — a tokens/s win that isn't a samples/s win may
  just be more padding chewed.
- `tokens_per_s = Δ(num_input_tokens_seen) / Δelapsed` over post-warmup steps (captured live via
  `--include_num_input_tokens_seen true`; can't be back-filled).
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

## Parameters

**Effective batch — a fixed input**
- `bs × ga × world_size` = samples per gradient update. User sets the target; loop only picks the
  `bs × ga` factorization that hits it. (No outer sweep over effective batches in v1 — one search per target.)

**Phase 1 lever — bs (memory-bound)**

| Parameter | Role | Search space |
|---|---|---|
| `per_device_train_batch_size` | samples per GPU micro-batch | 1 → 2 → 4 → … (double each step) |
| `gradient_accumulation_steps` | derived (not set) | `ga = effective_batch / (bs × world_size)` |

- `bs` = runtime lever the agent picks (not a config field); `ga` derived by the probe (bad `bs` → `status: error`).
- Sweep ceiling = `bs × world_size = effective_batch` (ga=1) → no room to double bs further. Done (or
  try extensions if OOM hit first).

**Dataloader tuning (reactive — only after throughput plateaus)**
- Not a separate phase. Tune ONLY once throughput plateaued AND the idle is actually data (idle high +
  CPU busy — see below). `cpu.p90` first tells you IF it's data, then which lever:
  - **cpu p90 ~idle (< ~30%)** → the idle is NOT data (GC/optimizer/sync structure or launch gaps — the
    workers aren't even busy). Leave the dataloader alone.
  - **cpu p90 active (~30-70%)** → data-bound, under-parallelized: try `dataloader_num_workers` (cap
    `nproc // 2`), then `dataloader_prefetch_factor`. Measure with a longer probe (`--max-steps 20
    --warmup 5`) so spin-up amortizes.
  - **cpu p90 pinned (~100%)** → preprocessing wall, more workers won't help. `dataset_num_proc`
    parallelizes the upfront tokenize (auto-disables `lazy_tokenize`); beyond that = decode/IO wall (out of scope for now).
  - **idle_frac <~0.05** → compute-bound, dataloader can't help — leave it.

---

## Ceiling facts (not a recipe)

When you hit a ceiling, these are the *physics* of the levers — what they do, not a sequence. You decide what (if anything) to try.

- **`padding_free` / `packing`** — drop wasted compute on padding / fill the block with real tokens.
  Positive for throughput **and** free some memory → help against both compute saturation and OOM.
- **`gradient_checkpointing`** (+ **`vit_gradient_checkpointing`** for a VLM's vision tower) —
  recompute activations to **trade compute for memory** (~30% slower/step). Useful only when
  MEMORY-bound (to fit a bigger `bs`); pure overhead on a compute-bound GPU. `vit_` targets image-token memory.
- **Saturation / plateau** = the last 3 lever attempts each improved <5% over the current best →
  ceiling reached, stop. This is the loop's "throughput flat" branch.

---

## Guardrails

- **Effective batch is fixed** — you only choose the `bs × ga` factorization (`ga` derived, world_size
  = `hardware_number`) that hits it.
- **Inputs vs levers** — never edit the `config.json` inputs (backend, model, dataset,
  effective_batch, hardware, hardware_number); only runtime levers (`bs` + flags) are yours.
- **Only set knobs that don't change the model's output** — quality-affecting knobs are off-limits
  (`search_space.md`); your judgment, not code-enforced.
- **Hard cap: 20 probes** (count `history` entries) so the loop can't run away.
- **Every throughput number comes from a real probe** — never assert a metric.

---

## Search state — the loop's memory (`search_state.json`)

- `probe.py` appends every probe to `search_state.json` (gitignored, local) — the durable record the
  loop reasons over. One file per **session**: `setup` = the fixed inputs; a different setup starts a fresh session.
- **Setup change = fresh session:** the code detects the mismatch and **overwrites** the file on the
  first probe (old history is lost) — archive `search_state.json` first if you want to keep the prior run.
- **Three memory tiers:** `history[].note` (per probe) → `beliefs` (per session) → **`knowledge.md`**
  (across runs — the curated KB, read at start, promoted to at end; see PRIORS/PROMOTE).

**Who writes what:**
- **Code owns `history` + `best` + `env`** — auto-written (`env` once by `--scan`, history per probe), can't be forgotten.
- **Agent owns each entry's `note`, plus `beliefs`** — it edits these; code preserves them (only appends).
- **`hypothesis` write-once at launch** — passed via `--hypothesis`, persisted with the probe before
  the result is known; not revised (`note` holds the review).

**The trail — `hypothesis` (at launch) → `note` (after) → `beliefs` (whole session):**
- `history[].hypothesis` = the **prediction** — why this lever, predicted throughput, what confirms/refutes. Before the outcome.
- `history[].note` = the **REVIEW** — prediction vs actual + what the gap taught (append-only, never revised). Pair makes each probe self-documenting.
- `beliefs` = the **head** — single cumulative *current* understanding across all probes, revised each iteration. Read it to decide the next probe.
- `env` = the **capability scan** (top-level, NOT in `setup` so it can't break session matching). Consult before kernel/compile levers.

```json
{
  "setup": { "backend": "modal", "model": "...", "dataset": "...", "max_length": 8192,
             "effective_batch": 8, "hardware": "A10G", "hardware_number": 1 },
  "env": { "torch": "2.6.0", "torch_cuda": "12.4", "driver_version": "550.x",
           "gpu_name": "NVIDIA H100", "compute_capability": "9.0", "gpu_count": 1,
           "libs": { "causal_conv1d": false, "liger_kernel": true, "flash_attn": true, "fla": true } },
  "history": [
    { "iter": 1,
      "config": { "...": "the exact levers that ran (identical to the probe-output config)" },
      "hypothesis": "padding_free at bs=2: predict ~+200% (long uniform samples, little padding to lose); OOM or <seed refutes",
      "result": { "status": "ok",
                  "throughput": { "tokens_per_s": 9421, "samples_per_s": 0.59, "steps_measured": 4 },
                  "memory": { "peak_mem_gb": 12.5, "gpu_mem_gb": 24.0 },
                  "utilization": { "gpu": {"idle_frac": 0.18, "p10": 35, "p50": 80, "p90": 99, "max": 100},
                                   "cpu": {"p10": 30, "p50": 55, "p90": 70, "max": 96},
                                   "window_steps": [3, 7], "n_samples": 18 },
                  "error_detail": null },
      "note": "padding_free at bs=2 → +200% vs seed; idle_frac 0.18 so not yet compute-bound" }
  ],
  "best": { "config": { "...": "..." }, "tokens_per_s": 9421 },
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

- `best` is ranked by **tokens/s**. If its samples/s disagrees (tokens/s up, samples/s flat/down),
  flag it — under variable sequence lengths that win can be padding, so take it with a grain of salt.

```markdown
# Optimal training config

| Model | Eff batch | bs | ga | workers | flags        | tokens/s |
|-------|-----------|----|----|---------|--------------|----------|
| 3B    | 16        | 2  | 8  | 1       | padding_free | 9421     |
```

This is the direct input to the production training run.
