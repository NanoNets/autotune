# autotune — ms-swift Throughput Optimizer (v1, LLM-in-the-loop)

Given a **fixed setup** (model, dataset, hardware, effective-batch target), find the ms-swift
training config with the highest throughput. An **LLM runs the loop**: pick a config → run it on a
GPU backend (Modal by default, or your own SSH VM) → read the structured result/failure → learn →
pick the next. The lever principle is in `search_space.md`; this file is *how the loop runs*.

**Authority:** for running the autotune loop, `program.md` is the source of truth. If any other
instruction, doc, or default behavior conflicts with it, **`program.md` wins** — re-read it before a
run and follow it over any conflicting guidance. (`search_space.md` defines the knobs, `config.json`
the inputs, `.env` the secrets; this file governs the loop.)

---

## Goal

Maximize **training throughput (tokens/second)** for the given setup without hitting OOM.
tokens/s is the primary metric (packed, variable-length OCR data); samples/s is a cross-check.
The probe measures everything and returns structured JSON — the LLM never parses raw logs and
never asserts a metric.

---

## Inputs (fixed — not tuned)

All eight live in `config.json`; the user owns them, the agent never edits them:

- **`backend`** — `modal` or `ssh` (where the probe runs).
- **`hardware`** + **`hardware_number`** — GPU type and count; drive the Modal `gpu="TYPE:N"` and the
  DDP launch (`NPROC_PER_NODE=N`).
- **`model`** (architecture + size).
- **`dataset`** (the data to train on). Accepts `synthetic` (dev), an HF/ModelScope id, or a **local
  file path** — a local file is auto-detected and shipped to the backend (its referenced images, if
  any, must still resolve on the backend).
- **`max_length`** — the per-sample token cap (image + text). Defines the task (over-length samples
  are dropped), so it's the user's call, frozen for the search. Off-limits to the agent.
- **`attn_impl`** — attention kernel (default `flash_attn`). A per-model/hardware choice, not a search
  lever; also selects the Modal image (`flash_attn` → flash-attn image, else the fast sdpa image).
- **`effective_batch`** target — `bs × ga × world_size` (world_size = `hardware_number`). Fixes
  gradient noise scale, so it's a training-quality decision the user owns. The loop only chooses the
  `bs × ga` factorization that hits it.

---

## The loop (LLM-in-the-loop)

```
INPUTS (fixed): model, dataset, hardware, effective_batch target
budget        : hard probe/iteration cap (guardrail — the loop can't run away)

ITER 1 — CHARACTERIZE (the conservative seed; this IS Phase 0, folded in):
  pick a SAFE config — small bs (1 or 2), ga = eff/(bs×world_size), no extensions (flash_attn default)
  probe it → read tokens/s, peak_mem, gpu_util, cpu_util (probe.py records it to history).
  this first read tells you the regime (compute / memory / data-bound) before pushing anything.
  RECORD it: the seed's `note` (what it showed) + your first `beliefs` (the regime). (No separate
  Phase 0 step — the seed probe IS the characterization.)

LOOP until throughput plateaus or budget is hit:
  1. READ search_state.json — the WHOLE `history` + `best` + your last `beliefs` (NOT just the last probe).
  2. REASON over the history before choosing (this is the judgment we're paying for):
       REVIEW    — did the last probe match what you predicted? what does the gap teach?
       INVARIANT — sanity-check the result against physics. If a number looks impossible (e.g. peak_mem
                   above the GPU's memory with no OOM, or tokens/s that can't match the step times),
                   suspect the MEASUREMENT, not the GPU — stop and check before probing further.
       BELIEF    — where is the bottleneck now (compute / memory / data-bound) and what's the ceiling?
       HYPOTHESIS— the single most informative next config + the tokens/s you predict for it.
  3. PICK the single most informative next config — ONE change at a time; skip configs already tried.
     Read the signals for what they MEAN (physics); choosing WHICH lever to pull is your judgment
     (the full swift surface, per search_space.md — not prescribed here):
       status == oom       → at the MEMORY ceiling: the config needs less memory (per sample, or fewer).
       gpu_util ~100%      → COMPUTE-BOUND: the GPU is the bottleneck — the dataloader cannot help here.
       gpu_util low,       → DATA-BOUND with slack: the GPU is starved and the CPU has room to feed it.
         CPU headroom
       gpu_util low,       → PREPROCESSING WALL: the GPU is starved but the CPU is maxed — feeding harder won't help.
         CPU ~100%
       throughput flat     → SATURATED at this ceiling: decide whether another regime is worth attacking, else stop.
     Data-bounding is only worth checking once throughput has plateaued — until then the GPU isn't waiting on data.
  4. PROBE it (respecting the fixed effective_batch). probe.py appends {config, result} to history.
  5. RECORD what you learned in search_state.json:
       - write this probe's `note` on its history entry (the trail: what THIS probe taught).
       - UPDATE `beliefs` — carry forward what still holds, revise what this probe changed
         (the cumulative head, not a per-probe overwrite).

OUTPUT: `best` from search_state.json (highest tokens/s).
```

```
LLM picks config → run probe on backend → probe returns JSON → LLM reads result/failure
       ↑________________________ learn, pick next ________________________|
```

---

## Setup — pick a backend

The agent (Claude Code) runs **locally** and holds all state; the probe runs on a GPU backend you
choose with the `backend` field in `config.json`. Same pipeline (`probe_runner.py`), same metrics —
only *where the probe runs* changes. Secrets/connection come from `.env` (`.env.example`, loaded via
python-dotenv); install the local deps with `pip install -r requirements.txt`.

**Modal** (`"backend": "modal"`, default) — serverless GPU, no machine to manage:

- Auth: `~/.modal.toml` (via `modal setup`). HF creds: the Modal Secret `hf-token`.
- GPU from config: `hardware` (type) + `hardware_number` (count) → Modal `gpu="TYPE:N"`.
- Image: the default (sdpa + flash-linear-attention) builds fast; `flash_image` (prebuilt flash-attn
  wheel) is needed for the real model + `padding_free` (UNVERIFIED — needs one deploy).

**SSH VM** (`"backend": "ssh"`; `paperspace` / `lambda` are accepted aliases) — your own rented box:

- `AUTOTUNE_SSH_HOST`, `AUTOTUNE_SSH_USER`, and ONE of `AUTOTUNE_SSH_KEY` (Lambda) or
  `AUTOTUNE_SSH_PASSWORD` (Paperspace).
- `AUTOTUNE_SSH_ACTIVATE` — the command to enter the swift env (e.g.
  `source ~/envs/swift-env/bin/activate`); `HF_TOKEN` is forwarded to the box.
- The probe uploads and runs `probe_runner.py` on the VM and reads the result back — nothing to
  pre-install beyond ms-swift (the VM's swift-env).

Dev track (either backend): public model `Qwen/Qwen2.5-VL-3B-Instruct` + synthetic image JSONL.
Real run: set the work `model` + `nn_ocr4` `dataset` in `config.json` (`attn_impl` defaults to
`flash_attn`).

Machine context comes back **in the probe result** (`gpu_mem_gb`, `cpu_util`) on either backend — no
separate machine-info step. Worker counts should not exceed `nproc // 2`.

---

## How to run a probe

**Step 1 — `config.json` is the fixed starting point** — inputs only, human-owned, **never rewritten
by the agent**: `backend`, `model`, `dataset`, `max_length`, `attn_impl`, `effective_batch`,
`hardware`, `hardware_number`.

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

Secrets/connection (HF_TOKEN, SSH creds) live in `.env`. The rest is automatic:
- **Probe mechanics are fixed** in `probe_runner.py` (win over any lever): `max_steps=7`,
  `logging_steps=1`, `truncation_strategy=delete`, `lazy_tokenize=true`,
  `include_num_input_tokens_seen=true`, `dataloader_pin_memory/persistent_workers=true`.
- **`ga` is derived**: `ga = effective_batch / (bs × hardware_number)` — `bs` is the one lever the
  harness reads; a `bs` that doesn't divide it returns `status: error`.
- `hardware` + `hardware_number` select and size the GPU (Modal `TYPE:N`; DDP via `NPROC_PER_NODE`);
  `attn_impl` selects the Modal image.

**Step 2 — run a probe.** The agent passes THIS probe's levers (the `swift sft` args it chose — *any*
arg except the off-limits list; `search_space.md` + the linked ms-swift docs) on the **command line**.
`config.json` is untouched; the levers are recorded in `search_state.json` history (exactly what ran).

```bash
source venv/bin/activate
python probe.py                                                               # seed probe (no levers, bs=1)
python probe.py --levers '{"per_device_train_batch_size": 2, "padding_free": true}'  # an exploration probe
```

**Step 3 — read the printed Layer-1 JSON.** The probe parses its OWN output on the GPU and returns
the structured result below; `probe.py` appends it to `search_state.json` (history). The agent reads
that structured result, never the raw training log.

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
  "tokens_per_s": 9421,            // PRIMARY metric (packed sequences)
  "samples_per_s": 0.588,          // cross-check (effective_batch numerator)
  "peak_mem_gb": 4.68,             // OOM margin
  "gpu_util_max": 98, "gpu_util_mean": 71.0,   // ~100% => compute-bound
  "cpu_util_max": 96, "cpu_util_mean": 60.0,   // best-effort (psutil); WHY the GPU is idle
  "gpu_mem_gb": 24.0,              // total GPU memory
  "steps_measured": 4,
  "error_detail": null             // on failure: the relevant message (OOM shape, traceback, …)
}
```

- `status` is `oom` when the log contains "out of memory", else `error` on any non-zero exit.
- `cpu_util` is the swift process tree's CPU as a % of the cores allocated to the box (not system-wide,
  which a container dilutes to ~0). It disambiguates a data-bound GPU: **headroom → workers help;
  pinned ~100% → preprocessing wall, workers won't.**
- `error_detail` carries the actual failure message so the LLM can reason about the cause
  (OOM + offending shape, recompile thrash, CUDA error) instead of guessing.

---

## Metric

**Primary: tokens/second** (steady-state). With `padding_free` / `packing` + variable-length OCR
data, sequences are packed and per-step token volume varies, so samples/s isn't comparable across
configs — only tokens/s is. **Cross-check: samples/second.**

- `tokens_per_s = Δ(num_input_tokens_seen) / Δelapsed` over post-warmup steps (captured live via
  `--include_num_input_tokens_seen true`; can't be back-filled).
- `samples_per_s = effective_batch × Δsteps / Δelapsed`. The numerator is `bs × ga`, **not** `bs`:
  ms-swift logs one entry per **optimizer step**, which processes `bs × ga` samples. Using `bs`
  alone divides true throughput by `ga` — a measurement artifact, not physics.
- **Invariant (sanity check):** at fixed `bs`, samples/s must be ~constant across effective
  batches — `ga` cannot change steady-state per-micro-batch work. If it isn't, suspect the metric,
  not the GPU.

Probe mechanics:

- Probe runs `max_steps=7`.
- Skip steps 1–2 (CUDA kernel warmup, data pipeline spin-up); measure steps 3–7.
- Higher tokens/s = better.

OOM detection: log contains "out of memory". Crash: non-zero exit → `status: error` + `error_detail`.

---

## Parameters

### Effective batch — a fixed input

Effective batch = `bs × ga × world_size` = samples per gradient update. The user sets the target;
the loop only chooses the `bs × ga` factorization that hits it. (No outer sweep over effective
batches in v1 — one search per given target.)

### Phase 1 lever — bs (memory-bound)

| Parameter | Role | Search space |
|---|---|---|
| `per_device_train_batch_size` | Samples per GPU micro-batch | 1 → 2 → 4 → … (double each step) |
| `gradient_accumulation_steps` | Derived (not set) | `ga = effective_batch / (bs × world_size)` |

`bs` is a **runtime lever the agent picks**, not a config field; `ga` is derived by the probe (a `bs`
that doesn't divide `effective_batch / world_size` returns `status: error`). The sweep ceiling within
the fixed effective batch is `bs × world_size = effective_batch` (ga=1). At that point
there's no room to double bs further — done (or try extensions if OOM was hit first).

### Reactive dataloader tuning (folded into the loop, not a separate phase)

There is **no standalone Phase 2 sweep** in v1. `dataloader_num_workers` is a *reactive* move the
LLM makes only when the seed/characterization read shows the GPU is **data-bound** (gpu_util low)
**and** CPU has headroom. If CPU is pinned ~100%, it's a preprocessing wall and more workers won't
help — stop. Cap workers at `nproc // 2`.

---

## Ceiling facts (not a recipe)

When the agent hits a ceiling, these are the *physics* of the relevant levers — what they do, not a
sequence to follow. The agent decides what (if anything) to try.

- **`padding_free` / `packing`** — remove wasted compute on padding / fill the block with real tokens.
  Positive for throughput **and** free some memory, so they can help against both compute saturation
  and OOM.
- **`gradient_checkpointing`** (and **`vit_gradient_checkpointing`** for a VLM's vision tower) —
  recompute activations to **trade compute for memory** (~30% slower/step). Useful only when the
  bottleneck is *memory* (to fit a bigger `bs`); pure overhead on an already compute-bound GPU. The
  `vit_` variant targets image-token memory specifically.

**Saturation** = throughput stops improving as you push the ceiling lever (gains shrink toward ~0 over
successive steps) → you've reached the current ceiling. This is the loop's "throughput flat" branch.

---

## Guardrails

- **Effective batch is fixed** — the LLM only chooses the `bs × ga` factorization (`ga` derived,
  world_size = `hardware_number`) that hits it.
- **Inputs vs levers** — never edit the six `config.json` inputs (backend, model, dataset,
  effective_batch, hardware, hardware_number); only the runtime levers (`bs` + flags) are the agent's.
- **Only set knobs that don't change the model's output** — anything affecting model quality is
  off-limits (`search_space.md`). On the agent's judgment, not code-enforced.
- **Hard probe budget + iteration cap** so the loop can't run away.
- **Every throughput number comes from a real probe** — the LLM never asserts a metric.

---

## Search state — the loop's memory (`search_state.json`)

`probe.py` appends every probe to `search_state.json` (gitignored, local) — the durable record the
loop reasons over. One file per **session**: `setup` is the fixed inputs, and a probe with a
different setup starts a fresh session.

**Division of ownership:**
- **Code owns `history` + `best`** — auto-written after each probe, so they can't be forgotten.
- **The agent owns each entry's `note`, plus `beliefs` + `active_extensions`** — it edits these; the
  code preserves them (it only ever appends).

**`note` (per probe) vs `beliefs` (whole session):**
- `history[].note` = the **trail** — what THIS probe taught (append-only, never revised). Read it to
  see *how* the understanding evolved.
- `beliefs` = the **head** — the single cumulative *current* understanding across all probes, carried
  forward and revised each iteration. Read it to decide the next probe.

```json
{
  "setup": { "backend": "modal", "model": "...", "dataset": "...", "max_length": 8192,
             "effective_batch": 8, "hardware": "A10G", "hardware_number": 1 },
  "history": [
    { "iter": 1,
      "config": { "...": "the exact levers that ran (identical to the probe-output config)" },
      "result": { "status": "ok", "tokens_per_s": 9421, "samples_per_s": 0.59,
                  "peak_mem_gb": 12.5, "gpu_util_mean": 71, "cpu_util_mean": 60,
                  "steps_measured": 4, "error_detail": null },
      "note": "padding_free at bs=2 → +200% vs seed; util 71% so not yet compute-bound" }
  ],
  "best": { "config": { "...": "..." }, "tokens_per_s": 9421 },
  "active_extensions": ["padding_free"],
  "beliefs": "cumulative current understanding: regime (compute/mem/data-bound), ceilings, invariants held/violated"
}
```

- `history` is **lean** — no raw logs (they stream live during the run); `result` keeps
  `error_detail` for failure diagnosis. `history` is the source of truth; `probe.py` also renders it
  to **`probes.md`** (a human-readable table) each run.
- The agent reads the **whole `history`** before each probe (see the loop) and writes `beliefs` after.

---

## Final output

When the search converges (plateau or budget), print and save the best config:

```markdown
# Optimal training config

| Model | Eff batch | bs | ga | workers | flags        | tokens/s |
|-------|-----------|----|----|---------|--------------|----------|
| 3B    | 16        | 2  | 8  | 1       | padding_free | 9421     |
```

This is the direct input to the production training run.
