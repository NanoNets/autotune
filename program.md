# autotune — ms-swift Throughput Optimizer (v1, LLM-in-the-loop)

Given a **fixed setup** (model, dataset, hardware, effective-batch target), find the ms-swift
training config with the highest throughput. An **LLM runs the loop**: pick a config → run it on a
GPU backend (Modal by default, or your own SSH VM) → read the structured result/failure → learn →
pick the next. The knobs are defined in `search_space.md`; this file is *how the loop runs*.

---

## Goal

Maximize **training throughput (tokens/second)** for the given setup without hitting OOM.
tokens/s is the primary metric (packed, variable-length OCR data); samples/s is a cross-check.
The probe measures everything and returns structured JSON — the LLM never parses raw logs and
never asserts a metric.

---

## Inputs (fixed — not tuned)

- **Hardware** (e.g. 4× H100) — set by the Modal `@app.function(gpu=...)`.
- **Model** (architecture + size).
- **Dataset** (the data to train on).
- **Effective-batch target** — `bs × ga × world_size`. Fixes gradient noise scale, so it's a
  training-quality decision the user owns. The loop only chooses the `bs × ga` factorization
  that hits it.

---

## The loop (LLM-in-the-loop)

```
INPUTS (fixed): model, dataset, hardware, effective_batch target
budget        : hard probe/iteration cap (guardrail — the loop can't run away)

ITER 0 — PRE-SCAN the dataset (once, before any GPU probe; see "Dataset pre-scan"):
  python dataset_token_scan.py → token_scan.json
  read the TRUE sequence-length distribution (image + text tokens) to seed bs sensibly and
  to know the memory-ceiling implication before burning a probe. Does NOT change max_length
  (off-limits); it produces the evidence for that separate, human-owned decision.

ITER 1 — CHARACTERIZE (the conservative seed; this IS Phase 0, folded in):
  pick a SAFE config — small bs (1 or 2), ga = eff/bs, no extensions, attn_impl=flash_attn
  probe it → read tokens/s, peak_mem, gpu_util, cpu_util
  this first read tells you the regime (compute / memory / data-bound) before pushing anything.
  No separate Phase 0 step — the seed probe is the characterization.

LOOP until throughput plateaus or budget is hit:
  read the last result and pick the SINGLE most informative next config:

    status == oom          → back off: drop bs (raise ga to hold effective_batch), or enable
                             gradient_checkpointing / vit_gradient_checkpointing
                             (the extension protocol below)
    gpu_util ~100%         → COMPUTE-BOUND: push throughput knobs (bs↑, padding_free, packing).
                             Do NOT touch dataloader_num_workers — the GPU isn't waiting on data.
    gpu_util low + CPU has → DATA-BOUND with headroom: add dataloader_num_workers.
      headroom
    gpu_util low + CPU     → PREPROCESSING WALL: more workers won't help. Stop tuning workers;
      pinned ~100%           the bottleneck is the data pipeline itself, not worker count.
    throughput plateaued   → SATURATED (see Saturation detection): stop.

  probe the chosen config (always respecting the fixed effective_batch).

OUTPUT: the best config found (highest tokens/s).
```

```
LLM picks config → run on Modal → probe returns JSON → LLM reads result/failure
       ↑________________________ learn, pick next ________________________|
```

---

## Dataset pre-scan (run once, before any probe)

Before the first GPU probe, measure what the model actually ingests. `dataset_token_scan.py`
computes the **true per-sample length = image tokens + text tokens** (not text-only): image
tokens from Qwen's `smart_resize` formula on each image's native `(W,H)` — read lazily from the
file header, no decode — pinned to the model's real processor config; text tokens from a fast
tokenizer pass. No model forward, no per-sample `AutoProcessor` call.

```bash
python dataset_token_scan.py --out token_scan.json          # full scan
python dataset_token_scan.py --max-samples 2000             # quick look
```

Read `token_scan.json` and use it to:

- **Seed `bs`** — the length distribution sets the activation-memory worst case; a dataset whose
  p99 is far below `max_length` has more memory headroom than the cap implies.
- **Know the OOM driver** — the per-sample `dominant` field (`image` vs `text`) says whether the
  longest samples are big images or long transcripts. For VLMs this routes the ceiling-breaker:
  image-dominated → `vit_gradient_checkpointing`; text-dominated → `gradient_checkpointing`.
- **Inform (not decide) `max_length`** — the `cut` block quantifies what a 15K→8K cut would
  truncate and whether it's lost *labels* (text-dominated overflow) or just downscaled pages.
  `max_length` stays off-limits to the loop; this is the evidence for the human decision.

**Caveat:** image tokens depend on `max_pixels`. The scan pins to the processor default (≈1280
merged tokens/image ceiling); if swift sets a different `max_pixels`, pass `--max-pixels` to match,
or the image-token counts (and the image-vs-text split) will be off.

---

## Setup — pick a backend

The agent (Claude Code) runs **locally** and holds all state; the probe runs on a GPU backend you
choose with the `AUTOTUNE_BACKEND` env var. Same pipeline (`probe_runner.py`), same metrics — only
*where the probe runs* changes. Secrets/connection come from env (`.env.example`); install the local
deps with `pip install -r requirements.txt`.

**Modal** (`AUTOTUNE_BACKEND=modal`, default) — serverless GPU, no machine to manage:

- Auth: `~/.modal.toml` (via `modal setup`). HF creds: the Modal Secret `hf-token`.
- Image: the default (sdpa + flash-linear-attention) builds fast; `flash_image` (prebuilt flash-attn
  wheel) is needed for the real model + `padding_free` (UNVERIFIED — needs one deploy).

**SSH VM** (`AUTOTUNE_BACKEND=ssh`; `paperspace` / `lambda` are accepted aliases) — your own rented box:

- `AUTOTUNE_SSH_HOST`, `AUTOTUNE_SSH_USER`, and ONE of `AUTOTUNE_SSH_KEY` (Lambda) or
  `AUTOTUNE_SSH_PASSWORD` (Paperspace).
- `AUTOTUNE_SSH_ACTIVATE` — the command to enter the swift env (e.g.
  `source ~/envs/swift-env/bin/activate`); `HF_TOKEN` is forwarded to the box.
- The probe uploads and runs `probe_runner.py` on the VM and reads the result back — nothing to
  pre-install beyond ms-swift (the VM's swift-env).

Dev track (either backend): public model `Qwen/Qwen2.5-VL-3B-Instruct` + synthetic image JSONL.
Real run: set the work model + `nn_ocr4` dataset in `config.json` and `attn_impl='flash_attn'`.

Machine context comes back **in the probe result** (`gpu_mem_gb`, `cpu_util`) on either backend — no
separate machine-info step. Worker counts should not exceed `nproc // 2`.

---

## How to run a probe

**Step 1 — set the config** (Edit `config.json`):

```json
{
  "model": "Qwen/Qwen2.5-VL-3B-Instruct",
  "dataset": "synthetic",
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 4,
  "max_steps": 7,
  "attn_impl": "flash_attn"
}
```

Optional knobs (add as needed): `padding_free`, `packing` (throughput levers);
`gradient_checkpointing`, `vit_gradient_checkpointing` (OOM ceiling-breakers); `dataloader_num_workers`
(reactive, only if data-bound).

**Step 2 — run it** — the same command for every backend:

```bash
source venv/bin/activate
AUTOTUNE_BACKEND=modal python probe.py      # or: ssh / paperspace / lambda
```

**Step 3 — read the printed Layer-1 JSON.** The probe parses its OWN output on the GPU and returns
the structured result below (also saved to `probe_result.json`) — the agent reads that, never the
raw training log.

---

## Probe output (the contract the LLM reads)

```json
{
  "status": "ok | oom | error",
  "config": { "model": "...", "bs": 2, "ga": 4, "effective_batch": 8, "attn_impl": "flash_attn" },
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
- `cpu_util` is the signal that disambiguates a data-bound GPU: **headroom → workers help; pinned
  ~100% → preprocessing wall, workers won't.**
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
| `per_device_train_batch_size` | Samples per GPU micro-batch | 1 → 2 → 4 → … → effective_batch (double each step) |
| `gradient_accumulation_steps` | Derived | `ga = effective_batch / bs` (halves as bs doubles) |

The sweep ceiling within the fixed effective batch is `bs = effective_batch` (ga=1). At that point
there's no room to double bs further — done (or try extensions if OOM was hit first).

### Reactive dataloader tuning (folded into the loop, not a separate phase)

There is **no standalone Phase 2 sweep** in v1. `dataloader_num_workers` is a *reactive* move the
LLM makes only when the seed/characterization read shows the GPU is **data-bound** (gpu_util low)
**and** CPU has headroom. If CPU is pinned ~100%, it's a preprocessing wall and more workers won't
help — stop. Cap workers at `nproc // 2`.

---

## Extension protocol (ceiling-breakers)

The extensions have different triggers because they have different natures.

| Priority | Extension | Trigger | Throughput effect |
|---|---|---|---|
| 1 | `padding_free` (and/or `packing`) | OOM **or** saturation | Positive — removes wasted compute on padding, can improve throughput directly |
| 2 | `gradient_checkpointing` (+ `vit_gradient_checkpointing` for VLMs) | OOM only | Negative — ~30% slower s/it, but a higher bs may compensate |

**`padding_free` on saturation:** saturation means the GPU is compute-bound at the current bs.
`padding_free` reduces per-sample compute (no padding tokens), which may shift the saturation point
upward and let the bs sweep continue with real gains.

**`gradient_checkpointing` on saturation:** never invoke — gc adds compute to an already-saturated
GPU. Only useful when the bottleneck is memory. For VLMs (Qwen2.5-VL / OCR), `vit_gradient_checkpointing`
is the same trade for the vision tower when image tokens are what blows the memory ceiling.

**Protocol:**

```
on OOM (before bs = effective_batch):
  try padding_free / packing if not active:
    → no OOM: mark active, resume sweep from oom_bs
    → still OOM: try next

  try gradient_checkpointing (+ vit_gradient_checkpointing for VLMs) if not active:
    → no OOM: mark active, resume sweep from oom_bs
    → still OOM: OOM ceiling is final → record best at oom_bs//2, stop

on SATURATION:
  try padding_free / packing if not active:
    → tokens/s > best * 1.05: mark active, reset saturation counter, resume sweep
    → no meaningful gain: saturation ceiling is final → stop
  (never try gradient_checkpointing on saturation)
```

When a memory extension activates, resume the bs sweep — it at bs=N may beat no-extension at
bs=N//2. The recorded best reflects the best `(bs, flags)` found.

---

## Saturation detection

Ceiling is "saturated" (not OOM) when throughput improvement over previous bs < 3%,
for 2 consecutive doublings. Try `padding_free` extension before declaring final.
If padding_free doesn't help, record best bs and stop.

```
bs=4,  ga=8  → 1.203 samp/sec  (+51% vs bs=2)  → continue
bs=8,  ga=4  → 1.231 samp/sec  (+2.3% vs bs=4) → non-improvement 1
bs=16, ga=2  → 1.235 samp/sec  (+0.3% vs bs=8) → non-improvement 2 → SATURATED → record bs=8
```

---

## Guardrails

- **Effective batch is fixed** — the LLM only chooses the `bs × ga` factorization that hits it.
- **Never touch the off-limits semantic args** (`search_space.md`): lr/vit_lr/aligner_lr,
  weight_decay, warmup, max_length, dropout, seed, bf16, and the effective-batch target.
- **Hard probe budget + iteration cap** so the loop can't run away.
- **Every throughput number comes from a real probe** — the LLM never asserts a metric.

---

## Search state

Persist after every probe so an interrupted run resumes from the last state. v1 keeps this light —
the per-run journal of probes (config → result) plus the running best. No cross-run learning DB
(deferred).

```json
{
  "machine": { "gpu_mem_gb": 24, "gpu": "A10G" },
  "model": "...", "effective_batch": 16,
  "active_extensions": ["padding_free"],
  "best": { "bs": 2, "ga": 8, "flags": ["padding_free"], "tokens_per_s": 9421 },
  "history": [ { "config": { "...": "..." }, "result": { "...": "..." } } ]
}
```

---

## Results log

Append one row per probe; never modify existing rows.

```
model  eff_batch  bs  ga  workers  flags                         tokens_per_s  status  notes
3B     16         1   16  1        -                             3120          ok      seed / characterization
3B     16         2   8   1        padding_free                  9421          ok      +200%
3B     16         4   4   1        padding_free                  9500          ok      +0.8% → non-improvement 1
3B     16         8   2   1        padding_free,gradient_ckpt    6100          oom→ok  gc broke ceiling but slower
```

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
