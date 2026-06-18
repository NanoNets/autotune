# autotune — the research loop

How the search/probe/measure loop actually runs **today**. This is the runbook: it must track
the code (`search.py`, `modal_probe.py`, `fake_probe.py`). When the loop changes, update this file
in the same change. The pre-Modal SSH/VM version lives in `archive/program.md`.

Goal: for a given model + dataset + GPU, find the config (batch size, grad accumulation,
dataloader/memory flags) that **maximizes training throughput** without OOM — automatically, in
as few probes as possible.

---

## What the loop is now

The loop is **code**, not a manual runbook. Two injected pieces meet in one tiny harness:

```
search(probe_fn, policy)            search.py
  loop:
    config = policy.propose(history)   # the brain — decides the next config
    if config is None: stop
    result = probe_fn(config)          # the hands — runs one experiment, measures
    history.append((config, result))
  return best(history) + full trace
```

- **policy** = the intelligence (which config to try next). Baseline: `SweepPolicy`.
- **probe_fn** = one measured training run. Two interchangeable backends, same contract:
  - `fake_probe.probe` — offline deterministic GPU simulator (no GPU, no Modal). For building/testing the loop.
  - `modal_probe.probe` — real `swift sft` run on a Modal GPU, returns measured metrics.

```
 LOCAL (agent + search loop, holds state)        MODAL (ephemeral GPU, pure compute)
 ┌───────────────────────────────┐  config dict  ┌─────────────────────────────────┐
 │ search() + policy             │ ─────────────▶ │ probe(config) → Layer-1 JSON     │
 │  decides next config          │ ◀───────────── │  runs swift sft (max_steps=7)    │
 │  ranks results, picks best    │  JSON result   │  parses its OWN output, no logs  │
 └───────────────────────────────┘                └─────────────────────────────────┘
        swap probe_fn = fake_probe.probe  → same loop runs fully offline
```

---

## The metric

**tokens/s is the intended primary objective**; samples/s is a secondary cross-check.
With `padding_free` + variable-length OCR, sequences are packed and per-step token volume
varies, so samples/s isn't comparable across configs — only tokens/s is. Rationale and the
token-source decision live in `MODAL_MIGRATION.md` §Layer-1.

Both probes already emit both metrics. The measurement is correct:
- `samples_per_s = effective_batch × Δsteps / Δelapsed` (numerator is `bs×ga`, the optimizer-step
  batch — not `bs`; using `bs` alone was the original bug).
- `tokens_per_s = Δ(num_input_tokens_seen) / Δelapsed`, captured live (can't be back-filled).
- Both measured over **post-warmup** steps (skip the first 2 — kernel/data spin-up), `warmup=2`.

> ⚠ **Pending switch:** `SweepPolicy` still ranks and detects saturation on `samples_per_s`
> (`search.py:61`, `search.py:83`). Moving both decision points to `tokens_per_s` is the
> outstanding change to make the loop optimize the metric we actually care about.

---

## The probe contract

**Input** — a config dict:

| Key | Meaning |
|---|---|
| `model` | HF/ModelScope model id |
| `dataset` | hub id, or `"synthetic"` (dev: tiny local JSONL, no hub dependency) |
| `per_device_train_batch_size` (`bs`) | samples per GPU micro-batch |
| `gradient_accumulation_steps` (`ga`) | micro-batches per optimizer step; `effective_batch = bs×ga` |
| `max_steps` | optimizer steps to run (7) |
| `attn_impl` | `sdpa` (dev image) or `flash_attn` (real) |
| `padding_free`, `gradient_checkpointing` | optional ceiling-breaker flags |

**Output** — structured Layer-1 JSON (agent never parses raw logs):

```json
{
  "status": "ok | oom | crash",
  "config": { "model": "...", "bs": 2, "ga": 4, "effective_batch": 8, "attn_impl": "sdpa" },
  "samples_per_s": 0.588,
  "tokens_per_s": 9421,
  "peak_mem_gb": 4.68,
  "gpu_util_max": 98, "gpu_util_mean": 71.0,
  "summary_samples_per_s": 0.57,
  "steps_measured": 4
}
```

`status` is `oom` when the log contains "out of memory", else `crash` on any non-zero exit.

---

## The baseline policy — `SweepPolicy`

One `SweepPolicy` instance searches **one effective_batch**. It ports the bs-doubling sweep +
OOM extension protocol; it's the measuring stick a smarter/LLM policy must beat (same optimum,
fewer probes).

```
bs = 1, ga = effective_batch          # ga = effective_batch / bs, halves as bs doubles
LOOP:
  probe(bs, ga, padding_free, gradient_checkpointing)

  status == ok:
    improvement = (sps - prev_sps) / prev_sps
    saturated if improvement < 3% for 2 consecutive doublings  → STOP, record best
    bs >= effective_batch (ga = 1 ceiling)                     → STOP, record best
    else: bs *= 2, continue

  status == oom:  run the extension protocol (below)
```

**Extension protocol — OOM only** (a ceiling-breaker, retried at the same bs that OOM'd):

```
1. padding_free if not active   → breaks OOM: continue sweep · still OOM: try next
2. gradient_checkpointing if not active → breaks OOM: continue · still OOM: STOP (ceiling final)
```

`padding_free` removes wasted compute on pad tokens (can raise throughput); `gradient_checkpointing`
trades ~slower steps for memory headroom to push bs higher. Once a flag breaks an OOM the sweep
resumes from that bs with the flag on, so the recorded best reflects the best `(bs, flags)` pair.

---

## How to run

**Offline (no GPU) — build/verify the loop:**
```bash
python test_search.py      # search() vs fake_probe; asserts the SIM's known optimum
```

**Dev Modal — one real probe end-to-end:**
```bash
source venv/bin/activate
modal run modal_probe.py   # runs CONFIG once, prints the Layer-1 JSON
```
Auth: `~/.modal.toml` (via `modal setup`). HF creds: the Modal Secret `hf-token`. Dev track uses
a public model (`Qwen2.5-0.5B-Instruct`) + synthetic JSONL on an A10G with `sdpa`.

**Real run:** swap `model`/`dataset` to the work targets and `attn_impl='flash_attn'` (needs
flash-attn in the image). Wiring `search()` to drive `modal_probe.probe.remote` is still pending
(see gaps).

---

## Current state & known gaps

What's wired today vs. what the old runbook had and the migration still owes:

- ✅ Codified: bs-doubling sweep, saturation detection, OOM→pf→gc extension protocol, corrected
  metric (samples/s + tokens/s), Layer-1 JSON, offline simulator + offline test.
- ⏳ **Metric ranking** still on `samples_per_s` — switch to `tokens_per_s` (above).
- ⏳ **Real-probe wiring** — `search()` runs against `fake_probe`; driving the Modal probe
  (`probe.remote`) from the loop is `MODAL_MIGRATION.md` Phase 4, not yet in the repo.
- ⏳ **Phase 2 dataloader sweep** (`dataloader_num_workers` / `prefetch_factor` / `dataset_num_proc`)
  is not in `SweepPolicy` — Phase 1 (memory/throughput) only.
- ⏳ **Outer loops** — multi effective_batch and multi-model are not in code; a caller would
  instantiate one `SweepPolicy` per `(model, effective_batch)`.
- ⏳ **Parallel fan-out** — adaptive bs-doubling stays sequential; independent points (separate
  effective_batches/models, Phase 2 grid) are the parallelizable ones (`MODAL_MIGRATION.md` Phase 5).
- ⏳ Extensions trigger on **OOM only**; the old `padding_free`-on-saturation path is not ported.
