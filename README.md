# auto_tune

> Applying Karpathy's Auto-Research loop to the unglamorous-but-critical problem of training throughput.

**auto_tune** optimizes the performance of training and fine-tuning jobs on GPUs. A poorly
configured run can leave 2–4x throughput on the table — entirely due to parameters like
batch size, gradient accumulation, and dataloader workers. Finding the sweet spot manually
means hours of trial-and-error OOM crashes and log-parsing. auto_tune eliminates that
entirely: it searches the configuration space and surfaces the settings that maximize
throughput for a given model, dataset, and GPU.

It runs the same agentic pattern behind Auto-Research — **propose → evaluate → update** — against
one well-defined objective: **maximize training throughput**. No babysitting required.

---

## The loop

```
observe history → policy proposes next config → probe runs it → measure → record → repeat
```

The intelligence (which config to try next) and the measurement (one probe) are separate. An
**LLM drives the loop**, reading two docs and running one probe at a time:

- `search_space.md` — the knobs it may set (throughput-safe) vs the training-semantic args that are
  off-limits, plus the fixed inputs (model, dataset, hardware, effective-batch target).
- `program.md` — how the loop runs: the LLM picks a config, runs a probe, reads the result/failure,
  learns, and picks the next. The first probe is a conservative seed that doubles as the
  bottleneck-characterization read; batch-size search, OOM → `padding_free`/`gradient_checkpointing`
  recovery, and saturation detection follow from there.
- `modal_probe.py` — one short `swift sft` run on a Modal serverless GPU that returns measured
  metrics JSON, never raw logs. Pinned image + GPU SKU → reproducible, production-matching
  measurement; per-second billing fits an idle-heavy search.

```
 LOCAL (LLM drives the loop, holds state)       MODAL (ephemeral GPU, pure compute)
 ┌──────────────────────────┐   config dict     ┌─────────────────────────────────┐
 │ LLM reads the two docs   │ ────────────────▶ │ probe(config) → metrics JSON     │
 │ picks next config        │ ◀──────────────── │ swift sft (7 steps), self-parsed │
 └──────────────────────────┘   result JSON      └─────────────────────────────────┘
```

Each probe is just **7 optimizer steps** — enough for a stable throughput signal (the first 2 are
discarded as warmup), cheap enough to run dozens. OOM recovery, extension activation, and saturation
detection are all automatic.

---

## The metric

Throughput is measured as **tokens/second** — the right objective once `padding_free` packs
variable-length sequences, where samples/second is no longer comparable across configs. The probe
captures it live (`Δ tokens / Δ elapsed` over the steady-state window) alongside the corrected
`samples/s = effective_batch / step_time` and peak memory.

> The probe emits tokens/s and samples/s; the loop **ranks by tokens/s** (samples/s is kept as
> a secondary cross-check).

---

## Run it

One-time setup: `pip install modal` then `modal setup` (writes `~/.modal.toml`), and store an HF
token as a Modal secret so the probe can pull the model/dataset:

```bash
modal secret create hf-token HF_TOKEN=hf_...
```

Then drive the loop one probe at a time — set the config in `modal_probe.py`'s `CONFIG` dict and run:

```bash
source venv/bin/activate
modal run modal_probe.py     # runs CONFIG once, prints the metrics JSON the LLM reads
```

Read the result, pick the next config per `program.md`, repeat. The dev track uses a public vision
model + synthetic image data (no special access needed) — swap the model/dataset in `CONFIG` for the
real run.

---

## Status

The v1 loop is defined in `search_space.md` + `program.md` and driven against `modal_probe.py`,
which returns the corrected tokens/s metric plus peak memory, GPU/CPU utilization, and structured
failure detail. Deploy-verifying one end-to-end probe (and the flash-attn image build) is the
next step.

Remaining: the production model (`Qwen3.5-0.8B` — a linear-attention model that needs the prod env's
`flash-linear-attention`/`triton` pins) and a real re-baseline against the work dataset (`nn_ocr4`)
under the work Modal workspace.

---

## Files

| File | Purpose |
|---|---|
| `search_space.md` | The knobs — throughput-safe (searchable) vs training-semantic (off-limits) + fixed inputs |
| `program.md` | How the loop runs — the authoritative runbook |
| `modal_probe.py` | The probe: runs `swift sft` on a Modal GPU, returns metrics JSON |
| `MODAL_MIGRATION.md` | The VM → Modal migration plan and probe (Layer-1) contract |
