# auto_tune

> Applying Karpathy's Auto-Research loop to the unglamorous-but-critical problem of training throughput.

**auto_tune** optimizes the performance of training and fine-tuning jobs on GPUs. A poorly
configured run can leave 2–4x throughput on the table — entirely due to parameters like
batch size, gradient accumulation, and dataloader workers. Finding the sweet spot manually
means hours of trial-and-error OOM crashes and log-parsing. auto_tune eliminates that
entirely: it searches the configuration space and surfaces the settings that maximize
throughput for a given model, dataset, and GPU.

It runs the same agentic pattern behind Auto-Research — **propose → evaluate → update** — against
one well-defined objective: **maximize training throughput without OOM**. No babysitting required.

---

## The loop

```
observe history → policy proposes next config → probe runs it → measure → record → repeat
```

The loop is a few lines of plumbing in `search.py`; the intelligence and the measurement are
**injected**, so the same harness runs offline or on a real GPU:

```python
search(probe_fn, policy)
  while (config := policy.propose(history)) is not None:
      history.append(probe_fn(config))   # run one experiment, measure
  return best(history)                   # highest-throughput config + full trace
```

- **policy** — decides the next config. Baseline `SweepPolicy`: double batch size until OOM or
  throughput saturates, breaking an OOM with `padding_free` then `gradient_checkpointing`. It's the
  measuring stick a smarter (e.g. LLM-driven) policy has to beat: same optimum, fewer probes.
- **probe_fn** — one short `swift sft` run that returns measured metrics, never raw logs. Two
  interchangeable backends, identical contract:
  - `fake_probe` — deterministic GPU simulator. No GPU, no Modal. Builds and tests the loop offline.
  - `modal_probe` — the real run on a Modal serverless GPU. Pinned image + GPU SKU → reproducible,
    production-matching measurement; per-second billing fits an idle-heavy search.

```
 LOCAL (search loop + state)                    MODAL (ephemeral GPU, pure compute)
 ┌──────────────────────────┐   config dict     ┌─────────────────────────────────┐
 │ search() + policy        │ ────────────────▶ │ probe(config) → metrics JSON     │
 │ ranks results, picks best│ ◀──────────────── │ swift sft (7 steps), self-parsed │
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

> Status: both probes emit tokens/s and samples/s; the policy's ranking is mid-switch from
> samples/s → tokens/s. See `program.md`.

---

## Run it

```bash
# Offline — exercise the whole loop with no GPU (asserts the simulator's known optimum)
python test_search.py

# Real — one probe end-to-end on a Modal GPU
source venv/bin/activate
modal run modal_probe.py
```

Modal auth lives in `~/.modal.toml` (`modal setup`); the HF token lives in the Modal Secret
`hf-token`. The dev track runs a public model + synthetic data so it works with no special access;
the real run swaps in the work model/dataset and `attn_impl='flash_attn'`.

---

## Status

The probe backend is mid-migration from a persistent SSH VM to Modal serverless GPUs (`MODAL_MIGRATION.md`).
The dev track works end-to-end; wiring `search()` to drive the Modal probe and re-baselining real
models against the corrected metric are the remaining steps. The prior VM-derived results and the old
SSH runbook are preserved under `archive/`.

---

## Files

| File | Purpose |
|---|---|
| `search.py` | The loop — `search(probe_fn, policy)` harness + baseline `SweepPolicy` |
| `modal_probe.py` | Real probe: runs `swift sft` on a Modal GPU, returns metrics JSON |
| `fake_probe.py` | Offline GPU simulator with the same probe contract |
| `test_search.py` | Runs the loop against the simulator; asserts it finds the known optimum |
| `program.md` | How the loop runs today — the authoritative runbook |
| `MODAL_MIGRATION.md` | The VM → Modal migration plan and probe (Layer-1) contract |
| `archive/` | Pre-Modal SSH loop, prior results, and the old runbook |
