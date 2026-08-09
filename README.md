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
- `probe_runner.py` — one short `swift sft` run that returns measured metrics JSON, never raw logs.
  It runs on a **GPU backend you pick** (`config.json` `backend`): Modal serverless, or your own SSH GPU
  VM (Paperspace, Lambda Labs, …). Same pipeline, same metrics — only *where it runs* changes.

```
 LOCAL (LLM drives the loop, holds state)       GPU BACKEND (ephemeral, pure compute)
 ┌──────────────────────────┐   config dict     ┌─────────────────────────────────┐
 │ LLM reads the two docs   │ ────────────────▶ │ probe(config) → metrics JSON     │
 │ picks next config        │ ◀──────────────── │ swift sft (7 steps), self-parsed │
 └──────────────────────────┘   result JSON      └── Modal  ·  or  ·  SSH VM ───────┘
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

One-time setup: `pip install -r requirements.txt`. Everything is configured in `config.json`; only
secrets go in `.env` (copy from `.env.example`). Set `"backend"` in `config.json`:

- **Modal** (`"backend": "modal"`): `modal setup` (writes `~/.modal.toml`) and store an HF token as a
  Modal secret — `modal secret create hf-token HF_TOKEN=hf_...`.
- **SSH VM** (`"backend": "ssh"`): fill the `ssh` block (`host`/`user`, optional `activate`) in
  `config.json`; put the auth secret (`AUTOTUNE_SSH_KEY` for Lambda or `AUTOTUNE_SSH_PASSWORD` for
  Paperspace) in `.env`.

Then drive the loop one probe at a time — set the config in `config.json` and run the same command
regardless of backend:

```bash
source venv/bin/activate
python probe.py
```

Read the printed result (also saved to `probe_result.json`), pick the next config per `program.md`,
repeat. The dev track uses a public model + synthetic data matched to the model's modality (text or
image, no special access needed) — swap the model/dataset in `config.json` for the real run.

---

## Status

The v1 loop is defined in `search_space.md` + `program.md` and driven against a backend-agnostic
probe (`probe_runner.py`) that returns the corrected tokens/s metric plus peak memory, GPU/CPU
utilization, and structured failure detail — on Modal or an SSH GPU VM. It runs for any model,
dataset, and GPU named in `config.json`: modality (text vs vision) is auto-detected, the Modal
image and GPU follow the config, and the loop searches the throughput knobs to the fastest config.
Point it at your setup and drive the loop.

---

## Files

| File | Purpose |
|---|---|
| `search_space.md` | The knobs — throughput-safe (searchable) vs training-semantic (off-limits) + fixed inputs |
| `program.md` | How the loop runs — the authoritative runbook |
| `probe.py` | Backend-agnostic entrypoint — reads `config.json`, dispatches on its `backend`, prints metrics JSON |
| `probe_runner.py` | The portable probe pipeline — runs `swift sft`, samples GPU/CPU, parses metrics (runs on whichever backend) |
| `modal_probe.py` | Modal backend — ships `probe_runner` to a Modal serverless GPU |
| `vm_probe.py` | SSH-VM backend — runs `probe_runner` on a Paperspace/Lambda/any SSH GPU box |
| `config.json` | The editable probe config (model, dataset, bs/ga, knobs) |
