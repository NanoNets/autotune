# auto_tune

> The Auto-Research loop applied to GPU training throughput.

auto_tune finds the GPU training config that runs fastest. A badly configured run can leave
2–4x throughput on the table — from parameters like batch size, gradient accumulation, and
dataloader workers. Tuning these by hand means hours of OOM crashes and log-reading. auto_tune
does the search for you: it explores the config space and reports the settings that maximize
throughput for a given model, dataset, and GPU.

It uses the Auto-Research pattern — **propose → evaluate → update** — against one objective:
maximize training throughput.

---

## The loop

```
read history → reason → propose next config → probe runs it → record → repeat
```

Two jobs stay separate: deciding which config to try next, and measuring one config. An LLM
decides; a probe measures. Each iteration the LLM reads the full history and picks the next move.

- `program.md` — the runbook for how the loop runs. It tells the agent which signals to read and
  what they mean, then asks it to propose the single most useful next config, run it, and record
  what it learned. Guidance, not a fixed script.
- `search_space.md` — the rule for what the agent may change. A **lever** is any `swift sft`
  argument that affects throughput or memory but not the trained model. Which lever to try is the
  agent's call.
- `probe_runner.py` — one short `swift sft` run that returns measured metrics as JSON. It runs on
  the GPU backend you pick in `config.json`: Modal serverless or your own SSH GPU VM (Paperspace,
  Lambda Labs, …). Same pipeline and metrics either way — only *where it runs* changes.

```
 LOCAL (LLM drives the loop, holds state)       GPU BACKEND (ephemeral, pure compute)
 ┌──────────────────────────┐   config+levers   ┌─────────────────────────────────┐
 │ LLM reads search_state   │ ────────────────▶ │ probe(config) → metrics JSON     │
 │ picks next config        │ ◀──────────────── │ swift sft (7 steps), self-parsed │
 └──────────────────────────┘   result JSON      └── Modal  ·  or  ·  SSH VM ───────┘
```

Each probe runs 7 optimizer steps — enough for a stable throughput reading (the first 2 are
dropped as warmup), cheap enough to run dozens. There's no fixed recovery flowchart: the probe
reports the regime signals (`gpu_util`, `peak_mem`, `cpu_util`) and the agent decides what to try.

---

## Inputs, levers, and state

- **Inputs** (`config.json`, set by you, static) — `backend`, `model`, `dataset`, `max_length`,
  `attn_impl`, `effective_batch`, `hardware`, `hardware_number`. The effective batch is fixed; the
  loop only picks the `bs × ga` split that hits it (`ga` is derived and multi-GPU aware).
- **Secrets** (`.env`) — `HF_TOKEN` and any SSH credentials. Nothing else.
- **Levers** — passed per probe on the command line (`--levers '{...}'`): any `swift sft` argument
  except the model-changing ones. `config.json` is never rewritten.
- **State** (`search_state.json`, one per session) — an append-only `history` of
  `{config, result, note}`, a running `best`, and cumulative `beliefs`. `probe.py` also writes
  `probes.md`, a readable table of the session.

---

## The metric

Throughput is measured in **tokens/second**. That's the right objective once `padding_free` or
`packing` pack variable-length sequences, where samples/second is no longer comparable across
configs. The probe measures it live as `Δ tokens / Δ elapsed` over the steady-state window. It
also reports `samples/s = effective_batch × Δsteps / Δelapsed` (so the numerator is
`bs × ga × world_size`, not `bs`) and peak memory.

> The loop ranks configs by tokens/s; samples/s is a cross-check.

---

## Run it

One-time setup — create a virtualenv, install deps, and add secrets:

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then fill in HF_TOKEN and any SSH creds
```

Then configure your backend:

- **Modal** (`"backend": "modal"`): run `modal setup` (writes `~/.modal.toml`) and store an HF
  token as a Modal secret — `modal secret create hf-token HF_TOKEN=hf_...`. The GPU comes from
  `hardware` + `hardware_number` in `config.json` (e.g. `"A100"` × `4` → `gpu="A100:4"`).
- **SSH VM** (`"backend": "ssh"`): set `AUTOTUNE_SSH_HOST` and `AUTOTUNE_SSH_USER`, plus one of
  `AUTOTUNE_SSH_KEY` (Lambda) or `AUTOTUNE_SSH_PASSWORD` (Paperspace), and `AUTOTUNE_SSH_ACTIVATE`.

Set the fixed inputs in `config.json`, then run the loop one probe at a time (the backend is read
from the config):

```bash
source venv/bin/activate
python probe.py                                                   # seed probe (no levers)
python probe.py --levers '{"per_device_train_batch_size": 2, "padding_free": true}'  # exploration probe
```

Each run appends to `search_state.json` and re-renders `probes.md`. The agent reads the history,
picks the next config per `program.md`, and repeats. The dev setup uses a public vision model and
synthetic image data, so no special access is needed — swap the model and dataset in `config.json`
for a real run.

---

## Status

The v1 loop is defined by `program.md` (the runbook) and `search_space.md` (the lever rule), and
driven against a backend-agnostic probe (`probe_runner.py`). The probe returns tokens/s plus peak
memory, GPU/CPU utilization, and structured failure detail — on Modal or an SSH GPU VM, single- or
multi-GPU (DDP). Multi-GPU is verified end-to-end on Modal (A10G×2): tokens/s is a valid global
metric — `num_input_tokens_seen` is all-reduced across ranks, so it needs no ×world_size correction.

Still to do: bring in the production model (`Qwen3.5-0.8B`, which needs the prod env's
`flash-linear-attention`/`triton` pins) and re-baseline against the work dataset (`nn_ocr4`) in the
work Modal workspace.

---

## Files

| File | Purpose |
|---|---|
| `program.md` | How the loop runs — the runbook |
| `search_space.md` | The lever rule — what the agent may change (principles only) |
| `config.json` | Fixed inputs: backend, model, dataset, max_length, attn_impl, effective_batch, hardware, hardware_number |
| `probe.py` | Entrypoint — reads `config.json` + `--levers`, dispatches on `config.backend`, writes `search_state.json` and `probes.md` |
| `probe_runner.py` | The portable probe — runs `swift sft`, samples GPU/CPU, parses metrics (runs on whichever backend) |
| `modal_probe.py` | Modal backend — ships `probe_runner` to a Modal serverless GPU |
| `vm_probe.py` | SSH-VM backend — runs `probe_runner` on a Paperspace/Lambda/any SSH GPU box |
| `search_state.json` | Per-session memory: history + best + beliefs (gitignored; `probes.md` is the readable view) |
