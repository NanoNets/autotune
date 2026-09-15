# autotune

> Find the fastest training config for your hardware — automatically, without touching model quality.

**Status: early. Rebuilding from scratch, profiler-first.** This is a working skeleton, not a finished tool.
**Single-GPU only for now** — multi-GPU/parallelism is planned (see "The plan," §4) but not yet built.

autotune is a research loop that profiles a real training step, sees where the time is going against
the hardware's roofline, and searches the configuration space for the settings that maximize throughput
— for any model, GPU, and framework. It only ever changes settings that affect *speed*, never ones that
change the model you get, so a faster config is never a worse model.

The intelligence is an LLM agent following the markdown in `loop/`; the measurement is deterministic
code in `profiler/`. **The agent decides; the code measures.**

## Where it's at

We build the measurement tools first, validate them by hand, then encode the agent. The profiler,
Frame, the HF runner and the local/ssh/modal backends are built and TDD-tested. Frame **measures**
the OOM ceiling — the largest `micro_batch` that runs at the `max_len` you declare — on manufactured
worst-case batches rather than hoping the dataset supplies one, so `micro_batch` is bounded by
evidence before the search starts.

**First full run is in** (`test_a10G`, Qwen2.5-0.5B on an A10): 23 experiments, **4.3 → 108.6
samples/s (25x)**, MFU 28.7% → 58.6%. The dominant win wasn't a kernel — it was not computing
padding. See `knowledge.md` for the write-up and `knowledge/` for the beliefs it produced.

Next: the surrogate + frontier in `search/`, so the loop ranks candidates instead of hand-searching.

## How a run works

`program.md` is the playbook the agent follows. It **routes** — each step points at the md that owns
the procedure. This is what actually happens at each one, and what lands on disk.

```
  YOU: run.toml          model · dataset · gpu · effective_batch · max_len · backend
       │
       ▼
  ┌─ STEP 1 ─ Environment ──────────────────────────── backends/README.md ─┐
  │  pick where probes run: local (this box) · ssh (your GPU box) · modal  │
  │  agree a run name                                    → runs/<name>/    │
  └───────────────────────────────────────────────────────────────────────┘
       ▼
  ┌─ STEP 2 ─ Frame, once ──────────────────────────────── frame/README.md ┐
  │  model.py     HF config, no weights   → params · attention · vocab     │
  │  dataset.py   tokenize a sample       → p50 / p90 / p99 lengths        │
  │  hardware.py  measured on the card    → peak FLOP/s · bandwidth · mem  │
  │  memory.py    analytical              → weights + grads + optimizer    │
  │  ceiling.py   REAL probes, worst case → max micro_batch at max_len     │
  │                                → runs/<name>/profile.json · ceiling.tsv│
  └───────────────────────────────────────────────────────────────────────┘
       ▼
  ┌─ STEP 3 ─ Confirm ──────────────────────────────────── runs/README.md ─┐
  │  agent reports summary + notes + the measured ceiling                  │
  │  YOU approve  ─────────────────────────▶  results.tsv created          │
  │  (last human checkpoint — "NEVER stop" applies from here)              │
  └───────────────────────────────────────────────────────────────────────┘
       ▼
  ┌─ STEP 4 ─ The loop, unattended ───────────────────────── loop/loop.md ─┐
  │  exp 1   BASELINE — shipped defaults, unmodified (never skipped)       │
  │  exp 2   REPLAY   — a matched prior from knowledge.md, if any          │
  │  exp 3+  HYPOTHESIZE ─▶ TEST ─▶ OBSERVE ─▶ LEARN ─┐   ~5 min each      │
  │             ▲                                     │   until you stop   │
  │             └─────────────────────────────────────┘   it               │
  │                   → results.tsv row + obs_N.json, per experiment       │
  └───────────────────────────────────────────────────────────────────────┘
       ▼
  ┌─ STEP 5 ─ Review ─────────────────────────────── knowledge/README.md ──┐
  │  → knowledge.md   dated run entry (append-only history)                │
  │  → knowledge/*.md beliefs (curated, context-keyed priors)              │
  └───────────────────────────────────────────────────────────────────────┘
```

### What each step reads and writes

| Step | Runs | Reads | Writes |
|---|---|---|---|
| 1 Environment | `backends/{local,ssh,modal}.py` | `run.toml`, `.env` | `runs/<name>/` |
| 2 Frame | `frame/` (+ one `capture` per ceiling trial) | hub config, dataset sample, the card | `profile.json`, `ceiling.tsv` |
| 3 Confirm | — | `profile.json` | `results.tsv` (header) |
| 4 Loop | `backends.run` → `profiler.capture` → `profiler.diagnostic` | `beliefs.md`, `results.tsv` | `results.tsv` row, `obs_N.json`, `research.md` |
| 5 Review | `runs/progress.py` | `results.tsv`, `research.md` | `knowledge.md`, `knowledge/*.md`, `progress.png` |

### The two memories

Different questions, different read cadence, **opposite** mutation rules — see `knowledge/README.md`.

- **`knowledge.md`** — *has this configuration been run before?* Read **once, at Frame**, matched on
  each entry's `Context:` line. **Append-only**: a past entry is dated evidence, never edited.
- **`knowledge/`** — *what do we believe about this lever?* Read **every hypothesis**, filtered by
  `context` match. **Curated**: amended, superseded, deleted as evidence lands.

### The invariants a dev should know

- **The agent decides; the code measures.** Every number comes from a real probe.
- **Only speed knobs are searchable.** `max_len`, `effective_batch`, precision, LR are off-limits —
  that's what makes "faster" a strict win, never a trade (`loop/throughput.md`).
- **The baseline is experiment 1, always.** A prior may be replayed as experiment 2, never as the
  baseline — measured against anything but shipped defaults, "+X%" means nothing.
- **Two ledgers.** Memory feasibility is measured on synthetic worst-case batches → `ceiling.tsv`.
  Throughput is measured on real data → `results.tsv`. Synthetic probes never enter the frontier.

## Getting started

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                  # creates .venv (Python 3.12), installs torch + pytest
uv run pytest                            # run the test suite
cp run.toml.example run.toml             # then fill it in — this is the input to Step 1
```

`run.toml` is the whole human-facing surface: model, dataset, gpu, backend, and the two constants
the loop may never change — `effective_batch` and `max_len`. Secrets go in `.env` (see
`.env.example`). Then hand `program.md` to an agent and it walks Steps 1-5 above.

Smoke-test the machinery without a GPU or a config:

```bash
uv run python -m profiler --device cpu --gpu A10G   # capture → trace → diagnose (MFU is
                                                    # meaningless on CPU; it proves the pipe runs)
```

The roofline / trace / diagnostic logic is pure-python; only the capture engine needs torch.

## Layout

```
program.md     the playbook the agent follows — routes to the md owning each step        [built]
loop/          the design — start here (contracts · loop spine · throughput objective)   [built]
profiler/      the measurement instrument (roofline · trace · capture · diagnostic · CLI)  [built]
frame/         profile the setup · measure the OOM ceiling · build the search space       [built]
frameworks/    runner adapters — hf (real) · torch (toy)                                   [built]
backends/      where a probe runs — local · ssh · modal                                    [built]
knowledge/     cross-run memory — knowledge.md log + belief files + index                  [live]
runs/          per-run outputs (gitignored) — profile · ceiling · results · observations   [built]
tests/         TDD                                                                         [built]

search/        surrogate · frontier · memory plumbing                                      [scaffold]
```

Every dir has a README describing what it holds; `[scaffold]` ones describe what they *will*. The
design lives in `loop/`: read `loop/loop.md` for how the loop works and `loop/contracts.md` for the
data shapes. Open dev questions — things not yet settled by evidence — are in
`design_improvements.md` (dev notes; deliberately **not** part of the agent's prompt hierarchy).

## The plan

1. **Profiler** *(done)* — single-step profile + roofline. Local, no GPU needed to build.
2. **Frame** *(done)* — profile model / hardware / env / dataset; measure the OOM ceiling.
3. **Intelligence** *(now)* — surrogate + frontier. Cross-run memory is live (`knowledge/`);
   candidate ranking in `search/` is not.
4. **Distributed** — parallelism as a search space, multi-GPU.

Read `loop/loop.md` for how the loop works and `loop/contracts.md` for the data shapes.
