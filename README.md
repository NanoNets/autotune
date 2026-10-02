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
padding. See `experience/test_a10G.md` for the write-up and `beliefs/` for the beliefs it produced.

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
  │  exp 2   REPLAY   — a matched prior from experience/, if any           │
  │  exp 3+  HYPOTHESIZE ─▶ TEST ─▶ OBSERVE ─▶ LEARN ─┐   ~5 min each      │
  │             ▲                                     │   until you stop   │
  │             └─────────────────────────────────────┘   it               │
  │                   → results.tsv row + obs_N.json, per experiment       │
  └───────────────────────────────────────────────────────────────────────┘
       ▼
  ┌─ STEP 5 ─ Review ─────────── experience/README.md · beliefs/README.md ─┐
  │  → experience/<run>.md  dated run entry + index line (append-only)     │
  │  → beliefs/*.md         beliefs + index line (curated, context-keyed)  │
  └───────────────────────────────────────────────────────────────────────┘
```

### What each step reads and writes

| Step | Runs | Reads | Writes |
|---|---|---|---|
| 1 Environment | `backends/{local,ssh,modal}.py` | `run.toml`, `.env` | `runs/<name>/` |
| 2 Frame | `frame/` (+ one `capture` per ceiling trial) | hub config, dataset sample, the card | `profile.json`, `ceiling.tsv` |
| 3 Confirm | — | `profile.json` | `results.tsv` (header) |
| 4 Loop | `backends.run` → `profiler.capture` → `profiler.diagnostic` | `beliefs.md`, `results.tsv` | `results.tsv` row, `obs_N.json`, `research.md` |
| 5 Review | `runs/progress.py` | `results.tsv`, `research.md` | `experience/<run>.md` + `experience/index.md`, `beliefs/*.md` + `beliefs/index.md`, `progress.png` |

### The two memories

Different questions, different read cadence, **opposite** mutation rules — see `experience/README.md` and `beliefs/README.md`.

- **`experience/`** — *has this configuration been run before?* Read **once, at Frame**, matched on
  each entry's `Context:` line. **Append-only**: a past entry is dated evidence, never edited.
- **`beliefs/`** — *what do we believe about this lever?* Read **every hypothesis**, filtered by
  `context` match. **Curated**: amended, superseded, deleted as evidence lands.

### Why the two memories matter

```
  ┌────────────────────────────────┐   ┌────────────────────────────────┐
  │  beliefs/        BELIEFS       │   │  experience/     EXPERIENCE    │
  │                                │   │                                │
  │  an engineer's opinion —       │   │  what actually happened —      │
  │  what someone who has done     │   │  dated runs, real numbers,     │
  │  this before would tell you    │   │  baseline → best, config by    │
  │  before you start              │   │  config                        │
  │                                │   │                                │
  │  "tf32 is inert under bf16 —   │   │  "Qwen 0.5B on an A10:         │
  │   don't spend the probe"       │   │   4.3 → 108.6 samples/s"       │
  └───────────────┬────────────────┘   └───────────────┬────────────────┘
                  │                                    │
                  └────────────────┬───────────────────┘
                                   ▼
                 ┌───────────────────────────────────────┐
                 │              THE AGENT                │
                 │                                       │
                 │   frames the job · hypothesizes ·     │
                 │   probes · learns                     │
                 │                                       │
                 │   starts from opinion + experience,   │
                 │   never from zero                     │
                 └───────────────────┬───────────────────┘
                                     │
      ┌──────────────────────────────┴──────────────────────────────┐
      │              every run writes back to both                  │
      ▼                                                             ▼
  OPINIONS EVOLVE                              EXPERIENCE ACCUMULATES
    re-confirmed  → probes 1 → 2 → 7             one more dated entry
    contradicted  → split by context,            a past run is never
                    never overwritten            edited — it is evidence
    proven wrong  → deleted
```

A belief behind seven runs on four card types is a different object from the same sentence behind
one. **More contributors means sharper opinions and wider experience** — and because it's all
markdown, contributing memory is a pull request, reviewed like code.

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

## Bring your own training script *(early; single GPU)*

By default autotune trains through its own runners (`frameworks/`). You can instead point it at
**your own training script**: an HF Trainer script, TRL, a hand-written PyTorch loop. autotune runs
it unchanged and measures it from the outside. Nothing in torch is patched (a guard fails the run if
anything is), and your script needs no autotune imports.

**What the script needs:**
- It runs as `python train.py <args>` on **one GPU**, with a **torch optimizer** and a **torch
  `DataLoader`** (HF Trainer, TRL, ms-swift's Python entry point, Lightning, and plain loops all
  qualify).
- A **step budget** you can pass on the command line (e.g. `--max-steps`), so a probe runs ~15
  steps and exits by itself.
- The settings you want tuned (micro-batch / grad accumulation, attention implementation,
  gradient checkpointing, compile, …), ideally reachable as **flags or a config file**. Hardcoded
  ones can be moved out with your permission. The agent sets them in your script's own terms;
  there's no adapter to write. Examples of each case: `examples/byo_script/`.
- A **fixed seed**, so every probe sees the same examples.

**The workflow:**
1. Put the script in the repo (e.g. `runs/<run_name>/train.py`). A repo-relative path is what lets
   the ssh backend run it on your GPU box.
2. Point autotune at it: `/run-loop … script=<path>`, plus `script_configs=<path>[,<path>…]` if it
   loads its settings from config files (a training YAML, say). Or set the same keys in
   `run.toml`. Without `script_configs`, the settings are taken to live in the script itself.
3. The agent reads your script and configs and maps where each speed setting lives. **If a
   setting is hardcoded,** it asks before touching anything, and offers to move such settings into
   a config file on a copy of your script, verified to train identically. It then runs the
   baseline as-is, and probes variations through your script's own flags and **per-probe copies**
   of your config files. Your files are never modified. Each probe is recorded in
   `runs/<run_name>/results.tsv`.
4. You get back the fastest settings, expressed as your script's own flags, plus what was tried and
   why.

Under the hood: `backends/local.py` runs `python -m profiler.launch -- <script> <args>`. It times
the optimizer steps, profiles one step, reads phases and token counts from torch's own trace, and
returns the same Observation a runner probe produces (`profiler/README.md`, "Attached capture").

**Not there yet** (tracked in `design_improvements.md` §10–§14):
- **Script probes are provisional evidence.** Their clean window is a fixed stretch of steps, not
  whole passes over Frame's sample, so under the Measurement axiom they don't count as `same-rows`
  comparisons, and no belief is promoted from their timings.
- **No OOM ceiling for scripts yet**: the agent won't raise the micro-batch above your baseline
  without asking you.
- GPU-active / idle fractions are not reported (no replayable reference batch).
- MFU reads far too low under LoRA / frozen weights.
- Multi-GPU and installed commands (e.g. `swift sft`, not a `.py` file) aren't supported.

## Layout

```
program.md     the playbook the agent follows — routes to the md owning each step        [built]
loop/          the design — start here (contracts · loop spine · throughput objective)   [built]
profiler/      the measurement instrument (roofline · trace · capture · diagnostic · CLI)  [built]
frame/         profile the setup · measure the OOM ceiling · build the search space       [built]
frameworks/    runner adapters — hf (real) · torch (toy)                                   [built]
backends/      where a probe runs — local · ssh · modal                                    [built]
experience/    cross-run memory: the run log, one file per run + index                     [live]
beliefs/       cross-run memory: the beliefs, one file per claim + index                   [live]
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
3. **Intelligence** *(now)* — surrogate + frontier. Cross-run memory is live (`beliefs/`);
   candidate ranking in `search/` is not.
4. **Distributed** — parallelism as a search space, multi-GPU.

Read `loop/loop.md` for how the loop works and `loop/contracts.md` for the data shapes.
