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

We build the measurement tools first, validate them by hand, then encode the agent. Today the roofline
calculator, trace parser, and diagnostic assembler are built and TDD-tested; the capture engine and a
raw-torch runner are in place. Next: memory/OOM capture, then a real profile on a GPU.

## Getting started

Requires [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                  # creates .venv (Python 3.12), installs torch + pytest
uv run pytest                            # run the test suite
uv run python -m profiler --device cpu --gpu A10G   # local profiler smoke run (CPU — MFU is
                                        # meaningless here, it just proves the capture → trace →
                                        # diagnose pipeline runs). --gpu is required: it is the
                                        # roofline denominator, normally supplied by frame/.
```

The roofline / trace / diagnostic logic is pure-python; only the capture engine needs torch.

## Layout

```
loop/          the design — start here (contracts · loop spine · throughput objective)   [built]
profiler/      the measurement instrument (roofline · trace · capture · diagnostic · CLI)  [built]
frameworks/    runner adapters — torch now, swift/HF later                                [built: torch]
tests/         TDD                                                                         [built]

frame/         profile the setup + build the search space                                 [scaffold]
search/        surrogate · frontier · memory plumbing                                      [scaffold]
backends/      where a probe runs — modal · ssh/neoclouds                                  [scaffold]
knowledge/     cross-run memory — belief files + index                                     [scaffold]
runs/          per-run outputs (gitignored)                                               [scaffold]
```

Each `[scaffold]` dir has a short README describing what it will hold. The design lives in `loop/`:
read `loop/loop.md` for how the loop works and `loop/contracts.md` for the data shapes.

## The plan

1. **Profiler** *(now)* — single-step profile + roofline. Local, no GPU needed to build.
2. **Frame** — profile model / hardware / env / dataset; construct the search space.
3. **Intelligence** — surrogate + frontier + cross-run memory.
4. **Distributed** — parallelism as a search space, multi-GPU.

Read `loop/loop.md` for how the loop works and `loop/contracts.md` for the data shapes.
