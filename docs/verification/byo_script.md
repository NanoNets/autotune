# Verification: bring-your-own-script, the agent workflow

> Private paper trail for the PR that adds the agent ↔ script / config interaction
> (`program.md` "Script mode", `/run-loop` `script=` / `script_configs=`, backend `script_files`,
> `examples/byo_script/`). In the style of `docs/verification/pr14.md`; remove before
> open-sourcing. Run 2026-10-01.

## Method

- **Box:** 1× NVIDIA A10 (23 GB), torch 2.7.0 / CUDA 12.8 (system), Python 3.10.12.
- **Probe venv** `~/autotune/.venv_at`:
  - transformers 5.15.1, accelerate 1.14.0, peft 0.20.0, huggingface-hub 1.28.0, tokenizers 0.22.2,
    safetensors 0.8.0, datasets 5.0.1;
  - numpy 1.24.4 and Pillow 12.3.0, pinned over the image's 1.21.5 / 9.0.1, which break the
    Trainer import;
  - PyYAML 5.4.1 (system).
- **Driver:** probes went through `backends.run(config)` with the **ssh** backend from a laptop,
  exactly as the agent calls it (row 13: the `local` backend on the box).
- **Workload:** every example script trains `HuggingFaceTB/SmolLM2-360M` (bf16, SDPA, seq 512) on
  seeded random tokens (`examples/byo_script/_data.py`), `effective_batch` 32 = micro-batch 8 ×
  accumulation 4.
- **Frame** ran on the box per scenario: profile only for script runs; `--ceiling` and
  `--sample-out` for row 1.
- **Each scenario** was a short supervised session following `program.md` "Script mode": Step 1
  reading, Step 2 knob audit (`runs/byo_tNN/research.md`), the 2b question where one applies, any
  2c edit and equivalence check, then the baseline plus at most 2 experiments in `results.tsv`.
  - **The user's answers** at the 2b prompt were the ones fixed in the test matrix beforehand.
  - **Per-scenario runs** were kept in `runs/byo_tNN/` (gitignored), so the user's root `run.toml`
    was not rewritten by each launch.
- **Probe settings:** `warmup` 3, `clean_steps` 10, step budget 17. Frozen peaks are from each
  run's Frame (87.2 TFLOP/s bf16, 444.7 GB/s).

## Results

**Matrix: 17 / 17 pass.**

| # | Scenario | User's answer | What the agent did | Probes (`results.tsv`) | Pass |
|---|---|---|---|---|---|
| 1 | **No script** (our HF runner): regression | — | Frame with ceiling (micro-batch max 19 at 512) + drawn sample; runner probes | baseline 12.83 → micro-batch 16: 24.30 samples/s; PR #14 `population` (`sample_hash`, `steps_per_pass`), no script labels | ✅ |
| 2 | `trainer_flags.py` (all flags) | — | knob audit: every knob a flag, no edits; changed only `script_args` | 24.83 → 25.85 samples/s; 4/4 → 2/2 micro-batches | ✅ |
| 3 | `trainer_yaml.py` + `train.yaml` | — | step budget is a config key, so a per-probe copy for **every** probe, baseline included (`probes/exp1.yaml`, `exp2.yaml`) | 24.92 → 25.74; user's `train.yaml` sha256 unchanged | ✅ |
| 4 | `trainer_hardcoded.py` | **move to config** | asked once, listing 5 hardcoded settings; copy `runs/byo_t04/train.py` + `train_config.yaml` (+7/−6 lines); equivalence 5 steps; probes via per-probe config copies | equivalence **identical** (12.56, 12.24, 12.09, 12.04, 11.99); 24.99 → 25.81 | ✅ |
| 5 | `trainer_hardcoded.py` | **expose as flags** | copy with 6 new flags, defaults = the hardcoded values (+13/−5); equivalence; probes via the new flags | equivalence **identical**; 24.77 → 25.73 | ✅ |
| 6 | `trainer_hardcoded.py` | **leave alone** | no edits; every speed knob reported out of the search; baseline only. Per-step logging is hardcoded, so it couldn't be turned off for the probe | 24.74; no further experiments | ✅ |
| 7 | `trainer_yaml_fixed_path.py` + `train.yaml` (config at a **hardcoded path**) | **expose the path as a flag** | copy adds `--config` (default: the same file); equivalence at steps 10 and 20 (the user's config logs every 10); per-probe config copies | equivalence **identical** (11.84, 11.41); 24.51 → 25.61; user's `train.yaml` unchanged | ✅ |
| 8 | `trainer_no_budget.py` (**no step budget**) | **allow adding the flag** | copy adds `--max_steps` (default −1 = unset, so it still trains 3 epochs); equivalence on the same uncapped schedule, stopped after 5 logged losses (steps 10–50); **then** probes | equivalence **identical** (11.80, 11.20, 11.05, 10.99, 10.96); 24.35 → 25.54. *See Deviations* | ✅ |
| 9 | `trainer_no_budget.py` | **decline** | Frame ran (Step 2), the 2b answer was no, so it **stopped before Step 3** and told the user to add a step budget or allow the edit | none (0 rows), by design | ✅ |
| 10 | `torch_flags.py` (plain PyTorch, no Trainer) | — | knob audit: flags; gradient checkpointing / compile **absent** (reported as not tunable, not "hardcoded") | 25.01 → 25.93; 4/4 → 2/2 | ✅ |
| 11 | `trainer_yaml.py` + **two** configs (`train.yaml`, `data.yaml`) | — | both passed (`--config` copy, `--data-config` original); `data.yaml` holds only never-touch `seq_len`, so untouched | 24.72 → 25.73; both files in `script_files` | ✅ |
| 12 | script + `_data.py` + config **under `runs/`** | — | everything under `runs/`, which the code sync excludes, so the backend **pushed** the script, its helper and the probe configs | 24.74 → 25.69 | ✅ |
| 13 | `trainer_flags.py`, **`local` backend** on the box | — | `backends.run({"backend": "local", …})` on the VM, no ssh hop | 24.74, step time +0.36% vs row 2 (within round-to-round spread) | ✅ |
| 14 | `trainer_flags.py`, an experiment that **OOMs** | — | micro-batch 32 / accum 1 (effective 32 kept) | baseline 24.79; exp 2 **`oom`** ("CUDA out of memory… 1.50 GiB"), logged, run continued | ✅ |
| 15 | row 3, **resume** | — | read `results.tsv`, continued at exp 3, new per-probe copy (gradient checkpointing on) | exp 3: 20.14 samples/s, 5.28 GB (vs 10.12): the expected checkpointing trade; numbering continued; user config unchanged | ✅ |
| 16 | `script_configs` with **no `script`** | — | `/run-loop` §2: refused before editing `run.toml` or any GPU time ("`script_configs` needs `script`…") | none | ✅ |
| 17 | `script` path **doesn't exist** | — | `/run-loop` §2: refused, naming the path; the backend independently raises `FileNotFoundError` before launch (`tests/test_backends_script.py`) | none | ✅ |

**Consistency check across scenarios:** every script baseline ran the same workload
(micro-batch 8 × accum 4) and landed at **24.35–25.01 samples/s**, a 2.7% spread across 12
different launches and fixtures. The micro-batch 16 / accum 2 experiment landed at
**25.54–25.93**. The plumbing adds no fixture-dependent artefact. Row 1's runner baseline (12.83)
isn't comparable: our runner's shipped defaults differ, and its probe trains on Frame's drawn
sample rather than the script's data.

## Deviations, recorded rather than smoothed over

- **Row 8, first attempt voided.**
  - **What happened:** the first equivalence run captured **no losses**. The harness stopped the
    uncapped runs with `timeout`, and piped, block-buffered stdout was lost.
  - **The violation:** the harness then ran probes anyway, which breaks `program.md` 2c ("prove
    equivalence before the baseline").
  - **How it was handled:** those rows were discarded (`runs/byo_t08/results.discarded-before-
    equivalence.tsv`, `obs.discarded/`). The check was re-run with `PYTHONUNBUFFERED=1`, passed,
    and only then were the probes repeated.
  - **Lesson for the agent flow:** an equivalence check that **captures nothing is a failure, not
    a pass**. The comparison returned "lengths 0 vs 0" → FAIL, which is the right verdict; the
    harness just didn't stop on it.
- **Equivalence precision:** Trainer prints losses to 2 decimals, so "identical" is to that
  precision. The rule's tolerance (~1e-3 relative) is finer than what's printed.
- **Row 7's equivalence compares 2 losses, not 5:** the user's config logs every 10 steps, and the
  original can't be made to log more often without editing it, which isn't allowed.

## Limits of this verification

- **The agent was this session, following `program.md` by hand.** No fresh agent was launched
  cold with `/run-loop`. The answers at each 2b prompt were the matrix's, not a live user's.
- **The `/run-loop` → `run.toml` merge was not exercised against the real root `run.toml`;** each
  scenario kept its own `runs/byo_tNN/`. Rows 16–17 check the skill's new validation rules as
  written.
- **Synthetic data, one model, one card, one framework family** (HF Trainer, plus one plain-
  PyTorch loop). TRL, Lightning and ms-swift's Python entry point weren't run.
- **Each scenario ran 1–3 probes,** enough to exercise the workflow, not a tuning session.
- **All script probes are provisional evidence by design** (`population.driver: "script"`; PR #17,
  "Known gaps"). Nothing here was promoted to `knowledge/`.

## Raw records

`runs/byo_t01` … `runs/byo_t14` on the machine that ran this (gitignored): `research.md` (the
agent's knob audit, the 2b question and answer, diffs, equivalence results), `results.tsv`,
`profile.json`, `obs/exp<N>.json` (the full Observation of every probe), and `probes/` (every
per-probe config copy).
