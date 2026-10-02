---
name: run-loop
description: Run an autonomous throughput-tuning session in this repo by following program.md, with the run's inputs taken from the command line instead of gathered by back-and-forth.
argument-hint: "[run_name] [model=…] [dataset=…[:config]] [max_len=N] [effective_batch=N] [host=user@ip | local] [gpu=…] [script=path] [script_configs=path[,path…]] [resume | review] [smoke] [budget=… (placeholder)]"
disable-model-invocation: true
---

# /run-loop

`program.md` is the playbook and stays the single source of truth. Read it now, in full, and follow
it step by step. This skill only changes **where the inputs come from**: the arguments below answer
the questions `program.md` would otherwise ask, so the human can launch a night's run in one line.

Arguments: `$ARGUMENTS`

## 1. Parse the arguments

Tokens are space-separated. A bare first token that isn't a keyword is the **run name**. Everything
else is `key=value` or a bare flag. Unknown keys: stop and say which ones, don't guess.

| Argument | Answers | Effect |
|---|---|---|
| `<run_name>` (first bare token) | Step 1 "ask the user for the run name" | Use it as-is; don't ask. Absent → ask once, default `<model>_<dataset>` as `program.md` says |
| `model=<hf_id>` | `run.toml` `model` | Write into `run.toml` |
| `dataset=<hf_id>[:<config>]` | `run.toml` `dataset` + `name` | The part after `:` is the subset (`name`). No `:` → remove `name` |
| `max_len=<N>` | `run.toml` `max_len` | Human-owned constant. Write it, never tune it |
| `effective_batch=<N>` | `run.toml` `effective_batch` | Human-owned constant. Write it, never tune it |
| `host=<user>@<ip>` | `backend`, `[ssh] host/user` | Sets `backend = "ssh"` and fills `[ssh]`. The credential stays in `.env` |
| `local` | `backend` | Sets `backend = "local"` |
| `gpu=<key>` | `run.toml` `gpu` | Must be a key in `profiler/roofline.py` `GPU_SPECS` |
| `script=<path>` | `run.toml` `script` | **Optional.** The user's own training script; probes then run it unchanged instead of our `frameworks/` runners (`program.md`, "Script mode"). Absent → our runners, as before. Repo-relative (the ssh backend runs the same path on the box) |
| `script_configs=<path>[,<path>…]` | `run.toml` `script_configs` | **Only with `script`.** The config file(s) the script loads its settings from (a training YAML, a JSON, …), comma-separated. Absent → the settings live in the script itself (its flags, or hardcoded) |
| `budget=<Nh>` or `budget=<N>x` | *(placeholder, not built)* | Reserved for a wall-clock (`8h`) or experiment-count (`40x`) limit. Accept it, echo it in the Step 3 summary as "noted, not enforced in v1", and otherwise ignore it. "NEVER stop" still applies |
| `resume` | Skips Steps 1–3 | Needs an existing `runs/<run_name>/results.tsv`. See §3 |
| `review` | Runs Step 5 only | Needs an existing `runs/<run_name>/results.tsv` |
| `smoke` | `device` | Sets `device = "cpu"` for a pipeline smoke test. Say in the Step 3 summary that its MFU means nothing and no result gets promoted to `beliefs/` |

`resume`, `review` and `smoke` exclude each other.

## 2. Merge with `run.toml`

**Check the script arguments first, before touching `run.toml` or any GPU time:**
- `script_configs` given without `script` → stop: "`script_configs` needs `script`: it lists the
  files *that script* loads."
- every `script` / `script_configs` path must exist and be relative to the repo root. If not,
  stop and name the missing or absolute path.
- write `script` as a string and `script_configs` as a list of strings.

- Arguments override `run.toml`. Edit only the fields you were given and keep the file's comments.
- No `run.toml` → copy `run.toml.example` first, then apply the arguments.
- After merging, if any required field (`model`, `dataset`, `gpu`, `effective_batch`, `max_len`,
  `backend`, plus `[ssh]` when the backend is `ssh`) is still missing or still holds the example's
  placeholder value, ask for **all of them in one message**, not one at a time.
- Arguments that changed `run.toml` are listed in the Step 3 summary as `field: old → new`, so an
  argument that silently replaced last night's config is visible before any GPU time is spent.

## 3. Mode

- **Default:** `program.md` Steps 1→5. The Step 3 confirmation gate stays. It is the last point a
  human looks before an unattended night, and this skill never skips it.
- **`resume`:** `runs/<run_name>/` has to exist with `results.tsv` and `profile.json`. Re-check that
  the box is reachable and `torch.cuda.is_available()` (the box changes between runs). Read the full
  `results.tsv`, `research.md` and `ceiling.tsv`, then say in one short summary where the run stands:
  best experiment, last experiment, and what was queued next. Before the next probe, check the
  ceiling with `python -m frame.ceiling --config <candidate> --run runs/<run>` (`program.md` Step 4):
  it re-measures only if a memory knob moved. A `profile.json` from before 2026-10-02 stamps only
  `dtype`, so it reads as stale once and gets one re-measure. Continue
  Step 4, numbering experiments after the last row. If `run.toml` no longer matches the run's
  `profile.json` on model, dataset, `max_len` or `effective_batch`, stop and say so. That is a new
  run, not a resume.
- **`review`:** Step 5 only. Never start a probe.

## Examples

```
/run-loop smollm2_byo model=HuggingFaceTB/SmolLM2-360M host=ubuntu@gpu-box gpu=A10 script=examples/byo_script/trainer_yaml.py script_configs=examples/byo_script/train.yaml
/run-loop smollm2_byo_flags model=HuggingFaceTB/SmolLM2-360M local gpu=A10 script=examples/byo_script/trainer_flags.py
```

```
/run-loop qwen3.5_2b_a100_wikitext model=Qwen/Qwen3.5-2B dataset=Salesforce/wikitext:wikitext-2-raw-v1 max_len=1024 effective_batch=16 host=ubuntu@gpu-box gpu=A100
/run-loop qwen3.5_2b_a100_wikitext resume
/run-loop smollm2_1.7b_a100_wikitext review
/run-loop smoke_test model=Qwen/Qwen2.5-0.5B local smoke
/run-loop                     # everything from run.toml; asks only for the run name
```
