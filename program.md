# Program

The operational playbook for running an autonomous throughput-tuning session in this repo. It
**routes** — each step below points at the md that owns the actual procedure, so this file stays
the map, not the territory. `loop/` (`contracts.md`, `loop.md`, `throughput.md`) is the design
reference this is built from — read it for the underlying model.

**Goal:** maximize **samples/s** (training examples per second) on the given hardware, without
degrading model quality. tokens/s is the cross-check, MFU a diagnostic (`loop/throughput.md`).

---

## Global rules

Apply these for the whole session, not just one step below.

- **Contracts:** Only tune knobs that change **speed**, never ones that change the **model**.
  Full safe/off-limits lever list: `loop/throughput.md`. Unsure whether a knob is safe → default
  to off-limits. Quality is never touched — this is what makes "faster" always a strict win, never
  a trade.
- **Simplicity:** Apply on every keep/discard decision. Small improvement + ugly complexity →
  discard. Equal-or-better result + less code → keep. (Karpathy's autoresearch heuristic — full
  quote below.)
- **NEVER stop:** Once Step 3 is confirmed, never pause to ask the human "should I keep going?" —
  run unattended until manually interrupted. The human may be away entirely and expects a full
  night of logged experiments (~5 min/experiment → ~12/hour → ~100/night).
- **Timeout:** Each experiment ≈5 min. If a run exceeds 10 min, kill it, log it as a failure,
  discard the change, move on.
- **Crashes:** IF easy fix (typo, missing import, off-by-one) → fix and re-run. IF the idea itself
  is fundamentally broken → don't try to rescue it — log `crash` as the status and move on.
- **One number, one population:** compare or divide two numbers only when they were measured on
  the same work by the same instrument (Measurement axiom, `loop/contracts.md`). Throughput
  compares across probes that timed the same rows (`population.sample_hash`, whole
  `clean_passes`). Trace durations compare only on the same reference batch
  (`population.reference.batch_key`). Kernel/gap counts and provenance compare across any batch.
  An Observation with `status: invalid` is a physically impossible reading: never score it,
  never explain it away. Stop and find the mixed population or the broken instrument first.

> All else being equal, simpler is better. A small improvement that adds ugly complexity is not
> worth it. Conversely, removing something and getting equal or better results is a great
> outcome — that's a simplification win. When evaluating whether to keep a change, weigh the
> complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20
> lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code?
> Definitely keep. An improvement of ~0 but much simpler code? Keep.

---

## Steps

Do these in order. Don't start experimenting until Step 3 is confirmed back to the user.

**IF `run.toml` sets `script`** (the user brought their own training script): follow each step
below as written, **plus** its "Script mode" line in the section after Step 5. Probes then run the
user's script unchanged instead of a `frameworks/` runner (`backends/README.md`, "Script probes").

### Step 1 — Environment & access → `backends/README.md`

- Identify local vs. remote (SSH). Credentials go in `.env`; non-secret config (model, dataset,
  gpu, `effective_batch`, `max_len`, backend, `device`, SSH host/user) goes in `run.toml` (copy from
  `run.toml.example`).
- Set `device` deliberately. `cpu` is a smoke test whose MFU is meaningless, never a result — a
  probe that falls back to it says so on stderr and in the Observation's `device_type`.
- Confirm the box is reachable AND that `torch.cuda.is_available()` is True before moving on.
- Determine the run name — ask the user; IF they decline → default to `<model_name>_<dataset>`
  (from `run.toml`, `/` sanitized to `-`). Create `runs/<run_name>/` now, so Step 2's research
  notes have somewhere to land.
- Skim the root `README.md` and `profiler/README.md`.

### Step 2 — Frame the job → `frame/README.md`

- Confirm the model is retrievable; build a real architecture picture.
- Confirm the dataset loads and matches the expected schema.
- **Draw the probe sample once, here:** `python -m frame ... --sample-out runs/<run_name>/sample.jsonl`.
  Frame draws it length-stratified from a random pool over the split. **Every probe of the run
  passes `sample_file: runs/<run_name>/sample.jsonl`**, so all probes time the same rows, and those
  rows carry the corpus's length distribution. Never time a probe on the head-slice fallback: its
  Observation says `population.sample_method: head`, and a result on it is not evidence.
- **Measure the OOM ceiling at `max_len`** before any tuning — `micro_batch` is bounded by
  measurement, never by guesswork. Procedure: `frame/README.md`.

### Step 3 — Initialize run tracking, then confirm → `runs/README.md`

- Create `runs/<run_name>/results.tsv` (the directory already exists from Step 1).
- Summarize environment, model, dataset, `effective_batch`, `max_len` + the measured ceiling,
  every Profile `note`, and the run name back to the user.
- IF the ceiling says `micro_batch=1` OOMs → say so here: it's a memory regime, and Step 4 starts
  from recompute/offload/sharding, not from batch knobs.
- Once confirmed → proceed to Step 4. **"NEVER stop" (Global rules) applies from here on.**

### Step 4 — Run the experimental loop → `loop/loop.md`

Cycle: `[ HYPOTHESIZE → TEST → OBSERVE → LEARN ] → repeat`. The first Config is the baseline —
the training script as-is, unmodified (see `loop/loop.md` Frame §5) — log it as experiment 1
before proposing any change. **NEVER** start from a prior's winning config: measured against
anything but the shipped defaults, "faster by X%" means nothing.

- IF Frame matched a previous run in `experience/` (`loop/loop.md` Frame §3) → run that run's
  winning config as **experiment 2**, right after the baseline. It re-validates the prior on this
  hardware and banks the biggest known jump before searching. IF it fails to reproduce → that
  contradiction outranks any hypothesis you had queued; chase it.
- **ALWAYS**, before proposing a hypothesis: read `beliefs/index.md` + the belief files whose
  context matches (`beliefs/README.md`), and this run's full `results.tsv` (`runs/README.md`).
  A probe costs ~5 min of real GPU time — think, don't guess.
- `experience/` is read ONCE, at Frame, through `experience/index.md`: open only the entries whose
  `Context:` line matches. "Has this configuration been run before" is a framing question, not a
  per-hypothesis one, and the log grows with every run. Re-open an entry only when a result
  contradicts the prior Frame matched.
- **ALWAYS** state the hypothesis's reasoning explicitly: what prior it's built on, what you
  expect, what would surprise you. Can't articulate why it should help → you're guessing → think
  more, don't spend the probe.
- **ALWAYS** check the ceiling before running a probe: it is NOT a run constant. Write the
  candidate config to `runs/<run_name>/candidate.json` and run
  `python -m frame.ceiling --config runs/<run_name>/candidate.json --run runs/<run_name>`. IF the
  config moves a memory knob the ceiling was stamped with (recompute, attention, optimizer, dtype,
  sharding, chunked CE, the allocator setting, **compile**) it re-measures from the previous
  ceiling, usually in 1-2 short trials; otherwise it runs nothing. Then choose the probe's
  `micro_batch` against the bound it reports (`frame/README.md`). `compile` is in that list on measured evidence: fusing the
  cross-entropy cut peak memory 5.4GB and moved a measured ceiling from 2 to 4 (`experience/test_a10G.md`).
- **NEVER** raise `micro_batch` above the ceiling because a probe reported spare `headroom_gb`.
  Under `padding="longest"` a probe's peak reflects the rows it drew, not the dataset's longest —
  the ceiling is the bound, the observed peak is not. Dynamic padding is safe *because*
  `micro_batch <= ceiling(max_len)`, not despite it.
- **NEVER** propose a `dtype` change to make something fit — precision changes the model. IF the
  memory regime blocks the run, put it to the human with the measured numbers (`loop/throughput.md`).
- **NEVER** log a ceiling trial as an experiment. Synthetic probes measure memory, not throughput
  — they go to `ceiling.tsv`, never `results.tsv` (`runs/README.md`).
- IF the change works → keep it, commit it (the branch now reflects the new best config).
  IF it doesn't → discard, move on.
- Rewind to an earlier point only very sparingly, if ever — prefer branching your thinking over
  branching git history.
- Out of ideas → dig deeper (re-read in-scope files, profiler output, websearch papers/model
  cards) rather than treat it as a stopping condition. Websearch findings go to
  `runs/<run_name>/research.md` (`runs/README.md`).

### Step 5 — End-of-run review → `experience/README.md`, then `beliefs/README.md`

- Review the full `results.tsv` + `research.md`.
- Write a dated entry to `experience/<run_name>.md` (findings, surprises, open hypotheses for next
  time) and append its line, `Context:` verbatim, to `experience/index.md` (`experience/README.md`).
- Promote only what's **net-new** — re-confirmed facts stay in `research.md`.
- **ALWAYS curate the existing beliefs, don't just write new ones** (`beliefs/README.md`). Re-read
  every belief whose context matches this run: re-confirmed → bump its evidence; contradicted →
  narrow its context and split, never overwrite; proven wrong → delete it and say so. Skipping this
  is how `probes: 1` stays `probes: 1` forever and beliefs stop being opinions worth holding.

---

## Script mode: the per-step differences when `run.toml` sets `script`

The user's training script, plus any `script_configs` it loads, is the thing being tuned. **The
user's own files are never modified.** Every change reaches the script through **its own** inputs:
- **its flags,** via `script_args`;
- **its config files,** via a per-probe **copy** the script is pointed at;
- **an edit,** only when the user allows one, on a **copy** of the script (Step 2c).

Everything you create lives under `runs/<run_name>/` and is passed to probes as `script_files`, so
the ssh backend ships it to the box (`backends/README.md`, "Script probes").

### Step 1: read the script and its configs, then run it once

Read `script` and every file in `script_configs`. Write down in `runs/<run_name>/research.md`:
- the framework (HF Trainer, TRL, Lightning, a plain loop);
- where its optimizer and DataLoader come from;
- its seed;
- **how each config file is loaded:** by a flag (e.g. `--config train.yaml`), a positional
  argument, or a **hardcoded path** in the code;
- **which flag or config key sets the step budget** (e.g. `--max_steps`, `training.max_steps`);
- **where its data source is set:** a flag (`--data_path`), a config key (`dataset.path`), a
  hardcoded path, or a hub id with no path setting (`load_dataset("org/name")`); and whether it
  reads text rows or pre-tokenized ones (a separate preprocessing step).

IF it has no torch DataLoader or no torch optimizer → stop and tell the user: it can't be probed
yet. Otherwise run it once on the backend with its baseline inputs and a small step budget, and
confirm it exits cleanly. (If there's no step budget, that's a Step 2b finding: ask before running
it for long.)

### Step 2: Frame, plus the knob audit

Frame the model and data the script uses as usual (`model` / `dataset` in `run.toml`).
- **LoRA / PEFT:** if the script trains adapters on a frozen base, pass its rank and target
  modules, as the script sets them, to Frame: `--lora-r <r> --lora-targets <names | all-linear>`
  (`frame/README.md`, "LoRA"). Without them, Frame's memory and FLOPs assume a full fine-tune and
  overstate this run several times over. They stay on the never-touch list (2a).
- **The probe sample:** the script trains on its own data. Frame's `sample.jsonl` isn't a format a
  user's framework can load yet, so script probes don't carry `sample_hash`. Keep the script's data,
  seed and sampler identical across every probe.
- **The ceiling** (`frame/README.md`, "User scripts"). Micro-batch is a lever for a script like
  for our runner, bounded by the ceiling; only `effective_batch` is fixed.
  1. Frame with `--ceiling-rows-out runs/<run_name>/ceiling_rows.jsonl` (plus
     `--ceiling-row-template` for chat / prompt data, or `--ceiling-rows-format ids` for
     pre-tokenized JSON; for a separate preprocessing step, run the user's own step on the rows).
  2. Write the trial config, `runs/<run_name>/ceiling_candidate.json`: the baseline probe config,
     with the script's data source pointed at the rows (a flag in `script_args`, or a config
     template in `ceiling_templates`), `{micro_batch}` where the micro-batch goes, accumulation 1,
     a step budget of at least 3, and `memory_knobs` (the script's current value for every memory
     knob, under its own names). A hardcoded data source or a hub id only is a 2b item.
  3. `python -m frame.ceiling --config runs/<run_name>/ceiling_candidate.json --run runs/<run_name>`.
     A trial that fails on the batch shape means the rows didn't reach the model as
     `[micro_batch, max_len]`: fix the pointer (or re-Frame at the script's real length), don't
     work around it. IF the user declines the 2b edit for a hardcoded data source → no ceiling:
     say so at Step 3, and keep the micro-batch at or below the baseline's unless the user OKs more.

**2a. The knob audit.** For every speed knob in `loop/throughput.md` (micro-batch / grad
accumulation, attention implementation, gradient checkpointing, compile, dataloader workers, …),
record in `research.md` **where it lives**:

| Where | Example | Tunable? |
|---|---|---|
| a flag | `--per_device_train_batch_size 8` | yes, via `script_args` |
| a config key in a file loaded **by flag** | `training.per_device_train_batch_size: 8` in `train.yaml`, loaded with `--config` | yes, via a per-probe copy (Step 4) |
| a config key in a file at a **hardcoded path** | the script always opens `train.yaml` | **not without an edit** (2b) |
| **hardcoded** in the code | `TrainingArguments(per_device_train_batch_size=8)` | **not without an edit** (2b) |
| absent | no way to set it | not tunable; note it |

Record the **data source** the same way: the ceiling trials point it at the ceiling rows.

Also record the **never-touch list:** learning rate, schedule, dataset, `max_len`, precision, seed,
LoRA rank, and anything else that changes the model (`loop/throughput.md`).

**2b. If anything needed is hardcoded, ask the user before Step 3.** That covers a speed knob, a
config file's path, the step budget, or the data source (a hardcoded path, or a hub id only). Ask once, listing every such item, and offer:
1. **Move them into a config file** (recommended): a copy of the script at `runs/<run_name>/train.py`
   that reads those values from `runs/<run_name>/train_config.yaml` (passed by a new flag), so every
   probe's settings live in a file on disk.
2. **Expose them as flags:** a copy of the script with new command-line flags for them.
3. **Leave them as they are:** tune only what's already reachable. The hardcoded knobs are out of
   the search, and you say which.

A missing step budget gets the same question: (1)/(2) add one, on a copy. Declining it means the
script **can't be probed** (a probe must stop after ~15 steps): stop and tell the user, before
running any probe.

**2c. Rules for any edit (options 1–2):**
- **Only with the user's explicit yes** to that option.
- **On a copy:** `runs/<run_name>/train.py` (and `train_config.yaml`). Never the user's files.
- **Move values, never change them.** The copy, run with the defaults you moved, must train
  exactly as the original did. No other code changes.
- **Record the diff** (original → copy) in `research.md`.
- **Prove equivalence before the baseline.** Run the original and the copy for the same few steps
  with logging every step, then compare.
  - **Pass:** the logged losses match at every step (within bf16 noise, ~1e-3 relative), and the
    step time is within noise.
  - **Fail:** discard the copy, and fall back to option 3 for those knobs (or stop, if the step
    budget was the problem).
- From then on, the copy **is** the script: `script` = `runs/<run_name>/train.py`, and it's listed
  in every probe's `script_files`.

### Step 3: confirm the script contract with the user

On top of the usual summary, report:
- the baseline command;
- the knob map (flag / config key / excluded) and the never-touch list;
- any edit you made, with its diff and the equivalence result;
- the knobs left out of the search;
- the measured ceiling, or that there's none (a hardcoded data source the user kept);
- that script probes are provisional evidence (Step 4).

### Step 4: every probe is the script with changed inputs

- **The baseline** is the script with its own defaults: the user's original, or the verified copy
  from 2c. Only the probe settings are added.
- **Probe settings, every probe:** a step budget of at least `warmup + clean_steps + 1` plus a
  little slack; logging, eval and checkpoint saving off. Use flags, or the config copy, whichever
  holds them.
- **Before a probe that moves a memory knob** (compile, checkpointing, attention, …), re-check the
  ceiling with the **trial form** of that probe: your `ceiling_candidate.json` (Step 2) with the
  same knob changed (its template or flag) and `memory_knobs` updated to the new values. Not the
  probe config itself: a trial reads the ceiling rows, at `{micro_batch}` and accumulation 1.
  A micro-batch / accumulation change alone needs no check.
- **A knob change:**
  - **a flag** → change it in `script_args`;
  - **a config key** → copy the config to `runs/<run_name>/probes/exp<N>.<ext>`, change **only**
    the knob keys, point the script at the copy through its own config flag, and list the copy in
    `script_files`. The user's config is never modified, and every probe's exact config stays on
    disk.

  Grad accumulation is derived, so `effective_batch` stays fixed (`loop/throughput.md`). In
  `results.tsv`'s hypothesis column, name the exact flags / keys changed.
- **Read `population.driver: "script"` on every Observation.** Its `clean_window:
  "fixed-position"` means the rows timed are the loader's next steps, not whole passes over a
  shared sample. So:
  - **rank with care:** re-probe a would-be champion, interleaved with the incumbent, before
    keeping it;
  - **never compare a script probe's throughput with a runner probe's.**
- `population.trace_basis: "traced-step"` means GPU-active / idle aren't reported. Read the phases
  and kernels instead.

### Step 5: script probes are observations, not belief evidence

- Their timings aren't `same-rows` and their trace durations aren't `same-reference-batch`
  (Measurement axiom, corollary 7), so record what they showed in the run's `experience/` entry.
- Only **width-invariant** findings (kernel / gap counts, which kernels ran, dtypes) may update a
  belief.
- **Report the winner in the user's own terms:** the flags to pass and/or a diff against their
  config file, plus any edit from 2c they'd need to adopt.

---

## Output Format

*(TBD — to be filled in later.)*
