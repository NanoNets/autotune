# Program

The operational playbook for running an autonomous throughput-tuning session in this repo. It
**routes** — each step below points at the md that owns the actual procedure, so this file stays
the map, not the territory. `loop/` (`contracts.md`, `loop.md`, `throughput.md`) is the design
reference this is built from — read it for the underlying model.

**Goal:** maximize training throughput (tokens/s and/or samples/s) on the given hardware, without
degrading model quality.

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

- IF Frame matched a previous run in `knowledge.md` (`loop/loop.md` Frame §3) → run that run's
  winning config as **experiment 2**, right after the baseline. It re-validates the prior on this
  hardware and banks the biggest known jump before searching. IF it fails to reproduce → that
  contradiction outranks any hypothesis you had queued; chase it.
- **ALWAYS**, before proposing a hypothesis: read `knowledge/beliefs.md` + the belief files whose
  context matches (`knowledge/README.md`), and this run's full `results.tsv` (`runs/README.md`).
  A probe costs ~5 min of real GPU time — think, don't guess.
- `knowledge.md` is read ONCE, at Frame — "has this configuration been run before" is a framing
  question, not a per-hypothesis one, and the file grows with every run. Re-open it only when a
  result contradicts the prior Frame matched.
- **ALWAYS** state the hypothesis's reasoning explicitly: what prior it's built on, what you
  expect, what would surprise you. Can't articulate why it should help → you're guessing → think
  more, don't spend the probe.
- **ALWAYS** check the ceiling before proposing a `micro_batch`: it is NOT a run constant. IF the
  Config moves a knob the ceiling was measured under (recompute, attn_kernel, optimizer, dtype,
  sharding, **compile**) → the ceiling is stale → re-measure it (`frame/README.md`), then
  hypothesize against the new bound. `compile` is in that list on measured evidence: fusing the
  cross-entropy cut peak memory 5.4GB and moved a measured ceiling from 2 to 4 (`knowledge.md`,
  test_a10G).
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

### Step 5 — End-of-run review → `knowledge/README.md`

- Review the full `results.tsv` + `research.md`.
- Append a dated entry to `knowledge.md`: findings, surprises, open hypotheses for next time.
- Promote only what's **net-new** — re-confirmed facts stay in `research.md`.
- **ALWAYS curate the existing beliefs, don't just write new ones** (`knowledge/README.md`). Re-read
  every belief whose context matches this run: re-confirmed → bump its evidence; contradicted →
  narrow its context and split, never overwrite; proven wrong → delete it and say so. Skipping this
  is how `probes: 1` stays `probes: 1` forever and beliefs stop being opinions worth holding.

---

## Script mode: the per-step differences when `run.toml` sets `script`

The user's training script is the thing being tuned. Treat it as theirs: **never edit its code.**
Everything you change reaches it through its own command-line flags or config file, via
`script_args`.

- **Step 1: read the script before anything else.** Find, and write down in
  `runs/<run_name>/research.md`:
  - the framework (HF Trainer, TRL, Lightning, a plain loop);
  - where its optimizer and DataLoader come from;
  - its seed;
  - which flag sets its **step budget** (e.g. `--max_steps`);
  - which flags or config keys control each speed knob in `loop/throughput.md`, and which ones
    must never change (learning rate, dataset, `max_len`, precision, LoRA rank, …).

  IF it has no torch DataLoader, no torch optimizer, or no step-budget flag → stop and tell the
  user; it can't be probed yet. Then run it once on the backend with the baseline `script_args`
  plus a small step budget, and confirm it exits cleanly.
- **Step 2: Frame the model and data it uses.** Set `model` / `dataset` in `run.toml` to what the
  script loads, and build the Profile as usual.
  - **The probe sample:** the script trains on its own data. Frame's `sample.jsonl` isn't a format a
    user's framework can load yet, so script probes don't carry `sample_hash`. Keep the script's
    data, seed and sampler identical across every probe.
  - **The ceiling:** there's no OOM ceiling for scripts yet (`design_improvements.md` §10). Don't
    raise the micro-batch above the baseline's value without the user's OK, and say so at Step 3.
- **Step 3: confirm the script contract with the user:**
  - the baseline command;
  - the step-budget flag;
  - the knob → flag mapping you found;
  - the flags you'll never touch;
  - that there's no ceiling.
- **Step 4: every probe is the script with changed `script_args`.**
  - The **baseline** is the script as-is.
  - For every probe, set the step budget to at least `warmup + clean_steps + 1` plus a little slack,
    and turn off logging, eval and checkpoint saving through the script's own flags, so no step pays
    for them.
  - A knob change is a flag change, applied in the script's own terms (grad accumulation is
    derived, so `effective_batch` stays fixed: `loop/throughput.md`).
  - Read `population.driver: "script"` on every Observation. Its `clean_window: "fixed-position"`
    means the rows timed are the loader's next steps, not whole passes over a shared sample. So:
    - **rank with care:** re-probe a would-be champion, interleaved with the incumbent, before
      keeping it;
    - **never compare a script probe's throughput with a runner probe's.**
  - `population.trace_basis: "traced-step"` means GPU-active / idle aren't reported. Read the
    phases and kernels instead.
- **Step 5: script probes are observations, not belief evidence.**
  - Their timings aren't `same-rows` and their trace durations aren't `same-reference-batch`
    (Measurement axiom, corollary 7), so record what they showed in `knowledge.md`.
  - Only **width-invariant** findings (kernel / gap counts, which kernels ran, dtypes) may update a
    belief.
  - Report the final config to the user as their script's own flags.

---

## Output Format

*(TBD — to be filled in later.)*
