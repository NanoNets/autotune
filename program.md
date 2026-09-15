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

> All else being equal, simpler is better. A small improvement that adds ugly complexity is not
> worth it. Conversely, removing something and getting equal or better results is a great
> outcome — that's a simplification win. When evaluating whether to keep a change, weigh the
> complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20
> lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code?
> Definitely keep. An improvement of ~0 but much simpler code? Keep.

---

## Steps

Do these in order. Don't start experimenting until Step 3 is confirmed back to the user.

### Step 1 — Environment & access → `backends/README.md`

- Identify local vs. remote (SSH). Credentials go in `.env`; non-secret config (model, dataset,
  gpu, `effective_batch`, `max_len`, backend, SSH host/user) goes in `run.toml` (copy from
  `run.toml.example`).
- Confirm the box is reachable before moving on.
- Determine the run name — ask the user; IF they decline → default to `<model_name>_<dataset>`
  (from `run.toml`, `/` sanitized to `-`). Create `runs/<run_name>/` now, so Step 2's research
  notes have somewhere to land.
- Skim the root `README.md` and `profiler/README.md`.

### Step 2 — Frame the job → `frame/README.md`

- Confirm the model is retrievable; build a real architecture picture.
- Confirm the dataset loads and matches the expected schema.
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
- IF `padding="longest"` → **check the loader's `sample` is representative before trusting any
  throughput number.** Batch width is the rows drawn, so compare the sample's length distribution
  against frame's scan and raise `sample` until the tail agrees (`frameworks/README.md`). A speedup
  measured on an unrepresentative sample is an upper bound — log it as one.
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

## Output Format

*(TBD — to be filled in later.)*
