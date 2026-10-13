# The Loop

Find the fastest training config for the given hardware, without degrading quality.
Method: the scientific method, run by an agent, on real hardware.

## Principles

- **The agent decides; code executes.** Mechanical work (run a probe, compute physics) is delegated to code. Every judgment stays with the agent.
- **Nothing is hardcoded.** Priors are starting points the agent overrides. Knowledge is derived at Frame, computed by physics, or learned from probes — never baked in.
- **Every number comes from a real probe**, verified to have actually applied.
- **Quality is never touched.** Only knobs that change speed, not the model, are searchable.

## Actors

- **Human** — gives the goal: model, dataset, hardware, effective_batch, max_len, and optionally
  their own training script (`program.md` "Script mode"). Nothing else. Interrupting the run is
  theirs too.
- **Agent** — frames, hypothesizes, learns, judges convergence and reports it. Never stops on its
  own (`program.md`, "NEVER stop").
- **Hands (code)** — runs probes, measures, persists state.



## The cycle

```
FRAME (once) → [ HYPOTHESIZE → TEST → OBSERVE → LEARN ] → repeat → OUTPUT
```

---



## Frame (once)

Understand the situation, then set up the search.

1. **Profile** the model, hardware, environment, dataset (→ `Profile`, saved to
   `runs/<run>/profile.json`). Measure the card's peak FLOP/s and bandwidth (the roofline).
2. **Measure the ceiling** — the largest `micro_batch` that runs at `max_len`, on manufactured
   worst-case batches, not on whatever the dataset happened to supply (`frame/README.md`). It
   bounds the `micro_batch` knob in step 3. IF `micro_batch=1` OOMs → memory regime: recompute /
   offload / sharding come first, batch knobs are not the lever.
3. **Pull priors** — two questions, two stores. Experience is read HERE and not again per
   hypothesis; Beliefs are retrieved here and re-read before every hypothesis (`program.md` Step 4):
   - **Has this configuration been run before?** Match this Profile against each `Context:` line
     in `experience/index.md` (model · dataset · hardware · dtype · `effective_batch` · `max_len`),
     then open only the entries that match or nearly match.
     A hit hands you that run's winning config, its dead levers, and its open hypotheses. A
     near-miss is still a prior — name the dimension that differs and what it predicts (same model,
     different card → the ceiling moves, the padding story doesn't).
   - **What do we believe about the levers?** Retrieve Beliefs whose `context` matches
     (`beliefs/index.md`); research external priors (model card, tuning guides).
4. **Build the SearchSpace** — project the canonical knobs onto what this framework/model/hardware supports. Classify each knob `safe`/`off_limits` (unsure → off_limits).
5. **Seed** — the first Config is the training script's **default, as-is** settings, unmodified.
   When the user brought their own script (`run.toml` `script`), "as-is" means their script and
   config files exactly as given, or the verified copy from `program.md` "Script mode" Step 2c.
   Not a surrogate guess, and **not a prior's winning config**. It is the baseline every later
   Config is measured against, so the eventual "faster by X%" claim is against real shipped
   defaults.
   **ALWAYS run the baseline first, whatever step 3 found.** A prior that starts the run has
   nothing to quantify itself against.
6. **Replay the prior** — IF step 3 matched a previous run, run ITS winning config as experiment 2,
   immediately after the baseline. That re-validates the prior on this hardware and banks the
   biggest known jump before the search starts; if it does NOT reproduce, that contradiction is
   itself the most valuable thing the run has learned so far (→ Learn). No match → skip.
   The search proper starts from the first Config that is neither the baseline nor a replay.



## Hypothesize

Propose the next Config.

- Reason from the Profile, the run's `results.tsv`, and Beliefs.
- Rank candidates yourself: predict each one's effect from Beliefs and the Profile, and spend the probe on the best prediction.
- Keep more than one best in view, not just the fastest: best (throughput), lean (memory), diverse (different strategy), noted in `research.md`. **Cross over** complementary winners.
- Every Config carries a rationale and an expected outcome.
- **Check the ceiling is still valid** before proposing a `micro_batch`. It was measured under one
  set of memory knobs (recompute, attention, optimizer, dtype, sharding, offload, packing, compile,
  chunked CE, the allocator setting) and is
  stale the moment one moves. Stale → re-measure from the previous ceiling
  (`python -m frame.ceiling`, `frame/README.md`), then hypothesize against the new bound. A ceiling that widens on
  `recompute` is a *result*, not an inconvenience.



## Test

Run the Config — the cheapest test that answers the question.

1. **Analytical** (free): predict from the Profile and Beliefs. Drop predicted-OOM or predicted-worse-than-best configs here. `micro_batch` above the measured ceiling is killed on evidence, not on a prediction.
2. **Probe** (~5 min, `program.md`): run on real hardware. A fixed warmup → clean window → trace split, the same for every config, so probes rank against each other. Cost-capped by the probe config's `timeout`.
3. **Extended probe** (longer, champion only): confirm the number holds over time (fragmentation, drift, periodic stalls). Never a full training run. The ceiling is measured at a fixed shape and so does NOT predict fragmentation from varying real lengths — this is where that shows up.

**Ceiling probes are not experiments.** They run on worst-case batches (synthetic for our runner, `ceiling_rows.jsonl` for a user script) and answer "does it fit", never "how fast". Frame's initial ceiling is in `profile.json`'s `ceiling`; re-measures by `python -m frame.ceiling` go to `ceiling.tsv`. Only real-data probes are scored or ranked.



## Observe

Read the `Observation` — raw and rich. Interpret it yourself; nothing is pre-labeled.

- Rank on samples/s. Read MFU as context — how far the step is from the hardware limit — never as
  the ranking key (`loop/throughput.md`).
- Check **provenance**: did the config apply? did the kernel engage? A number that didn't apply is worse than no number.



## Learn

Fold the result into the run's records. Beliefs are curated at the end of the run, not here.

- **Calibrate** — compare the result with the outcome you predicted; the miss is what to learn from.
- **Record** — the `conclusion` in `results.tsv` (kept or dropped, and why). IF the result says anything about a Belief, add or update its line in `research.md`'s `## Working beliefs`: the belief's name or `[new]` · re-confirmed / contradicted / new · the experiment numbers compared · the evidence basis (`runs/README.md`). The next hypothesis reads it; the end-of-run curation promotes from it (`program.md` Step 5, `beliefs/README.md`: re-confirm, narrow and split on contradiction, mark superseded — never overwrite).
- **Update the best-so-far** — the best, lean and diverse configs in `research.md`.
- **Judge convergence** — when the best-so-far stops moving on samples/s, say so in `research.md` and the final report, then dig deeper rather than stop: the loop runs until the human interrupts it. MFU says how much room was left; it is not a convergence rule. No fixed plateau rule.



## Output

The best config + the explanation: regime, ceilings, what's dead, samples/s baseline → best, and MFU reached as context.