# The Loop

Find the fastest training config for the given hardware, without degrading quality.
Method: the scientific method, run by an agent, on real hardware.

## Principles

- **The agent decides; code executes.** Mechanical work (run a probe, detect steady-state, compute physics) is delegated to code. Every judgment stays with the agent.
- **Nothing is hardcoded.** Priors are starting points the agent overrides. Knowledge is derived at Frame, computed by physics, or learned from probes — never baked in.
- **Every number comes from a real probe**, verified to have actually applied.
- **Quality is never touched.** Only knobs that change speed, not the model, are searchable.

## Actors

- **Human** — gives the goal: model, dataset, hardware, effective_batch, max_len. Nothing else.
- **Agent** — frames, hypothesizes, learns, decides when to stop.
- **Hands (code)** — runs probes, measures, persists state.



## The cycle

```
FRAME (once) → [ HYPOTHESIZE → TEST → OBSERVE → LEARN ] → repeat → OUTPUT
```

---



## Frame (once)

Understand the situation, then set up the search.

1. **Profile** the model, hardware, environment, dataset (→ `Profile`). Compute the roofline.
2. **Measure the ceiling** — the largest `micro_batch` that runs at `max_len`, on manufactured
   worst-case batches, not on whatever the dataset happened to supply (`frame/README.md`). It
   bounds the `micro_batch` knob in step 3. IF `micro_batch=1` OOMs → memory regime: recompute /
   offload / sharding come first, batch knobs are not the lever.
3. **Pull priors** — two questions, two stores, both answered HERE and not again per hypothesis:
   - **Has this configuration been run before?** Match this Profile against each `knowledge.md`
     entry's `Context:` line (model · dataset · hardware · dtype · `effective_batch` · `max_len`).
     A hit hands you that run's winning config, its dead levers, and its open hypotheses. A
     near-miss is still a prior — name the dimension that differs and what it predicts (same model,
     different card → the ceiling moves, the padding story doesn't).
   - **What do we believe about the levers?** Retrieve Beliefs whose `context` matches
     (`knowledge/beliefs.md`); research external priors (model card, tuning guides).
4. **Build the SearchSpace** — project the canonical knobs onto what this framework/model/hardware supports. Classify each knob `safe`/`off_limits` (unsure → off_limits).
5. **Seed** — the first Config is the training script's **default, as-is** settings, unmodified.
   Not a surrogate guess, and **not a prior's winning config**. It is the baseline every later
   Config is measured against, so the eventual "faster by X%" claim is against real shipped
   defaults; and the surrogate has no calibration data yet to seed anything with.
   **ALWAYS run the baseline first, whatever step 3 found.** A prior that starts the run has
   nothing to quantify itself against.
6. **Replay the prior** — IF step 3 matched a previous run, run ITS winning config as experiment 2,
   immediately after the baseline. That re-validates the prior on this hardware and banks the
   biggest known jump before the search starts; if it does NOT reproduce, that contradiction is
   itself the most valuable thing the run has learned so far (→ Learn). No match → skip.
   The surrogate starts ranking from the first Config that is neither the baseline nor a replay.



## Hypothesize

Propose the next Config.

- Reason from the Profile, the frontier, and Beliefs.
- Use the **surrogate** for the numeric search — it ranks candidates cheaply. Don't hand-search a high-dim space; let the surrogate propose, then judge.
- Keep a **frontier**, not one best: roles — best (throughput), lean (memory), diverse (different strategy). **Cross over** complementary winners.
- Every Config carries a rationale and an expected outcome.
- **Check the ceiling is still valid** before proposing a `micro_batch`. It was measured under one
  set of memory knobs (recompute, attn_kernel, optimizer, dtype, sharding) and is stale the moment
  one moves. Stale → re-measure, then hypothesize against the new bound. A ceiling that widens on
  `recompute` is a *result*, not an inconvenience.



## Test

Run the Config — the cheapest test that answers the question.

1. **Analytical** (free): predict via the surrogate. Kill predicted-OOM or predicted-worse-than-frontier configs here. `micro_batch` above the measured ceiling is killed on evidence, not on a prediction.
2. **Probe** (seconds): run on real hardware. Cost-capped and adaptive — stop when the reading is steady within the noise floor, or the cost cap is hit. The agent sets the cap.
3. **Extended probe** (minutes, champion only): confirm the number holds over time (fragmentation, drift, periodic stalls). Never a full training run. The ceiling is measured at a fixed shape and so does NOT predict fragmentation from varying real lengths — this is where that shows up.

**Ceiling probes are not experiments.** They run on synthetic batches and answer "does it fit", never "how fast". They go to `ceiling.tsv`; only real-data probes are scored, ranked, or put on the frontier.



## Observe

Read the `Observation` — raw and rich. Interpret it yourself; nothing is pre-labeled.

- Ground the objective against the roofline (% of achievable), not in isolation.
- Check **provenance**: did the config apply? did the kernel engage? A number that didn't apply is worse than no number.



## Learn

Fold the result into knowledge.

- **Calibrate** the surrogate with the measured residual.
- **Update Beliefs** — write/curate claims (regime, ceilings, dead levers, calibration). Amend + link on contradiction; key by context.
- **Update the frontier** — promote, archive, log why each config was kept or dropped.
- **Decide convergence** — stop when the frontier stops producing and you're near the roofline, judged against the noise floor. Hard probe cap as backstop. No fixed plateau rule.



## Output

The best config + the explanation: regime, ceilings, what's dead, and % of roofline reached.