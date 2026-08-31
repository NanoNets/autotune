# The Loop

Find the fastest training config for the given hardware, without degrading quality.
Method: the scientific method, run by an agent, on real hardware.

## Principles

- **The agent decides; code executes.** Mechanical work (run a probe, detect steady-state, compute physics) is delegated to code. Every judgment stays with the agent.
- **Nothing is hardcoded.** Priors are starting points the agent overrides. Knowledge is derived at Frame, computed by physics, or learned from probes — never baked in.
- **Every number comes from a real probe**, verified to have actually applied.
- **Quality is never touched.** Only knobs that change speed, not the model, are searchable.

## Actors

- **Human** — gives the goal: model, dataset, hardware, effective_batch. Nothing else.
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
2. **Pull priors** — retrieve Beliefs matching this profile; research external priors (model card, tuning guides).
3. **Build the SearchSpace** — project the canonical knobs onto what this framework/model/hardware supports. Classify each knob `safe`/`off_limits` (unsure → off_limits).
4. **Seed** — the first Config is the training script's **default, as-is** settings, unmodified —
   not a surrogate guess. It's the baseline every later Config is measured against (so the eventual
   "faster by X%" claim is against real shipped defaults), and the surrogate has no calibration
   data yet to seed anything with. The surrogate starts ranking candidates from the *second*
   Config onward (Hypothesize).



## Hypothesize

Propose the next Config.

- Reason from the Profile, the frontier, and Beliefs.
- Use the **surrogate** for the numeric search — it ranks candidates cheaply. Don't hand-search a high-dim space; let the surrogate propose, then judge.
- Keep a **frontier**, not one best: roles — best (throughput), lean (memory), diverse (different strategy). **Cross over** complementary winners.
- Every Config carries a rationale and an expected outcome.



## Test

Run the Config — the cheapest test that answers the question.

1. **Analytical** (free): predict via the surrogate. Kill predicted-OOM or predicted-worse-than-frontier configs here.
2. **Probe** (seconds): run on real hardware. Cost-capped and adaptive — stop when the reading is steady within the noise floor, or the cost cap is hit. The agent sets the cap.
3. **Extended probe** (minutes, champion only): confirm the number holds over time (fragmentation, drift, periodic stalls). Never a full training run.



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