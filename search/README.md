# search/ &mdash; *(scaffold, not yet built)*

The loop's decision-support machinery — the tools the agent wields to hypothesize and learn.

- **surrogate** — a calibrated analytical model (roofline; candidate: Calculon) that predicts memory/throughput, so most candidates are ranked without spending a probe
- **frontier** — a population of role-diverse elites (fastest / leanest / most-different) with crossover, instead of a single best guess
- **memory** — plumbing to read/write the belief store in `knowledge/`

The judgment lives in `loop/loop.md`; this is the code it calls.
