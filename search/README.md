# search/ &mdash; *(scaffold, not yet built)*

The loop's decision-support machinery — the tools the agent wields to hypothesize and learn.

- **surrogate** — a calibrated analytical model (roofline; candidate: Calculon) that predicts memory/throughput, so most candidates are ranked without spending a probe
- **frontier** — a population of role-diverse elites (fastest / leanest / most-different) with crossover, instead of a single best guess
- **memory** — plumbing to read/write the belief store in `beliefs/`

The judgment lives in `loop/loop.md`; this is the code it would call. Until it's built, the agent
does this reasoning itself: it predicts from Beliefs and the Profile, and keeps its best-so-far set
in `results.tsv` / `research.md`.
