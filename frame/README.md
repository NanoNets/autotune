# frame/ &mdash; *(scaffold, not yet built)*

Frame runs **once** at the start of a run: profile the setup, then construct the search space.

- profile the **model** (internals from the HF config), **hardware** (topology, versions, roofline specs), and **dataset** (length distribution, volume)
- build the **SearchSpace** — project the canonical levers onto what this framework/model/hardware supports, and classify each `safe` / `off_limits`
- pull **priors** from `knowledge/` (past runs) and external sources

Produces the `Profile` and `SearchSpace` contracts — see `loop/contracts.md`.
