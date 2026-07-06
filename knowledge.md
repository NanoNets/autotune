# Knowledge base — what we've learned across runs

The loop's **durable, cross-run memory** (above per-run `search_state.json`). The agent READS this at
run start and CURATES it at run end (see `program.md`). It is curated ground-truth: writes are gated
and rare — only confirmed findings, each tagged with the context it holds in.

**Entry schema** (one `##` per claim, grouped by topic):

- **Claim** — the finding, with magnitude.
- **Context** — *required.* The setup it was learned in (model, GPU, max_length, data shape, regime).
  An entry without context is unusable — you can't tell if it applies.
- **Why** — the mechanism, when known.
- **Scope** — `invariant` (a mechanism/direction that holds broadly; only magnitude is context-specific)
  or `contextual` (true in that regime; a HYPOTHESIS to re-test elsewhere).
- **Evidence** — run date + probe(s).

**How to use it (summary; rules in `program.md`):** invariant → apply as a prior. contextual +
context matches → strong prior. contextual + context differs → don't assume, test it as a hypothesis.

---