# knowledge/ &mdash; the memory store

Cross-run memory, mirroring how Claude's own memory works: one **belief** per markdown file with
frontmatter, plus an always-loaded index (`beliefs.md`).

Each belief is an evidence-backed claim keyed by context (`{arch_family, hardware_class, dataset_shape}`)
so the next run retrieves what's relevant and starts smart. Authored and curated by the agent.

See the `Belief` contract in `loop/contracts.md`. *(Convention defined; population begins at the intelligence stage.)*

---

## `knowledge.md` — the current running log

Until the `Belief` system above is populated, `knowledge.md` at repo root is the whole memory: one
lightweight, always-read, dated running log across *all* runs (not per-run). Read it — along with
this folder's `beliefs.md` and any per-belief files — before every hypothesis in the loop (see
`loop/loop.md`, Hypothesize): a matching entry is a strong prior to build on or deliberately test
against, not background color.

It's also where a new model's architecture summary gets captured the first time this repo sees it
(`program.md` Step 2 / `frame/README.md`) — worth a durable entry so the next run starts from an
answer, not from scratch.

As the `Belief` system comes online, durable claims from `knowledge.md` should get promoted into
proper `Belief` files here.

## Setup checklist (`program.md` Step 5 — end-of-run review)

When the loop ends (interrupted by the human, or converged), stop and do one more pass before
handing back: review the *entire* `results.tsv` **and** `research.md` for that run as a whole
(see `runs/README.md`), not just the last few rows or the most recent search. Pull out what's
durable — the things worth knowing before the *next* run starts, on this codebase or a different
model/hardware/dataset:

- **Findings** — what actually moved throughput, what didn't, ceilings you hit (memory, comm,
  compute-bound), levers that turned out to be dead on this stack.
- **Surprises** — anything that contradicted the going-in expectation, or contradicted a prior
  entry in `knowledge.md` (note the contradiction explicitly rather than silently overwriting).
- **Working hypotheses** — things you didn't get to test, or tested inconclusively, that are worth
  trying first next time.

Append (don't overwrite) a dated entry to `knowledge.md`. Suggested shape per entry:

```markdown
## <run_name> — <date>
Context: <model> · <dataset> · <hardware>
Baseline: <tokens/s or samples/s>  →  Best: <tokens/s or samples/s>  (+X%)

Findings:
- ...

Surprises / contradictions:
- ...

Open hypotheses for next run:
- ...
```

Keep entries short and empirical — claims backed by what's actually in `results.tsv`, not vibes.
This is what makes the *next* autonomous run start smarter instead of re-discovering the same dead
ends.

From `research.md`, promote only what's **net-new** — not already covered by an existing
`knowledge.md` entry or belief. A websearch that just re-confirmed something already known isn't
worth a duplicate entry; leave it in that run's `research.md` and move on.
