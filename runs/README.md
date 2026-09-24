# runs/ &mdash; per-run outputs *(gitignored)*

Each run gets a folder holding its journal (`journal.jsonl` — every config &rarr; observation), its
research notes (`research.md`), and the final best config. The contents are ignored by git; this
README is the only tracked file here.

---

## Setup checklist (`program.md` Step 3)

Before the loop starts, create `runs/<run_name>/results.tsv` with this header row:

```
experiment_number	token/s	samples/s	memory	status	hypothesis	conclusion
```

- `status` — e.g. `ok` / `oom` / `crash` / `error`.
- `hypothesis` — what you changed and why you expected it to help, written *before* running.
- `conclusion` — kept or discarded, and why, written *after* observing the result.
- A **worst-case probe** — the champion at `padding="max_length"` at `max_len`, the fit check
  `loop/throughput.md` requires — is a real probe and is logged here, with `WORST-CASE PROOF` in
  its conclusion. `progress.py` keeps it off the record staircase; it is never a champion candidate.

This file is the running record every hypothesis after the baseline gets built on — treat it as the
single source of truth for "what have we already tried." Every experiment (starting with the
as-is baseline, logged first, before any change is proposed) gets a row.

**Read it back before proposing a new experiment.** Look for patterns across the full history, not
just the last row (e.g. "everything with recompute=full has been worse") — this is what keeps the
loop from re-running something already tried.

## `ceiling.tsv` — the memory ledger, kept apart from the throughput one

Ceiling trials (`frame/README.md`) run on **synthetic** max_len batches and answer *does it fit*,
never *how fast*. Scoring them as experiments would rank throughput on fake data. They get their
own file:

```
seq_len	micro_batch	status	peak_gb	reserved_gb	config	note
```

- One row per trial, written as the search runs. `config` — the memory knobs it was measured under.
- **NEVER** put a ceiling trial in `results.tsv`, and never on the frontier.
- Re-run the search when a memory knob moves; append the new trials, don't overwrite the old ones —
  the two ceilings side by side are the evidence for what that knob bought.

## `progress.py` — the run's progress plot

```bash
python runs/progress.py [run_name]        # -> runs/<run_name>/progress.png
```

Reads `results.tsv` and draws every probe as a point with the monotone best-so-far as a staircase.
The staircase is the point: it only steps on an experiment that beat everything before it, so the
**flat stretches are where the search spent probes without buying anything** — which is the thing
worth looking at when deciding whether a line of attack is exhausted. Crashes and timeouts are
plotted too, below the baseline, so a run's failures stay visible rather than vanishing from the
record. A second panel tracks peak memory against the card limit on the same x-axis (its own panel,
never a second y-scale).

## `research.md` — the run's websearch landing zone

Anything found via websearch during this run (model card details, papers, tuning guides — see
`frame/README.md` and `program.md` Step 4) gets written here first, as it's found. No header, no
fixed shape — free-form notes, created on first write.

This file is *not* durable memory on its own. At Step 5's end-of-run review, it gets read
alongside `results.tsv` and anything net-new (not already covered by an existing `knowledge.md`
entry or belief) gets promoted into `knowledge.md` — see `knowledge/README.md`. A finding that just
re-confirms something already known stays here; it doesn't get re-added upstream. This keeps
`knowledge.md` a signal-dense run log instead of a transcript of every search made along the way.
(It is append-only — never edited once written. Curation happens in `knowledge/`, where beliefs are
amended and superseded; see `knowledge/README.md`.)
