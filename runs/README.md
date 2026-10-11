# runs/ &mdash; per-run outputs *(gitignored)*

Each run gets a folder, `runs/<run_name>/`. Its contents are ignored by git; this README and
`progress.py` are the only tracked files here. What a run's folder holds:

| File | Written by | What |
|---|---|---|
| `results.tsv` | the agent (Step 3 on) | one row per experiment — the throughput ledger, below |
| `research.md` | the agent | notes, websearch findings, the best-so-far, convergence calls |
| `profile.json` | the agent saves `python -m frame`'s printed Profile JSON (Step 2); `python -m frame.ceiling` updates its `ceiling` | read by `frame.ceiling`, `progress.py`, `/run-loop resume` |
| `sample.jsonl` | `python -m frame --sample-out` | the probe sample every probe passes as `sample_file` |
| `candidate.json` | the agent | the candidate probe config `frame.ceiling` checks |
| `ceiling.tsv` | `python -m frame.ceiling` | the memory ledger, below |
| `ceiling_rows.jsonl`, `ceiling_candidate.json` | Frame / the agent (script mode) | a user script's ceiling rows and trial config |
| `probes/` | the agent (script mode) | per-probe config copies |
| `progress.png` | `progress.py` (Step 5) | the progress plot |
| `annotations.json` | the agent, optional | plot captions and `records_from` for `progress.py` |

---

## Setup checklist (`program.md` Step 3)

Before the loop starts, create `runs/<run_name>/results.tsv` with this header row:

```
experiment_number	token/s	samples/s	memory	status	hypothesis	conclusion
```

- `samples/s` — **the ranked column**, the Observation's `samples_per_s`. Configs are compared on it and the progress staircase is
  drawn from it; `token/s` (the Observation's `tokens_per_s`) is recorded alongside to tell a real
  win from a sample artifact, never to rank (`loop/throughput.md`). Empty for a probe with no
  reading.
- `memory` — the Observation's `peak_mem_gb`, a number (a trailing `GB` is allowed).
- `status` — e.g. `ok` / `oom` / `crash` / `error`, or the Observation's `invalid`. Only an `ok` row
  can be the baseline or the best in `progress.py`.
- `hypothesis` — what you changed and why you expected it to help, written *before* running. Start
  it with a `[k=v k=v ...]` snapshot of the probe's knobs (e.g. `[micro_batch=8 compile=default]`):
  `progress.py` labels each staircase step with the knobs that changed from the previous best.
- `conclusion` — kept or discarded, and why, written *after* observing the result. `progress.py`
  reads two words in it: `CONTROL` (a repeat / isolation probe, drawn hollow) and `UNRELIABLE` (a
  reading the run rejected; it can't set a record).

This file is the running record every hypothesis after the baseline gets built on — treat it as the
single source of truth for "what have we already tried." Every experiment (starting with the
as-is baseline, logged first, before any change is proposed) gets a row.

**Read it back before proposing a new experiment.** Look for patterns across the full history, not
just the last row (e.g. "everything with recompute=full has been worse") — this is what keeps the
loop from re-running something already tried.

## `ceiling.tsv` — the memory ledger, kept apart from the throughput one

Ceiling trials (`frame/README.md`) run on worst-case `max_len` batches (synthetic for our runner,
Frame's `ceiling_rows.jsonl` for a user's script) and answer *does it fit*, never *how fast*.
Scoring them as experiments would rank throughput on fake data. Frame's initial ceiling (Step 2,
`python -m frame --ceiling`) is recorded in `profile.json`'s `ceiling`, not here. Every
`python -m frame.ceiling` run that measures appends its trials to their own file:

```
seq_len	micro_batch	status	peak_gb	reserved_gb	config	note
```

- One row per `frame.ceiling` trial. `config`: every memory knob it was measured under, defaults included.
  `note`: `CEILING` / `oom_at`, and `fresh` or `re-measure from <n>`.
- **NEVER** put a ceiling trial in `results.tsv`, and never rank one.
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
record. A row can't set a record (it's drawn, off the staircase) when its `status` isn't `ok`, when its
`conclusion` contains `UNRELIABLE` (a reading the run rejected), or when it predates the
`records_from` experiment number in `runs/<run>/annotations.json` (measured on a basis since
corrected). A second panel tracks peak memory against the card limit on the same x-axis (its own panel,
never a second y-scale).

## `research.md` — the run's websearch landing zone

Anything found via websearch during this run (model card details, papers, tuning guides — see
`frame/README.md` and `program.md` Step 4) gets written here first, as it's found. No header, no
fixed shape — free-form notes, created on first write.

This file is *not* durable memory on its own. At Step 5's end-of-run review, it gets read
alongside `results.tsv` and anything net-new (not already covered by an existing `experience/`
entry or belief) gets promoted into the run's `experience/` entry — see `experience/README.md`. A finding that just
re-confirms something already known stays here; it doesn't get re-added upstream. This keeps
`experience/` a signal-dense run log instead of a transcript of every search made along the way.
(It is append-only — never edited once written. Curation happens in `beliefs/`, where beliefs are
amended and superseded; see `beliefs/README.md`.)
