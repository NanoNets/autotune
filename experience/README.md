# experience/ &mdash; the run log

What actually happened: one dated entry per **run**, across all runs, one file each
(`experience/<run_name>.md`), plus an always-read index (`index.md`). Its partner is `beliefs/`,
which holds what we believe; see `beliefs/README.md` for how the two differ.

- **Answers:** *has this configuration been run before?*
- **Read:** **once, at Frame** (`loop/loop.md` §3). Read `index.md`, match on each `Context:` line,
  and open only the entries that match or nearly match. Not per hypothesis: the log grows with every
  run, and "have we run this before" is a framing question, asked once. Re-open an entry only when
  a result contradicts the prior Frame matched.
- **Mutation:** **append-only.** Never edit a past entry; it is dated evidence. A new run adds a
  new file and a new index line.

A hit hands the next run a winning config to replay as experiment 2, and a list of dead levers not
to re-probe. A near-miss is still a prior: name the dimension that differs and what it predicts.

An entry is also where a new model's architecture summary gets captured the first time this repo
sees it (`program.md` Step 2 / `frame/README.md`), so the next run starts from an answer, not from
scratch.

## Writing an entry (`program.md` Step 5, first half)

When the loop ends (interrupted by the human, or converged), stop and do one more pass before
handing back: review the *entire* `results.tsv` **and** `research.md` for that run as a whole
(see `runs/README.md`), not just the last few rows or the most recent search. Pull out what's
durable, the things worth knowing before the *next* run starts, on this codebase or a different
model/hardware/dataset:

- **Findings:** what actually moved throughput, what didn't, ceilings you hit (memory, comm,
  compute-bound), levers that turned out to be dead on this stack.
- **Surprises:** anything that contradicted the going-in expectation, or contradicted a prior
  entry here (note the contradiction explicitly rather than silently overwriting).
- **Working hypotheses:** things you didn't get to test, or tested inconclusively, that are worth
  trying first next time.

Write it to `experience/<run_name>.md`, then **append its line to `index.md`**. The `Context:` line
is not decoration: it is the key Frame matches the *next* run against, so write every field, and
copy it verbatim into the index line. Write the winning config in full: a match replays it as that
run's experiment 2.

Entry (`experience/<run_name>.md`):

```markdown
## <run_name> — <date>
Context: <model> · <dataset> · <hardware> · <dtype> · effective_batch <n> · max_len <n>
Baseline: <tokens/s or samples/s>  →  Best: <tokens/s or samples/s>  (+X%)
Winning config: <the knob assignments, in full — a match replays this as experiment 2>

Findings:
- ...

Surprises / contradictions:
- ...

Open hypotheses for next run:
- ...
```

Index line (`index.md`):

```markdown
- [<run_name>](<run_name>.md) — <date>
  Context: <the entry's Context line, verbatim, on one line>
  Baseline: <...> → Best: <...> (+X%)
```

Keep entries short and empirical: claims backed by what's actually in `results.tsv`, not vibes.
This is what makes the *next* autonomous run start smarter instead of re-discovering the same dead
ends. A comparison that doesn't meet the evidence gate (`beliefs/README.md`) stays here as an
observation; it never becomes belief evidence.

From `research.md`, promote only what's **net-new**: not already covered by an existing entry or
belief. A websearch that just re-confirmed something already known isn't worth a duplicate entry;
leave it in that run's `research.md`.

Then curate the beliefs (`beliefs/README.md`). Writing the entry is only half of Step 5.
