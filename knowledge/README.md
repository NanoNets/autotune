# knowledge/ &mdash; the memory store

Cross-run memory, mirroring how Claude's own memory works: one **belief** per markdown file with
frontmatter, plus an always-loaded index (`beliefs.md`).

Each belief is an evidence-backed claim keyed by context (`{arch_family, hardware_class, dataset_shape}`)
so the next run retrieves what's relevant and starts smart. Authored and curated by the agent.

See the `Belief` contract in `loop/contracts.md`. **Live** — populated from the `test_a10G` run.

---

## Two stores, two jobs

They are not two copies of the same memory. They answer different questions, are read at different
times, and have **opposite** mutation rules.

| | `knowledge.md` (repo root) | `knowledge/` (here) |
|---|---|---|
| Unit | one dated entry per **run** | one file per **claim** |
| Answers | *has this configuration been run before?* | *what do we believe about this lever?* |
| Read | **ONCE, at Frame** (`loop/loop.md` §3) | every hypothesis, filtered by `context` match |
| Mutation | **append-only.** Never edit a past entry — it is dated evidence | **curated.** Amend, supersede, delete as evidence lands |

**`knowledge.md` — the run log.** One dated entry per run, across *all* runs. Every entry opens
with a `Context:` line (model · dataset · hardware · dtype · `effective_batch` · `max_len`); that
line is the **match key** Frame searches. A hit hands the next run a winning config to replay as
experiment 2 and a list of dead levers not to re-probe. Read it at Frame, not per hypothesis — it
grows with every run and "have we run this before" is a framing question, asked once.

It's also where a new model's architecture summary gets captured the first time this repo sees it
(`program.md` Step 2 / `frame/README.md`) — so the next run starts from an answer, not from scratch.

**`knowledge/` — the belief store.** Context-keyed, so a new configuration retrieves only what
transfers. This is what keeps memory from becoming noise as the log grows.

**Promotion copies, it does not move.** A durable claim gets extracted from a run entry into a
belief file; the run entry stays intact and cross-references it (`knowledge.md` test_a10G does
this). Gutting a past entry would falsify the record of what was believed when.

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

Append (don't overwrite) a dated entry to `knowledge.md`. The `Context:` line is not decoration —
it is the key Frame matches the *next* run against (`loop/loop.md` §3), so write every field, and
write the winning config in full: a match replays it as that run's experiment 2. Shape per entry:

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

Keep entries short and empirical — claims backed by what's actually in `results.tsv`, not vibes.
This is what makes the *next* autonomous run start smarter instead of re-discovering the same dead
ends.

### Then curate the beliefs — they are supposed to change

Writing the run entry is only half of Step 5. **Re-read every belief whose `context` matches this
run** and update it, or beliefs only ever get created and `probes: 1` stays `probes: 1` forever.
For each match:

| What the run did to it | What to do |
|---|---|
| **Re-confirmed** it | bump `evidence.probes`, raise `confidence`, add this run to the evidence line. This is the whole point — a claim behind 7 runs on 4 card types is a different object from the same sentence behind 1 |
| **Contradicted** it | **NEVER overwrite.** Narrow its `context` to where it still holds and write a second belief for the new regime, `links:` between them. "True on Ampere, false on Hopper" is two beliefs, not an argument |
| **Superseded** it | keep it, mark it superseded, link forward — a reader needs to know the old claim existed and why it was replaced |
| **Proved it wrong** outright | delete the file, drop its line from `beliefs.md`, and say so in the run entry. A wrong prior is worse than none: it spends probes in the wrong direction |
| Didn't touch it | leave it alone |

Only then write beliefs for what's genuinely new. A belief that merely restates an existing one is a
duplicate — strengthen the original instead.

This is the difference between memory and a log. `knowledge.md` accumulates and is never edited;
`knowledge/` is *revised* — that's what makes an opinion worth holding.

From `research.md`, promote only what's **net-new** — not already covered by an existing
`knowledge.md` entry or belief. A websearch that just re-confirmed something already known isn't
worth a duplicate entry; leave it in that run's `research.md` and move on.
