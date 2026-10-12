# beliefs/ &mdash; the belief store

Cross-run memory, mirroring how Claude's own memory works: one **belief** per markdown file with
frontmatter (the `Belief` contract's fields: `claim`, `context`, `type`, `evidence`, `confidence`,
`links`, `superseded`), plus an always-loaded index (`index.md`).

Each belief is an evidence-backed claim keyed by context (`{arch_family, hardware_class, dataset_shape}`)
so the next run retrieves what's relevant and starts smart. Authored and curated by the agent.

See the `Belief` contract in `loop/contracts.md`. **Live** — populated from the A10 runs of
2026-10-01..09 (`experience/index.md`).

---

## Two stores, two jobs

They are not two copies of the same memory. They answer different questions, are read at different
times, and have **opposite** mutation rules.

| | `experience/` | `beliefs/` (here) |
|---|---|---|
| Unit | one dated entry per **run** | one file per **claim** |
| Answers | *has this configuration been run before?* | *what do we believe about this lever?* |
| Read | **ONCE, at Frame** (`loop/loop.md`, Frame item 3), via `experience/index.md` | every hypothesis, via `index.md`, filtered by `context` match |
| Mutation | **append-only.** Never edit a past entry — it is dated evidence | **curated** at the end of each run: re-confirm, narrow and split, supersede, delete — never overwrite |

**`experience/` — the run log.** One file per run plus an index of their `Context:` lines,
the match key Frame searches. See `experience/README.md`.

**`beliefs/` — the belief store.** Context-keyed, so a new configuration retrieves only what
transfers. This is what keeps memory from becoming noise as the log grows.

**Promotion copies, it does not move.** A durable claim gets extracted from a run entry into a
belief file; the run entry stays intact and cross-references it. Gutting a past entry would falsify the record of what was believed when.

## Curating beliefs (`program.md` Step 5, second half)

First write the run's entry in `experience/` (`experience/README.md`). That is only half of Step
5. **Re-read every belief whose `context` matches this
run** and update it, or beliefs only ever get created and `probes: 1` stays `probes: 1` forever.
For each match:

| What the run did to it | What to do |
|---|---|
| **Re-confirmed** it | bump `evidence.probes`, raise `confidence`, add this run to the evidence line. This is the whole point — a claim behind 7 runs on 4 card types is a different object from the same sentence behind 1 |
| **Contradicted** it | **NEVER overwrite.** Narrow its `context` to where it still holds and write a second belief for the new regime, `links:` between them. "True on Ampere, false on Hopper" is two beliefs, not an argument |
| **Superseded** it | keep it, mark it superseded, link forward — a reader needs to know the old claim existed and why it was replaced |
| **Proved it wrong** outright | delete the file, drop its line from `index.md`, and say so in the run entry. A wrong prior is worse than none: it spends probes in the wrong direction |
| Didn't touch it | leave it alone |

Only then write beliefs for what's genuinely new. A belief that merely restates an existing one is a
duplicate — strengthen the original instead.

**Gate every promotion, new beliefs and re-confirmations alike, on the evidence basis**
(Measurement axiom, `loop/contracts.md` corollary 7). Each comparison behind the claim must be one
of:

| Basis | Valid for | Check in the Observations |
|---|---|---|
| `same-rows` | throughput, step time | `population.sample_hash` matches and `clean_passes` is whole |
| `same-reference-batch` | trace durations: busy time, phase times, kernel times | `population.reference.batch_key` matches |
| `width-invariant` | counts and provenance: kernel/gap count, which kernels ran, dtypes | any batch |

Rates must come from totals (mean tokens over mean step time), never from a median. A comparison
that meets none of these stays in the run's `experience/` entry as an observation. It does not
become belief evidence and does not bump `probes`. Write the basis into the belief's `evidence`
line. If a run changed its measurement method partway through, every claim measured under the old
method is re-probed or flagged before it is promoted: a result reached with a biased instrument is
not a result.

This is the difference between memory and a log. `experience/` accumulates and is never edited;
`beliefs/` is *revised* — that's what makes an opinion worth holding.
