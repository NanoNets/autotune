# Program

The operational playbook for running an autonomous throughput-tuning session in this repo.
`loop/` (`contracts.md`, `loop.md`, `throughput.md`) is the design reference this is built from —
read it if you want the underlying model; this doc is what you actually execute.

Goal: **maximize training throughput (tokens/s and/or samples/s) on the given hardware, without
degrading model quality.** Everything below exists in service of that.

---

## 1. Setup

Do these in order. Don't start experimenting until all of them are confirmed.

### 1.1 Identify the environment
Ask the user where the experiments will actually run:
- **Local** — training runs on this machine.
- **VM / remote box** (Paperspace, Lambda, RunPod, your own cluster, ...) — you'll need SSH
  access. Ask for host/user, and get the credential via the env file, never inline in chat or
  in a tracked file:
  - `AUTOTUNE_SSH_PASSWORD` (password auth) or `AUTOTUNE_SSH_KEY` (path to a private key), set
    in `.env` (gitignored — see `.env.example` for the template).
  - Everything non-secret (backend, model, dataset, gpu, knobs, SSH host/user) belongs in
    `config.json` at repo root, not `.env`.
- Confirm you can actually reach the box (e.g. a trivial `ssh` command succeeds) before moving on.

### 1.2 Read the in-scope files
- `README.md` — repository context: what's built, what's scaffold, the overall plan.
- `profiler/` — the measurement instrument (roofline, trace parsing, capture, diagnostic). This
  is the thing that turns a run into numbers you can trust; understand what it measures and how
  before you start interpreting its output.

### 1.3 Verify the dataset
- **Local path** — confirm it exists, is readable, and matches the format the training script
  expects (schema, modality, on-disk layout). Don't assume; open a sample and check.
- **HuggingFace dataset** — confirm you can actually pull it (valid `HF_TOKEN` in `.env`,
  correct repo id/revision, gated-access resolved). A dataset you can't load is a setup failure,
  not an experiment result.

### 1.4 Initialize the run log
Create `runs/<run_name>/results.tsv` with the header row (see §5, Logging Results). This is the
running record every hypothesis after the baseline gets built on — treat it as the single source
of truth for "what have we already tried."

### 1.5 Confirm
Summarize the environment, dataset, and run name back to the user. Once confirmed, proceed to
Experimentation — and once the loop itself starts (§6), don't stop to check in again ("Never
stop", within §6).

---

## 2. Experimentation

### 2.1 Contracts — what you can and can't do
Only knobs that change **speed**, never ones that change the **model**, are searchable. When
unsure whether a knob is safe, default to off-limits.

**Safe** (change speed, not the model):
batch factorization (`micro_batch`/`grad_accum`), sharding/parallelism (dp, shard_degree/FSDP/
ZeRO, tp, pp, sequence_parallel, expert_parallel), recompute/activation offload, optimizer/param
offload, kernel swaps that are numerically-safe (attn kernel, fused kernels, fused optimizer,
chunked cross-entropy), `compile`, CUDA graphs, packing/padding-free/length-grouped batching,
dataloader tuning (workers, prefetch, num_proc), `tf32`.

**Off-limits** (change the model): learning rate, weight decay, warmup, dropout, optimizer
*algorithm*, precision used for training (bf16/fp16/fp8-train), `max_length`, seed, LoRA rank,
and the `effective_batch` target itself. These are human-owned; don't touch them even if they'd
help throughput.

Quality is never touched — that's what makes "faster" always a strict win, never a trade.

### 2.2 Optimization metric
Primary: **tokens/s** and/or **samples/s**. Report both where possible; cross-check one against
the other (a config that improves one but tanks the other is suspicious — figure out why before
trusting it).

### 2.3 Simplicity criterion
(Karpathy's autoresearch heuristic — apply it to every keep/discard decision.)

> All else being equal, simpler is better. A small improvement that adds ugly complexity is not
> worth it. Conversely, removing something and getting equal or better results is a great
> outcome — that's a simplification win. When evaluating whether to keep a change, weigh the
> complexity cost against the improvement magnitude. A 0.001 val_bpb improvement that adds 20
> lines of hacky code? Probably not worth it. A 0.001 val_bpb improvement from deleting code?
> Definitely keep. An improvement of ~0 but much simpler code? Keep.

---

## 3. First run — the baseline

Your very first run is always the training script **as-is**, unmodified. This is the number
every subsequent experiment is measured against. Log it as experiment 1 in `results.tsv` before
proposing any change.

---

## 4. Output Format

*(TBD — to be filled in later.)*

---

## 5. Logging Results

Every experiment gets a row in `runs/<run_name>/results.tsv`, tab-separated, header:

```
experiment_number	token/s	samples/s	memory	status	hypothesis	conclusion
```

- `status` — e.g. `ok` / `oom` / `crash` / `error`.
- `hypothesis` — what you changed and why you expected it to help, written *before* running.
- `conclusion` — kept or discarded, and why, written *after* observing the result.

This file is the reference the next hypothesis is built on — read it back before proposing a new
experiment so you don't re-run something already tried.

---

## 6. The Experimental Loop

Frame (environment, dataset, contracts) is already done by §1–2. From here it's the cycle from
`loop/loop.md`, adapted into an autonomous run:

```
[ HYPOTHESIZE → TEST → OBSERVE → LEARN ] → repeat
```

- **Hypothesize** — propose one change, reasoned from the current frontier and what's in
  `results.tsv` (and any beliefs in `knowledge/`, if populated). State the expected outcome.
- **Test** — run it (real hardware, not analytical guessing) for one experiment (see Timeout,
  below).
- **Observe** — read the raw numbers (tokens/s, samples/s, memory, status) against the baseline
  and the current best. Don't pre-label — interpret it yourself.
- **Learn** — log the row in `results.tsv`. Apply the simplicity criterion (§2.3) and decide:
  **keep or discard.**

You are a completely autonomous researcher. If a change works, keep it — commit it, and the
current branch now reflects the new best config. If it doesn't, discard it and move on to the
next hypothesis. This is how you advance: each kept experiment moves the branch forward, so
progress compounds instead of resetting every time.

If you feel like you're getting stuck (a string of discards, no clear direction, or you've
painted yourself into a corner), you *may* rewind to a earlier known-good point on the branch —
but do this **very sparingly, if ever**. Rewinding throws away information; prefer branching your
thinking (try a different lever) over branching your git history.

When you run out of ideas: think harder before concluding there's nothing left. Re-read the
in-scope files for angles you missed, look at what `profiler/` is actually telling you about
where time is going, look for referenced papers/model cards worth pulling from, and try
combining previous near-misses or making a more radical structural change. "Out of ideas" is a
prompt to dig deeper, not a stopping condition.

### Timeout
Each experiment should take **~5 minutes** total (plus a few seconds of startup/eval overhead).
If a run exceeds **10 minutes**, kill it, treat it as a failure, discard the change, and move on.

### Crashes
If a run crashes (OOM, bug, etc.), use judgment:
- **Something dumb and easy to fix** (typo, missing import, off-by-one) — fix it and re-run.
- **The idea itself is fundamentally broken** — don't try to rescue it. Log `crash` as the status
  in `results.tsv` and move on.

### Never stop
Once the loop has begun (after §1's setup is confirmed), **do not pause to ask the human whether
to continue.** No "should I keep going?", no "is this a good stopping point?" — the human may be
away entirely and expects this to run unattended. Keep going until manually interrupted.

As a reference point: at ~5 minutes per experiment you get roughly 12/hour — about 100 over a
typical night. The expectation is that a human can start this before going to sleep and wake up
to a full night of completed, logged experiments.

---

## 7. End-of-run review — `knowledge.md`

When the loop ends for a run (interrupted by the human, or converged per §6), stop and do one
more pass before handing back: review the *entire* `results.tsv` for that run as a whole, not
just the last few rows.

Pull out what's durable — the things worth knowing before the *next* run starts, on this
codebase or a different model/hardware/dataset:
- **Findings** — what actually moved throughput, what didn't, ceilings you hit (memory, comm,
  compute-bound), levers that turned out to be dead on this stack.
- **Surprises** — anything that contradicted the going-in expectation, or contradicted a prior
  entry in `knowledge.md` (note the contradiction explicitly rather than silently overwriting).
- **Working hypotheses** — things you didn't get to test, or tested inconclusively, that are
  worth trying first next time.

Append (don't overwrite) a dated entry to `knowledge.md` at repo root — one running history
across all runs, not per-run. Suggested shape per entry:

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

Keep entries short and empirical — claims backed by what's actually in `results.tsv`, not
vibes. This is the thing that makes the *next* autonomous run start smarter instead of
re-discovering the same dead ends.

`knowledge.md` is the lightweight, always-read running log. It's distinct from the more
structured `knowledge/` folder described under the `Belief` contract in §8 (one evidence-backed
claim per markdown file, keyed by context, for the future intelligence stage) — as that system
comes online, durable claims from `knowledge.md` should get promoted into proper `Belief` files
there. Until then, `knowledge.md` is the whole memory.

---

## 8. Misc

Everything from the original `loop/` docs not already folded into a section above — kept here
for reference.

### Actors (from `loop.md`)
- **Human** — gives the goal: model, dataset, hardware, effective_batch. Nothing else.
- **Agent** — frames, hypothesizes, learns, decides when to stop.
- **Hands (code)** — runs probes, measures, persists state.

### Core principles (from `loop.md`)
- **The agent decides; code executes.** Mechanical work (run a probe, detect steady-state,
  compute physics) is delegated to code. Every judgment stays with the agent.
- **Nothing is hardcoded.** Priors are starting points the agent overrides. Knowledge is derived
  at Frame, computed by physics, or learned from probes — never baked in.
- **Every number comes from a real probe**, verified to have actually applied.
- **Quality is never touched.** Only knobs that change speed, not the model, are searchable.

### Test rigor (from `loop.md`)
The cheapest test that answers the question, escalating only as needed:
1. **Analytical** (free) — predict via a surrogate/back-of-envelope math; kill predicted-OOM or
   predicted-worse-than-frontier configs before spending GPU time.
2. **Probe** (seconds) — run on real hardware, cost-capped and adaptive: stop when the reading is
   steady within the noise floor, or the cost cap is hit.
3. **Extended probe** (minutes, champion configs only) — confirm the number holds over time
   (fragmentation, drift, periodic stalls). Never a full training run.

### Provenance check (from `loop.md`)
When observing a result, check whether the config actually applied — did the kernel engage, or
did it silently fall back? A number that didn't apply is worse than no number at all.

### The five contracts (from `contracts.md`)
The data shapes the full design moves between steps (relevant mainly once `frame/`, `search/`,
and `knowledge/` are built out — right now the loop above is the practical subset):

- **Profile** — what's known about the situation before searching (model internals, hardware,
  env, dataset, roofline, the human-owned `effective_batch` constraint, priors). Built once.
- **SearchSpace** — what may be varied and the safety bounds: knobs (canonical names), a
  `safe`/`off_limits`/`unknown` classification per knob (default `unknown → off_limits`), the
  framework's actual capability, and the fixed/human-owned constants.
- **Config** — one hypothesis: knob assignments in canonical names, plus a rationale (parent(s),
  predicted outcome).
- **Observation** — everything measured from running a Config: status, objective (tokens/s +
  MFU), memory (peak/headroom, OOM breakdown), raw signals (gpu/cpu util, comm, data-wait —
  un-derived, the model interprets), provenance, cost, reliability (noise/variance), context
  (framework hints/log tail), and structured error on failure.
- **Belief** — one durable, evidence-backed claim, stored as a markdown file in `knowledge/`:
  claim, context key (`{arch_family, hardware_class, dataset_shape}`) for retrieval, evidence,
  type (regime/ceiling/calibration/dead-lever/safety/invariant), links to related beliefs.
  Authored and curated by the agent; injected at Frame by context match.

### Canonical levers (from `throughput.md`)
Framework-agnostic vocabulary; Frame projects these onto what the actual framework supports.

| Family | Levers |
|---|---|
| Batch | `micro_batch`, `grad_accum` (derived to hold effective_batch) |
| Parallelism | `dp`, `shard_degree` (FSDP/ZeRO 1-3), `tp`, `pp`, `sequence_parallel`, `expert_parallel`; mesh: dp×shard×tp×pp = world_size; comm overlap |
| Memory/recompute | `recompute` (full/selective), activation offload, optimizer/param offload, 8-bit optimizer states |
| Kernels | `attn_kernel` (flash/sdpa/FA3), fused kernels (Liger), fused optimizer, chunked cross-entropy |
| Compile | `compile` (off/default/max-autotune), CUDA graphs |
| Sequence | `packing`, `padding_free`, length-grouped batching |
| Dataloader | `workers`, `prefetch`, `num_proc` |
| Precision (safe subset) | `tf32` |

Applicability is model- and framework-dependent (Liger needs supported layers; FA3 needs
Hopper; TP needs the framework to expose it). Frame decides which apply; the surrogate predicts
effects; probes verify.

### Wall-clock framing (from `throughput.md`)
`wall_clock_for_a_fixed_run = total_tokens / tokens_per_s` — so higher throughput is just
directly shorter training. This is why tokens/s (reported as MFU, % of the hardware roofline) is
the primary metric and samples/s is the cross-check, not the other way around.

### Frontier, not single-best (from `loop.md`)
Keep a frontier of configs, not one champion — roles like best (throughput), lean (memory), and
diverse (different strategy). Cross over complementary winners rather than always mutating the
single best. Convergence is judged, not scheduled: stop producing new hypotheses when the
frontier stops improving and you're near the roofline (judged against the noise floor), with a
hard probe-count cap as backstop — there's no fixed plateau rule.
