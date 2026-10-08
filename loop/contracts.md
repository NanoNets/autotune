# Contracts

The five data shapes the loop moves. Everything plugs into these — keep them stable.

Flow: Frame builds **Profile** + **SearchSpace** and pulls relevant **Beliefs** →
Hypothesize emits a **Config** → Test returns an **Observation** → Learn writes **Beliefs**. Repeat.

Rule: contracts are the only things that cross step boundaries.

---

## Profile
What we know about the situation, before searching. Built once by Frame.

| Field | Meaning |
|---|---|
| `model` | internals: layers, hidden, heads, kv_heads, intermediate, vocab, experts/top_k, arch, attention, params, modality, dtype |
| `hardware` | gpu type, compute capability, memory, count, interconnect, gpus_per_node, nodes, $/hr |
| `env` | library versions (torch, framework, flash-attn, deepspeed), available kernels |
| `dataset` | length distribution, total tokens/samples, modality, storage location, schema |
| `roofline` | peak FLOPs → achievable tokens/s ceiling (the "how good is possible") |
| `constraint` | effective_batch + max_len targets (human-owned, off-limits) |
| `ceiling` | MEASURED largest micro_batch that runs at max_len, on worst-case synthetic batches. Carries the `config` it was measured under — compare before trusting it. `null` = not measured, never "fine" |
| `notes` | what the agent must read before tuning: memory regime, padding-heavy max_len, a ceiling at the edge of the card |
| `priors` | Beliefs retrieved by context + researched external priors |

Consumed by: Hypothesize, the surrogate, memory (keys on it).

---

## SearchSpace
What the loop may vary, and the safety bounds. Built by Frame.

| Field | Meaning |
|---|---|
| `knobs` | searchable dims in **canonical** names; each: type, range/values, dependencies, cost-hint. `micro_batch` carries a `max` stamped from the measured `ceiling` (a framework's `knobs()` can only declare `min`, so the bound is added at Frame) |
| `safety` | per knob: `safe` / `off_limits` / `unknown`. Model-decided at Frame. Default `unknown → off_limits` |
| `capability` | canonical ∩ this framework's supported knobs (the projection) |
| `fixed` | human-owned constants: model, dataset, hardware, effective_batch, max_len |

Canonical vocabulary is fixed; the projection to a framework is per-run.
Consumed by: Hypothesize, the surrogate.

---

## Config
One hypothesis — a point in the space.

| Field | Meaning |
|---|---|
| `knobs` | assignments in **canonical** names, e.g. `{micro_batch: 4, shard_degree: 2, compile: default}` |
| `rationale` | why proposed: parent(s), surrogate prediction, expected outcome |

Canonical only. The framework adapter translates to concrete flags/API.
Consumed by: Test.

---

## Observation
Everything measured from running a Config. Raw and rich.

| Field | Meaning |
|---|---|
| `status` | ok / oom / error / invalid (a physically impossible reading — never scored) |
| `objective` | samples/s (**primary**, from clean-window totals) + tokens/s (discriminator) + MFU (diagnostic only — `loop/throughput.md`) |
| `memory` | peak, headroom; on OOM: breakdown (what/how much overshot) |
| `signals` | raw gpu/cpu util, comm, data-wait. **No derived labels** — the model interprets |
| `provenance` | did the config apply? which kernels engaged vs silently fell back? |
| `population` | which rows the clean window timed (and over how many whole passes), and which fixed reference batch the trace describes. A **script probe** (the user's own training script, `program.md` "Script mode") says so: `driver: "script"`, `clean_window: "fixed-position"` (not whole passes) and `trace_basis: "traced-step"` (not a reference batch), plus the `script`, `script_args` and `script_files` it ran. Under the axiom below, it is never `same-rows` or `same-reference-batch` evidence |
| `reliability` | step-time spread (`step_time_cv`), steps measured, passes completed |
| `error` | on failure: structured cause |

Invariant: raw, not pre-digested. The model reads and interprets; code does not label.
Consumed by: Learn, the surrogate, Hypothesize.

### Measurement axiom: one number, one population

**A measured quantity may only be compared with, or divided by, another quantity measured on the
same population of work under the same instrument.**

A *population* is a specific set of rows, in specific batches, processed a specific number of
times. An *instrument* is either the **clean window** (profiler off) or the **trace** (profiler on).

1. **Two populations, two jobs.** The clean window measures *the run*: throughput, step time,
   variance, MFU. The trace measures *the reference batch*: what a step looks like inside (phases,
   kernels, launches, dtypes, busy time). Neither stands in for the other; a trace number
   describes one batch, never the run's average.
2. **No mixed ratios.** A ratio never takes its numerator from one population and its denominator
   from another. GPU-active is the reference batch's busy time over *that same batch's*
   profiler-off wall; MFU is the clean window's total FLOPs over its total time.
3. **Rates come from totals.** Throughput is total work over total time across the clean window.
   A median, or any single-step statistic, is never the denominator of a rate.
4. **The population is fixed by rule, not by position.** The clean window covers whole passes over
   the sample, starting at a pass boundary, so every config times exactly the same rows the same
   number of times. The reference batch is a fixed set of rows chosen once by rule, so changing
   `clean_steps`, warmup or loader order cannot change which batch is traced.
5. **Recorded populations decide what is comparable.** Every Observation records its population.
   Two probes' **throughput** compares when they timed the same rows (`sample_hash` matches,
   whole passes) — different batches are fine, because re-batching the same rows is what a loader
   knob does. Their **trace durations** (busy time, phase times, kernel times) compare only when
   the reference batch matches (`batch_key`). **Counts and provenance** (kernel count, gap count,
   which kernels ran, dtypes) compare across any batch: they are set by layers x micro-steps, not
   by batch width.
6. **An impossible value is a population error until proven otherwise.** A fraction above 1 or a
   negative remainder (e.g. `wall - busy`) means two populations were mixed or the instrument
   failed. The Observation is marked `invalid` and is not scored.
7. **A belief is promoted only on evidence that satisfies this axiom.** Each comparison behind it
   must be *same-rows* (clean window, whole passes), *same-reference-batch* (trace durations), or
   *width-invariant* (counts and provenance), and every rate must come from totals. Anything else
   is recorded in the run's `experience/` entry as an observation, never as belief evidence, and
   the belief's `evidence` names which basis it rests on.

Why: every impossible reading in the smollm2 and qwen3.5 A100 runs broke this — one profiled
batch's busy time over the median clean step (GPU-active 126-560%), mean tokens over median time
(MFU up to 144%), and partial passes that let loader order decide which rows got timed. Under
static padding the axiom holds by accident, because every batch is identical; dynamic padding and
length grouping are what make it necessary.

---

## Belief
One durable thing learned. A markdown file in `beliefs/`, one claim each.

| Field | Meaning |
|---|---|
| `claim` | the empirical statement, e.g. "shard_degree=2 fits micro_batch=4 on this profile" |
| `context` | key for retrieval/transfer: `{arch_family, hardware_class, dataset_shape}` |
| `evidence` | supporting observations, confidence, probe count, and the **basis** each comparison rests on: `same-rows` / `same-reference-batch` / `width-invariant` (Measurement axiom, corollary 7) |
| `type` | regime / ceiling / calibration / dead-lever / safety / invariant |
| `links` | related beliefs |

Authored and curated by the model (code is plumbing). Injected at Frame by context match.
Consumed by: Hypothesize, future Frames, the surrogate (calibration beliefs).
