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
| `knobs` | searchable dims in **canonical** names; each: type, range/values, dependencies, cost-hint. `micro_batch`'s upper bound comes from the Profile's measured `ceiling`, not from a guess |
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
| `knobs` | assignments in **canonical** names, e.g. `{micro_batch: 4, shard_degree: 2, packing: true}` |
| `rationale` | why proposed: parent(s), surrogate prediction, expected outcome |

Canonical only. The framework adapter translates to concrete flags/API.
Consumed by: Test.

---

## Observation
Everything measured from running a Config. Raw and rich.

| Field | Meaning |
|---|---|
| `status` | ok / oom / error |
| `objective` | tokens/s + MFU (% of roofline) |
| `memory` | peak, headroom; on OOM: breakdown (what/how much overshot) |
| `signals` | raw gpu/cpu util, comm, data-wait. **No derived labels** — the model interprets |
| `provenance` | did the config apply? which kernels engaged vs silently fell back? `probe_kind`: `real` (throughput, real data) or `synthetic_ceiling` (memory only — NEVER scored for throughput) |
| `cost` | gpu-seconds, $ |
| `reliability` | measured noise/variance, steps measured, steady-state reached? |
| `context` | framework hints/warnings, log tail |
| `error` | on failure: structured cause |

Invariant: raw, not pre-digested. The model reads and interprets; code does not label.
Consumed by: Learn, the surrogate, Hypothesize.

---

## Belief
One durable thing learned. A markdown file in `knowledge/`, one claim each.

| Field | Meaning |
|---|---|
| `claim` | the empirical statement, e.g. "shard_degree=2 fits micro_batch=4 on this profile" |
| `context` | key for retrieval/transfer: `{arch_family, hardware_class, dataset_shape}` |
| `evidence` | supporting observations, confidence, probe count |
| `type` | regime / ceiling / calibration / dead-lever / safety / invariant |
| `links` | related beliefs |

Authored and curated by the model (code is plumbing). Injected at Frame by context match.
Consumed by: Hypothesize, future Frames, the surrogate (calibration beliefs).
