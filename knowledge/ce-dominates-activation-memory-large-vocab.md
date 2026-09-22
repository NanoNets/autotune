---
claim: On a small model with a large vocabulary, the logits/cross-entropy term dominates activation
  memory at long max_len — not attention — and recompute cannot touch it.
type: ceiling
context:
  arch_family: dense decoder-only, vocab LARGE relative to params — vocab*seq*4B comparable to or
    above the model state. Qwen2-0.5B at vocab 151936 / seq 2048 qualifies; SmolLM2-1.7B at
    vocab 49152 / seq 1024 does NOT (see the sibling belief).
  hardware_class: single mid-range datacenter GPU (A10 22GB)
  dataset_shape: any — the term is a function of shape, not content
evidence:
  probes: 1 ceiling search (test_a10G) + 4 ceiling searches and 4 probes
    (qwen3.5_2b_a100_wikitext, vocab 248320) + first-principles derivation
  confidence: high WHERE IT APPLIES; context narrowed 2026-09-21 (SmolLM2 counter-case) and
    re-confirmed 2026-09-21 at the largest vocab seen
links: [[compile-cuts-memory-not-just-time]],
  [[ce-does-not-dominate-activation-memory-modest-vocab]]
---

> **2026-09-21 (qwen3.5_2b_a100_wikitext) — strongest confirmation yet, and the corollary is now
> actionable.** Qwen3.5-2B at **vocab 248,320** / seq 1024 is the most extreme case in this repo.
> Measured directly by removing the term: chunking the cross-entropy cut peak memory **9.35GB at
> micro_batch=4** (36.449 -> 27.095GB), and at mb=8 it cut 34.174 -> 17.542GB. The discriminator in
> this belief's context line — `vocab*seq*4B` against the model state — picked the regime correctly
> in advance (248320 x 1024 x 4B = 1.02GB/sample of fp32 logits, against a 24.8GB model state).
>
> **The corollary "recompute cannot touch that term" is the load-bearing part** and it was confirmed
> the hard way: recompute alone took the ceiling only to 8, and the run's two OOM failures were both
> configs where recompute had removed the layer activations and the logits were what remained. The
> term needs its own lever. That lever now exists and is measured —
> [[chunked-ce-makes-memory-batch-independent]] — which turns this belief from a diagnosis into a
> prescription: at large `vocab*seq`, reach for chunked CE *before* spending the memory budget on
> recompute, because chunked CE is what makes peak memory batch-independent.

> **Context narrowed 2026-09-21.** This held on Qwen2-0.5B/vocab 151936/seq 2048 and was *not*
> reproduced on SmolLM2-1.7B/vocab 49152/seq 1024, where the same term is ~9% rather than 70%. The
> claim is sound but it is a claim about a *regime*, not about causal LMs in general. The
> discriminator is the `vocab x seq` product against the model state — see "How to spot the regime"
> below, which is what correctly predicted the counter-case. Sibling:
> [[ce-does-not-dominate-activation-memory-modest-vocab]].

Measured `peak_gb(mb) = 2.932 + 6.048·mb` at seq 2048 for `Qwen/Qwen2.5-0.5B` (494M params,
vocab 151,936). The 6.05 GB/sample decomposes as:

| Term | GB/sample |
|---|---|
| `lm_head` out bf16 `[2048, 151936]` | 0.580 |
| `logits.float()` upcast (transformers casts before CE) | 1.159 |
| `log_softmax` saved output fp32 | 1.159 |
| grad wrt float logits fp32 | 1.159 |
| **cross-entropy subtotal** | **4.057 (70%)** |
| 24 decoder layers (norms, qkv/o, gate/up/silu) | 1.770 (30%) |

**Why it matters for search:** `gradient_checkpointing` discards only the layer term, so it moves the
ceiling far less than the usual "2-3x" folklore. Predicted ceiling ≈4 under recompute; measured
exactly 2 → 4. The lever that attacks the 70% is chunked/fused linear-cross-entropy (Liger), which
was not implemented in the adapter and so was never reachable.

**How to spot the regime:** `vocab × seq × 4 bytes` comparable to, or larger than, the model state.
Here the embedding alone is 136M of 494M params (~28%), and it is tied.
