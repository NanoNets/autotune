---
claim: torch.compile can be a memory lever, not only a speed one — Inductor's fused cross-entropy
  avoids materializing the fp32 logits copies, which raises the measured OOM ceiling.
type: ceiling
context:
  arch_family: DENSE causal LM, any vocab size — the effect survived where CE does NOT dominate.
    NOT hybrid linear-attention, where compile does not complete at all (2026-09-21)
  hardware_class: any CUDA
  dataset_shape: any
evidence:
  probes: 5 across 2 runs, 2 models, 2 cards (test_a10G ceiling searches + experiments 3/21;
    smollm2_1.7b_a100_wikitext ceiling search + experiments 7 and 35)
  confidence: high — reproduced in ceiling trials and throughput probes on both runs
links: [[ce-dominates-activation-memory-large-vocab]],
  [[ce-does-not-dominate-activation-memory-modest-vocab]]
---

> **2026-09-21 (qwen3.5_2b_a100_wikitext) — context narrowed to DENSE models.** All five prior
> probes were dense transformers. On a hybrid linear-attention model `torch.compile` never reached a
> timed step at all — two 601s timeouts, with shape count explicitly ruled out as the cause — so this
> belief has no purchase there and its memory benefit must be obtained another way
> ([[chunked-ce-makes-memory-batch-independent]] attacks the same CE term directly). See
> [[compile-unusable-on-hybrid-linear-attention]]. The dense evidence is untouched.

**2026-09-21 (smollm2_1.7b_a100_wikitext).** Third confirmation, and it **widens the context**: this
claim was filed under "large-vocab LM where CE dominates activations", but SmolLM2-1.7B at vocab
49152 / seq 1024 is a model where CE does **not** dominate (~9% of activations, see
[[ce-does-not-dominate-activation-memory-modest-vocab]]) — and compile still cut memory. Measured
20.83 → 18.65 GB at mb=8 and 16.56 → 14.50 GB at mb=16, and the ceiling moved **8 → 12**. So the
memory saving is not contingent on the CE term being the dominant one; the arch_family key has been
loosened accordingly. Caveat carried forward: 12 did not divide `effective_batch=16`, so on this run
the lifted ceiling bought nothing — a ceiling only pays at a legal factorization.

Enabling `torch.compile` cut peak memory **16.63 → 11.23 GB** (-5.4 GB) at the same `micro_batch`,
alongside a +32.8% throughput gain. Re-measuring the ceiling under compile moved it **2 → 4** — and 4
divides `effective_batch=16`, so the compile flag *created* the batch headroom the next experiment
spent. Per-sample activation cost fell 6.05 → 3.73 GB, consistent with fusing away part of the
4.06 GB fp32 logits term.

Confirmed from the other direction at the end of the run: disabling compile on the final config cost
-43% **and raised** peak memory 5.95 → 8.08 GB.

**Two consequences for the loop:**
1. Treat `compile` as a knob that stales the OOM ceiling, the same as `recompute`/`dtype`/`attn_kernel`.
   `frame/README.md` does not list it; it should.
2. Compile's value **survived** every regime change in this run, unlike
   [[padding-regime-flips-attention-verdict]] — re-testing winners finds flips, it does not predict them.
