---
claim: torch.compile can be a memory lever, not only a speed one — Inductor's fused cross-entropy
  avoids materializing the fp32 logits copies, which raises the measured OOM ceiling.
type: ceiling
context:
  arch_family: large-vocab causal LM where CE dominates activations
  hardware_class: any CUDA
  dataset_shape: any
evidence:
  probes: 2 ceiling searches (uncompiled, compiled) + experiments 3 and 21
  confidence: high — reproduced in ceiling trials and in throughput probes
links: [[ce-dominates-activation-memory-large-vocab]]
---

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
