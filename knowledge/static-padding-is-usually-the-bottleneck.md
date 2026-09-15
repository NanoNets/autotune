---
claim: When max_len is declared far above the data's real length distribution, padding — not kernels —
  is the dominant cost, and fixing it at the loader outweighs every kernel-level lever combined.
type: regime
context:
  arch_family: any
  hardware_class: any
  dataset_shape: p50 far below max_len (here p50 10 tok vs max_len 2048)
evidence:
  probes: 5 (experiments 9, 15, 17, 18, 20)
  confidence: high on DIRECTION — reproduced across four configs, best confirmed twice to 0.6%.
    MEDIUM on magnitude — measured on a 256-row loader sample, so the speedup is an upper bound
    (see "Known gap" below).
links: [[padding-regime-flips-attention-verdict]], [[micro-batch-interacts-with-dynamic-padding]]
---

`frameworks/hf.py` padded every batch to `max_length`. On wikitext (p50 **10** tokens, p99 374,
max 520) against `max_len=2048`, ~97% of every step was pad tokens — positions masked out of
attention and set to `-100` in the labels, i.e. **work whose result is discarded by construction**.

Three levers, which only pay together:

| Lever | Effect |
|---|---|
| `padding="longest"` | 8.86 → 39.0 samples/s (+340%) |
| `length_grouped` | +45% on top (batches hold near-equal lengths, so pad-to-longest wastes little) |
| `pad_multiple` 64 → 8 | +20% more (p50 is 10 tok — a 64-bucket re-wastes what grouping just saved) |

Net: tokens/step **32,768 → 1,125** for identical gradients. Worth ~12x, where every kernel/compiler
lever combined was worth ~2x.

**Check this first, before any kernel work.** The diagnostic is free — compare the dataset's p50/p99
(frame reports both) against `max_len`. Frame already emits a note when `max_len` ≫ p99; treat it as
the headline, not a footnote.

**Known gap — the magnitude is an upper bound (added 2026-09-15).** The measurement compares a
distribution-*invariant* baseline (`padding="max_length"`: every batch 2048 wide, whichever rows you
draw) against a distribution-*sensitive* winner (`padding="longest"`: batch width **is** the rows you
drew). And the rows were few — frame scanned 1000, the loader trained on 256 (`sample`), out of a far
larger split. The full dataset holds rows up to the 2048 truncation cap that neither sample saw;
more long rows means wider batches and lower samples/s. So the **direction is solid and the ~12x is
a ceiling, not a point estimate.**

The general lesson, which outlives this run, is its own belief:
[[probe-sample-must-match-length-distribution]].

**Caveat on `pad_multiple`:** the interval (8, 64) is unexplored. 1 is unusable — it defeats shape
bucketing and Inductor recompiles without end.
