---
claim: When max_len is declared far above the data's real length distribution, padding — not kernels —
  is the dominant cost, and fixing it at the loader outweighs every kernel-level lever combined.
type: regime
context:
  arch_family: any
  hardware_class: any
  dataset_shape: p50 far below max_len (A10 run p50 10 vs 2048; A100 run p50 9 vs 1024)
evidence:
  # 2026-09-21: third dataset/model/card. Direction holds; magnitude is regime-dependent
  probes: 7 across 2 runs, 2 models, 2 cards (test_a10G experiments 9/15/17/18/20;
    smollm2_1.7b_a100_wikitext experiments 2 and 38)
  confidence: high on DIRECTION — now reproduced on a second architecture (Llama MHA 1.7B,
    vocab 49k) and a second card (A100 SM80), and isolated cleanly on a final config.
    MEDIUM on magnitude — still sample-conditional; see "Known gap" and
    [[probe-sample-must-match-length-distribution]], which now has a measured effect size.
links: [[padding-regime-flips-attention-verdict]], [[micro-batch-interacts-with-dynamic-padding]],
  [[probe-sample-must-match-length-distribution]], [[median-step-time-overstates-skewed-throughput]]
---

> **2026-09-21 (qwen3.5_2b_a100_wikitext) — third independent confirmation, with a bound on the
> magnitude.** p50 9 tokens against `max_len` 1024 (~99% pad), and the padding levers were again the
> largest single jump of the run (5.37x from the baseline). But isolated on the *final* config the
> lever was worth **3.58x**, and on an intermediate config only **2.01x**, against the prior run's
> 6.26x on a dense model.
>
> **The magnitude is capped by whatever else bounds the step.** Removing pad tokens removes GPU
> *work*; if the step is launch-bound, that work was partly free
> ([[dispatch-count-bounds-small-real-batches]]). The tell is stark here: under static padding this
> model runs at MFU 49% and GPU-active 100% — it is *efficient* per token and merely computes 6.8x
> too many of them. So the isolated value of the padding fix RISES as the dispatch wall lifts
> (2.01x -> 3.58x as grad_accum went 4 -> 1). Check p50/p99 against `max_len` first, as the belief
> says — but expect the payoff to scale with how compute-bound you already are.

**2026-09-21 (smollm2_1.7b_a100_wikitext).** Re-confirmed, and for the first time **isolated on the
shipped config** rather than inferred from a baseline-vs-winner gap: flipping only `padding`
on the final config gave 16.4 → 102.8 samples/s, **6.26x**, at `sample=1000` over 300 mean-scored
steps. p50 9 tok against `max_len` 1024. Worth noting what the isolation also showed: the static
config ran at **MFU 75% and GPU-active 99%** while delivering 6.3x less throughput — a busy GPU
computing padding still looks healthy on every utilization signal.

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
