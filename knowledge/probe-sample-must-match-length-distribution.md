---
claim: Under dynamic padding, a probe's row sample must represent the dataset's length distribution
  — batch width is the rows drawn, so an unrepresentative sample measures a workload the run will
  not train on. Static padding hides this entirely.
type: invariant
context:
  arch_family: any
  hardware_class: any
  dataset_shape: any — the effect is created by the padding lever, not by the data
evidence:
  probes: 5 direct (smollm2_1.7b_a100_wikitext experiments 23/26/27/28/33, sample 256→1000→2000→4000
    plus a sample-invariance control on the baseline), after 0 direct in test_a10G
  confidence: HIGH — mechanism structural AND magnitude measured at 1.9x on samples/s
links: [[static-padding-is-usually-the-bottleneck]], [[micro-batch-interacts-with-dynamic-padding]]
---

Under `padding="max_length"` every batch is `seq_len` wide whichever rows the loader drew, so
throughput is independent of the sample. Under `padding="longest"` batch width **is** the rows
drawn — so the sample's length distribution is part of the measurement.

That makes a baseline-vs-winner comparison across the two modes asymmetric: a
distribution-*invariant* config measured against a distribution-*sensitive* one. `test_a10G`'s 25x
was measured with frame scanning 1000 rows and the loader training on 256, neither of which saw the
long rows up to the truncation cap that real training meets. Direction solid; **magnitude an upper
bound**.

Worse than sample *size*: `frameworks/hf.py` `_usable_texts` takes the **first** N usable rows, so
enlarging `sample` still yields a head slice — an ordered file defeats it at any size.

**What to do:** compare the loader's sample against frame's `scan()` (p50/p90/p99/max — it already
computes exactly this) and treat a tail mismatch as a caveat on the number, not a detail. Report a
speedup measured on an unrepresentative sample as a ceiling.

**MEASURED — 2026-09-21 (smollm2_1.7b_a100_wikitext).** The experiment this belief asked for was
run, and the belief is upgraded from mechanism to fact.

| `sample` | tokens/step | samples/s (champion) |
|---|---|---|
| 256 (shipped) | 1088 | 264.4 |
| 1000 | 2057 | 138.1 |

**Effect size: 1.9x on samples/s from the row sample alone**, same config, same hardware. The first
256 non-empty rows of wikitext are far shorter than the first 1000.

Three things the measurement added that the mechanism argument did not predict:

- **The baseline really is sample-invariant** — 9.365 at 256 vs 9.342 at 1000, 0.25% apart, tokens/step
  identical. So a dynamic-padding config at any sample can still be compared against it honestly. That
  assumption was load-bearing for the whole run and it is now checked, not assumed.
- **`tokens/s` is sample-ROBUST while `samples/s` is sample-SENSITIVE** (17994 → 17758, 1.3%, across
  the same 1.9x). Exactly mirroring the cross-padding-mode problem where tokens/s inflates static
  padding and samples/s is the honest metric. **Neither metric is safe on its own** — read the pair
  against what changed.
- **Enlarging the head slice does not converge.** tokens/step at 1000 / 2000 / 4000 went
  2057 / 1719 / 1810 — non-monotone. `_usable_texts` takes the first N usable rows, so a bigger N is
  still a head slice of an ordered file. Confirms the "worse than sample size" note above: the fix is
  a stratified draw (`design_improvements.md` 1b), not a bigger number.

**What it cost when ignored:** this run's headline read 28.2x before the correction and 11.0x after,
and the probe sample was one of the two compounding causes (the other is
[[median-step-time-overstates-skewed-throughput]]). It also **flipped which config won** — see
[[recompute-pays-only-instrumentally]].
