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
  probes: 0 direct — inferred from test_a10G's measurement conditions, not yet measured head-on
  confidence: medium — the mechanism is structural, the magnitude is unquantified
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

**Not yet measured.** The experiment that would settle it — samples/s at 256 / 1k / 4k and against a
stratified draw — is in `design_improvements.md` 1b. Until then this is a mechanism argument with an
unquantified effect size; sharpen or retire it on the next run that uses dynamic padding.
