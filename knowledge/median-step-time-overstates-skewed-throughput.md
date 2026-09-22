---
claim: Throughput computed from a MEDIAN step time is biased upward whenever step times are
  right-skewed — and length-grouped dynamic padding makes them violently so. Report the mean.
type: invariant
context:
  arch_family: any
  hardware_class: any
  dataset_shape: variable-length rows under padding="longest" with length_grouped=true
evidence:
  probes: 8 (smollm2_1.7b_a100_wikitext experiments 27/28/30/31/32/33/34/38, including
    median/mean measured at 1.00 / 1.03 / 0.93 / 0.45 across four padding+grouping regimes)
  confidence: high — the failure is self-evident (MFU > 100%, GPU-active 560%) and the
    median/mean ratio tracks the grouping knob exactly
links: [[probe-sample-must-match-length-distribution]],
  [[micro-batch-interacts-with-dynamic-padding]], [[recompute-pays-only-instrumentally]]
---

Throughput is total work over total time. A rate built as `effective_batch / median_step_time` is
therefore only correct when mean == median. Under length-grouped dynamic padding it badly is not: a
few chunks hold the longest rows and most do not, so the median step lands on a narrow chunk while
the mean carries the wide ones.

**The tell is a physically impossible number.** Three probes reported **MFU > 100%** (110%, 115%,
144%) and one **GPU-active 560%**. Those are not noise — `mfu` and `gpu_active_frac` are also
computed against the median, and `gpu_busy_s` comes from the single profiled step (`profiler/README`,
"Two steps, two sources"), so a wide profiled step over a narrow median step explodes the ratio.

**Raising the step count does not fix it.** Going `clean_steps` 10 → 300 left MFU at 115%. More
samples of a skewed distribution estimate the median better; they do not make the median the right
statistic. It did fix a *second*, separate problem — tokens/step at `sample=256` moved 1088 → 1576,
so the token count had been undersampled too.

**The ratio tracks the grouping knob, which identifies the cause:**

| config | median/mean | interpretation |
|---|---|---|
| `padding=max_length` | **1.00** | batch width constant, no skew at all |
| `padding=longest`, `length_grouped=false` | **1.03** | width varies mildly (max of N random rows) |
| `padding=longest`, `length_grouped=true`, sample 1000 | **0.93** | grouped, 62 chunks |
| `padding=longest`, `length_grouped=true`, sample 256 | **0.45** | grouped, only 16 chunks |

**The lever that wins is the lever that biases its own measurement.** Length grouping was worth +96%
here — and it is the sole source of the skew. Fewer chunks make it worse, so a small probe sample is
doubly dangerous: it is unrepresentative *and* maximally skewed.

**What it cost when ignored.** This run reported 264.4 samples/s and a 28.2x speedup at experiment 16;
the same config measured with a mean-scored rate on a representative sample is 102.8 and **11.0x**.
The two corrections are independent and they compound (1.9x from the sample x ~2.2x from the median).
It also **changed which config won** — see [[recompute-pays-only-instrumentally]].

**What to do:** `backends/local.py` now returns `step_time_mean_s` and `clean_steps_measured`
alongside the Observation, and the probe harness scores on the mean when present. The profiler's own
`mfu` / `gpu_active_frac` / `roofline_floor_s` are **still median-based** and will still print
impossible values under skew — treat MFU > 100% as a broken measurement, not a great result, and
sanity-check any dynamic-padding rate against `median/mean` before believing it.
