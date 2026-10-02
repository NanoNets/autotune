---
claim: Throughput computed from a MEDIAN step time is biased upward whenever step times are
  right-skewed — and length-grouped dynamic padding makes them violently so. Report the mean.
type: invariant
context:
  arch_family: any
  hardware_class: any
  dataset_shape: variable-length rows under padding="longest" with length_grouped=true
evidence:
  # 2026-09-21: third sighting across two runs — treat as a standing repo defect
  probes: 8 (smollm2_1.7b_a100_wikitext experiments 27/28/30/31/32/33/34/38, including
    median/mean measured at 1.00 / 1.03 / 0.93 / 0.45 across four padding+grouping regimes)
  confidence: high — the failure is self-evident (MFU > 100%, GPU-active 560%) and the
    median/mean ratio tracks the grouping knob exactly
links: [[probe-sample-must-match-length-distribution]],
  [[micro-batch-interacts-with-dynamic-padding]], [[recompute-pays-only-instrumentally]]
---

> **2026-09-21 (qwen3.5_2b_a100_wikitext) — third sighting, and the fix is still not in the
> profiler.** The champion reported **`GPU active 126%`** — physically impossible, from
> `gpu_active_frac` dividing GPU busy by the MEDIAN step time at median/mean 0.781. `samples/s` was
> unaffected because the run harness divides by `step_time_mean_s`, so the two disagreed in the
> same Observation.
>
> Also measured here: **more steps does not fix skew, but it does bound it.** The same config at
> n=30/150/400 gave mean 0.7399/0.5807/0.5597 with cv 2.11/0.98/0.60 while the median sat still at
> 0.5105/0.5040/0.5106 — so most of that skew was amortizable warmup, and a real heavy tail
> remained. And at `grad_accum=1` the skew is **irreducible**: one optimizer step is one batch, so
> step time tracks batch width with no averaging at all (cv 1.34, sem ±6.7%, untightenable by more
> steps). `mfu`, `gpu_active_frac` and `roofline_floor_s` remain median-based in
> `profiler/diagnostic.py`.

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
