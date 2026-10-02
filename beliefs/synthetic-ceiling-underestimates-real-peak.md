---
claim: The synthetic OOM ceiling is a necessary but NOT sufficient bound — at matched shapes a real
  run peaks several GB above it, so a config sitting near the ceiling can pass the ceiling search
  and still OOM in training. Treat a ceiling above ~0.65 card_frac as unproven until a real
  full-width probe survives it.
type: ceiling
context:
  arch_family: any
  hardware_class: any CUDA; the margin needed scales with micro_batch, so it bites hardest on
    large cards running large micro_batches
  dataset_shape: any — but it is only DISCOVERABLE under dynamic padding, where the probe's own
    batches are narrow and never exercise the worst case the ceiling claims to bound
evidence:
  probes: 3 real-data probes against 4 ceiling searches (qwen3.5_2b_a100_wikitext experiments
    18, 20, 21) — two measured OOMs and one measured margin
  confidence: high — two independent counter-examples at different micro_batches, plus a clean
    positive control, and the mechanism is structural
links: [[probe-sample-must-match-length-distribution]],
  [[recompute-pays-only-instrumentally]],
  [[chunked-ce-makes-memory-batch-independent]]
---

`frame/README.md` states the guarantee that `padding="longest"` rests on: *"a config that fits the
synthetic worst case fits every batch the dataset can produce."* **Two measured counter-examples,
at matched shapes, on the same card and the same run:**

| config | ceiling peak | card_frac | real run at the SAME shapes |
|---|---|---|---|
| mb=8 + recompute | 34.174 GB | 0.878 | **OOM** |
| mb=4, no recompute | 36.449 GB | 0.928 | **OOM** (reached 35.697) |
| mb=4 + recompute | 22.353 GB | 0.566 | ok — real peak **25.862 GB** |

The positive control is the useful one: **+3.51 GB (+15.7%) over its own ceiling**. The two OOMs
imply a larger gap still at their (larger) micro_batches, so the margin appears to **grow with
micro_batch**.

**Mechanism.** `frame/synthetic.py` builds ONE worst-case batch, puts it on the device, and the
runner reuses it every step (`frameworks/hf.py` `_synthetic_batch`). A real run instead allocates
fresh host tensors per micro-step and `.to(device)`-copies them, so it carries per-micro-step
staging allocations and the allocator churn they cause on top of the same activation peak. The
ceiling measures activation memory correctly; it just isn't measuring the whole process.

**This is not a contradiction of the dynamic-padding guarantee so much as a measurement of its
fine print.** `frame/README.md` already says *"at a high `card_frac`, keep a margin below the
ceiling rather than running at it"* — that caveat was advisory and is now quantified. Every
configuration that failed sat at `card_frac >= 0.878`; everything verified sits at `<= 0.60`.

**What it cost.** It invalidated two champions mid-run. `micro_batch=8 + recompute` measured
28.585 samples/s (13.27x) and `micro_batch=4` measured 25.803 (11.98x); both OOM on a full-width
batch and neither can be shipped. Under `length_grouped=True` this is not a remote risk — grouping
*actively clusters the longest rows into the same batch*, so the worst case is something the
sampler seeks out rather than stumbles on.

**How to use it.** Do not ship a config on ceiling arithmetic. Run the candidate once at
`padding="max_length"`, which forces every batch to `[micro_batch, max_len]` — the exact worst
case — and costs one short probe because static padding is distribution-invariant (cv ~0.006, so
20 steps suffices). If it survives, the bound is demonstrated rather than inferred.
