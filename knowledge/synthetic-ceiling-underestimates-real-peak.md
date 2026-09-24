---
claim: The synthetic OOM ceiling is a necessary but NOT sufficient bound — at matched shapes a real
  run peaks several GB above it, so a config sitting near the ceiling can pass the ceiling search
  and still OOM in training. The gap is one gradient buffer (params × dtype bytes): a
  `grad_accum=1` trial has none resident at its peak, a `grad_accum >= 2` run does. Until the
  fix is re-measured, a ceiling is unproven until a real full-width probe survives it.
type: ceiling
context:
  arch_family: any
  hardware_class: any CUDA; the margin is one gradient buffer (params × dtype bytes), so it
    bites hardest on large models near the card limit
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

The offset is the same on three independent shapes: exp 21 **+3.509 GB** (25.862 vs 22.353);
exp 1, the shipped baseline at `mb=1`/`grad_accum=16`, **+3.529 GB** (22.477 vs the `mb=1`
trial's 18.948); and exp 18's OOM message reads **30.11 GiB allocated by PyTorch** at the 7.58 GiB
fp32-logits allocation — its ceiling's pre-logits peak 34.174 − 7.58 + 3.505 = 30.10. The shipped
`mb=16` config (exp 23, `grad_accum=1`) peaked 2.2 GB *below* its ceiling (see "How to use it" for
why). The gap is an additive **constant**, not a fraction.

**Mechanism (found on review; fixed in `frame/ceiling.py`).** `find_ceiling.fits()` ran every
trial at `effective_batch = micro_batch`, i.e. `grad_accum=1`, on the argument that accumulation
does not move the peak. It does: `zero_grad(set_to_none=True)` frees `.grad` at step start, so a
`grad_accum=1` trial reaches its peak with no gradient buffer resident, while any real config with
`grad_accum >= 2` runs micro-step 2 with the full bf16 buffer — params × 2 bytes = 1.8818B × 2 =
**3.505 GB**, against the measured +3.509 GB. Trials now accumulate twice whenever the real config
accumulates at all. Predicted from this arithmetic; not yet re-measured on a GPU.

**This is not a contradiction of the dynamic-padding guarantee so much as a measurement of its
fine print.** The gap is one gradient buffer, an additive term, so the check is `ceiling peak + params × dtype
bytes` fits — a `card_frac` fraction is the wrong model for it, even where it happens to separate
this run's failures (0.878, 0.928) from its verified configs (0.57, 0.52).

**What it cost.** It invalidated two champions mid-run. `micro_batch=8 + recompute` measured
28.585 samples/s (13.27x) and `micro_batch=4` measured 25.803 (11.98x); both OOM on a full-width
batch and neither can be shipped. Under `length_grouped=True` this is not a remote risk — grouping
*actively clusters the longest rows into the same batch*, so the worst case is something the
sampler seeks out rather than stumbles on.

**How to use it.** Do not ship a config on ceiling arithmetic. Run the candidate once at
`padding="max_length"`, which forces every batch to `[micro_batch, max_len]` — the *shape* worst
case — and costs one short probe: default warmup plus a couple of clean steps (optimizer state
lands on the first `opt.step()`, the gradient buffer is resident from micro-step 2). If it
survives, the shape bound is demonstrated rather than inferred.

**Under `chunked_ce` that probe is not the whole worst case.** `_chunked_loss` drops pad labels
before chunking, so a real `max_length` batch of short rows runs CE over a few thousand real
tokens, while the synthetic trial (labels = ids, nothing masked) runs it over all `mb × max_len`.
That is exp 23's −2.2 GB below its ceiling. So with chunked CE the real probe bounds the
transformer stack and the (fixed) synthetic ceiling bounds the CE term: **both are required**.
