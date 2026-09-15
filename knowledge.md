# knowledge.md — cross-run running log

One dated entry per run, across all runs. Read **at Frame** (`loop/loop.md` §3) to answer "has this
configuration been run before" — match on the `Context:` line; a hit is replayed as that run's
experiment 2, never as its baseline. Per-hypothesis priors come from `knowledge/beliefs.md`
instead, which is context-filtered. See `knowledge/README.md`. Keep entries short and empirical:
claims backed by what's in that run's `results.tsv`, not vibes.

---

## test_a10G — 2026-09-15
Context: `Qwen/Qwen2.5-0.5B` · `wikitext-2-raw-v1` · 1x NVIDIA A10 22.1GB (CC 8.6), bf16,
effective_batch 16, max_len 2048, ssh backend
Baseline: 4.338 samples/s → Best: 108.59 samples/s (**+2403%, 25.0x**), MFU 28.7% → peak 58.6%
23 experiments, 17 ok / 6 failed (2 crashes, 4 timeouts); 10 of the 17 set a new record.
Plot: `runs/test_a10G/progress.png` (`python runs/progress.py`). Beliefs promoted to
`knowledge/` — see `knowledge/beliefs.md`.

**Winning config:** `micro_batch=8`, `gradient_checkpointing=True`, `torch_compile=True`,
`attn_implementation=sdpa`, `padding=longest`, `length_grouped=True`, `pad_multiple=8`, `tf32=False`.
Confirmed twice (108.591 / 107.916, 0.6% apart).

### Findings

- **The dominant win was never a kernel — it was not computing padding.** On a p50=10-token dataset
  declared at `max_len=2048`, the shipped `padding="max_length"` loader spent ~97% of every step on
  pad tokens. Pad-to-longest + length-grouped batching + a small `pad_multiple` took tokens/step from
  32768 → 1125 for *identical* gradients. Everything else combined was worth ~2x; this was worth ~12x.
- **Cross-entropy, not attention, dominated activation memory** at `max_len` on a small model with a
  large vocab. Measured 6.05 GB/sample, of which ~4.06 GB (70%) is the `[2048, 151936]` logits tensor
  and its fp32 upcast. Predicted from first principles to within 3.7%. Corollary that held up:
  `gradient_checkpointing` cannot touch that term, so it moves the ceiling far less than folklore
  suggests (2 → 4 measured, not 8+).
- **`torch.compile` cut peak memory by 5.4 GB**, not just time, by fusing the CE and never
  materialising the fp32 logits copies. That *raised the OOM ceiling 2 → 4* and so paid for a bigger
  micro_batch. Compile's value survived every later regime change (-43% without it, even at the end).
- **`tf32` is inert on a pure-bf16 config.** No fp32 matmul exists in the graph for it to act on.
  Expect this to generalise to any bf16 training job.
- **Cudagraphs are incompatible with gradient accumulation in this runner.** Both `max-autotune` and
  `reduce-overhead` fail identically ("accessing tensor output of CUDAGraphs that has been
  overwritten"), and `cudagraph_mark_step_begin()` does not fix it — the backward's saved activations
  live in the graph pool and the next micro-batch overwrites them.
- **`max-autotune`'s search cost can exceed the experiment budget** (>10 min, never reached a timed
  step on a 0.5B model). A lever whose search is slower than the whole probe is not usable.

### Surprises / contradictions

- **A winner flipped when the regime changed.** `flash_attention_2` won +40.2% while batches were 97%
  padding (it unpads). Once padding was removed at the loader, its unpad/repad overhead made it a net
  *loss* and plain `sdpa` beat it by 34%. **Re-test winners after a regime change** — but note compile
  did *not* flip, so this is a discipline for finding flips, not an expectation that all of them do.
- **`micro_batch` and dynamic padding interact adversely.** A bigger micro_batch pads to the longest of
  more rows, so mb 4→8 was +15% more efficient *per token* yet -14% on samples/s (tokens/step
  2560→4572). Length grouping is what makes a larger micro_batch pay.
- **Occupancy was a red herring throughout.** It *fell* at nearly every winning step (43→45→27→25%)
  while throughput rose. GPU-active likewise: the final config runs at 75% active and beats a 100%-active
  one. MFU was the only signal that tracked reality — and only until padding was removed, after which
  even MFU misleads (it counts pad tokens as work).
- **High within-run variance did not mean unreliable comparisons.** cv reached 0.29-0.37 from varying
  batch widths, yet repeats of identical configs agreed to 0.4-0.9%. The median over steps is stable
  even when steps are not.

### Open hypotheses for next run

- **`pad_multiple` in (8, 64) is unexplored** — 8 is the measured best, 64 costs 20%, 1 explodes compile
  cost. The interval was never cleanly bracketed (exp23 was contaminated by a prior kill).
- **Packing / `padding_free`** should beat pad-to-longest outright: it fills every position with real
  tokens instead of merely trimming padding. Not implemented in `frameworks/hf.py`.
- **Chunked / fused linear-cross-entropy (Liger)** attacks the 70% memory term directly and was never
  reachable — not implemented in the adapter. Highest-value unbuilt lever for large-vocab small models.
- **The attention verdict was never confirmed on the shipped config.** `sdpa` > `flash_attention_2`
  by 34% rests on experiments 11/13, both measured at `pad_multiple=64`; the re-test on the final
  `pad_multiple=8` config timed out. Narrower batches should only *reduce* what FA2's unpadding can
  recover, so the ordering is very likely unchanged — but it is an assumption at the shipped setting,
  not a measurement. One cheap probe would close it.
- **The launch-bound wall is unbroken.** The final config leaves ~25% of the GPU idle with one CPU core
  saturated dispatching, and the two obvious fixes are both blocked: cudagraphs (incompatible with
  grad_accum) and a bigger micro_batch (bounded by the ceiling at the declared `max_len`).
- **`max_len=2048` may be mis-declared** for this data (observed max 520 tok). If the application truly
  needs only ~512, the ceiling would rise sharply and the launch-bound wall would likely fall. This is
  the human's call, not the loop's — it was raised at Step 3 and they chose to keep 2048.

### Addendum — 2026-09-15 (appended, findings above unedited)

**The 25x is an upper bound, not a point estimate.** The comparison is between a
distribution-*invariant* baseline (`padding="max_length"` — every batch 2048 wide regardless of the
rows drawn) and a distribution-*sensitive* winner (`padding="longest"` — batch width *is* the rows
drawn). Frame scanned 1000 rows; the loader trained on 256 (`sample`). Both are a small slice of the
split, and neither saw the long rows up to the 2048 truncation cap that real training will hit —
those widen batches and lower samples/s.

Unaffected: everything measured on synthetic batches (the ceiling 2 → 4 → 8) or derived from
mechanism (CE dominating activation memory, compile's 5.4GB, tf32 inert under bf16, cudagraphs x
grad_accum). The padding finding's *direction* is arithmetic — p50 10 against `max_len` 2048 means
~97% of static-padded compute is pad — only the magnitude is sample-bound.

**Doctrine this produced:** under dynamic padding the probe's row sample must represent the
dataset's length distribution, or throughput is measured on a workload the run will not train on.
Static padding hid this — every batch was full width regardless. Now in `frameworks/README.md` and
`program.md` Step 4; the deeper fix (draw the sample to match the distribution, don't just enlarge
it) is `design_improvements.md`.
