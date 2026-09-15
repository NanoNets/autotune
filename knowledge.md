# knowledge.md — cross-run running log

One dated entry per run. Read this (plus `knowledge/beliefs.md`) before every hypothesis — see
`knowledge/README.md`. Keep entries short and empirical: claims backed by what's in that run's
`results.tsv`, not vibes.

---

## test_a10G — 2026-09-15
Context: `Qwen/Qwen2.5-0.5B` · `wikitext-2-raw-v1` · 1x NVIDIA A10 22.1GB (CC 8.6), bf16,
effective_batch 16, max_len 2048, ssh backend
Baseline: 4.338 samples/s → Best: 108.59 samples/s (**+2403%, 25.0x**), MFU 28.7% → peak 58.6%
23 experiments, 17 ok / 6 failed.

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
- **The launch-bound wall is unbroken.** The final config leaves ~25% of the GPU idle with one CPU core
  saturated dispatching, and the two obvious fixes are both blocked: cudagraphs (incompatible with
  grad_accum) and a bigger micro_batch (bounded by the ceiling at the declared `max_len`).
- **`max_len=2048` may be mis-declared** for this data (observed max 520 tok). If the application truly
  needs only ~512, the ceiling would rise sharply and the launch-bound wall would likely fall. This is
  the human's call, not the loop's — it was raised at Step 3 and they chose to keep 2048.
