---
claim: At large vocab, slicing the cross-entropy over the token axis with a checkpoint per slice
  removes the only activation term that scales with batch — peak memory becomes nearly
  batch-independent, and the micro_batch ceiling stops being memory-bound at all.
type: ceiling
context:
  arch_family: any causal LM where vocab x seq is large relative to the model state; Qwen3.5-2B at
    vocab 248320 / seq 1024 is the strongest case this repo has seen
  hardware_class: any CUDA
  dataset_shape: any — the term is a function of shape, not content
evidence:
  probes: 3 ceiling searches + 4 throughput probes (qwen3.5_2b_a100_wikitext experiments 22-25)
    + a numerical equivalence check
  confidence: high — the memory effect is large, mechanical, and reproduced across every
    micro_batch tested
links: [[ce-dominates-activation-memory-large-vocab]],
  [[compile-cuts-memory-not-just-time]],
  [[recompute-pays-only-instrumentally]],
  [[dispatch-count-bounds-small-real-batches]]
---

The full-logits path materializes `[mb, seq, vocab]` twice: once in bf16 from the `lm_head`, again
as the fp32 copy `cross_entropy` upcasts to. At vocab 248,320 that is the largest activation in the
step.

**Measured, at `max_len`=1024 on a 39.5GB A100:**

| micro_batch | ceiling peak, recompute only | ceiling peak, recompute + `chunked_ce=8` |
|---|---|---|
| 1 | 17.542 | 17.542 |
| 2 | 17.542 | 17.542 |
| 4 | 22.353 | 17.542 |
| 8 | 34.174 | 17.542 |
| 16 | OOM | **19.036** (search terminated CAPPED by effective_batch) |

**Peak is flat at the model-state floor across a 8x range of micro_batch.** Recompute removes the
layer activations; chunked CE removes the logits; nothing that remains scales with batch. Without
chunking, chunked CE saved 9.35GB at mb=4 alone (36.449 -> 27.095).

**Why it is a speed lever and not a model change.** The loss is a sum of the same per-token terms
over the same `ignore_index` mask divided by the same count — only fp32 summation *order* differs,
the same reassociation any fused kernel performs. `loop/throughput.md` lists "chunked
cross-entropy" under Kernels. Verified rather than asserted: loss matched the unchunked path to
**7.8e-08 relative** at chunks = 1, 2, 4, 8 and 16.

**What it bought.** grad_accum=1 (`micro_batch=16`), the last rung of the dispatch ladder, at
36.006 samples/s — and simultaneously the *safest* config in the run, worst case proven at
16.805GB of 39.494. Speed and memory-safety had traded against each other all run; this is the
lever that broke the trade.

**Chunk count barely matters for speed** (4 vs 8 was +4.1%, inside a ±7.2% band) but it *does* set
the worst-case peak: halving the chunk count roughly doubles per-slice fp32 logits. Prefer the
count whose worst case you have proven.

**Implementation note.** Call the backbone (`model.model`) for hidden states rather than the
`ForCausalLM` wrapper, shift, drop `-100` positions *before* chunking so slices are equal in real
tokens, and wrap each slice in `torch.utils.checkpoint(..., use_reentrant=False)` so its logits are
freed after the forward and rebuilt in the backward.
