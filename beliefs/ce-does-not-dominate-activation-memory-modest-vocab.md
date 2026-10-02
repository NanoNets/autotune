---
claim: At a modest vocab-to-sequence product, the cross-entropy term is a small share of activation
  memory and the per-layer term dominates — so recompute moves the OOM ceiling a lot, not a little.
type: ceiling
context:
  arch_family: dense decoder-only, vocab MODEST relative to params — vocab*seq*4B well below the
    model state (SmolLM2-1.7B, vocab 49152, seq 1024: 0.2GB against a 12.8GB model state)
  hardware_class: any CUDA with room for the model state
  dataset_shape: any — the term is a function of shape, not content
evidence:
  probes: 3 ceiling searches (uncompiled, +recompute, +compile) + experiments 9 and 10
  confidence: high — the ceiling moved as the decomposition predicted, on measured trials
links: [[ce-dominates-activation-memory-large-vocab]], [[compile-cuts-memory-not-just-time]],
  [[recompute-pays-only-instrumentally]]
---

The sibling of [[ce-dominates-activation-memory-large-vocab]], and the counter-case that narrowed it.
Same shape of analysis, opposite answer, because the discriminator that belief names —
`vocab x seq` against the model state — comes out the other way.

Measured on `HuggingFaceTB/SmolLM2-1.7B-Instruct` (1.711B params, 24 layers, hidden 2048, **MHA**
32/32 heads, vocab 49152) at seq 1024, from the ceiling trials
(`runs/smollm2_1.7b_a100_wikitext/ceiling.tsv`): peak rose 22.404 → 35.229 GB from mb 4 → 8, i.e.
**~3.2 GB/sample**, of which the logits path is roughly:

| Term | GB/sample | share |
|---|---|---|
| `lm_head` out bf16 `[1024, 49152]` + fp32 upcast and CE intermediates | ~0.3 | **~9%** |
| 24 decoder layers (norms, qkv/o, gate/up/silu) | ~2.9 | **~91%** |

**The consequence is the one that matters for search.** Because recompute discards the layer term —
the 91% here rather than the 30% there — it moves the ceiling *hard*:

| knobs | measured ceiling @ seq 1024 |
|---|---|
| none | 8 (OOM at 9) |
| `torch_compile` | 12 (OOM at 13) |
| `gradient_checkpointing` | **>=16** (capped at effective_batch; peak only 20.3GB, card_frac 0.57) |

Standalone, recompute cost 35% of throughput (52.1 vs 80.7 samples/s) — so it is not a throughput
lever. Its value was entirely in the ceiling it bought, which is
[[recompute-pays-only-instrumentally]].

**How to tell which sibling you are in, before spending a probe:** compute `vocab x seq x 4 bytes`
and compare it to the model state. Qwen2-0.5B at 151936 x 2048 gives ~1.2GB against a 7.4GB state and
CE dominated; SmolLM2-1.7B at 49152 x 1024 gives ~0.2GB against a 12.8GB state and it did not. The
folklore that recompute buys "2-3x" is regime-dependent in exactly this way, and the regime is
cheap to identify from `frame`'s Profile without running anything.
