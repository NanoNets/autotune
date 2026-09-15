---
claim: On a small model with a large vocabulary, the logits/cross-entropy term dominates activation
  memory at long max_len — not attention — and recompute cannot touch it.
type: ceiling
context:
  arch_family: dense decoder-only (Qwen2), large vocab relative to params
  hardware_class: single mid-range datacenter GPU (A10 22GB)
  dataset_shape: any — the term is a function of shape, not content
evidence:
  probes: 1 ceiling search (4 synthetic trials) + first-principles derivation
  confidence: high (predicted 5.83 GB/sample vs 6.05 measured, 3.7% error)
links: [[compile-cuts-memory-not-just-time]]
---

Measured `peak_gb(mb) = 2.932 + 6.048·mb` at seq 2048 for `Qwen/Qwen2.5-0.5B` (494M params,
vocab 151,936). The 6.05 GB/sample decomposes as:

| Term | GB/sample |
|---|---|
| `lm_head` out bf16 `[2048, 151936]` | 0.580 |
| `logits.float()` upcast (transformers casts before CE) | 1.159 |
| `log_softmax` saved output fp32 | 1.159 |
| grad wrt float logits fp32 | 1.159 |
| **cross-entropy subtotal** | **4.057 (70%)** |
| 24 decoder layers (norms, qkv/o, gate/up/silu) | 1.770 (30%) |

**Why it matters for search:** `gradient_checkpointing` discards only the layer term, so it moves the
ceiling far less than the usual "2-3x" folklore. Predicted ceiling ≈4 under recompute; measured
exactly 2 → 4. The lever that attacks the 70% is chunked/fused linear-cross-entropy (Liger), which
was not implemented in the adapter and so was never reachable.

**How to spot the regime:** `vocab × seq × 4 bytes` comparable to, or larger than, the model state.
Here the embedding alone is 136M of 494M params (~28%), and it is tied.
