---
claim: Naive sequence packing silently CHANGES THE MODEL on a hybrid linear-attention stack —
  documents attend and recur across block boundaries — and neither position_id resets nor passing
  cu_seq_lens to the public forward stopped it on Qwen3.5 / transformers 5.17. Packing is a quality
  risk here, not a free throughput lever.
type: invariant
context:
  arch_family: hybrid linear-attention / SSM-transformer (Qwen3.5; likely any gated-delta or
    Mamba-family hybrid, where the recurrence carries state across the sequence axis)
  hardware_class: any CUDA
  dataset_shape: any — this is a correctness property, not a performance one
evidence:
  probes: 3 direct logit-equivalence tests against standalone forwards, with a determinism control
    (qwen3.5_2b_a100_wikitext, post-run investigation)
  confidence: high that naive packing is unsafe here; MEDIUM that it is unreachable — one plumbing
    route was tested and failed, others may exist
links: [[static-padding-is-usually-the-bottleneck]],
  [[linear-attention-kernel-library-is-the-lever]],
  [[probe-sample-must-match-length-distribution]]
---

Both prior runs listed packing / `padding_free` as **the highest-value unbuilt lever**. Before
building it, test whether the model actually isolates packed documents. On Qwen3.5 it does not.

**Method.** Two random documents A (12 tok) and B (20 tok). Run each alone, then concatenated, and
compare logits. If boundaries are honoured, B's logits in the packed run equal its standalone ones.

| run | doc A max|Δ| | doc B max|Δ| |
|---|---|---|
| packed, naive positions | 0.2891 | **12.94** |
| packed, position_ids reset per doc | 0.2891 | **15.17** |
| packed, explicit `cu_seq_lens_q/k` + `max_length_q/k` | 0.2891 | **16.97** |
| control: identical call twice | — | **0.0000** |

**The control is what makes this conclusive.** The model is bit-deterministic on a repeated call,
so the ~15-17 differences are not sampling noise. Doc A's 0.2891 bounds the *shape-dependent*
numerical variation (a 32-token forward selects different kernel tilings than a 12-token one), and
doc B's difference is two orders of magnitude above it. Doc B is reading doc A.

**Position resets alone made it WORSE, not better** (15.17 vs 12.94). They change RoPE without
changing what any layer may attend to or recur over — so they alter the computation while leaving
the leak in place. That is the trap: it *looks* like the standard packing recipe.

**Where the plumbing stands.** The leaf modules are ready — `Qwen3_5GatedDeltaNet.forward` takes
`**kwargs: Unpack[TransformersKwargs]` and pops `cu_seq_lens_q`, and the decoder layer forwards
`**kwargs` to both `linear_attn` and `self_attn`. transformers even ships
`prepare_fa_kwargs_from_position_ids`. But passing `cu_seq_lens_q/k` into the *public* forward did
not change the result, so something between the wrapper and the layers drops or ignores them. This
is a plumbing gap, not a missing capability — worth one focused attempt next time, instrumented
inside the layer rather than inferred from logits.

**Why this matters beyond one model.** The full-attention half of a hybrid can be fixed with a
block-diagonal mask or FA2 varlen. **The linear-attention half cannot** — the gated delta rule is a
*recurrence* along the sequence, so document isolation requires resetting recurrent state at each
boundary, which no attention mask expresses. Any packing implementation on a hybrid must prove the
recurrence resets, not just that attention is masked.

**Also note what packing would do to `effective_batch`.** Under packing, 16 blocks of 1024 tokens
is not 16 documents — it is however many documents happen to fit. `effective_batch` is human-owned
and quality-bearing, so changing what a "sample" means is a decision to put to the human, not a
knob to turn. Raise it at Step 3, not mid-loop.
