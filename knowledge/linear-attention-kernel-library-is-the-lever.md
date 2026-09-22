---
claim: On a hybrid linear-attention model, `attn_implementation` is a near-dead lever and the
  linear-attention KERNEL LIBRARY is where the equivalent value lives. transformers silently falls
  back to reference PyTorch for those layers, and the fallback is the default state of a fresh box.
type: regime
context:
  arch_family: hybrid linear-attention / SSM-transformer (Qwen3.5, and expect the same shape for
    Mamba-2 hybrids, Jamba, Zamba — anything with a `layer_types` mix and a gated delta rule)
  hardware_class: any CUDA
  dataset_shape: any
evidence:
  probes: 3 (qwen3.5_2b_a100_wikitext experiments 5, 9, 17) + direct resolution of transformers'
    decorator lookup
  confidence: high for the ordering (12x separation); medium for the magnitudes, which are
    sequence-length dependent
links: [[dispatch-count-bounds-small-real-batches]],
  [[padding-regime-flips-attention-verdict]],
  [[compile-unusable-on-hybrid-linear-attention]]
---

Qwen3.5-2B is 24 layers with `full_attention_interval=4`: **18 linear-attention + 6 full
attention**. So `attn_implementation` reaches a quarter of the stack and nothing else.

Measured on one model, one card, one dataset:

| lever | reaches | measured |
|---|---|---|
| `flash-linear-attention` (gated delta rule) | 18 of 24 layers | **+20.7%** |
| `causal-conv1d` | 18 of 24 layers, thin op | -0.5% (noise) |
| `attn_implementation` sdpa vs eager | 6 of 24 layers | -1.7% (noise) |

The prior dense-model run measured sdpa over eager at 5.4%; a quarter of that is ~1.4%, against
1.7% measured. **The dense prior transfers, scaled by layer share.**

**The trap is that the fallback is silent and is the default.** `transformers` decorates these
kernels with `@use_kernel_func_from_hub_with_fallback` (`integrations/hub_kernels.py`), resolving
at import: HF kernels hub, then the original package, then reference PyTorch. Its own source
comment on the torch path: *"significantly slower: for `chunk_gated_delta_rule` the gap is more
than an order of magnitude on an H100."* A fresh GPU box has none of `kernels`, `fla`,
`causal_conv1d` installed, so **the default state of a new box is the slow path in 75% of layers.**

`pip install --no-deps flash-linear-attention fla-core einops` is the whole intervention — a pure
Python wheel over Triton kernels, no CUDA build, and `--no-deps` keeps pip from moving torch or
transformers. `causal-conv1d` is sdist-only and needs a matching nvcc (12.8 worked against torch's
CUDA 12.8), takes a few minutes to build, and on this evidence is not worth the toolchain unless
sequences are long.

**Check provenance by resolving the lookup, not by the absence of a warning.** The warning is
suppressed under `is_torchdynamo_compiling()`, so a compiled run on the slow path prints nothing.
Import `_KERNELS_INTERNAL_PATH_MAPPINGS` and `resolve_internal_import` and print what binds. The
throughput-side tells that the swap engaged: `gap_count` roughly halved, every `ampere_sgemm_*`
(fp32) kernel disappeared, and peak memory fell 2.4GB.
