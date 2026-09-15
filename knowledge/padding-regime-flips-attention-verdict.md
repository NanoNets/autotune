---
claim: flash_attention_2 beats sdpa only while batches carry heavy padding; once padding is removed at
  the loader, its unpad/repad overhead makes it a net loss.
type: regime
context:
  arch_family: any transformer using HF attn_implementation
  hardware_class: any CUDA with flash-attn installed
  dataset_shape: short sequences declared at a much longer max_len (padding-heavy)
evidence:
  probes: 4 (experiment 6 FA2 win; 11 and 13 sdpa; 9 and 12 FA2 under dynamic padding)
  confidence: medium-high — two samples per arm, non-overlapping ranges, but measured at
    pad_multiple=64 and never re-confirmed on the final pad_multiple=8 config
links: [[static-padding-is-usually-the-bottleneck]]
---

**With `padding="max_length"`** (~97% pad): FA2 won **+40.2%** (12940 → 18136 tokens/s). `sdpa` was
falling back to cutlass *memory-efficient* attention (`fmha_cutlassB/F`) because a padding mask
disqualifies its flash backend. Transformers' FA2 path unpads via `flash_attn_varlen_func`, so
attention went from 30.9% of GPU time to absent from the top-8 entirely.

**With `padding="longest"`**: `sdpa` beat FA2 by **+34%** (52.59 vs 39.19 samples/s, two samples each).
The padding FA2 existed to skip was already gone, while its unpad/repad round trip still costs extra
kernel launches — expensive in the launch-bound regime that dynamic padding creates.

**The methodological point is the transferable one:** a winner must be re-tested when the regime that
made it win changes. This is a discipline for *catching* flips, not an expectation that everything
flips — `torch.compile` was re-tested the same way and held (-43% without it).

**Known gap:** the verdict rests on probes taken at `pad_multiple=64`. The re-test on the final
`pad_multiple=8` config (experiment 22) timed out inconclusively, so the ordering is assumed, not
measured, at the shipped setting.
