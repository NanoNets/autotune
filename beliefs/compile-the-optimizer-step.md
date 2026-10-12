---
claim: torch.compile(optimizer.step) fuses the foreach AdamW chain to about one bandwidth-bound pass over params, grads
  and both moments, the same algorithm with <= 1 bf16 ulp differences. It pays in proportion to the optimizer's share of
  the step, which is largest once grad_accum is small and the step is GPU-bound. It also cuts the optimizer's temporary
  memory. Unlike fused=True, which was slower on bf16 params.
type: invariant
context:
  arch_family: dense decoder, full fine-tune, bf16 weights, foreach AdamW (SmolLM2-1.7B, Qwen2.5-1.5B)
  hardware_class: A10, torch 2.7.0
  dataset_shape: any; it matters most at small grad_accum
evidence:
  probes: 2 (a10_smollm2-1.7b_wikitext exp 8 -> 10: +22.0%, same-rows; optimizer phase 0.146 -> 0.050s,
    same-reference-batch; noise floor 0.03% from exp 10/11 repeat; mb16 ceiling trial reserved 21.0 -> 17.6 GB;
    a10_qwen2.5-1.5b_cosmopedia exp 7 -> 8: +3.25% at accum 16, same-rows, optimizer GPU 0.133 -> 0.045s)
  confidence: medium-high (two models; the gain tracks the optimizer's share: 30% of the step -> +22%, 5% -> +3%)
links: [[fused-adamw-slower-on-bf16-params]], [[micro-step-count-bounds-small-real-batches]]
---

hf runner knob `compile_optimizer` (branch run/a10_smollm2-1.7b_wikitext). Equivalence on-box, 5 steps on
1.7B-shaped bf16 tensors: 0.008% of elements differ by 1 bf16 ulp, exp_avg identical, exp_avg_sq <= 1 ulp. No
recompiles with a constant python-float lr (no scheduler). Untested with an lr scheduler, which may need lr as a tensor.
The floor: one pass over 1.71B params of bf16 p/g/m/v is ~24 GB, ~54 ms at 446 GB/s; it measured 50 ms.
