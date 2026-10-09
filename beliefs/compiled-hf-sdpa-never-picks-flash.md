---
claim: Under torch.compile, HF transformers' SDPA attention never gets the flash kernel. While tracing, the causal-mask
  builder skips its is_causal shortcut and materializes a 4D mask even with attention_mask=None, so SDPA runs
  mem-efficient (with repeat_kv'd K/V on GQA). For compiled training, use attn_implementation="flash_attention_2" with
  no mask (right padding + drop_pad_mask, or no pad), which calls flash_attn_func(causal=True) and compiles clean.
type: invariant
context:
  arch_family: HF transformers 5.15 decoders under torch.compile (Qwen2.5-1.5B, GQA head_dim 128)
  hardware_class: A10 (flash-attn 2.8.3), torch 2.7.0
  dataset_shape: any; it matters most on long rows, where attention backward is a real share of the step
evidence:
  probes: 3 (a10_qwen2.5-1.5b_cosmopedia exp 4 -> 5: compile alone -1.9%; exp 6: + attention_mask=None, identical
    -1.9%; exp 7: FA2 + compile + no mask +7.8% over eager SDPA-flash, all same-rows. Kernel identity width-invariant:
    fmha_cutlassB 0.404s in exps 5/6 vs pytorch_flash / flash bwd 0.055s in exps 4/7. Cause read from
    transformers/masking_utils.py `_ignore_causal_mask_sdpa`: `if is_tracing(padding_mask): return False`)
  confidence: high for this transformers version (the cause is in the source); re-check on upgrades
links: [[padded-mask-costs-flash-attention]], [[hf-fa2-padded-batches-break-compile]]
---

Compile's other fusions roughly paid for the slower attention in exp 5 (forward GPU 0.78 -> 0.65s), so a "compile is
neutral" reading hides a ~8% win that only shows once attention is fixed.
