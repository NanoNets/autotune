---
claim: Any zero in the attention mask takes HF SDPA off flash. On GQA it also repeat_kv's K/V and runs the masked
  mem-efficient kernel, whose backward was 7.5x flash's on ~700-token rows. With RIGHT padding a causal LM doesn't need
  the mask: real tokens only attend to earlier, real tokens, and pad positions carry label -100. So drop it (hf runner
  `drop_pad_mask`), or leave no pad (pad_multiple 1 at mb1). The loss is bit-identical and grads differ by less than the
  kernel-swap floor.
type: invariant
context:
  arch_family: HF decoder causal LMs, right padding (Qwen2.5-1.5B, GQA 12/2, head_dim 128)
  hardware_class: A10, torch 2.7.0, transformers 5.15
  dataset_shape: long real rows (cosmopedia, mean ~700 tok); attention matters less on short rows
evidence:
  probes: 2 (a10_qwen2.5-1.5b_cosmopedia exp 3 -> 4: pad_multiple 64 -> 1 at mb1, +17.4% same-rows; attention backward
    fmha_cutlassB 0.414s -> flash 0.055s, width-invariant kernel identity; tokens only -4.4%). Equivalence (not a probe,
    ~/autotune_tools/maskeq.py): loss identical, grads 1.0% median rel-err mask vs no mask, below flash-vs-math 3.8%
    a10_smollm2-360m_lora_script exp 3 (script probe; kernel identity is width-invariant): SmolLM2-360M, GQA 15/5,
    head_dim 64, padded mask, compiled: the attention backward ran fmha_cutlassB (PyTorchMemEffAttention), not flash
  confidence: medium (two models for the kernel identity; one same-rows throughput number)
links: [[compiled-hf-sdpa-never-picks-flash]], [[static-padding-is-the-first-lever]]
---

The runner's drop_pad_mask raises on a batch that isn't right-padded. Left padding (generation) needs the mask.
