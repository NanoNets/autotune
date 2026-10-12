---
claim: torch.compile mode max-autotune-no-cudagraphs did not beat inductor's default on A10. Its Triton GEMM templates
  lost or tied cuBLAS/cutlass on both skinny (~900 x 896) and mid-size (~1800 x 2048/8192) training GEMMs, and autotuning
  every dynamic shape costs minutes of wall time.
type: dead-lever
context:
  arch_family: dense decoder, full fine-tune (Qwen2.5-0.5B, SmolLM2-1.7B)
  hardware_class: A10, torch 2.7.0
  dataset_shape: dynamic widths under padding=longest
evidence:
  probes: 2 (a10_qwen2.5-0.5b_wikitext exp 12: -0.7%, 351s wall; a10_smollm2-1.7b_wikitext exp 16: -0.5%, 509s wall,
    GPU step 0.365 -> 0.367s same-reference-batch), basis same-rows
  confidence: medium
links: []
---
