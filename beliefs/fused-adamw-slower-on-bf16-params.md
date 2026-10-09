---
claim: torch AdamW(fused=True) on bf16 params was SLOWER than the default foreach implementation (optimizer GPU
  phase +29%). Don't expect it to pay without a trace showing the optimizer as a large share of the step.
type: dead-lever
context:
  arch_family: dense decoder, full fine-tune, bf16 weights, no fp32 master (Qwen2.5-0.5B, SmolLM2-1.7B)
  hardware_class: A10, torch 2.7.0
  dataset_shape: any
evidence:
  probes: 2 (a10_qwen2.5-0.5b_wikitext exp 11: -1.4%, same-rows, 7x the noise floor; optimizer 0.042 -> 0.054s;
    a10_smollm2-1.7b_wikitext exp 9: -3.9%, optimizer 0.146 -> 0.178s, same-reference-batch, with the optimizer at 30%
    of the step)
  confidence: medium (two models; one card, one torch version)
links: [[compile-the-optimizer-step]]
---
