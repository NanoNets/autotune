---
claim: A script that loads with from_pretrained and never calls model.train() trains its HF layers in eval mode. peft's
  new LoRA modules default to train mode, so adapter dropout still runs and nothing looks wrong, but transformers gates
  gradient checkpointing on `self.training`, so gradient_checkpointing_enable() is a silent no-op. Check the decoder
  layers' `.training`, or a ceiling trial with and without it, before trusting a recompute result on a user script.
type: safety
context:
  arch_family: HF transformers 5.15 decoder LMs + peft 0.20 LoRA, user scripts with a plain training loop
  hardware_class: any
  dataset_shape: any
evidence:
  probes: 1 (a10_smollm2-360m_lora_script: ceiling trial at mb8, same ceiling rows: 19.92 GB with
    gradient_checkpointing_enable() vs 19.87 without; after model.train() mb16 fit at 7.80 GB where it had OOMed.
    On-box: from_pretrained -> training False; after get_peft_model the PeftModel says True, layer 0 says False).
    Basis: memory trials on identical rows + module flags (width-invariant)
  confidence: high for the mechanism (transformers' GradientCheckpointingLayer checks self.training)
links: [[recompute-pays-when-it-cuts-micro-steps]]
---

Calling model.train() changes nothing numerically only when the base has no active dropout (SmolLM2: attention_dropout
0.0). Assert that before flipping it; otherwise it is a model change, not a speed knob, and goes to the user. Worth
reporting to the user either way: their script may not be training the way they think.
