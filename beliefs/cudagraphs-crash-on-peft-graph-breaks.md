---
claim: torch.compile(mode="reduce-overhead") on a peft-wrapped HF model crashes in training with "accessing tensor output
  of CUDAGraphs that has been overwritten by a subsequent run", at a dynamo resume function after a graph break (peft's
  forward wrapper, transformers' config __getattribute__). torch.compiler.cudagraph_mark_step_begin() before each
  micro-step does not fix it. Don't spend a probe on it until the graph breaks are gone.
type: dead-lever
context:
  arch_family: HF transformers 5.15 decoder + peft 0.20 LoRA, compiled with dynamic=True, grad accumulation
  hardware_class: any (torch 2.7.0)
  dataset_shape: any
evidence:
  probes: 1 (a10_smollm2-360m_lora_script exp 5: two on-box attempts, same error at modeling_llama.py:403; a crash is
    width-invariant)
  confidence: medium (one stack)
links: [[full-gc-after-compile-stalls-one-step]]
---

The launch overhead CUDA graphs would remove was partly recovered by Inductor's C++ wrapper (cpp_wrapper=True; needs
pybind11 headers next to torch): +1.8-2.0% in that run, versus a crash.
