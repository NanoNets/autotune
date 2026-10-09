---
claim: peft keeps LoRA adapter weights in fp32 on a bf16 base (autocast_adapter_dtype), so the adapter GEMMs run as fp32
  SIMT sgemm with no tensor cores. torch.backends.cuda.matmul.allow_tf32 = True moves them onto TF32 tensor-core
  kernels; the base stays bf16 and the loss stays inside run-to-run noise.
type: invariant
context:
  arch_family: HF decoder + peft 0.20 LoRA (r=16, 7 projections), bf16 base
  hardware_class: Ampere+ (A10, CC 8.6), torch 2.7.0
  dataset_shape: any; it pays where the step is GPU-bound
evidence:
  probes: 1 (a10_smollm2-360m_lora_script exps 18 -> 19/21: kernels ampere_sgemm_* -> cutlass tensorop_s1688gemm
    (TF32), width-invariant kernel identity. Loss, 40 compiled steps: tf32 vs no-tf32 max rel 2.43e-3, mean 6.2e-4;
    no-tf32 vs itself max 2.50e-3, mean 5.3e-4. Speed +3.4-4.6% in two interleaved script-probe pairs: an
    observation, not same-rows evidence)
  confidence: medium (one model; the kernel swap is direct)
links: [[padded-mask-costs-flash-attention]]
---

loop/throughput.md lists tf32 as the safe precision subset. Changing the adapters' storage dtype instead is a precision
change to the trained weights: off-limits.
