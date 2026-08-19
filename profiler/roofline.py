"""Roofline: the theoretical floor on step time, MFU, and the GPU spec table.

Training-FLOP estimate: ~6 * N * D, where N = model params and D = tokens processed this
step (forward 2ND + backward 4ND) — the standard MFU estimate (PaLM / nanoGPT). It ignores
the attention term (~12 * layers * d_model * seq per token), a later refinement; for most
fine-tuning shapes 6ND is within a few percent.

GPU_SPECS are vendor datasheet peaks (dense bf16, non-sparse) — the standard MFU denominator.
MFU (achieved / peak), not util%, is the real "is the GPU utilized well" signal.
"""

GPU_SPECS = {
    #          peak dense bf16 TFLOP/s   mem bandwidth GB/s   memory GB
    "A100": {"bf16_tflops": 312.0,  "mem_bw_gb_s": 2039.0, "mem_gb": 80},
    "H100": {"bf16_tflops": 989.4,  "mem_bw_gb_s": 3350.0, "mem_gb": 80},
    "A10G": {"bf16_tflops": 125.0,  "mem_bw_gb_s": 600.0,  "mem_gb": 24},
    "L4":   {"bf16_tflops": 121.0,  "mem_bw_gb_s": 300.0,  "mem_gb": 24},
}


def training_flops(params: int, tokens: int) -> float:
    """Approx training FLOPs to process `tokens` tokens through a `params`-param model."""
    return 6.0 * params * tokens


def step_time_floor_s(flops: float, gpu: str) -> float:
    """Compute-bound lower bound on step time (seconds) at the GPU's peak bf16 rate."""
    peak = GPU_SPECS[gpu]["bf16_tflops"] * 1e12
    return flops / peak


def mfu(flops: float, step_time_s: float, gpu: str) -> float:
    """Model FLOPs Utilization = achieved FLOP/s / peak FLOP/s. In [0, 1]."""
    peak = GPU_SPECS[gpu]["bf16_tflops"] * 1e12
    achieved = flops / step_time_s
    return achieved / peak
