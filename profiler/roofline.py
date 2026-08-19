"""Roofline: the theoretical floor on step time, and MFU.

Training-FLOP estimate: ~6 * N * D, where N = model params and D = tokens processed this
step (forward 2ND + backward 4ND) — the standard MFU estimate (PaLM / nanoGPT). It ignores
the attention term (~12 * layers * d_model * seq per token), a later refinement; for most
fine-tuning shapes 6ND is within a few percent.
"""
from profiler.specs import GPU_SPECS


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
