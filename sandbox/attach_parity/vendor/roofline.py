"""Roofline: the theoretical floor on step time and MFU, against a MEASURED peak.

Training-FLOP estimate: ~6 * N * D, where N = model params and D = tokens processed this
step (forward 2ND + backward 4ND) — the standard MFU estimate (PaLM / nanoGPT). It ignores
the attention term (~12 * layers * d_model * seq per token), a later refinement; for most
fine-tuning shapes 6ND is within a few percent.

The denominator is measured on the actual device (see `capture.measure_peak_flops`), not read
from a datasheet table: a table can't tell an A10 from an A10G, quietly quotes sparse numbers,
and hard-fails on any chip it has never heard of. A measured peak also prices in clock
throttling and whatever the cloud is really giving us. This module stays torch-free — it just
takes the number.
"""


def training_flops(params: int, tokens: int) -> float:
    """Approx training FLOPs to process `tokens` tokens through a `params`-param model."""
    return 6.0 * params * tokens


def step_time_floor_s(flops: float, peak_flops: float) -> float:
    """Compute-bound lower bound on step time (seconds) at the device's measured peak."""
    return flops / peak_flops if peak_flops else None


def mfu(flops: float, step_time_s: float, peak_flops: float) -> float:
    """Model FLOPs Utilization = achieved FLOP/s / measured peak FLOP/s. In [0, 1]."""
    if not peak_flops or not step_time_s:
        return None
    return (flops / step_time_s) / peak_flops
