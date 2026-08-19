"""Assemble the Observation from a clean step time + a profiled trace + the roofline.

Torch-free: operates on the parsed trace and plain numbers, so it's unit-testable without
a GPU. Reports raw fractions (MFU, GPU-active, data-wait, idle) and leaves interpretation
to the reader — no prescriptive lever suggestions.
"""
from profiler.roofline import training_flops, step_time_floor_s, mfu
from profiler.trace import load_trace, phase_times_us, kernels, gpu_busy_us, provenance
from profiler.schema import Observation

PHASES = ["forward", "backward", "optimizer", "dataloader"]


def diagnose(clean_step_time_s, trace, params, tokens, gpu):
    """clean_step_time_s: measured wall-clock (profiler off). trace: dict or path."""
    trace = load_trace(trace)
    flops = training_flops(params, tokens)
    floor = step_time_floor_s(flops, gpu)
    mfu_v = mfu(flops, clean_step_time_s, gpu)

    ph_s = {k: v / 1e6 for k, v in phase_times_us(trace, PHASES).items()}
    busy = gpu_busy_us(trace) / 1e6
    data_wait = ph_s.get("dataloader", 0.0)
    wall = clean_step_time_s
    gap = max(wall - busy, 0.0)

    def frac(x):
        return x / wall if wall else 0.0

    summary = (f"MFU {mfu_v:.0%} | GPU active {frac(busy):.0%} | "
               f"data-wait {frac(data_wait):.0%} | idle {frac(gap):.0%}")

    return Observation(
        status="ok", gpu=gpu, step_time_s=wall,
        flops=flops, roofline_floor_s=floor, mfu=mfu_v,
        gpu_busy_s=busy, gpu_active_frac=frac(busy),
        phase_times_s=ph_s, data_wait_s=data_wait, data_wait_frac=frac(data_wait),
        gap_s=gap, gap_frac=frac(gap),
        provenance=provenance(trace), top_kernels=kernels(trace, top=8),
        summary=summary,
    )
