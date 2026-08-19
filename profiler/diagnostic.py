"""Assemble the Observation from a capture (clean timing + memory) and the profiled trace.

Torch-free: `diagnose` takes a capture-like object (any object with .status, .step_times_s,
.trace, .memory, .error) plus plain numbers, so it's unit-testable without a GPU. Reports raw
signals — MFU (the "utilized well" number), GPU-active, data-wait, idle, noise floor, memory —
and leaves interpretation to the reader. No derived labels, no lever suggestions.
"""
import statistics
from dataclasses import dataclass, field

from profiler.roofline import training_flops, step_time_floor_s, mfu
from profiler.trace import load_trace, phase_times_us, kernels, gpu_busy_us, provenance

PHASES = ["forward", "backward", "optimizer", "dataloader"]


@dataclass
class Observation:
    status: str = "ok"                 # ok / oom / error
    gpu: str = ""
    step_time_s: float = None          # clean wall-clock per step (profiler off), median
    step_time_cv: float = None         # noise floor: coefficient of variation across clean steps
    flops: float = None
    roofline_floor_s: float = None
    mfu: float = None                  # achieved / peak — the "is the GPU utilized well" signal
    gpu_busy_s: float = None
    gpu_active_frac: float = None      # gpu_busy / wall (is the GPU busy at all)
    phase_times_s: dict = field(default_factory=dict)
    data_wait_s: float = None
    data_wait_frac: float = None
    gap_s: float = None                # GPU idle (wall - busy)
    gap_frac: float = None
    peak_mem_gb: float = None          # torch.cuda peak allocated
    reserved_mem_gb: float = None      # allocator reserved (reserved - peak = fragmentation)
    mem_total_gb: float = None
    headroom_gb: float = None          # total - reserved (how close to the OOM ceiling)
    provenance: dict = field(default_factory=dict)
    top_kernels: list = field(default_factory=list)
    error: str = None
    summary: str = ""


def diagnose(capture, params, tokens, gpu):
    """capture: an object with .status, .step_times_s, .trace, .memory, .error."""
    mem = capture.memory or {}
    base = dict(
        status=capture.status, gpu=gpu, error=capture.error,
        peak_mem_gb=mem.get("peak_gb"), reserved_mem_gb=mem.get("reserved_gb"),
        mem_total_gb=mem.get("total_gb"), headroom_gb=mem.get("headroom_gb"),
    )

    times = capture.step_times_s or []
    if capture.status != "ok" or not times:            # oom / error / nothing measured
        note = f"status={capture.status}" + (f": {capture.error}" if capture.error else "")
        return Observation(summary=note, **base)

    wall = statistics.median(times)
    cv = round(statistics.pstdev(times) / wall, 4) if len(times) > 1 and wall else 0.0
    trace = load_trace(capture.trace)
    flops = training_flops(params, tokens)
    floor = step_time_floor_s(flops, gpu)
    mfu_v = mfu(flops, wall, gpu)

    ph_s = {k: v / 1e6 for k, v in phase_times_us(trace, PHASES).items()}
    busy = gpu_busy_us(trace) / 1e6
    data_wait = ph_s.get("dataloader", 0.0)
    gap = max(wall - busy, 0.0)

    def frac(x):
        return x / wall if wall else 0.0

    mem_str = f" | mem {mem['peak_gb']}/{mem['total_gb']}GB" if mem else ""
    summary = (f"MFU {mfu_v:.0%} | GPU active {frac(busy):.0%} | data-wait {frac(data_wait):.0%} | "
               f"idle {frac(gap):.0%} | noise +/-{cv:.0%}{mem_str}")

    return Observation(
        step_time_s=wall, step_time_cv=cv, flops=flops, roofline_floor_s=floor, mfu=mfu_v,
        gpu_busy_s=busy, gpu_active_frac=frac(busy), phase_times_s=ph_s,
        data_wait_s=data_wait, data_wait_frac=frac(data_wait), gap_s=gap, gap_frac=frac(gap),
        provenance=provenance(trace), top_kernels=kernels(trace, top=8), summary=summary, **base,
    )
