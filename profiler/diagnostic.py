"""Assemble the Observation from a capture (clean timing + memory) and the profiled trace.

Torch-free: `diagnose` takes a capture-like object (any object with .status, .step_times_s,
.trace, .memory, .error) plus plain numbers, so it's unit-testable without a GPU. Reports raw
signals — MFU (achieved / MEASURED peak, the "utilized well" number), GPU-active, data-wait, idle, noise floor, memory —
plus coverage stats (how much the top-8 kernels represent, idle shape) so the agent can tell
when to drill into the raw trace, and a `trace_path` pointer so it can. No labels, no lever
suggestions.
"""
import statistics
from dataclasses import dataclass, field

from profiler.roofline import training_flops, step_time_floor_s, mfu
from profiler.trace import (load_trace, phase_times_us, gpu_phase_times_us, kernels,
                            kernel_gaps, provenance, device_props, occupancy)

# The phases the fixture runner emits. NOT a filter — every annotation is reported; this is only
# the list `data_wait_s` is pulled from and what the README documents as the common case.
PHASES = ["forward", "backward", "optimizer", "dataloader"]


@dataclass
class Observation:
    status: str = "ok"                 # ok / oom / error
    gpu: str = ""                      # hardware LABEL from the input — identifies, never priced
    peak_flops: float = None           # measured dense-GEMM peak on this device (MFU denominator)
    peak_dtype: str = None             # dtype the peak was measured at — MFU is only apples-to-
                                       #   apples when it matches the dtype the model ran in
    peak_mem_bw_gb_s: float = None     # measured achievable bandwidth: the roofline's other axis,
                                       #   for compute-bound vs bandwidth-bound
    step_time_s: float = None          # clean wall-clock per step (profiler off), median
    step_time_cv: float = None         # noise floor: coefficient of variation across clean steps
    flops: float = None
    roofline_floor_s: float = None
    mfu: float = None                  # achieved / peak — the "is the GPU utilized well" signal
    gpu_busy_s: float = None
    gpu_active_frac: float = None      # gpu_busy / wall (is the GPU busy at all)
    profiled_step_s: float = None      # wall of the profiled step; / step_time_s = profiler overhead
    phase_times_s: dict = field(default_factory=dict)   # HOST dispatch time per annotation
    gpu_phase_times_s: dict = field(default_factory=dict)  # GPU time per annotation (correlation ids)
    gpu_phase_coverage: float = None   # fraction of gpu_busy_s attributed to some annotation
    data_wait_s: float = None
    data_wait_frac: float = None
    gap_s: float = None                # GPU idle (wall - busy)
    gap_frac: float = None
    cpu_util_mean: float = None        # over the clean window; GPU-idle + CPU pinned = preprocessing wall
    cpu_util_max: float = None
    cpu_cores_busy: float = None       # CPU-seconds burned per wall-second by this process tree
                                       #   (dataloader workers included). A counter delta, so it
                                       #   survives windows too short to sample. >1 = multi-core.
    nproc: int = None
    peak_mem_gb: float = None          # torch.cuda peak allocated
    reserved_mem_gb: float = None      # allocator reserved (reserved - peak = fragmentation)
    mem_total_gb: float = None
    headroom_gb: float = None          # total - reserved (how close to the OOM ceiling)
    provenance: dict = field(default_factory=dict)
    top_kernels: list = field(default_factory=list)   # [[name, seconds], ...] top 8 by total time
    kernel_count: int = None           # distinct kernels; with top8_coverage exposes the long tail
    top8_coverage: float = None        # fraction of GPU-busy in the top-8 kernels
    gap_count: int = None              # idle-shape: number of gaps on the GPU-kernel timeline
    max_gap_s: float = None            #   largest single gap (one big stall vs many small)
    achieved_occupancy_pct: float = None   # runtime-weighted; the "why is compute slow" signal
    limiting_factor: str = None            # what caps occupancy (REGS / shared mem / warps)
    device: dict = field(default_factory=dict)   # real hw from the trace: name, mem, sms, cc
    device_type: str = None            # what it RAN on (cuda/cpu/mps). A cpu probe is a smoke test:
                                       #   MFU and every GPU signal are meaningless — never compare
                                       #   it against a cuda probe, and never learn a belief from it
    trace_path: str = None             # raw trace on disk, for on-demand drill-down
    error: str = None
    summary: str = ""


def _clean_timing(times):
    """Median wall-clock + noise floor over the clean steps. {} if nothing was measured."""
    if not times:
        return {}
    wall = statistics.median(times)
    return {"step_time_s": wall,
            "step_time_cv": round(statistics.pstdev(times) / wall, 4) if len(times) > 1 and wall else 0.0}


def diagnose(capture, params, tokens, gpu):
    """capture: an object with .status, .step_times_s, .trace, .memory, .error."""
    mem = capture.memory or {}
    cpu = getattr(capture, "cpu", None) or {}
    trace_path = capture.trace if isinstance(capture.trace, str) else None
    peak = getattr(capture, "peak_flops", None)
    base = dict(
        status=capture.status, gpu=gpu, error=capture.error, trace_path=trace_path,
        peak_flops=peak, peak_dtype=getattr(capture, "peak_dtype", None),
        peak_mem_bw_gb_s=getattr(capture, "peak_mem_bw_gb_s", None),
        device_type=getattr(capture, "device", None),
        peak_mem_gb=mem.get("peak_gb"), reserved_mem_gb=mem.get("reserved_gb"),
        mem_total_gb=mem.get("total_gb"), headroom_gb=mem.get("headroom_gb"),
        cpu_util_mean=cpu.get("mean"), cpu_util_max=cpu.get("max"), nproc=cpu.get("nproc"),
        cpu_cores_busy=cpu.get("cores_busy"),
    )

    dev = getattr(capture, "device", None)              # a cpu probe must never read as a real one
    dev_str = f"[{dev.upper()} — not a GPU probe] " if dev and dev != "cuda" else ""

    times = capture.step_times_s or []
    timing = _clean_timing(times)                      # kept even on oom/error: steps that DID run
    if capture.status != "ok" or not times:            # oom / error / nothing measured
        note = f"status={capture.status}" + (f": {capture.error}" if capture.error else "")
        return Observation(summary=dev_str + note, **timing, **base)

    wall, cv = timing["step_time_s"], timing["step_time_cv"]
    trace = load_trace(capture.trace)
    flops = training_flops(params, tokens)
    floor = step_time_floor_s(flops, peak)
    mfu_v = mfu(flops, wall, peak)

    ph_s = {k: v / 1e6 for k, v in phase_times_us(trace).items()}   # all of them, not just PHASES
    ks = kernels(trace)                                # all kernels, by name (desc)
    busy_us = sum(d for _, d in ks)
    busy = busy_us / 1e6
    top8 = [[n, d / 1e6] for n, d in ks[:8]]            # seconds — every duration here is `_s`
    top8_cov = round(sum(d for _, d in ks[:8]) / busy_us, 3) if busy_us else None
    gap_count, max_gap_us = kernel_gaps(trace)
    occ = occupancy(trace)

    gpu_ph_us, gpu_ph_cov = gpu_phase_times_us(trace)
    gpu_ph_s = {k: v / 1e6 for k, v in gpu_ph_us.items()}

    data_wait = ph_s.get("dataloader", 0.0)
    gap = wall - busy                                  # null, not 0: a negative gap means the two
    if gap < 0:                                        # sources disagree, which is information
        gap = None

    def frac(x):
        return x / wall if wall and x is not None else None

    mem_str = f" | mem {mem['peak_gb']}/{mem['total_gb']}GB" if mem else ""
    if cpu.get("mean") is not None:
        cpu_str = f" | cpu {cpu['mean']:.0f}%"
    elif cpu.get("cores_busy") is not None:            # short window: the counter still has it
        cpu_str = f" | cpu {cpu['cores_busy']:.2f} cores"
    else:
        cpu_str = ""
    occ_pct = occ.get("achieved_occupancy_pct")
    occ_str = f" | occ {occ_pct:.0f}%" if occ_pct is not None else ""
    idle_str = f"{frac(gap):.0%}" if gap is not None else "n/a"
    mfu_str = f"{mfu_v:.0%}" if mfu_v is not None else "n/a"
    summary = (f"{dev_str}MFU {mfu_str} | GPU active {frac(busy):.0%} | "
               f"data-wait {frac(data_wait):.0%} | "
               f"idle {idle_str}{cpu_str} | noise +/-{cv:.0%}{occ_str}{mem_str}")

    return Observation(
        step_time_s=wall, step_time_cv=cv, flops=flops, roofline_floor_s=floor, mfu=mfu_v,
        gpu_busy_s=busy, gpu_active_frac=frac(busy), phase_times_s=ph_s,
        profiled_step_s=getattr(capture, "profiled_step_s", None),
        gpu_phase_times_s=gpu_ph_s, gpu_phase_coverage=gpu_ph_cov,
        data_wait_s=data_wait, data_wait_frac=frac(data_wait), gap_s=gap, gap_frac=frac(gap),
        provenance=provenance(trace), top_kernels=top8,
        kernel_count=len(ks), top8_coverage=top8_cov,
        gap_count=gap_count, max_gap_s=None if max_gap_us is None else max_gap_us / 1e6,
        achieved_occupancy_pct=occ_pct, limiting_factor=occ.get("limiting_factor"),
        device=device_props(trace),
        summary=summary, **base,
    )
