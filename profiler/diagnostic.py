"""Assemble the Observation from a capture (clean timing + memory) and the profiled trace.

Torch-free: `diagnose` takes a capture-like object (any object with .status, .step_times_s,
.trace, .memory, .error) plus plain numbers, so it's unit-testable without a GPU. Reports raw
signals — MFU (achieved / MEASURED peak, the "utilized well" number), GPU-active, data-wait, idle, noise floor, memory —
plus coverage stats (how much the top-8 kernels represent, idle shape) so the agent can tell
when to drill into the raw trace, and a `trace_path` pointer so it can. No labels, no lever
suggestions.

Every ratio stays inside one population (the Measurement axiom, `loop/contracts.md`). Rates —
`mfu`, `tokens_per_s`, `roofline_floor_s` — come from clean-window TOTALS: mean tokens over mean
step time, never over the median. Trace fractions — `gpu_active_frac`, `gap_s` — divide the
reference batch's busy time by that same batch's profiler-off wall (`reference_step_s`);
`data_wait_frac` divides a host annotation by the wall of the profiled step it was recorded in.
A physically impossible result (a fraction above 1, beyond tolerance) means two populations were
mixed or the instrument failed, so the Observation is marked `invalid` rather than scored.
"""
import statistics
from dataclasses import dataclass, field

from profiler.roofline import training_flops, step_time_floor_s, mfu
from profiler.trace import (load_trace, phase_times_us, gpu_phase_times_us, kernels, gpu_busy_us,
                            kernel_gaps, provenance, device_props, occupancy)

# The phases the fixture runner emits. NOT a filter — every annotation is reported; this is only
# the list `data_wait_s` is pulled from and what the README documents as the common case.
PHASES = ["forward", "backward", "optimizer", "dataloader"]

# How far past 1.0 a fraction may read before it is called impossible. Covers run-to-run jitter
# between the reference batch's two executions (busy is read from one, wall from the other) and
# the measured peak being a floor on the true one (`capture.measure_peak_flops`), not headroom for
# mixing populations — a mixed ratio lands at 1.3-5.6x (smollm2/qwen3.5 A100 runs). Calibrated on
# an A100, 50 traced/untraced pairings per config: healthy GPU-active never passed 0.996, and the
# most GPU-saturated config's mean + 3 sd was 1.017. Starts tight at 2%; revisit if healthy,
# saturated probes get flagged invalid.
_TOLERANCE = 0.02


@dataclass
class Observation:
    status: str = "ok"                 # ok / oom / error / invalid (impossible reading, see error)
    gpu: str = ""                      # hardware LABEL from the input — identifies, never priced
    peak_flops: float = None           # measured dense-GEMM peak on this device (MFU denominator)
    peak_dtype: str = None             # dtype the peak was measured at — MFU is only apples-to-
                                       #   apples when it matches the dtype the model ran in
    peak_mem_bw_gb_s: float = None     # measured achievable bandwidth: the roofline's other axis,
                                       #   for compute-bound vs bandwidth-bound
    step_time_s: float = None          # clean wall-clock per step (profiler off), MEDIAN — a
                                       #   distribution statistic, never the denominator of a rate
    step_time_mean_s: float = None     # clean wall-clock per step, MEAN — what every rate divides by
    step_time_cv: float = None         # noise floor: coefficient of variation across clean steps
    clean_steps_measured: int = None   # clean steps timed (rounded up to whole passes when known)
    clean_steps_requested: int = None  # what the caller asked for; differs when rounding applied
    clean_passes: float = None         # passes COMPLETED over the sample (None: unknown)
    tokens_per_step: float = None      # mean tokens per clean step — the numerator of every rate
    tokens_per_s: float = None         # total clean tokens / total clean time
    flops: float = None                # training FLOPs of the mean clean step (6ND)
    roofline_floor_s: float = None
    mfu: float = None                  # achieved / peak over clean-window totals
    gpu_busy_s: float = None           # reference batch: union of kernel intervals, profiled run
    gpu_active_frac: float = None      # gpu_busy_s / reference_step_s — same batch on both sides
    reference_step_s: float = None     # wall of the reference batch, profiler OFF
    profiled_step_s: float = None      # wall of the same batch, profiler ON; / reference_step_s =
                                       #   profiler overhead
    population: dict = field(default_factory=dict)   # which rows the window timed, which batch the
                                                     #   trace describes (runner.population())
    phase_times_s: dict = field(default_factory=dict)   # HOST dispatch time per annotation
    gpu_phase_times_s: dict = field(default_factory=dict)  # GPU time per annotation (correlation ids)
    gpu_phase_coverage: float = None   # share of summed kernel time attributed to an annotation
    data_wait_s: float = None
    data_wait_frac: float = None
    gap_s: float = None                # reference batch GPU idle (reference_step_s - busy)
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
    provenance: dict = field(default_factory=dict)   # + `runner`: WHICH adapter produced this
                                       #   reading. A belief learned from one runner does not
                                       #   transfer to another — different loop, different
                                       #   bottlenecks (see frameworks/README.md)
    top_kernels: list = field(default_factory=list)   # [[name, seconds], ...] top 8 by total time
    kernel_count: int = None           # distinct kernels; with top8_coverage exposes the long tail
    top8_coverage: float = None        # share of summed kernel time in the top-8 kernels
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


def _clean_timing(times, tokens):
    """Clean-window statistics. {} if nothing was measured.

    `step_time_s` (median) and `step_time_cv` describe the distribution; `step_time_mean_s` and
    `tokens_per_step` (mean over the same steps) are what rates are built from.
    """
    if not times:
        return {}
    wall = statistics.median(times)
    out = {"step_time_s": wall,
           "step_time_mean_s": sum(times) / len(times),
           "clean_steps_measured": len(times),
           "step_time_cv": round(statistics.pstdev(times) / wall, 4) if len(times) > 1 and wall else 0.0}
    if tokens:
        out["tokens_per_step"] = sum(tokens) / len(tokens)
    return out


def diagnose(capture, params, tokens, gpu):
    """capture: an object with .status, .step_times_s, .trace, .memory, .error.

    tokens: tokens per step, used only when the capture carries no per-step counts
    (`capture.step_tokens`) — a runner whose steps are all the same size.
    """
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
        clean_passes=getattr(capture, "clean_passes", None),
        clean_steps_requested=getattr(capture, "clean_steps_requested", None),
        population=getattr(capture, "population", None) or {},
    )

    dev = getattr(capture, "device", None)              # a cpu probe must never read as a real one
    dev_str = f"[{dev.upper()} — not a GPU probe] " if dev and dev != "cuda" else ""

    times = capture.step_times_s or []
    timing = _clean_timing(times, getattr(capture, "step_tokens", None))   # kept even on oom/error:
    if capture.status != "ok" or not times:            # steps that DID run / nothing measured
        note = f"status={capture.status}" + (f": {capture.error}" if capture.error else "")
        return Observation(summary=dev_str + note,
                           provenance={"runner": getattr(capture, "runner", None)},
                           **timing, **base)

    cv, mean_wall = timing["step_time_cv"], timing["step_time_mean_s"]
    step_tok = timing.setdefault("tokens_per_step", tokens)   # per-step counts, else the caller's
    trace = load_trace(capture.trace)
    known = bool(step_tok and params)                  # tokens unknown (a script whose model input
    flops = training_flops(params, step_tok) if known else None   # had no token tensor): no FLOP
    floor = step_time_floor_s(flops, peak) if known else None     # count, so no MFU, rather
    mfu_v = mfu(flops, mean_wall, peak) if known else None        # than a crash or a guess
    tok_s = step_tok / mean_wall if step_tok and mean_wall else None

    ph_s = {k: v / 1e6 for k, v in phase_times_us(trace).items()}   # all of them, not just PHASES
    ks = kernels(trace)                                # all kernels, by name (desc)
    kernel_us = sum(d for _, d in ks)                  # summed per kernel: top-8 share only
    busy = gpu_busy_us(trace) / 1e6                    # union of intervals: the time GPU was busy
    if not ks and dev == "cuda":                       # a CUDA step with no kernels in its trace is
        busy = None                                    # a trace that failed, not an idle GPU
    top8 = [[n, d / 1e6] for n, d in ks[:8]]            # seconds — every duration here is `_s`
    top8_cov = round(sum(d for _, d in ks[:8]) / kernel_us, 3) if kernel_us else None
    gap_count, max_gap_us = kernel_gaps(trace)
    occ = occupancy(trace)

    gpu_ph_us, gpu_ph_cov = gpu_phase_times_us(trace)
    gpu_ph_s = {k: v / 1e6 for k, v in gpu_ph_us.items()}

    ref_wall = getattr(capture, "reference_step_s", None)   # same batch as the trace, profiler off
    profiled = getattr(capture, "profiled_step_s", None)    # same batch, profiler on
    data_wait = ph_s.get("dataloader", 0.0)

    def frac(x, over):                                 # a trace fraction, over ITS OWN batch's wall
        return x / over if over and x is not None else None

    active = frac(busy, ref_wall)
    gap = ref_wall - busy if ref_wall and busy is not None else None   # negative gap: the two
    impossible = []                                    # executions disagree, which is information
    if active is not None and active > 1 + _TOLERANCE:
        impossible.append(f"gpu_active_frac {active:.2f} > 1")
    if mfu_v is not None and mfu_v > 1 + _TOLERANCE:
        impossible.append(f"mfu {mfu_v:.2f} > 1")
    if gap is not None and gap < 0:
        gap = None
    if impossible:                                     # never scored: a population error until
        base = {**base, "status": "invalid",           # proven otherwise (loop/contracts.md)
                "error": "impossible reading: " + "; ".join(impossible)}

    mem_str = f" | mem {mem['peak_gb']}/{mem['total_gb']}GB" if mem else ""
    if cpu.get("mean") is not None:
        cpu_str = f" | cpu {cpu['mean']:.0f}%"
    elif cpu.get("cores_busy") is not None:            # short window: the counter still has it
        cpu_str = f" | cpu {cpu['cores_busy']:.2f} cores"
    else:
        cpu_str = ""
    occ_pct = occ.get("achieved_occupancy_pct")
    occ_str = f" | occ {occ_pct:.0f}%" if occ_pct is not None else ""
    def pct(x):
        return f"{x:.0%}" if x is not None else "n/a"

    wait = frac(data_wait, profiled)                   # host annotation over its own profiled wall
    summary = (f"{dev_str}MFU {pct(mfu_v)} | ref-batch GPU active {pct(active)} | "
               f"data-wait {pct(wait)} | "
               f"idle {pct(frac(gap, ref_wall))}{cpu_str} | noise +/-{cv:.0%}{occ_str}{mem_str}")
    if busy is None:
        summary += " | trace recorded no kernels: GPU signals unknown"
    if impossible:
        summary = f"{dev_str}INVALID ({'; '.join(impossible)}) | " + summary[len(dev_str):]

    return Observation(
        **timing, tokens_per_s=tok_s, flops=flops, roofline_floor_s=floor, mfu=mfu_v,
        gpu_busy_s=busy, gpu_active_frac=active, phase_times_s=ph_s,
        reference_step_s=ref_wall, profiled_step_s=profiled,
        gpu_phase_times_s=gpu_ph_s, gpu_phase_coverage=gpu_ph_cov,
        data_wait_s=data_wait, data_wait_frac=wait, gap_s=gap, gap_frac=frac(gap, ref_wall),
        provenance={**provenance(trace), "runner": getattr(capture, "runner", None)},
        top_kernels=top8,
        kernel_count=len(ks), top8_coverage=top8_cov,
        gap_count=gap_count, max_gap_s=None if max_gap_us is None else max_gap_us / 1e6,
        achieved_occupancy_pct=occ_pct, limiting_factor=occ.get("limiting_factor"),
        device=device_props(trace),
        summary=summary, **base,
    )
