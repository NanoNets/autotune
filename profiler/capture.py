"""Capture engine: roofline ceilings, clean step timing, CPU util, memory, OOM, profiled step.

Roofline ceilings (peak FLOP/s + memory bandwidth) are MEASURED on the device, never read from a
datasheet table — so any hardware works. They are hardware properties: measure once per run, then
pass them back in (`peak_flops=`, `peak_mem_bw_gb_s=`) for every later probe.

The profiler distorts timing, so the real wall-clock comes from a warm, profiler-OFF window —
all clean step times are kept (for the noise floor), and CPU utilization is sampled over that
same window (via psutil, lazily imported). Memory (peak / reserved / headroom) is read from the
torch.cuda allocator. An OOM is CAUGHT and returned as a result (status="oom"), not a crash.
Returns a `Capture`.

Two populations, never mixed (the Measurement axiom, `loop/contracts.md`):
- the CLEAN WINDOW measures the run. When the runner reports `steps_per_pass`, the window is
  rounded up to whole passes over the sample and started at a pass boundary, so every config
  times exactly the same rows the same number of times;
- the REFERENCE BATCH is what the trace describes. When the runner offers `reference_step()`, it
  is a fixed set of rows chosen by rule, not whichever batch the loader happened to be on. It is
  run once profiler-OFF (its own clean wall, `reference_step_s`) and once profiler-ON (the trace),
  so busy time is only ever divided by the wall of the same batch.
Runners without a reference batch (`has_reference` false: the torch fixture, synthetic ceiling
trials) reuse one fixed batch, so the clean window already times the reference. Its mean step is
`reference_step_s`, and only the traced step is extra.
"""
import math
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field

import torch
from torch.profiler import ProfilerActivity, profile, record_function


@dataclass
class Capture:
    status: str                                       # ok / oom / error
    step_times_s: list = field(default_factory=list)  # clean per-step wall times (profiler off)
    trace: object = None                              # Chrome-trace path (or dict); None on failure
    memory: dict = None                               # {peak_gb, reserved_gb, total_gb, headroom_gb}
    cpu: dict = None                                  # {mean, max, nproc} over the clean window
    step_tokens: list = field(default_factory=list)   # tokens per clean step, when the runner counts
    clean_steps_requested: int = None                 # what the caller asked for, before rounding
    clean_passes: float = None                        # clean steps COMPLETED / steps_per_pass
    reference_step_s: float = None                    # wall of the reference batch, profiler OFF
    profiled_step_s: float = None                     # wall of the same batch, profiler ON (distorted)
    population: dict = None                           # runner.population(): which rows, which batch
    peak_flops: float = None                          # measured dense-GEMM peak (MFU denominator)
    peak_dtype: str = None                            # the dtype that peak was measured at
    peak_mem_bw_gb_s: float = None                    # measured achievable bandwidth (roofline)
    device: str = None                                # the device this actually ran on (cuda/cpu/mps)
    error: str = None


_MIN_SAMPLE_S = 0.02   # below this, psutil's tick deltas quantize to 0.0 — drop, don't record


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def _gemm_flops(device, dtype, size, iters):
    """FLOP/s for one square GEMM shape, timed at steady state. None if it can't run."""
    try:
        a = torch.randn(size, size, device=device, dtype=dtype)
        b = torch.randn(size, size, device=device, dtype=dtype)
        for _ in range(3):                             # warm up autotuning/clocks
            a @ b
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            a @ b
        _sync(device)
        elapsed = time.perf_counter() - t0
        return (2.0 * size ** 3 * iters) / elapsed if elapsed > 0 else None
    except Exception:
        return None
    finally:
        a = b = None
        if device == "cuda":
            torch.cuda.empty_cache()


_PEAK_CACHE = {}   # (device, chip, dtype) -> FLOP/s. Peak is a property of the hardware, not of
                   # the config, so a search re-measuring it every probe burns ~40% per capture.


def measure_peak_flops(device, dtype, sizes=(2048, 4096, 8192), iters=20):
    """Achievable peak FLOP/s: the BEST of several large dense GEMMs at the run's dtype.

    Replaces a datasheet table — this is what the chip actually delivers under its real clocks
    and power cap, for any hardware, named or not. Several shapes because peak is shape-dependent:
    one size can leave throughput on the table and then the model beats it, yielding MFU > 100%.
    Even so this is a measured floor on the true ceiling, so MFU slightly over 1.0 is possible.
    None if nothing ran — a missing denominator beats a wrong one. Cached per (device, chip, dtype);
    pass `capture(..., peak_flops=...)` to skip it when the caller already knows the number.
    """
    chip = torch.cuda.get_device_name() if device == "cuda" else device   # don't share across a fleet
    key = (device, chip, str(dtype))
    if key not in _PEAK_CACHE:
        measured = [f for f in (_gemm_flops(device, dtype, n, iters) for n in sizes) if f]
        _PEAK_CACHE[key] = max(measured) if measured else None
    return _PEAK_CACHE[key]


_BW_CACHE = {}


def measure_peak_bandwidth_gb_s(device, size_mb=256, iters=20):
    """Achievable memory bandwidth (GB/s): a large device-to-device copy, timed at steady state.

    The other half of the roofline. Without it "is this kernel compute-bound or bandwidth-bound?"
    — the first question asked of a low MFU — has no answer, and most transformer training is
    bandwidth-bound (attention, layernorm, elementwise). Buffer is sized well past L2 so it is
    real traffic. A copy moves 2 * nbytes (one read + one write). None if it can't run.
    """
    chip = torch.cuda.get_device_name() if device == "cuda" else device
    if chip in _BW_CACHE:
        return _BW_CACHE[chip]
    a = b = None
    try:
        n = size_mb * 1024 ** 2 // 4                    # float32 elements
        a = torch.empty(n, device=device, dtype=torch.float32)
        b = torch.empty_like(a)
        for _ in range(3):
            b.copy_(a)
        _sync(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            b.copy_(a)
        _sync(device)
        elapsed = time.perf_counter() - t0
        moved_gb = (2.0 * a.numel() * a.element_size() * iters) / 1024 ** 3
        _BW_CACHE[chip] = moved_gb / elapsed if elapsed > 0 else None
    except Exception:
        _BW_CACHE[chip] = None
    finally:
        a = b = None
        if device == "cuda":
            torch.cuda.empty_cache()
    return _BW_CACHE[chip]


def device_facts(device):
    """Static card facts read live off the device: total memory, name, SM count.

    The pre-probe counterpart to `trace.device_props`, which reads the same fields from a trace
    header *after* a probe. `frame` runs before any trace exists, so it reads them live here to
    fill the Profile's hardware section. None-valued off-GPU (CPU/MPS).
    """
    if device != "cuda":
        return {"mem_gb": None, "device_name": device, "num_sms": None}
    p = torch.cuda.get_device_properties(device)
    return {"mem_gb": round(p.total_memory / 1024 ** 3, 1),
            "device_name": p.name, "num_sms": p.multi_processor_count}


def _read_memory(device):
    """Peak / reserved / headroom from the CUDA allocator. None off-GPU (CPU/MPS)."""
    if device != "cuda":
        return None
    gb = 1024 ** 3
    _free, total = torch.cuda.mem_get_info()
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    return {
        "peak_gb": round(peak / gb, 3),
        "reserved_gb": round(reserved / gb, 3),        # reserved - peak = fragmentation
        "total_gb": round(total / gb, 3),
        "headroom_gb": round((total - reserved) / gb, 3),
    }


def _tree_cpu_seconds():
    """Cumulative CPU seconds for this process AND its children (dataloader workers).

    A counter, not a sample: differencing it over the window works at ANY window length, where
    `cpu_percent` sampling quantizes to nothing below ~10ms. Falls back to this process only.
    """
    try:
        import psutil
    except Exception:
        return time.process_time()
    try:
        proc = psutil.Process()
        total = 0.0
        for pr in [proc] + proc.children(recursive=True):
            try:
                t = pr.cpu_times()
                total += t.user + t.system
            except Exception:                          # a worker can exit mid-read
                pass
        return total
    except Exception:
        return time.process_time()


def _cpu_stats(samples):
    """Mean/max CPU % over the sampled window — disambiguates a data-bound GPU (workers help vs
    preprocessing wall). None if psutil is unavailable."""
    if not samples:
        return None
    return {"mean": round(sum(samples) / len(samples), 1),
            "max": round(max(samples), 1), "nproc": os.cpu_count()}


def _default_trace_path():
    """One trace file per capture — a shared path would leave old Observations pointing at it."""
    return os.path.join(tempfile.gettempdir(),
                        f"autotune_trace_{os.getpid()}_{time.time_ns()}.json")


def capture(runner, config, warmup=3, clean_steps=10, trace_path=None, peak_flops=None,
            peak_mem_bw_gb_s=None):
    """Profile one config. Returns a `Capture`.

    peak_flops / peak_mem_bw_gb_s: the roofline ceilings. Measured here when not supplied, but a
    SEARCH SHOULD MEASURE THEM ONCE AND FREEZE THEM: read both off the first probe's Observation
    and pass them into every subsequent probe. They are properties of the hardware, not of the
    config, and re-measuring per probe both re-pays the cost and injects ~28% run-to-run noise
    into a comparison whose real signal is ~1%. A shared constant cancels out of every ranking.
    """
    device = config.get("device", "cpu")
    if trace_path is None:                             # unique per capture — the Observation keeps
        trace_path = _default_trace_path()             # this pointer, so runs must not overwrite
    times = []                                         # hoisted: kept even if a later step OOMs
    cpu_samples, cpu_stop, cpu_ready = [], threading.Event(), threading.Event()

    def _cpu_sampler():
        try:
            import psutil
        except Exception:
            cpu_ready.set()                            # no psutil -> no cpu data, degrade gracefully
            return
        psutil.cpu_percent()                           # prime: psutil keys its baseline by THREAD id,
        cpu_ready.set()                                # so prime in THIS thread, before the window
        last = time.perf_counter()
        while True:
            stopped = cpu_stop.wait(0.05)              # pace, but exit the moment the window closes
            now = time.perf_counter()
            if now - last >= _MIN_SAMPLE_S:            # too short a delta quantizes to a bogus 0.0
                cpu_samples.append(psutil.cpu_percent())   # ...one last real reading on the way out
                last = now
            if stopped:
                return

    sampler = threading.Thread(target=_cpu_sampler, daemon=True)

    def _stop_sampler():
        cpu_stop.set()
        if sampler.is_alive():                         # False if setup failed before start()
            sampler.join(timeout=0.5)                  # let the in-flight reading land

    dtype = config.get("dtype") or torch.get_default_dtype()
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    peak = peak_flops or measure_peak_flops(device, dtype)   # before setup: most memory free
    peak_bw = peak_mem_bw_gb_s or measure_peak_bandwidth_gb_s(device)
    peak_dtype = str(dtype).replace("torch.", "")

    step_tokens, per_pass, requested = [], None, clean_steps

    def passes():                                      # completed, not planned: an OOM mid-window
        return len(times) / per_pass if per_pass else None   # must not report a whole pass

    try:
        runner.setup(config)
        has_ref = getattr(runner, "has_reference", False)
        reference = runner.reference_step if has_ref else runner.step
        count_tokens = getattr(runner, "last_step_tokens", None)
        per_pass = getattr(runner, "steps_per_pass", None)
        if per_pass:                                   # whole passes, so every config times the
            clean_steps = max(1, math.ceil(clean_steps / per_pass)) * per_pass   # same rows
            if clean_steps != requested:
                print(f"autotune: clean_steps {requested} -> {clean_steps} "
                      f"({clean_steps // per_pass} whole pass(es) of {per_pass} steps)",
                      file=sys.stderr)

        for _ in range(warmup):                        # kernel/cache warmup, discarded
            runner.step()
        if has_ref:                                    # warm the reference as long as the loader:
            for _ in range(warmup):                    # under compile, call 1 compiles and call 2
                reference()                            # captures; neither may be the timed one
        if hasattr(runner, "start_pass"):
            runner.start_pass()                        # the window starts at a pass boundary
        _sync(device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()       # peak is measured over the timed window only

        sampler.start()                                # sample CPU only over the clean window...
        cpu_ready.wait(timeout=1.0)                    # ...but not before it has primed
        cpu0, wall0 = _tree_cpu_seconds(), time.perf_counter()
        for _ in range(clean_steps):                   # clean wall-clock, profiler off (kept for noise)
            t0 = time.perf_counter()
            runner.step()
            _sync(device)
            times.append(time.perf_counter() - t0)
            if count_tokens:
                step_tokens.append(count_tokens())
        cpu_busy = _tree_cpu_seconds() - cpu0          # exact over the window, any length
        window = time.perf_counter() - wall0
        _stop_sampler()
        memory = _read_memory(device)                  # the window's peak, before the reference runs

        if has_ref:                                    # the reference batch, profiler OFF: the
            t0 = time.perf_counter()                   # denominator for everything the trace says
            reference()
            _sync(device)
            reference_s = time.perf_counter() - t0
        else:                                          # one fixed batch: the window WAS the reference
            reference_s = sum(times) / len(times) if times else None

        activities = [ProfilerActivity.CPU]            # attribution: the same batch, profiled
        if device == "cuda":
            activities.append(ProfilerActivity.CUDA)
        # Occupancy + device-props come from standard CUDA profiling (no flags). profile_memory /
        # with_flops were tried and dropped — their data isn't in the chrome-trace export.
        with profile(activities=activities) as prof:
            t0 = time.perf_counter()                   # its profiled wall, kept only to expose the
            with record_function("step"):              # profiler's overhead (/ reference_step_s);
                reference()                            # `step` annotation is host-dispatch only
            _sync(device)
            profiled_s = time.perf_counter() - t0
        prof.export_chrome_trace(trace_path)

        cpu = _cpu_stats(cpu_samples) or {"mean": None, "max": None, "nproc": os.cpu_count()}
        cpu["cores_busy"] = round(cpu_busy / window, 3) if window > 0 else None
        population = runner.population() if hasattr(runner, "population") else None

        return Capture(status="ok", step_times_s=times, trace=trace_path,
                       step_tokens=step_tokens, clean_steps_requested=requested,
                       clean_passes=passes(),
                       reference_step_s=reference_s, profiled_step_s=profiled_s,
                       population=population,
                       peak_flops=peak, peak_dtype=peak_dtype, peak_mem_bw_gb_s=peak_bw,
                       memory=memory, cpu=cpu, device=device)

    except torch.cuda.OutOfMemoryError as e:           # OOM is a valid result — the memory ceiling
        _stop_sampler()
        mem = _read_memory(device)
        torch.cuda.empty_cache()
        first = str(e).strip().splitlines()[0] if str(e).strip() else "CUDA out of memory"
        return Capture(status="oom", step_times_s=times, memory=mem,    # steps that DID run are
                       step_tokens=step_tokens, clean_passes=passes(),  # evidence, not noise
                       clean_steps_requested=requested,
                       cpu=_cpu_stats(cpu_samples), peak_flops=peak,
                       peak_dtype=peak_dtype, peak_mem_bw_gb_s=peak_bw, device=device, error=first)
    except Exception as e:                             # any other failure, surfaced not swallowed
        _stop_sampler()
        return Capture(status="error", device=device, error=f"{type(e).__name__}: {e}")
