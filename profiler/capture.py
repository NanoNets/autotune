"""Capture engine: clean step timing, CPU util, memory, OOM handling, then one profiled step.

The profiler distorts timing, so the real wall-clock comes from a warm, profiler-OFF window —
all clean step times are kept (for the noise floor), and CPU utilization is sampled over that
same window (via psutil, lazily imported). Memory (peak / reserved / headroom) is read from the
torch.cuda allocator. An OOM is CAUGHT and returned as a result (status="oom"), not a crash.
Returns a `Capture`.
"""
import os
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
    error: str = None


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


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


def _cpu_stats(samples):
    """Mean/max CPU % over the sampled window — disambiguates a data-bound GPU (workers help vs
    preprocessing wall). None if psutil is unavailable."""
    if not samples:
        return None
    return {"mean": round(sum(samples) / len(samples), 1),
            "max": round(max(samples), 1), "nproc": os.cpu_count()}


def capture(runner, config, warmup=3, clean_steps=10, trace_path="/tmp/autotune_trace.json"):
    device = config.get("device", "cpu")
    cpu_samples, cpu_stop = [], threading.Event()

    def _cpu_sampler():
        try:
            import psutil
        except Exception:
            return                                     # no psutil -> no cpu data, degrade gracefully
        psutil.cpu_percent()                           # prime the first (throwaway) reading
        while not cpu_stop.is_set():
            cpu_samples.append(psutil.cpu_percent())
            time.sleep(0.05)

    sampler = threading.Thread(target=_cpu_sampler, daemon=True)
    try:
        runner.setup(config)
        for _ in range(warmup):                        # kernel/cache warmup, discarded
            runner.step()
        _sync(device)
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()       # peak is measured over the timed window only

        sampler.start()                                # sample CPU only over the clean window
        times = []                                     # clean wall-clock, profiler off (kept for noise)
        for _ in range(clean_steps):
            t0 = time.perf_counter()
            runner.step()
            _sync(device)
            times.append(time.perf_counter() - t0)
        cpu_stop.set()

        activities = [ProfilerActivity.CPU]            # attribution, one profiled step
        if device == "cuda":
            activities.append(ProfilerActivity.CUDA)
        # Occupancy + device-props come from standard CUDA profiling (no flags). profile_memory /
        # with_flops were tried and dropped — their data isn't in the chrome-trace export.
        with profile(activities=activities) as prof:
            with record_function("step"):
                runner.step()
            _sync(device)
        prof.export_chrome_trace(trace_path)

        return Capture(status="ok", step_times_s=times, trace=trace_path,
                       memory=_read_memory(device), cpu=_cpu_stats(cpu_samples))

    except torch.cuda.OutOfMemoryError as e:           # OOM is a valid result — the memory ceiling
        cpu_stop.set()
        mem = _read_memory(device)
        torch.cuda.empty_cache()
        first = str(e).strip().splitlines()[0] if str(e).strip() else "CUDA out of memory"
        return Capture(status="oom", memory=mem, error=first)
    except Exception as e:                             # any other failure, surfaced not swallowed
        cpu_stop.set()
        return Capture(status="error", error=f"{type(e).__name__}: {e}")
