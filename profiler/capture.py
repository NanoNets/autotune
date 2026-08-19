"""Capture engine: measure the clean step time, then a separate profiled step.

The profiler distorts timing, so the real wall-clock comes from a warm, profiler-OFF window
(median over `clean_steps`); attribution comes from ONE profiled step exported as a Chrome
trace. Returns (clean_step_time_s, trace_path) — hand both to `diagnostic.diagnose`.
"""
import statistics
import time

import torch
from torch.profiler import ProfilerActivity, profile, record_function


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def capture(runner, config, warmup=3, clean_steps=10, trace_path="/tmp/autotune_trace.json"):
    runner.setup(config)
    device = config.get("device", "cpu")

    for _ in range(warmup):                          # kernel/cache warmup, discarded
        runner.step()
    _sync(device)

    times = []                                        # clean wall-clock, profiler off
    for _ in range(clean_steps):
        t0 = time.perf_counter()
        runner.step()
        _sync(device)
        times.append(time.perf_counter() - t0)
    clean = statistics.median(times)

    activities = [ProfilerActivity.CPU]               # attribution, one profiled step
    if device == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities) as prof:
        with record_function("step"):
            runner.step()
        _sync(device)
    prof.export_chrome_trace(trace_path)

    return clean, trace_path
