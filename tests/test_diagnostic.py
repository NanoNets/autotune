"""Diagnostic tests — assemble an Observation from a fixture capture (no torch, no GPU).

`diagnose` is duck-typed on the capture (any object with .status/.step_times_s/.trace/.memory/
.error), so we hand it a SimpleNamespace and never import the torch-backed capture engine.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiler.diagnostic import diagnose
from profiler.roofline import training_flops, mfu

TRACE = {"traceEvents": [
    {"cat": "user_annotation", "name": "forward", "dur": 100.0},
    {"cat": "user_annotation", "name": "backward", "dur": 200.0},
    {"cat": "user_annotation", "name": "optimizer", "dur": 50.0},
    {"cat": "user_annotation", "name": "dataloader", "dur": 300.0},
    {"cat": "kernel", "name": "sm80_gemm_bf16", "dur": 80.0, "ts": 0.0},     # [0, 80]
    {"cat": "kernel", "name": "flash_attention_fwd", "dur": 40.0, "ts": 100.0},  # [100,140] -> 20us gap
]}

PARAMS, TOKENS, GPU = 1_000_000, 1_000, "A100"
STEP_TIMES = [1.0e-3, 1.1e-3, 0.9e-3]   # median 1.0 ms, with spread -> nonzero noise floor
MEMORY = {"peak_gb": 12.0, "reserved_gb": 14.0, "total_gb": 80.0, "headroom_gb": 66.0}


def _cap(status="ok", step_times=None, trace=TRACE, memory=MEMORY, error=None):
    return SimpleNamespace(
        status=status, trace=trace, memory=memory, error=error,
        step_times_s=STEP_TIMES if step_times is None else step_times)


def _obs():
    return diagnose(_cap(), params=PARAMS, tokens=TOKENS, gpu=GPU)


def test_trace_fields_flow_through():
    o = _obs()
    assert o.gpu_busy_s == 120e-6                       # 80 + 40 us
    assert o.data_wait_s == 300e-6
    assert o.phase_times_s["backward"] == 200e-6


def test_mfu_uses_median_step_time():
    o = _obs()
    assert abs(o.mfu - mfu(training_flops(PARAMS, TOKENS), 1.0e-3, GPU)) < 1e-12   # median = 1.0 ms


def test_noise_floor():
    assert _obs().step_time_cv > 0                      # spread -> nonzero cv
    o1 = diagnose(_cap(step_times=[1.0e-3]), params=PARAMS, tokens=TOKENS, gpu=GPU)
    assert o1.step_time_cv == 0.0                       # single sample -> 0


def test_memory_flows_through():
    o = _obs()
    assert o.peak_mem_gb == 12.0
    assert o.mem_total_gb == 80.0
    assert o.headroom_gb == 66.0


def test_coverage_and_idle_shape():
    o = _obs()
    assert o.kernel_count == 2
    assert o.top8_coverage == 1.0                       # both kernels are in the top-8
    assert o.gap_count == 1                             # one idle gap on the timeline
    assert abs(o.max_gap_s - 20e-6) < 1e-12             # the 20 us gap


def test_oom_short_circuits_but_keeps_memory_and_trace_path():
    o = diagnose(_cap(status="oom", step_times=[], trace="/tmp/probe.json", error="CUDA out of memory"),
                 params=PARAMS, tokens=TOKENS, gpu=GPU)
    assert o.status == "oom"
    assert o.mfu is None                                # no throughput computed on OOM
    assert o.peak_mem_gb == 12.0                        # but memory is still surfaced
    assert o.trace_path == "/tmp/probe.json"            # and the raw-trace pointer is kept
    assert "oom" in o.summary


def test_provenance_present():
    assert _obs().provenance["flash_attention"] is True


if __name__ == "__main__":
    for _n, _f in list(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f(); print(f"ok  {_n}")
    print("all passed")
