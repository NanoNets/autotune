"""Diagnostic tests — assemble an Observation from a fixture trace (no torch, no GPU)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiler.diagnostic import diagnose
from profiler.roofline import training_flops, mfu

TRACE = {"traceEvents": [
    {"cat": "user_annotation", "name": "forward", "dur": 100.0},
    {"cat": "user_annotation", "name": "backward", "dur": 200.0},
    {"cat": "user_annotation", "name": "optimizer", "dur": 50.0},
    {"cat": "user_annotation", "name": "dataloader", "dur": 300.0},
    {"cat": "kernel", "name": "sm80_gemm_bf16", "dur": 80.0},
    {"cat": "kernel", "name": "flash_attention_fwd", "dur": 40.0},
]}

PARAMS, TOKENS, GPU = 1_000_000, 1_000, "A100"
STEP_TIME = 1e-3   # 1 ms clean wall-clock


def _obs():
    return diagnose(STEP_TIME, TRACE, params=PARAMS, tokens=TOKENS, gpu=GPU)


def test_trace_fields_flow_through():
    o = _obs()
    assert o.gpu_busy_s == 120e-6                       # 80 + 40 us
    assert o.data_wait_s == 300e-6                      # dataloader annotation
    assert o.phase_times_s["backward"] == 200e-6


def test_mfu_is_consistent_with_roofline():
    o = _obs()
    assert abs(o.mfu - mfu(training_flops(PARAMS, TOKENS), STEP_TIME, GPU)) < 1e-12


def test_fractions_use_wall_clock():
    o = _obs()
    assert abs(o.gpu_active_frac - 120e-6 / STEP_TIME) < 1e-12
    assert abs(o.data_wait_frac - 300e-6 / STEP_TIME) < 1e-12
    assert abs(o.gap_s - (STEP_TIME - 120e-6)) < 1e-12


def test_provenance_present():
    assert _obs().provenance["flash_attention"] is True


if __name__ == "__main__":
    for _n, _f in list(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f(); print(f"ok  {_n}")
    print("all passed")
