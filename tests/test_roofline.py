"""Roofline tests — hand-verifiable against the 6ND estimate and datasheet peaks.

Runs with bare python (`python3 tests/test_roofline.py`) or pytest.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiler.roofline import training_flops, step_time_floor_s, mfu, GPU_SPECS


def test_training_flops_is_6ND():
    # forward 2ND + backward 4ND = 6ND
    assert training_flops(params=124_000_000, tokens=1_000) == 6 * 124_000_000 * 1_000


def test_step_time_floor_matches_peak():
    flops = training_flops(124_000_000, 1_000)          # 7.44e11
    peak = GPU_SPECS["A100"]["bf16_tflops"] * 1e12       # 3.12e14 FLOP/s
    assert abs(step_time_floor_s(flops, "A100") - flops / peak) < 1e-15


def test_mfu_at_half_peak_is_50pct():
    flops = training_flops(124_000_000, 1_000)
    peak = GPU_SPECS["A100"]["bf16_tflops"] * 1e12
    step_time = flops / (0.5 * peak)                     # a step running at exactly half peak
    assert abs(mfu(flops, step_time, "A100") - 0.5) < 1e-9


def test_mfu_at_floor_is_100pct():
    flops = training_flops(124_000_000, 1_000)
    floor = step_time_floor_s(flops, "A100")             # running at the compute-bound floor
    assert abs(mfu(flops, floor, "A100") - 1.0) < 1e-9


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok  {_name}")
    print("all passed")
