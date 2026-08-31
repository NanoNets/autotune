"""Roofline tests — hand-verifiable against the 6ND estimate and a measured peak.

Runs with bare python (`python3 tests/test_roofline.py`) or pytest.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiler.roofline import training_flops, step_time_floor_s, mfu

PEAK = 1.25e14          # a measured peak, e.g. what an A10-class card actually delivers


def test_training_flops_is_6ND():
    # forward 2ND + backward 4ND = 6ND
    assert training_flops(params=124_000_000, tokens=1_000) == 6 * 124_000_000 * 1_000


def test_step_time_floor_matches_peak():
    flops = training_flops(124_000_000, 1_000)          # 7.44e11
    assert abs(step_time_floor_s(flops, PEAK) - flops / PEAK) < 1e-15


def test_mfu_at_half_peak_is_50pct():
    flops = training_flops(124_000_000, 1_000)
    step_time = flops / (0.5 * PEAK)                     # a step running at exactly half peak
    assert abs(mfu(flops, step_time, PEAK) - 0.5) < 1e-9


def test_mfu_at_floor_is_100pct():
    flops = training_flops(124_000_000, 1_000)
    floor = step_time_floor_s(flops, PEAK)               # running at the compute-bound floor
    assert abs(mfu(flops, floor, PEAK) - 1.0) < 1e-9


def test_no_peak_yields_none_not_a_wrong_number():
    """If the measurement failed there is no denominator — say so, don't substitute one."""
    assert mfu(1e12, 1.0, None) is None
    assert step_time_floor_s(1e12, None) is None


def test_any_hardware_works():
    """No table, no key: an unnamed chip measured at some peak prices exactly the same."""
    assert abs(mfu(1e12, 1.0, 4e12) - 0.25) < 1e-12


if __name__ == "__main__":
    for _name, _fn in list(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"ok  {_name}")
    print("all passed")
