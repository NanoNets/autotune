"""Hardware tests — the pure assembler (no torch/GPU). `measure()` is the on-device shell and
isn't unit-tested here; peak comes in as an already-measured number."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame.hardware import hardware, versions


def test_assembles_from_measured_numbers():
    hw = hardware("A10G", peak_flops=1.21e14, mem_gb=24.0, dtype="bfloat16",
                  device_name="NVIDIA A10", num_sms=72)
    assert hw["count"] == 1
    assert hw["peak_flops"] == 1.21e14 and hw["peak_dtype"] == "bfloat16"
    assert hw["mem_gb"] == 24.0 and hw["num_sms"] == 72
    assert hw["device_name"] == "NVIDIA A10"


def test_gpu_is_only_a_label_any_hardware_supported():
    # No validation, no lookup — the peak is measured, so an unknown chip name is fine.
    hw = hardware("some-brand-new-chip", peak_flops=9e14, mem_gb=141.0, dtype="bfloat16")
    assert hw["gpu"] == "some-brand-new-chip"          # echoed verbatim, no KeyError
    assert hw["device_name"] is None                   # optional fields default to None


def test_versions_shape():
    v = versions()
    assert set(v) == {"torch", "transformers", "flash_attn"}   # None-safe, never crashes


if __name__ == "__main__":
    for _n, _f in list(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f(); print(f"ok  {_n}")
    print("all passed")
