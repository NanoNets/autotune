"""Trace-parser tests — deterministic against a hand-built Chrome trace fixture (no torch)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiler.trace import phase_times_us, kernels, gpu_busy_us, provenance

TRACE = {"traceEvents": [
    {"cat": "user_annotation", "name": "forward", "dur": 100.0},
    {"cat": "user_annotation", "name": "backward", "dur": 200.0},
    {"cat": "user_annotation", "name": "optimizer", "dur": 50.0},
    {"cat": "user_annotation", "name": "dataloader", "dur": 300.0},
    {"cat": "kernel", "name": "sm80_gemm_bf16", "dur": 80.0},
    {"cat": "kernel", "name": "flash_attention_fwd", "dur": 40.0},
    {"cat": "kernel", "name": "elementwise_kernel", "dur": 20.0},
]}


def test_phase_times():
    assert phase_times_us(TRACE, ["forward", "backward", "optimizer", "dataloader"]) == {
        "forward": 100.0, "backward": 200.0, "optimizer": 50.0, "dataloader": 300.0}


def test_gpu_busy_sums_kernels():
    assert gpu_busy_us(TRACE) == 140.0


def test_kernels_ranked_top():
    assert kernels(TRACE, top=2) == [("sm80_gemm_bf16", 80.0), ("flash_attention_fwd", 40.0)]


def test_provenance_reads_kernel_names():
    p = provenance(TRACE)
    assert p["flash_attention"] is True
    assert "bf16" in p["dtypes"]


if __name__ == "__main__":
    for _n, _f in list(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f(); print(f"ok  {_n}")
    print("all passed")
