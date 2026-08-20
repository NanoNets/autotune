"""Trace-parser tests — deterministic against a hand-built Chrome trace fixture (no torch)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiler.trace import phase_times_us, kernels, gpu_busy_us, provenance, kernel_gaps

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


def test_kernel_gaps():
    # k1 [0,80], k2 [100,140] -> one 20us gap; k3 [130,160] overlaps k2 -> no gap
    trace = {"traceEvents": [
        {"cat": "kernel", "name": "k1", "ts": 0.0,   "dur": 80.0},
        {"cat": "kernel", "name": "k2", "ts": 100.0, "dur": 40.0},
        {"cat": "kernel", "name": "k3", "ts": 130.0, "dur": 30.0},
    ]}
    assert kernel_gaps(trace) == (1, 20.0)
    assert kernel_gaps(TRACE) == (0, 0.0)          # fixture kernels have no ts -> no timeline


if __name__ == "__main__":
    for _n, _f in list(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f(); print(f"ok  {_n}")
    print("all passed")
