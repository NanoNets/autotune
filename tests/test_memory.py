"""Memory-model tests — the analytical model-state breakdown (pure math)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame.memory import training_memory

P = 7e9   # 7B params


def _close(a, b, tol=0.01):
    return abs(a - b) / b < tol


def test_bf16_adamw_is_16_bytes_per_param():
    m = training_memory(P, param_bytes=2, optimizer="adamw")   # master_weights defaults on
    # ZeRO accounting: 2 + 2 + (4+4) optimizer + 4 master = 16 bytes/param
    assert _close(m["model_state_gb"], 16 * P / 1024**3)
    assert _close(m["weights_gb"], 2 * P / 1024**3)
    assert _close(m["optimizer_gb"], 8 * P / 1024**3)          # m + v, fp32
    assert _close(m["master_gb"], 4 * P / 1024**3)


def test_fp32_full_training_no_master():
    m = training_memory(P, param_bytes=4, optimizer="adamw")   # master off at fp32
    assert m["master_gb"] == 0
    assert _close(m["model_state_gb"], 16 * P / 1024**3)       # 4 + 4 + 8


def test_adamw_8bit_shrinks_optimizer():
    m = training_memory(P, param_bytes=2, optimizer="adamw_8bit")
    assert _close(m["optimizer_gb"], 2 * P / 1024**3)          # m + v, int8


def test_sgd_nomomentum_has_no_optimizer_state():
    m = training_memory(P, param_bytes=2, optimizer="sgd_nomomentum")
    assert m["optimizer_gb"] == 0


if __name__ == "__main__":
    for _n, _f in list(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f(); print(f"ok  {_n}")
    print("all passed")
