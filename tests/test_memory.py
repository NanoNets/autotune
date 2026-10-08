"""Memory-model tests — the analytical model-state breakdown (pure math)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame.memory import training_memory

P = 7e9   # 7B params


def _close(a, b, tol=0.01):
    return abs(a - b) / b < tol


def test_bf16_adamw_is_8_bytes_per_param():
    """bf16 weights + torch AdamW: the state follows the weights' dtype, and there's no master
    copy unless the training code keeps one (final_boss_report T5-C1: Frame said 16, and the
    repo's own bf16 runner trained Qwen2.5-1.5B at micro-batch 2 where Frame said it didn't fit)."""
    m = training_memory(P, param_bytes=2, optimizer="adamw")
    assert _close(m["model_state_gb"], 8 * P / 1024**3)        # 2 + 2 + (2+2)
    assert _close(m["weights_gb"], 2 * P / 1024**3)
    assert _close(m["optimizer_gb"], 4 * P / 1024**3)          # m + v, bf16
    assert m["master_gb"] == 0


def test_a_master_copy_only_when_the_code_keeps_one():
    m = training_memory(P, param_bytes=2, optimizer="adamw", master_weights=True)
    assert _close(m["master_gb"], 4 * P / 1024**3)


def test_fp32_full_training_no_master():
    m = training_memory(P, param_bytes=4, optimizer="adamw")   # fp32 load + autocast: no master
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


def test_full_fine_tune_is_unchanged_when_everything_trains():
    assert training_memory(P, trainable=P) == training_memory(P)


def test_lora_frozen_base_is_only_stored():
    """bf16 base, fp32 adapters (peft's default): base 2 B/param; adapters 4 + 4 + 8, no master."""
    t = 40e6
    m = training_memory(P, param_bytes=2, trainable=t, trainable_bytes=4)
    assert _close(m["weights_gb"], ((P - t) * 2 + t * 4) / 1024**3)
    assert _close(m["grads_gb"], t * 4 / 1024**3)
    assert _close(m["optimizer_gb"], t * 8 / 1024**3)
    assert m["master_gb"] == 0                         # adapters already fp32
    assert m["model_state_gb"] < training_memory(P)["model_state_gb"] / 3   # ~14GB vs ~52GB
