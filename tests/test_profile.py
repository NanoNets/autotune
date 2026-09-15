"""Profile-assembly tests — build() from fixture parts (no transformers/datasets/GPU).

The peak + memory arrive as already-measured numbers via a hand-made hardware dict, so build()
is exercised without a GPU. The live frame() (which measures the peak) isn't tested here.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame.model import internals
from frame.hardware import hardware
from frame.dataset import distribution
from frame.profile import build, _framework_knobs

LLAMA3_8B = dict(model_type="llama", architectures=["LlamaForCausalLM"],
                 num_hidden_layers=32, hidden_size=4096, num_attention_heads=32,
                 num_key_value_heads=8, head_dim=128, vocab_size=128256,
                 intermediate_size=14336, max_position_embeddings=8192,
                 hidden_act="silu", torch_dtype="bfloat16")


def _profile(mem_gb=24.0, dtype="bfloat16"):
    hw = hardware("A10G", peak_flops=1.21e14, mem_gb=mem_gb, dtype=dtype)
    return build(internals(LLAMA3_8B), distribution([256, 512, 1024, 2048]),
                 hw, {"torch": "2.4", "transformers": "4.44", "flash_attn": None}, dtype=dtype)


def test_roofline_and_peak_carried():
    p = _profile()
    assert abs(p.flops_per_token - 6 * p.model["params"]) < 1        # 6N
    assert p.hardware["peak_flops"] == 1.21e14                        # measured peak threaded through


def test_fit_uses_measured_mem():
    small = _profile(mem_gb=24.0)                        # 8B * 16B/param ~ 128GB >> 24GB
    assert small.model_state_fits is False and small.activation_headroom_gb < 0
    assert "does NOT fit" in small.summary
    big = _profile(mem_gb=200.0)                         # ~128GB < 200GB
    assert big.model_state_fits is True and "fits" in big.summary


def test_missing_mem_degrades_gracefully():
    p = _profile(mem_gb=None)                            # e.g. a CPU/MPS smoke path
    assert p.model_state_fits is None and p.activation_headroom_gb is None
    assert "fit n/a" in p.summary


def test_dtype_drives_param_bytes_and_memory():
    bf16 = _profile(dtype="bfloat16")
    fp32 = _profile(dtype="float32")
    assert bf16.default_config == {"dtype": "bfloat16", "param_bytes": 2, "optimizer": "adamw"}
    assert fp32.default_config["param_bytes"] == 4
    assert fp32.memory["weights_gb"] > bf16.memory["weights_gb"]     # fp32 weights are heavier


def test_dataset_carried():
    assert _profile().dataset["p50"] == 512


def test_mismatched_peak_dtype_is_rejected():
    """A peak measured at another precision is not a ceiling for this run."""
    hw = hardware("A10G", peak_flops=1.21e14, mem_gb=24.0, dtype="float32")   # peak at fp32...
    try:
        build(internals(LLAMA3_8B), distribution([256]), hw, {}, dtype="bfloat16")   # ...run bf16
    except ValueError as e:
        assert "peak was measured at" in str(e)
        return
    assert False, "expected ValueError for a peak/run dtype mismatch"


def test_bandwidth_is_carried_for_the_loop_to_freeze():
    hw = hardware("A10G", peak_flops=1.21e14, mem_gb=24.0, dtype="bfloat16",
                  peak_mem_bw_gb_s=428.0)
    p = build(internals(LLAMA3_8B), distribution([256]), hw, {}, dtype="bfloat16")
    assert p.hardware["peak_mem_bw_gb_s"] == 428.0    # probes must not re-measure this


def test_effective_batch_and_knobs_carried():
    hw = hardware("A10G", peak_flops=1.21e14, mem_gb=200.0, dtype="bfloat16")
    p = build(internals(LLAMA3_8B), distribution([256]), hw, {}, dtype="bfloat16",
              effective_batch=32, knobs={"micro_batch": {"type": "int"}})
    assert p.effective_batch == 32 and "eff-batch 32" in p.summary
    assert "micro_batch" in p.knobs                    # the framework's searchable surface, in the Profile


def test_framework_knobs_reads_the_reconciled_surface():
    k = _framework_knobs("hf")                          # safe levers only; run params excluded
    assert "micro_batch" in k and "tf32" in k
    assert "dtype" not in k and "seq_len" not in k and "grad_accum" not in k


if __name__ == "__main__":
    for _n, _f in list(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f(); print(f"ok  {_n}")
    print("all passed")
