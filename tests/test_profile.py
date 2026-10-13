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


def _ceiling(micro_batch_max=11, seq_len=2048, **kw):
    """A find_ceiling record, hand-made — build() only carries and reads it."""
    return {"seq_len": seq_len, "micro_batch_max": micro_batch_max,
            "fits": micro_batch_max > 0, "capped": False, "oom_at": micro_batch_max + 1,
            "peak_gb": 21.9, "reserved_gb": 22.3, "frag_gb": 0.4, "card_frac": 0.93,
            "config": {"dtype": "bfloat16", "optimizer": "adamw"}, "method": "synthetic",
            "seed": 0, "trials": [], **kw}


def _profile(mem_gb=24.0, dtype="bfloat16", lengths=(256, 512, 1024, 2048), **kw):
    hw = hardware("A10G", peak_flops=1.21e14, mem_gb=mem_gb, dtype=dtype)
    return build(internals(LLAMA3_8B), distribution(list(lengths)),
                 hw, {"torch": "2.4", "transformers": "4.44", "flash_attn": None}, dtype=dtype,
                 **kw)


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
    assert bf16.default_config == {"dtype": "bfloat16", "param_bytes": 2, "optimizer": "adamw",
                                   "finetune": "full"}
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


# --- max_len + the measured OOM ceiling ---------------------------------------------------

def test_max_len_beyond_the_architecture_is_rejected():
    """Caught at Frame, where it is one line of config — not as a crash an hour into the run."""
    try:
        _profile(max_len=99999)                          # LLAMA3_8B fixture: context_len 8192
    except ValueError as e:
        assert "exceeds the model's context length" in str(e)
        return
    assert False, "expected ValueError for max_len > context_len"


def test_ceiling_is_carried_and_bounds_the_summary():
    p = _profile(max_len=2048, ceiling=_ceiling(11))
    assert p.max_len == 2048 and p.ceiling["micro_batch_max"] == 11
    assert "ceiling @2048: micro_batch<=11 (measured)" in p.summary


def test_an_unmeasured_ceiling_says_so_out_loud():
    """Silence would read as 'checked and fine'. It must read as 'not checked'."""
    p = _profile(max_len=2048)
    assert p.ceiling is None and "ceiling NOT measured" in p.summary


def test_nothing_fits_is_flagged_as_a_memory_regime():
    p = _profile(max_len=2048, ceiling=_ceiling(0, fits=False, oom_at=1))
    assert "does NOT fit at micro_batch 1" in p.summary
    assert any("MEMORY regime" in n for n in p.notes)


def test_max_len_dwarfing_the_data_is_flagged_not_rejected():
    """p99 317 vs max_len 2048: mostly padding. The human's call, but the agent must see it."""
    p = _profile(lengths=[16, 40, 317], max_len=2048, ceiling=_ceiling(11))
    assert any("padding" in n and "packing" in n for n in p.notes)


def test_a_ceiling_at_the_edge_of_the_card_warns_about_fragmentation():
    """Fixed-shape probing does not reproduce what varying real lengths do to the allocator."""
    p = _profile(max_len=2048, ceiling=_ceiling(11, card_frac=0.99))
    assert any("margin" in n for n in p.notes)


def test_a_capped_ceiling_says_memory_is_not_the_constraint():
    p = _profile(max_len=2048, ceiling=_ceiling(16, capped=True))
    assert "micro_batch<=16+ (measured)" in p.summary
    assert any("real ceiling is higher" in n for n in p.notes)


def test_the_ceiling_bounds_the_micro_batch_knob():
    """The bound has to land on the KNOB. A framework's knobs() can only declare min:1 — it is
    called with no arguments and cannot know a per-card measurement."""
    hw = hardware("A10G", peak_flops=1.21e14, mem_gb=200.0, dtype="bfloat16")
    p = build(internals(LLAMA3_8B), distribution([256]), hw, {}, dtype="bfloat16",
              knobs={"micro_batch": {"type": "int", "min": 1}, "tf32": {"type": "bool"}},
              max_len=2048, ceiling=_ceiling(11))
    assert p.knobs["micro_batch"]["max"] == 11
    assert p.knobs["micro_batch"]["min"] == 1 and p.knobs["tf32"] == {"type": "bool"}


def test_an_unmeasured_ceiling_leaves_the_knob_unbounded():
    """Never invent a bound. No measurement -> the knob stays as the framework declared it."""
    hw = hardware("A10G", peak_flops=1.21e14, mem_gb=200.0, dtype="bfloat16")
    p = build(internals(LLAMA3_8B), distribution([256]), hw, {}, dtype="bfloat16",
              knobs={"micro_batch": {"type": "int", "min": 1}}, max_len=2048)
    assert "max" not in p.knobs["micro_batch"]


def test_bounding_does_not_mutate_the_frameworks_declaration():
    """knobs() is a shared module-level surface; stamping onto it would leak across runs."""
    k = {"micro_batch": {"type": "int", "min": 1}}
    hw = hardware("A10G", peak_flops=1.21e14, mem_gb=200.0, dtype="bfloat16")
    build(internals(LLAMA3_8B), distribution([256]), hw, {}, dtype="bfloat16",
          knobs=k, max_len=2048, ceiling=_ceiling(11))
    assert k == {"micro_batch": {"type": "int", "min": 1}}


def test_a_clean_profile_carries_no_notes():
    assert _profile(mem_gb=200.0, max_len=2048, ceiling=_ceiling(11)).notes == []


def test_a_multimodal_checkpoint_says_what_isnt_counted():
    vlm = {**internals(LLAMA3_8B), "modality": "multimodal"}
    hw = hardware("A10G", peak_flops=1.21e14, mem_gb=200.0, dtype="bfloat16")
    p = build(vlm, distribution([256]), hw, {}, dtype="bfloat16")
    assert any("vision tower isn't counted" in n for n in p.notes)


if __name__ == "__main__":
    for _n, _f in list(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f(); print(f"ok  {_n}")
    print("all passed")


# ---- LoRA: the adapters are the only trained weights -----------------------------------------

def test_lora_flops_and_memory_count_frozen_and_trained_separately():
    full = _profile()
    lora = _profile(lora={"r": 16, "targets": "all-linear"})
    t = lora.lora["params_trainable"]
    n = full.model["params"] + t
    assert lora.lora["params_total"] == n and t > 0
    assert abs(lora.flops_per_token - (4 * n + 2 * t)) < 1           # the profiler's formula
    assert lora.flops_per_token < full.flops_per_token
    assert lora.memory["model_state_gb"] < full.memory["model_state_gb"] / 3
    assert lora.default_config["finetune"] == "lora" and "LoRA model-state" in lora.summary


def test_lora_flips_the_fit_for_an_8b_on_a_24gb_card():
    assert _profile(mem_gb=24.0).model_state_fits is False                       # full FT: ~120GB
    assert _profile(mem_gb=24.0, lora={"r": 16, "targets": "all-linear"}).model_state_fits is True


def test_lora_targets_it_cant_count_keep_full_numbers_and_say_so():
    full = _profile()
    p = _profile(lora={"r": 16, "targets": ["c_attn"]})
    assert p.lora["params_trainable"] is None and p.flops_per_token == full.flops_per_token
    assert p.memory == full.memory and "full-FT model-state" in p.summary
    assert any("can't be counted" in n for n in p.notes)


# ---- a quantized base (QLoRA) -----------------------------------------------------------------

def test_a_4bit_base_charges_its_linear_weights_at_half_a_byte():
    """With no quantization input a 4-bit base was charged 2 B/weight
    (Qwen2.5-7B: 14.79 GiB vs 5.65 measured on an A10)."""
    lora = {"r": 16, "targets": "all-linear"}
    plain, q4 = _profile(lora=lora), _profile(lora=lora, base_quant="4bit")
    m = internals(LLAMA3_8B)
    linear = m["params"] - m["vocab"] * m["hidden"] * (1 if m["tied"] else 2)   # embed/head: bf16
    saved = linear * (2 - (0.5 + 4 / 64)) / 1024 ** 3
    assert abs((plain.memory["weights_gb"] - q4.memory["weights_gb"]) - saved) < 0.02
    assert q4.default_config["base_quant"] == "4bit" and "QLoRA 4bit base" in q4.summary
    q8 = _profile(lora=lora, base_quant="8bit")
    assert q4.memory["weights_gb"] < q8.memory["weights_gb"] < plain.memory["weights_gb"]


def test_a_quantized_base_needs_the_lora_setup():
    import pytest
    with pytest.raises(ValueError, match="can't be fully fine-tuned"):
        _profile(base_quant="4bit")
    with pytest.raises(ValueError, match="base_quant must be"):
        _profile(lora={"r": 16, "targets": "all-linear"}, base_quant="nf3")
