"""Model-internals tests — normalize an HF config dict (no transformers, no download).

`internals` is a pure function on a config mapping, so we feed it fixture dicts that
match real HF configs and check the derived facts + the analytical param estimate
against known model sizes. `load` (AutoConfig) is the thin shell and isn't tested here.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame.model import internals

# Real config fields (llama family schema). Values are the published ones.
LLAMA2_7B = dict(model_type="llama", architectures=["LlamaForCausalLM"],
                 num_hidden_layers=32, hidden_size=4096, num_attention_heads=32,
                 num_key_value_heads=32, vocab_size=32000, intermediate_size=11008,
                 max_position_embeddings=4096, hidden_act="silu",
                 tie_word_embeddings=False, torch_dtype="float16")

LLAMA3_8B = dict(model_type="llama", architectures=["LlamaForCausalLM"],
                 num_hidden_layers=32, hidden_size=4096, num_attention_heads=32,
                 num_key_value_heads=8, head_dim=128, vocab_size=128256,
                 intermediate_size=14336, max_position_embeddings=8192,
                 hidden_act="silu", tie_word_embeddings=False, torch_dtype="bfloat16")

# Qwen2.5-0.5B ties word embeddings (small models do) — the output embedding must not be double-counted.
QWEN25_05B = dict(model_type="qwen2", architectures=["Qwen2ForCausalLM"],
                  num_hidden_layers=24, hidden_size=896, num_attention_heads=14,
                  num_key_value_heads=2, vocab_size=151936, intermediate_size=4864,
                  max_position_embeddings=32768, hidden_act="silu",
                  tie_word_embeddings=True, torch_dtype="bfloat16")

MIXTRAL_8X7B = dict(model_type="mixtral", architectures=["MixtralForCausalLM"],
                    num_hidden_layers=32, hidden_size=4096, num_attention_heads=32,
                    num_key_value_heads=8, head_dim=128, vocab_size=32000,
                    intermediate_size=14336, num_local_experts=8, num_experts_per_tok=2,
                    max_position_embeddings=32768, hidden_act="silu",
                    tie_word_embeddings=False, torch_dtype="bfloat16")


def _close(a, b, tol=0.01):
    return abs(a - b) / b < tol


def test_llama2_7b_params_and_dims():
    m = internals(LLAMA2_7B)
    assert m["layers"] == 32 and m["hidden"] == 4096 and m["head_dim"] == 128
    assert m["attention"] == "mha"                     # kv_heads == heads
    assert m["gated_mlp"] is True                      # silu
    assert m["moe"] is False
    assert _close(m["params"], 6.738e9)                # published 6.738B
    assert m["params_total"] == m["params"]            # dense: total == active


def test_llama3_8b_gqa():
    m = internals(LLAMA3_8B)
    assert m["kv_heads"] == 8 and m["attention"] == "gqa"
    assert m["dtype"] == "bfloat16" and m["context_len"] == 8192
    assert _close(m["params"], 8.03e9)                 # published 8.03B


def test_mixtral_moe_active_vs_total():
    m = internals(MIXTRAL_8X7B)
    assert m["moe"] is True
    assert m["experts"] == 8 and m["experts_per_tok"] == 2
    assert _close(m["params"], 12.9e9, tol=0.02)       # active-per-token ~12.9B
    assert _close(m["params_total"], 46.7e9, tol=0.02) # total ~46.7B


def test_tied_embeddings_not_double_counted():
    m = internals(QWEN25_05B)
    assert m["attention"] == "gqa" and m["kv_heads"] == 2
    assert _close(m["params"], 0.494e9)                # true 0.494B; untied would over-count by vocab*hidden
    # the same config untied would be ~136M heavier (the lm_head)
    untied = internals({**QWEN25_05B, "tie_word_embeddings": False})
    assert untied["params"] - m["params"] == 151936 * 896


def test_dtype_alias_transformers5():
    m = internals({**QWEN25_05B, "torch_dtype": None, "dtype": "bfloat16"})   # 5.x renamed the field
    assert m["dtype"] == "bfloat16"


def test_mqa_and_defaults():
    cfg = dict(model_type="x", num_hidden_layers=2, hidden_size=64,
               num_attention_heads=8, num_key_value_heads=1, vocab_size=100,
               intermediate_size=256, hidden_act="silu")
    m = internals(cfg)
    assert m["attention"] == "mqa"                     # kv_heads == 1
    assert m["context_len"] is None                    # missing field -> None, no crash
    assert m["kv_heads"] == 1


def test_kv_heads_defaults_to_heads_when_absent():
    cfg = dict(num_hidden_layers=1, hidden_size=64, num_attention_heads=8,
               vocab_size=100, intermediate_size=256)   # no num_key_value_heads
    m = internals(cfg)
    assert m["kv_heads"] == 8 and m["attention"] == "mha"


if __name__ == "__main__":
    for _n, _f in list(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f(); print(f"ok  {_n}")
    print("all passed")


# ---- LoRA adapters, counted from the dims ---------------------------------------------------

from frame.model import lora_params  # noqa: E402

SMOLLM2_360M = dict(model_type="llama", architectures=["LlamaForCausalLM"],
                    num_hidden_layers=32, hidden_size=960, num_attention_heads=15,
                    num_key_value_heads=5, vocab_size=49152, intermediate_size=2560,
                    max_position_embeddings=8192, hidden_act="silu",
                    tie_word_embeddings=True, torch_dtype="bfloat16")


def test_lora_params_match_the_counts_peft_reported():
    """r=16 on all 7 projections: exactly what peft's optimizer held in BYO experiment 7 and on
    the A10."""
    assert lora_params(internals(SMOLLM2_360M), 16, "all-linear") == 8_683_520
    assert lora_params(internals(QWEN25_05B), 16, "all-linear") == 8_798_208
    names = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    assert lora_params(internals(QWEN25_05B), 16, names) == 8_798_208


def test_lora_params_per_projection():
    m = internals(LLAMA3_8B)                            # q 4096->4096, k/v 4096->1024 (GQA)
    assert lora_params(m, 8, ["q_proj"]) == 32 * 8 * (4096 + 4096)
    assert lora_params(m, 8, ["q_proj", "v_proj"]) == 32 * 8 * ((4096 + 4096) + (4096 + 1024))
    assert lora_params(m, 8, ["down_proj"]) == 32 * 8 * (14336 + 4096)


def test_lora_params_unknown_rather_than_guessed():
    assert lora_params(internals(LLAMA3_8B), 8, ["c_attn"]) is None              # GPT-2 naming
    assert lora_params(internals(MIXTRAL_8X7B), 8, ["gate_proj"]) is None        # MoE experts
    assert lora_params(internals(MIXTRAL_8X7B), 8, ["q_proj"]) is not None       # attention is fine
