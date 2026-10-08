"""Exact counts from a meta-device build (frame.model.with_exact_counts), on tiny configs built
locally: no hub, no weights. Ground truth is the same model instantiated for real on CPU, and peft
itself for LoRA when it's installed (final_boss_report T7-C1, T7-C3, VLM theme)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

transformers = pytest.importorskip("transformers")
import torch  # noqa: E402

from frame.model import internals, lora_trainable, with_exact_counts, build_meta  # noqa: E402

SMALL = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2, num_attention_heads=4,
             num_key_value_heads=2, vocab_size=256, max_position_embeddings=128)


def _configs():
    return {
        "llama": transformers.LlamaConfig(**SMALL),
        # Gemma-3: gated, but the activation lives in `hidden_activation` (the template misses it)
        "gemma3": transformers.Gemma3TextConfig(**SMALL, head_dim=16),
        # Phi-3: fused qkv_proj / gate_up_proj
        "phi3": transformers.Phi3Config(**SMALL, pad_token_id=0),
        # Qwen2-VL: a vision tower the text template can't see
        "qwen2vl": transformers.Qwen2VLConfig(
            text_config={**SMALL, "rope_scaling": {"type": "mrope", "mrope_section": [2, 2, 4]}},
            vision_config=dict(depth=2, embed_dim=32, hidden_size=64, num_heads=2, mlp_ratio=2,
                               patch_size=14, spatial_merge_size=2, in_channels=3)),
    }


def _real(config):
    """The same model, instantiated for real: ground truth."""
    m = build_meta(config)
    real = type(m)._from_config(config)
    return real, sum(p.numel() for p in real.parameters())


@pytest.mark.parametrize("name", ["llama", "gemma3", "phi3", "qwen2vl"])
def test_meta_count_is_the_real_count(name):
    cfg = _configs()[name]
    info = with_exact_counts(internals(cfg), cfg)
    _, real = _real(cfg)
    assert info["params_source"].startswith("exact") and info["params_total"] == real
    assert info["params"] == real


def test_the_template_misses_gemma_gating_and_the_exact_count_does_not():
    cfg = _configs()["gemma3"]
    est = internals(cfg)["params"]
    assert est < with_exact_counts(internals(cfg), cfg)["params"]   # read as ungated


def test_a_vlm_counts_its_vision_tower():
    cfg = _configs()["qwen2vl"]
    info = with_exact_counts(internals(cfg), cfg)
    real, _ = _real(cfg)
    vision = sum(p.numel() for n, p in real.named_parameters() if ".visual." in f".{n}")
    assert info["vision_params"] == vision > 0


def test_a_config_that_cant_build_falls_back_to_a_labelled_estimate(monkeypatch):
    cfg = _configs()["llama"]
    import frame.model as fm
    monkeypatch.setattr(fm, "build_meta", lambda c: (_ for _ in ()).throw(RuntimeError("boom")))
    info = with_exact_counts(internals(cfg), cfg, lora={"r": 8, "targets": ["q_proj"]})
    assert info["params"] == internals(cfg)["params"] and "lora_trainable" not in info
    assert info["params_source"].startswith("estimated") and "boom" in info["params_source"]


def test_fused_phi3_targets_that_dont_exist_are_refused_not_counted():
    cfg = _configs()["phi3"]
    n, why = lora_trainable(build_meta(cfg), 8, ["q_proj", "k_proj", "v_proj"])
    assert n is None and "match no module" in why
    n, _ = lora_trainable(build_meta(cfg), 8, ["qkv_proj"])
    assert n == 2 * 8 * (64 + 3 * 64 - 2 * 32)                       # 2 layers, GQA-fused qkv


@pytest.mark.parametrize("name,targets", [
    ("llama", "all-linear"), ("llama", ["q_proj", "v_proj"]), ("gemma3", "all-linear"),
    ("phi3", "all-linear"), ("phi3", ["qkv_proj", "gate_up_proj"]), ("qwen2vl", "all-linear"),
    ("qwen2vl", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])])
def test_lora_count_matches_peft(name, targets):
    peft = pytest.importorskip("peft")
    cfg = _configs()[name]
    ours, _ = lora_trainable(build_meta(cfg), 8, targets)
    real, _ = _real(cfg)
    pm = peft.get_peft_model(real, peft.LoraConfig(r=8, target_modules=targets))
    assert ours == sum(p.numel() for p in pm.parameters() if p.requires_grad)


@pytest.mark.parametrize("name", ["llama", "phi3", "qwen2vl"])
def test_quantizable_params_are_the_linear_weights_but_the_head(name):
    cfg = _configs()[name]
    info = with_exact_counts(internals(cfg), cfg)
    real, _ = _real(cfg)
    head = real.get_output_embeddings()
    want = sum(m.weight.numel() for m in real.modules()
               if isinstance(m, torch.nn.Linear) and m is not head)
    assert info["quantizable_params"] == want > 0
