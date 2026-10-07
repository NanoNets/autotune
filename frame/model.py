"""Model internals from an HF config.

`internals(config)` is a pure function on a config mapping (or any object with
`.to_dict()`), so it's testable without `transformers`. `load(model)` is the thin
shell that does `AutoConfig.from_pretrained` (needs transformers, hits the hub/path).

Reports the physical facts the roofline and memory model hang off: the dims, an
analytical parameter estimate (GQA / gated-MLP / MoE-aware; active-per-token for
MoE, which is the N the 6ND roofline wants), and the mechanically-derived facts
(attention regime, dtype, modality). No labels beyond what falls out of the config.
"""

# hidden_act values that mean a gated MLP (3 matrices: gate/up/down) rather than 2.
# Covers llama/qwen/mistral/mixtral (silu) and gemma (gelu_pytorch_tanh); plain
# gelu/gelu_new (GPT-style, ungated) deliberately excluded.
_GATED_ACTS = {"silu", "swish", "swiglu", "geglu", "gelu_pytorch_tanh"}


def _flatten(cfg):
    """Multimodal configs nest the language model's dims under `text_config` (Qwen3.5, Gemma3,
    Llama4) while `architectures` / `vision_config` / `tie_word_embeddings` stay at the top level.
    Read the text config as the base and overlay the top level's set fields, so a nested config
    presents the same flat surface a text-only one does. Top-level `None`s (transformers writes
    `dtype: null` on the wrapper) must not shadow the text config's real values.
    """
    text = cfg.get("text_config")
    if not isinstance(text, dict):
        return cfg
    merged = dict(text)
    merged.update({k: v for k, v in cfg.items() if v is not None})
    return merged


def _get(cfg, *names, default=None):
    """First present field among aliases (HF field names vary across archs)."""
    for n in names:
        if n in cfg and cfg[n] is not None:
            return cfg[n]
    return default


def _params(layers, hidden, heads, kv_heads, head_dim, vocab, inter, gated,
            experts_per_tok, experts, tied):
    """Analytical parameter count. Returns (active_per_token, total)."""
    q = hidden * heads * head_dim
    kv = 2 * hidden * kv_heads * head_dim
    o = heads * head_dim * hidden
    attn = q + kv + o

    mlp_mult = 3 if gated else 2
    one_expert = mlp_mult * hidden * inter
    if experts:                                        # MoE: top-k experts run per token
        mlp_active = experts_per_tok * one_expert
        mlp_total = experts * one_expert
    else:
        mlp_active = mlp_total = one_expert

    embed = vocab * hidden * (1 if tied else 2)         # input embedding (+ lm_head if untied)
    active = layers * (attn + mlp_active) + embed
    total = layers * (attn + mlp_total) + embed
    return int(active), int(total)


def internals(config):
    """config: a mapping of HF config fields (or anything with `.to_dict()`)."""
    cfg = config.to_dict() if hasattr(config, "to_dict") else dict(config)
    cfg = _flatten(cfg)

    layers = _get(cfg, "num_hidden_layers", "n_layer")
    hidden = _get(cfg, "hidden_size", "n_embd")
    heads = _get(cfg, "num_attention_heads", "n_head")
    kv_heads = _get(cfg, "num_key_value_heads", default=heads)   # GQA; absent -> MHA
    head_dim = _get(cfg, "head_dim", default=hidden // heads)
    vocab = _get(cfg, "vocab_size")
    inter = _get(cfg, "intermediate_size", "moe_intermediate_size", "n_inner")
    experts = _get(cfg, "num_local_experts", "num_experts")
    experts_per_tok = _get(cfg, "num_experts_per_tok")
    gated = _get(cfg, "hidden_act", "activation_function", default="") in _GATED_ACTS
    tied = _get(cfg, "tie_word_embeddings", default=True)   # HF default; large models set False

    active, total = _params(layers, hidden, heads, kv_heads, head_dim, vocab, inter,
                            gated, experts_per_tok, experts, tied)

    if kv_heads == heads:
        attention = "mha"
    elif kv_heads == 1:
        attention = "mqa"
    else:
        attention = "gqa"

    return {
        "arch": (_get(cfg, "architectures", default=[None])[0]
                 or _get(cfg, "model_type")),
        "params": active,                              # N for the 6ND roofline
        "params_total": total,                         # differs only for MoE
        "layers": layers,
        "hidden": hidden,
        "heads": heads,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "vocab": vocab,
        "intermediate": inter,
        "context_len": _get(cfg, "max_position_embeddings", "n_positions"),
        "attention": attention,
        "gated_mlp": gated,
        "dtype": _get(cfg, "torch_dtype", "dtype"),    # transformers 5.x renamed torch_dtype -> dtype
        "modality": "multimodal" if _get(cfg, "vision_config") else "text",
        "moe": bool(experts),
        "experts": experts,
        "experts_per_tok": experts_per_tok,
    }


# LoRA targets by module name (llama-family naming, which most decoder LMs use).
_ATTN_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_TARGETS = ("gate_proj", "up_proj", "down_proj")


def lora_params(internals, r, targets):
    """Trainable parameters LoRA adds: `r * (d_in + d_out)` per targeted projection per layer
    (A is d_in x r, B is r x d_out), from the dims `internals` already read off the config.

    targets: module names (`q_proj`, ..., `down_proj`) or peft's "all-linear" (every projection in
    the blocks, not `lm_head`). Returns None, never a guess, for a name this doesn't know (other
    families name their layers differently), an ungated MLP, or MLP targets on an MoE (where the
    expert weights are laid out is model- and version-specific).
    """
    targets = [targets] if isinstance(targets, str) else list(targets)
    if targets == ["all-linear"]:
        targets = list(_ATTN_TARGETS + _MLP_TARGETS)
    if any(t not in _ATTN_TARGETS + _MLP_TARGETS for t in targets):
        return None
    if any(t in _MLP_TARGETS for t in targets) and (not internals["gated_mlp"] or internals["moe"]):
        return None
    h, inter = internals["hidden"], internals["intermediate"]
    q_out = internals["heads"] * internals["head_dim"]
    kv_out = internals["kv_heads"] * internals["head_dim"]
    shapes = {"q_proj": (h, q_out), "k_proj": (h, kv_out), "v_proj": (h, kv_out),
              "o_proj": (q_out, h), "gate_proj": (h, inter), "up_proj": (h, inter),
              "down_proj": (inter, h)}
    per_layer = sum(r * (shapes[t][0] + shapes[t][1]) for t in set(targets))
    return internals["layers"] * per_layer


def load(model):
    """model: an HF hub id or local path. Needs `transformers`."""
    from transformers import AutoConfig
    return internals(AutoConfig.from_pretrained(model, trust_remote_code=True))
