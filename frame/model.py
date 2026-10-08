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
        "tied": tied,
        "dtype": (str(_get(cfg, "dtype", "torch_dtype")).replace("torch.", "")   # the checkpoint's:
                  if _get(cfg, "dtype", "torch_dtype") is not None else None),   # 5.x renamed it
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


# ---- exact counts: the real module tree, built with no weights ----------------------------
# The analytical count above is a llama-shaped template: it misses gating it can't see in the
# activation name (Gemma), hybrid layers (Qwen3.5), fused projections for LoRA (Phi-3) and the vision
# tower (VLMs). Building the model on torch's `meta` device runs transformers' own model code on the
# config: every tensor has its real shape and no storage, so it costs ~1 s and no memory, and the
# counts are exact for any architecture transformers ships. The template stays as the fallback,
# always labelled an estimate (`params_source`).

def build_meta(config):
    """The model transformers would build for `config`, on the meta device: the class
    `AutoModelForCausalLM` maps it to (what our runner and most SFT scripts load), else the class
    the config's `architectures` names, else `AutoModelForImageTextToText` (VLMs). Raises when
    none builds."""
    import torch
    import transformers
    from transformers import AutoModelForCausalLM
    if not type(config).__module__.startswith("transformers."):
        raise ValueError(f"{type(config).__name__} comes from the checkpoint's own code, "
                         "which Frame doesn't run")
    with torch.device("meta"):
        try:
            return AutoModelForCausalLM.from_config(config)
        except ValueError:                             # no causal-LM mapping (e.g. Qwen2-VL)
            arch = (getattr(config, "architectures", None) or [None])[0]
            cls = getattr(transformers, arch or "", None)
            if cls is not None:
                return cls._from_config(config)
            return transformers.AutoModelForImageTextToText.from_config(config)


_VISION_PREFIXES = ("visual", "vision_tower", "vision_model", "vision_encoder")


def exact_counts(model):
    """(total, vision) parameters of a built model. `parameters()` yields a shared (tied) weight
    once, so tied embeddings count once, as they should."""
    total = vision = 0
    for name, p in model.named_parameters():
        total += p.numel()
        if any(part in _VISION_PREFIXES for part in name.split(".")):
            vision += p.numel()
    return total, vision


def quantizable_params(model):
    """Weights a 4-/8-bit load quantizes: every `Linear` / `Conv1D` weight but the output head's.
    Embeddings, norms, biases and the head stay at the compute dtype."""
    import torch
    from transformers.pytorch_utils import Conv1D
    out = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
    seen, n = set(), 0
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, Conv1D)) and m is not out and id(m.weight) not in seen:
            seen.add(id(m.weight))
            n += m.weight.numel()
    return n


def lora_trainable(model, r, targets):
    """(params, why_not): the parameters LoRA adds to a built model, matched the way peft matches
    them. "all-linear": every `nn.Linear` / `Conv1D` except the output head
    (peft `_maybe_include_all_linear_layers`); a list: every module whose name is a target or ends
    with `.<target>` (peft `check_target_module_exists`). Each adds `r * (d_in + d_out)`.
    params None, with why: no module matches (peft would refuse the config), or a match is a layer
    type this doesn't count."""
    import torch
    from transformers.pytorch_utils import Conv1D
    modules = dict(model.named_modules())
    if targets == "all-linear" or targets == ["all-linear"]:
        out = model.get_output_embeddings() if hasattr(model, "get_output_embeddings") else None
        chosen = [n for n, m in modules.items() if isinstance(m, (torch.nn.Linear, Conv1D)) and m is not out]
    else:
        ts = [targets] if isinstance(targets, str) else list(targets)
        chosen = [n for n in modules if n and any(n == t or n.endswith("." + t) for t in ts)]
    if not chosen:
        return None, (f"LoRA targets {targets} match no module of {type(model).__name__}: "
                      f"peft would refuse this config")
    total = 0
    for n in chosen:
        m = modules[n]
        if isinstance(m, torch.nn.Linear):
            total += r * (m.in_features + m.out_features)
        elif isinstance(m, Conv1D):                    # weight is (d_in, d_out)
            total += r * (m.weight.shape[0] + m.weight.shape[1])
        elif isinstance(m, torch.nn.Embedding):
            total += r * (m.num_embeddings + m.embedding_dim)
        else:
            return None, f"LoRA target {n} is a {type(m).__name__}, which this doesn't count"
    return total, None


def with_exact_counts(info, config, lora=None):
    """`internals(config)` plus exact counts from a meta-device build. On success: `params` /
    `params_total` exact (dense; an MoE keeps the analytical active count as `params`),
    `vision_params`, and with `lora`, `lora_trainable` (or None + `lora_note`). On failure: the
    analytical numbers, labelled estimates, with the reason in `params_source`."""
    info = dict(info)
    try:
        model = build_meta(config)
    except Exception as e:                             # custom code, or a model that can't build
        info["params_source"] = f"estimated from config dims ({type(e).__name__}: {str(e)[:160]})"
        return info
    total, vision = exact_counts(model)
    info["params_total"] = total
    if not info.get("moe"):
        info["params"] = total
    info["vision_params"] = vision
    info["quantizable_params"] = quantizable_params(model)
    info["params_source"] = f"exact: {type(model).__name__} built on the meta device"
    if lora:
        info["lora_trainable"], info["lora_note"] = lora_trainable(model, lora["r"], lora["targets"])
    return info


def load(model, lora=None):
    """model: an HF hub id or local path. Needs `transformers`. lora: {"r", "targets"}."""
    from transformers import AutoConfig
    try:                                               # transformers' own class first: a hub repo
        config = AutoConfig.from_pretrained(model)     # may ship a copy of a native one (Phi-3)
    except ValueError:                                 # an architecture transformers doesn't know
        config = AutoConfig.from_pretrained(model, trust_remote_code=True)
    return with_exact_counts(internals(config), config, lora)
