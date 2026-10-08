"""The attention term of the FLOP count: which layers attend over what, read off a model's config.

MFU follows PaLM's definition of model FLOPs (Chowdhery et al., 2022, "PaLM: Scaling Language
Modeling with Pathways", arXiv:2204.02311, Appendix B): per token, 6N for the weights plus
12 * L * H * Q * T for self-attention, where L is the attention layers, H the query heads, Q the
head dimension and T the attended sequence length. Per layer the forward does two matmuls of
2 * H * Q FLOPs per (query, key) pair (Q K^T, then the weights times V), and the backward twice
that: 12 = 4 forward x 3. Summed over a step, T * tokens becomes the number of (query, key) PAIRS
the step attends over, so the term is 12 * L * H * Q * pairs. As in PaLM, the causal mask isn't
discounted.

`spec(config)` lists the attention parts of a model (no torch needed); the capture paths count
each part's pairs per step:

- `lm_full`: full-attention layers of the language model. Pairs per sequence = length^2, a
  packed row's documents separately (a sequence attends within its own document).
- `lm_window`: sliding-window layers (Gemma, Mistral): length * min(length, window).
- `vision`: a VLM's vision tower, which attends within each image: patches^2 per image.

A part whose cost isn't attention over a sequence (linear attention, state-space layers) or whose
layout isn't known here (a vision tower's windowed blocks) is left out and named in `skipped`, so
the term can only read LOW, never high.
"""

_FULL = {"full_attention", "attention", "global", "global_attention"}
_WINDOW = {"sliding_attention", "sliding_window", "local", "local_attention"}


def _get(cfg, *names):
    for n in names:
        v = cfg.get(n) if isinstance(cfg, dict) else getattr(cfg, n, None)
        if v is not None:
            return v
    return None


def _lm(cfg, backward):
    text = _get(cfg, "text_config") or cfg
    layers = _get(text, "num_hidden_layers", "n_layer")
    heads = _get(text, "num_attention_heads", "n_head")
    hidden = _get(text, "hidden_size", "n_embd")
    if not (layers and heads and hidden):
        return [], ["language model: no layer / head / width in the config"]
    hq = heads * (_get(text, "head_dim") or hidden // heads)
    window = _get(text, "sliding_window")
    types = _get(text, "layer_types")
    parts, skipped = [], []
    if types:
        full = sum(t in _FULL for t in types)
        win = sum(t in _WINDOW for t in types)
        other = sorted({t for t in types if t not in _FULL | _WINDOW})
        if other:
            skipped.append(f"language model: {len(types) - full - win} layers of {', '.join(other)} "
                           f"(not attention over the sequence)")
    elif window and _get(text, "use_sliding_window") is not False:
        full, win = 0, layers
    else:
        full, win = layers, 0
    if full:
        parts.append({"part": "lm_full", "layers": full, "hq": hq, "window": None, "backward": backward})
    if win and window:
        parts.append({"part": "lm_window", "layers": win, "hq": hq, "window": int(window),
                      "backward": backward})
    elif win:
        skipped.append(f"language model: {win} sliding-window layers with no window size")
    return parts, skipped


def _vision(cfg, backward):
    v = _get(cfg, "vision_config")
    if not v:
        return [], []
    layers = _get(v, "depth", "num_hidden_layers")
    width = _get(v, "embed_dim", "hidden_size")    # Qwen2-VL: embed_dim is the attention width
    if not (layers and width):
        return [], ["vision tower: no layer count / width in the config"]
    full = _get(v, "fullatt_block_indexes")
    if full is not None:                           # windowed blocks (Qwen2.5-VL): only the full ones
        skipped = [f"vision tower: {layers - len(full)} windowed blocks"] if layers > len(full) else []
        layers = len(full)
    else:
        skipped = []
    parts = [{"part": "vision", "layers": layers, "hq": width, "window": None, "backward": backward}]
    return (parts if layers else []), skipped


def spec(config, backward=True, vision_backward=None):
    """(parts, skipped) for a model config (an HF config object or its dict). `backward`: gradients
    flow back through the language model (anything trains). `vision_backward`: through the vision
    tower (something in it trains); None = same as `backward`."""
    if config is None:
        return [], ["no model config"]
    parts, skipped = _lm(config, backward)
    vp, vs = _vision(config, backward if vision_backward is None else vision_backward)
    return parts + vp, skipped + vs


def describe(parts, skipped):
    """One line for the Observation: what the attention term covers."""
    if not parts:
        return "unknown: " + ("; ".join(skipped) or "no attention layout")
    covered = ", ".join(f"{p['part']} {p['layers']} layers x {p['hq']}" +
                        (f" (window {p['window']})" if p.get("window") else "") +
                        ("" if p["backward"] else " forward only") for p in parts)
    return f"PaLM 12*L*H*Q*pairs: {covered}" + (f"; left out: {'; '.join(skipped)}" if skipped else "")
