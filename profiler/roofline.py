"""Roofline: the theoretical floor on step time and MFU, against a MEASURED peak.

Training FLOPs follow PaLM's model-FLOPs definition (Chowdhery et al., 2022, "PaLM: Scaling
Language Modeling with Pathways", arXiv:2204.02311, Appendix B): per token 6N for the weights
(forward 2N + backward 4N, N = model params) plus 12 * L * H * Q * T for self-attention. The weight
term is `training_flops` / `vlm_training_flops`; the attention term is `attention_flops`, over the
(query, key) pairs a step attends over (`profiler/attention.py`). Leaving attention out understates
the work by a share that grows with sequence length: 6ND alone was 0.66-0.92x of the true FLOPs at
2,048 tokens (final_boss_report T3-C1), and ~22% of a Qwen2-VL step at 672x672.

With frozen weights (LoRA / PEFT) the backward's 4ND splits: gradients back through the
activations (2ND) still flow through every weight, but weight gradients (2ND) exist only for the
trained ones. So it is (4 * N_total + 2 * N_trainable) * D, which is exactly 6ND when every
weight trains. These are model FLOPs, not executed ones: recompute (gradient checkpointing) is not
counted, so MFU can't rise while training slows.

The denominator is measured on the actual device (see `capture.measure_peak_flops`), not read
from a datasheet table: a table can't tell an A10 from an A10G, quietly quotes sparse numbers,
and hard-fails on any chip it has never heard of. A measured peak also prices in clock
throttling and whatever the cloud is really giving us. This module stays torch-free — it just
takes the number.
"""


def training_flops(params: int, tokens: int, trainable: int = None) -> float:
    """Approx training FLOPs to process `tokens` tokens through a `params`-param model, of which
    `trainable` get weight gradients (None: all of them, i.e. 6ND)."""
    trainable = params if trainable is None else trainable
    return (4.0 * params + 2.0 * trainable) * tokens


def vlm_training_flops(params: int, tokens: float, trainable: int, vision: int,
                       vision_trainable: int, patches: float) -> float:
    """A vision-language model: the language model runs once per sequence token (image tokens
    included), the vision tower once per image PATCH (before patches merge into tokens: 2,304 for a
    672x672 image on Qwen2-VL, which become 576 tokens). Each part gets its own D:

        (4 * N_lm + 2 * N_lm_trainable) * tokens + (k * N_vision + 2 * N_vision_trainable) * patches

    k = 4 when anything in the vision tower trains (gradients flow back through it), else 2: a frozen
    tower below the trained weights runs forward only. Same convention as `training_flops`: weight
    FLOPs, no attention term."""
    lm = training_flops(params - vision, tokens, trainable - vision_trainable)
    k = 4.0 if vision_trainable else 2.0
    return lm + (k * vision + 2.0 * vision_trainable) * patches


def attention_flops(parts, pairs) -> float:
    """PaLM's attention term for one step: for each attention part (`profiler.attention.spec`),
    12 * layers * (heads x head_dim) * pairs, or 4 * ... when no gradient flows back through it (a
    frozen vision tower below the trained weights). `pairs`: {part: (query, key) pairs this step}."""
    return sum((12.0 if p["backward"] else 4.0) * p["layers"] * p["hq"] * pairs.get(p["part"], 0)
               for p in parts or [])


def step_time_floor_s(flops: float, peak_flops: float) -> float:
    """Compute-bound lower bound on step time (seconds) at the device's measured peak."""
    return flops / peak_flops if peak_flops else None


def mfu(flops: float, step_time_s: float, peak_flops: float) -> float:
    """Model FLOPs Utilization = achieved FLOP/s / measured peak FLOP/s. In [0, 1]."""
    if not peak_flops or not step_time_s:
        return None
    return (flops / step_time_s) / peak_flops
