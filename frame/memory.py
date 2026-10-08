"""Analytical model-state memory (weights + grads + optimizer state).

Pairs with the profiler's *measured* peak: measured_peak - model_state = activations,
the part you can't get from a config. GiB (1024^3) to match the profiler's memory units.

Everything follows the dtype the trained weights are held in, as the training code holds them
(never a recipe Frame assumes): grads match the weights, and torch's AdamW keeps its state in the
weights' dtype. bf16 weights + AdamW = 2 + 2 + 2x2 = 8 bytes/param; fp32 weights (fp32 load, bf16
autocast compute) = 4 + 4 + 2x4 = 16. A separate fp32 master copy exists only when the training
code keeps one (multi-GPU frameworks' mixed precision, out of scope): `master_weights=True`.

With frozen weights (LoRA / PEFT) only the trained ones carry grads, optimizer state and a master
copy; a frozen weight is just stored. peft keeps LoRA adapters in fp32 by default, so they are
4-byte weights with 4-byte grads and need no separate master copy.
"""

_GB = 1024 ** 3

# optimizer state slots per param (m, v, momentum, ...). fp32 unless 8-bit.
_OPT_STATES = {"adamw": 2, "adam": 2, "lion": 1, "sgd": 1, "sgd_nomomentum": 0}


def training_memory(params, param_bytes=2, optimizer="adamw", master_weights=None,
                    trainable=None, trainable_bytes=None, quantized=0, quant_bytes=None):
    """params: parameter count, frozen and trained. param_bytes: 2 (bf16/fp16) or 4 (fp32).
    quantized: how many of the frozen weights are stored quantized (QLoRA), at quant_bytes each
    (bytes per weight, scaling constants included).

    trainable: how many of them train (None: all, a full fine-tune). trainable_bytes: their
    precision (None: param_bytes; LoRA adapters under peft: 4).
    master_weights: the training code keeps an extra fp32 copy of the trained weights (default no).
    """
    trainable = params if trainable is None else trainable
    trainable_bytes = param_bytes if trainable_bytes is None else trainable_bytes
    master_weights = bool(master_weights) and trainable_bytes < 4
    state_bytes = 1 if "8bit" in optimizer else trainable_bytes   # AdamW: the weights' dtype
    n_states = _OPT_STATES.get(optimizer.replace("_8bit", ""), 2)

    quantized = min(quantized or 0, params - trainable)
    weights = ((params - trainable - quantized) * param_bytes + trainable * trainable_bytes
               + quantized * (quant_bytes or 0))
    grads = trainable * trainable_bytes
    optim = trainable * n_states * state_bytes
    master = trainable * 4 if master_weights else 0

    gb = lambda b: round(b / _GB, 2)
    total = weights + grads + optim + master
    return {
        "weights_gb": gb(weights),
        "grads_gb": gb(grads),
        "optimizer_gb": gb(optim),
        "master_gb": gb(master),
        "model_state_gb": gb(total),                   # everything but activations
    }
