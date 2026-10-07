"""Analytical model-state memory (weights + grads + optimizer + fp32 master).

Pairs with the profiler's *measured* peak: measured_peak - model_state = activations,
the part you can't get from a config. It's a function of the actual knobs (precision,
optimizer), not a fixed assumption, so it tracks what the agent changes. GiB (1024^3)
to match the profiler's memory units. The standard ZeRO accounting: bf16 + AdamW mixed
precision = 16 bytes/param (2 weight + 2 grad + 12 optimizer state).

With frozen weights (LoRA / PEFT) only the trained ones carry grads, optimizer state and a master
copy; a frozen weight is just stored. peft keeps LoRA adapters in fp32 by default, so they are
4-byte weights with 4-byte grads and need no separate master copy.
"""

_GB = 1024 ** 3

# optimizer state slots per param (m, v, momentum, ...). fp32 unless 8-bit.
_OPT_STATES = {"adamw": 2, "adam": 2, "lion": 1, "sgd": 1, "sgd_nomomentum": 0}


def training_memory(params, param_bytes=2, optimizer="adamw", master_weights=None,
                    trainable=None, trainable_bytes=None):
    """params: parameter count, frozen and trained. param_bytes: 2 (bf16/fp16) or 4 (fp32).

    trainable: how many of them train (None: all, a full fine-tune). trainable_bytes: their
    precision (None: param_bytes; LoRA adapters under peft: 4).
    master_weights: keep an fp32 copy of the trained weights; defaults on when they are <fp32.
    """
    trainable = params if trainable is None else trainable
    trainable_bytes = param_bytes if trainable_bytes is None else trainable_bytes
    if master_weights is None:
        master_weights = trainable_bytes < 4
    state_bytes = 1 if "8bit" in optimizer else 4
    n_states = _OPT_STATES.get(optimizer.replace("_8bit", ""), 2)

    weights = (params - trainable) * param_bytes + trainable * trainable_bytes
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
