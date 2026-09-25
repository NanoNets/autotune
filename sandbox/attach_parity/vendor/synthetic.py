"""Worst-case synthetic batches — the input to the OOM ceiling.

Memory is decided by the LARGEST batch a run will ever see, not the average one. On a dataset
whose p50 is 16 tokens and p99 is 317, a batch drawn from real data almost never holds a max_len
sequence, so a memory reading taken from it is a random variable: a config can pass ten clean
steps at Frame and OOM hours into training, when a long batch finally arrives. Measuring the
ceiling on real data is hoping to hit the worst case; this manufactures it.

Activation memory is a function of SHAPE, not of content, so the worst case can be built from
nothing: every sequence exactly `seq_len`, attention mask all ones (no padding, no early exit,
full attention matrix). No dataset, no tokenizer, no dataloader — which is what makes bisecting
the ceiling cheap enough to do before the loop starts.

`spec(...)` is pure (shape + validation); `random_batch(...)` is the torch shell.

Ids are drawn uniformly from [0, vocab) and seeded. Content-independence is why uniform ids are
sound; the bound matters because an id >= vocab is a device-side assert in the embedding lookup —
a crash, not the OOM being measured.
"""


def spec(vocab, micro_batch, seq_len):
    """Validate + describe one worst-case batch. Pure."""
    for name, v in (("vocab", vocab), ("micro_batch", micro_batch), ("seq_len", seq_len)):
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise ValueError(f"{name} must be a positive int, got {v!r}")
    return {"vocab": vocab, "micro_batch": micro_batch, "seq_len": seq_len,
            "tokens": micro_batch * seq_len}


def random_batch(vocab, micro_batch, seq_len, device="cpu", seed=0):
    """One worst-case batch, built on `device`. Needs torch.

    Built ONCE and reused across the trial's steps: the ceiling is a property of the step, not of
    data loading, and re-drawing per step would add host-to-device traffic the question doesn't
    involve. Seeded, so a ceiling is reproducible.
    """
    import torch
    s = spec(vocab, micro_batch, seq_len)
    g = torch.Generator(device=device).manual_seed(seed)
    ids = torch.randint(0, vocab, (micro_batch, seq_len), device=device, generator=g)
    return {
        "input_ids": ids,
        "attention_mask": torch.ones_like(ids),   # all ones: zero padding IS the worst case
        "labels": ids.clone(),
        "spec": s,
    }
