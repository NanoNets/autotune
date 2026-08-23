"""Dataset shape for the Profile: how long are the sequences, how many tokens.

`distribution(lengths)` is pure (a list of per-sample token counts -> percentiles +
volume), so it's testable without `datasets`/`transformers`. `scan(...)` is the shell
that loads a dataset, tokenizes a sample of rows, and feeds their lengths in. The
length distribution drives batch/packing decisions; total tokens is the D in the roofline.
"""
import math


def _pct(sorted_vals, p):
    """Nearest-rank percentile on an already-sorted list (rank = ceil(p/100 * n))."""
    if not sorted_vals:
        return None
    rank = max(1, math.ceil(p / 100 * len(sorted_vals)))
    return sorted_vals[min(rank, len(sorted_vals)) - 1]


def distribution(lengths):
    """lengths: per-sample token counts (from the tokenized sample)."""
    xs = sorted(lengths)
    n = len(xs)
    if not n:
        return {"n_samples": 0, "total_tokens": 0, "mean_len": None,
                "p50": None, "p90": None, "p99": None, "max_len": None}
    total = sum(xs)
    return {
        "n_samples": n,
        "total_tokens": total,
        "mean_len": round(total / n, 1),
        "p50": _pct(xs, 50),
        "p90": _pct(xs, 90),
        "p99": _pct(xs, 99),
        "max_len": xs[-1],
    }


def scan(dataset, tokenizer, text_field="text", split="train", sample=1000, name=None):
    """Tokenize up to `sample` rows and summarize their lengths. Needs `datasets`.

    dataset: an HF hub id or local path. name: dataset config (e.g. 'wikitext-2-raw-v1').
    tokenizer: an HF tokenizer (from `model.load`'s model).
    """
    from datasets import load_dataset
    ds = load_dataset(dataset, name, split=split, streaming=True)
    lengths = []
    for i, row in enumerate(ds):
        if i >= sample:
            break
        lengths.append(len(tokenizer(row[text_field]).input_ids))
    return distribution(lengths)
