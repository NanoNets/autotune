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
    """Tokenize the FIRST `sample` rows and summarize their lengths. Needs `datasets`.

    A head slice, blank rows included: kept for callers that want a quick look, but `frame()` no
    longer uses it. The head of an ordered file is not the file (wikitext opens with short header
    lines), so the Profile's distribution now comes from `draw_sample`'s random pool instead.

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


# --- the probe sample: drawn once at Frame, loaded by every probe -------------------------------
#
# Under dynamic padding a probe's throughput is set by the lengths of the rows it draws, so the
# rows ARE part of the measurement (loop/contracts.md, the Measurement axiom). The first N rows of
# a file are not a sample of it. `draw_sample` takes a uniform random POOL over the whole split,
# measures its token lengths, and picks N rows at evenly spaced length quantiles of that pool, so
# the sample carries the corpus's length distribution, tail included, at any N. Frame draws it
# once, writes it next to the run, and every probe loads exactly those rows, which also makes the
# length statistics the agent plans from and the rows it times one population.

SAMPLE_METHOD = "length-stratified"
POOL_FACTOR = 20        # pool = 20x the sample: enough rows that its p99 tail is well populated


def usable(text):
    """A row worth training on: non-blank. A blank row tokenizes to all padding, whose loss is
    0/0 = NaN (`frameworks/hf.py` `_usable_texts` skips them for the same reason)."""
    return bool(text and text.strip())


def reservoir(rows, k, text_field="text", seed=0, max_scan=None):
    """A uniform random sample of `k` usable rows from a stream, in one pass (Algorithm R).

    Returns (pool, usable_seen): `pool` is [(stream_index, text), ...] sorted by stream index;
    `usable_seen` counts the usable rows scanned. Blank rows are skipped before sampling, so the
    pool describes the rows a trainer would actually use. `max_scan` caps the rows read, for a
    corpus too large to stream end to end; the pool is then uniform over that prefix only, which
    the caller records rather than hides.
    """
    import random
    rng = random.Random(seed)
    pool, seen = [], 0
    for i, row in enumerate(rows):
        if max_scan is not None and i >= max_scan:
            break
        text = row.get(text_field, "")
        if not usable(text):
            continue
        if len(pool) < k:
            pool.append((i, text))
        else:
            j = rng.randrange(seen + 1)                # keep each row with probability k/(seen+1)
            if j < k:
                pool[j] = (i, text)
        seen += 1
    return sorted(pool), seen


def stratify(lengths, n):
    """Indices of `n` items at evenly spaced quantiles of `lengths`, returned in input order.

    Systematic sampling over the length-sorted pool: item j is taken at quantile (j + 0.5) / n.
    The result reproduces the pool's length distribution by construction (every 1/n slice of it
    contributes one row), so the long tail is represented at any n instead of by luck. Ties break
    on index, so the draw is deterministic.
    """
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    if n >= len(order):
        return list(range(len(lengths)))
    return sorted(order[int((j + 0.5) * len(order) / n)] for j in range(n))


def whole_steps(n, effective_batch):
    """`n` rounded DOWN to a multiple of `effective_batch`, so one pass is whole optimizer steps.

    Rounded here, at the draw, rather than by trimming later: a trim after stratifying would cut
    whichever rows sit last, and a later sort could make those the longest.
    """
    return n - n % effective_batch if effective_batch else n


def draw_sample(dataset, tokenizer, n, text_field="text", split="train", name=None, seed=0,
                effective_batch=None, max_len=None, pool_size=None, max_scan=None):
    """Draw the probe sample: `n` usable rows, length-stratified from a random pool. Needs `datasets`.

    Lengths are measured AFTER truncation at `max_len`, because that is the length the loader
    batches. `n` is rounded down to a multiple of `effective_batch`. Returns
    {"meta": {...}, "rows": [{"row", "text", "tokens"}, ...]}, rows in stream order. `meta` holds
    the pool's distribution (the Profile's dataset shape) next to the sample's, so the two can be
    compared.
    """
    from datasets import load_dataset
    n = whole_steps(n, effective_batch)
    k = pool_size or POOL_FACTOR * n
    ds = load_dataset(dataset, name, split=split, streaming=True)
    pool, seen = reservoir(ds, k, text_field=text_field, seed=seed, max_scan=max_scan)
    enc = tokenizer([t for _, t in pool], truncation=max_len is not None, max_length=max_len)
    lengths = [len(ids) for ids in enc["input_ids"]]
    picks = stratify(lengths, n)
    return {
        "meta": {
            "method": SAMPLE_METHOD, "dataset": dataset, "name": name, "split": split,
            "text_field": text_field, "seed": seed, "max_len": max_len,
            "effective_batch": effective_batch, "n": len(picks), "pool_size": len(pool),
            "usable_rows_scanned": seen, "max_scan": max_scan,
            "pool": distribution(lengths),
            "sample": distribution([lengths[p] for p in picks]),
        },
        "rows": [{"row": pool[p][0], "text": pool[p][1], "tokens": lengths[p]} for p in picks],
    }


def write_sample(path, drawn):
    """Write a drawn sample as JSON lines: one `{"meta": ...}` header, then one row per line."""
    import json
    import os
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write(json.dumps({"meta": drawn["meta"]}) + "\n")
        for r in drawn["rows"]:
            f.write(json.dumps(r) + "\n")


def read_sample(path):
    """Read a sample written by `write_sample`. Returns (meta, rows)."""
    import json
    with open(path) as f:
        lines = [json.loads(line) for line in f if line.strip()]
    if not lines or "meta" not in lines[0]:
        raise ValueError(f"{path}: not a probe sample (no meta header)")
    return lines[0]["meta"], lines[1:]
