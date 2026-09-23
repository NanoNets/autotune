"""Dataset tests — the length-distribution summary and the probe-sample draw (no GPU, no network)."""
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame.dataset import (distribution, reservoir, stratify, whole_steps, draw_sample,
                           write_sample, read_sample)


def test_distribution_basics():
    d = distribution([10, 20, 30, 40, 100])
    assert d["n_samples"] == 5
    assert d["total_tokens"] == 200
    assert d["mean_len"] == 40.0
    assert d["max_len"] == 100
    assert d["p50"] == 30                               # nearest-rank median


def test_percentiles_track_the_tail():
    d = distribution(list(range(1, 101)))               # 1..100
    assert d["p50"] == 50 and d["p90"] == 90 and d["p99"] == 99 and d["max_len"] == 100


def test_empty_is_safe():
    d = distribution([])
    assert d["n_samples"] == 0 and d["total_tokens"] == 0 and d["p50"] is None


# --- the probe sample --------------------------------------------------------------------------

def _stream(n):
    """Rows like an ordered file: every 3rd row blank, length growing along the file."""
    return [{"text": "" if i % 3 == 0 else "w " * (i // 10 + 1)} for i in range(n)]


def test_reservoir_skips_blanks_and_reaches_past_the_head():
    pool, seen = reservoir(_stream(3000), 100, seed=0)
    assert seen == 2000                                  # 3000 rows, every 3rd blank
    assert len(pool) == 100 and all(t.strip() for _, t in pool)
    assert [i for i, _ in pool] == sorted(i for i, _ in pool)   # returned in stream order
    assert max(i for i, _ in pool) > 1500                # a head slice would stop at ~150


def test_reservoir_is_deterministic_per_seed_and_differs_across_seeds():
    a, _ = reservoir(_stream(3000), 100, seed=0)
    assert reservoir(_stream(3000), 100, seed=0)[0] == a
    assert reservoir(_stream(3000), 100, seed=1)[0] != a


def test_reservoir_max_scan_caps_the_rows_read():
    pool, seen = reservoir(_stream(3000), 100, seed=0, max_scan=600)
    assert seen == 400 and max(i for i, _ in pool) < 600


def test_reservoir_keeps_everything_when_the_stream_is_smaller_than_the_pool():
    pool, seen = reservoir(_stream(30), 100)
    assert len(pool) == seen == 20


def test_stratify_reproduces_the_length_distribution_including_the_tail():
    lengths = [1] * 900 + [50] * 90 + [1000] * 10        # 1% of the pool is very long
    picks = stratify(lengths, 100)
    got = sorted(lengths[i] for i in picks)
    assert got.count(1) == 90 and got.count(50) == 9 and got.count(1000) == 1   # 90/9/1, exactly
    assert picks == sorted(picks) and len(set(picks)) == 100      # input order, no duplicates


def test_stratify_takes_everything_when_n_covers_the_pool():
    assert stratify([5, 1, 3], 10) == [0, 1, 2]


def test_whole_steps_rounds_down_to_the_effective_batch():
    assert whole_steps(1000, 16) == 992 and whole_steps(992, 16) == 992 and whole_steps(7, None) == 7


class _Tok:
    """Stand-in tokenizer: one token per whitespace word, truncated like an HF tokenizer."""
    def __call__(self, texts, truncation=False, max_length=None):
        ids = [t.split() for t in texts]
        return {"input_ids": [i[:max_length] if truncation else i for i in ids]}


def test_draw_sample_is_stratified_rounded_and_recorded():
    fake = types.ModuleType("datasets")
    fake.load_dataset = lambda *a, **k: iter(_stream(30000))
    real = sys.modules.get("datasets")
    sys.modules["datasets"] = fake
    try:
        drawn = draw_sample("wiki", _Tok(), 1000, effective_batch=16, max_len=20, seed=0)
    finally:
        if real is not None:
            sys.modules["datasets"] = real
        else:
            del sys.modules["datasets"]
    meta, rows = drawn["meta"], drawn["rows"]
    assert meta["method"] == "length-stratified" and meta["n"] == 992 == len(rows)
    assert meta["pool_size"] == 20 * 992 and meta["usable_rows_scanned"] == 20000   # pool = 20x n
    assert [r["row"] for r in rows] == sorted(r["row"] for r in rows)   # stream order
    assert max(r["tokens"] for r in rows) <= 20                  # measured after truncation
    for q in ("p50", "p90", "p99"):                              # the sample carries the pool's shape
        assert abs(meta["sample"][q] - meta["pool"][q]) <= 1


def test_sample_file_round_trips():
    drawn = {"meta": {"method": "length-stratified", "n": 2},
             "rows": [{"row": 3, "text": "a b", "tokens": 2}, {"row": 9, "text": "c", "tokens": 1}]}
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "run", "sample.jsonl")
        write_sample(path, drawn)
        meta, rows = read_sample(path)
    assert meta == drawn["meta"] and rows == drawn["rows"]


if __name__ == "__main__":
    for _n, _f in list(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f(); print(f"ok  {_n}")
    print("all passed")
