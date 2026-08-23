"""Dataset tests — the pure length-distribution summary (no datasets/tokenizer)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame.dataset import distribution


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


if __name__ == "__main__":
    for _n, _f in list(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f(); print(f"ok  {_n}")
    print("all passed")
