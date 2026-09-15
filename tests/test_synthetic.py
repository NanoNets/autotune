"""Synthetic-batch tests — the worst case must actually be the worst case.

`spec` is pure. `random_batch` needs torch but runs on CPU.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame.synthetic import random_batch, spec


def test_spec_counts_tokens():
    s = spec(vocab=32000, micro_batch=4, seq_len=2048)
    assert s["tokens"] == 4 * 2048


def test_spec_rejects_nonsense():
    for bad in (dict(vocab=0, micro_batch=1, seq_len=1),
                dict(vocab=10, micro_batch=-1, seq_len=1),
                dict(vocab=10, micro_batch=1, seq_len=0),
                dict(vocab=10.5, micro_batch=1, seq_len=1)):
        try:
            spec(**bad)
        except ValueError:
            continue
        assert False, f"expected ValueError for {bad}"


def test_every_sequence_is_exactly_seq_len():
    """The whole point: no short rows, so the batch is the worst case and not a sample of one."""
    b = random_batch(vocab=1000, micro_batch=3, seq_len=64)
    assert tuple(b["input_ids"].shape) == (3, 64)
    assert b["attention_mask"].sum().item() == 3 * 64      # all ones == zero padding


def test_ids_stay_inside_the_vocab():
    """An id >= vocab is a device-side assert in the embedding lookup — a crash, not an OOM."""
    b = random_batch(vocab=17, micro_batch=8, seq_len=128)
    assert int(b["input_ids"].max()) < 17 and int(b["input_ids"].min()) >= 0


def test_seeded_batches_are_reproducible():
    """A ceiling has to be re-derivable; an unseeded batch would make it a one-off reading."""
    a = random_batch(vocab=1000, micro_batch=2, seq_len=32, seed=7)
    b = random_batch(vocab=1000, micro_batch=2, seq_len=32, seed=7)
    c = random_batch(vocab=1000, micro_batch=2, seq_len=32, seed=8)
    assert a["input_ids"].equal(b["input_ids"])
    assert not a["input_ids"].equal(c["input_ids"])


if __name__ == "__main__":
    for _n, _f in list(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f(); print(f"ok  {_n}")
    print("all passed")
