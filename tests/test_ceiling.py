"""Ceiling tests — the search ladder (pure) and the live record (CPU, fake OOM).

`search` takes the fit test as an argument, so the ladder is exercised with no GPU at all. The
end-to-end test fakes the OOM by raising the exception `capture` catches, which walks the real
path: capture -> status="oom" -> bisect -> record.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame.ceiling import find_ceiling, is_stale, search


def _counted(limit):
    """A monotone fit test that fits up to `limit`, plus the trials it was asked for."""
    calls = []

    def fits(n):
        calls.append(n)
        return n <= limit
    return fits, calls


def test_search_finds_the_exact_boundary():
    for limit in (1, 2, 3, 7, 11, 64, 100):
        fits, _ = _counted(limit)
        assert search(fits) == limit, limit


def test_search_is_logarithmic_not_linear():
    """A probe costs real seconds — the ladder must not walk 1..n."""
    fits, calls = _counted(100)
    search(fits)
    assert len(calls) < 20                              # ~2*log2(100); a linear walk would be 101


def test_search_returns_zero_when_even_one_does_not_fit():
    """0 is the memory-regime signal: no batch knob rescues this, only recompute/offload/shard."""
    assert search(lambda n: False) == 0


def test_search_never_exceeds_the_cap():
    """Above effective_batch, grad_accum would be < 1 — those configs are not legal."""
    fits, calls = _counted(1000)
    assert search(fits, cap=16) == 16
    assert max(calls) <= 16                             # and it never even TRIED a bigger one


def test_search_bisects_below_a_cap_it_cannot_reach():
    fits, _ = _counted(11)
    assert search(fits, cap=16) == 11


class _FakeRunner:
    """OOMs above `limit`, exactly as a real card would, so the live path is walked on CPU."""
    def __init__(self, limit):
        self.limit = limit
        self.setups = []

    def setup(self, config):
        self.setups.append(config)
        import torch
        if config["micro_batch"] > self.limit:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")

    def step(self):
        pass

    def tokens_per_step(self):
        return 1


def _find(limit, **kw):
    r = _FakeRunner(limit)
    rec = find_ceiling(r, {"device": "cpu", "dtype": "bfloat16", "optimizer": "adamw"},
                       seq_len=2048, vocab=32000, warmup=1, clean_steps=1,
                       peak_flops=1e12, peak_mem_bw_gb_s=1.0, **kw)
    return r, rec


def test_live_ceiling_records_the_boundary_and_the_oom():
    r, rec = _find(11, cap=16)
    assert rec["micro_batch_max"] == 11 and rec["fits"] is True
    assert rec["oom_at"] == 12                          # the SMALLEST failure, not the last one
    assert rec["seq_len"] == 2048 and rec["method"] == "synthetic"
    assert [t["micro_batch"] for t in rec["trials"]] == [1, 2, 4, 8, 16, 12, 10, 11]


def test_every_trial_is_a_worst_case_batch():
    """A ceiling measured on anything but full-length batches is the bug this exists to fix."""
    r, _ = _find(11, cap=16)
    assert all(c["data"] == "synthetic" and c["max_len"] == 2048 for c in r.setups)


def test_nothing_fits_is_reported_not_raised():
    _, rec = _find(0)
    assert rec["micro_batch_max"] == 0 and rec["fits"] is False
    assert rec["oom_at"] == 1


def test_capped_says_the_real_ceiling_is_higher():
    _, rec = _find(1000, cap=8)
    assert rec["micro_batch_max"] == 8 and rec["capped"] is True


def test_a_crash_is_not_read_as_a_memory_answer():
    """status="error" means the trial told us nothing about memory — never bisect on it."""
    class Broken(_FakeRunner):
        def setup(self, config):
            raise RuntimeError("boom")

    try:
        find_ceiling(Broken(1), {"device": "cpu"}, seq_len=8, vocab=10,
                     warmup=1, clean_steps=1, peak_flops=1e12, peak_mem_bw_gb_s=1.0)
    except RuntimeError as e:
        assert "not an OOM" in str(e)
        return
    assert False, "expected the error to surface, not be scored as an OOM"


def test_stale_when_a_memory_knob_moved():
    """The ceiling is not a run constant — recompute/attn/optimizer all move it."""
    _, rec = _find(11, cap=16)
    assert rec["config"] == {"dtype": "bfloat16", "optimizer": "adamw"}
    assert is_stale(rec, {"dtype": "bfloat16", "optimizer": "adamw"}) is False
    assert is_stale(rec, {"dtype": "bfloat16", "optimizer": "adamw_8bit"}) is True
    assert is_stale(None, {}) is True


if __name__ == "__main__":
    for _n, _f in list(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f(); print(f"ok  {_n}")
    print("all passed")
