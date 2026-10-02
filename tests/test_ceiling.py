"""Ceiling tests: the divisor search (pure), the re-measure rules, the stamp, the live trial path
(CPU, fake OOM), and `python -m frame.ceiling` against a fake backend.

`search_divisors` takes the fit test as an argument, so the search is exercised with no GPU at all.
The live tests fake the OOM by raising the exception a real card raises, which walks the real
path: trial -> status="oom" -> search -> record.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame import ceiling as C  # noqa: E402
from frame.ceiling import (divisors, find_ceiling, is_stale, measure, search_divisors,  # noqa: E402
                           stamp)


def _counted(limit):
    """A monotone fit test that fits up to `limit`, plus the trials it was asked for."""
    calls = []

    def fits(n):
        calls.append(n)
        return n <= limit
    return fits, calls


# ---- the search ---------------------------------------------------------------------------
def test_a_fresh_search_answers_with_the_largest_fitting_divisor():
    for limit, want in ((1, 1), (2, 2), (3, 2), (7, 4), (11, 8), (16, 16), (100, 16)):
        fits, _ = _counted(limit)
        assert search_divisors(fits, 16) == want, limit


def test_a_fresh_search_climbs_the_divisors_and_stops_at_the_first_failure():
    fits, calls = _counted(11)
    search_divisors(fits, 16)
    assert calls == [1, 2, 4, 8, 16]                   # never 11, 12, 10: no config can use them


def test_a_fresh_search_never_tries_above_effective_batch():
    fits, calls = _counted(1000)
    assert search_divisors(fits, 32) == 32
    assert calls == [1, 2, 4, 8, 16, 32]


def test_nothing_fits_is_zero():
    """0 is the memory-regime signal: no batch knob rescues this, only recompute/offload/shard."""
    assert search_divisors(lambda n: False, 16) == 0


def test_a_remeasure_after_freeing_memory_climbs_from_the_previous_ceiling():
    fits, calls = _counted(8)                          # was 2; compile + checkpointing freed memory
    assert search_divisors(fits, 16, previous=2) == 8
    assert calls == [4, 8, 16]                         # 2 itself is implied by 4 fitting


def test_a_remeasure_that_gains_nothing_costs_two_trials():
    fits, calls = _counted(5)
    assert search_divisors(fits, 16, previous=4) == 4
    assert calls == [8, 4]


def test_a_remeasure_after_adding_memory_steps_down():
    fits, calls = _counted(1)                          # was 4; eager attention added memory
    assert search_divisors(fits, 16, previous=4) == 1
    assert calls == [8, 4, 2, 1]


def test_a_remeasure_from_the_top_checks_the_top_first():
    fits, calls = _counted(32)
    assert search_divisors(fits, 32, previous=32) == 32 and calls == [32]


def test_divisors():
    assert divisors(16) == [1, 2, 4, 8, 16] and divisors(12) == [1, 2, 3, 4, 6, 12]


# ---- the stamp and staleness --------------------------------------------------------------
HF_DEFAULTS = {"dtype": "bfloat16", "gradient_checkpointing": False, "attn_implementation": "sdpa",
               "torch_compile": False, "chunked_ce": 0, "micro_batch": 1, "padding": "max_length"}


def _rec(config, defaults=HF_DEFAULTS, seq_len=2048, eb=16):
    return {"seq_len": seq_len, "effective_batch": eb, "micro_batch_max": 2, "fits": True,
            "config": stamp(config, defaults)}


def test_the_stamp_includes_defaults_the_config_never_set():
    """Regression: Frame passed only {device, dtype, model, dataset}, so the stamp held only dtype
    and switching compile / checkpointing / attention read as 'not stale'."""
    s = stamp({"dtype": "bfloat16", "model": "m"}, HF_DEFAULTS)
    assert s == {"dtype": "bfloat16", "gradient_checkpointing": False,
                 "attn_implementation": "sdpa", "torch_compile": False, "chunked_ce": 0}


@pytest.mark.parametrize("change", [{"gradient_checkpointing": True}, {"torch_compile": True},
                                    {"attn_implementation": "eager"}, {"chunked_ce": 8},
                                    {"env": {"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"}},
                                    {"max_len": 4096}, {"effective_batch": 32}])
def test_a_memory_change_makes_it_stale(change):
    base = {"dtype": "bfloat16", "model": "m", "max_len": 2048, "effective_batch": 16}
    assert is_stale(_rec(base), {**base, **change}, HF_DEFAULTS) is True


@pytest.mark.parametrize("change", [{"micro_batch": 8}, {"grad_accum": 2}, {"dataloader_workers": 4},
                                    {"padding": "longest"}, {"length_grouped": True}])
def test_a_non_memory_change_does_not(change):
    base = {"dtype": "bfloat16", "model": "m", "max_len": 2048, "effective_batch": 16}
    assert is_stale(_rec(base), {**base, **change}, HF_DEFAULTS) is False


def test_a_dtype_only_stamp_from_before_is_stale():
    """Records written before the full stamp hold only dtype: re-measure rather than trust them."""
    old = {"seq_len": 2048, "micro_batch_max": 2, "fits": True, "config": {"dtype": "bfloat16"}}
    assert is_stale(old, {"dtype": "bfloat16"}, HF_DEFAULTS) is True
    assert is_stale(None, {}) is True


def test_a_script_stamps_its_declared_memory_knobs():
    cfg = {"script": "train.py", "dtype": "bfloat16",
           "memory_knobs": {"gradient_checkpointing": False, "torch_compile": False}}
    rec = {"seq_len": 512, "effective_batch": 32, "fits": True, "config": stamp(cfg)}
    assert rec["config"] == {"dtype": "bfloat16", "gradient_checkpointing": False, "torch_compile": False}
    moved = {**cfg, "memory_knobs": {"gradient_checkpointing": False, "torch_compile": True}}
    assert is_stale(rec, moved) is True
    assert is_stale(rec, {**cfg, "script_args": ["--per_device_train_batch_size", "16"]}) is False
    with pytest.raises(ValueError, match="memory_knobs"):
        stamp({"script": "train.py"})


# ---- the live trial path (CPU, fake OOM) --------------------------------------------------
class _FakeRunner:
    """OOMs above `limit`, exactly as a real card would, so the live path is walked on CPU."""
    def __init__(self, limit, fail_on="setup"):
        self.limit, self.fail_on = limit, fail_on
        self.setups, self.steps = [], 0

    def setup(self, config):
        self.setups.append(config)
        self.cfg = config
        if self.fail_on == "setup":
            self._maybe_oom()

    def _maybe_oom(self):
        import torch
        if self.cfg["micro_batch"] > self.limit:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (simulated)")

    def step(self):
        self.steps += 1
        if self.fail_on == "step2" and self.steps % 3 == 2:
            self._maybe_oom()                          # the peak arrives at step 2, as with AdamW


def _find(limit, eb=16, **kw):
    r = _FakeRunner(limit, **{k: kw.pop(k) for k in ("fail_on",) if k in kw})
    rec = find_ceiling(r, {"device": "cpu", "dtype": "bfloat16"}, seq_len=2048, vocab=32000,
                       effective_batch=eb, defaults=HF_DEFAULTS, **kw)
    return r, rec


def test_live_ceiling_records_the_divisor_and_the_oom():
    r, rec = _find(11)
    assert rec["micro_batch_max"] == 8 and rec["fits"] is True and rec["capped"] is False
    assert rec["oom_at"] == 16 and rec["seq_len"] == 2048 and rec["effective_batch"] == 16
    assert [t["micro_batch"] for t in rec["trials"]] == [1, 2, 4, 8, 16]
    assert rec["config"]["attn_implementation"] == "sdpa"         # defaults stamped


def test_every_trial_is_a_short_worst_case_run_at_accumulation_one():
    r, _ = _find(4)
    assert all(c["data"] == "synthetic" and c["max_len"] == 2048 for c in r.setups)
    assert all(c["effective_batch"] == c["micro_batch"] for c in r.setups)
    assert r.steps == 3 * 3                            # 3 trials that got past setup, 3 steps each


def test_an_oom_at_step_two_is_caught_by_a_three_step_trial():
    """A one-step trial would miss it: AdamW's state first exists at step 2."""
    _, rec = _find(2, fail_on="step2")
    assert rec["micro_batch_max"] == 2 and rec["oom_at"] == 4


def test_nothing_fits_is_reported_not_raised():
    _, rec = _find(0)
    assert rec["micro_batch_max"] == 0 and rec["fits"] is False and rec["oom_at"] == 1


def test_capped_says_the_real_ceiling_is_at_least_effective_batch():
    _, rec = _find(1000, eb=8)
    assert rec["micro_batch_max"] == 8 and rec["capped"] is True


def test_a_crash_is_not_read_as_a_memory_answer():
    class Broken(_FakeRunner):
        def setup(self, config):
            raise RuntimeError("boom")
    with pytest.raises(RuntimeError, match="not an OOM"):
        find_ceiling(Broken(1), {"device": "cpu"}, seq_len=8, vocab=10, effective_batch=4)


def test_a_remeasure_records_where_it_started():
    _, first = _find(2)
    _, again = _find(8, previous=first)
    assert again["micro_batch_max"] == 8 and again["remeasured_from"] == 2
    assert [t["micro_batch"] for t in again["trials"]] == [4, 8, 16]


# ---- through a backend: per-trial configs and the CLI -------------------------------------
def test_a_script_trial_config_fills_in_the_micro_batch(tmp_path):
    tmpl = tmp_path / "ceiling.yaml"
    tmpl.write_text("data_path: runs/r/ceiling_rows.jsonl\nper_device_train_batch_size: {micro_batch}\n")
    out = str(tmp_path / "probes" / "ceiling_mb{micro_batch}.yaml")
    cfg = C._render({"script": "train.py", "script_args": ["--bs", "{micro_batch}", "--config", out],
                     "ceiling_templates": {out: str(tmpl)}}, 4, 3)
    assert cfg["probe_kind"] == "ceiling_trial" and cfg["micro_batch"] == cfg["effective_batch"] == 4
    assert cfg["script_args"] == ["--bs", "4", "--config", out.replace("{micro_batch}", "4")]
    written = tmp_path / "probes" / "ceiling_mb4.yaml"
    assert written.read_text().endswith("per_device_train_batch_size: 4\n")
    assert cfg["script_files"] == [str(written)]


def _fake_backend(monkeypatch, limit):
    import backends
    calls = []

    def run(cfg):
        calls.append(cfg)
        assert cfg["probe_kind"] == "ceiling_trial"
        ok = cfg["micro_batch"] <= limit
        return {"status": "ok" if ok else "oom", "error": None if ok else "CUDA out of memory",
                "peak_gb": cfg["micro_batch"] * 1.0, "reserved_gb": cfg["micro_batch"] * 1.1,
                "total_gb": 22.0}
    monkeypatch.setattr(backends, "run", run)
    return calls


def _run_dir(tmp_path, ceiling=None):
    d = tmp_path / "run"
    d.mkdir()
    prof = {"max_len": 2048, "effective_batch": 16, "model": {"vocab": 32000},
            "knobs": {"micro_batch": {"type": "int", "min": 1, "max": 2}}, "ceiling": ceiling}
    (d / "profile.json").write_text(json.dumps(prof))
    return d


def test_the_cli_remeasures_a_stale_ceiling_from_the_previous_one(tmp_path, monkeypatch):
    base = {"backend": "local", "framework": "hf", "model": "m", "dtype": "bfloat16"}
    old = {"seq_len": 2048, "effective_batch": 16, "micro_batch_max": 2, "fits": True,
           "config": stamp(base, C.framework_defaults("hf"))}
    d = _run_dir(tmp_path, old)
    calls = _fake_backend(monkeypatch, limit=8)
    cfg = tmp_path / "cand.json"
    cfg.write_text(json.dumps({**base, "gradient_checkpointing": True}))
    rec = C.main(["--config", str(cfg), "--run", str(d)])
    assert rec["micro_batch_max"] == 8 and [c["micro_batch"] for c in calls] == [4, 8, 16]
    assert all(c["data"] == "synthetic" and c["vocab"] == 32000 for c in calls)
    prof = json.loads((d / "profile.json").read_text())
    assert prof["ceiling"]["micro_batch_max"] == 8 and prof["ceiling_history"] == [old]
    assert prof["knobs"]["micro_batch"]["max"] == 8
    rows = (d / "ceiling.tsv").read_text().splitlines()
    assert rows[0].startswith("seq_len") and len(rows) == 4 and "re-measure from 2" in rows[1]


def test_the_cli_runs_nothing_when_the_ceiling_still_holds(tmp_path, monkeypatch):
    base = {"backend": "local", "framework": "hf", "model": "m", "dtype": "bfloat16"}
    old = {"seq_len": 2048, "effective_batch": 16, "micro_batch_max": 2, "fits": True,
           "config": stamp(base, C.framework_defaults("hf"))}
    d = _run_dir(tmp_path, old)
    calls = _fake_backend(monkeypatch, limit=8)
    cfg = tmp_path / "cand.json"
    cfg.write_text(json.dumps({**base, "micro_batch": 2, "dataloader_workers": 4}))
    assert C.main(["--config", str(cfg), "--run", str(d)]) == old
    assert calls == [] and not (d / "ceiling.tsv").exists()


def test_measure_passes_trial_records_through():
    rec = measure(lambda n: {"status": "ok" if n <= 4 else "oom", "error": None,
                             "peak_gb": 1.0, "reserved_gb": 2.0, "total_gb": 4.0},
                  16, 512, {"dtype": "bfloat16"})
    assert rec["micro_batch_max"] == 4 and rec["card_frac"] == 0.5 and rec["frag_gb"] == 1.0
