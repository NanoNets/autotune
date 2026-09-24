"""HF-adapter tests — the pure parts (knobs surface + config resolution). No torch/transformers.

setup()/step()/_build_loader need a real model, data, and a GPU, so they're validated live, not
here. `_resolve` and `knobs` are import-clean (heavy imports are lazy), so they test on bare python3.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frameworks.hf import (knobs, _resolve, _usable_texts, _whole_steps, _reference_indices,
                           _rows_hash)

BASE = {"model": "Qwen/Qwen2.5-0.5B", "dataset": "Salesforce/wikitext"}


def test_knobs_are_safe_levers_only():
    k = knobs()
    assert set(k) == {"micro_batch", "gradient_checkpointing",
                      "attn_implementation", "torch_compile", "tf32", "fused_optimizer",
                      "padding", "length_grouped", "chunked_ce"}
    # dtype (precision) and seq_len (max_length) are off-limits run params, NOT searchable knobs;
    # grad_accum is derived, not free.
    assert "dtype" not in k and "seq_len" not in k and "grad_accum" not in k
    assert k["attn_implementation"]["options"] == ["eager", "sdpa", "flash_attention_2"]
    # compile is off/default/max-autotune in loop/throughput.md, so the knob is a choice, not a bool
    assert k["torch_compile"]["options"] == [False, "default", "reduce-overhead",
                                            "max-autotune", "max-autotune-no-cudagraphs"]
    # chunked_ce slices the cross-entropy over the token axis and checkpoints each slice. The loss
    # is a sum of the same per-token terms over the same ignore_index mask, divided by the same
    # count -- only the fp32 summation order differs. loop/throughput.md lists "chunked
    # cross-entropy" under Kernels, so it is a memory/kernel lever, not a model change.
    assert k["chunked_ce"]["min"] == 0          # 0 = off, so the default surface is unchanged
    # fused_optimizer swaps AdamW's foreach kernel chain for one fused kernel -- the SAME AdamW
    # algorithm, so it is a kernel swap (loop/throughput.md lists "fused optimizer" under Kernels),
    # not the optimizer-algorithm change that would be off-limits.
    assert k["fused_optimizer"] == {"type": "bool"}
    assert _resolve(BASE)["fused_optimizer"] is False         # default = the shipped behaviour


def test_padding_is_a_speed_lever_not_a_model_change():
    # pad-to-longest is masked out of attention and set to -100 in labels, so it changes only how
    # much compute is spent, never the gradients. seq_len still truncates, so max_len is respected.
    assert knobs()["padding"]["options"] == ["max_length", "longest"]
    assert _resolve(BASE)["padding"] == "max_length"          # default = the shipped behaviour
    assert _resolve({**BASE, "padding": "longest"})["pad_multiple"] == 64


def test_length_grouping_reduces_padding_waste():
    import torch
    from frameworks.hf import _length_grouped
    rows = [torch.arange(n) for n in [5, 400, 12, 380, 7, 410]]
    lens = [r.numel() for r in _length_grouped(rows, micro_batch=2)]
    # every pair should now be near-equal in length (one chunk boundary aside)
    spreads = sorted(max(lens[i:i + 2]) - min(lens[i:i + 2]) for i in range(0, 6, 2))
    assert spreads[0] <= 10 and spreads[1] <= 10           # two of three pairs are tight
    assert sorted(lens) == [5, 7, 12, 380, 400, 410]       # no row lost or duplicated


def test_torch_compile_accepts_bool_or_mode():
    assert _resolve(BASE)["torch_compile"] is False                     # default: off
    assert _resolve({**BASE, "torch_compile": True})["torch_compile"] is True
    assert _resolve({**BASE, "torch_compile": "max-autotune"})["torch_compile"] == "max-autotune"


def test_resolve_fills_defaults():
    c = _resolve(BASE)
    assert c["device"] == "cuda" and c["dtype"] == "bfloat16" and c["seq_len"] == 1024
    assert c["micro_batch"] == 1 and c["effective_batch"] == 16
    assert c["grad_accum"] == 16                          # effective_batch // micro_batch


def test_grad_accum_is_derived_to_hold_effective_batch():
    c = _resolve({**BASE, "effective_batch": 32, "micro_batch": 8})
    assert c["grad_accum"] == 4                           # 32 // 8
    assert c["tokens_per_step"] == 32 * c["seq_len"]      # effective_batch * seq_len


def test_indivisible_effective_batch_raises():
    try:
        _resolve({**BASE, "effective_batch": 10, "micro_batch": 4})   # 10 % 4 != 0
    except ValueError as e:
        assert "divisible" in str(e)
        return
    assert False, "expected ValueError when micro_batch doesn't divide effective_batch"


def test_dtype_and_seq_len_are_config_not_knobs():
    c = _resolve({**BASE, "dtype": "float32", "seq_len": 2048})   # still accepted as run params
    assert c["dtype"] == "float32" and c["seq_len"] == 2048


def test_missing_model_or_dataset_raises():
    for bad in ({"dataset": "x"}, {"model": "x"}, {}):
        try:
            _resolve(bad)
        except ValueError:
            continue
        assert False, f"expected ValueError for {bad}"


def test_sample_smaller_than_effective_batch_raises():
    """A pass over the sample must hold at least one whole optimizer step."""
    try:
        _resolve({**BASE, "sample": 8, "micro_batch": 4})         # 8 rows < effective_batch 16
    except ValueError as e:
        assert "no whole optimizer step" in str(e)
        return
    assert False, "expected ValueError when sample < effective_batch"


def test_sample_is_trimmed_to_whole_steps_from_the_end():
    """Every config keeps the SAME rows: the tail is dropped, file order is kept."""
    texts = [f"row{i}" for i in range(1000)]
    kept = _whole_steps(texts, 16)
    assert len(kept) == 992 and kept == texts[:992]            # 62 whole steps of 16
    assert _whole_steps(texts[:992], 16) == texts[:992]        # already whole: unchanged


def test_reference_rows_are_chosen_by_rule_around_the_median():
    lengths = [5, 900, 9, 12, 300, 9, 40, 7, 9, 11]
    ref = _reference_indices(lengths, 4)
    assert sorted(lengths[i] for i in ref) == [9, 9, 11, 12]   # the 4 rows centred on the median
    assert [lengths[i] for i in ref] == sorted(lengths[i] for i in ref)   # ordered by length
    assert _reference_indices(lengths, 4) == ref               # deterministic: ties break on index
    assert _reference_indices(lengths, 2) == ref[1:3]          # nested: same centre, fewer rows
    assert len(_reference_indices(lengths[:4], 4)) == 4        # sample == n: every row


def test_rows_hash_identifies_the_population():
    a, b = [[1, 2, 3], [4, 5]], [[1, 2, 3], [4, 6]]
    assert _rows_hash(a) == _rows_hash([[1, 2, 3], [4, 5]])    # same rows, same hash
    assert _rows_hash(a) != _rows_hash(b)                      # one token differs
    assert _rows_hash([[1, 2], [3]]) != _rows_hash([[1], [2, 3]])   # row boundaries matter


def _write_sample(d, **meta):
    from frame.dataset import write_sample
    path = os.path.join(d, "sample.jsonl")
    write_sample(path, {"meta": {"method": "length-stratified", "dataset": BASE["dataset"],
                                 "name": None, "split": "train", "text_field": "text", **meta},
                        "rows": [{"row": 5, "text": "hello there", "tokens": 2},
                                 {"row": 40, "text": "a much longer row", "tokens": 4}]})
    return path


def test_sample_file_supplies_the_rows_and_skips_the_head_check():
    import tempfile
    from frameworks.hf import _sample_file_texts
    with tempfile.TemporaryDirectory() as d:
        c = _resolve({**BASE, "sample_file": _write_sample(d), "sample": 1})   # sample ignored
        texts, meta = _sample_file_texts(c)
    assert texts == ["hello there", "a much longer row"] and meta["method"] == "length-stratified"


def test_sample_file_drawn_from_another_dataset_is_refused():
    """A different dataset/split/field is a different population: fail, don't mis-measure."""
    import tempfile
    from frameworks.hf import _sample_file_texts
    with tempfile.TemporaryDirectory() as d:
        c = _resolve({**BASE, "sample_file": _write_sample(d, split="validation")})
        try:
            _sample_file_texts(c)
        except ValueError as e:
            assert "split" in str(e)
            return
    assert False, "expected ValueError for a sample drawn from another split"


def test_usable_texts_skips_blanks_missing_fields_and_caps():
    rows = [{"text": "hello"}, {"text": "  "}, {"text": ""}, {"other": "x"},
            {"text": "world"}, {"text": "\n"}, {"text": "again"}]
    # blank/whitespace-only rows and the missing-field row are skipped (all-pad batch -> NaN loss)
    assert _usable_texts(rows, "text", sample=10) == ["hello", "world", "again"]
    assert _usable_texts(rows, "text", sample=2) == ["hello", "world"]         # capped at sample
    assert _usable_texts(rows, "text", sample=10) and _usable_texts([], "text", 10) == []


# --- the synthetic data source (what a ceiling search drives) --------------------------------

def test_max_len_is_translated_to_this_frameworks_seq_len():
    """`max_len` is the canonical name (loop/throughput.md); `seq_len` is this adapter's local one."""
    c = _resolve({**BASE, "max_len": 2048})
    assert c["seq_len"] == 2048 and c["tokens_per_step"] == 16 * 2048


def test_synthetic_needs_no_dataset():
    """It builds its batch — reading rows would defeat the point of a worst case."""
    c = _resolve({"model": BASE["model"], "data": "synthetic", "max_len": 2048})
    assert c["seq_len"] == 2048 and c["data"] == "synthetic"


def test_real_probes_still_require_a_dataset():
    try:
        _resolve({"model": BASE["model"]})
    except ValueError as e:
        assert "dataset" in str(e)
        return
    assert False, "expected ValueError: a throughput probe needs real rows"


def test_synthetic_skips_the_row_count_check():
    """`sample` bounds the loader, and synthetic has no loader — the check would be a false OOM."""
    c = _resolve({"model": BASE["model"], "data": "synthetic", "micro_batch": 512,
                  "effective_batch": 512, "sample": 8})
    assert c["micro_batch"] == 512 and c["grad_accum"] == 1


def test_fingerprint_is_a_token_mean_independent_of_micro_batch():
    """Same reference rows, mean over real tokens: neither half may move when only micro_batch
    (and so the batching) changes -- or a batch knob would read as a numerical change."""
    import torch
    from frameworks.hf import HFRunner
    r = HFRunner()
    r.device = "cpu"
    r.model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        r.model.weight.fill_(1.0)
    r.opt = torch.optim.SGD(r.model.parameters(), lr=0.0)
    rows = [torch.arange(1, 6), torch.arange(1, 12), torch.arange(1, 4)]   # 4 + 10 + 2 real tokens

    def fake_loss(ids, att, labels):                       # a true per-token mean, times the weight
        y = labels[:, 1:]
        return y[y != -100].float().mean() * r.model.weight.sum()
    r._loss = fake_loss

    expected = (sum(range(2, 6)) + sum(range(2, 12)) + sum(range(2, 4))) / 16   # 5.25
    for mb in (1, 2, 3):
        r.cfg = {"pad_multiple": 8, "seq_len": 32}
        r._reference = [r._pad_batch(rows[j:j + mb], 0) for j in range(0, len(rows), mb)]
        fp = r.reference_fingerprint()
        assert abs(fp["loss"] - expected) < 1e-6, mb                  # d(loss)/d(weight) = the
        assert abs(fp["grad_norm"] - expected) < 1e-6, mb             #   same token mean
        assert r.model.weight.grad is None                            # nothing left behind


def test_a_ceiling_trial_accumulates_once():
    """find_ceiling sets effective_batch = micro_batch: same peak memory, 1/grad_accum the time."""
    c = _resolve({"model": BASE["model"], "data": "synthetic",
                  "micro_batch": 4, "effective_batch": 4})
    assert c["grad_accum"] == 1


if __name__ == "__main__":
    for _n, _f in list(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f(); print(f"ok  {_n}")
    print("all passed")
