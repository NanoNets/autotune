"""HF-adapter tests — the pure parts (knobs surface + config resolution). No torch/transformers.

setup()/step()/_build_loader need a real model, data, and a GPU, so they're validated live, not
here. `_resolve` and `knobs` are import-clean (heavy imports are lazy), so they test on bare python3.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frameworks.hf import knobs, _resolve, _usable_texts

BASE = {"model": "Qwen/Qwen2.5-0.5B", "dataset": "Salesforce/wikitext"}


def test_knobs_are_safe_levers_only():
    k = knobs()
    assert set(k) == {"micro_batch", "gradient_checkpointing",
                      "attn_implementation", "torch_compile", "tf32"}
    # dtype (precision) and seq_len (max_length) are off-limits run params, NOT searchable knobs;
    # grad_accum is derived, not free.
    assert "dtype" not in k and "seq_len" not in k and "grad_accum" not in k
    assert k["attn_implementation"]["options"] == ["eager", "sdpa", "flash_attention_2"]


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


def test_sample_smaller_than_micro_batch_raises():
    try:
        _resolve({**BASE, "sample": 4, "micro_batch": 8})
    except ValueError as e:
        assert "loader would be empty" in str(e)
        return
    assert False, "expected ValueError when sample < micro_batch"


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
