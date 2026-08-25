"""HF-adapter tests — the pure parts (knobs surface + config resolution). No torch/transformers.

setup()/step()/_build_loader need a real model, data, and a GPU, so they're validated live, not
here. `_resolve` and `knobs` are import-clean (heavy imports are lazy), so they test on bare python3.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frameworks.hf import knobs, _resolve, _usable_texts

BASE = {"model": "Qwen/Qwen2.5-0.5B", "dataset": "Salesforce/wikitext"}


def test_knobs_are_speed_levers_only():
    k = knobs()
    assert set(k) == {"micro_batch", "grad_accum", "seq_len", "dtype",
                      "gradient_checkpointing", "attn_implementation", "torch_compile"}
    assert k["dtype"]["options"] == ["bfloat16", "float16", "float32"]
    assert k["micro_batch"]["type"] == "int"


def test_resolve_fills_defaults():
    c = _resolve(BASE)
    assert c["device"] == "cuda" and c["dtype"] == "bfloat16"
    assert c["micro_batch"] == 1 and c["grad_accum"] == 1 and c["seq_len"] == 1024


def test_tokens_per_step_counts_accumulation():
    c = _resolve({**BASE, "micro_batch": 4, "seq_len": 512, "grad_accum": 8})
    assert c["tokens_per_step"] == 4 * 512 * 8            # one optimizer step's worth of tokens


def test_overrides_win_over_defaults():
    c = _resolve({**BASE, "dtype": "float32", "attn_implementation": "flash_attention_2"})
    assert c["dtype"] == "float32" and c["attn_implementation"] == "flash_attention_2"


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


if __name__ == "__main__":
    for _n, _f in list(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f(); print(f"ok  {_n}")
    print("all passed")
