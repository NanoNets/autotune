"""User-script ceiling trials, end to end on CPU: Frame's ceiling rows -> the script's own data
setting -> `profiler.launch --memory-only` -> a trial record -> the divisor search.

Each test writes a small user-style training script (its own argparse, JSONL reader, collator and
DataLoader, no autotune imports). Memory reads as None on CPU, so an OOM is simulated by the
script raising the exception a real card raises, at step 2, where AdamW's state first exists.
"""
import json
import os
import subprocess
import sys
import textwrap

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import backends  # noqa: E402
from frame import ceiling as C  # noqa: E402
from frame.ceiling_rows import write_ceiling_rows  # noqa: E402

SCRIPT = textwrap.dedent('''
    """A user's training script: reads JSONL rows of token ids, truncates, pads to the longest."""
    import argparse, json, torch
    from torch.utils.data import DataLoader
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--bs", type=int, default=2)
    ap.add_argument("--max_len", type=int, default=8)
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--oom_above", type=int, default=0)      # simulate a card that fits <= this
    ap.add_argument("--ignore_data", action="store_true")    # a bug: reads its own default rows
    ap.add_argument("--reference_first", action="store_true")
    a = ap.parse_args()

    def read(path):
        with open(path) as f:
            return [[hash(w) % 60 + 2 for w in json.loads(l)["text"].split()] for l in f]
    rows = [[5, 6, 7]] * 64 if (a.ignore_data or not a.data_path) else read(a.data_path)
    rows = [r[:a.max_len] for r in rows]

    def collate(batch):
        n = max(len(r) for r in batch)
        return torch.tensor([r + [0] * (n - len(r)) for r in batch])
    loader = DataLoader(rows, batch_size=a.bs, collate_fn=collate, drop_last=True)
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Embedding(64, 8), torch.nn.Linear(8, 64))
    ref = torch.nn.Sequential(torch.nn.Embedding(64, 8), torch.nn.Linear(8, 64)).requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters())
    step = 0
    while step < a.max_steps:
        for x in loader:
            if a.reference_first:
                ref(torch.zeros(3, 5, dtype=torch.long))
            model(x).logsumexp(-1).mean().backward()
            opt.step(); opt.zero_grad()
            step += 1
            if a.oom_above and a.bs > a.oom_above and step == 2:
                raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
            if step >= a.max_steps:
                break
''')


class _Tok:
    """Whitespace 'tokenizer': id i <-> word w<i>. Ids 0 and 1 are special."""
    all_special_ids = [0, 1]

    def __len__(self):
        return 64

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"w{i}" for i in ids if not (skip_special_tokens and i in (0, 1)))

    def encode(self, text, add_special_tokens=False):
        return [int(w[1:]) for w in text.split()]


@pytest.fixture
def setup(tmp_path):
    script = tmp_path / "train.py"
    script.write_text(SCRIPT)
    rows = tmp_path / "ceiling_rows.jsonl"
    write_ceiling_rows(str(rows), _Tok(), max_len=8, n_rows=16 * 3)
    return script, rows


def _launch(tmp_path, script, *args, steps=3, expect="2,8"):
    out = tmp_path / "trial.json"
    p = subprocess.run([sys.executable, "-m", "profiler.launch", "--out", str(out), "--device", "cpu",
                        "--memory-only", "--steps", str(steps), "--expect-shape", expect, "--",
                        str(script), *args], capture_output=True, text=True, cwd=REPO)
    return p, json.loads(out.read_text())["trial"]


def test_the_rows_encode_past_max_len_and_skip_special_tokens(tmp_path):
    p = tmp_path / "rows.jsonl"
    write_ceiling_rows(str(p), _Tok(), max_len=32, n_rows=5)
    rows = [json.loads(l) for l in p.read_text().splitlines()]
    assert len(rows) == 5
    for r in rows:
        ids = _Tok().encode(r["text"])
        assert len(ids) >= 32 and not {0, 1} & set(ids)


def test_a_row_template_puts_the_text_where_the_script_reads_it(tmp_path):
    p = tmp_path / "rows.jsonl"
    tmpl = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "{TEXT}"}]}
    write_ceiling_rows(str(p), _Tok(), max_len=16, n_rows=2, template=json.dumps(tmpl))
    row = json.loads(p.read_text().splitlines()[0])
    assert row["messages"][0]["content"] == "hi" and len(row["messages"][1]["content"].split()) >= 16
    with pytest.raises(ValueError, match="TEXT"):
        write_ceiling_rows(str(p), _Tok(), max_len=16, n_rows=1, template='{"text": "no placeholder"}')


def test_pretokenized_rows_are_exactly_max_len(tmp_path):
    p = tmp_path / "rows.jsonl"
    write_ceiling_rows(str(p), _Tok(), max_len=12, n_rows=3, fmt="ids")
    assert all(len(json.loads(l)["input_ids"]) == 12 for l in p.read_text().splitlines())


def test_a_trial_runs_three_steps_and_stops_the_script(tmp_path, setup):
    script, rows = setup
    p, t = _launch(tmp_path, script, "--data_path", str(rows), "--bs", "2")
    assert p.returncode == 0, p.stderr
    assert t["status"] == "ok" and t["steps"] == 3          # the script asked for 50
    assert t["shapes_seen"] == [[2, 8]]


def test_a_trial_that_ooms_at_step_two_is_oom(tmp_path, setup):
    script, rows = setup
    p, t = _launch(tmp_path, script, "--data_path", str(rows), "--bs", "4", "--oom_above", "2",
                   expect="4,8")
    assert t["status"] == "oom" and "CUDA out of memory" in t["error"]


@pytest.mark.parametrize("args,why", [(["--ignore_data"], "fed [2, 3]"),
                                      (["--max_len", "6"], "fed [2, 6]")])
def test_a_batch_that_is_not_the_worst_case_fails_the_trial(tmp_path, setup, args, why):
    """The script ignored the rows, or truncates at another length: the trial measured some other
    workload, so it can't be a memory answer, even an `oom`."""
    script, rows = setup
    p, t = _launch(tmp_path, script, "--data_path", str(rows), "--bs", "2", *args)
    assert t["status"] == "error" and why in t["error"]


def test_a_model_called_with_keyword_inputs_is_checked(tmp_path):
    """HF Trainer calls model(**batch): the one-shot global hook can't see kwargs, so the model's
    own pre-hook must catch every later call, and a kwargs-only first call must not read as a
    missing token tensor."""
    script = tmp_path / "kw.py"
    script.write_text(textwrap.dedent('''
        import torch
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.e, self.o = torch.nn.Embedding(64, 8), torch.nn.Linear(8, 64)
            def forward(self, input_ids=None, labels=None):
                return self.o(self.e(input_ids)).logsumexp(-1).mean()
        m = M(); opt = torch.optim.AdamW(m.parameters())
        for _ in range(10):
            m(input_ids=torch.zeros(2, 8, dtype=torch.long)).backward(); opt.step(); opt.zero_grad()
    '''))
    p, t = _launch(tmp_path, script)
    assert t["status"] == "ok", t
    assert t["shapes_seen"] == [[2, 8]]


def test_an_oom_in_the_first_kwargs_forward_still_counts_as_does_not_fit(tmp_path):
    """Found on the A10: micro-batch 32 OOMed inside step 1's model(**batch), before any shape was
    readable. That's still "doesn't fit" (marked unchecked): it can only make the ceiling lower."""
    script = tmp_path / "kw_oom.py"
    script.write_text(textwrap.dedent('''
        import torch
        class M(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.e = torch.nn.Embedding(64, 8)
            def forward(self, input_ids=None):
                raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 3.00 GiB")
        m = M(); opt = torch.optim.AdamW(m.parameters())
        m(input_ids=torch.zeros(2, 8, dtype=torch.long))
    '''))
    p, t = _launch(tmp_path, script)
    assert t["status"] == "oom" and t["shape_checked"] is False


def test_a_frozen_model_called_first_is_not_the_one_checked(tmp_path, setup):
    script, rows = setup
    p, t = _launch(tmp_path, script, "--data_path", str(rows), "--bs", "2", "--reference_first")
    assert t["status"] == "ok", t


def test_a_script_that_stops_early_is_an_error(tmp_path, setup):
    script, rows = setup
    p, t = _launch(tmp_path, script, "--data_path", str(rows), "--max_steps", "1")
    assert t["status"] == "error" and t["error"] == "script ended after 1 optimizer steps; needs 3"


def test_one_step_is_refused(tmp_path, setup):
    script, rows = setup
    p, t = _launch(tmp_path, script, "--data_path", str(rows), steps=1)
    assert p.returncode == 2 and t["status"] == "error" and "needs >= 2" in t["error"]


def test_a_backend_trial_returns_the_record(setup):
    script, rows = setup
    t = backends.run({"backend": "local", "probe_kind": "ceiling_trial", "device": "cpu",
                      "script": str(script), "script_args": ["--data_path", str(rows), "--bs", "2"],
                      "micro_batch": 2, "max_len": 8, "trial_steps": 3})
    assert t["status"] == "ok" and t["steps"] == 3


def test_a_script_ceiling_end_to_end(setup):
    """The divisor search over real launches: fits <= 4 of effective_batch 16 -> 4, in 4 trials."""
    script, rows = setup
    cfg = {"backend": "local", "device": "cpu", "script": str(script), "dtype": "float32",
           "script_args": ["--data_path", str(rows), "--bs", "{micro_batch}", "--oom_above", "4"],
           "memory_knobs": {"compile": False}, "max_len": 8}
    rec = C.measure(C.backend_trial(cfg), 16, 8, C.stamp(cfg), method="ceiling-rows")
    assert rec["micro_batch_max"] == 4 and rec["oom_at"] == 8
    assert [t["micro_batch"] for t in rec["trials"]] == [1, 2, 4, 8]
    assert rec["config"] == {"dtype": "float32", "compile": False}
