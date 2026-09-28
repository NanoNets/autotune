"""Trace-format drift check: does THIS torch still emit what derive.py reads? CPU only.

derive.py depends on names torch writes into its traces, not on a documented contract:
`enumerate(DataLoader)#...`, `autograd::engine::evaluate_function: ...`, `Optimizer.step#...`, and
the `Input Dims` field (record_shapes=True). This runs a real tiny training step under the profiler
and checks both the names and what derive recovers from them. Run it on every supported torch
version; a failure here means a rename, before it silently drops a phase from real probes.
"""
import json
import os
import sys

import torch
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import derive  # noqa: E402

VOCAB, BATCH, SEQ, ACCUM = 100, 4, 16, 2


def _profiled_step(tmp_path):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Embedding(VOCAB, 32), torch.nn.Linear(32, VOCAB))
    opt = torch.optim.AdamW(model.parameters())
    loader = iter(DataLoader(TensorDataset(torch.randint(0, VOCAB, (BATCH * ACCUM, SEQ))),
                             batch_size=BATCH))
    for _ in range(2):                                       # warm up outside the profiler
        x, = next(iter(DataLoader(TensorDataset(torch.randint(0, VOCAB, (BATCH, SEQ))),
                                  batch_size=BATCH)))
        model(x).logsumexp(-1).mean().backward()
        opt.step()
        opt.zero_grad()
    with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
        with record_function("step"):
            for _ in range(ACCUM):
                x, = next(loader)
                (model(x).logsumexp(-1).mean() / ACCUM).backward()
            opt.step()
            opt.zero_grad()
    path = tmp_path / "trace.json"
    prof.export_chrome_trace(str(path))
    return json.loads(path.read_text())


def test_torch_still_emits_the_events_derive_reads(tmp_path):
    ev = _profiled_step(tmp_path)["traceEvents"]
    names = {e.get("name", "") for e in ev}
    assert any(n.startswith(derive.LOADER) for n in names), f"torch {torch.__version__}: DataLoader label"
    assert any(n.startswith(derive.EVALF) for n in names), f"torch {torch.__version__}: autograd events"
    assert any(n.startswith("Optimizer.step#") for n in names), f"torch {torch.__version__}: optimizer label"
    emb = [e for e in ev if e.get("name") == "aten::embedding"]
    assert emb and "Input Dims" in (emb[0].get("args") or {}), f"torch {torch.__version__}: Input Dims"


def test_derive_recovers_micro_batches_and_tokens(tmp_path):
    t = _profiled_step(tmp_path)
    found = derive.augment(t)
    assert (found["forwards"], found["backwards"]) == (ACCUM, ACCUM), found
    assert found["tokens"] == ACCUM * BATCH * SEQ and found["token_source"] == "embedding"
    assert found["notes"] == []
