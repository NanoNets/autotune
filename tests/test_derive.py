"""derive.py on hand-built traces. Torch-free: these pin down the RULES, independent of any torch
version. `test_trace_format.py` checks that a real torch still emits what these rules read.

  .venv/bin/python -m pytest tests/test_derive.py -q
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root

from profiler import derive  # noqa: E402

MAIN, AUTOGRAD = 1, 2
LOADER = "enumerate(DataLoader)#_SingleProcessDataLoaderIter.__next__"


def op(name, ts, dur, tid=MAIN, dims=None):
    e = {"ph": "X", "cat": "cpu_op", "name": name, "ts": ts, "dur": dur, "pid": 0, "tid": tid}
    if dims is not None:
        e["args"] = {"Input Dims": dims}
    return e


def ann(name, ts, dur, tid=MAIN):
    return {"ph": "X", "cat": "user_annotation", "name": name, "ts": ts, "dur": dur,
            "pid": 0, "tid": tid}


def micro_batch(b, backward_tid, loader=True, embed=True):
    """One micro-batch starting at ts=b: fetch, copy, forward (+ loss), backward."""
    ev = []
    if loader:
        ev.append(ann(LOADER, b + 10, 30))
    ev.append(op("aten::to", b + 45, 5))                            # batch -> device
    if embed:
        ev.append(op("aten::embedding", b + 55, 5, dims=[[1000, 16], [4, 32]]))
        ev.append(op("aten::linear", b + 60, 40, dims=[[4, 32, 16], [16, 16], [16]]))
    else:
        ev.append(op("aten::addmm", b + 55, 45, dims=[[16], [128, 16], [16, 16]]))
    ev.append(op("aten::mean", b + 100, 5))                          # the loss, outside any module
    ev.append(op("autograd::engine::evaluate_function: MeanBackward0", b + 110, 40, backward_tid))
    ev.append(op("aten::mm", b + 120, 10, backward_tid))             # nested in the node above
    ev.append(op("autograd::engine::evaluate_function: AddmmBackward0", b + 150, 50, backward_tid))
    return ev


def trace(n_micro=2, backward_tid=AUTOGRAD, loader=True, embed=True, step=True):
    ev = [ann("step", 0, 1000)] if step else []
    for k in range(n_micro):
        ev += micro_batch(k * 400, backward_tid, loader, embed)
    t_opt = n_micro * 400 + 50                                       # after the last micro-batch
    ev.append(ann("Optimizer.step#AdamW.step", t_opt, 100))
    ev.append(op("aten::_foreach_add_", t_opt + 10, 40))             # inside the optimizer label
    if step:
        ev[0]["dur"] = t_opt + 150
    return {"traceEvents": ev}


def spans(t, name):
    return sorted((e["ts"], e["ts"] + e["dur"]) for e in t["traceEvents"]
                  if e.get("cat") == "user_annotation" and e["name"] == name
                  and (e.get("args") or {}).get("derived") == "attach.derive")


def test_cuda_style_backward_on_its_own_thread():
    t = trace(backward_tid=AUTOGRAD)
    found = derive.augment(t)
    assert (found["forwards"], found["backwards"]) == (2, 2)
    assert found["tokens"] == 2 * 4 * 32 and found["token_source"] == "embedding"
    assert found["notes"] == []
    # forward starts at the first NON-copy op after the batch arrives, and ends where backward starts
    assert spans(t, "forward") == [(55, 110), (455, 510)]
    assert spans(t, "backward") == [(110, 200), (510, 600)]


def test_cpu_style_backward_on_the_main_thread():
    t = trace(backward_tid=MAIN)                                     # nested aten::mm is on MAIN too
    found = derive.augment(t)
    assert (found["forwards"], found["backwards"]) == (2, 2)
    assert spans(t, "backward") == [(110, 200), (510, 600)]


def test_forward_includes_the_loss_and_excludes_the_copy():
    t = trace()
    derive.augment(t)
    f0, f1 = spans(t, "forward")[0]
    assert f0 == 55                                                  # after aten::to (45-50)
    assert f1 >= 105                                                 # aten::mean (100-105) inside


def test_dataloader_span_is_torchs_own_label_renamed():
    t = trace()
    derive.augment(t)
    assert spans(t, "dataloader") == [(10, 40), (410, 440)]


def test_no_dataloader_means_no_dataloader_span():
    t = trace(loader=False)
    found = derive.augment(t)
    assert spans(t, "dataloader") == []
    assert (found["forwards"], found["backwards"]) == (2, 2)


def test_tokens_fall_back_to_the_first_linear():
    t = trace(embed=False)                                           # addmm(bias, input[128,16], w)
    found = derive.augment(t)
    assert found["tokens"] == 2 * 128 and found["token_source"] == "first-linear"


def test_optimizer_ops_are_never_forward():
    t = trace(n_micro=1)                                             # optimizer label at 450-550
    derive.augment(t)
    assert all(b <= 450 for _, b in spans(t, "forward"))


def test_grad_accum_count_follows_the_micro_batches():
    for n in (1, 3, 4):
        found = derive.augment(trace(n_micro=n))
        assert (found["forwards"], found["backwards"]) == (n, n)


def test_no_step_label_is_a_note_not_a_crash():
    found = derive.augment(trace(step=False))
    assert found["forwards"] == found["backwards"] == 0
    assert found["tokens"] is None and found["notes"] == ["no `step` annotation"]


def test_derived_spans_are_marked():
    t = trace()
    derive.augment(t)
    derived = [e for e in t["traceEvents"] if (e.get("args") or {}).get("derived")]
    assert {e["name"] for e in derived} == {"forward", "backward", "dataloader"}
    assert all(e["cat"] == "user_annotation" and e["tid"] == MAIN for e in derived)
