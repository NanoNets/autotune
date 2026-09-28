"""Attach in-process on CPU: the state machine, the Capture, and that it leaves nothing behind."""
import os
import sys

import torch
from torch.optim.optimizer import _global_optimizer_post_hooks

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nopatch  # noqa: E402
from attach import Attach  # noqa: E402

WARMUP, CLEAN = 2, 3


def _run(steps, second_optimizer=False):
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Embedding(50, 16), torch.nn.Linear(16, 50))
    before = nopatch.snapshot()
    att = Attach(device="cpu", warmup=WARMUP, clean_steps=CLEAN, peak_flops=1.0, peak_mem_bw_gb_s=1.0)
    att.install()
    opt = torch.optim.AdamW(model[1].parameters())
    other = torch.optim.SGD(model[0].parameters(), lr=0.1) if second_optimizer else None
    for _ in range(steps):
        model(torch.randint(0, 50, (4, 8))).logsumexp(-1).mean().backward()
        opt.step()
        if other is not None:
            other.step()
        opt.zero_grad()
        if other is not None:
            other.zero_grad()
    nopatch.verify(before)
    return att, model


def test_state_machine_and_capture():
    att, model = _run(steps=WARMUP + CLEAN + 3)
    cap = att.result()
    assert att.done and cap.status == "ok"
    assert len(cap.step_times_s) == CLEAN and all(t > 0 for t in cap.step_times_s)
    assert cap.trace and os.path.exists(cap.trace)
    assert att.micro_batches["forwards"] == att.micro_batches["backwards"] == 1
    assert att.tokens == 4 * 8


def test_params_come_from_the_first_optimizer_only():
    att, model = _run(steps=WARMUP + CLEAN + 3, second_optimizer=True)
    assert att.params == sum(p.numel() for p in model[1].parameters())   # AdamW, not the SGD
    assert len(att.result().step_times_s) == CLEAN                       # one boundary per step


def test_nothing_left_registered_after_done():
    n_before = len(_global_optimizer_post_hooks)
    att, _ = _run(steps=WARMUP + CLEAN + 3)
    assert att.done and len(_global_optimizer_post_hooks) == n_before


def test_too_few_steps_never_finishes_and_uninstall_stops_the_profiler():
    att, _ = _run(steps=WARMUP + CLEAN)                                  # profiled step never ends
    assert not att.done and att.result() is None and att._profiling
    att.uninstall()
    assert not att._profiling
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]):
        pass                                                             # a new profiler can start
