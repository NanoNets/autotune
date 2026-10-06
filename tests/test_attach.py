"""Attach in-process on CPU: the state machine, the Capture, and that it leaves nothing behind."""
import os
import sys

import torch
from torch.optim.optimizer import _global_optimizer_post_hooks

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root

from profiler import nopatch  # noqa: E402
from profiler.attach import Attach  # noqa: E402

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


# ---- B1: tokens are counted on every clean step (decision 2026-09-27) ----------------------

def _variable_width_run(widths, accum=1, float_input=False):
    """A loop whose batch width changes every step. Returns (att, per-step padded tokens)."""
    torch.manual_seed(0)
    before = nopatch.snapshot()
    att = Attach(device="cpu", warmup=WARMUP, clean_steps=CLEAN, peak_flops=1.0, peak_mem_bw_gb_s=1.0)
    att.install()
    if float_input:
        model = torch.nn.Linear(16, 16)
    else:
        model = torch.nn.Sequential(torch.nn.Embedding(50, 16), torch.nn.Linear(16, 50))
    opt = torch.optim.AdamW(model.parameters())
    expected = []
    for w in widths:
        for _ in range(accum):
            x = torch.randn(4, w, 16) if float_input else torch.randint(0, 50, (4, w))
            model(x).square().mean().backward() if float_input else \
                model(x).logsumexp(-1).mean().backward()
        opt.step()
        opt.zero_grad()
        expected.append(4 * w * accum)
    nopatch.verify(before)                                               # still patch-free
    return att, expected


# steps 1-2 warmup, 3-5 clean, 6 traced (much narrower than the clean mean), 7 after
WIDTHS = [8, 8, 30, 90, 60, 5, 8]


def test_b1_tokens_come_from_the_clean_steps_not_the_traced_step():
    att, expected = _variable_width_run(WIDTHS)
    clean = expected[WARMUP:WARMUP + CLEAN]                               # steps 3-5
    assert att.step_tokens == clean                                       # one count per timed step
    assert att.step_samples == [4] * CLEAN                                # rows, same steps
    assert att.tokens == sum(clean) / len(clean)                          # rate numerator = their mean
    traced = att.micro_batches["traced_step_tokens"]
    assert traced["hook"] == traced["trace"] == expected[WARMUP + CLEAN]  # hook and trace agree...
    assert traced["trace"] < 0.2 * att.tokens                             # ...but one step would have
    assert att.micro_batches["token_source"].startswith("hook: mean of 3")  # been ~10x off (the bug)


def test_b1_grad_accum_sums_every_micro_batch_of_a_step():
    att, expected = _variable_width_run(WIDTHS, accum=2)
    assert att.step_tokens == expected[WARMUP:WARMUP + CLEAN]
    assert att.step_samples == [8] * CLEAN                                # 2 micro-batches of 4
    assert att.micro_batches["forwards"] == att.micro_batches["backwards"] == 2


def test_b1_float_activation_input_counts_rows_times_seq():
    att, expected = _variable_width_run(WIDTHS, float_input=True)
    assert att.step_tokens == expected[WARMUP:WARMUP + CLEAN]


def test_b1_no_hook_left_on_the_model():
    att, _ = _variable_width_run(WIDTHS)
    assert att._tok_hook is None and att._root_hook is None
    assert att._root is not None and len(att._root._forward_pre_hooks) == 0


# ---- samples under packing: documents, not rows -------------------------------------------

class _KwModel(torch.nn.Module):
    """A model called HF-style, with keyword inputs; packing metadata rides alongside."""
    def __init__(self):
        super().__init__()
        self.emb, self.head = torch.nn.Embedding(50, 16), torch.nn.Linear(16, 50)

    def forward(self, input_ids, position_ids=None, cu_seqlens=None, cu_seq_lens_q=None):
        return self.head(self.emb(input_ids))


def _packed_run(batches):
    """One optimizer step per batch; each batch is the kwargs the model is called with."""
    torch.manual_seed(0)
    before = nopatch.snapshot()
    att = Attach(device="cpu", warmup=WARMUP, clean_steps=CLEAN, peak_flops=1.0, peak_mem_bw_gb_s=1.0)
    att.install()
    model = _KwModel()
    opt = torch.optim.AdamW(model.parameters())
    for kw in batches:
        model(**kw).logsumexp(-1).mean().backward()
        opt.step()
        opt.zero_grad()
    nopatch.verify(before)
    return att


def _pos(*doc_lens):
    """position_ids for one row packing documents of the given lengths."""
    return torch.cat([torch.arange(n) for n in doc_lens])


def test_packed_rows_count_documents_via_position_ids():
    ids = torch.randint(0, 50, (2, 12))                    # 2 rows, 3 + 2 documents packed in them
    pos = torch.stack([_pos(4, 4, 4), _pos(6, 6)])
    att = _packed_run([{"input_ids": ids, "position_ids": pos}] * (WARMUP + CLEAN + 2))
    assert att.step_samples == [5] * CLEAN                 # documents, not the 2 rows
    assert att.step_tokens == [24] * CLEAN                 # tokens unchanged: padded numel
    assert att.micro_batches["sample_source"] == "position_ids"


def test_broadcast_position_row_counts_every_batch_row():
    ids = torch.randint(0, 50, (4, 8))
    pos = torch.arange(8)[None]                            # [1, S], as HF-style scripts pass it
    att = _packed_run([{"input_ids": ids, "position_ids": pos}] * (WARMUP + CLEAN + 2))
    assert att.step_samples == [4] * CLEAN                 # one zero, broadcast over 4 rows


def test_flattened_packing_counts_via_cu_seqlens():
    ids = torch.randint(0, 50, (30,))                      # 1-D: three documents of 10
    cu = torch.tensor([0, 10, 20, 30])
    att = _packed_run([{"input_ids": ids, "cu_seqlens": cu}] * (WARMUP + CLEAN + 2))
    assert att.step_samples == [3] * CLEAN
    assert att.micro_batches["sample_source"] == "cu_seqlens"


def test_flattened_input_without_boundaries_is_unknown_not_the_token_count():
    ids = torch.randint(0, 50, (30,))
    att = _packed_run([{"input_ids": ids}] * (WARMUP + CLEAN + 2))
    assert att.step_samples == []                          # never 30
    assert att.step_tokens == [30] * CLEAN                 # tokens still known
    assert att.micro_batches["sample_source"].startswith("unknown")
