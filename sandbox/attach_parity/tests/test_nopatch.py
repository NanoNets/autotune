"""The monkeypatch guard: passes on a clean process, fails on each kind of patch. CPU only."""
import os
import sys

import pytest
import torch
import torch.nn.modules.module as _module

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nopatch  # noqa: E402


def test_clean_process_passes():
    before = nopatch.snapshot()
    assert nopatch.verify(before) == {"replaced": [], "global_module_hooks": {}}


def test_replaced_torch_function_fails():
    before = nopatch.snapshot()
    orig = torch.autograd.backward
    torch.autograd.backward = lambda *a, **k: orig(*a, **k)
    try:
        with pytest.raises(SystemExit, match="torch.autograd.backward"):
            nopatch.verify(before)
    finally:
        torch.autograd.backward = orig
    nopatch.verify(before)                                     # restored -> passes again


def test_replaced_dataloader_iterator_fails():
    from torch.utils.data.dataloader import _BaseDataLoaderIter
    before = nopatch.snapshot()
    orig = _BaseDataLoaderIter.__next__
    _BaseDataLoaderIter.__next__ = lambda self: orig(self)
    try:
        with pytest.raises(SystemExit, match="__next__"):
            nopatch.verify(before)
    finally:
        _BaseDataLoaderIter.__next__ = orig


def test_leftover_global_module_hook_fails():
    before = nopatch.snapshot()
    h = _module.register_module_forward_pre_hook(lambda m, a: None)
    try:
        with pytest.raises(SystemExit, match="_global_forward_pre_hooks"):
            nopatch.verify(before)
    finally:
        h.remove()
    nopatch.verify(before)
