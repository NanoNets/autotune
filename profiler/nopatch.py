"""Guard: prove a run left torch exactly as it found it (bring-your-own-script, `profiler/launch.py`).

`snapshot()` before installing, `verify(before)` after the run. Fails if any of the torch
functions the old sandbox used to monkeypatch was replaced, or if any global module hook is
registered. torch's own wrapping of `Optimizer.step` (done in `Optimizer.__init__`) is torch's
behaviour, not ours, and isn't checked here.
"""
import torch
import torch.nn.modules.module as _module
from torch.utils.data.dataloader import _BaseDataLoaderIter

_GLOBAL_HOOK_TABLES = ("_global_forward_pre_hooks", "_global_forward_hooks",
                       "_global_backward_pre_hooks", "_global_backward_hooks",
                       "_global_module_registration_hooks", "_global_parameter_registration_hooks",
                       "_global_buffer_registration_hooks")


def snapshot():
    return {
        "torch.autograd.backward": torch.autograd.backward,
        "torch.Tensor.backward": torch.Tensor.backward,
        "_BaseDataLoaderIter.__next__": _BaseDataLoaderIter.__next__,
        "torch.nn.Module.__call__": torch.nn.Module.__call__,
    }


def verify(before):
    after = snapshot()
    replaced = [k for k in before if after[k] is not before[k]]
    hooks = {t: len(getattr(_module, t)) for t in _GLOBAL_HOOK_TABLES
             if hasattr(_module, t) and len(getattr(_module, t))}
    if replaced or hooks:
        raise SystemExit(f"monkeypatch guard: replaced={replaced} global module hooks={hooks}")
    return {"replaced": [], "global_module_hooks": {}}
