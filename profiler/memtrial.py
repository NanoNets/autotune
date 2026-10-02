"""A memory-only ceiling trial inside a user's script: run `steps` optimizer steps, then stop.

`python -m profiler.launch --memory-only ...` installs this instead of `Attach`. It answers one
question, "does this micro-batch fit?", so it times nothing and runs no profiler:

- **Steps.** `register_optimizer_step_post_hook` (public, global) counts optimizer steps. After
  `steps` of them (default 3) it reads the allocator's peaks and stops the script by raising
  `TrialDone` out of the hook. Step 1 alone isn't the answer: AdamW allocates its state during
  step 1's `opt.step()`, after the activations are freed, so the steady-state peak first appears at
  step 2. Step 3 covers CUDA-graph pools (compile `reduce-overhead` / `max-autotune`) and early
  allocator growth.
- **Shape check.** A ceiling trial feeds the script a file of over-length rows through its own
  data setting (`frame/ceiling_rows.py`), so every batch should be exactly `[micro_batch, max_len]`.
  A one-shot global `register_module_forward_pre_hook` finds the model (the first module called
  that has a trainable parameter, so a parameter-free transform or a frozen reference model is
  passed over), then a per-module pre-hook reads the shape of every input it gets. Any other shape
  fails the trial: the script ignored the file, dropped or split the long rows, packed them, or
  runs at a different sequence length. Shapes only, so no GPU sync.

Nothing is patched; every hook is removed before the process exits (`nopatch.verify`).
"""
import torch
import torch.nn.modules.module as _module
from torch.optim.optimizer import register_optimizer_step_post_hook


class TrialDone(BaseException):
    """Raised from the optimizer hook once the trial has its answer. A BaseException, so a script's
    own `except Exception:` can't swallow it and keep training."""


def _input_shape(args, kwargs):
    """(batch, seq) of the token tensor the model was called with, or None if it has none."""
    x = kwargs.get("input_ids")
    if x is None:
        for a in args:
            if isinstance(a, torch.Tensor):
                x = a
                break
            if isinstance(a, dict) and isinstance(a.get("input_ids"), torch.Tensor):
                x = a["input_ids"]
                break
    if not isinstance(x, torch.Tensor) or x.dim() < 2:
        return None
    return int(x.shape[0]), int(x.shape[1])


def _memory(device):
    if device != "cuda":
        return {"peak_gb": None, "reserved_gb": None, "total_gb": None}
    gb = 1024 ** 3
    return {"peak_gb": round(torch.cuda.max_memory_allocated() / gb, 3),
            "reserved_gb": round(torch.cuda.max_memory_reserved() / gb, 3),
            "total_gb": round(torch.cuda.mem_get_info()[1] / gb, 3)}


class MemoryTrial:
    def __init__(self, device="cuda", steps=3, expect_shape=None):
        if steps < 2:
            raise ValueError(f"steps={steps}: needs >= 2 (AdamW's state first appears at step 2)")
        self.device, self.steps, self.expect = device, steps, expect_shape
        self.n = 0
        self.shapes = []                                 # every (batch, seq) the model was fed
        self.mismatch = None                             # the first shape that wasn't expected
        self.memory = None
        self._opt = self._step_hook = self._root_hook = self._shape_hook = None

    def install(self):
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        self._step_hook = register_optimizer_step_post_hook(self._on_step)
        self._root_hook = _module.register_module_forward_pre_hook(self._find_root)

    def uninstall(self):
        for name in ("_step_hook", "_root_hook", "_shape_hook"):
            h = getattr(self, name)
            if h is not None:
                h.remove()
                setattr(self, name, None)

    def _find_root(self, mod, args):
        if not any(p.requires_grad for p in mod.parameters()):
            return                                        # a transform, or a frozen model
        self._shape_hook = mod.register_forward_pre_hook(self._see, with_kwargs=True)
        self._root_hook.remove()
        self._root_hook = None
        self._see(mod, args, {})                          # this call counts too

    def _see(self, mod, args, kwargs):
        shape = _input_shape(args, kwargs)
        self.shapes.append(shape)
        if self.expect and shape != tuple(self.expect) and self.mismatch is None:
            self.mismatch = shape or ()                   # () = called with no token tensor

    def _on_step(self, opt, args, kwargs):
        if self._opt is None:                             # the first optimizer; ignore others
            self._opt = opt
        if opt is not self._opt:
            return
        self.n += 1
        if self.n >= self.steps:
            if self.device == "cuda":
                torch.cuda.synchronize()
            self.memory = _memory(self.device)
            raise TrialDone()

    def result(self, status, error=None):
        """The trial record. A wrong or unreadable batch shape overrides any status: the trial
        then measured some other workload, so its memory says nothing about this ceiling."""
        if self.memory is None:
            self.memory = _memory(self.device)
        if self.expect and status in ("ok", "oom"):
            if self.mismatch is not None:
                status, error = "error", (
                    f"script fed {list(self.mismatch) if self.mismatch else 'no token tensor'}, "
                    f"not {list(self.expect)}: it ignored the ceiling rows, dropped / split / "
                    "packed them, or runs at another sequence length")
            elif not any(self.shapes):
                status, error = "error", "the batch shape couldn't be checked: no model input seen"
        return {"status": status, "error": error, "steps": self.n,
                "shapes_seen": sorted({s for s in self.shapes if s}), **self.memory}
