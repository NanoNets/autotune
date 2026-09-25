"""SANDBOX — hook-driven capture: profile a training loop autotune does not own.

`capture()` drives a runner — it calls `step()` itself and the runner wraps its phases in
`record_function`. A user's script (swift, TRL, a hand-written loop) offers neither. `Attach`
inverts control: the user's loop runs as-is, and process-global torch hooks tell us where the
step boundaries are and label the phases. It produces the same `Capture` that `capture()` does,
so `diagnose()` is unchanged. See `bring_your_own_script.md` §2.

    att = Attach(device="cuda", peak_flops=..., peak_mem_bw_gb_s=...)
    att.install()              # before the user builds the optimizer
    ... user's loop runs ...   # att.done flips after warmup + clean + 1 profiled step
    cap = att.result()

Step boundary: one optimizer step of the FIRST optimizer seen. The first step is discovered
through the global post-hook, which runs INSIDE torch's `Optimizer.step#...` annotation; starting
or stopping the profiler in there would cut that annotation in half. So on discovery the
optimizer's class `step` is wrapped once more from outside, and every later boundary fires after
torch's annotation has closed.

Label hooks (forward / backward / tokens) are registered only for the profiled step. The model
the token counter watches is found by a one-shot hook on the first module call, during warmup. Any global
module hook disables `Module.__call__`'s no-hook fast path for EVERY module, so leaving them on
during the clean window would put our overhead into `step_time_s`.
"""
import os
import threading
import time

import torch
import torch.nn.modules.module as _module
from torch.autograd.profiler import record_function
from torch.optim.optimizer import register_optimizer_step_post_hook
from torch.profiler import ProfilerActivity, profile
from torch.utils.data.dataloader import _BaseDataLoaderIter

from vendor.capture import (Capture, _MIN_SAMPLE_S, _cpu_stats, _default_trace_path,
                              _read_memory, _sync, _tree_cpu_seconds, measure_peak_bandwidth_gb_s,
                              measure_peak_flops)


def _count_tokens(args, kwargs):
    """Tokens in one micro-batch, from the outermost module's input. None if it can't tell.

    Integer ids (B, S) count every element; a float activation (B, S, D) counts B*S. An
    `attention_mask` kwarg wins when present, since it excludes padding. The mask count is
    returned as a device TENSOR: `.item()` here would force a CUDA sync inside the profiled step.
    """
    mask = kwargs.get("attention_mask")
    if isinstance(mask, torch.Tensor):
        return mask.sum()
    x = kwargs.get("input_ids")
    if x is None:
        x = next((a for a in args if isinstance(a, torch.Tensor)), None)
    if not isinstance(x, torch.Tensor):
        return None
    if not x.is_floating_point():
        return x.numel()
    return x.shape[0] * x.shape[1] if x.dim() >= 3 else x.shape[0]


class _Labels:
    """forward / backward annotations + token and micro-batch counting, installed around one
    profiled step. `forwards` / `backwards` per optimizer step are the grad-accum count as it
    actually ran: torch has no notion of grad accumulation, only of N fwd/bwd between steps."""

    def __init__(self, root):
        self.depth, self.fwd, self.root, self.tokens = 0, None, root, 0
        self.forwards, self.backwards = 0, 0
        self.handles, self._orig_backward, self._orig_next = [], None, None

    def install(self):
        def pre(mod, args):
            self.depth += 1
            if self.depth == 1:                        # outermost module call only
                self.fwd = record_function("forward")
                self.fwd.__enter__()

        def post(mod, args, out):
            self.depth -= 1
            if self.depth == 0 and self.fwd is not None:
                self.fwd.__exit__(None, None, None)
                self.fwd = None

        def tok(mod, args, kwargs):                    # per-module hook: global ones get no kwargs
            self.forwards += 1
            self._tokens_from(mod, args, kwargs)

        self.handles += [_module.register_module_forward_pre_hook(pre),
                         _module.register_module_forward_hook(post)]
        if self.root is not None:
            self.handles.append(self.root.register_forward_pre_hook(tok, with_kwargs=True))

        orig = self._orig_backward = torch.autograd.backward

        def backward(*a, **k):                         # Tensor.backward resolves this at call time
            self.backwards += 1
            with record_function("backward"):
                return orig(*a, **k)
        torch.autograd.backward = backward

        # dataloader: torch already annotates `enumerate(DataLoader)#<Iter>.__next__`, but the
        # Observation's `data_wait_s` reads a label named `dataloader`. Emit that name too, over
        # the same call, so the diagnostic needs no alias. Class-level: every loader in the process.
        orig_next = self._orig_next = _BaseDataLoaderIter.__next__

        def next_(it):
            with record_function("dataloader"):
                return orig_next(it)
        _BaseDataLoaderIter.__next__ = next_

    def _tokens_from(self, mod, args, kwargs):
        n = _count_tokens(args, kwargs)
        if n is not None:
            self.tokens = self.tokens + n                          # int, or a device tensor

    def token_count(self):
        """Resolve the count. Called after the profiler stops, so any sync lands outside it."""
        t = self.tokens
        return int(t.item()) if isinstance(t, torch.Tensor) else int(t)

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []
        if self._orig_backward is not None:
            torch.autograd.backward = self._orig_backward
            self._orig_backward = None
        if self._orig_next is not None:
            _BaseDataLoaderIter.__next__ = self._orig_next
            self._orig_next = None


class Attach:
    def __init__(self, device="cpu", warmup=3, clean_steps=10, trace_path=None, dtype=None,
                 peak_flops=None, peak_mem_bw_gb_s=None):
        self.device, self.warmup, self.clean_steps = device, warmup, clean_steps
        self.trace_path = trace_path or _default_trace_path()
        dtype = dtype or torch.get_default_dtype()
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.peak_flops, self.peak_mem_bw_gb_s = peak_flops, peak_mem_bw_gb_s
        self.done, self.params, self.tokens = False, None, None
        self.micro_batches = None                      # {"forwards", "backwards"} in the profiled step
        self.on_done = None                            # callback(Capture): the launcher writes it out
        self._n, self._opt, self._cls, self._inner, self._hook = 0, None, None, None, None
        self._root, self._root_hook = None, None
        self._times, self._t_last, self._cpu = [], None, None
        self._labels, self._prof, self._step_rf, self._t_prof = None, None, None, None
        self._cap = None

    # ---- lifecycle -------------------------------------------------------------------------
    def install(self):
        """Measure the ceilings (while memory is still free) and start watching for a step."""
        if self.peak_flops is None:
            self.peak_flops = measure_peak_flops(self.device, self.dtype)
        if self.peak_mem_bw_gb_s is None:
            self.peak_mem_bw_gb_s = measure_peak_bandwidth_gb_s(self.device)
        self._hook = register_optimizer_step_post_hook(self._discover)
        self._root_hook = _module.register_module_forward_pre_hook(self._find_root)

    def _find_root(self, mod, args):
        """The first module called is outermost by construction: count tokens there. One-shot,
        during warmup, so the no-hook fast path is back before anything is timed."""
        self._root = mod
        self._root_hook.remove()
        self._root_hook = None

    def uninstall(self):
        for h in (self._hook, self._root_hook):
            if h is not None:
                h.remove()
        self._hook = self._root_hook = None
        if self._cls is not None:
            self._cls.step = self._inner
            self._cls = None
        if self._labels is not None:
            self._labels.remove()

    def result(self):
        return self._cap

    def tokens_per_step(self):
        return self.tokens

    # ---- step boundary ---------------------------------------------------------------------
    def _discover(self, opt, args, kwargs):
        """First step of the first optimizer: adopt it, and move the boundary outside torch's
        annotation for every later step."""
        self._hook.remove()
        self._hook = None
        self._opt, self._cls = opt, type(opt)
        self._inner = inner = self._cls.step
        self.params = sum(p.numel() for g in opt.param_groups for p in g["params"])

        def step(opt_self, *a, **k):
            out = inner(opt_self, *a, **k)
            if opt_self is self._opt and not self.done:
                self._on_step()
            return out
        step.hooked = True                             # stop torch re-wrapping it
        self._cls.step = step
        self._on_step()                                # step 1 is warmup; inside is harmless

    def _on_step(self):
        self._n += 1
        n, w, c = self._n, self.warmup, self.clean_steps
        if n < w:
            return
        if n > w + c:                                  # the profiled step ended: close `step`
            self._stop_profiled()                      # BEFORE syncing, or the GPU drain lands
            return                                     # in its host-dispatch time
        _sync(self.device)
        now = time.perf_counter()
        if n == w:                                     # warmup over: open the clean window
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            self._start_cpu()
            self._t_last = now
        elif n <= w + c:                               # one clean step ended
            self._times.append(now - self._t_last)
            self._t_last = now
            if n == w + c:
                self._stop_cpu()
                self._start_profiled()

    # ---- profiled step ---------------------------------------------------------------------
    def _start_profiled(self):
        activities = [ProfilerActivity.CPU]
        if self.device == "cuda":
            activities.append(ProfilerActivity.CUDA)
        self._labels = _Labels(self._root)
        self._labels.install()
        self._prof = profile(activities=activities)
        self._prof.__enter__()
        self._t_prof = time.perf_counter()
        self._step_rf = record_function("step")
        self._step_rf.__enter__()

    def _stop_profiled(self):
        self._step_rf.__exit__(None, None, None)
        _sync(self.device)
        profiled_s = time.perf_counter() - self._t_prof
        self._prof.__exit__(None, None, None)
        self._labels.remove()
        self.tokens = self._labels.token_count() or None
        self.micro_batches = {"forwards": self._labels.forwards,
                              "backwards": self._labels.backwards}
        self._prof.export_chrome_trace(self.trace_path)
        self._cap = Capture(status="ok", step_times_s=self._times, trace=self.trace_path,
                            profiled_step_s=profiled_s, peak_flops=self.peak_flops,
                            peak_dtype=str(self.dtype).replace("torch.", ""),
                            peak_mem_bw_gb_s=self.peak_mem_bw_gb_s,
                            memory=_read_memory(self.device), cpu=self._cpu, device=self.device)
        self.done = True
        self.uninstall()
        if self.on_done is not None:
            self.on_done(self._cap)

    # ---- CPU window (mirrors capture()'s sampler) ------------------------------------------
    def _start_cpu(self):
        self._samples, self._cpu_stop, ready = [], threading.Event(), threading.Event()

        def sampler():
            try:
                import psutil
            except Exception:
                ready.set()
                return
            psutil.cpu_percent()
            ready.set()
            last = time.perf_counter()
            while True:
                stopped = self._cpu_stop.wait(0.05)
                now = time.perf_counter()
                if now - last >= _MIN_SAMPLE_S:
                    self._samples.append(psutil.cpu_percent())
                    last = now
                if stopped:
                    return

        self._sampler = threading.Thread(target=sampler, daemon=True)
        self._sampler.start()
        ready.wait(timeout=1.0)
        self._cpu0, self._wall0 = _tree_cpu_seconds(), time.perf_counter()

    def _stop_cpu(self):
        busy = _tree_cpu_seconds() - self._cpu0
        window = time.perf_counter() - self._wall0
        self._cpu_stop.set()
        self._sampler.join(timeout=0.5)
        cpu = _cpu_stats(self._samples) or {"mean": None, "max": None, "nproc": os.cpu_count()}
        cpu["cores_busy"] = round(busy / window, 3) if window > 0 else None
        self._cpu = cpu
