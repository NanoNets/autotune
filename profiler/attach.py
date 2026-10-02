"""Bring-your-own-script capture: profile a training loop autotune does not own.

`capture()` drives a runner: it calls `step()` itself, and the runner wraps its phases in
`record_function`. A user's script (swift, TRL, a hand-written loop) offers neither. `Attach`
inverts control: the user's loop runs as-is (started by `profiler/launch.py`), and it produces the
same `Capture` that `capture()` does, so `diagnose()` is unchanged. Design and evidence:
`bring_your_own_script.md`; what it does and doesn't measure yet: `integration_plan.md` Appendix A.

    att = Attach(device="cuda", peak_flops=..., peak_mem_bw_gb_s=...)
    att.install()              # before the user builds the optimizer
    ... user's loop runs ...   # att.done flips after warmup + clean + 1 profiled step
    cap = att.result()

NO MONKEYPATCHING. Only public APIs, used as designed:
  - `register_optimizer_step_post_hook`  step boundary (the FIRST optimizer seen) + its params
  - `torch.cuda.synchronize` + a clock  clean step time
  - `torch.cuda` memory stats, psutil    memory, CPU
  - `model.register_forward_pre_hook(with_kwargs=True)` on the user's model: TOKENS PER CLEAN
    STEP (decision B1, 2026-09-27). It counts padded `input_ids.numel()` (shape metadata, no GPU
    sync), the same definition as OG's `frameworks/hf.py`. A hook on one module only; the rest of
    the model keeps torch's no-hook fast path.
  - `register_module_forward_pre_hook` (global), ONE-SHOT during warmup: finds the model and
    attaches the per-module hook above, then removes itself before any timed step. "The model" is
    the first module called AFTER the first optimizer step whose parameters include the ones that
    optimizer trains. That rules out modules that run first but aren't the model being trained: an
    `nn.Module` transform in the data pipeline (no parameters), or a frozen reference / teacher
    model (none of the trained ones). Hence warmup >= 2: step 1 identifies the trained parameters,
    and step 2 finds the module that owns them.
  - `torch.profiler.profile(record_shapes=True)` + one `record_function("step")` of our own,
    for the one profiled step.
Phases and the micro-batch count are recovered from the trace torch recorded (`derive.py`). The
trace's token count for the traced step is kept only as a cross-check of the hook's.

Why tokens are counted on every clean step (B1): a rate's numerator and denominator must describe
the same steps (the Measurement axiom, `loop/contracts.md`). Under variable batch width, one
traced step's tokens over the clean window's time would be off by (traced width / mean width),
silently, and differently for every config.

The boundary fires inside torch's own `Optimizer.step#...` label (that's where the post-hook
runs), so the profiled window opens and closes inside one. `derive.py` only reads spans fully
inside our `step` label.
"""
import json
import os
import threading
import time

import torch
import torch.nn.modules.module as _module
from torch.autograd.profiler import record_function
from torch.optim.optimizer import register_optimizer_step_post_hook
from torch.profiler import ProfilerActivity, profile

from profiler import derive
from profiler.capture import (Capture, _MIN_SAMPLE_S, _cpu_stats, _default_trace_path,
                              _read_memory, _sync, _tree_cpu_seconds, measure_peak_bandwidth_gb_s,
                              measure_peak_flops)

# What every attached capture records about itself, so the loop can tell it apart from a runner
# probe (Measurement axiom, `loop/contracts.md`, corollary 5). The clean window is NOT yet whole
# passes over Frame's sample, and the trace is one step of the script's own loop, not a replayed
# reference batch, so throughput and trace durations from these probes are not `same-rows` /
# `same-reference-batch` evidence (integration_plan.md Appendix A: B2, B4).
POPULATION = {"driver": "script", "clean_window": "fixed-position", "trace_basis": "traced-step"}
MIN_WARMUP = 2      # step 1 identifies the trained parameters; step 2 finds the model that owns them


def _padded_tokens(args, kwargs):
    """Tokens in one micro-batch as the model receives it: padded, like OG (`input_ids.numel()`).

    Reads shapes only, so it never syncs the GPU. Integer ids (B, S) count every element; a float
    activation (B, S, D) counts B*S. None when the first input isn't a tensor.
    """
    x = kwargs.get("input_ids")
    if x is None:
        x = next((a for a in args if isinstance(a, torch.Tensor)), None)
    if not isinstance(x, torch.Tensor):
        return None
    if not x.is_floating_point():
        return x.numel()
    return x.shape[0] * x.shape[1] if x.dim() >= 3 else x.shape[0]


class Attach:
    def __init__(self, device="cpu", warmup=3, clean_steps=10, trace_path=None, dtype=None,
                 peak_flops=None, peak_mem_bw_gb_s=None):
        if warmup < MIN_WARMUP:
            raise ValueError(f"warmup={warmup}: needs >= {MIN_WARMUP} (step 1 identifies the trained "
                             "parameters, step 2 finds the model that owns them)")
        if clean_steps < 1:
            raise ValueError(f"clean_steps={clean_steps}: needs >= 1")
        self.device, self.warmup, self.clean_steps = device, warmup, clean_steps
        self.trace_path = trace_path or _default_trace_path()
        dtype = dtype or torch.get_default_dtype()
        self.dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
        self.peak_flops, self.peak_mem_bw_gb_s = peak_flops, peak_mem_bw_gb_s
        self.done, self.params, self.tokens = False, None, None
        self.micro_batches = None                      # what derive.py found in the profiled step
        self.on_done = None                            # callback(Capture): the launcher writes it out
        self._n, self._opt, self._hook = 0, None, None
        self._times, self._t_last, self._cpu = [], None, None
        self._prof, self._step_rf, self._t_prof, self._cap = None, None, None, None
        self._profiling = False                        # True between profiler enter and exit
        self.step_tokens = []                          # padded tokens of each CLEAN step (B1)
        self._acc, self._traced_tokens = 0, None       # running count within the current step
        self._root, self._root_hook, self._tok_hook = None, None, None
        self._trained = set()                          # ids of the parameters the optimizer trains
        self._step_bad = False                         # a call in this step had no token tensor
        self._tokens_unknown = None                    # why tokens can't be reported, if they can't
        self.model_trained = None                      # the model's checkpoint name, when it has one

    # ---- lifecycle -------------------------------------------------------------------------
    def install(self):
        """Measure the ceilings (while memory is still free) and start watching for a step."""
        if self.peak_flops is None:
            self.peak_flops = measure_peak_flops(self.device, self.dtype)
        if self.peak_mem_bw_gb_s is None:
            self.peak_mem_bw_gb_s = measure_peak_bandwidth_gb_s(self.device)
        self._hook = register_optimizer_step_post_hook(self._on_optimizer_step)

    def _find_root(self, mod, args):
        """The model: the first module called that owns parameters the optimizer trains. In a
        forward pass the outermost call comes first, so that's the whole model, not a layer. A
        module that owns none of them (a transform, a frozen reference or teacher model) is passed
        over. One-shot: attach the per-step token counter, then remove this global hook, so the
        no-hook fast path is back for every module before anything is timed. (The call in progress
        isn't counted: it's warmup.)"""
        if not any(id(p) in self._trained for p in mod.parameters()):
            return
        self._root = mod
        self._tok_hook = mod.register_forward_pre_hook(self._count, with_kwargs=True)
        self._root_hook.remove()
        self._root_hook = None
        cfg = getattr(mod, "config", None)              # HF models (and peft wrappers) carry the
        self.model_trained = getattr(cfg, "_name_or_path", None) or None   # checkpoint they came from

    def _count(self, mod, args, kwargs):
        n = _padded_tokens(args, kwargs)
        if n is None:                                   # no token tensor: this step's count is
            self._step_bad = True                       # unknown, not a smaller number
        else:
            self._acc += n

    def uninstall(self):
        """Leave the process as it was: hook removed, and a profiler still running (the run ended
        inside the profiled step) stopped rather than leaked."""
        for name in ("_hook", "_root_hook", "_tok_hook"):
            h = getattr(self, name)
            if h is not None:
                h.remove()
                setattr(self, name, None)
        self._stop_profiler_quietly()

    def _stop_profiler_quietly(self):
        if not self._profiling:
            return
        self._profiling = False
        for obj in (self._step_rf, self._prof):
            try:
                obj.__exit__(None, None, None)
            except Exception:
                pass

    def result(self):
        return self._cap

    def failed(self, status, error):
        """The Capture for a run that ended before the profiled step finished.

        Like `capture()` on an OOM: clean steps that already ran are kept (evidence the config ran
        that far), memory is read at the point of failure, and nothing is scored off it.
        """
        self._stop_profiler_quietly()                     # died inside the profiled step
        cpu = self._cpu
        if cpu is None and getattr(self, "_cpu_stop", None) is not None:
            try:                                           # died inside the clean window
                self._stop_cpu()
                cpu = self._cpu
            except Exception:
                pass
        self.uninstall()
        return Capture(status=status, step_times_s=list(self._times), memory=_read_memory(self.device),
                       step_tokens=list(self.step_tokens), clean_steps_requested=self.clean_steps,
                       population={**POPULATION, "model_trained": self.model_trained},
                       cpu=cpu, peak_flops=self.peak_flops,
                       peak_dtype=str(self.dtype).replace("torch.", ""),
                       peak_mem_bw_gb_s=self.peak_mem_bw_gb_s, device=self.device, error=error)

    def tokens_per_step(self):
        """Mean padded tokens per clean step: the same steps the clean window timed (B1)."""
        return self.tokens

    # ---- step boundary ---------------------------------------------------------------------
    def _on_optimizer_step(self, opt, args, kwargs):
        if self._opt is None:                          # adopt the first optimizer; ignore others
            self._opt = opt
            self.params = sum(p.numel() for g in opt.param_groups for p in g["params"])
            self._trained = {id(p) for g in opt.param_groups for p in g["params"]}
            self._root_hook = _module.register_module_forward_pre_hook(self._find_root)
        if opt is self._opt and not self.done:
            self._on_step()

    def _on_step(self):
        self._n += 1
        n, w, c = self._n, self.warmup, self.clean_steps
        step_tok, self._acc = self._acc, 0             # tokens of the step that just ended
        step_bad, self._step_bad = self._step_bad, False
        if n < w:
            return
        if n > w + c:                                  # the profiled step ended: close `step`
            self._traced_tokens = step_tok
            self._stop_profiled()                      # BEFORE syncing, or the GPU drain lands
            return                                     # in its host-dispatch time
        _sync(self.device)
        now = time.perf_counter()
        if n == w:                                     # warmup over: open the clean window
            if self.device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            self._start_cpu()
            self._t_last = now
        else:                                          # one clean step ended
            self._times.append(now - self._t_last)
            self._t_last = now
            if self._tok_hook is None:                 # same step as the time just recorded
                self._tokens_unknown = self._tokens_unknown or "no module owning the trained parameters was called"
            elif step_bad or step_tok == 0:
                self._tokens_unknown = self._tokens_unknown or "the model's input carried no token tensor"
            else:
                self.step_tokens.append(step_tok)
            if n == w + c:
                self._stop_cpu()
                self._start_profiled()

    # ---- profiled step ---------------------------------------------------------------------
    def _start_profiled(self):
        activities = [ProfilerActivity.CPU]
        if self.device == "cuda":
            activities.append(ProfilerActivity.CUDA)
        self._prof = profile(activities=activities, record_shapes=True)
        self._prof.__enter__()
        self._profiling = True
        self._t_prof = time.perf_counter()
        self._step_rf = record_function("step")
        self._step_rf.__enter__()

    def _stop_profiled(self):
        self._step_rf.__exit__(None, None, None)
        _sync(self.device)
        profiled_s = time.perf_counter() - self._t_prof
        self._prof.__exit__(None, None, None)
        self._profiling = False
        self._prof.export_chrome_trace(self.trace_path)

        with open(self.trace_path) as f:               # recover phases/tokens from the trace
            trace = json.load(f)
        found = derive.augment(trace)
        with open(self.trace_path, "w") as f:
            json.dump(trace, f)
        if self._tokens_unknown or not self.step_tokens:   # say so; never a guessed or partial count
            self.step_tokens, self.tokens = [], None
            found["token_source"] = f"unknown: {self._tokens_unknown or 'no per-step count'}"
        else:                                          # B1: the timed steps' own tokens
            self.tokens = sum(self.step_tokens) / len(self.step_tokens)
            found["token_source"] = f"hook: mean of {len(self.step_tokens)} clean steps (padded)"
        found["traced_step_tokens"] = {"hook": self._traced_tokens, "trace": found.get("tokens")}
        found["tokens"] = self.tokens
        self.micro_batches = found

        self._cap = Capture(status="ok", step_times_s=self._times, trace=self.trace_path,
                            step_tokens=list(self.step_tokens),
                            clean_steps_requested=self.clean_steps,
                            profiled_step_s=profiled_s,
                            population={**POPULATION, "model_trained": self.model_trained},
                            peak_flops=self.peak_flops,
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
