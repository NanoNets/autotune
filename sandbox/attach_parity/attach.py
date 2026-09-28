"""SANDBOX — patch-free capture: profile a training loop autotune does not own.

`capture()` drives a runner: it calls `step()` itself, and the runner wraps its phases in
`record_function`. A user's script (swift, TRL, a hand-written loop) offers neither. `Attach`
inverts control: the user's loop runs as-is, and it produces the same `Capture` that
`capture()` does, so `diagnose()` is unchanged. See `bring_your_own_script.md` §2.

    att = Attach(device="cuda", peak_flops=..., peak_mem_bw_gb_s=...)
    att.install()              # before the user builds the optimizer
    ... user's loop runs ...   # att.done flips after warmup + clean + 1 profiled step
    cap = att.result()

NO MONKEYPATCHING, and no module hooks. Only public APIs, used as designed:
  - `register_optimizer_step_post_hook`  step boundary (the FIRST optimizer seen) + its params
  - `torch.cuda.synchronize` + a clock  clean step time
  - `torch.cuda` memory stats, psutil    memory, CPU
  - `torch.profiler.profile(record_shapes=True)` + one `record_function("step")` of our own,
    for the one profiled step
Phases, tokens and the micro-batch count are then recovered from the trace torch recorded
(`derive.py`), not from code we inject into the user's run.

The boundary fires inside torch's own `Optimizer.step#...` label (that's where the post-hook
runs), so the profiled window opens and closes inside one. `derive.py` only reads spans fully
inside our `step` label.
"""
import json
import os
import threading
import time

import torch
from torch.autograd.profiler import record_function
from torch.optim.optimizer import register_optimizer_step_post_hook
from torch.profiler import ProfilerActivity, profile

import derive
from vendor.capture import (Capture, _MIN_SAMPLE_S, _cpu_stats, _default_trace_path,
                            _read_memory, _sync, _tree_cpu_seconds, measure_peak_bandwidth_gb_s,
                            measure_peak_flops)


class Attach:
    def __init__(self, device="cpu", warmup=3, clean_steps=10, trace_path=None, dtype=None,
                 peak_flops=None, peak_mem_bw_gb_s=None):
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

    # ---- lifecycle -------------------------------------------------------------------------
    def install(self):
        """Measure the ceilings (while memory is still free) and start watching for a step."""
        if self.peak_flops is None:
            self.peak_flops = measure_peak_flops(self.device, self.dtype)
        if self.peak_mem_bw_gb_s is None:
            self.peak_mem_bw_gb_s = measure_peak_bandwidth_gb_s(self.device)
        self._hook = register_optimizer_step_post_hook(self._on_optimizer_step)

    def uninstall(self):
        """Leave the process as it was: hook removed, and a profiler still running (the run ended
        inside the profiled step) stopped rather than leaked."""
        if self._hook is not None:
            self._hook.remove()
            self._hook = None
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
                       cpu=cpu, peak_flops=self.peak_flops,
                       peak_dtype=str(self.dtype).replace("torch.", ""),
                       peak_mem_bw_gb_s=self.peak_mem_bw_gb_s, device=self.device, error=error)

    def tokens_per_step(self):
        return self.tokens

    # ---- step boundary ---------------------------------------------------------------------
    def _on_optimizer_step(self, opt, args, kwargs):
        if self._opt is None:                          # adopt the first optimizer; ignore others
            self._opt = opt
            self.params = sum(p.numel() for g in opt.param_groups for p in g["params"])
        if opt is self._opt and not self.done:
            self._on_step()

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
        else:                                          # one clean step ended
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
        self.tokens = found["tokens"]
        self.micro_batches = found

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
