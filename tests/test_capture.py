"""Capture tests — the frozen-ceiling contract a search depends on. Needs torch, CPU only."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiler.capture import capture


class _Runner:
    """Trivial runner: the capture engine is what's under test, not the workload."""
    def setup(self, config):
        self.params = 10

    def step(self):
        pass

    def tokens_per_step(self):
        return 1


def test_supplied_ceilings_are_used_verbatim_and_not_remeasured():
    """Frame freezes these off probe 1 and passes them to probes 2..N — they must survive intact.

    Passing deliberately absurd values: if capture ever re-measured, they would come back changed.
    """
    cap = capture(_Runner(), {"device": "cpu"}, warmup=1, clean_steps=2,
                  peak_flops=1.23e15, peak_mem_bw_gb_s=999.0)
    assert cap.status == "ok", cap.error
    assert cap.peak_flops == 1.23e15
    assert cap.peak_mem_bw_gb_s == 999.0


def test_frozen_ceiling_flows_into_the_observation():
    """The Observation is where frame reads them back off probe 1."""
    from profiler.diagnostic import diagnose
    cap = capture(_Runner(), {"device": "cpu"}, warmup=1, clean_steps=2,
                  peak_flops=1.0e15, peak_mem_bw_gb_s=500.0)
    o = diagnose(cap, params=10, tokens=1, gpu="whatever")
    assert o.peak_flops == 1.0e15 and o.peak_mem_bw_gb_s == 500.0


def test_cpu_signal_survives_a_window_too_short_to_sample():
    """psutil sampling quantizes to nothing under ~10ms; the counter delta still reports."""
    import time

    class Busy(_Runner):
        def step(self):
            t = time.perf_counter()
            while time.perf_counter() - t < 0.002:      # 2ms of real CPU work
                pass

    cap = capture(Busy(), {"device": "cpu"}, warmup=1, clean_steps=3,
                  peak_flops=1e12, peak_mem_bw_gb_s=1.0)
    assert cap.cpu["cores_busy"] > 0.5                  # a busy core is visible...
    assert cap.cpu["mean"] is None                      # ...even though sampling saw nothing


class _PassRunner(_Runner):
    """A runner that exposes the population hooks: passes, a reference batch, token counts."""
    steps_per_pass = 4
    has_reference = True

    def setup(self, config):
        super().setup(config)
        self.log, self.n = [], 0

    def step(self):
        self.n += 1
        self.log.append("step")

    def reference_step(self):
        self.log.append("ref")

    def start_pass(self):
        self.log.append("pass")

    def last_step_tokens(self):
        return 10 * self.n                               # every step a different width

    def last_step_samples(self):
        return 16

    def population(self):
        return {"sample_hash": "abc", "reference": {"batch_key": "k1"}}


def test_clean_window_is_whole_passes_and_the_trace_is_the_reference_batch():
    r = _PassRunner()
    cap = capture(r, {"device": "cpu"}, warmup=2, clean_steps=5,
                  peak_flops=1e12, peak_mem_bw_gb_s=1.0)
    assert cap.status == "ok", cap.error
    assert len(cap.step_times_s) == 8 and cap.clean_passes == 2.0   # 5 rounded up to 2 passes of 4
    assert cap.clean_steps_requested == 5
    # warmup, warm the reference as many times (compile, then capture), restart at a pass
    # boundary, window, then the reference twice: profiler off (its wall), then on (the trace)
    assert r.log == ["step"] * 2 + ["ref"] * 2 + ["pass"] + ["step"] * 8 + ["ref", "ref"]
    assert cap.step_tokens == [10 * n for n in range(3, 11)]   # clean steps only, not warmup
    assert cap.step_samples == [16] * 8
    assert cap.reference_step_s is not None and cap.profiled_step_s is not None
    assert cap.population["reference"]["batch_key"] == "k1"


def test_a_runner_without_a_reference_pays_no_extra_steps():
    """One fixed batch (torch fixture, synthetic ceiling): the clean window already times it."""
    class Counting(_Runner):
        def setup(self, config):
            super().setup(config)
            self.steps = 0

        def step(self):
            self.steps += 1

    r = Counting()
    cap = capture(r, {"device": "cpu"}, warmup=1, clean_steps=3,
                  peak_flops=1e12, peak_mem_bw_gb_s=1.0)
    assert r.steps == 1 + 3 + 1                         # warmup + window + the traced step only
    assert len(cap.step_times_s) == 3 and cap.clean_passes is None   # no rounding without passes
    assert cap.step_tokens == [] and cap.population is None
    assert abs(cap.reference_step_s - sum(cap.step_times_s) / 3) < 1e-12   # the window's mean


def test_an_oom_mid_window_reports_the_passes_completed_not_planned():
    """Review: an OOM after 3 of 8 steps must not claim 2 whole passes (same-rows evidence)."""
    import torch

    class OomAfter3(_PassRunner):
        def step(self):
            super().step()
            if self.log.count("pass") and self.n > 2 + 3:  # 2 warmup, then 3 clean steps
                raise torch.cuda.OutOfMemoryError("CUDA out of memory")

    cap = capture(OomAfter3(), {"device": "cpu"}, warmup=2, clean_steps=5,
                  peak_flops=1e12, peak_mem_bw_gb_s=1.0)
    assert cap.status == "oom"
    assert len(cap.step_times_s) == 3 and cap.clean_passes == 3 / 4   # 0.75, not 2.0


def test_rounding_up_to_whole_passes_is_announced():
    """Review: 10 clean steps silently becoming 250 is a probe-length surprise."""
    import contextlib
    import io
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        capture(_PassRunner(), {"device": "cpu"}, warmup=1, clean_steps=5,
                peak_flops=1e12, peak_mem_bw_gb_s=1.0)
    assert "clean_steps 5 -> 8" in err.getvalue()


class _FingerprintRunner(_PassRunner):
    def reference_fingerprint(self):
        return {"loss": 2.5, "grad_norm": 0.75}


def test_fingerprint_flows_from_runner_to_observation():
    """Taken after setup, before warmup; both halves land on the Observation and its summary."""
    from profiler.diagnostic import diagnose
    cap = capture(_FingerprintRunner(), {"device": "cpu"}, warmup=1, clean_steps=2,
                  peak_flops=1e12, peak_mem_bw_gb_s=1.0)
    assert cap.fingerprint == {"loss": 2.5, "grad_norm": 0.75}
    o = diagnose(cap, params=10, tokens=1, gpu="x")
    assert o.reference_loss == 2.5 and o.reference_grad_norm == 0.75
    assert "ref-loss 2.5000 grad 0.75" in o.summary


def test_a_runner_without_a_fingerprint_reports_none():
    """Optional: the torch fixture and synthetic ceiling trials have no reference rows."""
    from profiler.diagnostic import diagnose
    cap = capture(_Runner(), {"device": "cpu"}, warmup=1, clean_steps=2,
                  peak_flops=1e12, peak_mem_bw_gb_s=1.0)
    assert cap.fingerprint is None
    o = diagnose(cap, params=10, tokens=1, gpu="x")
    assert o.reference_loss is None and o.reference_grad_norm is None
    assert "ref-loss" not in o.summary


if __name__ == "__main__":
    for _n, _f in list(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f(); print(f"ok  {_n}")
    print("all passed")
