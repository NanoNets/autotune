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

    def population(self):
        return {"sample_hash": "abc", "reference": {"batch_key": "k1"}}


def test_clean_window_is_whole_passes_and_the_trace_is_the_reference_batch():
    r = _PassRunner()
    cap = capture(r, {"device": "cpu"}, warmup=2, clean_steps=5,
                  peak_flops=1e12, peak_mem_bw_gb_s=1.0)
    assert cap.status == "ok", cap.error
    assert len(cap.step_times_s) == 8 and cap.clean_passes == 2.0   # 5 rounded up to 2 passes of 4
    # warmup, warm the reference, restart at a pass boundary, window, then the reference twice:
    # once profiler-off (its wall), once profiler-on (the trace) — never a loader batch
    assert r.log == ["step"] * 2 + ["ref", "pass"] + ["step"] * 8 + ["ref", "ref"]
    assert cap.step_tokens == [10 * n for n in range(3, 11)]   # clean steps only, not warmup
    assert cap.reference_step_s is not None and cap.profiled_step_s is not None
    assert cap.population["reference"]["batch_key"] == "k1"


def test_a_runner_without_hooks_is_its_own_reference():
    """The torch fixture and synthetic ceiling trials reuse one batch: step() is the reference."""
    cap = capture(_Runner(), {"device": "cpu"}, warmup=1, clean_steps=3,
                  peak_flops=1e12, peak_mem_bw_gb_s=1.0)
    assert len(cap.step_times_s) == 3 and cap.clean_passes is None   # no rounding without passes
    assert cap.step_tokens == [] and cap.population is None
    assert cap.reference_step_s is not None


if __name__ == "__main__":
    for _n, _f in list(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f(); print(f"ok  {_n}")
    print("all passed")
