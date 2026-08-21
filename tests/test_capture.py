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


if __name__ == "__main__":
    for _n, _f in list(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f(); print(f"ok  {_n}")
    print("all passed")
