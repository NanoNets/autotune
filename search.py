"""Search harness + a baseline deterministic policy.

search(probe_fn, policy) is dumb plumbing: ask the policy for a config, probe it, record,
repeat until the policy stops; return the best (highest samples_per_s) config seen. Both the
PROBE and the POLICY are injected, so the same harness runs fake_probe or modal_probe, and a
grid-walker or (later) a smarter/LLM policy. The intelligence lives in the policy.

SweepPolicy is the baseline (port of program.md): double bs until OOM or saturation, break an
OOM with padding_free then gradient_checkpointing. It's the measuring stick a smart policy
must beat (fewer probes to the same optimum).
"""


class SweepPolicy:
    """Baseline: bs-doubling sweep + OOM extension protocol, for one effective_batch."""

    def __init__(self, effective_batch, sat_threshold=0.03):
        self.eff = effective_batch
        self.sat_threshold = sat_threshold
        self.bs = 1
        self.pf = False
        self.gc = False
        self.sat_count = 0
        self.prev_sps = None
        self.pending_ext = None   # "pf"/"gc" while awaiting an extension attempt's result

    def _config(self):
        return {"per_device_train_batch_size": self.bs,
                "gradient_accumulation_steps": self.eff // self.bs,
                "padding_free": self.pf,
                "gradient_checkpointing": self.gc}

    def propose(self, history):
        if not history:
            return self._config()
        last = history[-1]["result"]

        if self.pending_ext is not None:          # awaiting an extension attempt's result
            if last["status"] == "ok":
                self.pending_ext = None
                return self._after_ok(last)        # extension broke the OOM -> continue sweep
            if self.pending_ext == "pf" and not self.gc:
                self.gc = True
                self.pending_ext = "gc"
                return self._config()              # pf didn't help -> try gc
            return None                            # gc also failed -> OOM ceiling is final

        if last["status"] == "oom":                # start the extension protocol
            if not self.pf:
                self.pf = True
                self.pending_ext = "pf"
                return self._config()
            if not self.gc:
                self.gc = True
                self.pending_ext = "gc"
                return self._config()
            return None
        return self._after_ok(last)

    def _after_ok(self, last):
        sps = last["samples_per_s"]
        if self.prev_sps is not None:
            improved = (sps - self.prev_sps) / self.prev_sps
            self.sat_count = self.sat_count + 1 if improved < self.sat_threshold else 0
        self.prev_sps = sps
        if self.sat_count >= 2 or self.bs >= self.eff:   # saturated, or bs==eff (ga=1 ceiling)
            return None
        self.bs *= 2
        return self._config()


class PredictivePolicy:
    """Predict the OOM ceiling from peak_mem, jump near it, confirm/bisect, escalate extensions;
    never hard-skip (every boundary is probed). Drop-in for SweepPolicy (same propose interface).

    Per flag set: anchor the linear memory model (peak ~= base + slope*bs), jump to the predicted
    max-safe bs, confirm by probing one above; bisect only if the prediction missed. Escalate to
    padding_free then gradient_checkpointing while the ceiling sits below eff. The harness picks the
    highest-throughput probe, so this only has to explore the right configs and stop. UNTESTED.
    """

    FLAGS = [(False, False), (True, False), (True, True)]  # (padding_free, gradient_checkpointing)

    def __init__(self, effective_batch, anchors=(1, 2)):
        self.eff = effective_batch
        self.anchors = list(anchors)
        self.ai = 0
        self.gpu_mem = None
        self.base = None
        self.slope_f = None          # memory slope (GB/sample) for the current flag set
        self.fi = 0                  # flag-set index
        self.lo = None               # highest confirmed-fit bs (current flag set)
        self.hi = None               # lowest confirmed-OOM bs (current flag set)
        self.stage = "anchor"

    def _cfg(self, bs):
        pf, gc = self.FLAGS[self.fi]
        return {"per_device_train_batch_size": bs,
                "gradient_accumulation_steps": max(1, self.eff // bs),
                "padding_free": pf, "gradient_checkpointing": gc}

    def _predict(self):
        ceil = int((self.gpu_mem - self.base) / self.slope_f)   # largest bs whose peak fits
        return max(1, min(ceil, self.eff))                      # cap at eff (bs can't exceed it)

    def _finish_flagset(self, max_safe):
        if max_safe >= self.eff or self.fi == len(self.FLAGS) - 1:
            return None                                # reached the eff cap, or out of extensions
        self.fi += 1                                   # escalate: pf, then pf+gc
        self.lo = self.hi = self.slope_f = None
        self.stage = "enter_flag"
        return self._cfg(max_safe)                     # re-probe max_safe with new flags -> learn its slope

    def propose(self, history):
        last = history[-1]["result"] if history else None
        last_bs = history[-1]["config"]["per_device_train_batch_size"] if history else None

        if self.stage == "anchor":
            if self.ai < len(self.anchors):
                bs = self.anchors[self.ai]
                self.ai += 1
                return self._cfg(bs)
            (b1, p1), (b2, p2) = [(h["config"]["per_device_train_batch_size"],
                                   h["result"]["peak_mem_gb"]) for h in history[-2:]]
            self.gpu_mem = last["gpu_mem_gb"]
            self.slope_f = (p2 - p1) / (b2 - b1)       # base + slope memory model
            self.base = p1 - self.slope_f * b1
            self.lo = b2
            self.stage = "predict"
            return self.propose(history)

        if self.stage == "enter_flag":
            self.slope_f = (last["peak_mem_gb"] - self.base) / last_bs   # this flag set's slope
            self.lo = last_bs
            self.stage = "predict"
            return self.propose(history)

        if self.stage == "predict":
            self.stage = "check"
            return self._cfg(self._predict())

        if self.stage == "check":
            if last["status"] == "oom":
                self.hi = last_bs
                self.stage = "bisect"
                return self.propose(history)
            self.lo = last_bs
            if last_bs >= self.eff:
                return self._finish_flagset(self.eff)
            self.stage = "confirm"
            return self._cfg(last_bs + 1)              # probe one above to confirm the ceiling

        if self.stage == "confirm":
            if last["status"] == "oom":
                return self._finish_flagset(self.lo)   # confirmed: max-safe = lo
            self.lo = last_bs                          # under-predicted -> climb
            if last_bs >= self.eff:
                return self._finish_flagset(self.eff)
            self.slope_f = (last["peak_mem_gb"] - self.base) / last_bs   # refine, re-predict
            nxt = self._predict()
            self.stage = "check" if nxt > last_bs else "confirm"
            return self._cfg(nxt if nxt > last_bs else last_bs + 1)

        if self.stage == "bisect":
            if last["status"] == "oom":
                self.hi = last_bs
            else:
                self.lo = last_bs
            if self.hi - self.lo <= 1:
                return self._finish_flagset(self.lo)
            return self._cfg((self.lo + self.hi) // 2)

        return None


def search(probe_fn, policy, base_config=None):
    """Run policy<->probe until the policy stops; return the best ok config + full trace."""
    base_config = base_config or {}
    history = []
    while True:
        proposal = policy.propose(history)
        if proposal is None:
            break
        result = probe_fn({**base_config, **proposal})
        history.append({"config": proposal, "result": result})
    oks = [h for h in history if h["result"]["status"] == "ok"]
    best = max(oks, key=lambda h: h["result"]["samples_per_s"], default=None)
    return {"best": best, "n_probes": len(history), "history": history}
