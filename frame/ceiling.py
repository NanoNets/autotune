"""Measure the OOM ceiling: the largest micro_batch that fits at a given sequence length.

`search(fits)` is pure — the exponential-then-bisect ladder with the fit test injected, so the
ladder is testable without a GPU. `find_ceiling(...)` is the live shell: it drives real probes on
worst-case synthetic batches (`frame.synthetic`) and returns the record the Profile carries.

Why measure it instead of discovering it: it turns `micro_batch` from an unbounded search
dimension into a bounded one BEFORE the loop spends a 5-minute probe on a config that was never
going to run. ~2*log2(n) trials of a few seconds each — no dataset, no tokenizer, no dataloader.

The ceiling is NOT a run constant. Every memory-relevant knob moves it (recompute, attn_kernel,
optimizer, sharding, dtype), so the record carries the config it was measured under: comparing
that to the current config is how a stale ceiling is detected. Re-measure, don't extrapolate.

Recorded, not hidden: fixed-shape probing avoids the allocator fragmentation that varying real
lengths cause, so `frag_gb` understates a long run's drift. `card_frac` is reported so a margin
can be kept below the ceiling — how much margin is the agent's call, not a constant in here.
"""

# The knobs that move the ceiling. Whichever are present are stamped on the record, so a later
# config can be compared against it and the ceiling retired when one of them has changed.
_MEMORY_KNOBS = ("dtype", "optimizer", "recompute", "attn_kernel", "shard_degree",
                 "offload", "packing")


def search(fits, cap=None):
    """Largest n in [1, cap] with `fits(n)` True. Returns 0 if even n=1 does not fit.

    Assumes `fits` is monotone (if n fits, n-1 fits) — true of batch size against memory.
    Exponential up to bracket the boundary, then bisect: ~2*log2(n) calls instead of n.
    cap: largest n worth testing; when everything up to it fits, cap is returned.
    """
    if cap is not None and cap < 1:
        raise ValueError(f"cap must be >= 1, got {cap!r}")
    if not fits(1):
        return 0

    lo, hi = 1, None                                   # lo: known to fit. hi: known to OOM.
    while hi is None:
        nxt = lo * 2
        if cap is not None and nxt > cap:              # the doubling would overshoot the cap
            if lo >= cap:
                return lo
            if fits(cap):                              # everything up to the cap fits
                return cap
            hi = cap
        elif fits(nxt):
            lo = nxt
        else:
            hi = nxt

    while hi - lo > 1:                                 # bisect the bracketed boundary
        mid = (lo + hi) // 2
        if fits(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _release(runner, device):
    """Drop a trial's allocations — the next trial must start from a clean card, or the ladder
    measures leftovers instead of the config."""
    import gc

    import torch
    if hasattr(runner, "teardown"):
        runner.teardown()
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def find_ceiling(runner, config, seq_len, vocab, cap=None, seed=0, warmup=2, clean_steps=2,
                 peak_flops=None, peak_mem_bw_gb_s=None, on_trial=None):
    """Largest micro_batch that fits at `seq_len`, measured on worst-case synthetic batches.

    runner: a `setup(config)` / `step()` runner — the REAL training step, or the ceiling describes
      something other than the job. An optional `teardown()` is called between trials.
    config: the runner's base config (include whatever the runner needs to build itself — e.g.
      `model` for `frameworks.hf`). `data="synthetic"`, `micro_batch`, `effective_batch`,
      `max_len`, `vocab` and `seed` are injected per trial; everything else passes through.
    cap: largest micro_batch worth testing — pass `effective_batch` (above it grad_accum < 1).
    peak_flops / peak_mem_bw_gb_s: Frame's frozen roofline ceilings, threaded through so no trial
      re-measures them.
    on_trial: called with each trial record as it lands (progress output).

    warmup >= 2 on purpose: AdamW allocates its state lazily on the first `opt.step()`, so a
    single-step peak omits optimizer state and reports a ceiling that is too high.
    """
    from profiler.capture import capture

    device = config.get("device", "cpu")
    trials = []

    def fits(n):
        # effective_batch = n makes grad_accum 1. Peak memory is unchanged by accumulation —
        # each micro-batch's activations are freed at its backward, and the .grad buffers are
        # full-size either way — so a 16-way accumulation would cost 16x the time for exactly
        # the same answer.
        cfg = {**config, "data": "synthetic", "micro_batch": n, "effective_batch": n,
               "max_len": seq_len, "vocab": vocab, "seed": seed}
        cap_ = capture(runner, cfg, warmup=warmup, clean_steps=clean_steps,
                       peak_flops=peak_flops, peak_mem_bw_gb_s=peak_mem_bw_gb_s)
        mem = cap_.memory or {}
        trial = {"micro_batch": n, "status": cap_.status, "peak_gb": mem.get("peak_gb"),
                 "reserved_gb": mem.get("reserved_gb"), "total_gb": mem.get("total_gb"),
                 "error": cap_.error}
        trials.append(trial)
        if on_trial:
            on_trial(trial)
        _release(runner, device)
        if cap_.status == "error":                     # a crash is not a memory answer — never
            raise RuntimeError(                        # let it be read as one
                f"ceiling trial micro_batch={n} failed with an error, not an OOM: {cap_.error}")
        return cap_.status == "ok"

    best = search(fits, cap=cap)

    ooms = [t["micro_batch"] for t in trials if t["status"] == "oom"]
    at_best = next((t for t in reversed(trials)
                    if t["micro_batch"] == best and t["status"] == "ok"), None) or {}
    peak, reserved, total = (at_best.get("peak_gb"), at_best.get("reserved_gb"),
                             at_best.get("total_gb"))
    return {
        "seq_len": seq_len,
        "micro_batch_max": best,
        "fits": best > 0,                              # False => memory regime, not a batch problem
        "capped": cap is not None and best == cap,     # True => the real ceiling is >= this
        "oom_at": min(ooms) if ooms else None,
        "peak_gb": peak,
        "reserved_gb": reserved,
        "frag_gb": round(reserved - peak, 3) if peak is not None and reserved is not None else None,
        "card_frac": round(reserved / total, 3) if reserved and total else None,
        "config": {k: config[k] for k in _MEMORY_KNOBS if k in config},
        "method": "synthetic",
        "seed": seed,
        "trials": trials,
    }


def is_stale(ceiling, config):
    """True when `config` has moved a knob the ceiling was measured under — re-measure, don't
    extrapolate. Compares only the knobs stamped on the record."""
    if not ceiling:
        return True
    measured = ceiling.get("config") or {}
    return any(config.get(k) != v for k, v in measured.items())
