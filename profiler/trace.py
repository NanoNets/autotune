"""Parse a torch.profiler Chrome trace (export_chrome_trace) — no torch needed to read it.

A Chrome trace is `{"traceEvents": [{cat, name, dur(us), ...}, ...]}`. We read:
- phase times: duration of the `record_function` annotations the runner wraps each phase in
- kernels: GPU kernel events aggregated by name (→ GPU busy time, top kernels)
- provenance: what actually ran, read off kernel names (flash attention, fused kernels, dtype)
Durations are microseconds, as emitted by the profiler.
"""
import bisect
import json

_KERNEL_CATS = {"kernel", "gpu_memcpy", "gpu_memset"}
_ANNOTATION_CATS = {"user_annotation"}


def load_trace(path_or_dict):
    """Accept a parsed dict or a path to the trace json."""
    if isinstance(path_or_dict, dict):
        return path_or_dict
    with open(path_or_dict) as f:
        return json.load(f)


def _events(trace):
    return trace.get("traceEvents", [])


def phase_times_us(trace, phases=None):
    """Sum annotation durations (us) per phase name.

    phases=None reports EVERY user annotation the runner emitted — a closed list would silently
    drop whatever it doesn't know about (comm, checkpoint, eval). Pass a list to restrict.
    """
    totals = {p: 0.0 for p in phases} if phases is not None else {}
    for e in _events(trace):
        if e.get("cat") not in _ANNOTATION_CATS:
            continue
        name = e.get("name")
        if phases is not None and name not in totals:
            continue
        totals[name] = totals.get(name, 0.0) + e.get("dur", 0.0)
    return totals


def kernels(trace, top=None):
    """GPU kernel events aggregated by name → [(name, total_us), ...] sorted desc."""
    agg = {}
    for e in _events(trace):
        if e.get("cat") in _KERNEL_CATS and "dur" in e:
            agg[e["name"]] = agg.get(e["name"], 0.0) + e["dur"]
    ranked = sorted(agg.items(), key=lambda kv: -kv[1])
    return ranked[:top] if top else ranked


def gpu_busy_us(trace):
    """Total GPU kernel time (us)."""
    return sum(dur for _, dur in kernels(trace))


def kernel_gaps(trace):
    """Idle shape on the GPU-kernel timeline: (gap_count, max_gap_us).

    One big gap (data-wait) and many small gaps (launch overhead) have the same TOTAL idle but
    look different here — that's the signal the aggregate would otherwise hide. Sort kernels by
    start, sum positive gaps between them (running max end handles async overlap → clamp to 0).
    First-cut single-timeline view; needs `ts` on kernel events (present on real GPU traces);
    (None, None) without them — unknown, not "gapless".
    """
    spans = sorted((e["ts"], e["ts"] + e.get("dur", 0.0))
                   for e in _events(trace)
                   if e.get("cat") in _KERNEL_CATS and "ts" in e and "dur" in e)
    if not spans:                                      # no timeline (CPU trace) -> unknown
        return None, None
    gaps, running_end = [], None
    for start, end in spans:
        if running_end is not None:
            if start - running_end > 0:
                gaps.append(start - running_end)
            running_end = max(running_end, end)
        else:
            running_end = end
    return len(gaps), (max(gaps) if gaps else 0.0)


# GEMM kernels encode precision in the name (sgemm=fp32, hgemm=fp16, dgemm=fp64) — without this
# an fp32 run reports no dtype at all, indistinguishable from a run we learned nothing about.
_GEMM_DTYPE = {"sgemm": "fp32", "hgemm": "fp16", "dgemm": "fp64"}


_LAUNCH_CATS = {"cuda_runtime", "cuda_driver"}


def gpu_phase_times_us(trace):
    """GPU time attributed to each annotation (us) — the fix for host-dispatch-vs-GPU-time.

    A `record_function` annotation measures HOST dispatch, and CUDA launches are async, so its
    duration says nothing about GPU work. Every kernel carries a `correlation` id matching the
    runtime launch that issued it, so a kernel belongs to whichever annotation window contains
    that launch. Scopes nest (`step` contains the rest), so these do not sum to the total.
    Returns ({name: gpu_us}, attributed_fraction_of_busy); ({}, None) with no correlation data.
    """
    corr = {}
    for e in _events(trace):
        if e.get("cat") in _KERNEL_CATS and "dur" in e:
            c = (e.get("args") or {}).get("correlation")
            if c is not None:
                corr[c] = corr.get(c, 0.0) + e["dur"]
    if not corr:
        return {}, None

    launches = sorted((e["ts"], corr[(e.get("args") or {}).get("correlation")])
                      for e in _events(trace)
                      if e.get("cat") in _LAUNCH_CATS and "ts" in e
                      and (e.get("args") or {}).get("correlation") in corr)
    starts = [ts for ts, _ in launches]

    totals, seen = {}, set()
    for e in _events(trace):
        if e.get("cat") not in _ANNOTATION_CATS or "ts" not in e or "dur" not in e:
            continue
        lo = bisect.bisect_left(starts, e["ts"])
        hi = bisect.bisect_right(starts, e["ts"] + e["dur"])
        totals[e["name"]] = totals.get(e["name"], 0.0) + sum(d for _, d in launches[lo:hi])
        seen.update(range(lo, hi))

    busy = sum(corr.values())
    covered = sum(launches[i][1] for i in seen)
    return totals, (round(covered / busy, 3) if busy else None)


def provenance(trace):
    """What actually ran, read off kernel names — the 'did it apply?' check.

    A HINT, not a verdict: substring matching on kernel names is lossy, so the raw names stay in
    `top_kernels` for the model to read. None (not False) when no kernels ran — absent evidence is
    not evidence of absence.
    """
    ks = kernels(trace)
    if not ks:
        return {"flash_attention": None, "fused_kernels": None, "dtypes": []}
    names = " ".join(name.lower() for name, _ in ks)
    dtypes = {d for d in ("bf16", "fp16", "tf32", "fp8") if d in names}
    dtypes |= {v for k, v in _GEMM_DTYPE.items() if k in names}
    return {
        "flash_attention": any(k in names for k in ("flash", "fmha", "mem_efficient")),
        # "cutlass" is a GEMM backend, not evidence of fusion — it made every plain matmul look fused
        "fused_kernels": any(k in names for k in ("fused", "liger", "multi_tensor")),
        "dtypes": sorted(dtypes),
    }


def device_props(trace):
    """Real hardware from the trace header — beats hardcoded specs. {} if absent (CPU trace)."""
    devs = trace.get("deviceProperties") or []
    if not devs:
        return {}
    d = devs[0]
    total = d.get("totalGlobalMem")
    return {
        "name": d.get("name"),
        "mem_gb": round(total / 1024 ** 3, 1) if total else None,
        "compute_capability": (f"{d['computeMajor']}.{d['computeMinor']}"
                               if "computeMajor" in d else None),
        "num_sms": d.get("numSms"),
    }


def occupancy(trace):
    """Runtime-weighted achieved occupancy across kernels + the dominant limiting factor.

    Reads 'est. achieved occupancy %' and occupancy.limitingFactors from the kernel args (present
    on real CUDA traces). This is the deep 'why is compute slow' signal — busy but under-fed, and
    *why* (REGS / shared mem / warps). {} if absent (CPU / no kernels).
    """
    tot_dur, weighted, limiters = 0.0, 0.0, {}
    for e in _events(trace):
        if e.get("cat") not in _KERNEL_CATS:
            continue
        a = e.get("args") or {}
        occ, dur = a.get("est. achieved occupancy %"), e.get("dur", 0.0)
        if occ is not None and dur:
            weighted += occ * dur
            tot_dur += dur
            lf = (a.get("occupancy") or {}).get("limitingFactors")
            if lf:
                limiters[lf] = limiters.get(lf, 0.0) + dur
    if not tot_dur:
        return {}
    return {
        "achieved_occupancy_pct": round(weighted / tot_dur, 1),
        "limiting_factor": max(limiters, key=limiters.get) if limiters else None,
    }
