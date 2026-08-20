"""Parse a torch.profiler Chrome trace (export_chrome_trace) — no torch needed to read it.

A Chrome trace is `{"traceEvents": [{cat, name, dur(us), ...}, ...]}`. We read:
- phase times: duration of the `record_function` annotations the runner wraps each phase in
- kernels: GPU kernel events aggregated by name (→ GPU busy time, top kernels)
- provenance: what actually ran, read off kernel names (flash attention, fused kernels, dtype)
Durations are microseconds, as emitted by the profiler.
"""
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


def phase_times_us(trace, phases):
    """Sum annotation durations (us) per phase name (e.g. forward/backward/optimizer/dataloader)."""
    totals = {p: 0.0 for p in phases}
    for e in _events(trace):
        if e.get("cat") in _ANNOTATION_CATS and e.get("name") in totals:
            totals[e["name"]] += e.get("dur", 0.0)
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
    First-cut single-timeline view; needs `ts` on kernel events (present on real GPU traces).
    """
    spans = sorted((e["ts"], e["ts"] + e.get("dur", 0.0))
                   for e in _events(trace)
                   if e.get("cat") in _KERNEL_CATS and "ts" in e and "dur" in e)
    gaps, running_end = [], None
    for start, end in spans:
        if running_end is not None:
            if start - running_end > 0:
                gaps.append(start - running_end)
            running_end = max(running_end, end)
        else:
            running_end = end
    return len(gaps), (max(gaps) if gaps else 0.0)


def provenance(trace):
    """What actually ran, read off kernel names — the 'did it apply?' check."""
    names = " ".join(name.lower() for name, _ in kernels(trace))
    return {
        "flash_attention": any(k in names for k in ("flash", "fmha", "mem_efficient")),
        "fused_kernels": any(k in names for k in ("fused", "cutlass", "liger")),
        "dtypes": sorted({d for d in ("bf16", "fp16", "tf32", "fp8") if d in names}),
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
