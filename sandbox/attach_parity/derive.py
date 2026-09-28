"""Recover phases, tokens and micro-batches from a torch.profiler trace alone. Torch-free.

The patch-free replacement for the label hooks. Everything here is read out of what torch
records by itself when the profiler runs (with `record_shapes=True`):

  backward   `autograd::engine::evaluate_function: ...` events, one per autograd node. One
             backward pass is a cluster of them, split wherever main-thread work or a DataLoader
             fetch happens in between. Detected by EVENT TYPE, not by thread, so it holds whether
             the engine runs backward on its own thread (CUDA) or on the caller's (CPU).
  dataloader torch's own `enumerate(DataLoader)#<Iter>.__next__` label, re-emitted under the
             name `dataloader` so `diagnose()` reads it unchanged.
  forward    everything on the main thread from the first non-copy op after the batch arrives
             up to the start of that micro-batch's backward. This includes the loss, like our
             runners' `forward` label, and excludes the host->device batch copy, which our
             runners also leave unlabelled.
  tokens     the index input of the first `aten::embedding` in each forward (`Input Dims`),
             else the leading dims of the first linear/matmul's input. This is the PADDED count,
             the same thing `frameworks/hf.py:264` counts (`input_ids.numel()`).

`augment(trace)` adds the derived spans to the trace as ordinary `user_annotation` events, each
marked `args.derived = "attach.derive"`, so the frozen diagnostic consumes them unchanged, and
returns what it found. Nothing is inferred from timing gaps or thresholds.
"""
import bisect

EVALF = "autograd::engine::evaluate_function"
LOADER = "enumerate(DataLoader)"
OPTIM = "Optimizer."
# Ops that move or allocate the incoming batch before the model runs. A forward span starts at
# the first main-thread op that is NOT one of these.
COPY_OPS = {"aten::to", "aten::_to_copy", "aten::copy_", "aten::empty_strided", "aten::empty",
            "aten::pin_memory", "aten::_pin_memory", "aten::is_pinned", "aten::lift_fresh",
            "aten::detach", "aten::detach_", "aten::alias", "aten::view", "aten::contiguous"}
LINEAR_OPS = ("aten::linear", "aten::addmm", "aten::matmul", "aten::mm")
MARK = {"derived": "attach.derive"}


def _end(e):
    return e["ts"] + e.get("dur", 0.0)


def _top_level(events):
    """Events on one thread that aren't nested inside another one."""
    out, end = [], float("-inf")
    for e in sorted(events, key=lambda e: (e["ts"], -e.get("dur", 0.0))):
        if e["ts"] >= end:
            out.append(e)
            end = _end(e)
        else:
            end = max(end, _end(e))
    return out


def _dims(e, i):
    d = (e.get("args") or {}).get("Input Dims") or []
    return d[i] if len(d) > i and isinstance(d[i], list) else None


def _prod(xs):
    n = 1
    for x in xs:
        n *= x
    return n


def _tokens_in(ops):
    """(tokens, source) for one forward span's ops, in time order."""
    for e in ops:
        if e["name"] == "aten::embedding":
            idx = _dims(e, 1)                           # (weight, indices, ...)
            if idx:
                return _prod(idx), "embedding"
    for e in ops:
        if e["name"] in LINEAR_OPS:
            x = _dims(e, 1 if e["name"] == "aten::addmm" else 0)   # addmm(bias, input, weight)
            if x:
                return (_prod(x[:-1]) if len(x) > 1 else x[0]), "first-linear"
    return None, None


def augment(trace):
    """Add derived forward / backward / dataloader spans; return what was found.

    Returns {"forwards", "backwards", "tokens", "token_source", "notes"}. Mutates `trace`.
    """
    ev = trace.get("traceEvents", [])
    steps = [e for e in ev if e.get("cat") == "user_annotation" and e.get("name") == "step"
             and "dur" in e]
    if not steps:
        return {"forwards": 0, "backwards": 0, "tokens": None, "token_source": None,
                "notes": ["no `step` annotation"]}
    step = max(steps, key=lambda e: e["dur"])
    s0, s1, main, pid = step["ts"], _end(step), step.get("tid"), step.get("pid")

    def inside(e):
        return s0 <= e["ts"] <= s1

    ops = [e for e in ev if e.get("cat") == "cpu_op" and "dur" in e and inside(e)]
    anns = [e for e in ev if e.get("cat") == "user_annotation" and "dur" in e and inside(e)]
    loaders = sorted((e for e in anns if e["name"].startswith(LOADER)), key=lambda e: e["ts"])
    optim = [e for e in anns if e["name"].startswith(OPTIM)]

    def in_optim(e):
        return any(o["ts"] <= e["ts"] <= _end(o) for o in optim)

    main_ops = [e for e in ops if e.get("tid") == main]
    main_top = [e for e in _top_level(main_ops) if not e["name"].startswith(EVALF)]
    main_work = [e for e in main_top if not in_optim(e)]

    # backward: evaluate_function events, clustered; a split wherever main-thread work (not
    # inside an evaluate_function) or a DataLoader fetch starts between two of them
    evalf = sorted((e for e in ops if e["name"].startswith(EVALF)), key=lambda e: e["ts"])
    markers = sorted([e["ts"] for e in main_work] + [e["ts"] for e in loaders])
    clusters = []
    for e in evalf:
        if clusters:
            prev_end = clusters[-1][1]
            i = bisect.bisect_left(markers, prev_end)
            if i < len(markers) and markers[i] < e["ts"]:
                clusters.append([e["ts"], _end(e)])
            else:
                clusters[-1][1] = max(prev_end, _end(e))
        else:
            clusters.append([e["ts"], _end(e)])

    # forward k: from the first non-copy main-thread op after the batch arrives, to backward k
    forwards, tokens, sources, notes = [], 0, set(), []
    prev = s0
    for b0, b1 in clusters:
        arrived = max([prev] + [_end(l) for l in loaders if prev <= l["ts"] < b0])
        cands = [e for e in main_work if arrived <= e["ts"] < b0 and e["name"] not in COPY_OPS]
        if cands:
            f0 = cands[0]["ts"]
            forwards.append((f0, b0))
            span_ops = sorted((e for e in main_ops if f0 <= e["ts"] < b0), key=lambda e: e["ts"])
            n, src = _tokens_in(span_ops)
            if n is None:
                notes.append(f"no embedding/linear op in forward at ts={f0}")
            else:
                tokens += n
                sources.add(src)
        else:
            notes.append(f"no forward work before backward at ts={b0}")
        prev = b1

    def span(name, t0, t1):
        return {"ph": "X", "cat": "user_annotation", "name": name, "ts": t0, "dur": t1 - t0,
                "pid": pid, "tid": main, "args": dict(MARK)}

    ev += [span("forward", a, b) for a, b in forwards]
    ev += [span("backward", a, b) for a, b in clusters]
    ev += [span("dataloader", l["ts"], _end(l)) for l in loaders]
    return {"forwards": len(forwards), "backwards": len(clusters),
            "tokens": tokens if sources else None,
            "token_source": "+".join(sorted(sources)) or None, "notes": notes}
