"""Measure the OOM ceiling: the largest micro_batch that fits at a given sequence length.

Why measure it instead of discovering it: it turns `micro_batch` from an unbounded search
dimension into a bounded one BEFORE the loop spends a probe on a config that was never going to
run. A ceiling run is a set of short **trials**, never `results.tsv` rows.

- **A trial is memory-only:** `TRIAL_STEPS` optimizer steps at one micro-batch, no timing, no
  profiler, then `ok` (fits) / `oom` (doesn't) / `error` (a crash: never a memory answer). Step 1
  alone isn't enough: AdamW allocates its state during step 1's `opt.step()`, so the steady-state
  peak first appears at step 2. The third step covers CUDA-graph pools and early allocator growth.
  Each trial runs at accumulation 1 (`effective_batch = micro_batch`): accumulation doesn't change
  the peak, only the time.
- **Only divisors of `effective_batch` are tried** (`search_divisors`): micro_batch must divide it,
  so a ceiling of 11 against 16 is a number no config can use. At most 5 trials for 16.
- **The ceiling is NOT a run constant.** Every memory knob moves it, so the record stamps the
  FULL set of memory knobs it was measured under, defaults included (`stamp`), and `is_stale`
  compares a candidate config against that stamp. Stale -> re-measure **from the previous ceiling**
  (usually 1-2 trials), don't extrapolate.

Two trial providers, one search:
- `runner_trial`: in-process, on our runner with `data="synthetic"` (worst-case batches,
  `frame.synthetic`). Frame uses it at Step 2, on the box.
- `backend_trial`: one `backends.run` per trial with `probe_kind: "ceiling_trial"`, for our runner
  or a user's script (the script reads Frame's `ceiling_rows.jsonl` through its own data setting
  and runs under `profiler.launch --memory-only`). `python -m frame.ceiling` re-measures with it.

Recorded, not hidden: fixed-shape probing avoids the allocator fragmentation that varying real
lengths cause, so `frag_gb` understates a long run's drift. `card_frac` is reported so a margin
can be kept below the ceiling; how much margin is the agent's call, not a constant in here.
"""
import json
import os

TRIAL_STEPS = 3

# The knobs that move the ceiling. Every one present in the config or the framework's defaults is
# stamped on the record, defaults included, so a later config can be compared against it and the
# ceiling retired when one of them has changed.
#
# Canonical names (loop/throughput.md) AND the adapter spellings they project onto. A name this list
# misses is a knob whose change goes UNDETECTED. A user's script declares its own names instead
# (`memory_knobs` in the probe config, from the knob audit).
#
# `dtype` is here because a HUMAN changing precision invalidates the measurement, NOT because the
# loop may turn it to buy memory. Precision is off-limits (loop/throughput.md).
_MEMORY_KNOBS = (
    "dtype", "optimizer", "shard_degree", "offload", "packing",
    "recompute", "gradient_checkpointing",              # canonical, hf
    "attn_kernel", "attn_implementation",               # canonical, hf
    "compile", "torch_compile",                         # canonical, hf: compile fuses the CE and
                                                        #   moved a measured ceiling 2 -> 4
    "chunked_ce",                                       # hf: frees the [tokens, vocab] logits
    "compile_optimizer",                                # hf: changes the optimizer's temporaries
)
_ALLOC_ENV = "PYTORCH_CUDA_ALLOC_CONF"                  # moves fragmentation, so the ceiling


def divisors(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def search_divisors(fits, effective_batch, previous=None):
    """Largest divisor of `effective_batch` with `fits(d)` True; 0 if even 1 doesn't fit.

    Assumes `fits` is monotone (if d fits, every smaller d fits), true of batch size against memory.

    previous=None: a fresh search, ascending, stopping at the first failure.
    previous=<a divisor>: a re-measure after a config change. Try the next divisor UP first: if it
    fits, keep climbing (and `previous` fits too, by monotonicity). If it doesn't, check `previous`
    itself, and if that fails as well, step DOWN until one fits. So a change that frees memory
    costs 1 trial per step gained plus one, a change that leaves it flat costs 2, and a change
    that adds memory costs 2 plus one per step lost.
    """
    ds = divisors(effective_batch)
    if not previous or previous not in ds:
        best = 0
        for d in ds:
            if not fits(d):
                break
            best = d
        return best
    i = ds.index(previous)
    if i + 1 < len(ds) and fits(ds[i + 1]):
        best, j = ds[i + 1], i + 2
        while j < len(ds) and fits(ds[j]):
            best, j = ds[j], j + 1
        return best
    if fits(previous):
        return previous
    for d in reversed(ds[:i]):
        if fits(d):
            return d
    return 0


def framework_defaults(framework):
    """The framework runner's resolved defaults ({} if it declares none)."""
    import importlib
    try:
        return dict(getattr(importlib.import_module(f"frameworks.{framework}"), "_DEFAULTS", {}))
    except ImportError:
        return {}


def stamp(config, defaults=None):
    """The memory knobs a ceiling is measured under, DEFAULTS INCLUDED.

    Our runner: every `_MEMORY_KNOBS` name the config or the framework's defaults carries. A user's
    script: `dtype` plus its own `memory_knobs` (the script's names and current values, from the
    knob audit). The CUDA allocator setting, when set, either way.
    """
    if config.get("script"):
        if not isinstance(config.get("memory_knobs"), dict):
            raise ValueError("a script's ceiling needs `memory_knobs`: the script's current value "
                             "for every memory knob (knob audit, program.md Step 2a)")
        out = {"dtype": config.get("dtype"), **config["memory_knobs"]}
    else:
        d = defaults if defaults is not None else framework_defaults(config.get("framework", "torch"))
        out = {k: config.get(k, d.get(k)) for k in _MEMORY_KNOBS if k in config or k in d}
    alloc = (config.get("env") or {}).get(_ALLOC_ENV)
    if alloc:
        out[_ALLOC_ENV] = alloc
    for k in ("max_image_size", "max_images"):         # a VLM's image worst case: another image
        if config.get(k):                              # size or count is another ceiling
            out[k] = config[k]
    return out


def is_stale(ceiling, config, defaults=None):
    """True when `config` would not run under the conditions the ceiling was measured in: another
    `max_len` or `effective_batch`, or any memory knob moved. Re-measure, don't extrapolate.
    Micro-batch, accumulation and every other non-memory knob never make it stale."""
    if not ceiling:
        return True
    want = config.get("max_len") or config.get("seq_len")
    if want and ceiling.get("seq_len") and want != ceiling["seq_len"]:
        return True
    eb = config.get("effective_batch")
    if eb and ceiling.get("effective_batch") and eb != ceiling["effective_batch"]:
        return True
    old, new = ceiling.get("config") or {}, stamp(config, defaults)
    return any(old.get(k) != new.get(k) for k in set(old) | set(new))


def measure(trial, effective_batch, seq_len, config_stamp, previous=None, on_trial=None,
            method="synthetic", seed=0, steps=TRIAL_STEPS):
    """Run the divisor search with `trial(n) -> {"status", "error", "peak_gb", ...}` and return the
    ceiling record. `previous`: the stale record to re-measure from (same seq_len and
    effective_batch), or None for a fresh search."""
    trials = []

    def fits(n):
        t = {"micro_batch": n, **trial(n)}
        trials.append(t)
        if on_trial:
            on_trial(t)
        if t["status"] == "error":                     # a crash is not a memory answer: never
            raise RuntimeError(                        # let it be read as one
                f"ceiling trial micro_batch={n} failed with an error, not an OOM: {t.get('error')}")
        return t["status"] == "ok"

    prev = previous.get("micro_batch_max") if previous else None
    best = search_divisors(fits, effective_batch, previous=prev)
    ooms = [t["micro_batch"] for t in trials if t["status"] == "oom"]
    at_best = next((t for t in reversed(trials)
                    if t["micro_batch"] == best and t["status"] == "ok"), None) or {}
    peak, reserved, total = at_best.get("peak_gb"), at_best.get("reserved_gb"), at_best.get("total_gb")
    return {
        "seq_len": seq_len,
        "effective_batch": effective_batch,
        "micro_batch_max": best,
        "fits": best > 0,                              # False => memory regime, not a batch problem
        "capped": best == effective_batch,             # True => the real ceiling is >= this
        "oom_at": min(ooms) if ooms else None,
        "peak_gb": peak,
        "reserved_gb": reserved,
        "frag_gb": round(reserved - peak, 3) if peak is not None and reserved is not None else None,
        "card_frac": round(reserved / total, 3) if reserved and total else None,
        "config": config_stamp,
        "method": method,
        "steps_per_trial": steps,
        "seed": seed,
        "remeasured_from": prev,
        "trials": trials,
    }


def _release(runner, device):
    """Drop a trial's allocations: the next trial must start from a clean card, or the search
    measures leftovers instead of the config."""
    import gc

    import torch
    if hasattr(runner, "teardown"):
        runner.teardown()
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def _memory(device):
    import torch
    if device != "cuda":
        return {"peak_gb": None, "reserved_gb": None, "total_gb": None}
    gb = 1024 ** 3
    return {"peak_gb": round(torch.cuda.max_memory_allocated() / gb, 3),
            "reserved_gb": round(torch.cuda.max_memory_reserved() / gb, 3),
            "total_gb": round(torch.cuda.mem_get_info()[1] / gb, 3)}


def _first_line(e):
    msg = str(e).strip()
    return msg.splitlines()[0] if msg else type(e).__name__


def runner_trial(runner, config, seq_len, vocab=None, steps=TRIAL_STEPS, seed=0):
    """`trial(n)` on our runner, in this process: worst-case synthetic batches at `seq_len`."""
    def trial(n):
        import torch
        cfg = {**config, "data": "synthetic", "micro_batch": n, "effective_batch": n,
               "max_len": seq_len, "vocab": vocab, "seed": seed}
        device = cfg.get("device", "cpu")
        if device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        try:
            runner.setup(cfg)
            for _ in range(steps):
                runner.step()
            if device == "cuda":
                torch.cuda.synchronize()
            out = {"status": "ok", "error": None, **_memory(device)}
        except torch.cuda.OutOfMemoryError as e:       # the answer "doesn't fit"
            out = {"status": "oom", "error": _first_line(e), **_memory(device)}
        except Exception as e:                         # anything else tells us nothing about memory
            out = {"status": "error", "error": f"{type(e).__name__}: {_first_line(e)}",
                   **_memory(device)}
        finally:
            _release(runner, device)
        return out
    return trial


def find_ceiling(runner, config, seq_len, vocab, effective_batch, seed=0, steps=TRIAL_STEPS,
                 on_trial=None, previous=None, defaults=None):
    """The ceiling on our runner, in-process (Frame Step 2). `config`: the runner's base config
    (e.g. `model` for `frameworks.hf`); `data`, `micro_batch`, `effective_batch`, `max_len`,
    `vocab` and `seed` are set per trial. `defaults`: the framework's, for the stamp."""
    return measure(runner_trial(runner, config, seq_len, vocab, steps, seed), effective_batch,
                   seq_len, stamp(config, defaults if defaults is not None else {}),
                   previous=previous, on_trial=on_trial, seed=seed, steps=steps)


# ---- through a backend: re-measures, and user scripts -------------------------------------
def _render(config, n, steps):
    """The probe config for one trial at micro-batch n. For a script, `{micro_batch}` is filled in
    `script_args`, and each `ceiling_templates` entry ({out_path: template_path}) is rendered to
    a per-trial file (e.g. `runs/<run>/probes/ceiling_mb{micro_batch}.yaml`) added to
    `script_files`, so the backend ships it."""
    cfg = {**config, "probe_kind": "ceiling_trial", "micro_batch": n, "effective_batch": n,
           "trial_steps": steps}
    if config.get("script"):
        cfg["script_args"] = [str(a).replace("{micro_batch}", str(n))
                              for a in config.get("script_args") or []]
        files = list(config.get("script_files") or [])
        for out, tmpl in (config.get("ceiling_templates") or {}).items():
            path = out.replace("{micro_batch}", str(n))
            with open(tmpl) as f:
                text = f.read().replace("{micro_batch}", str(n))
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            with open(path, "w") as f:
                f.write(text)
            files.append(path)
        cfg["script_files"] = files
    else:
        cfg["data"] = "synthetic"
    return cfg


def backend_trial(config, steps=TRIAL_STEPS):
    """`trial(n)` as one `backends.run` each (a fresh process: local, ssh or modal)."""
    def trial(n):
        import backends
        return backends.run(_render(config, n, steps))
    return trial


_TSV_HEADER = "seq_len\tmicro_batch\tstatus\tpeak_gb\treserved_gb\tconfig\tnote\n"


def append_tsv(path, rec, note=""):
    """Append a ceiling run's trials to `runs/<run>/ceiling.tsv`."""
    new = not os.path.exists(path)
    with open(path, "a") as f:
        if new:
            f.write(_TSV_HEADER)
        for t in rec["trials"]:
            mark = "CEILING" if t["micro_batch"] == rec["micro_batch_max"] and t["status"] == "ok" else ""
            if t["micro_batch"] == rec["oom_at"]:
                mark = "oom_at"
            f.write(f"{rec['seq_len']}\t{t['micro_batch']}\t{t['status']}\t{t.get('peak_gb')}\t"
                    f"{t.get('reserved_gb')}\t{json.dumps(rec['config'], separators=(',', ':'))}\t"
                    f"{' '.join(x for x in (mark, note) if x)}\n")


def main(argv=None):
    """python -m frame.ceiling --config probe.json --run runs/<run> [--force]

    Re-measure (or, for a script, first measure) the ceiling for a candidate probe config, through
    the same backend a probe uses. Reads the current ceiling from `runs/<run>/profile.json`; when
    it isn't stale for this config, says so and runs nothing. Otherwise searches from it, appends
    the trials to `runs/<run>/ceiling.tsv`, and replaces the Profile's ceiling (the old one is kept
    under `ceiling_history`, and `knobs.micro_batch.max` follows).
    """
    import argparse
    ap = argparse.ArgumentParser(prog="python -m frame.ceiling")
    ap.add_argument("--config", required=True, help="the candidate probe config (JSON file)")
    ap.add_argument("--run", required=True, help="the run directory, e.g. runs/<run>")
    ap.add_argument("--force", action="store_true", help="measure even if not stale")
    ap.add_argument("--fresh", action="store_true",
                    help="measure from scratch, ignoring the previous ceiling (implies --force)")
    ap.add_argument("--steps", type=int, default=TRIAL_STEPS)
    a = ap.parse_args(argv)

    with open(a.config) as f:
        config = json.load(f)
    prof_path = os.path.join(a.run, "profile.json")
    profile = {}
    if os.path.exists(prof_path):
        with open(prof_path) as f:
            profile = json.load(f)
    seq_len = config.get("max_len") or profile.get("max_len")
    eb = config.get("effective_batch") or profile.get("effective_batch")
    if not seq_len or not eb:
        raise SystemExit("needs max_len and effective_batch (in the config or the run's profile.json)")
    config = {**config, "max_len": seq_len, "effective_batch": eb}
    images = profile.get("images") or {}               # a VLM: the rows carry maxed images
    if images and config.get("script"):
        config = {**config, "max_image_size": images["max_image_size"],
                  "max_images": images["max_images"],
                  "expect_image_tokens": images.get("image_tokens_per_row"),
                  "image_token_id": images.get("image_token_id")}
        if images.get("ceiling_image"):                # ships with the rows (ssh)
            files = list(config.get("script_files") or [])
            if images["ceiling_image"] not in files:
                config["script_files"] = files + [images["ceiling_image"]]
    vocab = (profile.get("model") or {}).get("vocab")
    if vocab and not config.get("script"):             # the embedding width: an id past it is a
        config.setdefault("vocab", vocab)              # device-side assert, not the OOM measured
    current = profile.get("ceiling")
    if current and not (a.force or a.fresh) and not is_stale(current, config):
        print(f"ceiling still valid: micro_batch <= {current['micro_batch_max']} at {seq_len} "
              f"(no memory knob moved)")
        print(json.dumps(current, default=str))
        return current
    previous = None
    if current and not a.fresh and current.get("seq_len") == seq_len \
            and current.get("effective_batch", eb) == eb and current.get("fits"):
        previous = current

    def on_trial(t):
        print(f"  mb={t['micro_batch']:<5} {t['status']:<5} peak={t.get('peak_gb')} "
              f"reserved={t.get('reserved_gb')}" + (f"  {t['error']}" if t.get("error") else ""),
              flush=True)

    print(f"ceiling: {'re-measuring from ' + str(previous['micro_batch_max']) if previous else 'measuring'}"
          f" at max_len={seq_len}, effective_batch={eb}", flush=True)
    rec = measure(backend_trial(config, a.steps), eb, seq_len, stamp(config), previous=previous,
                  on_trial=on_trial, method="ceiling-rows" if config.get("script") else "synthetic",
                  seed=config.get("seed", 0), steps=a.steps)
    append_tsv(os.path.join(a.run, "ceiling.tsv"), rec,
               note=f"re-measure from {previous['micro_batch_max']}" if previous else "fresh")
    if profile or os.path.isdir(a.run):
        if current:
            profile.setdefault("ceiling_history", []).append(current)
        profile["ceiling"] = rec
        knobs = profile.get("knobs") or {}
        if "micro_batch" in knobs and rec["fits"]:
            knobs["micro_batch"]["max"] = rec["micro_batch_max"]
            knobs["micro_batch"]["max_source"] = "measured ceiling at max_len"
        with open(prof_path, "w") as f:
            json.dump(profile, f, indent=2, default=str)
    print(f"ceiling: micro_batch <= {rec['micro_batch_max']}{'+' if rec['capped'] else ''} "
          f"({len(rec['trials'])} trials)")
    print(json.dumps(rec, default=str))
    return rec


if __name__ == "__main__":
    main()
