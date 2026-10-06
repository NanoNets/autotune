"""Assemble the Profile: the once-per-run framing the agent reads.

`build(...)` is pure (takes the already-extracted parts), so the assembly + memory-fit logic
is testable without transformers/datasets/GPU. `frame(model, dataset, gpu)` is the live shell:
measure the peak, load the HF config + tokenizer, scan the dataset, then build.

The peak FLOP/s (MFU denominator) is MEASURED once here at the run dtype and carried in the
Profile; the loop threads it to every probe via `capture(..., peak_flops=...)`. Measuring it
once matters: on cloud GPUs the measured peak swings ~28% run-to-run while step time varies
~1%, so if each probe measured its own, probe-to-probe MFU would be mostly measurement noise.

`model_state_fits` covers the model state ONLY — weights/grads/optimizer. Activations are not in
it, so a positive `activation_headroom_gb` is what actually decides whether a config runs.

`max_len` is the human's declared application sequence length (off-limits, like effective_batch).
It is the yardstick the OOM `ceiling` is measured at (`frame.ceiling`): memory is decided by the
worst batch a run will ever see, so the ceiling is measured on manufactured max_len batches rather
than hoping real data supplies one. build() only validates and carries it — measuring is the
caller's job, because a real ceiling needs a real training step.

The memory numbers assume a default full-fine-tune config (the run dtype + AdamW) — recorded in
`default_config` so it's clear they move with the knobs, not a fixed truth.
"""
from dataclasses import dataclass, field

from frame.memory import training_memory

_DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4, "float64": 8}


@dataclass
class Profile:
    model: dict = field(default_factory=dict)          # internals: dims, params, attention, dtype
    dataset: dict = field(default_factory=dict)        # length distribution + volume
    hardware: dict = field(default_factory=dict)       # measured peak FLOP/s + memory, single-gpu
    versions: dict = field(default_factory=dict)       # library versions (gate which knobs exist)
    memory: dict = field(default_factory=dict)         # analytical model-state breakdown
    default_config: dict = field(default_factory=dict) # the (dtype, optimizer) the memory assumes
    effective_batch: int = None                        # human-owned target; the loop factorizes it, never changes it
    knobs: dict = field(default_factory=dict)          # the framework's SEARCHABLE surface (agent classifies at Frame)
    flops_per_token: float = None                      # 6N; loop multiplies by tokens/step for the floor
    activation_headroom_gb: float = None               # mem - model_state (room for activations)
    model_state_fits: bool = None                      # does MODEL STATE alone fit — says nothing
                                                       #   about activations; see the headroom
    max_len: int = None                                # human-declared sequence length (off-limits)
    ceiling: dict = None                               # MEASURED micro_batch ceiling at max_len;
                                                       #   None when it wasn't measured
    notes: list = field(default_factory=list)          # things the agent must see, not errors
    summary: str = ""


# max_len this many times the dataset's p99 means most of every batch is padding — worth the
# agent's attention (packing is likely a large win), never an error: it's the human's call.
_PADDING_FLAG = 4


def _ceiling_str(max_len, ceiling):
    """The summary's memory-ceiling clause."""
    if not ceiling:
        return f"max_len {max_len} (ceiling NOT measured)" if max_len else "no max_len"
    if not ceiling.get("fits"):
        return f"ceiling @{ceiling['seq_len']}: does NOT fit at micro_batch 1"
    plus = "+" if ceiling.get("capped") else ""        # capped: the real ceiling is >= this
    return (f"ceiling @{ceiling['seq_len']}: micro_batch<={ceiling['micro_batch_max']}{plus} "
            f"(measured)")


def _bound_knobs(knobs, ceiling):
    """Stamp the measured ceiling onto `micro_batch`'s range.

    A framework's `knobs()` is a static declaration of what it SUPPORTS — it is called with no
    arguments and cannot know a per-card, per-config measurement, so it can only say `min: 1`.
    Left at that, the SearchSpace advertises an unbounded int and the bound holds only if the
    agent remembers to cross-reference the ceiling itself. Putting `max` here is what actually
    makes `micro_batch` a bounded dimension.
    """
    out = {k: dict(v) for k, v in (knobs or {}).items()}
    if ceiling and ceiling.get("fits") and "micro_batch" in out:
        out["micro_batch"]["max"] = ceiling["micro_batch_max"]
        out["micro_batch"]["max_source"] = "measured ceiling at max_len"
    return out


def _notes(dataset, max_len, ceiling):
    """Things the agent must see before it starts tuning. Facts + consequences, no labels."""
    notes = []
    p99 = dataset.get("p99")
    if max_len and p99 and max_len >= _PADDING_FLAG * p99:
        notes.append(
            f"max_len {max_len} is {max_len // p99}x the dataset p99 ({p99} tok): most of every "
            f"unpacked batch is padding. Expect packing to be a large win, or max_len is set "
            f"higher than this data needs — confirm with the human.")
    if ceiling and not ceiling.get("fits"):
        notes.append(
            f"micro_batch=1 OOMs at max_len {ceiling['seq_len']}: this job does not fit on this "
            f"card as configured. This is a MEMORY regime — start from recompute / offload / "
            f"sharding / 8-bit optimizer states, not from batch knobs. NEVER drop dtype to make it "
            f"fit: precision changes the model, so put it to the human with these numbers and let "
            f"them decide.")
    if ceiling and ceiling.get("capped"):
        notes.append(
            f"the ceiling search hit its cap at micro_batch {ceiling['micro_batch_max']}; the "
            f"real ceiling is higher. Memory is not the binding constraint here.")
    if ceiling and ceiling.get("card_frac") and ceiling["card_frac"] > 0.97:
        notes.append(
            f"the ceiling sits at {ceiling['card_frac']:.0%} of the card. Fixed-shape probing "
            f"does not reproduce the fragmentation real varying lengths cause — keep a margin "
            f"below micro_batch {ceiling['micro_batch_max']} rather than running at it.")
    return notes


def build(model, dataset, hardware, versions, dtype="bfloat16", optimizer="adamw",
          effective_batch=None, knobs=None, max_len=None, ceiling=None):
    """model/dataset/hardware/versions: the extracted parts (see the sibling modules).

    dtype: the run compute dtype — drives the memory precision AND is the dtype the peak was
    measured at (carried in default_config so the loop passes the matching config["dtype"]).
    effective_batch: the human's target (carried, not searched). knobs: the framework's searchable
    surface, so the agent classifies safe/off-limits at Frame from what's actually tunable.
    max_len: the human-declared sequence length. Validated against the architecture here.
    ceiling: the record from `frame.ceiling` at max_len, or None if not measured.
    """
    peak_dtype = hardware.get("peak_dtype")
    if peak_dtype is not None and peak_dtype != dtype:  # a peak measured at another precision is
        raise ValueError(                               # not a ceiling for THIS run
            f"peak was measured at {peak_dtype!r} but the run dtype is {dtype!r}; "
            "MFU against a mismatched peak is meaningless")

    ctx = model.get("context_len")
    if max_len is not None and ctx and max_len > ctx:    # caught here, not as a crash mid-run
        raise ValueError(
            f"max_len {max_len} exceeds the model's context length {ctx}; the architecture "
            "cannot take sequences that long")

    params = model["params"]
    param_bytes = _DTYPE_BYTES.get(dtype, 2)
    mem = training_memory(params, param_bytes=param_bytes, optimizer=optimizer)

    mem_gb = hardware.get("mem_gb")
    headroom = round(mem_gb - mem["model_state_gb"], 1) if mem_gb is not None else None
    fits = headroom > 0 if headroom is not None else None

    name = model.get("arch") or "model"
    peak = hardware.get("peak_flops")
    peak_str = f" | peak {peak/1e12:.0f} TFLOP/s (measured)" if peak else ""
    mem_str = (f"{mem_gb}GB x{hardware.get('count', 1)}" if mem_gb is not None else "mem n/a")
    fit_str = (f"-> model state {'fits' if fits else 'does NOT fit'} (headroom {headroom}GB)"
               if fits is not None else "-> fit n/a")
    ds = dataset
    ds_str = (f"{ds['n_samples']} samples, p50 {ds['p50']}/p99 {ds['p99']} tok"
              if ds.get("n_samples") else "no dataset")
    eb_str = f" | eff-batch {effective_batch}" if effective_batch else ""
    summary = (f"{name} ({model['attention']}, {dtype}) | {params/1e9:.1f}B params{eb_str} | "
               f"{hardware.get('gpu')} {mem_str}{peak_str} | "
               f"full-FT model-state {mem['model_state_gb']}GB {fit_str} | "
               f"{_ceiling_str(max_len, ceiling)} | {ds_str}")

    return Profile(
        model=model, dataset=dataset, hardware=hardware, versions=versions, memory=mem,
        default_config={"dtype": dtype, "param_bytes": param_bytes, "optimizer": optimizer},
        effective_batch=effective_batch, knobs=_bound_knobs(knobs, ceiling),
        flops_per_token=6.0 * params, activation_headroom_gb=headroom,
        model_state_fits=fits, max_len=max_len, ceiling=ceiling,
        notes=_notes(dataset, max_len, ceiling), summary=summary,
    )


def _framework_knobs(framework):
    """The searchable surface the named framework exposes ({} if it declares none, e.g. the fixture)."""
    import importlib
    fn = getattr(importlib.import_module(f"frameworks.{framework}"), "knobs", None)
    return fn() if callable(fn) else {}


def framework_runner(framework):
    """The named framework's runner class ({}.Runner convention: `HFRunner`, `TorchRunner`).

    Kept separate from `frame()`'s `runner=` argument on purpose: constructing a runner loads real
    weights, so measuring a ceiling stays something the caller asks for, never a side effect of
    profiling.
    """
    import importlib
    mod = importlib.import_module(f"frameworks.{framework}")
    for name in (f"{framework.upper()}Runner", f"{framework.capitalize()}Runner"):
        if hasattr(mod, name):
            return getattr(mod, name)
    raise ValueError(f"frameworks.{framework} exposes no runner class")


def frame(model, dataset, gpu, max_len, dtype="bfloat16", device="cuda", framework="hf",
          effective_batch=16, text_field="text", split="train", sample=1000, name=None,
          runner=None, seed=0, on_trial=None, sample_path=None, sample_seed=0, max_scan=None,
          ceiling_rows_path=None, ceiling_row_template=None, ceiling_rows_format="text"):
    """Live: measure the peak, load config + tokenizer, scan the dataset, measure the ceiling,
    assemble.

    Needs torch + a GPU (peak), transformers + datasets (model/tokenizer/dataset). `gpu` is a
    free-form label. Measuring the peak first also fails fast if there's no usable device.

    runner: supply one to MEASURE the OOM ceiling at max_len. It must be the run's REAL training
    step — a ceiling measured on a different workload describes that workload, not this job. No
    runner, or a non-cuda device, and the Profile carries `ceiling=None`, which the summary says
    out loud rather than implying a ceiling was checked.
    effective_batch: the human's target. The ceiling search tries only its divisors (micro_batch
    must divide it), up to effective_batch itself.
    ceiling_rows_path / ceiling_row_template / ceiling_rows_format: for a user's script, write the
    over-length rows its ceiling trials read (`frame.ceiling_rows`), `effective_batch * 3` of them.
    sample / sample_path / sample_seed / max_scan: the PROBE SAMPLE (`frame.dataset.draw_sample`).
    `sample` rows, rounded down to a multiple of effective_batch, are drawn length-stratified from
    a random pool over the split. The pool's distribution becomes the Profile's dataset shape, and
    when `sample_path` is given the drawn rows are written there for every probe to load
    (`sample_file` in the probe config), so Frame's statistics and the probes' rows are one
    population.
    """
    from frame.hardware import hardware, versions, measure
    m = measure(device, dtype)                          # measure the peak ONCE, at the run dtype
    from transformers import AutoTokenizer
    from frame.model import load
    from frame.dataset import draw_sample, write_sample

    mi = load(model)
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    drawn = draw_sample(dataset, tok, sample, text_field=text_field, split=split, name=name,
                        seed=sample_seed, effective_batch=effective_batch, max_len=max_len,
                        max_scan=max_scan)
    if sample_path:
        write_sample(sample_path, drawn)
    ds = drawn["meta"]["pool"]                          # the corpus's shape, not the head's
    hw = hardware(gpu, peak_flops=m["peak_flops"], mem_gb=m["mem_gb"], dtype=dtype,
                  device_name=m["device_name"], num_sms=m["num_sms"],
                  peak_mem_bw_gb_s=m["peak_mem_bw_gb_s"])

    if ceiling_rows_path:                               # a user's script reads these in a trial
        from frame.ceiling_rows import write_ceiling_rows
        write_ceiling_rows(ceiling_rows_path, tok, max_len, effective_batch * 3,
                           text_field=text_field, template=ceiling_row_template,
                           fmt=ceiling_rows_format, seed=seed)

    ceiling = None
    if runner is not None and device == "cuda":
        from frame.ceiling import find_ceiling, framework_defaults
        ceiling = find_ceiling(                         # memory-only trials: no peaks re-measured
            runner, {"device": device, "dtype": dtype, "model": model, "framework": framework},
            seq_len=max_len, vocab=mi["vocab"], effective_batch=effective_batch, seed=seed,
            on_trial=on_trial, defaults=framework_defaults(framework))

    return build(mi, ds, hw, versions(), dtype=dtype,
                 effective_batch=effective_batch, knobs=_framework_knobs(framework),
                 max_len=max_len, ceiling=ceiling)
