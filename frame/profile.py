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

The memory numbers assume a default config (the run dtype + AdamW, a full fine-tune unless `lora`
says otherwise) — recorded in `default_config` so it's clear they move with the knobs, not a fixed
truth. With `lora` (rank + target modules, read from the user's script), the adapters are counted
on the model built on the meta device, matched the way peft matches them (`frame.model.load`), or
from the config's dims when it can't be built (`frame.model.lora_params`, an estimate that says
so), and only they carry grads and optimizer state:
FLOPs per token become 4 * N_total + 2 * N_trainable, the profiler's own formula.
"""
from dataclasses import dataclass, field

from frame.memory import training_memory
from frame.model import lora_params

_LORA_ADAPTER_BYTES = 4                                # peft keeps adapters in fp32 by default
# A quantized frozen base (QLoRA), bytes per quantized weight. 4-bit: half a byte plus one fp32
# scale per 64-weight block. With nested ("double") quantization the scales cost ~0.05 B/weight
# less, so this overstates slightly, the safe side. 8-bit: one byte; per-row scales are negligible.
_QUANT_BYTES = {"4bit": 0.5 + 4 / 64, "8bit": 1.0}

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
    flops_per_token: float = None                      # 6N (LoRA: 4N_total + 2N_trainable); the loop
                                                       #   multiplies by tokens/step for the floor
    lora: dict = None                                  # {r, targets, params_trainable, params_total};
                                                       #   None for a full fine-tune
    activation_headroom_gb: float = None               # mem - model_state (room for activations)
    model_state_fits: bool = None                      # does MODEL STATE alone fit — says nothing
                                                       #   about activations; see the headroom
    max_len: int = None                                # human-declared sequence length (off-limits)
    images: dict = None                                # VLM worst case: max_image_size, max_images,
                                                       #   image_tokens_per_row (frame/images.py)
    ceiling: dict = None                               # MEASURED micro_batch ceiling at max_len;
                                                       #   None when it wasn't measured
    notes: list = field(default_factory=list)          # things the agent must see, not errors
    summary: str = ""


# max_len this many times the dataset's p99 means most of every batch is padding — worth the
# agent's attention, never an error: it's the human's call.
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
            f"unpacked batch is padding. Dynamic padding / padding-free batching recover it; "
            f"packing would too, but it changes examples per optimizer step, so it's off-limits "
            f"to the loop. Or max_len is set higher than this data needs — confirm with the human.")
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


def build(model, dataset, hardware, versions, dtype=None, optimizer="adamw",
          effective_batch=None, knobs=None, max_len=None, ceiling=None, lora=None,
          base_quant=None, compute_dtype=None, images=None):
    """model/dataset/hardware/versions: the extracted parts (see the sibling modules).

    dtype: the dtype the trained weights are held in, as the training code holds them (never
    defined by Frame: the script's, else the checkpoint's). Drives the memory model.
    compute_dtype: the dtype the matmuls run in when it differs (autocast); the peak must have
    been measured at it. None: same as dtype.
    effective_batch: the human's target (carried, not searched). knobs: the framework's searchable
    surface, so the agent classifies safe/off-limits at Frame from what's actually tunable.
    max_len: the human-declared sequence length. Validated against the architecture here.
    ceiling: the record from `frame.ceiling` at max_len, or None if not measured.
    lora: {"r": rank, "targets": module names or "all-linear"} when the run trains LoRA adapters
    on a frozen base; None for a full fine-tune.
    base_quant: "4bit" / "8bit" when the script loads that frozen base with quantized weights
    (QLoRA). Its linear weights are then charged at `_QUANT_BYTES`; embeddings, norms and the
    output head stay at `dtype`.
    """
    if base_quant is not None and base_quant not in _QUANT_BYTES:
        raise ValueError(f"base_quant must be one of {sorted(_QUANT_BYTES)}, got {base_quant!r}")
    if base_quant and not lora:
        raise ValueError("a quantized base can't be fully fine-tuned: pass the script's LoRA "
                         "rank and targets too")
    dtype = dtype or model.get("dtype") or "float32"   # the code's, else the checkpoint's
    peak_dtype = hardware.get("peak_dtype")
    run_compute = compute_dtype or dtype
    if peak_dtype is not None and peak_dtype != run_compute:   # a peak at another precision is
        raise ValueError(                               # not a ceiling for THIS run
            f"peak was measured at {peak_dtype!r} but the run computes in {run_compute!r}; "
            "MFU against a mismatched peak is meaningless")

    ctx = model.get("context_len")
    if max_len is not None and ctx and max_len > ctx:    # caught here, not as a crash mid-run
        raise ValueError(
            f"max_len {max_len} exceeds the model's context length {ctx}; the architecture "
            "cannot take sequences that long")

    params = model["params"]
    param_bytes = _DTYPE_BYTES.get(dtype, 2)
    lora_rec, extra_notes = None, []
    if lora and "lora_trainable" in model:              # matched on the built model, as peft does
        trainable = model["lora_trainable"]
    elif lora:                                          # template estimate: llama-style names only
        trainable = lora_params(model, lora["r"], lora["targets"])
    else:
        trainable = None
    if str(model.get("params_source", "")).startswith("estimated"):
        extra_notes.append(
            f"params are an estimate from the config's dims (a llama-shaped template), because "
            f"the model couldn't be built: {model['params_source']}. Gated activations it can't "
            f"see, hybrid layers, fused projections and vision towers make it wrong.")
    if lora:
        lora_rec = {"r": lora["r"], "targets": lora["targets"], "params_trainable": trainable,
                    "params_total": params + trainable if trainable else None}
    if trainable:                                       # base + adapters run forward and backward;
        params += trainable                             # only the adapters get weight grads + state
        quantized = 0
        if base_quant:                                  # the base's linear weights, quantized
            quantized = model.get("quantizable_params")
            if quantized is None:                       # template fallback: all but the embeddings
                quantized = model["params"] - model["vocab"] * model["hidden"] * (1 if model.get("tied", True) else 2)
        mem = training_memory(params, param_bytes=param_bytes, optimizer=optimizer,
                              trainable=trainable, trainable_bytes=_LORA_ADAPTER_BYTES,
                              quantized=quantized, quant_bytes=_QUANT_BYTES.get(base_quant))
        flops_per_token = 4.0 * params + 2.0 * trainable
    else:
        mem = training_memory(params, param_bytes=param_bytes, optimizer=optimizer)
        flops_per_token = 6.0 * params
        if lora:                                        # declared, but not countable from the config
            why = model.get("lora_note") or ("unknown module names, an ungated MLP, or MLP "
                                             "targets on an MoE")
            extra_notes.append(
                f"LoRA targets {lora['targets']} can't be counted from this config ({why}), so the "
                f"model-state memory and FLOPs above are FULL fine-tune numbers: they overstate "
                f"memory for this run. Trust the measured ceiling.")

    mem_gb = hardware.get("mem_gb")
    headroom = round(mem_gb - mem["model_state_gb"], 1) if mem_gb is not None else None
    fits = headroom > 0 if headroom is not None else None
    if images and images.get("image_tokens_per_row") and max_len \
            and images["image_tokens_per_row"] >= max_len:
        raise ValueError(
            f"{images['max_images']} image(s) of {images['max_image_size']} make "
            f"{images['image_tokens_per_row']} image tokens, more than max_len {max_len}: no text "
            f"fits; the declared image size / count and max_len disagree")
    if model.get("modality") == "multimodal":           # final_boss_report VLM theme
        counted = (f"params include the vision tower ({model['vision_params'] / 1e6:.0f}M), but "
                   f"flops_per_token charges it per text token, while it runs per image: the FLOP "
                   f"estimate is approximate" if model.get("vision_params") else
                   "params, flops_per_token and model-state memory count the text tower only; the "
                   "vision tower isn't counted")
        rows = (f"Ceiling rows carry {images['max_images']} image(s) of "
                f"{images['max_image_size'][0]}x{images['max_image_size'][1]} each"
                + (f" ({images['image_tokens_per_row']} image tokens a row, checked every trial)"
                   if images.get("image_tokens_per_row") else
                   " (image tokens unchecked: the processor couldn't count them here)")
                if images else
                "Ceiling rows are text-only: if the vision tower trains (e.g. LoRA all-linear), "
                "declare the largest image (--max-image-size WxH) so they carry it, or the "
                "ceiling can read 2x too high")
        extra_notes.append(f"multimodal checkpoint: {counted}. {rows}.")

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
               f"{('QLoRA ' + base_quant + ' base') if trainable and base_quant else 'LoRA' if trainable else 'full-FT'} "
               f"model-state {mem['model_state_gb']}GB {fit_str} | "
               f"{_ceiling_str(max_len, ceiling)} | {ds_str}")

    return Profile(
        model=model, dataset=dataset, hardware=hardware, versions=versions, memory=mem,
        default_config={"dtype": dtype, "param_bytes": param_bytes, "optimizer": optimizer,
                        "finetune": "lora" if trainable else "full",
                        **({"base_quant": base_quant} if base_quant else {}),
                        **({"compute_dtype": compute_dtype}
                           if compute_dtype and compute_dtype != dtype else {})},
        effective_batch=effective_batch, knobs=_bound_knobs(knobs, ceiling),
        flops_per_token=flops_per_token, lora=lora_rec, activation_headroom_gb=headroom,
        model_state_fits=fits, max_len=max_len, ceiling=ceiling, images=images,
        notes=_notes(dataset, max_len, ceiling) + extra_notes, summary=summary,
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


def frame(model, dataset, gpu, max_len, dtype=None, device="cuda", framework="hf",
          effective_batch=16, text_field="text", split="train", sample=1000, name=None,
          runner=None, seed=0, on_trial=None, sample_path=None, sample_seed=0, max_scan=None,
          ceiling_rows_path=None, ceiling_row_template=None, ceiling_rows_format="text",
          lora=None, base_quant=None, compute_dtype=None, max_image_size=None, max_images=None,
          image_field=None):
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
    lora: {"r", "targets"} read from a user's LoRA script (see `build`).
    base_quant: "4bit" / "8bit" when that script loads its frozen base quantized (see `build`).
    max_image_size / max_images / image_field: a VLM's image worst case (`frame/images.py`):
    (W, H) from the training code's image cap; images per sample from the code, else the most in
    any row of `image_field` (found automatically when None).
    dtype / compute_dtype: only what the training code sets (see `build`). Never defined here:
    dtype None is the checkpoint's own (`config.json`), which is what transformers loads when the
    code names none; compute_dtype None is dtype.
    """
    from frame.hardware import hardware, versions, measure
    from transformers import AutoTokenizer
    from frame.model import load
    from frame.dataset import draw_sample, write_sample

    mi = load(model, lora=lora)                          # exact counts when it builds (meta device)
    dtype = dtype or mi.get("dtype") or "float32"       # the checkpoint's, else torch's default
    compute_dtype = compute_dtype or dtype
    m = measure(device, compute_dtype)                   # the peak ONCE, at the matmuls' dtype
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    drawn = draw_sample(dataset, tok, sample, text_field=text_field, split=split, name=name,
                        seed=sample_seed, effective_batch=effective_batch, max_len=max_len,
                        max_scan=max_scan)
    if sample_path:
        write_sample(sample_path, drawn)
    ds = drawn["meta"]["pool"]                          # the corpus's shape, not the head's
    hw = hardware(gpu, peak_flops=m["peak_flops"], mem_gb=m["mem_gb"], dtype=compute_dtype,
                  device_name=m["device_name"], num_sms=m["num_sms"],
                  peak_mem_bw_gb_s=m["peak_mem_bw_gb_s"])

    images = None
    if max_image_size:                                  # a VLM's image worst case
        from frame.images import image_tokens, scan_images
        if mi.get("modality") != "multimodal":
            raise ValueError(f"--max-image-size is for a multimodal checkpoint; {model} is text-only")
        src = "declared"
        if max_images is None:                          # not in the code: the data says
            s = scan_images(dataset, name=name, split=split, image_field=image_field,
                            max_scan=max_scan)
            if not s["max"]:
                raise ValueError(
                    "couldn't read images per sample from the data (no image column found, or no "
                    "row holds an image); pass the training code's limit as --max-images, or the "
                    "column as --image-field")
            max_images = s["max"]
            src = f"data scan: most images in any of {s['rows']} rows of `{s['field']}`"
        per_row, tid = image_tokens(model, tuple(max_image_size), max_images)
        images = {"max_image_size": list(max_image_size), "max_images": max_images,
                  "max_images_source": src, "image_tokens_per_row": per_row, "image_token_id": tid}

    if ceiling_rows_path:                               # a user's script reads these in a trial
        from frame.ceiling_rows import write_ceiling_rows
        image = None
        if images:                                      # beside the rows, so it ships with them
            import os
            from frame.images import write_image
            w, h = images["max_image_size"]
            images["ceiling_image"] = write_image(
                os.path.join(os.path.dirname(ceiling_rows_path), f"ceiling_image_{w}x{h}.png"), (w, h))
            image = (images["ceiling_image"], images["max_images"])
        write_ceiling_rows(ceiling_rows_path, tok, max_len, effective_batch * 3,
                           text_field=text_field, template=ceiling_row_template,
                           fmt=ceiling_rows_format, seed=seed, image=image)

    ceiling = None
    if runner is not None and device == "cuda":
        from frame.ceiling import find_ceiling, framework_defaults
        ceiling = find_ceiling(                         # memory-only trials: no peaks re-measured
            runner, {"device": device, "dtype": dtype, "model": model, "framework": framework},
            seq_len=max_len, vocab=mi["vocab"], effective_batch=effective_batch, seed=seed,
            on_trial=on_trial, defaults=framework_defaults(framework))

    return build(mi, ds, hw, versions(), dtype=dtype,
                 effective_batch=effective_batch, knobs=_framework_knobs(framework),
                 max_len=max_len, ceiling=ceiling, lora=lora, base_quant=base_quant,
                 compute_dtype=compute_dtype, images=images)
