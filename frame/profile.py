"""Assemble the Profile: the once-per-run framing the agent reads.

`build(...)` is pure (takes the already-extracted parts), so the assembly + memory-fit logic
is testable without transformers/datasets/GPU. `frame(model, dataset, gpu)` is the live shell:
measure the peak, load the HF config + tokenizer, scan the dataset, then build.

The peak FLOP/s (MFU denominator) is MEASURED once here at the run dtype and carried in the
Profile; the loop threads it to every probe via `capture(..., peak_flops=...)`. Measuring it
once matters: on cloud GPUs the measured peak swings ~28% run-to-run while the step-time noise
floor is ~1%, so if each probe measured its own, probe-to-probe MFU would be mostly noise.

`model_state_fits` covers the model state ONLY — weights/grads/optimizer. Activations are not in
it, so a positive `activation_headroom_gb` is what actually decides whether a config runs.

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
    flops_per_token: float = None                      # 6N; loop multiplies by tokens/step for the floor
    activation_headroom_gb: float = None               # mem - model_state (room for activations)
    model_state_fits: bool = None                      # does MODEL STATE alone fit — says nothing
                                                       #   about activations; see the headroom
    summary: str = ""


def build(model, dataset, hardware, versions, dtype="bfloat16", optimizer="adamw"):
    """model/dataset/hardware/versions: the extracted parts (see the sibling modules).

    dtype: the run compute dtype — drives the memory precision AND is the dtype the peak was
    measured at (carried in default_config so the loop passes the matching config["dtype"]).
    """
    peak_dtype = hardware.get("peak_dtype")
    if peak_dtype is not None and peak_dtype != dtype:  # a peak measured at another precision is
        raise ValueError(                               # not a ceiling for THIS run
            f"peak was measured at {peak_dtype!r} but the run dtype is {dtype!r}; "
            "MFU against a mismatched peak is meaningless")

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
    summary = (f"{name} ({model['attention']}, {dtype}) | {params/1e9:.1f}B params | "
               f"{hardware.get('gpu')} {mem_str}{peak_str} | "
               f"full-FT model-state {mem['model_state_gb']}GB {fit_str} | {ds_str}")

    return Profile(
        model=model, dataset=dataset, hardware=hardware, versions=versions, memory=mem,
        default_config={"dtype": dtype, "param_bytes": param_bytes, "optimizer": optimizer},
        flops_per_token=6.0 * params, activation_headroom_gb=headroom,
        model_state_fits=fits, summary=summary,
    )


def frame(model, dataset, gpu, dtype="bfloat16", device="cuda",
          text_field="text", split="train", sample=1000, name=None):
    """Live: measure the peak, load config + tokenizer, scan the dataset, assemble.

    Needs torch + a GPU (peak), transformers + datasets (model/tokenizer/dataset). `gpu` is a
    free-form label. Measuring the peak first also fails fast if there's no usable device.
    """
    from frame.hardware import hardware, versions, measure
    m = measure(device, dtype)                          # measure the peak ONCE, at the run dtype
    from transformers import AutoTokenizer
    from frame.model import load
    from frame.dataset import scan

    mi = load(model)
    tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    ds = scan(dataset, tok, text_field=text_field, split=split, sample=sample, name=name)
    hw = hardware(gpu, peak_flops=m["peak_flops"], mem_gb=m["mem_gb"], dtype=dtype,
                  device_name=m["device_name"], num_sms=m["num_sms"],
                  peak_mem_bw_gb_s=m["peak_mem_bw_gb_s"])
    return build(mi, ds, hw, versions(), dtype=dtype)
