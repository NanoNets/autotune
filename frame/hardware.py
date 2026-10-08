"""Hardware + software facts for the Profile.

The MFU denominator is no longer a datasheet lookup — it's the peak FLOP/s the chip actually
delivers, MEASURED once per run at the run's dtype (`profiler.capture.measure_peak_flops`) and
then threaded to every probe. `gpu` is just a free-form label for the record; any hardware is
supported, named or not. `versions()` reads installed library versions so the loop knows which
levers exist (no flash-attn installed -> that knob is off).

`hardware(...)` is the pure assembler (testable without a GPU); `measure(...)` is the on-device
shell that produces the peak + memory it takes.
"""


def hardware(gpu, peak_flops, mem_gb, dtype, device_name=None, num_sms=None,
             peak_mem_bw_gb_s=None):
    """Assemble the hardware facts from already-measured numbers.

    gpu: a free-form label (NOT validated — the peak is measured, so any chip works).
    peak_flops: measured peak FLOP/s at `dtype` — the MFU denominator, measured once per run.
    mem_gb: the card's real total memory. device_name/num_sms: from the card, if known.
    peak_mem_bw_gb_s: measured achievable bandwidth — the roofline's memory ceiling, frozen for
    the run exactly like peak_flops so probes don't each re-measure it.
    """
    return {
        "gpu": gpu,                        # label for the record only
        "count": 1,                        # single-GPU for now
        "peak_flops": peak_flops,          # measured on-device at `dtype`
        "peak_dtype": dtype,               # the dtype the peak was measured at
        "peak_mem_bw_gb_s": peak_mem_bw_gb_s,   # measured; frozen for the run like peak_flops
        "mem_gb": mem_gb,                  # real card total
        "device_name": device_name,
        "num_sms": num_sms,
    }


def measure(device, dtype):
    """On-device: measure BOTH roofline ceilings + read the card's facts. Needs torch + the GPU.

    Both ceilings are measured once here and threaded to every probe
    (`capture(..., peak_flops=..., peak_mem_bw_gb_s=...)`) — they describe the hardware, not the
    config, and re-measuring per probe re-pays the cost and adds ~28% noise to MFU comparisons.

    dtype is a torch dtype name ('bfloat16' / 'float16' / 'float32') — the peak MUST be measured at
    the dtype the model runs in (fp32 model vs a bf16 peak = MFU wrong by ~10x).
    """
    import torch
    from profiler.capture import measure_peak_flops, measure_peak_bandwidth_gb_s, device_facts

    return {"peak_flops": measure_peak_flops(device, getattr(torch, dtype)),
            "peak_mem_bw_gb_s": measure_peak_bandwidth_gb_s(device),
            **device_facts(device)}


def versions():
    """Installed versions of the libraries whose presence gates knobs. None if absent."""
    def _v(mod):
        try:
            return __import__(mod).__version__
        except Exception:
            return None
    return {"torch": _v("torch"), "transformers": _v("transformers"),
            "flash_attn": _v("flash_attn")}
