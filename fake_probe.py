"""Local simulator probe — same contract as modal_probe.probe(config), no GPU/Modal.

Deterministic GPU model so the search loop can be built/tested offline against a known optimum.
  memory: peak = base + bs*mem_per_sample*mem_factor (ga-INDEPENDENT) -> OOM if > gpu_mem
  speed:  samples_per_s = bs/(k*bs*compute_factor + overhead) (rises & saturates; ga-INDEPENDENT)
  flags:  padding_free 0.6x mem / 0.7x compute; gradient_checkpointing 0.4x mem / 1.5x compute
Default SIM optimum: bs=8, padding_free, no gc (gc fits bs=16 but its tax makes it slower).
Swap fake_probe.probe for modal_probe.probe to run for real.
"""

SIM = {
    "gpu_mem_gb": 24.0,
    "base_mem_gb": 4.0,          # weights + optimizer state (bs-independent)
    "mem_per_sample_gb": 2.8,    # activation memory per sample
    "k": 0.5,                    # compute time per sample
    "overhead": 2.0,             # fixed per-step overhead (amortized as bs grows)
    "tokens_per_sample": 512,
}


def probe(config, sim=SIM):
    bs = config["per_device_train_batch_size"]
    ga = config["gradient_accumulation_steps"]
    eff = bs * ga
    gc = config.get("gradient_checkpointing", False)
    pf = config.get("padding_free", False)

    mem_factor = (0.4 if gc else 1.0) * (0.6 if pf else 1.0)
    compute_factor = (1.5 if gc else 1.0) * (0.7 if pf else 1.0)

    peak = sim["base_mem_gb"] + bs * sim["mem_per_sample_gb"] * mem_factor
    cfg = {"bs": bs, "ga": ga, "effective_batch": eff,
           "gradient_checkpointing": gc, "padding_free": pf}

    if peak > sim["gpu_mem_gb"]:
        return {"status": "oom", "config": cfg, "effective_batch": eff,
                "peak_mem_gb": round(peak, 2), "samples_per_s": None,
                "tokens_per_s": None, "gpu_util_max": None}

    micro = sim["k"] * bs * compute_factor + sim["overhead"]
    sps = bs / micro  # ga-independent: the corrected-metric invariant
    return {
        "status": "ok",
        "config": cfg,
        "effective_batch": eff,
        "samples_per_s": round(sps, 4),
        "tokens_per_s": round(sps * sim["tokens_per_sample"], 1),
        "peak_mem_gb": round(peak, 2),
        "gpu_util_max": round(100 * (sim["k"] * bs * compute_factor) / micro, 1),
    }
