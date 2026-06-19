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
    "pf_mem": 0.6, "pf_compute": 0.7,    # padding_free: 0.6x memory, 0.7x compute (a win)
    "gc_mem": 0.4, "gc_compute": 1.5,    # gradient_checkpointing: 0.4x memory, 1.5x compute (mem cure, speed tax)
    "data_rate": 1000.0,                 # dataloader samples/s per worker (high default => compute-bound)
}

# Benchmark scenarios — distinct GPU landscapes with known (brute-forced) optima, each
# exercising a different policy path. Ground truth + regret live in test_search.py.
#   oom_pf          — OOM forces padding_free; optimum bs=8 +pf (gc fits deeper but is slower)
#   gc_wins         — memory so tight the optimum needs pf+gc at the ceiling (bigger bs beats gc tax)
#   saturates_early — compute-bound, saturates below OOM; no extension ever triggers
SIMS = {
    "oom_pf": SIM,
    "gc_wins": {**SIM, "k": 0.1},
    "saturates_early": {**SIM, "gpu_mem_gb": 1000.0, "mem_per_sample_gb": 1.0, "k": 2.0, "overhead": 0.5},
}


def probe(config, sim=SIM):
    bs = config["per_device_train_batch_size"]
    ga = config["gradient_accumulation_steps"]
    eff = bs * ga
    gc = config.get("gradient_checkpointing", False)
    pf = config.get("padding_free", False)

    mem_factor = (sim["gc_mem"] if gc else 1.0) * (sim["pf_mem"] if pf else 1.0)
    compute_factor = (sim["gc_compute"] if gc else 1.0) * (sim["pf_compute"] if pf else 1.0)

    peak = sim["base_mem_gb"] + bs * sim["mem_per_sample_gb"] * mem_factor
    cfg = {"bs": bs, "ga": ga, "effective_batch": eff,
           "gradient_checkpointing": gc, "padding_free": pf}

    if peak > sim["gpu_mem_gb"]:
        return {"status": "oom", "config": cfg, "effective_batch": eff,
                "gpu_mem_gb": sim["gpu_mem_gb"], "peak_mem_gb": round(peak, 2),
                "samples_per_s": None, "tokens_per_s": None, "gpu_util_max": None}

    micro = sim["k"] * bs * compute_factor + sim["overhead"]
    compute_sps = bs / micro                          # GPU-side throughput (ga-independent)
    workers = config.get("dataloader_num_workers", 1)
    data_sps = sim["data_rate"] * workers             # dataloader delivery rate
    sps = min(compute_sps, data_sps)                  # GPU starves if data can't keep up
    util = round(100 * sps / compute_sps, 1)          # <100 => data-bound (more workers can help)
    return {
        "status": "ok",
        "config": cfg,
        "effective_batch": eff,
        "gpu_mem_gb": sim["gpu_mem_gb"],          # total GPU memory (real telemetry the policy needs)
        "samples_per_s": round(sps, 4),
        "tokens_per_s": round(sps * sim["tokens_per_sample"], 1),
        "peak_mem_gb": round(peak, 2),
        "gpu_util_max": util,
    }
