"""Wire the search loop to the real Modal probe.   Run:  modal run run_search.py::loop

Drives search() against modal_probe.probe (one real GPU probe per config), journaling to a state
file so an interrupted run resumes for free. Uses the dev model + synthetic data by default — swap
DEV_BASE for the real model/dataset when work access lands.

UNVERIFIED end-to-end (needs a deploy). Note: with the tiny dev model + synthetic data the search
won't OOM or saturate, so this validates the *wiring*, not a meaningful optimization.
"""
from modal_probe import app, probe
from search import search, SweepPolicy, PredictivePolicy

# Fixed fields the policy doesn't search — merged into every probed config.
DEV_BASE = {
    "model": "Qwen/Qwen3.5-0.8B",        # the production-target model
    "dataset": "synthetic",
    "max_steps": 7,
    "attn_impl": "sdpa",
}


@app.local_entrypoint()
def loop(effective_batch: int = 16, policy: str = "predictive"):
    pol = PredictivePolicy(effective_batch) if policy == "predictive" else SweepPolicy(effective_batch)
    out = search(lambda cfg: probe.remote(cfg), pol,
                 base_config=DEV_BASE, state_path="loop_state.json")
    print(f"\nprobes: {out['n_probes']}")
    for h in out["history"]:
        c, r = h["config"], h["result"]
        print(f"  bs={c['per_device_train_batch_size']:<3} pf={int(c.get('padding_free', 0))} "
              f"gc={int(c.get('gradient_checkpointing', 0))} -> {r['status']:<4} tok/s={r.get('tokens_per_s')}")
    if out["best"]:
        b = out["best"]["config"]
        print(f"\nbest: bs={b['per_device_train_batch_size']} pf={int(b.get('padding_free', 0))} "
              f"gc={int(b.get('gradient_checkpointing', 0))} tok/s={out['best']['result'].get('tokens_per_s')}")
