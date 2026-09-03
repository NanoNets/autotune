# backends/

**Where a probe runs.** One contract — **`run(config) -> Observation`** (a dict) — with swappable
implementations picked by `config["backend"]`. The backend runs a single probe (build the runner →
`capture` → `diagnose`) and hands back the Observation the agent reads. We assume probe hardware =
training hardware, so measurements transfer by definition.

```
loop ──▶ backends.run(config) ──▶ [ build runner → capture → diagnose ] ──▶ Observation
                 │
   config["backend"] routes:  local (in-process)  ·  modal (cloud GPU)  ·  ssh (deferred)
```

## Use it

```python
from backends import run

obs = run({
    "backend": "modal", "gpu": "A10G",          # where + which GPU
    "framework": "hf", "model": "Qwen/Qwen2.5-0.5B",
    "dataset": "Salesforce/wikitext", "name": "wikitext-2-raw-v1",
    "dtype": "bfloat16", "effective_batch": 16, "micro_batch": 4, "seq_len": 1024,
    "peak_flops": 1.21e14, "peak_mem_bw_gb_s": 428.0,   # frozen ceilings from Frame
})
print(obs["summary"])
```

## The config

Everything a probe needs, in one dict: **run params** (`model`, `dataset`, `name`, `dtype`,
`seq_len`, `gpu`, `device`), the **knob assignment** for this probe (`micro_batch`,
`gradient_checkpointing`, `attn_implementation`, `torch_compile`, `tf32`, `effective_batch`), and the
**frozen ceilings** the loop measured once at Frame (`peak_flops`, `peak_mem_bw_gb_s`) so every probe
is scored against the same denominator (not a per-probe re-measure).

## Backends

| Backend | What | Status |
|---|---|---|
| **local** | in-process on this machine's device — dev, or a box you're already on | built; runs any runner (fixture on CPU, or a real GPU) |
| **modal** | serverless GPU: ships the repo to Modal, runs on the requested GPU, returns the Observation. `config["gpu"]` is the GPU request | built (Modal 1.5); **live-validated end-to-end on an A10G** |
| **ssh** | any rented box / neocloud (Lambda, RunPod, …) or your own cluster | deferred |

## Adding a runner

`local` maps `config["framework"]` → a runner class in `_RUNNERS` (`torch` fixture, `hf` real).
Add a line per adapter. A declared framework must implement the runner interface
(`setup`/`step`/`tokens_per_step` + `.params`) — see `frameworks/`.

## Layout

- `__init__.py` — the `run(config)` dispatch (routes by `config["backend"]`).
- `local.py` — in-process `capture → diagnose`; the framework→runner registry.
- `modal.py` — Modal execution (lazy `import modal`).

Heavy imports (torch, modal) are lazy, so the modules import on bare `python3`; the dispatch +
registry are unit-tested there. Real execution (local on a GPU box, modal on Modal) is validated live.
