"""backends/ — where a probe runs.

One contract: **`run(config) -> Observation`** (as a dict). The config carries the run params +
the knob assignments + the frozen roofline ceilings (`peak_flops` / `peak_mem_bw_gb_s`) the loop
measured once at Frame; the backend runs one probe and hands back the Observation the agent reads.
Swappable implementations, picked by `config["backend"]`:

- **local** — in-process on this machine's device (dev, or a box you're already on).
- **modal** — serverless GPU: elastic, on-demand (the default for cloud probing).
- **ssh**  — any rented box / neocloud / your own cluster (system ssh/rsync, no new deps).

We assume probe hardware = training hardware, so measurements transfer by definition.
"""

_BACKENDS = {"local", "modal", "ssh"}


def run(config):
    """Run one probe on `config["backend"]` (default "local") and return its Observation (a dict)."""
    backend = config.get("backend", "local")
    if backend not in _BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; known: {sorted(_BACKENDS)}")
    import importlib
    return importlib.import_module(f"backends.{backend}").run(config)
