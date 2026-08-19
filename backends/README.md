# backends/ &mdash; *(scaffold, not yet built)*

Where a probe runs. One contract — `run(config) -> Observation` — with swappable implementations.

- **modal** — serverless GPU: elastic, on-demand, good for wide parallel probing
- **ssh** — any rented box / neocloud (Lambda, CoreWeave, RunPod, …) or your own cluster

We assume probe hardware = training hardware, so measurements transfer by definition.
