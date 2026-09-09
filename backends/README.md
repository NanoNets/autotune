# backends/ &mdash; *(scaffold, not yet built)*

Where a probe runs. One contract — `run(config) -> Observation` — with swappable implementations.

- **modal** — serverless GPU: elastic, on-demand, good for wide parallel probing
- **ssh** — any rented box / neocloud (Lambda, CoreWeave, RunPod, …) or your own cluster

We assume probe hardware = training hardware, so measurements transfer by definition.

**Single GPU only for now** — a box/cluster can host many, but `frame/` and the loop only profile
and tune one at a time. Multi-GPU parallelism is a planned search dimension, not yet built.

---

## Setup checklist (`program.md` Step 1)

Do this before anything else in a session — nothing downstream works without a place to run.

Ask the user where the experiments will actually run:

- **Local** — training runs on this machine.
- **VM / remote box** (Paperspace, Lambda, RunPod, your own cluster, ...) — you'll need SSH access.
  Ask for host/user, and get the credential via the env file, never inline in chat or in a tracked
  file:
  - `AUTOTUNE_SSH_PASSWORD` (password auth) or `AUTOTUNE_SSH_KEY` (path to a private key), set in
    `.env` (gitignored — see `.env.example` for the template).
  - Everything non-secret (backend, model, dataset, gpu, `effective_batch`, SSH host/user)
    belongs in `run.toml` (copy from the committed `run.toml.example`), not `.env`.
- Confirm you can actually reach the box (e.g. a trivial `ssh` command succeeds) before moving on.

Determine the run name — ask the user for one. IF they decline, default to
`<model_name>_<dataset>` (from `run.toml`'s `model`/`dataset` fields, `/` sanitized to `-`, e.g.
`Qwen2.5-0.5B_wikitext`). Create `runs/<run_name>/` now — Step 2 (`frame/README.md`) needs
somewhere to write `research.md` before Step 3 exists.

Also skim `README.md` (repo context: what's built, what's scaffold, the overall plan) and
`profiler/README.md` (the measurement instrument) — you'll be reading its output every iteration
of the loop, so understand what it measures and how before you start interpreting it.
