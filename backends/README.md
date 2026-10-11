# backends/

**Where a probe runs.** One contract — **`run(config) -> Observation`** (a dict) — with swappable
implementations picked by `config["backend"]`. The backend runs a single probe (build the runner →
`capture` → `diagnose`) and hands back the Observation the agent reads. We assume probe hardware =
training hardware, so measurements transfer by definition.

```
loop ──▶ backends.run(config) ──▶ [ build runner → capture → diagnose ] ──▶ Observation
                 │
   config["backend"] routes:  local (in-process)  ·  modal (cloud GPU)  ·  ssh (your own box)
```

**Single GPU only for now** — a box/cluster can host many, but `frame/` and the loop only profile
and tune one at a time. Multi-GPU parallelism is a planned search dimension, not yet built.

---

## Setup checklist (`program.md` Step 1)

Do this before anything else in a session — nothing downstream works without a place to run.

Ask the user where the experiments will actually run:

- **Local** (`backend = "local"`, `device = "cuda"`) — the probe runs in-process on this machine's
  device. What you use when you're already on the GPU box, including a rented one.
- **Modal** (`backend = "modal"`) — serverless cloud GPU. Needs the `modal` package and auth
  (`modal setup`, or `MODAL_TOKEN_ID`/`MODAL_TOKEN_SECRET` in `.env`). The GPU is taken from
  `run.toml`'s `gpu` field (e.g. `A10G`, `A100`, `H100`), and the repo is shipped to the worker
  automatically — nothing to provision.
- **VM / remote box over SSH** (`backend = "ssh"`) — any rented box / neocloud / your own
  cluster. Built on the system `ssh`/`rsync` binaries, so there's no new Python dependency and your
  existing `~/.ssh/config`, agent and known_hosts keep working. You'll need host/user, and the
  credential must come from the env file, never inline in chat or in a tracked file:
  - `AUTOTUNE_SSH_KEY` (path to a private key) or `AUTOTUNE_SSH_PASSWORD` (password auth, needs
    the `sshpass` binary), set in `.env` (gitignored — see `.env.example` for the template). Set
    **neither** if the box is already reachable via your ssh agent/config — `ssh.py` falls through
    to the system ssh's own credential resolution.
  - The box needs a Python with `torch` + `transformers` + `datasets`; point `[ssh] python` at it
    (default `~/autotune/.venv_at/bin/python`). The repo is rsynced to `[ssh] remote_dir`
    (default `~/autotune`) on every probe — it's ~700KB, so the sync is not the cost. `[ssh] sync`
    (default `true`) turns that code sync off; script files under `runs/` are still pushed.
  - Everything non-secret (backend, model, dataset, gpu, `effective_batch`, SSH host/user)
    belongs in `run.toml` (copy from the committed `run.toml.example`), not `.env`.
  - Confirm you can actually reach the box (e.g. a trivial `ssh` command succeeds) before moving on.

Then confirm the device. `device` is a decision, not a detail: with `backend = "local"` it defaults
to `cuda` and falls back to `cpu` only when torch reports no CUDA — loudly, on stderr and in the
Observation's `device_type`. A `cpu` probe returns a complete, well-formed Observation whose MFU and
GPU signals are meaningless, so it is a smoke test you ask for deliberately. Never rank a `cpu` row
against a `cuda` row, and never promote a belief from one. Single-GPU only, so `cuda:0` is refused:
pick the GPU with `CUDA_VISIBLE_DEVICES`.

Determine the run name — ask the user for one. IF they decline, default to
`<model_name>_<dataset>` (from `run.toml`'s `model`/`dataset` fields, `/` sanitized to `-`, e.g.
`Qwen2.5-0.5B_wikitext`). Create `runs/<run_name>/` now — Step 2 (`frame/README.md`) needs
somewhere to write `research.md` before Step 3 exists.

Also skim `README.md` (repo context: what's built, what's scaffold, the overall plan) and
`profiler/README.md` (the measurement instrument) — you'll be reading its output every iteration
of the loop, so understand what it measures and how before you start interpreting it.

---

## Use it

```python
from backends import run

obs = run({
    "backend": "modal", "gpu": "A10G",          # where + which GPU
    "framework": "hf", "model": "Qwen/Qwen2.5-0.5B",
    "dataset": "Salesforce/wikitext", "name": "wikitext-2-raw-v1",
    "effective_batch": 16, "micro_batch": 4, "seq_len": 1024,   # no dtype: the checkpoint's
    "peak_flops": 1.21e14, "peak_mem_bw_gb_s": 428.0,   # frozen ceilings from Frame
})
print(obs["summary"])
```

## The config

Everything a probe needs, in one dict: **run params** (`framework`, `model`, `dataset`, `name`,
`dtype`, `seq_len`, `effective_batch`, `sample_file`, `gpu`, `device`), the **knob assignment** for
this probe (`micro_batch`, `gradient_checkpointing`, `attn_implementation`, `torch_compile`, `tf32`,
`fused_optimizer`, `padding`, `length_grouped`, `chunked_ce`; see `frameworks/README.md`), and the
**frozen ceilings** the loop measured once at Frame (`peak_flops`, `peak_mem_bw_gb_s`) so every probe
is scored against the same denominator (not a per-probe re-measure).

- `framework` picks the runner (`hf` real, `torch` the MLP fixture). It defaults to **`torch`**, so
  a real probe must set `framework: "hf"`.
- `warmup` / `clean_steps` (default 3 / 10) set the timing window.
- `timeout` (seconds) kills a hung probe and returns an `error` Observation. Default 1800 for a
  subprocess (script probes and their ceiling trials; the ssh connection waits `timeout` + 300 so the
  box's answer wins); modal's function timeout defaults to 600. An in-process runner probe on
  `local` has none. `program.md` sets `timeout: 600` in every probe config.
- `env` sets environment variables before the probe process starts (e.g.
  `PYTORCH_CUDA_ALLOC_CONF`): on the box for ssh, and for a local script probe's process.

## Script probes (bring your own training script)

Set `script` (and usually `script_args`) in the config, and the backend runs the **user's training
script unchanged** instead of a `frameworks/` runner:

```
backends/local.py: run_script → python -m profiler.launch --out cap.json -- <script> <script_args>
                              → the script runs in its own process, measured from the outside
                              → cap.json → diagnose() → the same Observation a runner probe returns
```

ssh and modal reach this through `backends.local.run` on the box, so all three backends support it.

**Ceiling trials** use the same path with `probe_kind: "ceiling_trial"` (built by `frame.ceiling`,
never by hand): a memory-only run of `trial_steps` optimizer steps at `micro_batch`, returning
`{"status": ok|oom|error, "peak_gb", "reserved_gb", ...}` instead of an Observation. For our runner
it's worst-case synthetic batches; for a script, `profiler.launch --memory-only` on the ceiling
rows, with every batch checked to be `[micro_batch, max_len]` (`frame/README.md`, "User scripts").

```python
from backends import run

obs = run({
    "backend": "ssh", "ssh": {"host": "gpu-box", "user": "ubuntu"}, "gpu": "A10",
    "script": "runs/my_run/train.py",                        # repo-relative: same path on the box
    "script_args": ["--config", "runs/my_run/train.yaml",    # the user's baseline flags...
                    "--max_steps", "16"],                    # ...+ a step budget >= warmup + clean + 1
    "model": "Qwen/Qwen2.5-0.5B",                             # the script's checkpoint: its dtype
    "warmup": 3, "clean_steps": 10,                           # is the peak's unless the script sets
                                                              # `dtype` / `compute_dtype` (then pass them)
    "peak_flops": 8.9e13, "peak_mem_bw_gb_s": 445.0,         # frozen ceilings from Frame
})
print(obs["summary"], obs["population"]["driver"])           # ... "script"
```

| Key | Meaning |
|---|---|
| `script` | the training script. Absolute or repo-relative for `local`; **repo-relative for `ssh`**, which runs the same path after the code sync. A script under `runs/` (excluded from the sync) is pushed like `sample_file` |
| `script_args` | its arguments: the baseline flags, plus the step budget and knob changes the agent sets per probe |
| `script_files` | other repo-relative files the script reads, e.g. a per-probe config copy (`runs/<run>/probes/exp3.yaml`). Checked before launch; the ssh backend pushes any under `runs/` |
| `script_cwd` | working directory for the script (default: the repo root) |
| `python` | interpreter for the probe process (default: the current one) |
| `warmup`, `clean_steps`, `peak_flops`, `peak_mem_bw_gb_s`, `timeout`, `env` | as for runner probes |
| `dtype`, `compute_dtype` | only what the script sets: its weights' load dtype, and its matmul dtype when it autocasts. Neither set: the peak is measured at the checkpoint's dtype (`model`), what the script loads when it names none |

**What the Observation says about itself:**
- `population.driver: "script"`;
- `clean_window: "fixed-position"` (not whole passes over Frame's sample);
- `trace_basis: "traced-step"` (one step of the script's own loop, so `gpu_active_frac` / `gap_*`
  are null);
- the `script` and `script_args` it ran;
- the micro-batch count seen per step;
- any `script_error` raised after the measurement finished.

Statuses match runner probes:
- a CUDA OOM → `oom`;
- another exception → `error` with its type and first line;
- a killed process (no result written) → `error: launcher wrote no result`;
- a step budget too small → `error: script ended after N optimizer steps; needs M`.

## Backends

| Backend | What | Status |
|---|---|---|
| **local** | in-process on this machine's device — dev, or a box you're already on | built; runs any runner (fixture on CPU, or a real GPU) |
| **modal** | serverless GPU: ships the repo to Modal, runs on the requested GPU, returns the Observation. `config["gpu"]` is the GPU request | built (Modal 1.5); **live-validated end-to-end on an A10G** |
| **ssh** | any rented box / neocloud (Lambda, RunPod, …) or your own cluster | built (system ssh/rsync, no new deps); **live-validated end-to-end on an A10** |

## Adding a runner

`local` maps `config["framework"]` → a runner class in `_RUNNERS` (`torch` fixture, `hf` real).
Add a line per adapter. A declared framework must implement the runner interface
(`setup`/`step`/`tokens_per_step` + `.params`) — see `frameworks/`.

## Layout

- `__init__.py` — the `run(config)` dispatch (routes by `config["backend"]`).
- `local.py` — in-process `capture → diagnose`; the framework→runner registry.
- `modal.py` — Modal execution (lazy `import modal`).
- `ssh.py` — rsync the repo to a box, run `local.run` there over ssh, parse the Observation back
  out of stdout (sentinel-framed, since transformers/datasets share that stream).

Heavy imports (torch, modal) are lazy, so the modules import on bare `python3`; the dispatch +
registry are unit-tested there. Real execution (local on a GPU box, modal on Modal) is validated live.
