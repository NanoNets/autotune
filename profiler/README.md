# profiler/

The measurement instrument. Profiles a single training step and turns it into an **Observation** —
the ~30-field diagnostic the agent reads. Framework-agnostic (it's `torch.profiler` underneath),
and validated live on an A10G.

## Run it

```bash
python -m profiler --device cpu     # fixture MLP on CPU (pipeline smoke)
python -m profiler --device cuda    # on a GPU (real kernels / memory / occupancy)
```

Prints the one-line summary + the full Observation JSON.

## How it works

Profiles one step, **start → start of the next** (so it catches data-wait too), in two tiers:
- **Clean timing** (profiler OFF) → the real wall-clock: median + noise floor over N steps. The
  profiler distorts timing, so the real number is measured with it off.
- **One profiled step** (`torch.profiler`) → attribution: phases, kernels, occupancy.

Then it computes the **roofline** (peak FLOPs for the GPU) and reports **MFU = achieved / peak** — the
"is the GPU utilized well" signal (util% lies; MFU doesn't).

## The Observation (~30 fields)

- **objective** — `step_time_s`, `step_time_cv` (noise), `mfu`, `roofline_floor_s`, `flops`
- **utilization** — `gpu_active_frac`, `achieved_occupancy_pct` + `limiting_factor`, `cpu_util_mean/max`, `nproc`
- **time breakdown** — `phase_times_s` (fwd/bwd/opt/dataloader), `data_wait_s/frac`, `gap_s/frac` (idle)
- **memory** — `peak / reserved / total / headroom` (torch.cuda)
- **kernels** — `top_kernels`, `kernel_count`, `top8_coverage`, `gap_count`, `max_gap_s` (idle shape), `provenance`
- **hardware** — `device` (name/mem/cc/sms, read from the trace header)
- **status** — `ok / oom / error`, `error`, `trace_path`

The agent reads this compact object — **never the raw trace.** `trace_path` points to the raw Chrome
trace for on-demand drill-down (open it in <https://ui.perfetto.dev>).

## Layout

- `roofline.py` — FLOPs (6ND) + GPU specs → step-time floor, MFU.
- `trace.py` — parse a torch.profiler Chrome trace → phases, kernels, gaps, occupancy, device props, provenance.
- `capture.py` — run the step: clean timing + CPU sampler (psutil) + memory + OOM handling + one profiled step. *(needs torch)*
- `diagnostic.py` — assemble the Observation (torch-free; the `Observation` dataclass lives here).
- `__main__.py` — the CLI (`python -m profiler`).

## Adapters

`capture` drives a **runner** — anything with `setup(config)` + `step()`. `frameworks/torch.py` is a
minimal fixture (an MLP on random data → *shape*, not real numbers). A real framework adapter
(`frameworks/swift.py` / HF) profiles a real model; the profiler engine is unchanged.

## Status

Engine complete and **GPU-validated** (A10G): every field populates, the OOM path fires, the CLI runs
clean. Deferred: memory breakdown (→ Frame, analytical), per-op FLOPs (not in the trace export),
GPU-time-per-phase (via correlation ids — a refinement).
