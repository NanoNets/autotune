# profiler/

The measurement instrument. Profiles a single training step and turns it into an **Observation** —
the 40-field diagnostic the agent reads. Framework-agnostic (it's `torch.profiler` underneath),
and validated live on an A10G.

## Run it

```bash
python -m profiler --device cpu  --gpu A10G   # fixture MLP on CPU (pipeline smoke)
python -m profiler --device cuda --gpu A10G   # on a GPU (real kernels / memory / occupancy)
```

Prints the one-line summary + the full Observation JSON.

### Roofline ceilings: measure once, freeze, reuse

`peak_flops` and `peak_mem_bw_gb_s` are measured on the device (no datasheet table, so any chip
works). They describe the *hardware*, not the config, so a search must measure them **once** and
reuse them:

```python
first = capture(runner, cfg)                       # measures both, reports them in the Observation
rest  = capture(runner, cfg2, peak_flops=first.peak_flops,
                peak_mem_bw_gb_s=first.peak_mem_bw_gb_s)
```

Re-measuring per probe re-pays the cost every time and injects ~28% run-to-run variance into
comparisons whose real signal is ~1% — a frozen constant cancels out of every ranking instead.

`--gpu` is **required**, but it is only a **label** for the record — it is never looked up or priced.
The MFU denominator is measured on the device itself, so any hardware works, including chips that
shipped yesterday, and there is no table to keep correct. The trace's real `device` is reported
alongside the label for comparison.

## How it works

Profiles one step, **start → start of the next** (so it catches data-wait too), in two tiers:
- **Clean timing** (profiler OFF) → the real wall-clock: median + noise floor over N steps. The
  profiler distorts timing, so the real number is measured with it off.
- **One profiled step** (`torch.profiler`) → attribution: phases, kernels, occupancy.

Then it **measures** the device's achievable peak (a large dense GEMM at the run's dtype) and reports
**MFU = achieved / measured peak** — the "is the GPU utilized well" signal (util% lies; MFU doesn't).
The peak is measured, never read from a table: tables can't tell an A10 from an A10G, quietly quote
sparse numbers, and fail on hardware they've never heard of. Measuring supports any chip and prices
in real clocks and power caps.

## The Observation (40 fields)

A **flat JSON object** — one level deep, except four container fields (`phase_times_s`,
`provenance`, `device`, `top_kernels`). `python -m profiler` prints the one-line `summary`, then
`json.dumps(obs.__dict__, indent=2, default=str)`. Anything not measured is `null` — never 0,
so "unknown" can't be mistaken for "measured zero".

| # | Field | Meaning | `null` when |
|---|---|---|---|
| 1 | `status` | `ok` / `oom` / `error` | never |
| 2 | `gpu` | hardware **label** from `--gpu`, for the record only — it is never priced or looked up | never |
| 3 | `peak_flops` | dense-GEMM peak **measured on this device** — the MFU denominator. No datasheet table, so any hardware works and throttling is priced in | measurement failed (then `mfu` is `null` too) |
| 4 | `peak_dtype` | dtype the peak was measured at. MFU is only apples-to-apples if it matches what the model ran in | ditto |
| 5 | `peak_mem_bw_gb_s` | achievable memory bandwidth, **measured** by a large device-to-device copy — the roofline's other axis. Without it, a low MFU can't be diagnosed as compute-bound vs bandwidth-bound, and most transformer training is the latter | measurement failed |
| 6 | `step_time_s` | clean wall-clock per step, profiler off, **median** | no clean step completed |
| 7 | `step_time_cv` | noise floor: coefficient of variation across clean steps | ditto |
| 8 | `flops` | estimated training FLOPs for the step (6ND) | not `ok` |
| 9 | `roofline_floor_s` | compute-bound floor on step time at peak bf16 | not `ok` |
| 10 | `mfu` | achieved / peak — the "utilized well" signal | not `ok` |
| 11 | `gpu_busy_s` | total GPU kernel time | not `ok` |
| 12 | `gpu_active_frac` | `gpu_busy_s / step_time_s` | not `ok` |
| 13 | `profiled_step_s` | wall of the ONE profiled step. `profiled_step_s / step_time_s` is the profiler's overhead factor — exposed so the distortion is visible rather than folded into the numbers | not `ok` |
| 14 | `phase_times_s` | **every** `record_function` annotation, seconds — the four above plus whatever the runner/torch emit (`step`, `Optimizer.step#AdamW.step`, comm, …). Two traps: scopes **nest** (do not sum them), and these are **host dispatch times, not GPU time** — CUDA launches are async, so `forward` is how long the host spent *launching* forward, not how long the GPU ran it. Compare against `gpu_busy_s`, never against these | `{}` when not `ok` |
| 15 | `gpu_phase_times_s` | **GPU** time per annotation, seconds — kernels linked to the enclosing annotation by `correlation` id. This is the real fwd/bwd/opt split; `phase_times_s` is only host dispatch. Nests the same way | `{}` with no correlation data (CPU trace) |
| 16 | `gpu_phase_coverage` | fraction of `gpu_busy_s` attributed to some annotation — 1.0 means nothing was missed | ditto |
| 17 | `data_wait_s` | dataloader annotation time | not `ok` |
| 18 | `data_wait_frac` | `data_wait_s / step_time_s` | not `ok` |
| 19 | `gap_s` | GPU idle (`step_time_s - gpu_busy_s`) | not `ok`; **`null` when negative** — the two come from different steps, so a negative value means they disagree rather than "no idle" |
| 20 | `gap_frac` | `gap_s / step_time_s` | not `ok` |
| 21 | `cpu_util_mean` | mean CPU % over the clean window | no psutil, or window too short to sample |
| 22 | `cpu_util_max` | max CPU % over the same window | ditto |
| 23 | `nproc` | host CPU count (denominator for the above) | ditto |
| 24 | `cpu_cores_busy` | CPU-seconds burned per wall-second by this process **and its children** (dataloader workers included). A counter delta rather than a sampled rate, so unlike the two fields above it survives windows too short to sample. `1.0` = one core saturated; `>1` = genuinely parallel | psutil missing and `process_time` unavailable |
| 25 | `peak_mem_gb` | `torch.cuda` peak allocated | off-GPU (CPU/MPS) |
| 26 | `reserved_mem_gb` | allocator reserved (`reserved - peak` = fragmentation) | off-GPU |
| 27 | `mem_total_gb` | device total | off-GPU |
| 28 | `headroom_gb` | `total - reserved` — distance to the OOM ceiling | off-GPU |
| 29 | `provenance` | `{flash_attention, fused_kernels, dtypes}` — a **hint** from substring-matching kernel names, not a verdict; the raw names are in `top_kernels`. Booleans are `null` when no kernels ran (unknown ≠ False) | `{}` when not `ok` |
| 30 | `top_kernels` | top 8 as `[[name, seconds], …]` | `[]` when not `ok` |
| 31 | `kernel_count` | distinct kernels — with `top8_coverage`, exposes the long tail | not `ok` |
| 32 | `top8_coverage` | fraction of GPU-busy inside the top 8 | not `ok`, or no kernels |
| 33 | `gap_count` | number of gaps on the GPU-kernel timeline | kernel events carry no `ts` (CPU trace) |
| 34 | `max_gap_s` | largest single gap — one big stall vs many small | ditto |
| 35 | `achieved_occupancy_pct` | runtime-weighted; the "why is compute slow" signal | trace has no occupancy args (CPU / no kernels) |
| 36 | `limiting_factor` | what caps occupancy (`REGS` / shared mem / warps) | ditto |
| 37 | `device` | real hw from the trace: `{name, mem_gb, compute_capability, num_sms}` | `{}` on a CPU trace |
| 38 | `trace_path` | raw Chrome trace on disk, for drill-down | no trace exported (e.g. OOM before the profiled step) |
| 39 | `error` | failure cause, first line | `status == "ok"` |
| 40 | `summary` | the one-line human/agent digest | never |

**Two steps, two sources.** `step_time_s` is the median of the clean steps (profiler **off**);
everything read from the trace — `gpu_busy_s`, the phase maps, gaps, occupancy — comes from the one
profiled step (profiler **on**). Mixing them is deliberate: kernel durations are measured on-device
by CUPTI and carry across steps, so `gpu_busy_s / step_time_s` is the best estimate of real GPU
utilization, whereas host-side annotation times are inflated by instrumentation and only mean
anything relative to the step they were measured in. `profiled_step_s` exposes that inflation.
Note `gpu_busy_s` sums kernel durations, so overlapping streams can push `gpu_active_frac` above 1.

On `oom` / `error` the throughput fields short-circuit to `null`, but **clean steps that already
ran are still reported** (`step_time_s`, `step_time_cv`) along with memory and CPU — evidence the
config was viable up to the ceiling, without scoring anything off it.

### What it looks like

Real output, A10G, 8×2048 MLP at micro-batch 32 / seq 512 (kernel names truncated):

```json
{
  "status": "ok",
  "gpu": "A10G",
  "peak_flops": 18155829441063.945,
  "peak_dtype": "float32",
  "step_time_s": 0.24849169600000032,
  "step_time_cv": 0.0067,
  "flops": 3300145496064.0,
  "roofline_floor_s": 0.181767817701564,
  "mfu": 0.7314844746424193,
  "gpu_busy_s": 0.2389987620000001,
  "gpu_active_frac": 0.9617977817657125,
  "profiled_step_s": 0.24137529099999888,
  "phase_times_s": {
    "step": 0.008329072,
    "dataloader": 2.6890000000000002e-05,
    "forward": 0.002659956,
    "backward": 0.004563027,
    "optimizer": 0.000957638,
    "Optimizer.step#AdamW.step": 0.000855763,
    "Optimizer.zero_grad#AdamW.zero_grad": 3.6358e-05
  },
  "gpu_phase_times_s": {
    "step": 0.23899876200000011,
    "dataloader": 0.0,
    "forward": 0.09197536300000002,
    "backward": 0.14134233699999996,
    "optimizer": 0.005681062,
    "Optimizer.step#AdamW.step": 0.005681062,
    "Optimizer.zero_grad#AdamW.zero_grad": 0.0
  },
  "gpu_phase_coverage": 1.0,
  "data_wait_s": 2.6890000000000002e-05,
  "data_wait_frac": 0.00010821287162851497,
  "gap_s": 0.00949293400000023,
  "gap_frac": 0.03820221823428747,
  "cpu_util_mean": 5.8,
  "cpu_util_max": 7.1,
  "nproc": 17,
  "peak_mem_gb": 2.016,
  "reserved_mem_gb": 2.021,
  "mem_total_gb": 22.058,
  "headroom_gb": 20.037,
  "provenance": {
    "flash_attention": false,
    "fused_kernels": true,
    "dtypes": [
      "fp32"
    ]
  },
  "top_kernels": [
    [
      "ampere_sgemm_128x64_tn",
      0.091137085
    ],
    [
      "void cutlass::Kernel2<cutlass_80_simt_sgemm_256x128_...(...)",
      0.071707412
    ]
  ],
  "kernel_count": 19,
  "top8_coverage": 0.98,
  "gap_count": 62,
  "max_gap_s": 2.462890625e-06,
  "achieved_occupancy_pct": 26.6,
  "limiting_factor": "REGS",
  "device": {
    "name": "NVIDIA A10",
    "mem_gb": 22.1,
    "compute_capability": "8.6",
    "num_sms": 72
  },
  "trace_path": "/tmp/autotune_trace_2_1787343066222349493.json",
  "error": null,
  "summary": "MFU 73% | GPU active 96% | data-wait 0% | idle 4% | cpu 6% | noise +/-1% | occ 27% | mem 2.016/22.058GB"
}
```

The agent reads this object — **never the raw trace.** `trace_path` points at the raw Chrome trace
for on-demand drill-down (open it in <https://ui.perfetto.dev>).

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

Engine complete and **GPU-validated** (A10G): every field populates, the OOM path fires — including
an OOM *after* clean steps, which keeps the timing it measured — and the CLI runs clean. Deferred: memory breakdown (→ Frame, analytical), per-op FLOPs (not in the trace export). GPU-time-per-phase now ships (`gpu_phase_times_s`, via correlation ids).
