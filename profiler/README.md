# profiler/

The measurement instrument. Times a window of training steps, traces a fixed reference batch, and
turns both into an **Observation**: the 48-field diagnostic the agent reads. Framework-agnostic
(it's `torch.profiler` underneath), and validated live on an A10G.

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

Each step is timed **start → start of the next** (so it catches data-wait too). Two instruments,
each measuring its own population (the Measurement axiom, `loop/contracts.md`):
- **Clean window** (profiler OFF) → the run: throughput, step-time distribution, noise floor, MFU.
  The profiler distorts timing, so these are measured with it off. When the runner reports
  `steps_per_pass`, the window is rounded up to **whole passes** over the sample and starts at a
  pass boundary, so every config times exactly the same rows the same number of times.
- **Reference batch** (`torch.profiler`) → what a step looks like inside: phases, kernels,
  launches, occupancy. A fixed set of rows chosen by rule (the runner's `reference_step()`), run
  once with the profiler off (its wall, `reference_step_s`) and once on (the trace). Changing
  `clean_steps`, warmup or loader order can't change which batch is traced.

Runner hooks, all optional and duck-typed (`frameworks/hf.py` implements them):
`steps_per_pass`, `start_pass()`, `reference_step()`, `last_step_tokens()`, `population()`.
A runner without them (the torch fixture, a synthetic ceiling trial) reuses one fixed batch, so its
`step()` already is the reference.

Then it **measures** the device's achievable peak (a large dense GEMM at the run's dtype) and reports
**MFU = achieved / measured peak** — the "is the GPU utilized well" signal (util% lies; MFU doesn't).
The peak is measured, never read from a table: tables can't tell an A10 from an A10G, quietly quote
sparse numbers, and fail on hardware they've never heard of. Measuring supports any chip and prices
in real clocks and power caps.

## The Observation (48 fields)

A **flat JSON object** — one level deep, except five container fields (`population`, `phase_times_s`,
`provenance`, `device`, `top_kernels`). `python -m profiler` prints the one-line `summary`, then
`json.dumps(obs.__dict__, indent=2, default=str)`. Anything not measured is `null` — never 0,
so "unknown" can't be mistaken for "measured zero".

| # | Field | Meaning | `null` when |
|---|---|---|---|
| 1 | `status` | `ok` / `oom` / `error` / `invalid`. `invalid` = a physically impossible reading (a fraction above 1 beyond a 5% tolerance): two populations were mixed or the instrument failed, so the probe is never scored. The values stay in place and `error` names the reading | never |
| 2 | `gpu` | hardware **label** from `--gpu`, for the record only — it is never priced or looked up | never |
| 3 | `peak_flops` | dense-GEMM peak **measured on this device** — the MFU denominator. No datasheet table, so any hardware works and throttling is priced in | measurement failed (then `mfu` is `null` too) |
| 4 | `peak_dtype` | dtype the peak was measured at. MFU is only apples-to-apples if it matches what the model ran in | ditto |
| 5 | `peak_mem_bw_gb_s` | achievable memory bandwidth, **measured** by a large device-to-device copy — the roofline's other axis. Without it, a low MFU can't be diagnosed as compute-bound vs bandwidth-bound, and most transformer training is the latter | measurement failed |
| 6 | `step_time_s` | clean wall-clock per step, profiler off, **median**. A distribution statistic only: never the denominator of a rate (under length-grouped padding the median sat at 0.45x the mean) | no clean step completed |
| 7 | `step_time_mean_s` | clean wall-clock per step, **mean**. The denominator of every rate (throughput is total work over total time) | no clean step completed |
| 8 | `step_time_cv` | noise floor: coefficient of variation across clean steps | ditto |
| 9 | `clean_steps_measured` | clean steps actually timed: `clean_steps` rounded **up** to whole passes when the runner reports `steps_per_pass` | ditto |
| 10 | `clean_passes` | passes over the sample the clean window covered. A whole number means every row was timed the same number of times, so two configs on the same `population.sample_hash` timed identical work | runner doesn't report passes (torch fixture, synthetic) |
| 11 | `tokens_per_step` | mean tokens (padded width x rows) per **clean** step, counted per step; falls back to the caller's constant for runners that don't count | not measured |
| 12 | `tokens_per_s` | total clean tokens / total clean time | not `ok` |
| 13 | `flops` | estimated training FLOPs of the **mean** clean step (6ND over `tokens_per_step`) | not `ok` |
| 14 | `roofline_floor_s` | compute-bound floor on the mean step's time at peak | not `ok` |
| 15 | `mfu` | achieved / peak over the clean window's **totals**: `flops / step_time_mean_s / peak_flops` | not `ok` |
| 16 | `gpu_busy_s` | time the GPU was busy on the **reference batch**: the union of kernel intervals in its trace, so concurrent streams don't count twice | not `ok` |
| 17 | `gpu_active_frac` | `gpu_busy_s / reference_step_s`: both sides are the same batch. Describes the reference batch, not the run average | not `ok`, or no `reference_step_s` |
| 18 | `reference_step_s` | wall of the **reference batch** with the profiler **off**: the denominator for every trace fraction | not `ok` |
| 19 | `profiled_step_s` | wall of the reference batch **with** the profiler on. `profiled_step_s / reference_step_s` is the profiler's overhead factor (≈1.0 when GPU-bound, up to ~1.9x launch-bound), exposed so the distortion is visible rather than folded into the numbers | not `ok` |
| 20 | `population` | what was measured. `sample_method` (`length-stratified` from a Frame-drawn `sample_file`, or `head` for the unrepresentative fallback) / `sample_file` / `sample_rows` / `sample_hash` / `steps_per_pass` / `loader_seed` identify the rows the clean window timed; `reference.{row_ids, rows_hash, real_tokens, batch_shapes, batch_key}` identify the batch the trace describes. Compare duration metrics across probes only when `batch_key` matches | `{}` for runners without the hooks (torch fixture, synthetic) |
| 21 | `phase_times_s` | **every** `record_function` annotation, seconds — the four above plus whatever the runner/torch emit (`step`, `Optimizer.step#AdamW.step`, comm, …). Two traps: scopes **nest** (do not sum them), and these are **host dispatch times, not GPU time** — CUDA launches are async, so `forward` is how long the host spent *launching* forward, not how long the GPU ran it. Compare against `gpu_busy_s`, never against these | `{}` when not `ok` |
| 22 | `gpu_phase_times_s` | **GPU** time per annotation, seconds — kernels linked to the enclosing annotation by `correlation` id. This is the real fwd/bwd/opt split; `phase_times_s` is only host dispatch. Nests the same way | `{}` with no correlation data (CPU trace) |
| 23 | `gpu_phase_coverage` | fraction of `gpu_busy_s` attributed to some annotation — 1.0 means nothing was missed | ditto |
| 24 | `data_wait_s` | dataloader annotation time | not `ok` |
| 25 | `data_wait_frac` | `data_wait_s / profiled_step_s`: a host annotation over the wall of the profiled run it was recorded in | not `ok` |
| 26 | `gap_s` | reference-batch GPU idle (`reference_step_s - gpu_busy_s`) | not `ok`; **`null` when negative**: busy comes from the profiled run and wall from the unprofiled one, so a negative value means the two runs disagree, not "no idle" |
| 27 | `gap_frac` | `gap_s / reference_step_s` | not `ok` |
| 28 | `cpu_util_mean` | mean CPU % over the clean window | no psutil, or window too short to sample |
| 29 | `cpu_util_max` | max CPU % over the same window | ditto |
| 30 | `nproc` | host CPU count (denominator for the above) | ditto |
| 31 | `cpu_cores_busy` | CPU-seconds burned per wall-second by this process **and its children** (dataloader workers included). A counter delta rather than a sampled rate, so unlike the two fields above it survives windows too short to sample. `1.0` = one core saturated; `>1` = genuinely parallel | psutil missing and `process_time` unavailable |
| 32 | `peak_mem_gb` | `torch.cuda` peak allocated over the clean window (read before the reference batch runs) | off-GPU (CPU/MPS) |
| 33 | `reserved_mem_gb` | allocator reserved (`reserved - peak` = fragmentation) | off-GPU |
| 34 | `mem_total_gb` | device total | off-GPU |
| 35 | `headroom_gb` | `total - reserved` — distance to the OOM ceiling | off-GPU |
| 36 | `provenance` | `{flash_attention, fused_kernels, dtypes}` — a **hint** from substring-matching kernel names, not a verdict; the raw names are in `top_kernels`. Booleans are `null` when no kernels ran (unknown ≠ False) | `{}` when not `ok` |
| 37 | `top_kernels` | top 8 as `[[name, seconds], …]` | `[]` when not `ok` |
| 38 | `kernel_count` | distinct kernels — with `top8_coverage`, exposes the long tail | not `ok` |
| 39 | `top8_coverage` | fraction of GPU-busy inside the top 8 | not `ok`, or no kernels |
| 40 | `gap_count` | number of gaps on the GPU-kernel timeline | kernel events carry no `ts` (CPU trace) |
| 41 | `max_gap_s` | largest single gap — one big stall vs many small | ditto |
| 42 | `achieved_occupancy_pct` | runtime-weighted; the "why is compute slow" signal | trace has no occupancy args (CPU / no kernels) |
| 43 | `limiting_factor` | what caps occupancy (`REGS` / shared mem / warps) | ditto |
| 44 | `device` | real hw from the trace: `{name, mem_gb, compute_capability, num_sms}` | `{}` on a CPU trace |
| 45 | `device_type` | what it RAN on (`cuda` / `cpu` / `mps`). A `cpu` probe is a smoke test: MFU and every GPU signal are meaningless, so never rank it against a `cuda` probe or learn a belief from it | capture never reached a device |
| 46 | `trace_path` | raw Chrome trace on disk, for drill-down | no trace exported (e.g. OOM before the profiled step) |
| 47 | `error` | failure cause, first line; for `invalid`, which reading was impossible | `status == "ok"` |
| 48 | `summary` | the one-line human/agent digest (prefixed `[CPU — not a GPU probe]` when `device_type` isn't `cuda`, and `INVALID (...)` when `status` is `invalid`) | never |

**Two populations, never mixed.** Rates (`mfu`, `tokens_per_s`, `roofline_floor_s`) come only from
the clean window's totals: mean tokens over mean step time. Everything read from the trace
(`gpu_busy_s`, the phase maps, gaps, occupancy, kernels) describes the reference batch only, and
its fractions divide by that batch's own wall: `reference_step_s` for GPU time (profiler off, so
the profiler's host overhead doesn't deflate it) and `profiled_step_s` for host annotations (which
the profiler inflates, so they only mean anything against the run they were recorded in). No ratio
takes its numerator from one population and its denominator from the other.

This replaces an earlier design that divided one profiled step's busy time by the median clean
step, on the assumption that kernel durations "carry across steps". They do when every batch is
the same shape. Under length-grouped dynamic padding they don't: it printed GPU-active of 126-560%
and MFU of 110-144% on the smollm2/qwen3.5 A100 runs. Comparing across probes follows the same
rule. Throughput and step time compare when `population.sample_hash` matches and `clean_passes`
is whole. Trace durations compare only when `population.reference.batch_key` matches. Counts and
provenance (`gap_count`, `kernel_count`, which kernels ran, `dtypes`) compare across any batch.

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
- `capture.py` — the clean window (whole passes, per-step tokens) + CPU sampler (psutil) + memory + OOM handling + the reference batch, timed profiler-off then traced. *(needs torch)*
- `diagnostic.py` — assemble the Observation (torch-free; the `Observation` dataclass lives here).
- `__main__.py` — the CLI (`python -m profiler`).

## Adapters

`capture` drives a **runner** — anything with `setup(config)` + `step()`. `frameworks/torch.py` is a
minimal fixture (an MLP on random data → *shape*, not real numbers). A real framework adapter
(`frameworks/swift.py` / HF) profiles a real model; the profiler engine is unchanged.

## Status

Engine complete and **GPU-validated** (A10G): every field populates, the OOM path fires — including
an OOM *after* clean steps, which keeps the timing it measured — and the CLI runs clean. Deferred: memory breakdown (→ Frame, analytical), per-op FLOPs (not in the trace export). GPU-time-per-phase now ships (`gpu_phase_times_s`, via correlation ids).
