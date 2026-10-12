# profiler/

The measurement instrument. Times a window of training steps, traces a fixed reference batch, and
turns both into an **Observation**: the 57-field diagnostic the agent reads. Framework-agnostic
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
- **Clean window** (profiler OFF) → the run: throughput, step-time distribution and spread, MFU.
  The profiler distorts timing, so these are measured with it off. When the runner reports
  `steps_per_pass`, the window is rounded up to **whole passes** over the sample and starts at a
  pass boundary, so every config times exactly the same rows the same number of times.
- **Reference batch** (`torch.profiler`) → what a step looks like inside: phases, kernels,
  launches, occupancy. A fixed set of rows chosen by rule (the runner's `reference_step()`), run
  once with the profiler off (its wall, `reference_step_s`) and once on (the trace). Changing
  `clean_steps`, warmup or loader order can't change which batch is traced.

Runner hooks, all optional and duck-typed (`frameworks/hf.py` implements them):
`steps_per_pass`, `start_pass()`, `has_reference`, `reference_step()`, `last_step_tokens()`, `last_step_samples()`,
`last_step_attention_pairs()`, `attention_parts` / `attention_skipped`, `population()`.
A runner without a reference batch (`has_reference` false: the torch fixture, a synthetic ceiling
trial) reuses one fixed batch, so its clean window already times the reference: `reference_step_s`
is the window's mean step, and only the traced step is extra.

Then it **measures** the device's achievable peak (a large dense GEMM at the run's dtype) and reports
**MFU = achieved / measured peak** — the "is the GPU utilized well" signal (util% lies; MFU doesn't).
The peak is measured, never read from a table: tables can't tell an A10 from an A10G, quietly quote
sparse numbers, and fail on hardware they've never heard of. Measuring supports any chip and prices
in real clocks and power caps.

## The Observation (57 fields)

A **flat JSON object** — one level deep, except six container fields (`population`, `phase_times_s`,
`gpu_phase_times_s`, `provenance`, `device`, `top_kernels`). `python -m profiler` prints the one-line `summary`, then
`json.dumps(obs.__dict__, indent=2, default=str)`. Anything not measured is `null` — never 0,
so "unknown" can't be mistaken for "measured zero".

| # | Field | Meaning | `null` when |
|---|---|---|---|
| 1 | `status` | `ok` / `oom` / `error` / `invalid`. `invalid` = a physically impossible reading (a fraction above 1 beyond a 2% tolerance, calibrated on an A100): two populations were mixed or the instrument failed, so the probe is never scored. The values stay in place and `error` names the reading | never |
| 2 | `gpu` | hardware **label** from `--gpu`, for the record only — it is never priced or looked up | never |
| 3 | `peak_flops` | dense-GEMM peak **measured on this device** — the MFU denominator. No datasheet table, so any hardware works and throttling is priced in | measurement failed (then `mfu` is `null` too), or a bare `error` (below) |
| 4 | `peak_dtype` | dtype the peak was measured at, never defined by the profiler: the config's `compute_dtype` (the code autocasts), else its `dtype`, else what the code loads when it names none: the checkpoint's own (our runner: `frameworks/hf.py` `resolve_dtype`; a script: `backends/local.py`). MFU is only apples-to-apples if it matches what the matmuls ran in | a bare `error` |
| 5 | `peak_mem_bw_gb_s` | achievable memory bandwidth, **measured** by a large device-to-device copy — the roofline's other axis. Without it, a low MFU can't be diagnosed as compute-bound vs bandwidth-bound, and most transformer training is the latter | measurement failed |
| 6 | `step_time_s` | clean wall-clock per step, profiler off, **median**. A distribution statistic only: never the denominator of a rate (under length-grouped padding the median sat at 0.45x the mean) | no clean step completed |
| 7 | `step_time_mean_s` | clean wall-clock per step, **mean**. The denominator of every rate (throughput is total work over total time) | no clean step completed |
| 8 | `step_time_cv` | std / mean of the clean step times | ditto |
| 9 | `clean_steps_measured` | clean steps actually timed: `clean_steps` rounded **up** to whole passes when the runner reports `steps_per_pass` | ditto |
| 10 | `clean_steps_requested` | the `clean_steps` the caller asked for. Differs from `clean_steps_measured` when the window was rounded up to whole passes, which is also printed to stderr | a bare `error` (set on `oom` even when no step completed) |
| 11 | `clean_passes` | passes over the sample the clean window **completed** (so an OOM mid-window reports a partial pass). A whole number means every row was timed the same number of times, so two configs on the same `population.sample_hash` timed identical work | runner doesn't report passes (torch fixture, synthetic), and script probes |
| 12 | `tokens_per_step` | mean tokens (padded width x rows) per **clean** step, counted per step; falls back to the caller's constant for runners that don't count | not measured |
| 13 | `tokens_per_s` | total clean tokens / total clean time | not `ok` |
| 14 | `samples_per_step` | mean samples (training examples) per **clean** step, counted per step. Runners count rows; a script probe counts rows when unpacked and documents when packed (`position_ids` / `cu_seqlens`; see "Attached capture" below) | samples not counted (runner without the hook; a script micro-batch with no boundaries to read) |
| 15 | `samples_per_s` | total clean samples / total clean time. **The primary metric**: what configs are ranked on (`loop/throughput.md`) | not `ok`, or samples not counted |
| 16 | `flops` | estimated training FLOPs of the **mean** clean step, PaLM's model-FLOPs definition (Chowdhery et al. 2022, arXiv:2204.02311, App. B): weights plus attention. Weights: `(4·params_total + 2·params_trainable)·D` over `tokens_per_step` (6ND when every weight trains); a VLM charges the vision tower per image patch instead, `(k·params_vision + 2·its trained part)·patches_per_step`, k = 4 if any of the tower trains, else 2 (`roofline.vlm_training_flops`). Attention: `attention_flops`. Model FLOPs, not executed: recompute isn't counted, and as in PaLM the causal mask isn't discounted | not `ok`, or `params_total` or `tokens_per_step` unknown |
| 17 | `params_total` | N: every weight of the model, trained or frozen. Runner: the model's parameters. Script: the parameters of the module the optimizer's weights live in (under PEFT, the base model plus the adapters) | unknown: no model found, a quantized base (any storage dtype), or an image input whose vision tower or patch count can't be read (no `visual` / `vision_tower` / `vision_model` / `vision_encoder` module, or an image tensor that's neither `[patches, dim]` nor `[..., C, H, W]`) — `population.micro_batches.params_source` says why |
| 18 | `attention_flops` | PaLM's attention term of `flops`: `12·L·H·Q·pairs` per attention part, the (query, key) pairs the step attended over (`profiler/attention.py`): `width²` a row, a packed row's documents separately, `width·min(width, window)` for sliding-window layers, `patches²` per image for a vision tower; `4·L·H·Q·pairs` (forward only) for a frozen tower | not `ok`, `flops` unknown, no attention layout read off the config, or pairs not counted on every clean step |
| 19 | `attention_source` | what `attention_flops` covers (each part's layers × width, window, forward-only) and what it leaves out (linear-attention / state-space layers, windowed vision blocks, an image layout with no per-image count). Left-out parts make it read low, never high. `unknown: <why>` when no layout could be read (e.g. `unknown: no model config`) | not `ok`; a script whose model was never found; a runner without `attention_parts` (torch fixture) |
| 20 | `params_vision` | a VLM: the vision tower's share of `params_total`, the weights charged per image patch (parameters under a `visual` / `vision_tower` / `vision_model` / `vision_encoder` module) | not a VLM probe, or its FLOPs are unknown |
| 21 | `patches_per_step` | a VLM: mean image patches per clean step, from the image tensors' shapes (a `[patches, dim]` tensor's rows; else images × (H/patch)·(W/patch)) | as `params_vision` |
| 22 | `params_trainable` | the weights the optimizer trains: its parameters with `requires_grad`, so a frozen weight handed to it (`AdamW(model.parameters())` under LoRA) doesn't count. Equals `params_total` in a full fine-tune; under LoRA, the adapters only | no optimizer step seen |
| 23 | `roofline_floor_s` | compute-bound floor on the mean step's time at peak | not `ok` |
| 24 | `mfu` | achieved / peak over the clean window's **totals**: `flops / step_time_mean_s / peak_flops` | not `ok`; `peak_dtype` float32 while the trace's kernels ran bf16 / fp16 (the summary says so; `roofline_floor_s` is `null` too) |
| 25 | `gpu_busy_s` | time the GPU was busy on the **reference batch**: the union of kernel intervals in its trace, so concurrent streams don't count twice | not `ok`; **`null` when a CUDA trace recorded no kernels** (a failed trace, not an idle GPU) |
| 26 | `gpu_active_frac` | `gpu_busy_s / reference_step_s`: both sides are the same batch. Describes the reference batch, not the run average | not `ok`, or no `reference_step_s` |
| 27 | `reference_step_s` | wall of the **reference batch** with the profiler **off**: the denominator for every trace fraction | not `ok`, or a script probe |
| 28 | `profiled_step_s` | wall of the reference batch **with** the profiler on. `profiled_step_s / reference_step_s` is the profiler's overhead factor (≈1.0 when GPU-bound, up to ~1.9x launch-bound), exposed so the distortion is visible rather than folded into the numbers | not `ok` |
| 29 | `population` | what was measured. `sample_method` (`length-stratified` from a Frame-drawn `sample_file`, or `head` for the unrepresentative fallback) / `sample_file` / `sample_rows` / `sample_hash` / `steps_per_pass` / `loader_seed` identify the rows the clean window timed; `reference.{row_ids, rows_hash, real_tokens, batch_shapes, batch_key}` identify the batch the trace describes. Compare duration metrics across probes only when `batch_key` matches | `{}` for runners without the hooks (torch fixture, synthetic) |
| 30 | `phase_times_s` | **every** `record_function` annotation, seconds — the runners' `forward` / `backward` / `optimizer` / `dataloader` (a script's are derived from torch's own events, `derive.py`) plus whatever the runner/torch emit (`step`, `Optimizer.step#AdamW.step`, comm, …). Two traps: scopes **nest** (do not sum them), and these are **host dispatch times, not GPU time** — CUDA launches are async, so `forward` is how long the host spent *launching* forward, not how long the GPU ran it. Compare against `gpu_busy_s`, never against these | `{}` when not `ok` |
| 31 | `gpu_phase_times_s` | **GPU** time per annotation, seconds — kernels linked to the enclosing annotation by `correlation` id. This is the real fwd/bwd/opt split; `phase_times_s` is only host dispatch. Nests the same way | `{}` with no correlation data (CPU trace) |
| 32 | `gpu_phase_coverage` | share of **summed** kernel time attributed to some annotation (summed, not `gpu_busy_s`'s union, so overlapping kernels count once each) — 1.0 means nothing was missed | ditto |
| 33 | `data_wait_s` | dataloader annotation time | not `ok`, or the trace has no `dataloader` annotation |
| 34 | `data_wait_frac` | `data_wait_s / profiled_step_s`: a host annotation over the wall of the profiled run it was recorded in | not `ok`, or no `data_wait_s` |
| 35 | `gap_s` | reference-batch GPU idle (`reference_step_s - gpu_busy_s`) | not `ok`; **`null` when negative**: busy comes from the profiled run and wall from the unprofiled one, so a negative value means the two runs disagree, not "no idle" |
| 36 | `gap_frac` | `gap_s / reference_step_s` | not `ok` |
| 37 | `cpu_util_mean` | mean **whole-machine** CPU % (psutil, all cores averaged) over the clean window: other processes on the host count too. `cpu_cores_busy` is this process alone | no psutil, or window too short to sample |
| 38 | `cpu_util_max` | max whole-machine CPU % over the same window | ditto |
| 39 | `nproc` | host CPU count (denominator for the above) | a runner `oom` with no CPU sample, a script that failed before its clean window, or a bare `error` |
| 40 | `cpu_cores_busy` | CPU-seconds burned per wall-second by this process **and its children** (dataloader workers included). A counter delta rather than a sampled rate, so unlike the two fields above it survives windows too short to sample. `1.0` = one core saturated; `>1` = genuinely parallel | a runner `oom`, a script that failed before its clean window, or a bare `error` (without psutil it falls back to this process's `process_time`) |
| 41 | `peak_mem_gb` | `torch.cuda` peak allocated over the clean window (read when the window closes, before the reference batch or traced step runs) | off-GPU (CPU/MPS) |
| 42 | `reserved_mem_gb` | allocator reserved (`reserved - peak` = fragmentation) | off-GPU |
| 43 | `mem_total_gb` | device total | off-GPU |
| 44 | `headroom_gb` | `total - reserved` — distance to the OOM ceiling | off-GPU |
| 45 | `provenance` | `{flash_attention, fused_kernels, dtypes}` — a **hint** from substring-matching kernel names, not a verdict; the raw names are in `top_kernels`. Booleans are `null` when no kernels ran (unknown ≠ False) | `{}` when not `ok` |
| 46 | `top_kernels` | top 8 as `[[name, seconds], …]` | `[]` when not `ok` |
| 47 | `kernel_count` | distinct kernels — with `top8_coverage`, exposes the long tail | not `ok` |
| 48 | `top8_coverage` | share of **summed** kernel time inside the top 8 (a share of kernel time, not of `gpu_busy_s`) | not `ok`, or no kernels |
| 49 | `gap_count` | gaps between consecutive kernel events on the GPU timeline, of any size | kernel events carry no `ts` (CPU trace) |
| 50 | `max_gap_s` | largest single gap — one big stall vs many small | ditto |
| 51 | `achieved_occupancy_pct` | runtime-weighted; the "why is compute slow" signal | trace has no occupancy args (CPU / no kernels) |
| 52 | `limiting_factor` | what caps occupancy (`REGS` / shared mem / warps) | ditto |
| 53 | `device` | real hw from the trace: `{name, mem_gb, compute_capability, num_sms}` | `{}` on a CPU trace |
| 54 | `device_type` | what it RAN on (`cuda` / `cpu` / `mps`). A `cpu` probe is a smoke test: MFU and every GPU signal are meaningless, so never rank it against a `cuda` probe or learn a belief from it | capture never reached a device |
| 55 | `trace_path` | raw Chrome trace on disk, for drill-down | no trace exported (e.g. OOM before the profiled step) |
| 56 | `error` | failure cause, first line; for `invalid`, which reading was impossible | `status == "ok"` |
| 57 | `summary` | the one-line human/agent digest: samples/s, tokens/s, MFU, then the GPU signals (prefixed `[CPU — not a GPU probe]` when `device_type` isn't `cuda`, and `INVALID (...)` when `status` is `invalid`) | never |

**Two populations, never mixed.** Rates (`samples_per_s`, `tokens_per_s`, `mfu`, `roofline_floor_s`) come only from
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
config was viable up to the ceiling, without scoring anything off it. That holds for `capture()`'s
OOM and for a script that ended early (`Attach.failed`). A **bare `error`** carries only `status`,
`device_type` and `error`: `capture()`'s non-OOM exception, or a script probe that wrote no result
(timed out, killed, invalid settings — `backends/local.py`).

### What it looks like

An old-format illustration: real output from an earlier version, A10G, 8×2048 MLP at micro-batch
32 / seq 512 (kernel names truncated). Its fields, numbers and summary line predate the current
field set and formulas (e.g. `gpu_active_frac` here is busy time over the median clean step, the
design replaced above), so read the table above for the current shape:

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

- `roofline.py` — FLOPs, PaLM's definition (arXiv:2204.02311, App. B): weights (6ND, or `(4·N_total + 2·N_trainable)·D` with frozen weights; per image patch for a VLM's vision tower) + attention (`12·L·H·Q·pairs`) → step-time floor, MFU.
- `attention.py` — which layers attend over what, read off the model config (full / sliding-window / vision parts), for the attention term.
- `trace.py` — parse a torch.profiler Chrome trace → phases, kernels, gaps, occupancy, device props, provenance.
- `capture.py` — the clean window (whole passes, per-step tokens) + CPU sampler (psutil) + memory + OOM handling + the reference batch, timed profiler-off then traced. *(needs torch)*
- `diagnostic.py` — assemble the Observation (torch-free; the `Observation` dataclass lives here).
- `__main__.py` — the CLI (`python -m profiler`).
- `launch.py` / `attach.py` / `derive.py` / `nopatch.py` / `memtrial.py` — the attached capture for user scripts, and its memory-only ceiling trial (below).

## Attached capture: user scripts

`capture()` drives a runner. For a **user's own training script**, the same Observation comes from
the **attached capture** instead, started by `python -m profiler.launch -- train.py …`
(`backends/README.md`, "Script probes"):

| File | Does |
|---|---|
| `launch.py` | runs the script unchanged in-process (`runpy`), with `Attach` installed first; writes `cap.json`; `ok` / `oom` / `error` like `capture()` |
| `attach.py` | step boundary = torch's public optimizer post-hook; clean timing; **tokens counted on every clean step** by a per-module pre-hook on the model (padded `input_ids.numel()`, as the runners count; forwards under `no_grad` / `inference_mode` — eval, `generate()`, a reference pass — are not counted, though their time stays in the step, so switch eval off for probes. A training-side `no_grad` forward of the same model (a reference pass for a KL term) is right to skip, since its rows are already counted, but its ~2N·D forward FLOPs aren't in the FLOP formula, so MFU reads low (~1/3) for such loops); one profiled step with `record_shapes=True` |
| `derive.py` | torch-free: forward / backward / dataloader spans and the micro-batch count, read from torch's own trace events (the script has no `record_function` labels) |
| `nopatch.py` | guard: the run fails if any torch function was replaced or a global module hook is left registered |
| `memtrial.py` | `launch.py --memory-only`: a ceiling trial. Counts optimizer steps, stops the script after N (default 3), reads the allocator's peaks, and checks every model input is `[micro_batch, max_len]`. No timing, no profiler |

**How it differs from `capture()`, recorded in `population`** (`driver: "script"`):
- **The clean window is a fixed stretch of the script's own loader** (`clean_window:
  "fixed-position"`), not whole passes over Frame's sample. So `clean_passes` and `sample_hash`
  are absent, and throughput isn't `same-rows` evidence (Measurement axiom, `loop/contracts.md`).
- **The trace is one step of the script's loop** (`trace_basis: "traced-step"`), not a replayed
  reference batch. So `reference_step_s`, `gpu_active_frac`, `gap_s` and `gap_frac` are null.
- **Step time reads ~0.25% high** from where the per-step sync sits relative to the framework's own
  bookkeeping, so don't compare it with runner probes.
- **LoRA / PEFT works.** `params_trainable` is the optimizer's count (the adapters) and
  `params_total` every weight of the model the optimizer's parameters live in, so MFU counts the
  forward and activation gradients through the frozen base and weight gradients only for the
  adapters. A quantized base (QLoRA's 4-bit / int8, GPTQ, AWQ) packs weights, so `params_total` and
  MFU are null (`population.micro_batches.params_source` says why). Expect the kernel signals to move: LoRA
  adds small rank-r GEMMs (higher `kernel_count` and `gap_count`, lower `top8_coverage` and
  occupancy) and drops the frozen weights' gradients (smaller backward and optimizer phases).
- **Samples are counted from the boundaries the batch carries.** `population.micro_batches.sample_source`
  records which rule applied (`rows`, `position_ids`, `cu_seqlens`, or `unknown: <why>`).
  - **Supported:** unpacked 2-D batches (a row is a sample); packed batches whose `position_ids`
    reset to 0 per document, passed as a keyword (`[B, S]`, `[1, S]` or multi-axis such as mRoPE
    `[3, B, S]`), with padding masked, or at position 0 unmasked (a run of 0s that ends a row is
    read as padding: TRL `padding_free` + `pad_to_multiple_of`); flattened 1-D batches with a 1-D
    `cu_seqlens` / `cu_seq_lens_q` keyword. A 1-D batch with neither is `null`.
  - **Underestimates** (documents counted as rows, or merged): concat-and-chunk packing with no
    markers; packing through a 4-D block-diagonal mask; packing without position resets (EOS only);
    boundaries in other kwargs (`seq_idx`, 2-D `cu_seqlens`, `packed_seq_params`); `position_ids`
    passed positionally; a one-token document followed by another (their 0s form one run), or
    ending a row with no `attention_mask` (read as padding); left
    padding with broadcast `[1, S]` positions `0…S-1` and an `attention_mask` (each padded row's
    position 0 is a masked pad, so that row counts 0; left padding is for generation, outside
    SFT / PEFT).
  - **Overestimates:** documents split across rows (counted once per row); context / sequence
    parallel shards that start mid-document; unmasked padding at position 0 *between* documents
    (one extra per pad run).


## Adapters

`capture` drives a **runner** — anything with `setup(config)` + `step()`. `frameworks/torch.py` is a
minimal fixture (an MLP on random data → *shape*, not real numbers). A real framework adapter
(`frameworks/hf.py`) profiles a real model; the profiler engine is unchanged.

## Status

Engine complete and **GPU-validated** (A10G): every field populates, the OOM path fires — including
an OOM *after* clean steps, which keeps the timing it measured — and the CLI runs clean. Deferred: memory breakdown (→ Frame, analytical), per-op FLOPs (not in the trace export). GPU-time-per-phase now ships (`gpu_phase_times_s`, via correlation ids).
