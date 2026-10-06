# Metrics audit

Every metric the profiler reports: what it is, how the code calculates it, why it is needed, and
whether the code matches the docs. Audited against `main` at `592d074` (2026-10-05), by reading the
code. The seven items it flagged are fixed in PR #24 (`metrics-audit`), which this doc ships with.
`samples_per_step` and `samples_per_s` are added by PR #25 (`metric`).

**Matches docs:** ✅ yes · ✅ *n* fixed by this PR, see item *n* below · ❌ a real gap.

"Clean window" means the timed training steps with the profiler off. "Reference batch" means the
one fixed batch that is traced with the profiler on.

| Metric | What it is | How it's calculated | Why it's needed | Matches docs |
|---|---|---|---|---|
| `status` | Whether the probe produced a usable reading | `ok`, `oom`, `error`, or `invalid` when `gpu_active_frac` or `mfu` reads above 1.02 | Stops a failed or impossible reading from being scored | ✅ |
| `gpu` | Hardware label | Copied from the `--gpu` input; never looked up | Identifies the run in records | ✅ |
| `error` | Why the probe failed | First line of the exception, or which reading was impossible | Lets the agent tell an OOM from a crash from a bad measurement | ✅ |
| `summary` | One-line digest | Formatted from MFU, GPU-active, data-wait, idle, CPU, noise, occupancy, memory | Quick read for a human or the agent | ✅ |
| `trace_path` | Where the raw trace is | Path of the exported Chrome trace | Lets the agent drill into the trace when the summary isn't enough | ✅ |
| `step_time_s` | Typical step time | Median of the clean step times | Describes a typical step; never used in a rate | ✅ |
| `step_time_mean_s` | Average step time | Sum of clean step times / number of steps | The denominator of every rate | ✅ |
| `step_time_cv` | Spread of step times | Population std of step times / mean | Shows how much the step times varied; the agent decides what that means | ✅ 1 |
| `clean_steps_measured` | Steps actually timed | Count of clean step times | Shows how much evidence is behind the timing | ✅ |
| `clean_steps_requested` | Steps the caller asked for | The `clean_steps` input, before rounding up to whole passes | Explains why measured differs from requested | ✅ |
| `clean_passes` | Passes over the sample completed | Steps completed / steps per pass | A whole number proves every row was timed equally, so configs are comparable | ✅ |
| `samples_per_step` | Rows per step (PR #25) | Mean of per-step row counts over the clean steps | Numerator of samples/s | ✅ |
| `samples_per_s` | Training examples per second (PR #25) | `samples_per_step / step_time_mean_s` | **The primary metric**: what configs are ranked on | ✅ in PR #25; ❌ absent on `main` |
| `tokens_per_step` | Tokens per step, padding included | Mean of per-step `input_ids.numel()` over the clean steps | Numerator of tokens/s and of the FLOP estimate | ✅ |
| `tokens_per_s` | Tokens per second | `tokens_per_step / step_time_mean_s` | Cross-check on samples/s: shows whether a win is real or a sampling artifact | ✅ |
| `peak_flops` | Fastest compute the device delivers | Best of three large matrix multiplies: `2 × n³ × iters / elapsed` | The denominator of MFU, measured so any hardware works | ✅ |
| `peak_dtype` | Dtype the peak was measured at | The run's dtype | MFU is only fair if this matches what the model ran in | ✅ |
| `peak_mem_bw_gb_s` | Memory bandwidth the device delivers | A 256 MB device copy: `2 × bytes × iters / elapsed` | Tells whether a low MFU is limited by compute or by memory traffic | ✅ |
| `flops` | Estimated work in one step | `6 × params × tokens_per_step` | Numerator of MFU | ✅ |
| `roofline_floor_s` | Fastest a step could possibly be | `flops / peak_flops` | Shows how far a step is from the hardware limit | ✅ |
| `mfu` | Share of the device's peak compute actually used | `(flops / step_time_mean_s) / peak_flops` | A diagnostic for how well the GPU is used | ✅ |
| `reference_step_s` | Time of the reference batch, profiler off | Wall time of one run of that batch. `null` for user scripts | Denominator for every GPU fraction from the trace | ✅ |
| `profiled_step_s` | Time of the same batch, profiler on | Wall time of the traced run | Exposes how much the profiler slows the step | ✅ |
| `gpu_busy_s` | Time the GPU was working in the reference batch | Union of kernel intervals on the trace timeline | Numerator of GPU-active | ✅ |
| `gpu_active_frac` | Share of the step the GPU was working | `gpu_busy_s / reference_step_s` | Shows whether the GPU is being kept fed | ✅ |
| `gap_s` | GPU idle time in the reference batch | `reference_step_s − gpu_busy_s`; `null` if negative | Sizes how much time is lost to waiting | ✅ |
| `gap_frac` | Idle share of the step | `gap_s / reference_step_s` | Same, as a fraction | ✅ |
| `gap_count` | Number of idle gaps | Count of positive gaps between consecutive kernel events, any size | With `max_gap_s`, meant to tell one big stall from many small ones | ✅ 5 |
| `max_gap_s` | Longest single idle gap | Largest of those gaps | Spots one big stall, such as a data wait | ✅ |
| `phase_times_s` | Host time per phase (forward, backward, …) | Sum of each annotation's duration, by name | Shows where the host spends its time launching work | ✅ |
| `gpu_phase_times_s` | GPU time per phase | Each kernel is assigned to the annotation its launch call started in | The real forward / backward / optimizer split | ✅ |
| `gpu_phase_coverage` | Share of kernel time assigned to a phase | Assigned kernel time / all kernel time | Shows whether the phase split missed anything | ✅ 6 |
| `data_wait_s` | Time waiting on the dataloader | The `dataloader` annotation's time; `null` if there is none | Flags a data-loading bottleneck | ✅ 2 |
| `data_wait_frac` | Data wait as a share of the step | `data_wait_s / profiled_step_s` | Same, as a fraction | ✅ |
| `cpu_util_mean` | Average CPU load, whole machine | Mean of whole-machine `psutil.cpu_percent()` samples over the clean window | With an idle GPU, shows whether CPU preprocessing is the limit | ✅ 4 |
| `cpu_util_max` | Peak CPU load, whole machine | Max of the same samples | Same | ✅ 4 |
| `nproc` | CPU cores on the host | `os.cpu_count()` | Denominator for reading the CPU numbers | ✅ |
| `cpu_cores_busy` | Cores this process tree kept busy | CPU seconds of the process and its children / wall seconds | A CPU reading that works even on very short windows | ✅ |
| `peak_mem_gb` | Peak GPU memory in use | `torch.cuda.max_memory_allocated()`, reset after warmup, read when the clean window closes | How close the config is to running out of memory | ✅ 7 |
| `reserved_mem_gb` | Peak GPU memory held by the allocator | `torch.cuda.max_memory_reserved()` | Reserved minus peak shows fragmentation | ✅ |
| `mem_total_gb` | Total memory on the card | `torch.cuda.mem_get_info()` | Denominator for headroom | ✅ |
| `headroom_gb` | Memory left before OOM | `mem_total_gb − reserved_mem_gb` | Shows room to raise the batch size | ✅ |
| `top_kernels` | The 8 GPU kernels that took the most time | Kernel durations summed by name, top 8 | Shows what the GPU actually spent time on | ✅ |
| `kernel_count` | Number of distinct kernels | Count of distinct kernel names | With `top8_coverage`, shows a long tail of small kernels | ✅ |
| `top8_coverage` | Share of kernel time in the top 8 | Top-8 time / summed time of all kernels | Says whether the top 8 tell the whole story | ✅ |
| `achieved_occupancy_pct` | How full the GPU's cores were | Each kernel's occupancy estimate, weighted by its duration | Explains why compute is slow when the GPU is busy | ✅ |
| `limiting_factor` | What capped occupancy | The limiting factor with the most kernel time behind it | Names the cause of low occupancy | ✅ |
| `provenance` | What actually ran | Substring search over kernel names: flash attention, fused kernels, dtypes | A hint that a config change took effect | ✅ |
| `device` | The real hardware | Name, memory, compute capability, SM count from the trace header | Checks the label against the actual card | ✅ |
| `device_type` | What the probe ran on | The device string (`cuda` / `cpu` / `mps`) | Stops a CPU smoke test being read as a GPU result | ✅ |
| `population` | Which rows were timed and which batch was traced | Reported by the runner: sample hash, steps per pass, reference batch key | Decides which probes can be compared with each other | ✅ |
| Ceiling trial: `peak_gb`, `reserved_gb`, `total_gb` | Memory of a memory-only trial | Allocator peaks from start until optimizer step 3 | Answers "does this batch size fit?" | ✅ |
| Ceiling trial: `shapes_seen`, `shape_checked` | Batch shapes the model was fed | Every `(batch, seq)` seen by a hook on the model | Proves the trial measured the intended worst case | ✅ |
| `gpu_seconds` | How long the probe held the device | Wall-clock from the start of `capture()` (or `Attach.install()`) to the end of the traced step | Cost = `gpu_seconds` × the card's $/hour / 3600 | ✅ 3 |

## What the PR fixed

1. **`step_time_cv`** divided by the median; a coefficient of variation is std / mean, and the mean
   is what samples/s divides by. It was also labelled the "noise floor", which under dynamic padding
   it isn't (it mostly measures batch-width spread). Now std / mean, described as exactly that.
2. **`data_wait_s`** read `0.0` when the trace had no `dataloader` span (a script without a torch
   `DataLoader`). Now `null`: unknown, not a measured zero.
3. **`cost`** was in `loop/contracts.md` but not in the code, along with a "steady-state reached?"
   flag, a `context` field and `provenance.probe_kind`. Added `gpu_seconds` (the probe's wall-clock
   on the device); removed the rest from the contract. Price is the user's to apply.
4. **`cpu_util_mean` / `cpu_util_max`** are whole-machine numbers; the README now says so and
   points to `cpu_cores_busy` for this process alone.
5. **`gap_count`** was described as "number of gaps". It counts gaps of any size between consecutive
   kernel events, so on a launch-bound step it lands near the event count. The README now says what
   it counts; what it means is left to the agent, next to `max_gap_s`.
6. **`gpu_phase_coverage`** left kernels without a correlation id out of its denominator, so an
   unattributable kernel didn't lower it. Now divides by all kernel time, as documented.
7. **`peak_mem_gb` for user scripts** was read after the traced step. Now read when the clean
   window closes, as for runner probes.

Doc slips fixed alongside: the README's container-field count (six, not five), the stale
`HFRunner.tokens_per_step` docstring, and the "real output" example is marked as predating the
current fields (it needs a GPU to regenerate).
