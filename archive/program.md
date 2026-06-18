# autotune — ms-swift Throughput Optimizer

Automated hyperparameter search for ms-swift training across multiple model sizes.
The agent runs probes, measures throughput, and finds the optimal config per model — unattended.

---

## Goal

Find the configuration that maximizes **training throughput (samples/second)** for each
model size without hitting OOM. samples/sec is the single metric — it captures GPU
utilization implicitly. If the GPU is starved by the data pipeline, samples/sec will be
low; the Phase 2 dataloader sweep will surface and fix it. No separate GPU monitoring needed.

---

## Setup

### 1. SSH into the VM

```
Host: 184.105.190.154
User: paperspace
Password: from local env var PAPERSPACE_PASSWORD
```

```bash
ssh paperspace@184.105.190.154
```

### 2. Activate the environment

```bash
source /home/paperspace/envs/swift-env/bin/activate
```

### 3. Navigate to the project root

```bash
cd Desktop/auto_tune/
```

All commands — including `python run_train.py` and log parsing — must be run from this directory.
The base config (`configs/sft_base.yaml`) and the run state (`completed_runs/runs.json`) are
resolved relative to this path.

---

## Machine context

On first run, SSH in and record:

```bash
nproc                         # total CPU cores available
free -h                       # total RAM
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
```

Log these to `machine_info.txt`. They inform safe upper bounds for worker counts
(`dataloader_num_workers` should not exceed `nproc // 2`).

---

## File locations

The agent (Claude Code) runs **locally**. The VM is a pure compute resource.
Files are owned as follows — never confuse the two:

| File | Lives on | How agent accesses it |
|---|---|---|
| `run_train.py` | **Local** (agent edits), pushed to VM before each probe | `Edit` tool locally → `scp` to VM |
| `search_state.json` | **Local** | `Read`/`Edit` tools directly |
| `results.tsv` | **Local** | `Read`/`Edit` tools directly |
| `probe.log` | VM `/tmp/probe.log`, pulled after each run | `scp` from VM → read locally |
| `configs/sft_base.yaml` | VM (read-only, never modified) | `ssh cat` to inspect if needed |

## How to run a probe

Each probe follows this exact sequence:

**Step 1 — Edit GRID locally** (Edit tool on local `run_train.py`):

```python
GRID: dict[str, list] = {
    "model":                          ["Qwen/Qwen3.5-0.8B"],
    "per_device_train_batch_size":    [4],
    "gradient_accumulation_steps":    [4],
    "dataset_num_proc":               [4],
    "dataloader_num_workers":         [4],
    "dataloader_prefetch_factor":     [2],
    "gradient_checkpointing":         [False],
    # "padding_free":                 [True],   # uncomment if extension active
}
```

**Step 2 — Push to VM** (bash):

```bash
scp run_train.py paperspace@184.105.190.154:~/Desktop/auto_tune/run_train.py
```

**Step 3 — Run training on VM** (bash):

```bash
ssh paperspace@184.105.190.154 \
  "source ~/envs/swift-env/bin/activate && \
   cd ~/Desktop/auto_tune && \
   python run_train.py --reset --max-steps 7 > /tmp/probe.log 2>&1"
```

**Step 4 — Pull log** (bash):

```bash
scp paperspace@184.105.190.154:/tmp/probe.log /tmp/probe.log
```

**Step 5 — Parse locally** (Read tool on `/tmp/probe.log`), then update `search_state.json` and `results.tsv` locally.

---

## Metric

**Throughput = `(per_device_train_batch_size × gradient_accumulation_steps) / mean_s_it`**
= `effective_batch / mean_s_it` (samples per second)

ms-swift logs one entry per **optimizer step**, and each optimizer step processes `bs × ga`
micro-batch samples. So the per-step time delta covers `ga` micro-batches — the numerator must
be `bs × ga`, not `bs`. Using `bs` alone divides true throughput by `ga`, making throughput
*look* like it falls as `ga` rises. That is a measurement artifact, not physics.

**Invariant (sanity check):** at fixed `bs`, true samples/s must be ~constant across effective
batches — `ga` cannot change steady-state per-micro-batch work. If it isn't, suspect the metric.

- Probe runs `max_steps=7`
- Skip steps 1–2 (CUDA kernel warmup, data pipeline spin-up)
- Measure `s/it` for steps 3–7 from ms-swift's step logs
- Compute `mean_s_it` over those 5 steps
- Higher throughput = better

Parse from `/tmp/probe.log` (ms-swift emits one dict per step with `--logging_steps 1`):

```
# Extract elapsed_time delta between consecutive steps to get per-step s/it
{'loss': ..., 'global_step/max_steps': '3/7', 'elapsed_time': '0:00:14.3', ...}
```

OOM detection: `grep -i "out of memory" /tmp/probe.log`
Crash detection: non-zero exit from `run_train.py` or `Traceback` in log.

---

## Parameters

### Outer loop — effective batch sizes

Run the full search (Phase 1 + extensions + Phase 2) once per effective batch size, per model.
Effective batch = `bs × ga` = samples per gradient update. This is a training quality decision
(affects gradient noise and convergence) so each value produces a separate optimal config.

**Effective batch values to evaluate: [16, 32, 64]**

Loop order: for each effective_batch → for each model_size (ascending).

### Phase 1 — Core (memory-bound, search first)


| Parameter                     | Role                                | Search space                                         |
| ----------------------------- | ----------------------------------- | ---------------------------------------------------- |
| `per_device_train_batch_size` | Samples per GPU micro-batch         | 1 → 2 → 4 → ... → effective_batch (double each step) |
| `gradient_accumulation_steps` | Derived from bs and effective batch | `ga = effective_batch / bs` (halves as bs doubles)   |


The sweep ceiling within a fixed effective batch is `bs = effective_batch` (ga=1).
At that point there is no room to double bs further without breaking the effective batch
constraint — declare Phase 1 done and move to Phase 2 (or try extensions if OOM was hit
before reaching bs = effective_batch).

Example for effective_batch=32:

```
bs=1,  ga=32  → probe
bs=2,  ga=16  → probe
bs=4,  ga=8   → probe
bs=8,  ga=4   → probe
bs=16, ga=2   → probe
bs=32, ga=1   → probe (ceiling of this effective batch)
```

### Phase 2 — Dataloader (CPU-bound, search after Phase 1)

Run Phase 2 at the optimal `bs` found in Phase 1, with the active extension flags.


| Parameter                    | Role                                          | Search space          | Notes                                             |
| ---------------------------- | --------------------------------------------- | --------------------- | ------------------------------------------------- |
| `dataset_num_proc`           | CPU processes for lazy tokenization (getitem) | [2, 4, 8, `nproc//2`] | Don't exceed `nproc - dataloader_num_workers - 2` |
| `dataloader_num_workers`     | Async prefetch workers alongside GPU          | [1, 2, 4, 8]          | Cap at `nproc//2`                                 |
| `dataloader_prefetch_factor` | Batches queued per worker                     | [2, 4, 8]             | Only meaningful when `num_workers > 0`            |


**Fixed (do not search — always wins):**

- `dataloader_pin_memory = True` — faster CPU→GPU transfers via DMA
- `dataloader_persistent_workers = True` — avoids worker restart overhead between steps

**CPU budget constraint:** `dataset_num_proc + dataloader_num_workers ≤ nproc - 2`
Reserve at least 2 cores for the main process and OS.

Phase 2 sweep order:

1. Fix `dataset_num_proc=4`, sweep `dataloader_num_workers` [1, 2, 4, 8] → find best `num_workers`
2. Fix `num_workers` at best, sweep `dataloader_prefetch_factor` [2, 4, 8]
3. Fix `prefetch_factor` at best, sweep `dataset_num_proc` [2, 4, 8, nproc//2]
4. Record the Phase 2 winner

---

## Extension protocol (ceiling-breakers)

The two extensions have different trigger conditions because they have fundamentally different natures.

**Phase 1 starts with `gradient_checkpointing=False`** to find the true throughput ceiling.

| Priority | Extension | Trigger | Throughput effect |
|---|---|---|---|
| 1 | `padding_free` | OOM **or** saturation | Positive — removes wasted compute on padding tokens, can improve throughput directly |
| 2 | `gradient_checkpointing` | OOM only | Negative — ~30% slower s/it, but higher bs may compensate |

**`padding_free` on saturation:** saturation means the GPU is compute-bound at the current bs.
`padding_free` reduces per-sample compute (no padding tokens processed), which may shift
the saturation point upward and allow the bs sweep to continue with real throughput gains.

**`gradient_checkpointing` on saturation:** never invoke — gc adds compute overhead to an
already compute-saturated GPU. Only useful when the bottleneck is memory.

**Protocol:**

```
on OOM (before bs = effective_batch):
  try padding_free if not active:
    probe(oom_bs, + padding_free)
    → no OOM: mark active, resume sweep from oom_bs
    → still OOM: mark no_effect, try next

  try gradient_checkpointing if not active:
    probe(oom_bs, + gradient_checkpointing)
    → no OOM: mark active, resume sweep from oom_bs
    → still OOM: mark no_effect — OOM ceiling is final → record best at oom_bs//2, move to Phase 2

on SATURATION:
  try padding_free if not active:
    probe(saturated_bs, + padding_free)
    → throughput > best * 1.05: mark active, reset saturation counter, resume sweep
    → no meaningful gain: mark no_effect — saturation ceiling is final → move to Phase 2
  (never try gradient_checkpointing on saturation)
```

When `gradient_checkpointing` activates, resume the bs sweep — gc at bs=N may yield better
throughput than no-gc at bs=N//2. The final recorded config reflects the best (bs, flags) found.

Active extensions carry over to larger effective batch sizes for the same model.

---

## Saturation detection

Ceiling is "saturated" (not OOM) when throughput improvement over previous bs < 3%,
for 2 consecutive doublings. Try `padding_free` extension before declaring final.
If padding_free doesn't help, record best bs and move to Phase 2.

```
bs=4,  ga=8  → 1.203 samp/sec  (+51% vs bs=2)  → continue
bs=8,  ga=4  → 1.231 samp/sec  (+2.3% vs bs=4) → non-improvement 1
bs=16, ga=2  → 1.235 samp/sec  (+0.3% vs bs=8) → non-improvement 2 → SATURATED → record bs=8
```

---

## Fixed training arguments

These live in `configs/sft_base.yaml` and do not change across any experiment.
Do not modify `sft_base.yaml`. Only the `GRID` dict in `run_train.py` changes per probe.

Key fixed values (for reference):

- dataset: `HF::nikhitrivedi1/nn_ocr4:v0_small`
- max_length: 15000 (maximum total tokens per sample — image tokens + text tokens combined; samples longer than this are truncated)
- tuner_type: full (full SFT — LoRA to be evaluated separately)
- learning_rate: 1e-4
- attn_impl: flash_attn (always on — no reason to use standard attention)
- gradient_checkpointing: false (Phase 1 baseline — only enabled as an extension on OOM)
- logging_steps: 1 (required for per-step s/it parsing)
- dataloader_pin_memory: true, dataloader_persistent_workers: true (always set)

---

## Model sizes

**Currently in scope: 0.8B only.** Additional model sizes to be added once 0.8B results are validated.

| Name | Model ID | Status |
|---|---|---|
| 0.8B | `Qwen/Qwen3.5-0.8B` | ✓ active |
| 1.5B | `Qwen/Qwen3.5-1.5B` | pending |
| 3B | `Qwen/Qwen3.5-3B` | pending |
| 7B | `Qwen/Qwen3.5-7B` | pending |
| 14B | `Qwen/Qwen3.5-14B` | pending |


OOM ceilings roughly scale with model size. If 0.8B ceiling is `bs=16`, expect 7B ceiling
around `bs=2` or `bs=4`. Use prior results to skip obviously-too-large configs.

---

## Search state

Persist `search_state.json` after every probe. If interrupted, resume from last state.
State is keyed by `(model, effective_batch)` pair.

```json
{
  "machine": { "nproc": 16, "gpu_mem_gb": 24 },
  "active_extensions": ["padding_free"],
  "runs": {
    "0.8B__16": {
      "model": "0.8B", "effective_batch": 16,
      "phase": "done",
      "tried_extensions": { "padding_free": "active", "flash_attn": "no_effect" },
      "phase1_best": { "bs": 16, "ga": 1, "throughput": 2.104 },
      "phase2_best": { "bs": 16, "ga": 1, "num_workers": 4, "prefetch_factor": 4, "dataset_num_proc": 8, "throughput": 2.341 }
    },
    "0.8B__32": {
      "model": "0.8B", "effective_batch": 32,
      "phase": "phase1",
      "current_bs": 4,
      "tried_extensions": {},
      "recent_throughputs": [0.412, 0.798, 1.203],
      "phase1_best": { "bs": 4, "ga": 8, "throughput": 1.203 }
    }
  }
}
```

---

## Results log

Append to `results.tsv` after every probe. Never modify existing rows.

```
model  eff_batch  phase  bs  ga  workers  prefetch  dataset_proc  flags           throughput  status  notes
0.8B   16         1      1   16  4        2         4             -               0.412       ok      baseline
0.8B   16         1      2   8   4        2         4             -               0.798       ok      +93%
0.8B   16         1      4   4   4        2         4             -               1.203       ok      +51%
0.8B   16         1      8   2   4        2         4             -               0.000       oom     hit ceiling, try extensions
0.8B   16         1      8   2   4        2         4             padding_free    1.891       ok      ext broke ceiling
0.8B   16         1      16  1   4        2         4             padding_free    2.104       ok      bs=eff_batch, phase1 done
0.8B   16         2      16  1   4        2         4             padding_free    2.341       ok      dataloader tuned
0.8B   16         -      -   -   -        -         -             -               -           done    optimal: bs=16 ga=1 w=4 pf=4 p=8
0.8B   32         1      1   32  4        2         4             padding_free    0.389       ok      baseline (pf carried over)
0.8B   32         1      2   16  4        2         4             padding_free    0.741       ok      ...
```

---

## The experiment loop

```
SETUP:
  ssh in → record machine_info (nproc, GPU mem)
  read search_state.json (or init if first run)

FOR EACH effective_batch in [16, 32, 64]:
  FOR EACH model_size in [0.8B, 1.5B, 3B, 7B, 14B]:

    skip if state[model][effective_batch].phase == "done"

    carry over active_extensions from smaller effective_batch runs of same model
    (if padding_free was active for eff_batch=16, start with it active for eff_batch=32)

    PHASE 1 — bs sweep:
      bs = 1 (or resume from state)
      saturation_count = 0
      LOOP:
        ga = effective_batch / bs
        probe(model, bs, ga, active_extensions, dataloader_defaults)
        log to results.tsv, save search_state.json

        if OOM and bs < effective_batch:
          run extension_protocol(oom_bs=bs)
          if extension unlocks: continue loop at same bs with new flags
          if no extension works: record best (bs//2, ga*2), break to Phase 2

        if bs == effective_batch (ga=1 and sweep complete):
          record best bs from sweep, break to Phase 2

        if throughput improved > 3%:
          saturation_count = 0, bs *= 2, continue

        else:
          saturation_count += 1
          if saturation_count >= 2: record best bs, break to Phase 2
          else: bs *= 2, continue

    PHASE 2 — dataloader sweep at Phase 1 best config:
      sweep num_workers [1, 2, 4, 8] (cap: nproc//2)
      fix best, sweep prefetch_factor [2, 4, 8]
      fix best, sweep dataset_num_proc [2, 4, 8, nproc//2]
      constraint: dataset_num_proc + num_workers ≤ nproc - 2
      log each, save state

    mark state[model][effective_batch].phase = "done"

PRINT SUMMARY TABLE when all (model, effective_batch) pairs are done.
```

**Never pause to ask the user if you should continue.** The user may be away. Run until
all runs are marked `done` or until manually interrupted. On restart, resume from `search_state.json`.

---

## Final output

When all runs are done, print and save `optimal_configs.md`:

```markdown
# Optimal training configs

| Model | Eff batch | bs | ga | workers | prefetch | dataset_proc | flags        | throughput   |
|-------|-----------|----|----|---------|----------|--------------|--------------|--------------|
| 0.8B  | 16        | 16 | 1  | 4       | 4        | 8            | padding_free | 2.401 samp/s |
| 0.8B  | 32        | 16 | 2  | 4       | 4        | 8            | padding_free | 2.198 samp/s |
| 0.8B  | 64        | 16 | 4  | 4       | 4        | 8            | padding_free | 2.071 samp/s |
| 1.5B  | 16        | 8  | 2  | 4       | 4        | 8            | padding_free | 1.204 samp/s |
| ...   |           |    |    |         |          |              |              |              |
```

This table is the direct input to your production training runs.
Pick the effective batch row that matches your training quality requirements.