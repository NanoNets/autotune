# auto_tune

> Applying Karpathy's Auto-Research loop to the unglamorous-but-critical problem of training throughput.

When you're running LLM fine-tuning jobs, a poorly configured training run can leave 2–4x throughput on the table — entirely due to parameters like batch size, gradient accumulation, and dataloader workers. Finding the sweet spot manually means hours of trial-and-error OOM crashes and log-parsing. **auto_tune** eliminates that entirely.

It implements an agentic experiment loop — the same "model proposes, environment evaluates, model updates" pattern behind Auto-Research — but applied to a single, well-defined objective: **maximize samples/second on your GPU without OOM**. Claude Code acts as the agent, SSHing into your training VM, running short probes, reading the logs, and autonomously navigating the search space until it finds the optimal config. No babysitting required.

---

## The loop

```
observe state → propose next config → run probe → measure throughput → update state → repeat
```

Each probe is just 7 training steps — enough to get a stable throughput signal, fast enough to run dozens without burning your GPU budget. The agent handles OOM recovery, extension activation, and saturation detection automatically.

**Phase 1 — Memory sweep** finds the largest batch size that fits in GPU memory, doubling each step. When it hits a ceiling it tries memory-saving extensions (`padding_free`, `gradient_checkpointing`) before accepting the limit.

**Phase 2 — CPU sweep** locks in the best batch size and sweeps dataloader parameters to eliminate any CPU-side pipeline stall. If the GPU is already the bottleneck (common), this phase confirms it in a few probes and moves on.

The result is a `swift sft` config you can drop directly into production.

---

## Results — Qwen3.5-0.8B on H100 80GB

| Effective batch | bs | ga | workers | prefetch | dataset_proc | Throughput |
|---|---|---|---|---|---|---|
| 16 | 16 | 1 | 1 | 4 | 4 | 0.169 samp/s |
| 32 | 16 | 2 | 8 | 2 | 4 | 0.084 samp/s |
| 64 | 16 | 4 | 2 | 2 | 4 | 0.046 samp/s |

Both `padding_free` and `gradient_checkpointing` were required to reach bs=16 — without them the run OOMs at bs=4. Dataloader tuning added <2%, confirming the GPU is the bottleneck on this model/hardware combination.

Pick the effective batch row that matches your training quality requirements (larger effective batch = less gradient noise).

---

## Production configs

Drop these overrides on top of your `sft_base.yaml`:

<details>
<summary>eff_batch=16</summary>

```yaml
per_device_train_batch_size: 16
gradient_accumulation_steps: 1
dataloader_num_workers: 1
dataloader_prefetch_factor: 4
dataset_num_proc: 4
gradient_checkpointing: true
padding_free: true
dataloader_persistent_workers: true
```
</details>

<details>
<summary>eff_batch=32</summary>

```yaml
per_device_train_batch_size: 16
gradient_accumulation_steps: 2
dataloader_num_workers: 8
dataloader_prefetch_factor: 2
dataset_num_proc: 4
gradient_checkpointing: true
padding_free: true
dataloader_persistent_workers: true
```
</details>

<details>
<summary>eff_batch=64</summary>

```yaml
per_device_train_batch_size: 16
gradient_accumulation_steps: 4
dataloader_num_workers: 2
dataloader_prefetch_factor: 2
dataset_num_proc: 4
gradient_checkpointing: true
padding_free: true
dataloader_persistent_workers: true
```
</details>

---

## Search space

| Phase | What's searched | Strategy |
|---|---|---|
| Phase 1 | `per_device_train_batch_size` | Binary sweep: 1→2→4→8→… until OOM or saturation |
| | `gradient_accumulation_steps` | Derived: `ga = effective_batch / bs` |
| Extensions | `padding_free` | Triggered on OOM or throughput saturation |
| | `gradient_checkpointing` | Triggered on OOM only (never on saturation — adds compute overhead) |
| Phase 2 | `dataloader_num_workers` | Swept first at optimal bs |
| | `dataloader_prefetch_factor` | Swept after num_workers is fixed |
| | `dataset_num_proc` | Swept last; constrained to `num_proc + num_workers ≤ nproc − 2` |
| Fixed | `attn_impl=flash_attn`, `pin_memory=true`, `persistent_workers=true` | Always-on; no reason to search |

---

## Models

| Model | Status |
|---|---|
| `Qwen/Qwen3.5-0.8B` | Done |
| `Qwen/Qwen3.5-1.5B` | Pending |
| `Qwen/Qwen3.5-3B` | Pending |
| `Qwen/Qwen3.5-7B` | Pending |
| `Qwen/Qwen3.5-14B` | Pending |

---

## Files

| File | Purpose |
|---|---|
| `run_train.py` | Probe script — agent edits the `GRID`, pushes to VM, and runs it |
| `search_state.json` | Persistent state; agent resumes here if interrupted |
| `results.tsv` | Append-only log of every probe |
| `optimal_configs.md` | Final output table of best configs per model × effective batch |
| `program.md` | Full algorithm spec — the agent's instructions |
| `search_space.md` | Parameter reference and search strategy |
| `check_token_lengths.py` | Utility for inspecting dataset token length distribution |
