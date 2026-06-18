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

## Results — Qwen3.5-0.8B on 80GB GPU

Throughput is `effective_batch / mean_s_it` (= `bs × ga / s_it`). An earlier version divided
by `bs` alone, which crowned `bs=16 + gradient_checkpointing` — in fact the **slowest** config
measured. Corrected ranking (`reported × ga`, see `optimal_configs.md` / `notes.md`):

| Effective batch | bs | ga | gc  | Throughput |
|---|---|---|---|---|
| 16 | 2 | 8  | off | ~0.59 samp/s |
| 32 | 2 | 16 | off | ~0.54 samp/s |
| 64 | 2 | 32 | off | ~0.58 samp/s |

The GPU saturates by `bs=2`; pushing `bs` higher buys nothing, and the `gradient_checkpointing`
it forced on only added ~30% recompute. `padding_free` alone fits bs=2 — no gc needed. True
throughput is flat across effective batches (it's GPU-bound); dataloader tuning added <2%.

Pick the effective batch row that matches your training quality requirements (larger effective batch = less gradient noise).

---

## Production configs

Drop these overrides on top of your `sft_base.yaml`:

<details>
<summary>eff_batch=16</summary>

```yaml
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
gradient_checkpointing: false
padding_free: true
dataloader_persistent_workers: true
```
</details>

<details>
<summary>eff_batch=32</summary>

```yaml
per_device_train_batch_size: 2
gradient_accumulation_steps: 16
gradient_checkpointing: false
padding_free: true
dataloader_persistent_workers: true
```
</details>

<details>
<summary>eff_batch=64</summary>

```yaml
per_device_train_batch_size: 2
gradient_accumulation_steps: 32
gradient_checkpointing: false
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
