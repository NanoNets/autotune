# Autotune search space

## Phase 1 — core (memory-bound, run first)

| Parameter | Search values | Strategy | Notes & dependencies |
|---|---|---|---|
| `per_device_train_batch_size` | 1 → 2 → 4 → 8 → 16 → ... | Binary sweep upward | Primary throughput lever. Double each step until OOM or saturation. Triggers extension protocol when ceiling hit. |
| `gradient_accumulation_steps` | Derived: `effective_batch / bs` | Not searched | Held so `bs × ga = effective_batch` (16, 32, or 64). Outer loop sets effective_batch; ga is derived per probe. |

## Extensions — ceiling-breakers (invoked reactively, in this order)

| Parameter | Search values | Trigger | Throughput effect | Notes |
|---|---|---|---|---|
| `padding_free` | False → True | OOM **or** saturation | Positive — removes wasted compute on padding | Free win. On saturation: reduces per-sample compute, may shift the saturation ceiling upward. Always try before gc. |
| `gradient_checkpointing` | False → True | OOM only | ~30% slower s/it | Trades compute for memory. Never invoke on saturation — adds overhead to an already compute-saturated GPU. Resume bs sweep after activating — higher bs may recover the loss. |

## Phase 2 — dataloader (CPU-bound, run at optimal bs from Phase 1)

| Parameter | Search values | Strategy | Notes & dependencies |
|---|---|---|---|
| `dataloader_num_workers` | [1, 2, 4, 8] | Sweep first, pick best | Async prefetch workers running alongside GPU. Cap at `nproc // 2`. Larger bs → GPU takes longer per batch → fewer workers needed to keep up. |
| `dataloader_prefetch_factor` | [2, 4, 8] | Sweep after num_workers fixed | Batches queued per worker. Total prefetched = `num_workers × prefetch_factor`. Only meaningful when `num_workers > 0`. |
| `dataset_num_proc` | [2, 4, 8, nproc//2] | Sweep last | CPU processes for lazy tokenization (getitem). **CPU budget constraint:** `num_proc + num_workers ≤ nproc − 2`. |

## Fixed — set once, never searched

| Parameter | Value | Reason |
|---|---|---|
| `attn_impl` | flash_attn | Fused attention kernel — lower memory and faster than standard attention on all modern NVIDIA GPUs. No reason to use anything else. |
| `gradient_checkpointing` | False (baseline) | Phase 1 starts with gc=False to find the true throughput ceiling. Only enabled as an extension when OOM persists after padding_free. |
| `dataloader_pin_memory` | True | Page-locked CPU RAM → GPU DMA transfer without extra OS copy. Always a win on GPU. |
| `dataloader_persistent_workers` | True | Workers stay alive between steps, avoiding ~0.5–1s respawn cost per step. |
| `max_length`, `lora_rank`, `lr`, ... | See program.md | Training quality hyperparameters held constant so throughput differences are attributable only to searched params. |
