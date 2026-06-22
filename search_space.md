# Autotune search space (v1)

Knobs the loop may set = **ms-swift `swift sft` CLI args only** — nothing custom. Two buckets:
throughput-safe (free to search) and off-limits (training-semantic — never touch). Effective batch
is a **fixed input**; the loop only chooses the `bs × ga` factorization that hits it.

## Throughput-safe knobs (searchable)

| Parameter | Search values | Strategy | Notes & dependencies |
|---|---|---|---|
| `per_device_train_batch_size` (bs) | 1 → 2 → 4 → 8 → … | Binary sweep upward | Primary throughput lever. Double until OOM or saturation; triggers the extension protocol at the ceiling. |
| `gradient_accumulation_steps` (ga) | Derived: `effective_batch / bs` | Not searched | Held so `bs × ga = effective_batch`. The user fixes effective_batch; ga is derived per probe. |
| `packing` | False → True | Throughput lever | Concatenates samples to fill `max_length` — fewer short/padded steps, higher token throughput on variable-length data. Interacts with `padding_free`. |
| `padding_free` | False → True | OOM **or** saturation | Removes wasted compute on pad tokens (free throughput win + memory headroom). Always try before gc. |
| `attn_impl` | flash_attn (default) | Set once | Fused attention — lower memory + faster on all modern NVIDIA GPUs. flash_attn is the default win; only change if the model lacks a kernel. |
| `gradient_checkpointing` | False → True | OOM only | ~30% slower s/it; trades compute for memory to fit a bigger bs. Never invoke on saturation. |
| `vit_gradient_checkpointing` | False → True | OOM only (VLM) | Same trade for the vision tower (Qwen2.5-VL / OCR). Checkpoint the ViT to fit image tokens when the model is image-token-memory-bound. |
| `dataloader_num_workers` | [1, 2, 4, 8] | Reactive — only if data-bound | Async prefetch workers. Add ONLY when the GPU is idle (data-bound) **and** CPU has headroom. Cap at `nproc // 2`. If CPU is pinned ~100% it's a preprocessing wall — more workers won't help. |

## Off-limits — training-semantic (never tune)

These change *what model you get*, not throughput. Hard block — the loop must never set them.

| Parameter | Why off-limits |
|---|---|
| `learning_rate`, `vit_lr`, `aligner_lr` | Optimization dynamics / final quality |
| `weight_decay`, `warmup`, `dropout` | Regularization / convergence |
| `max_length` | Data semantics — truncation changes the task |
| `seed` | Reproducibility, not throughput |
| `bf16` | Numerical precision / quality |
| effective-batch target | Gradient noise scale — fixed by the user as an input |

## Fixed (set once, not searched — always wins)

- `dataloader_pin_memory = True` — page-locked CPU→GPU DMA transfers.
- `dataloader_persistent_workers = True` — workers survive between steps (no respawn cost).
- `logging_steps = 1` — required for per-step parsing.
- `include_num_input_tokens_seen = True` — emits the token count → tokens/s.

## Deferred (not in v1)

`dataloader_prefetch_factor`, `dataset_num_proc` — secondary dataloader knobs from the fuller plan.
v1 tunes only `dataloader_num_workers`, and only reactively when data-bound.
