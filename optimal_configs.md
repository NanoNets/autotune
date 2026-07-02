# Optimal training configs — Qwen3.5-0.8B, heavy_256

Machine: 8× H100 80GB HBM3, 64 CPU cores, 1.6TB RAM
Dataset: HF::nikhitrivedi1/nn_ocr4:heavy_256
Base config: configs/lora_base.yaml (lora_rank=512, max_length=20000, pf+gc baked in)
Tuner: LoRA (rank 512, alpha 256, target all-linear)

## Memory ceiling

Per-device batch size ceiling: **bs=1** (bs=2 causes stochastic OOM — heavy_256 samples
can reach ~48–49 GiB per GPU, exceeding the 79 GiB limit on worst-case batches).
Both padding_free and gradient_checkpointing are already active in lora_base.yaml.

## Optimal dataloader config (same for all effective batches)

| Parameter              | Value | Notes                              |
|------------------------|-------|------------------------------------|
| dataloader_num_workers | 2     | Fewer workers = less CPU contention|
| dataloader_prefetch_factor | 4 | Ceiling slightly better than 2     |
| dataset_num_proc       | 16    | Peak at 16, drops at 8 and 32      |
| dataloader_pin_memory  | true  | Fixed (always wins)                |
| dataloader_persistent_workers | true | Fixed (always wins)         |

## Results by effective batch size

| Model | Eff batch | bs | ga | workers | prefetch | dataset_proc | flags | throughput (samp/sec/GPU) |
|-------|-----------|----|----|---------|----------|--------------|-------|--------------------------|
| 0.8B  | 16        | 1  | 2  | 2       | 4        | 16           | pf+gc | 0.0174                   |
| 0.8B  | 32        | 1  | 4  | 2       | 4        | 16           | pf+gc | 0.0086                   |
| 0.8B  | 64        | 1  | 8  | 2       | 4        | 16           | pf+gc | 0.0042                   |

## Notes

- Throughput is `per_device_train_batch_size / mean(s/it for steps 3–7)`, per GPU
- Total cluster throughput = throughput × 8 GPUs × ga (samples per gradient update per second):
  - eff_batch=16: 0.0174 × 8 × 2 = 0.278 samples/sec
  - eff_batch=32: 0.0086 × 8 × 4 = 0.275 samples/sec
  - eff_batch=64: 0.0042 × 8 × 8 = 0.269 samples/sec
- All three effective batch sizes give near-identical total cluster throughput (~0.275 samp/sec)
  — the choice between them is purely a training quality/convergence decision
- The dataloader Phase 2 gain was small (+2–7%) because heavy_256's bottleneck is GPU compute
  (large image token sequences), not CPU data loading
- Phase 2 optimal (w=2, pf=4, dp=16) was stable across all three eff_batch values,
  confirming it is driven by bs=1, not ga

## Max_length throughput study — eff_batch=16

Goal: compare throughput at 20K vs 15K to evaluate whether reducing max_length is worthwhile.
`truncation_strategy=right` used throughout (samples truncated not deleted; required for
heavy_256 at 15K since too many samples exceed 15K and deletion empties some shards).

All configs: Qwen3.5-0.8B, heavy_256, LoRA rank=512, ZeRO-2, pf+gc baked in.
True throughput = (ga × bs) / s_per_it — metric alone (bs/s_it) undercounts when ga > 1.

| config | max_length | bs | ga | s/it | metric tp | true tp/GPU | peak mem | 100K time |
|--------|------------|----|----|------|-----------|-------------|----------|-----------|
| 20K baseline | 20000 | 1 | 2 | 57.5s | 0.0174 | 0.0348 | ~49 GiB | ~100 hrs |
| 15K bs=2/ga=1 | 15000 | 2 | 1 | 85.8s | 0.0233 | 0.0233 | 44.71 GiB | ~149 hrs |
| **15K bs=1/ga=2** | **15000** | **1** | **2** | **49.6s** | **0.0202** | **0.0403** | **24.44 GiB** | **~86 hrs** |

**Key findings:**
- **15K/bs=1/ga=2 is the fastest config** — 14% faster than 20K baseline (86 vs 100 hrs),
  42% faster than 15K/bs=2/ga=1.
- The `metric throughput` (bs/s_it) was misleading: 15K/bs=2 looked better (0.0233 > 0.0174)
  but was actually the slowest because ga=1 vs ga=2 wasn't accounted for.
- 15K/bs=1/ga=2 is faster than 20K/bs=1/ga=2 because each micro-batch processes fewer
  tokens (~15K vs ~20K), making individual passes faster.
- Memory at bs=1/ga=2 (24.44 GiB) is dramatically lower than bs=2/ga=1 (44.71 GiB) —
  gradient accumulation keeps each micro-batch small; both samples are never in memory
  simultaneously.
- bs=2 at 15K packs ~30K tokens into one micro-batch (2 × 15K), which is slower than
  two sequential 15K passes via ga=2 due to super-linear recomputation cost in gc.

**Recommendation: `max_length: 15000`, `bs=1`, `ga=2`, `truncation_strategy: right`**
- 14% faster end-to-end training vs 20K
- 20 GiB less peak memory per GPU (headroom for future bs experiments)
- truncation_strategy=right required: heavy_256 has so many >15K samples that deletion
  empties some data shards, crashing training

Note: truncation_strategy=right preserves image tokens for multimodal; only text/response
tail is cut. Max generation length at inference is unaffected (controlled by max_new_tokens).

## GRID to use for production runs

```python
# eff_batch=16 (smallest gradient noise, best for early training)
GRID = {
    'model':                        ['Qwen/Qwen3.5-0.8B'],
    'per_device_train_batch_size':  [1],
    'gradient_accumulation_steps':  [2],
    'dataloader_num_workers':       [2],
    'dataloader_prefetch_factor':   [4],
    'dataset_num_proc':             [16],
}

# eff_batch=32
GRID = {
    'model':                        ['Qwen/Qwen3.5-0.8B'],
    'per_device_train_batch_size':  [1],
    'gradient_accumulation_steps':  [4],
    'dataloader_num_workers':       [2],
    'dataloader_prefetch_factor':   [4],
    'dataset_num_proc':             [16],
}

# eff_batch=64 (largest gradient batch, most stable gradients)
GRID = {
    'model':                        ['Qwen/Qwen3.5-0.8B'],
    'per_device_train_batch_size':  [1],
    'gradient_accumulation_steps':  [8],
    'dataloader_num_workers':       [2],
    'dataloader_prefetch_factor':   [4],
    'dataset_num_proc':             [16],
}
```
