# Autotune Results: Qwen/Qwen3.5-0.8B

**Machine:** 1x A100 80GB, 16 vCPUs  
**Extensions active on all runs:** `padding_free=true`, `gradient_checkpointing=true`

## Optimal Configurations

| eff_batch | bs | ga | workers | prefetch | dataset_proc | throughput (samples/s) | vs phase1_best |
|-----------|----|----|---------|----------|--------------|------------------------|----------------|
| 16        | 16 | 1  | 1       | 4        | 4            | 0.1691                 | +0.0%          |
| 32        | 16 | 2  | 8       | 2        | 4            | 0.0836                 | +0.2%          |
| 64        | 16 | 4  | 2       | 2        | 4            | 0.0455                 | +0.2%          |

## Key Findings

**Phase 1 (batch size sweep):**
- Max safe `per_device_train_batch_size` is **16** across all effective batch sizes.
- bs=32 OOM'd even with both extensions active; ceiling set at bs=16.
- OOM at bs=4 for eff_batch=16 without gradient_checkpointing prompted extension activation.
- Throughput roughly halves as eff_batch doubles (GPU-bound regime).

**Phase 2 (dataloader sweep):**
- Throughput is essentially flat across all worker/prefetch/dataset_proc values (max delta ~2%).
- GPU is the bottleneck; dataloader tuning provides no meaningful gain.
- Default `dataset_num_proc=4` was optimal in all cases.
- Higher `num_workers` helped marginally at eff_batch=32 (w=8 best), but hurt at eff_batch=64.

## swift sft Overrides

For production training, add these overrides on top of `sft_base.yaml`:

### eff_batch=16
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

### eff_batch=32
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

### eff_batch=64
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
