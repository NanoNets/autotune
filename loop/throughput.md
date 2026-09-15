# Objective: Training Throughput

**Maximize** training throughput (tokens/second) on the given hardware, **without degrading quality**.
Wall-clock for a fixed run = total_tokens / tokens_per_s, so higher throughput = shorter training.

- **Primary metric:** tokens/s, reported as **MFU** (% of the hardware roofline).
- **Cross-check:** samples/s.
- **Quality guarantee:** by construction — only knobs that change speed, not the model, are searchable.

## Safe vs off-limits
The agent classifies each knob at Frame (conservative: unsure → off_limits). Guidance:

- **Safe** (change speed, not the model): batch factorization, sharding/parallelism, recompute, offload, kernel swaps (verified numerically-safe), packing, compile, dataloader, tf32.
- **Off-limits** (change the model): learning rate, weight decay, warmup, dropout, optimizer *algorithm*, precision (bf16/fp16/FP8-train), `max_len`, seed, LoRA rank, and the effective_batch target.

`max_len` is off-limits *and* load-bearing: it is the length the OOM ceiling is measured at
(`frame/README.md`). Lowering it to make a config fit is changing the job, not tuning it.

## Canonical levers
Framework-agnostic. Frame projects these onto what the framework supports.

| Family | Levers |
|---|---|
| Batch | `micro_batch` (bounded above by the measured ceiling at `max_len`), `grad_accum` (derived to hold effective_batch) |
| Parallelism | `dp`, `shard_degree` (FSDP/ZeRO 1-3), `tp`, `pp`, `sequence_parallel`, `expert_parallel`; mesh: dp×shard×tp×pp = world_size; comm overlap |
| Memory/recompute | `recompute` (full/selective), activation offload, optimizer/param offload, 8-bit optimizer states |
| Kernels | `attn_kernel` (flash/sdpa/FA3), fused kernels (Liger), fused optimizer, chunked cross-entropy |
| Compile | `compile` (off/default/max-autotune), CUDA graphs |
| Sequence | `packing`, `padding_free`, length-grouped batching |
| Dataloader | `workers`, `prefetch`, `num_proc` |
| Precision (safe subset) | `tf32` |

Applicability is model- and framework-dependent (Liger needs supported layers; FA3 needs Hopper; TP needs the framework to expose it). Frame decides which apply; the surrogate predicts effects; probes verify.
