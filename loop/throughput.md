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

**Precision is the human's, not a memory lever.** IF nothing fits — `micro_batch=1` OOMs at
`max_len` — the safe moves are recompute, offload, sharding, 8-bit optimizer *states*, fused/chunked
CE. **NEVER** drop `dtype` to make a config fit: it changes the model you get, so it is a decision to
put to the human with the measured numbers, not one the loop may take. (`dtype` appears in the
ceiling's staleness list only because a human changing it invalidates the measurement.)

## The dynamic-padding guarantee

`padding="longest"` is safe **iff `micro_batch <= ceiling(max_len)`**. Every real batch is at most
`max_len` wide, so a config that fits the synthetic worst case fits *every* batch the dataset can
produce — including the long rows a short profiling sample never drew.

**NEVER raise `micro_batch` above the ceiling on the strength of a probe's `headroom_gb`.** Under
dynamic padding a probe's peak reflects the rows it happened to draw, not the dataset's longest, so
the headroom it reports is an artifact of the sample. Measured on `test_a10G`: the scan saw max 520
tok against `max_len=2048`, so no probe in 23 experiments ever ran the `[8, 2048]` batch the shipped
config will meet in training — only the ceiling did (17.4GB alloc / 20.7GB reserved of 22.1GB; 9
OOMs). The ceiling bounds `micro_batch`; an observed peak never does.

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
