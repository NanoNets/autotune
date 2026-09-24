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

## The ceiling is a bound, not a guarantee

`micro_batch > ceiling(max_len)` cannot run: every real batch is at most `max_len` wide, so a
config that OOMs on the synthetic worst case OOMs in training. `micro_batch <= ceiling` is
necessary, not sufficient. Measured on `qwen3.5_2b_a100_wikitext`: the ceiling passed
`mb=8+recompute` at 34.17GB and `mb=4` at 36.45GB, both OOM'd on real data at the same shape, and
the shipped `mb=16` config peaked *below* its ceiling
(`knowledge/synthetic-ceiling-underestimates-real-peak.md`).

The cause is arithmetic, and fixed in `frame/ceiling.py`: ceiling trials ran at `grad_accum=1`,
and with `zero_grad(set_to_none=True)` that leaves no gradient buffer resident at the peak. Any
config with `grad_accum >= 2` carries one — params × 2 bytes, 3.51GB on this 1.88B model, which
is the measured gap (exp 21: +3.51GB; exp 23 at `grad_accum=1`: no gap). Trials now accumulate
twice whenever the real config accumulates at all. The fix is predicted, not yet measured on a GPU,
so until a run confirms it: **before a config is called the champion, run it once on the real
runner at `padding="max_length"` at `max_len`.** Every batch is then `[micro_batch, max_len]`, the
widest shape training meets. Default `warmup` plus a couple of clean steps is enough — optimizer
state lands on the first `opt.step()`, the grad buffer is resident from micro-step 2. Log it to
`results.tsv` with `WORST-CASE PROOF` in the conclusion: it is a fit check, never a champion
candidate and never on the frontier. Under `chunked_ce` it is the *shape* worst case only — real
rows are mostly pad and chunked CE drops pad before chunking, while the synthetic trial's labels
are all real — so there the synthetic ceiling stays the bound on the CE term and both are required
(exp 23 sat 2.2GB below its ceiling for exactly this reason).

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
