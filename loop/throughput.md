# Objective: Training Throughput

**Maximize** training throughput on the given hardware, **without degrading quality**.
`effective_batch` is human-owned and fixed, so wall-clock for a fixed run = total_samples /
samples_per_s — a higher samples/s is strictly shorter training, whatever the tokens do.

- **Primary metric:** **samples/s**, the Observation's `samples_per_s`: total rows over total time
  across the clean window, never over the median step.
- **Read alongside it:** tokens/s — not a tie-breaker, a *discriminator*. The two respond to
  different changes, and the pair tells you which one you actually made.
- **Diagnostic only:** MFU (% of the measured roofline). Grounds a number against the hardware;
  never the thing being maximized, and never a ranking key.
- **Quality guarantee:** by construction — only knobs that change speed, not the model, are searchable.

## Why samples/s, and why tokens/s still stays on the page

`tokens/s` counts pad tokens as work, so it **rewards computing padding** — and removing padding has
been the single largest real lever in every run so far. Measured on `test_a10G`: `micro_batch` 4 → 8
was **+15% per token and -14% on samples/s** (`beliefs/micro-batch-interacts-with-dynamic-padding.md`).
One experiment, opposite verdicts. Rank on samples/s.

The converse is equally real, which is why tokens/s is not deleted. Under `padding="longest"` batch
width *is* the rows drawn, so `samples/s` is **sample-sensitive**: on
`smollm2_1.7b_a100_wikitext` the same config moved **1.9x** (264.4 → 138.1) on nothing but the
probe's row sample, while tokens/s moved **1.3%**
(`beliefs/probe-sample-must-match-length-distribution.md`).

**Packing.** A packed row holds several examples, so samples are counted from document boundaries
(`position_ids` resets or `cu_seqlens`) when a user's script passes them, and per row otherwise.
A script that packs without boundary markers reads low on samples/s; `sample_source` in the
Observation's `population` says how the count was made (`profiler/README.md`, "Attached capture").

| what moved | tokens/s | samples/s | reading |
|---|---|---|---|
| a real speed knob | up | up | genuine win |
| padding removed at the loader | down (less pad counted as work) | up | genuine win — believe samples/s |
| the probe's row sample | ~flat | moves a lot | measurement artifact, not a result |

**Neither metric is safe alone.** Rank on samples/s; read tokens/s to confirm the change you made is
the change you measured.

## MFU is a diagnostic: it is pad-inflated

It is computed from tokens/step, which counts pad, so it is comparable only *within* one padding
regime. `qwen3.5_2b_a100_wikitext` under static padding reports MFU 49% / GPU-active 100% —
genuinely efficient per token while computing 6.8x too many of them.

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
