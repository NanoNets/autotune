# frameworks/

The **actuator**. A *runner* actually runs one training step of a config, so the profiler can time
it and turn it into an Observation. Frame says *what we're working with*, the profiler says *how a
config ran* — a runner is the thing in between that makes a config **run**.

- `torch.py` — a random-data MLP fixture (no model/data download; exercises the whole
  capture→trace→diagnose pipeline on CPU/MPS/CUDA).
- `hf.py` — the **real** adapter: an actual `transformers` causal-LM trained with a raw PyTorch
  loop over real data.

## The runner interface

Anything the profiler drives implements:

| Member | Does |
|---|---|
| `setup(config)` | build the model + optimizer + data; set `self.params` |
| `step()` | run **one** optimizer step, each phase wrapped in `record_function("dataloader"/"forward"/"backward"/"optimizer")` |
| `tokens_per_step()` | tokens processed per optimizer step (the D in the roofline) |
| `.params` | parameter count (the N in the roofline) |

The profiler is unchanged across runners — it just calls these.

## Run the real adapter

```bash
python -m profiler --framework hf --device cuda --gpu A10G \
    --model Qwen/Qwen2.5-0.5B --dataset Salesforce/wikitext --name wikitext-2-raw-v1 \
    --dtype bfloat16 --effective-batch 16 --micro-batch 4 --seq 1024
```

Needs torch + a GPU, `transformers`, `datasets`. Loads real weights, so it's heavier than the fixture.

## Knobs — the tunable surface

`knobs()` reports the **safe** levers the loop may turn — speed only, nothing that changes the
learned model (per `loop/throughput.md`). Availability (e.g. flash-attn installed) is gated by
frame's `versions()`.

| Knob | Effect |
|---|---|
| `micro_batch` | per-step batch; more = better GPU fill, more memory. `grad_accum` is **derived** (`effective_batch // micro_batch`) so the effective batch is held fixed |
| `gradient_checkpointing` | recompute activations in bwd — less memory, more compute |
| `attn_implementation` | `eager` / `sdpa` / `flash_attention_2` |
| `torch_compile` | `False` / `"default"` / `"reduce-overhead"` / `"max-autotune"` / `"max-autotune-no-cudagraphs"`. **Both cudagraph-enabling modes are broken with `grad_accum`** — the backward's saved activations live in the graph pool and the next micro-batch overwrites them (`cudagraph_mark_step_begin()` does not fix it) |
| `tf32` | TF32 matmuls on Ampere+ — faster matmul, model unchanged. **Inert on a pure-bf16 config**: no fp32 matmul exists to accelerate |
| `padding` | `"max_length"` (every batch `seq_len` wide) or `"longest"` (pad to the batch's own longest, rounded to `pad_multiple`). Identical arithmetic — pad is masked in attention and `-100` in labels — so it only changes how much compute is spent. `seq_len` still truncates, so `max_len` is respected |
| `length_grouped` | batch similar-length rows together so pad-to-longest wastes less. Only meaningful with `padding="longest"`, and it is what makes a **larger** `micro_batch` pay there |

**Not knobs** (run *parameters*, set by the human, off-limits to search): `dtype` (precision) and
`seq_len` (max_length) change the model, so they're config inputs, not searchable. `effective_batch`
is the human's target — the loop only **factorizes** it into `micro_batch`×`grad_accum`, never changes it.

**Deferred levers** (in `throughput.md`, not yet implemented here): packing / `padding_free`,
dataloader `workers`/`prefetch`, fused kernels (incl. chunked cross-entropy), offload, and
parallelism (multi-GPU). `knobs()` reports only what this
adapter actually applies — a declared knob must really take effect.

## Design notes

- **Raw PyTorch loop, not HF `Trainer`.** The profiler needs an explicit `step()` with the phases
  wrapped in `record_function`; `Trainer` hides the loop.
- **Real weights, real data.** Unlike frame (config-only), the adapter downloads the model and streams
  the dataset — a genuine fwd/bwd/opt is what gets timed.
- **`grad_accum` is inside one `step()`** — grad_accum micro fwd/bwd passes (loss scaled by
  `1/grad_accum`), then one optimizer step. `tokens_per_step = effective_batch · seq_len`.
- **Padding is masked** in the labels (`-100`) and attention, so a padded step is still a valid
  training step. Fully blank/whitespace rows are **dropped** (an all-pad batch has all-`-100`
  labels → `0/0` NaN loss, which would poison every later step); a run with too few usable rows
  fails loudly in `setup` rather than mid-`step`.
- **Pure vs live split**, like frame/profiler: `knobs()` + `_resolve()` (defaults + token math) import
  and test on bare `python3`; `setup`/`step` lazily import torch/transformers/datasets.

## Limitations (deferred)

- **Pre-tokenized in-memory loader** → the `dataloader` phase is ~0 (no real data-wait), so
  `workers`/`prefetch` have nothing to recover. On-the-fly tokenization + packing come later.
- **`tokens_per_step()` is measured, not assumed** — the mean over steps actually run. Under
  `padding="longest"` the batch width varies, so `effective_batch * seq_len` overstates the work and
  inflates both tokens/s and MFU. Under `padding="max_length"` the measured value equals that
  product. **Compare padding modes on samples/s**, which counts the same work either way.
- Single-GPU; `AdamW` only; the LR is a fixed placeholder (it doesn't affect step timing).

## Status

Built; `knobs()` + `_resolve()` unit-tested on bare `python3`. `setup`/`step` need a real model +
GPU, so they're validated live (not in the bare-python3 suite).
