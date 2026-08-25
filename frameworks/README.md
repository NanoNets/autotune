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
    --dtype bfloat16 --micro-batch 4 --seq 1024
```

Needs torch + a GPU, `transformers`, `datasets`. Loads real weights, so it's heavier than the fixture.

## Knobs — the tunable surface

`knobs()` reports the levers the loop may turn. **Speed only** — nothing that changes the learned
model (no LR, weight decay, optimizer choice, or effective batch). Availability (e.g. flash-attn
installed) is gated by frame's `versions()`.

| Knob | Effect |
|---|---|
| `micro_batch` | per-step batch; more = better GPU fill, more memory |
| `grad_accum` | micro-batches per optimizer step (hold effective batch while changing `micro_batch`) |
| `seq_len` | sequence length (truncate/pad to this) |
| `dtype` | `bfloat16` / `float16` / `float32` |
| `gradient_checkpointing` | recompute activations in bwd — less memory, more compute |
| `attn_implementation` | `eager` / `sdpa` / `flash_attention_2` |
| `torch_compile` | `torch.compile` the model |

## Design notes

- **Raw PyTorch loop, not HF `Trainer`.** The profiler needs an explicit `step()` with the phases
  wrapped in `record_function`; `Trainer` hides the loop.
- **Real weights, real data.** Unlike frame (config-only), the adapter downloads the model and streams
  the dataset — a genuine fwd/bwd/opt is what gets timed.
- **`grad_accum` is inside one `step()`** — grad_accum micro fwd/bwd passes (loss scaled by
  `1/grad_accum`), then one optimizer step. `tokens_per_step = micro_batch · seq_len · grad_accum`.
- **Padding is masked** in the labels (`-100`) and attention, so a padded step is still a valid
  training step.
- **Pure vs live split**, like frame/profiler: `knobs()` + `_resolve()` (defaults + token math) import
  and test on bare `python3`; `setup`/`step` lazily import torch/transformers/datasets.

## Limitations (deferred)

- **Pre-tokenized in-memory loader** → the `dataloader` phase is ~0 (no real data-wait). On-the-fly
  tokenization + `num_workers`/packing as knobs come later.
- Single-GPU; `AdamW` only; the LR is a fixed placeholder (it doesn't affect step timing).

## Status

Built; `knobs()` + `_resolve()` unit-tested on bare `python3`. `setup`/`step` need a real model +
GPU, so they're validated live (not in the bare-python3 suite).
