# Verification: LoRA / PEFT in the profiler

> Private paper trail for the `peft-lora` branch (`design_improvements.md` §10: MFU under frozen
> weights). In the style of `docs/verification/byo_script.md`; remove before open-sourcing.
> Run 2026-10-07.

## Method

- **Box:** 1× NVIDIA A10 (23 GB), driver 580.105.08, torch 2.7.0 / CUDA 12.8 (system), Python
  3.10.12.
- **Probe venv** `~/autotune/.venv_at`: transformers 5.15.1, huggingface-hub 1.28.0, tokenizers
  0.22.2, safetensors 0.8.0, accelerate 1.14.0, peft 0.20.0, datasets 5.0.1, numpy 1.24.4,
  pillow 12.3.0, networkx 3.4.2. Same pins as `docs/verification/ceiling.md`.
- **Driver:** every probe went through `backends.run` with the ssh backend from a laptop, as the
  loop calls it: `profiler.launch` attached to the unchanged script (sandbox path).
- **Scripts:** `examples/byo_script/torch_lora.py` (new: PEFT LoRA r=16, alpha 32, dropout 0.05 on
  all 7 attention / MLP projections, optimizer over the adapters only) and `torch_flags.py` (full
  fine-tune). Both: bf16, SDPA, seq 512, micro_batch 8, grad_accum 4 (16,384 tokens per step),
  seeded random tokens. The LoRA settings match BYO experiment 7.
- **Probe settings:** `warmup` 3, `clean_steps` 10, step budget 17. The first probe measured the
  peak (91.96 TFLOP/s bf16, 445.6 GB/s); the other three reused it, so every MFU has the same
  denominator.

## Results

**4 / 4 pass.**

| Probe | `params_trainable` | `params_total` | Step (mean) | MFU, old formula | **MFU, new** |
|---|---|---|---|---|---|
| SmolLM2-360M, LoRA | 8,683,520 | 370,504,640 | 2.038 s | 0.46% | **13.1%** |
| Qwen2.5-0.5B, LoRA | 8,798,208 | 502,830,976 | 2.178 s | 0.43% | **16.6%** |
| SmolLM2-360M, full | 361,821,120 | 361,821,120 | 1.268 s | 30.5% | **30.5%** |
| Qwen2.5-0.5B, full | 494,032,768 | 494,032,768 | 1.471 s | 35.9% | **35.9%** |

"Old formula" is 6 × the optimizer's count × D, i.e. what the probe reported before this branch.
All four: `status` ok, `params_source` "root module: all parameters", `model_trained` the base
checkpoint (`model_matches_config` true), step-time cv 0.14–0.35%.

What each check shows:
1. **Counts are exact.** Both LoRA probes report exactly experiment 7's trainable / total counts
   (`sandbox-archive:bring_your_own_script.md`). The total is found through the `PeftModel`
   (the root module) and resolves to the base checkpoint's name.
2. **The counts agree across probes.** LoRA total − LoRA trainable = the full fine-tune total, to
   the parameter, on both models (370,504,640 − 8,683,520 = 361,821,120; 502,830,976 − 8,798,208 =
   494,032,768).
3. **The formula is applied as written.** `flops == (4·params_total + 2·params_trainable)·D` on
   every probe.
4. **Full fine-tune is unchanged.** `params_total == params_trainable`, and the new MFU equals the
   old 6ND MFU to 5 decimal places, so past MFUs in `experience/` stay comparable.
5. **LoRA MFU is fixed.** From ~0.45% (≈30× low) to 13–17%.

## Deviations, recorded rather than smoothed over

- **LoRA MFU is ~12–14% below §10's estimates (≈15.3% / ≈19.7%), and the cause is step time, not
  the count.** §10's figures came from experiment 7's HF Trainer arm. Its own record gives a
  SmolLM2 step of ≈1.82 s at an 88.4 TFLOP/s peak; the new formula on those inputs gives 15.3%,
  matching §10. Here `torch_lora.py` stepped in 2.038 s (SmolLM2) and 2.178 s (Qwen), about 12% and
  14% slower, against a slightly higher peak (91.96 vs 88.4). Same counts and formula, slower
  script: a workload difference, not a profiler one. Not investigated further.
- **A LoRA step is slower than a full fine-tune step** (SmolLM2 2.04 s vs 1.27 s; Qwen 2.18 s vs
  1.47 s), as experiment 7 also found (1.82 s vs 1.30 s). LoRA saves the frozen weights' gradients
  and optimizer work, but at this scale the extra small adapter GEMMs, dropout and dtype casts cost
  more. MFU correctly reads lower for LoRA: it counts the model FLOPs the work requires, not the
  overhead.

## Limits of this verification

- One card type (A10), two small models (360M, 0.5B), one LoRA config. Larger models and other
  ranks were not run.
- `torch_lora.py` hands the optimizer only the adapters. A script that passes `model.parameters()`
  (frozen weights included) used to count the frozen base as trainable; the count now keeps only
  parameters with `requires_grad`, covered by a CPU test
  (`test_frozen_weights_handed_to_the_optimizer_are_not_counted_as_trained`), not re-run on the GPU.
- Quantized bases (QLoRA, GPTQ, AWQ) were not run on a GPU; the null-MFU guard is covered only by a
  CPU test (`tests/test_attach.py::test_quantized_base_is_unknown_not_its_packed_count`).
- §10's step 3 (HFU, gradient checkpointing on vs off) was not run; it stays optional.

## Frame (analytical, CPU only)

Frame's LoRA estimates (`--lora-r` / `--lora-targets`) were checked against the A10 counts above:

| Model, r=16, all 7 projections | Frame `params_trainable` (from the config) | peft / the profiler (measured) |
|---|---|---|
| SmolLM2-360M | 8,683,520 | 8,683,520 |
| Qwen2.5-0.5B | 8,798,208 | 8,798,208 |

Frame's analytical base count is within 0.02% of the measured one (it leaves out norm and bias
weights). The model-state memory estimate (bf16 base stored only; fp32 adapters with grads and
AdamW state) was **not** checked on a GPU: that is `design_improvements.md` §14, item 1.

## CPU tests

`uv run pytest`: 264 passed, 1 failed (`test_cpu_signal_survives_a_window_too_short_to_sample`,
which also fails on `main`). New: `tests/test_roofline.py` (formula), `tests/test_attach.py`
(LoRA-shaped counts, full fine-tune, quantized guard), `tests/test_diagnostic.py` (MFU with both
counts, unchanged full fine-tune MFU, null MFU without a total), `tests/test_backends_script.py`
(both counts end to end through `backends.run`). Frame: `tests/test_model.py` (adapter counts exact against peft, per
projection, unknown names / MoE MLP targets → null), `tests/test_memory.py` (frozen base stored
only; unchanged when everything trains), `tests/test_profile.py` (LoRA FLOPs and memory, an 8B
flipping to `model_state_fits` on 24GB, uncountable targets keep full numbers with a note).

## Raw records

`runs/lora_verify/{smol,qwen}_{lora,full}.json` (gitignored): the full Observation of each probe.
