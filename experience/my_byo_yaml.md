## my_byo_yaml — 2026-10-01
Context: HuggingFaceTB/SmolLM2-360M · user script `examples/byo_script/trainer_yaml.py` + `train.yaml` (HF Trainer; trains on seeded random 512-token rows, NOT run.toml's wikitext-2) · 1x A10 23GB (CC 8.6) · bf16 · effective_batch 32 · max_len 512
Baseline: 12815 tok/s (25.0 samples/s, MFU 32%)  →  Best: 17465 tok/s (34.1 samples/s, MFU 43%)  (+36%)
Winning config (train.yaml keys): `per_device_train_batch_size: 16`, `gradient_accumulation_steps: 2`, `torch_compile: true`; all else as the user wrote it (sdpa, adamw_torch, no gradient checkpointing). Copy: `runs/my_byo_yaml/probes/exp7.yaml`.

**Script-mode run** (`population.driver: "script"`, `clean_window: fixed-position`): timings are
observations, not `same-rows` evidence. Every kept change was re-probed interleaved with the
incumbent (exps 4/5, 8/9) and reproduced within 0.7%. Loop stopped by the user after 10 experiments.

Findings:
- **torch.compile: +31% on top of micro 16** (13331 → 17465 tok/s, reproduced) and peak memory
  **17.36 → 12.75 GB** at the same micro-batch. Width-invariant: `gap_count` 8830 → 3616 (-59%) and
  distinct kernels 96 → 77, so the fusion is real launch reduction, not only GPU work.
- **micro 8 → 16 (accum 4 → 2): +3.6%** (two interleaved pairs), peak 10.1 → 17.4 GB uncompiled.
- **micro 32 / accum 1 OOMs even under compile** (peak 20.4 GB on 22.06; it wanted +3 GiB). Without
  recompute, 16 is the largest legal micro-batch on this card at seq 512.
- **Fused AdamW: -1.3%** uncompiled. Optimizer GPU time was 0.028s of a 1.26s step (2%); at 16k
  tokens/step there is no fixed optimizer cost to recover. Consistent with
  [[optimizer-fixed-cost-dominates-small-real-batches]], whose context is small real batches.
- Dataloader wait 0.1%: workers / pin memory are dead levers for this script.

Surprises / contradictions:
- **Misread the trace, worth remembering:** `phase_times_s['Optimizer.step#AdamW.step']` was 0.199s
  (wall) vs 0.028s on the GPU. The wall number is the CPU blocked on queued backward kernels, not
  launch overhead. Read `gpu_phase_times_s` before attributing a phase; `phase_times_s` overlaps.
- **Compile changed the attention kernel.** Uncompiled, sdpa ran flash (no attention kernel in the
  top 8). Compiled, it ran `fmha_cutlassB/F ... PyTorchMemEffAttention`, and the backward kernel
  became the **largest single kernel (0.123s/step, ~13%)**. Which kernels ran is width-invariant.
- Box trap: the image's system networkx 2.4 (`np.int`) breaks `torch.compile` under numpy 1.24
  (exp 6, env error, not a result). `pip install networkx==3.4.2` into `.venv_at`.

Open hypotheses for next run:
- **Recover flash attention under compile** (`attn_implementation: flash_attention_2` if the wheel
  installs, or check why inductor picks mem-efficient: GQA expansion / an explicit attention_mask).
  Up to ~0.08s/step (~9%) if the backward drops to flash's cost.
- micro 32 / accum 1 with `gradient_checkpointing: true` under compile: does the bigger GEMM beat
  recompute's ~30% extra forward? Per [[recompute-pays-only-instrumentally]], likely not, but 32
  is a legal factorization, so the arithmetic is worth one probe.
- Compile + fused AdamW (untested combination; expected ~0 at this tokens/step).
- No OOM ceiling exists for scripts (design_improvements.md §7); the micro-32 OOM was found by
  probing, which is exactly what a ceiling search should have told us up front.
