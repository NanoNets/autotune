---
claim: Recompiles inside a compiled probe's clean window show as a mean step time far above the median, with the
  reference-batch GPU step unchanged and a long wall. A mean/median of ~1.1-1.2 under length grouping at accum 1 is just
  length variation. Tell them apart with a multi-pass extended probe: recompiles inflate pass 1 only.
type: calibration
context:
  arch_family: any, under torch.compile with padding="longest"
  hardware_class: any
  dataset_shape: wide length spread, dynamic widths
evidence:
  probes: 2 (a10_smollm2-1.7b_wikitext exp 13: pad_multiple 1, median 0.395 vs mean 2.454s, samples/s -82%, GPU step
    unchanged; exp 14: 3-pass extended probe of the pad_multiple 8 champion within 0.4% of 1 pass, so its 1.14
    mean/median is length variation), basis same-rows
  confidence: medium
links: [[full-gc-after-compile-stalls-one-step]]
---

pad_multiple 8 kept recompiles in warmup on both A10 runs. pad_multiple 1 did not, even after dynamo's automatic
dynamic should have kicked in. Keep widths bucketed under compile.

2026-10-08, Qwen2.5-1.5B on cosmopedia: pad_multiple 1 compiled clean at mb1 and at grouped mb4 (mean/median <= 1.035 in
exps 5-8, 10, 11, warmup 3). So recompiling at pad_multiple 1 is not universal. It showed on SmolLM2 (mb16, short rows).
Keep checking mean/median rather than assuming either way.

2026-10-09, SmolLM2-360M LoRA user script (a10_smollm2-360m_lora_script, script probes; recompile counts are
width-invariant): default compile on per-batch widths recompiled on width and then on HF SDPA's alignment guards
(`width % 8 == 0`), ~95s each, until the probe timed out; `dynamic=True` from the start + widths padded to a multiple of
8 gave 0 recompiles in 70 steps. Keep widths bucketed. But **mean >> median is not proof of recompiles**: the same run
had a one-step ~1.3s stall with 0 recompiles, a full Python GC (see [[full-gc-after-compile-stalls-one-step]]). Check
TORCH_LOGS=recompiles before calling it.
