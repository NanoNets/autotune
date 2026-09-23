---
claim: Any cudagraph-enabling torch.compile mode breaks a runner that invokes the model several times
  per optimizer step, and marking step boundaries does not fix it.
type: safety
context:
  arch_family: any
  hardware_class: any CUDA
  dataset_shape: any
evidence:
  probes: 2 (test_a10G experiment 7 max-autotune, experiment 10 reduce-overhead)
  confidence: high for the CRASH, but the claim's scope was narrowed 2026-09-21 — at grad_accum=1
    the precondition is absent and cudagraphs do not crash. They fail a different way instead.
links: [[padding-regime-flips-attention-verdict]],
  [[cudagraphs-incompatible-with-dynamic-shapes]]
---

> **Scope narrowed 2026-09-21 (smollm2_1.7b_a100_wikitext).** This belief predicted a hard crash and
> implicitly that grad accumulation was the whole problem. Tested at `grad_accum=1`, where the
> precondition is gone: `reduce-overhead` **did not crash** — it ran and collapsed to 4.01 samples/s
> (GPU-active 3%, 39x slower than the same config without it). So the crash claim is correct and
> stays, but "remove the accumulation and cudagraphs become available" does **not** follow. A second,
> independent blocker sits behind it: [[cudagraphs-incompatible-with-dynamic-shapes]]. Do not spend a
> probe expecting grad_accum=1 to unlock cudagraphs under dynamic padding — this run spent that probe.

Both attempts failed identically in `Qwen2RMSNorm.forward`:

```
RuntimeError: accessing tensor output of CUDAGraphs that has been overwritten
by a subsequent run
```

`HFRunner.step` runs `grad_accum` forward/backward passes per optimizer step. CUDA graphs reuse
output buffers across replays, so a later micro-batch overwrites tensors an earlier one still needs.

**The prescribed fix does not work.** Torch's own error message recommends
`torch.compiler.cudagraph_mark_step_begin()` before each invocation; adding it (experiment 10)
produced the identical error. The cause is deeper than step boundaries — the backward's *saved
activations* live in the cudagraph memory pool and the next forward overwrites them.

**Consequence:** in a launch-bound regime, the obvious remedy (CUDA graphs) is unavailable to this
runner. Fixing it would need the accumulation loop restructured, not a flag. Plain `torch.compile`
(default mode) works fine and remains valuable — see [[compile-cuts-memory-not-just-time]].
