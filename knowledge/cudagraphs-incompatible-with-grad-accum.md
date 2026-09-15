---
claim: Any cudagraph-enabling torch.compile mode breaks a runner that invokes the model several times
  per optimizer step, and marking step boundaries does not fix it.
type: safety
context:
  arch_family: any
  hardware_class: any CUDA
  dataset_shape: any
evidence:
  probes: 2 (experiment 7 max-autotune, experiment 10 reduce-overhead)
  confidence: high — two independent modes, identical failure, prescribed fix ineffective
links: [[padding-regime-flips-attention-verdict]]
---

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
