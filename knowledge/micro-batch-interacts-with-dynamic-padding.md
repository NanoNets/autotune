---
claim: Under pad-to-longest, a larger micro_batch is cheaper per token but wastier per sample, because
  it pads to the longest of more rows. Length grouping is what makes a larger micro_batch pay.
type: invariant
context:
  arch_family: any
  hardware_class: any
  dataset_shape: variable-length rows under dynamic padding
evidence:
  probes: 3 (experiment 14 GC+mb8 without grouping; 15 with; 17 isolation control)
  confidence: high — mechanism measured via tokens/step, not inferred
links: [[static-padding-is-usually-the-bottleneck]]
---

Experiment 14 (`gradient_checkpointing` + `micro_batch` 4 → 8) **lost 13.9%** on samples/s while
being **+15% more efficient per token** and lifting GPU-active 66% → 76%. Both halves were real:

- the launch half worked — `grad_accum` halved, fewer model invocations
- the padding half backfired — padding to the longest of *8* mixed rows instead of 4 pushed
  tokens/step **2560 → 4572**, i.e. 49% more work for the same 16 samples

Adding `length_grouped` fixed only the second half and the config jumped **+71.7%** (tokens/step
4572 → 1756). An isolation control (experiment 17) attributed it honestly: grouping alone +45%,
GC+mb=8 a further +18.7%. Neither was redundant.

**Reading for the loop:** a losing probe whose *mechanism* partly succeeded is a diagnosis, not a dead
end. Check the work-normalized metric (tokens/s) against the objective (samples/s) before discarding —
when they disagree, the config is doing the right thing to the wrong amount of work.
