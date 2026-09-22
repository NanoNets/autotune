---
claim: Cudagraph-enabling compile modes are unusable under dynamic padding even when grad
  accumulation is absent — every distinct batch width forces a re-capture, and the capture cost
  swamps the step.
type: dead lever
context:
  arch_family: any
  hardware_class: any CUDA
  dataset_shape: variable-length rows under padding="longest" (i.e. many distinct batch shapes)
evidence:
  probes: 1 (smollm2_1.7b_a100_wikitext experiment 11, reduce-overhead at grad_accum=1)
  confidence: medium-high — one probe, but the magnitude is unmissable (39x) and the mechanism is
    the documented behaviour of cudagraph capture under changing shapes
links: [[cudagraphs-incompatible-with-grad-accum]], [[static-padding-is-usually-the-bottleneck]]
---

The second blocker behind [[cudagraphs-incompatible-with-grad-accum]], and the reason removing the
first one does not help.

`test_a10G` established that cudagraphs crash when the runner invokes the model several times per
optimizer step. That belief left an implication hanging: reach `grad_accum=1` and cudagraphs become
available. This run reached `grad_accum=1` (via `micro_batch=16`, bought with recompute) and tested it.

**Result: no crash, and a 39x slowdown.** `torch_compile="reduce-overhead"` ran to completion at
4.01 samples/s against 155.29 for the identical config on default compile — GPU-active **3%**, idle
**97%**. The GPU was doing essentially nothing while the host re-captured graphs.

**Mechanism.** Under `padding="longest"` with `pad_multiple=16`, batch width changes constantly, and
a CUDA graph is captured per shape. The workload presents new shapes faster than captures amortize,
so the run pays capture cost almost every step.

**Why this is worth a belief rather than a footnote.** The two conditions are in direct opposition on
exactly the workload where cudagraphs look most attractive:

- cudagraphs pay off when a run is **launch-bound** — many small kernels, GPU idle waiting on the host;
- a run becomes launch-bound here precisely **because** padding was removed at the loader, which is
  what makes batch widths dynamic in the first place ([[static-padding-is-usually-the-bottleneck]]).

So the lever is structurally unavailable in the regime that motivates it. The way out is not a flag:
it is making shapes static again *without* reintroducing padding — i.e. packing / `padding_free`, or
a fixed bucket set small enough to amortize captures. Until one of those exists in the adapter, treat
any cudagraph mode as dead under dynamic padding and do not spend the probe.
