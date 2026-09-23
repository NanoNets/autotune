---
claim: torch.compile cannot compile a hybrid linear-attention model in this runner — it never
  reaches a timed step, and the cause is the architecture, not the number of input shapes.
type: dead-lever
context:
  arch_family: hybrid linear-attention / SSM-transformer with a gated delta rule (Qwen3.5); expect
    to re-test rather than assume for other Mamba-family hybrids
  hardware_class: any CUDA
  dataset_shape: any — tested at both ~128 distinct batch widths and at 4
evidence:
  probes: 2 (qwen3.5_2b_a100_wikitext experiments 2 and 8), both 601s timeouts, 20 minutes total
  confidence: high for this architecture + runner; the discriminator was explicit and clean
links: [[compile-cuts-memory-not-just-time]],
  [[cudagraphs-incompatible-with-dynamic-shapes]],
  [[linear-attention-kernel-library-is-the-lever]]
---

Two 601s timeouts, neither reaching a single timed step.

The second was designed as a **discriminator, not a retry**. The obvious suspect was per-shape
Inductor recompilation: exp 2 ran at `pad_multiple=8`, which admits ~128 distinct batch widths. So
exp 8 ran the same config at `pad_multiple=256` — **four** possible widths — with exp 7 as the
no-compile control at that same bucketing (22.929 samples/s, so the config itself was healthy).

It still timed out. **Shape count is exonerated.** What remains is the architecture: the gated
delta rule's chunked recurrence carries data-dependent control flow, and once `fla` is installed
its Triton kernels are opaque to Inductor. transformers documents the tension itself — the kernel
decorator returns the reference torch path outright under `is_torchdynamo_exporting()`, because
these packages are export-incompatible.

**This NARROWS [[compile-cuts-memory-not-just-time]] rather than contradicting it.** That belief
has five probes across two runs, two models and two cards — all **dense** transformers, where
compile was worth +42% and reliably cut peak memory by fusing the CE. Nothing here disputes that.
The context line on that belief should be read as dense-only until a hybrid compiles successfully.

**Practical consequence:** on a hybrid linear-attention model, do not spend early probes on
compile, and do not treat compile as the answer to a dispatch-bound step even though it is the
canonical one. The memory win compile usually brings via CE fusion is available another way —
`chunked_ce` (see [[chunked-ce-makes-memory-batch-independent]]) — and the dispatch win has to come
from `grad_accum`.
