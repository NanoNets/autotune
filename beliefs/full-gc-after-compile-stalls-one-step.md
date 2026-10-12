---
claim: torch.compile leaves ~2-3M long-lived Python objects, and each generation-2 GC pass over them takes ~0.9-1.4 s.
  When the last of those passes lands inside a probe's clean window it shows as one ~1.3 s step: mean far above median
  with 0 recompiles. gc.collect() + gc.freeze() after the first (compiling) step moves them to the permanent generation;
  after that no slow GC ran in 199 steps.
type: measurement
context:
  arch_family: any model under torch.compile (seen with peft LoRA + Inductor dynamic shapes, both wrapper modes)
  hardware_class: any (host-side); seen on an A10 box, torch 2.7.0, Python 3.10
  dataset_shape: any
evidence:
  probes: 1 (a10_smollm2-360m_lora_script: stall in 3 of 6 probes on one box (exps 12, 15, 17: cv 92-95%, medians
    unaffected), 0 of 4 on another. gc.callbacks timing, 3 runs: ~20 gen-2 passes each, slowest 0.87-1.43 s (1.34 s
    collecting 1.30M objects), the same sequence every run. With gc.freeze after step 1: every slow pass before step 1
    ends, none in 199 steps; without: none in steps 1-198, one 0.83 s pass after step 199). Basis: GC pass timings
    and counts (width-invariant)
  confidence: medium (one stack; the mechanism is CPython's generational GC)
links: [[recompiles-show-as-mean-over-median]]
---

Tell it from a recompile with TORCH_LOGS=recompiles (0 here) and from length variation with the median (unchanged).
Fix in the training code: gc.freeze() once compiling is done (copy flag `--gc-freeze`). It changes no numerics. How
often a full GC recurs over thousands of steps without it is unmeasured.
