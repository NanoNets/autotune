"""The Observation — the profiler's output. First slice of loop/contracts.md, as code.

Raw and rich: the objective (MFU vs roofline), where the step's time went (phases, GPU
busy, data-wait, idle gap), provenance (what ran), and top kernels. No derived labels —
the reader interprets. Grows toward the full contract as later stages need it.
"""
from dataclasses import dataclass


@dataclass
class Observation:
    status: str                 # ok / oom / error
    gpu: str
    step_time_s: float          # clean wall-clock per step (profiler off)
    flops: float                # estimated training FLOPs/step
    roofline_floor_s: float     # compute-bound lower bound on step time
    mfu: float                  # achieved / peak, in [0, 1]
    gpu_busy_s: float           # total GPU kernel time (from the trace)
    gpu_active_frac: float      # gpu_busy / wall
    phase_times_s: dict         # forward/backward/optimizer/dataloader (CPU-side)
    data_wait_s: float          # time blocked on the dataloader
    data_wait_frac: float
    gap_s: float                # GPU idle time (wall - busy)
    gap_frac: float
    provenance: dict            # flash_attention / fused_kernels / dtypes actually seen
    top_kernels: list           # [(name, us), ...]
    summary: str = ""           # plain one-line statement of the fractions
