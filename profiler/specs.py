"""GPU specs for the roofline: peak dense bf16 throughput and memory bandwidth.

Numbers are vendor datasheet peaks (dense, non-sparse) — the standard denominator for MFU.
Extend as needed. Keys are canonical short names; normalize nvidia-smi's name to these at lookup.
"""

GPU_SPECS = {
    #          peak dense bf16 TFLOP/s   mem bandwidth GB/s   memory GB
    "A100": {"bf16_tflops": 312.0,  "mem_bw_gb_s": 2039.0, "mem_gb": 80},
    "H100": {"bf16_tflops": 989.4,  "mem_bw_gb_s": 3350.0, "mem_gb": 80},
    "A10G": {"bf16_tflops": 125.0,  "mem_bw_gb_s": 600.0,  "mem_gb": 24},
    "L4":   {"bf16_tflops": 121.0,  "mem_bw_gb_s": 300.0,  "mem_gb": 24},
}
