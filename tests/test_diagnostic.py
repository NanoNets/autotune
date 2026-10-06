"""Diagnostic tests — assemble an Observation from a fixture capture (no torch, no GPU).

`diagnose` is duck-typed on the capture (any object with .status/.step_times_s/.trace/.memory/
.error), so we hand it a SimpleNamespace and never import the torch-backed capture engine.
"""
import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiler.diagnostic import diagnose
from profiler.roofline import training_flops, mfu

TRACE = {
    "deviceProperties": [{"name": "NVIDIA A10", "totalGlobalMem": 23684841472,
                          "computeMajor": 8, "computeMinor": 6, "numSms": 72}],
    "traceEvents": [
        {"cat": "user_annotation", "name": "forward", "dur": 100.0},
        {"cat": "user_annotation", "name": "backward", "dur": 200.0},
        {"cat": "user_annotation", "name": "optimizer", "dur": 50.0},
        {"cat": "user_annotation", "name": "dataloader", "dur": 300.0},
        {"cat": "kernel", "name": "sm80_gemm_bf16", "dur": 80.0, "ts": 0.0,           # [0, 80]
         "args": {"est. achieved occupancy %": 40, "occupancy": {"limitingFactors": "REGS"}}},
        {"cat": "kernel", "name": "flash_attention_fwd", "dur": 40.0, "ts": 100.0,    # [100,140] 20us gap
         "args": {"est. achieved occupancy %": 60, "occupancy": {"limitingFactors": "REGS"}}},
    ],
}

PARAMS, TOKENS, GPU = 1_000_000, 1_000, "A100"     # GPU is a label now, not a spec key
PEAK = 3.12e14                                     # measured peak handed over by the capture
STEP_TIMES = [1.0e-3, 1.1e-3, 0.9e-3]   # median 1.0 ms, with spread -> nonzero cv
MEMORY = {"peak_gb": 12.0, "reserved_gb": 14.0, "total_gb": 80.0, "headroom_gb": 66.0}
CPU = {"mean": 62.0, "max": 88.0, "nproc": 16}


REF_WALL, PROFILED = 0.5e-3, 0.8e-3    # the reference batch: profiler off, then on


def _cap(status="ok", step_times=None, trace=TRACE, memory=MEMORY, cpu=CPU, error=None, peak=PEAK,
         ref=REF_WALL, profiled=PROFILED, step_tokens=None, device="cuda"):
    return SimpleNamespace(
        status=status, trace=trace, memory=memory, cpu=cpu, error=error, device=device,
        peak_flops=peak, peak_dtype="bfloat16", reference_step_s=ref, profiled_step_s=profiled,
        step_tokens=step_tokens or [],
        step_times_s=STEP_TIMES if step_times is None else step_times)


def _obs():
    return diagnose(_cap(), params=PARAMS, tokens=TOKENS, gpu=GPU)


def test_device_type_flows_through():
    assert _obs().device_type == "cuda"                 # the normal case, unremarkable
    o = diagnose(_cap(status="error", error="boom"), params=PARAMS, tokens=TOKENS, gpu=GPU)
    assert o.device_type == "cuda"                      # kept on the failure paths too


def test_a_cpu_probe_is_marked_as_one():
    # a cpu probe returns a well-formed Observation; without this it is indistinguishable from a
    # real GPU probe in results.tsv, and nothing downstream can tell the MFU is meaningless
    o = diagnose(_cap(device="cpu"), params=PARAMS, tokens=TOKENS, gpu=GPU)
    assert o.device_type == "cpu"
    assert o.summary.startswith("[CPU — not a GPU probe]")
    assert not _obs().summary.startswith("[")           # the cuda summary is unchanged

    # ...and on the paths that never reach a timing, where it is just as misleading
    bad = diagnose(_cap(device="cpu", status="error", error="boom"),
                   params=PARAMS, tokens=TOKENS, gpu=GPU)
    assert bad.summary.startswith("[CPU — not a GPU probe]") and "boom" in bad.summary


def test_trace_fields_flow_through():
    o = _obs()
    assert o.gpu_busy_s == 120e-6                       # 80 + 40 us
    assert o.data_wait_s == 300e-6
    assert o.phase_times_s["backward"] == 200e-6


def test_rates_come_from_clean_window_totals_not_the_median():
    """Right-skewed steps: the median understates the time, so a median-based rate reads high."""
    times, toks = [1.0e-3, 1.0e-3, 4.0e-3], [100, 100, 700]      # one wide step; median 1ms
    o = diagnose(_cap(step_times=times, step_tokens=toks), params=PARAMS, tokens=TOKENS, gpu=GPU)
    assert o.step_time_s == 1.0e-3                               # median kept, as a statistic
    assert abs(o.step_time_mean_s - 2.0e-3) < 1e-15
    assert o.tokens_per_step == 300.0                            # mean over the clean steps
    assert abs(o.tokens_per_s - 900 / 6.0e-3) < 1e-6             # total tokens / total time
    assert abs(o.mfu - mfu(training_flops(PARAMS, 300.0), 2.0e-3, PEAK)) < 1e-12
    assert o.clean_steps_measured == 3


def test_caller_tokens_are_the_fallback_when_steps_are_uncounted():
    o = _obs()                                                   # no per-step counts
    assert o.tokens_per_step == TOKENS
    assert abs(o.mfu - mfu(training_flops(PARAMS, TOKENS), 1.0e-3, PEAK)) < 1e-12   # mean = 1ms


def test_step_time_cv_is_std_over_mean():
    assert _obs().step_time_cv > 0                      # spread -> nonzero cv
    o = diagnose(_cap(step_times=[1.0e-3, 1.0e-3, 4.0e-3]), params=PARAMS, tokens=TOKENS, gpu=GPU)
    assert o.step_time_cv == 0.7071                     # pstdev 1.414ms / mean 2ms, not / median 1ms
    o1 = diagnose(_cap(step_times=[1.0e-3]), params=PARAMS, tokens=TOKENS, gpu=GPU)
    assert o1.step_time_cv == 0.0                       # single sample -> 0


def test_memory_flows_through():
    o = _obs()
    assert o.peak_mem_gb == 12.0
    assert o.mem_total_gb == 80.0
    assert o.headroom_gb == 66.0


def test_cpu_util_flows_through():
    o = _obs()
    assert o.cpu_util_mean == 62.0
    assert o.cpu_util_max == 88.0
    assert o.nproc == 16


def test_trace_fractions_stay_inside_the_reference_batch():
    """Busy time is divided by the SAME batch's profiler-off wall — never the clean median."""
    o = _obs()
    assert abs(o.gpu_active_frac - 120e-6 / REF_WALL) < 1e-12
    assert abs(o.gap_s - (REF_WALL - 120e-6)) < 1e-12
    assert abs(o.data_wait_frac - 300e-6 / PROFILED) < 1e-12   # host annotation: its own wall
    assert o.reference_step_s == REF_WALL and o.profiled_step_s == PROFILED


def test_no_reference_wall_means_no_trace_fractions():
    o = diagnose(_cap(ref=None, profiled=None), params=PARAMS, tokens=TOKENS, gpu=GPU)
    assert o.gpu_active_frac is None and o.gap_s is None and o.data_wait_frac is None
    assert o.gpu_busy_s == 120e-6                       # the trace itself still reports


def test_data_wait_is_unknown_without_a_dataloader_annotation():
    """A script that fetches without a DataLoader has no span to read: null, never a measured 0."""
    trace = {**TRACE, "traceEvents": [e for e in TRACE["traceEvents"] if e.get("name") != "dataloader"]}
    o = diagnose(_cap(trace=trace), params=PARAMS, tokens=TOKENS, gpu=GPU)
    assert o.data_wait_s is None and o.data_wait_frac is None
    assert _obs().data_wait_s == 300e-6                 # present: read as before


def test_gpu_seconds_flow_through_on_every_status():
    for status in ("ok", "oom", "error"):
        cap = _cap(status=status, step_times=[] if status != "ok" else None)
        cap.gpu_seconds = 12.5
        assert diagnose(cap, params=PARAMS, tokens=TOKENS, gpu=GPU).gpu_seconds == 12.5
    assert _obs().gpu_seconds is None                   # a capture without it: unknown


def test_a_cuda_trace_with_no_kernels_is_unknown_not_idle():
    """A failed trace (seen at 38.97/39.5GB reserved) must not read as a 0%-busy GPU."""
    empty = {**TRACE, "traceEvents": [e for e in TRACE["traceEvents"] if e["cat"] != "kernel"]}
    o = diagnose(_cap(trace=empty), params=PARAMS, tokens=TOKENS, gpu=GPU)
    assert o.gpu_busy_s is None and o.gpu_active_frac is None and o.gap_s is None
    assert o.status == "ok" and "no kernels" in o.summary     # rates still stand; GPU signals don't
    assert o.mfu is not None


def test_an_impossible_reading_is_invalid_not_scored():
    """smollm2 exp 28: one wide batch's busy over a narrow median read 560%. Never score it."""
    o = diagnose(_cap(ref=40e-6), params=PARAMS, tokens=TOKENS, gpu=GPU)   # 120us busy / 40us
    assert o.status == "invalid"
    assert "gpu_active_frac" in o.error and o.summary.startswith("INVALID")
    assert abs(o.gpu_active_frac - 3.0) < 1e-12         # the value is kept, so it can be read


def test_jitter_inside_tolerance_is_not_invalid():
    o = diagnose(_cap(ref=118e-6), params=PARAMS, tokens=TOKENS, gpu=GPU)  # 120/118 = 1.017
    assert o.status == "ok"


def test_the_tolerance_is_two_percent():
    """Calibrated on an A100: healthy mean + 3 sd reached 1.017; 1.03 is past the 2% line."""
    o = diagnose(_cap(ref=116.5e-6), params=PARAMS, tokens=TOKENS, gpu=GPU)  # 120/116.5 = 1.030
    assert o.status == "invalid"


def test_coverage_and_idle_shape():
    o = _obs()
    assert o.kernel_count == 2
    assert o.top8_coverage == 1.0                       # both kernels are in the top-8
    assert o.gap_count == 1                             # one idle gap on the timeline
    assert abs(o.max_gap_s - 20e-6) < 1e-12             # the 20 us gap


def test_top_kernels_are_seconds_like_every_other_duration():
    o = _obs()
    assert o.top_kernels[0] == ["sm80_gemm_bf16", 80e-6]      # 80 us, expressed in seconds
    assert abs(sum(d for _, d in o.top_kernels) - o.gpu_busy_s) < 1e-12   # same unit as the totals


def test_device_and_occupancy_flow_through():
    o = _obs()
    assert o.device["name"] == "NVIDIA A10"
    assert o.device["num_sms"] == 72
    assert o.achieved_occupancy_pct == 46.7             # (40*80 + 60*40) / 120, runtime-weighted
    assert o.limiting_factor == "REGS"


def test_oom_short_circuits_but_keeps_memory_and_trace_path():
    o = diagnose(_cap(status="oom", step_times=[], trace="/tmp/probe.json", error="CUDA out of memory"),
                 params=PARAMS, tokens=TOKENS, gpu=GPU)
    assert o.status == "oom"
    assert o.mfu is None                                # no throughput computed on OOM
    assert o.peak_mem_gb == 12.0                        # but memory is still surfaced
    assert o.trace_path == "/tmp/probe.json"            # and the raw-trace pointer is kept
    assert "oom" in o.summary


def test_idle_shape_is_null_without_a_kernel_timeline():
    trace = {**TRACE, "traceEvents": [{"cat": "kernel", "name": "k", "dur": 80.0}]}   # no ts
    o = diagnose(_cap(trace=trace), params=PARAMS, tokens=TOKENS, gpu=GPU)
    assert o.gap_count is None and o.max_gap_s is None   # unknown, not a gapless GPU


def test_oom_keeps_the_clean_steps_that_ran():
    o = diagnose(_cap(status="oom", error="CUDA out of memory"),
                 params=PARAMS, tokens=TOKENS, gpu=GPU)
    assert o.status == "oom"
    assert o.step_time_s == 1.0e-3                      # steps before the OOM are still evidence
    assert o.mfu is None                                # but nothing is scored off them


def test_negative_gap_is_null_not_zero():
    """busy is read from the profiled run, wall from the unprofiled one — they can disagree."""
    o = diagnose(_cap(ref=118e-6), params=PARAMS, tokens=TOKENS, gpu=GPU)  # 120us busy, 118us wall
    assert o.gap_s is None and o.gap_frac is None       # disagreement, not "zero idle"
    assert "idle n/a" in o.summary


def test_provenance_present():
    assert _obs().provenance["flash_attention"] is True


def test_any_hardware_label_is_accepted():
    """No table to validate against — the denominator is measured, so any chip is supported."""
    o = diagnose(_cap(), params=PARAMS, tokens=TOKENS, gpu="GB200-that-shipped-yesterday")
    assert o.gpu == "GB200-that-shipped-yesterday" and o.mfu is not None


def test_no_measured_peak_means_no_mfu():
    o = diagnose(_cap(peak=None), params=PARAMS, tokens=TOKENS, gpu="A100")
    assert o.mfu is None and o.roofline_floor_s is None   # no denominator, no number
    assert "MFU n/a" in o.summary


def test_readme_documents_every_observation_field():
    """The README's field table is the agent-facing contract — keep it from drifting."""
    from dataclasses import fields
    from profiler.diagnostic import Observation

    doc = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "profiler", "README.md")).read()
    names = [f.name for f in fields(Observation)]
    assert f"{len(names)} fields" in doc                # the count claim matches reality
    assert [n for n in names if f"`{n}`" not in doc] == []


if __name__ == "__main__":
    for _n, _f in list(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f(); print(f"ok  {_n}")
    print("all passed")
