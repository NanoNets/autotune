"""`python -m profiler.launch` failure-path tests. CPU only; each case runs a tiny script under it.

  .venv/bin/python -m pytest tests/test_launch.py -q

The statuses must match what `capture()` produces, because `frame/ceiling.py` reads them:
`oom` means "doesn't fit", and `error` is a crash it refuses to use as a memory answer.
"""
import json
import os
import subprocess
import sys
import textwrap

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WARMUP, CLEAN = 1, 2                                  # profiled step = optimizer step 4

SCRIPT = textwrap.dedent('''
    import sys, torch
    fail_at, kind, steps = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
    model = torch.nn.Linear(8, 8)
    opt = torch.optim.AdamW(model.parameters())
    for step in range(1, steps + 1):
        loss = model(torch.randn(4, 8)).square().mean()
        loss.backward()
        opt.step()
        opt.zero_grad()
        if step == fail_at:
            if kind == "oom":
                raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
            if kind == "error":
                raise ValueError("bad batch\\nsecond line")
            if kind == "exit0":
                sys.exit(0)
            if kind == "exit3":
                sys.exit(3)
''')


def run(tmp_path, fail_at, kind, steps=8):
    script = tmp_path / "train.py"
    script.write_text(SCRIPT)
    out = tmp_path / "cap.json"
    p = subprocess.run([sys.executable, "-m", "profiler.launch", "--out", str(out),
                        "--device", "cpu", "--warmup", str(WARMUP), "--clean", str(CLEAN),
                        "--peak-flops", "1", "--peak-bw", "1", "--",
                        str(script), str(fail_at), kind, str(steps)],
                       capture_output=True, text=True, cwd=REPO)
    return p, json.loads(out.read_text())


def test_normal_run_is_ok(tmp_path):
    p, rec = run(tmp_path, fail_at=0, kind="none")
    assert p.returncode == 0, p.stderr
    assert rec["capture"]["status"] == "ok"
    assert len(rec["capture"]["step_times_s"]) == CLEAN
    assert "script_error" not in rec


def test_oom_before_profiled_step_is_oom_and_keeps_clean_steps(tmp_path):
    p, rec = run(tmp_path, fail_at=2, kind="oom")                # during the clean window
    cap = rec["capture"]
    assert p.returncode != 0                                     # re-raised: exits non-zero
    assert "OutOfMemoryError" in p.stderr                        # full traceback still on stderr
    assert cap["status"] == "oom"
    assert cap["error"] == "OutOfMemoryError: CUDA out of memory. Tried to allocate 2.00 GiB"
    assert len(cap["step_times_s"]) == 1                         # the one clean step that ran


def test_oom_after_profiled_step_overrides_to_oom(tmp_path):
    p, rec = run(tmp_path, fail_at=6, kind="oom")
    cap = rec["capture"]
    assert p.returncode != 0
    assert cap["status"] == "oom"
    assert cap["error"].endswith("(after the profiled step)")
    assert len(cap["step_times_s"]) == CLEAN and cap["trace"]    # the measurement is kept


def test_other_exception_is_error_with_type_and_first_line(tmp_path):
    p, rec = run(tmp_path, fail_at=2, kind="error")
    assert p.returncode != 0
    assert rec["capture"]["status"] == "error"
    assert rec["capture"]["error"] == "ValueError: bad batch"


def test_exception_after_profiled_step_keeps_ok_and_notes_it(tmp_path):
    p, rec = run(tmp_path, fail_at=6, kind="error")
    assert p.returncode != 0
    assert rec["capture"]["status"] == "ok"
    assert rec["script_error"] == "ValueError: bad batch"


@pytest.mark.parametrize("kind,expect", [("exit0", "too few"), ("exit3", "SystemExit: 3")])
def test_early_exit(tmp_path, kind, expect):
    p, rec = run(tmp_path, fail_at=2, kind=kind)
    cap = rec["capture"]
    assert cap["status"] == "error"
    if expect == "too few":                                      # sys.exit(0) is a normal end...
        assert cap["error"] == "script ended after 2 optimizer steps; needs 4"   # ...but too early
    else:
        assert cap["error"] == expect


def test_too_few_steps_is_error(tmp_path):
    p, rec = run(tmp_path, fail_at=0, kind="none", steps=3)
    assert p.returncode == 0
    assert rec["capture"]["status"] == "error"
    assert rec["capture"]["error"] == "script ended after 3 optimizer steps; needs 4"
