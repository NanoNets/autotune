"""Bring-your-own-script, end to end through the backend the loop calls. CPU only.

`backends.run({"script": ...})` -> `backends/local.py: run_script` -> `python -m profiler.launch`
in a subprocess -> the user's unchanged script -> `cap.json` -> `diagnose()` -> Observation.
Each test writes a small user-style training script (its own argparse, DataLoader and loop, no
autotune imports) and checks the Observation the loop would receive.
"""
import os
import sys
import textwrap

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # repo root

import backends  # noqa: E402
from backends.ssh import _check_script  # noqa: E402

BATCH, SEQ = 4, 8
SCRIPT = textwrap.dedent(f'''
    """A user's training script: nothing autotune-specific in it."""
    import argparse, os, sys, time, torch
    from torch.utils.data import DataLoader, TensorDataset
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-steps", type=int, default=20)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--fail", default="")                  # "<step>:oom" | "<step>:kill"
    ap.add_argument("--mode", default="")                  # how the script calls its model
    a = ap.parse_args()
    torch.manual_seed(0)
    model = torch.nn.Sequential(torch.nn.Embedding(64, 16), torch.nn.Linear(16, 64))
    if a.mode == "named":                                   # what from_pretrained leaves behind
        model.config = type("Config", (), {{"_name_or_path": "org/tiny"}})()

    class TakesADict(torch.nn.Module):                      # model(batch), batch a dict
        def __init__(self, m):
            super().__init__()
            self.m = m
        def forward(self, batch):
            return self.m(batch["input_ids"])

    net = TakesADict(model) if a.mode == "dict" else model
    transform = torch.nn.Identity()                         # no parameters; called before the model
    reference = torch.nn.Sequential(torch.nn.Embedding(64, 16), torch.nn.Linear(16, 64))
    reference.requires_grad_(False)                         # frozen, e.g. a distillation teacher
    opt = torch.optim.AdamW(model.parameters())
    data = DataLoader(TensorDataset(torch.randint(0, 64, (256, {SEQ}))), batch_size={BATCH})
    it = iter(data)
    for step in range(1, a.max_steps + 1):
        if a.mode == "hang":
            time.sleep(600)
        for _ in range(a.grad_accum):
            x, = next(it)
            if a.mode == "transform_first":
                x = transform(x)
            if a.mode == "reference_first":                 # a different-size input, called first
                with torch.no_grad():
                    reference(torch.randint(0, 64, (3 * {BATCH}, {SEQ})))
            out = net({{"input_ids": x}}) if a.mode == "dict" else net(x)
            out.logsumexp(-1).mean().backward()
        opt.step()
        opt.zero_grad()
        if a.fail == f"{{step}}:oom":
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        if a.fail == f"{{step}}:kill":
            os._exit(9)                                   # no cleanup: like the OOM killer
''')


def probe(tmp_path, *script_args, clean=2, **config):
    path = tmp_path / "train.py"
    path.write_text(SCRIPT)
    return backends.run({"backend": "local", "script": str(path), "script_args": list(script_args),
                         "device": "cpu", "warmup": 2, "clean_steps": clean, "gpu": "test",
                         "peak_flops": 1e18, "peak_mem_bw_gb_s": 1e6, **config})


def test_a_user_script_becomes_an_observation(tmp_path):
    obs = probe(tmp_path, "--max-steps", "8")
    assert obs["status"] == "ok", obs["error"]
    assert obs["clean_steps_measured"] == 2
    assert obs["tokens_per_step"] == BATCH * SEQ                   # counted on the clean steps (B1)
    assert obs["tokens_per_s"] and obs["mfu"] is not None
    assert obs["trace_path"] and os.path.exists(obs["trace_path"])
    assert {"forward", "backward", "dataloader"} <= set(obs["phase_times_s"])   # derived from the trace
    pop = obs["population"]
    assert pop["driver"] == "script" and pop["trace_basis"] == "traced-step"
    assert pop["clean_window"] == "fixed-position"                 # not whole passes: said, not hidden
    assert pop["script"].endswith("train.py") and pop["script_args"] == ["--max-steps", "8"]
    assert pop["micro_batches"]["forwards"] == pop["micro_batches"]["backwards"] == 1


def test_grad_accum_is_seen_through_the_script(tmp_path):
    obs = probe(tmp_path, "--max-steps", "8", "--grad-accum", "3")
    assert obs["status"] == "ok", obs["error"]
    assert obs["tokens_per_step"] == 3 * BATCH * SEQ
    assert obs["population"]["micro_batches"]["forwards"] == 3


def test_an_oom_in_the_script_is_an_oom_observation(tmp_path):
    obs = probe(tmp_path, "--max-steps", "8", "--fail", "3:oom")   # inside the clean window
    assert obs["status"] == "oom"
    assert "CUDA out of memory" in obs["error"]
    assert obs["clean_steps_measured"] == 1                        # the step that ran is kept


def test_a_killed_script_is_an_error_not_a_crash(tmp_path):
    obs = probe(tmp_path, "--max-steps", "8", "--fail", "2:kill")
    assert obs["status"] == "error"
    assert obs["error"].startswith("launcher wrote no result")


def test_too_short_a_step_budget_is_an_error_that_says_so(tmp_path):
    obs = probe(tmp_path, "--max-steps", "2")                       # needs warmup 2 + clean 2 + 1
    assert obs["status"] == "error"
    assert obs["error"] == "script ended after 2 optimizer steps; needs 5"


def test_a_missing_script_fails_before_launching(tmp_path):
    with pytest.raises(FileNotFoundError):
        backends.run({"backend": "local", "script": str(tmp_path / "nope.py"), "device": "cpu"})


def test_ssh_needs_a_repo_relative_script(tmp_path):
    with pytest.raises(ValueError, match="relative to the repo root"):
        _check_script(str(tmp_path / "train.py"))
    with pytest.raises(FileNotFoundError):
        _check_script("runs/does-not-exist/train.py")
    _check_script("tests/test_backends_script.py")                  # any repo-relative file passes


def test_script_files_must_exist_before_launching(tmp_path):
    path = tmp_path / "train.py"
    path.write_text(SCRIPT)
    with pytest.raises(FileNotFoundError):
        backends.run({"backend": "local", "script": str(path), "device": "cpu",
                      "script_files": [str(tmp_path / "probes" / "exp1.yaml")]})


def test_script_files_are_recorded_on_the_observation(tmp_path):
    cfg = tmp_path / "exp1.yaml"
    cfg.write_text("micro_batch: 4\n")
    path = tmp_path / "train.py"
    path.write_text(SCRIPT)
    obs = backends.run({"backend": "local", "script": str(path), "device": "cpu",
                        "script_args": ["--max-steps", "8"], "script_files": [str(cfg)],
                        "warmup": 2, "clean_steps": 2, "peak_flops": 1e18, "peak_mem_bw_gb_s": 1e6})
    assert obs["status"] == "ok", obs["error"]
    assert obs["population"]["script_files"] == [str(cfg)]


def test_ssh_checks_every_script_file_is_repo_relative(tmp_path):
    from backends.ssh import run as ssh_run
    with pytest.raises(ValueError, match="relative to the repo root"):
        ssh_run({"backend": "ssh", "ssh": {"host": "unused"}, "sync": False,
                 "script": "tests/test_backends_script.py",
                 "script_files": [str(tmp_path / "exp1.yaml")]})


def test_a_hung_script_is_an_error_observation_not_a_crashed_loop(tmp_path):
    obs = probe(tmp_path, "--max-steps", "8", "--mode", "hang", timeout=5)
    assert obs["status"] == "error"
    assert obs["error"] == "script exceeded timeout of 5s (killed)"


def test_invalid_probe_settings_are_an_error_observation(tmp_path):
    obs = probe(tmp_path, "--max-steps", "8", warmup=1)
    assert obs["status"] == "error"
    assert obs["error"].startswith("invalid probe settings: warmup=1: needs >= 2")


def test_a_model_called_with_a_dict_has_unknown_tokens_not_a_guess(tmp_path):
    obs = probe(tmp_path, "--max-steps", "8", "--mode", "dict")
    assert obs["status"] == "ok", obs["error"]                      # timing still stands
    assert obs["tokens_per_step"] is None and obs["tokens_per_s"] is None
    assert obs["population"]["micro_batches"]["token_source"].startswith("unknown:")


@pytest.mark.parametrize("mode", ["transform_first", "reference_first"])
def test_tokens_are_counted_on_the_model_being_trained(tmp_path, mode):
    """The first module called isn't necessarily the model: a parameter-free transform, or a
    frozen reference model fed a different-size input. Tokens come from the one that owns the
    optimizer's parameters."""
    obs = probe(tmp_path, "--max-steps", "8", "--mode", mode)
    assert obs["status"] == "ok", obs["error"]
    assert obs["tokens_per_step"] == BATCH * SEQ


@pytest.mark.parametrize("declared,matches", [("org/tiny", True), ("org/other", False), (None, None)])
def test_the_trained_model_is_checked_against_the_declared_one(tmp_path, declared, matches):
    obs = probe(tmp_path, "--max-steps", "8", "--mode", "named",
                **({"model": declared} if declared else {}))
    assert obs["status"] == "ok", obs["error"]
    assert obs["population"]["model_trained"] == "org/tiny"
    assert obs["population"]["model_matches_config"] is matches


def test_ssh_pushes_run_files_even_without_a_code_sync(monkeypatch):
    """`sync = false` skips the code rsync, but a new per-probe config under runs/ still has to
    reach the box, or the probe runs on a stale copy."""
    import backends.ssh as ssh
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    d = os.path.join(repo, "runs", "_test_ship")
    os.makedirs(d, exist_ok=True)
    cfg = "runs/_test_ship/exp1.yaml"
    try:
        with open(os.path.join(repo, cfg), "w") as f:
            f.write("micro_batch: 4\n")
        pushed = []
        monkeypatch.setattr(ssh, "_push_sample", lambda s, f: pushed.append(f))
        monkeypatch.setattr(ssh, "_sync", lambda s: pytest.fail("sync = false must not rsync"))

        class Stop(Exception):
            pass

        def no_remote(*a, **k):
            raise Stop
        monkeypatch.setattr(ssh.subprocess, "run", no_remote)
        with pytest.raises(Stop):
            ssh.run({"backend": "ssh", "ssh": {"host": "unused", "sync": False},
                     "script": "tests/test_backends_script.py", "script_files": [cfg]})
        assert pushed == [cfg]                                     # tests/ rides the code sync
    finally:
        import shutil
        shutil.rmtree(d, ignore_errors=True)
