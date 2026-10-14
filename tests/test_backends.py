"""backends tests — the pure dispatch surface (no torch / GPU).

The local `run()` needs torch and a device, so end-to-end runs are validated live.
What's testable here: the dispatch/validation, and that the modules IMPORT clean on bare python3
(proving the heavy import — torch — is lazy inside `run`, not at module load).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backends
import backends.local
import backends.ssh
from backends.local import _make_runner, _resolve_device
from backends.ssh import _settings, _ssh_cmd, payload_config


def test_modules_import_clean_on_bare_python3():
    # If torch weren't lazy, these imports would have already failed on a bare interpreter.
    assert callable(backends.run)
    assert callable(backends.local.run)
    assert callable(backends.ssh.run)


def test_unknown_backend_raises():
    try:
        backends.run({"backend": "tpu-v9"})
    except ValueError as e:
        assert "unknown backend" in str(e)
        return
    assert False, "expected ValueError for an unknown backend"


def test_unknown_framework_raises_before_importing():
    try:
        _make_runner("jax")                            # not in the registry -> fails before any import
    except ValueError as e:
        assert "unknown framework" in str(e)
        return
    assert False, "expected ValueError for an unknown framework"


def test_ssh_needs_a_host():
    try:
        _settings({})
    except ValueError as e:
        assert "needs a host" in str(e)
        return
    assert False, "expected ValueError when no ssh host is configured"


def test_ssh_target_joins_user_and_host():
    assert _settings({"ssh": {"host": "1.2.3.4", "user": "ubuntu"}})["target"] == "ubuntu@1.2.3.4"
    assert _settings({"ssh": {"host": "box"}})["target"] == "box"        # user optional (~/.ssh/config)


def test_ssh_credential_precedence(monkeypatch=None):
    # No credential in env -> plain ssh, letting the system config/agent resolve it.
    for var in ("AUTOTUNE_SSH_KEY", "AUTOTUNE_SSH_PASSWORD"):
        os.environ.pop(var, None)
    base = _ssh_cmd(_settings({"ssh": {"host": "box"}}))
    assert base[0] == "ssh" and "-i" not in base

    os.environ["AUTOTUNE_SSH_KEY"] = "/tmp/k"
    keyed = _ssh_cmd(_settings({"ssh": {"host": "box"}}))
    assert keyed[0] == "ssh" and "-i" in keyed
    del os.environ["AUTOTUNE_SSH_KEY"]

    os.environ["AUTOTUNE_SSH_PASSWORD"] = "pw"
    pw = _ssh_cmd(_settings({"ssh": {"host": "box"}}))
    assert pw[0] == "sshpass"                          # password auth wraps ssh
    del os.environ["AUTOTUNE_SSH_PASSWORD"]


def test_ssh_never_ships_connection_details_to_the_box():
    # On password auth the ssh block holds the credential, so it must not reach a remote argv.
    payload = payload_config({"backend": "ssh", "micro_batch": 4,
                              "ssh": {"host": "box", "user": "ubuntu"}})
    assert "ssh" not in payload
    assert payload["micro_batch"] == 4                  # ...but the knobs do cross


def test_ssh_probes_on_cuda_regardless_of_local_device():
    # The launcher is a laptop (mps/cpu); the box is the GPU. The remote device is not negotiable.
    assert payload_config({"ssh": {"host": "box"}, "device": "mps"})["device"] == "cuda"


class _FakeTorch:
    """Stands in for torch so the device tests stay bare-python3 (no real torch import)."""
    def __init__(self, available):
        self.cuda = type("cuda", (), {"is_available": staticmethod(lambda: available)})


def _with_torch(available, config):
    saved = sys.modules.get("torch")
    sys.modules["torch"] = _FakeTorch(available)
    try:
        return _resolve_device(config)
    finally:
        sys.modules.pop("torch", None)
        if saved is not None:
            sys.modules["torch"] = saved


def test_device_defaults_to_cuda_on_a_gpu_box():
    # the real case: an agent fills in run.toml without thinking about `device`
    assert _with_torch(True, {}) == "cuda"


def test_explicit_device_always_wins():
    # including "cpu" — a smoke test you asked for is not a mistake
    assert _with_torch(True, {"device": "cpu"}) == "cpu"
    assert _with_torch(False, {"device": "cuda"}) == "cuda"
    assert _with_torch(True, {"device": "mps"}) == "mps"


def test_indexed_cuda_device_is_refused():
    # capture compares `device == "cuda"` exactly, so "cuda:0" would skip synchronisation and
    # memory stats and still return a well-formed Observation — the silent bad probe again
    try:
        _with_torch(True, {"device": "cuda:0"})
    except ValueError as e:
        assert "CUDA_VISIBLE_DEVICES" in str(e)
        return
    assert False, "expected ValueError for an indexed cuda device"


def test_cpu_fallback_announces_itself():
    # no CUDA -> cpu, but never silently: a silent fall-through logs CPU numbers to results.tsv
    # as if they were a real probe
    import contextlib
    import io
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        assert _with_torch(False, {}) == "cpu"
    assert "CPU" in err.getvalue() and "meaningless" in err.getvalue()


def test_sample_file_must_be_repo_relative_and_exist_before_anything_is_sent():
    """The box resolves sample_file under remote_dir, so an absolute path would miss it."""
    from backends.ssh import _push_sample
    s = _settings({"ssh": {"host": "box"}})
    for bad, err in (("/tmp/sample.jsonl", ValueError),
                     ("runs/no-such-run/sample.jsonl", FileNotFoundError)):
        try:
            _push_sample(s, bad)                         # raises before any ssh/rsync runs
        except err:
            continue
        assert False, f"expected {err.__name__} for {bad!r}"


if __name__ == "__main__":
    for _n, _f in list(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f(); print(f"ok  {_n}")
    print("all passed")
