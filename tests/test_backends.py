"""backends tests — the pure dispatch surface (no torch / GPU / Modal).

The local/modal `run()` needs torch, a device, or Modal, so end-to-end runs are validated live.
What's testable here: the dispatch/validation, and that the modules IMPORT clean on bare python3
(proving the heavy imports — torch, modal — are lazy inside `run`, not at module load).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import backends
import backends.local
import backends.modal
import backends.ssh
from backends.local import _make_runner
from backends.ssh import _settings, _ssh_cmd, payload_config


def test_modules_import_clean_on_bare_python3():
    # If torch/modal weren't lazy, these imports would have already failed on a bare interpreter.
    assert callable(backends.run)
    assert callable(backends.local.run) and callable(backends.modal.run)
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


if __name__ == "__main__":
    for _n, _f in list(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f(); print(f"ok  {_n}")
    print("all passed")
