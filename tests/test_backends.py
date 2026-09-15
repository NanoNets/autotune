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
from backends.local import _make_runner


def test_modules_import_clean_on_bare_python3():
    # If torch/modal weren't lazy, these imports would have already failed on a bare interpreter.
    assert callable(backends.run)
    assert callable(backends.local.run) and callable(backends.modal.run)


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


if __name__ == "__main__":
    for _n, _f in list(globals().items()):
        if _n.startswith("test_") and callable(_f):
            _f(); print(f"ok  {_n}")
    print("all passed")
