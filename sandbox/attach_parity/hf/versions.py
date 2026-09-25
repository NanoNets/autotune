"""Exact-version gate for experiment 4. Every HF arm calls `check()` before it trains."""
import sys

PINNED = {
    "python": "3.10.12",
    "torch": "2.7.0",
    "torch.cuda": "12.8",
    "transformers": "5.15.1",
    "accelerate": "1.14.0",
    "huggingface_hub": "1.28.0",
    "tokenizers": "0.22.2",
    "safetensors": "0.8.0",
    "numpy": "1.24.4",
    "scipy": "1.8.0",                      # system package, imported by Trainer; not pip-installed
}


def installed():
    import accelerate
    import numpy
    import scipy
    import huggingface_hub
    import safetensors
    import tokenizers
    import torch
    import transformers
    return {
        "python": ".".join(map(str, sys.version_info[:3])),
        "torch": torch.__version__.split("+")[0],
        "torch.cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "accelerate": accelerate.__version__,
        "huggingface_hub": huggingface_hub.__version__,
        "tokenizers": tokenizers.__version__,
        "safetensors": safetensors.__version__,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
    }


def check():
    got = installed()
    bad = {k: (PINNED[k], got[k]) for k in PINNED if got[k] != PINNED[k]}
    if bad:
        raise SystemExit(f"version mismatch (pinned, installed): {bad}")
    return got
