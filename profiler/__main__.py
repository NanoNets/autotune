"""Profile one config and print the diagnostic — the local tool.

  python -m profiler --device cpu    # torch fixture runner on CPU (smoke)

(Named __main__.py so it runs as `python -m profiler` — a top-level `profile.py` would shadow
Python's stdlib `profile` module and break torch's imports.)

capture (clean time + memory + profiled trace) -> diagnose (roofline + trace -> Observation).
Diagnostic / roofline / trace are torch-free; only capture + the runner need torch. `--gpu` is
the roofline denominator; on CPU MFU and memory are meaningless, it just exercises the pipeline.
"""
import argparse
import json

from profiler.capture import capture
from profiler.diagnostic import diagnose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cpu")       # cpu / cuda / mps
    ap.add_argument("--gpu", default="A100")         # roofline denominator
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    args = ap.parse_args()

    from frameworks.torch import TorchRunner
    runner = TorchRunner()
    config = {"device": args.device, "micro_batch": args.micro_batch, "seq": args.seq}

    cap = capture(runner, config)
    params = getattr(runner, "params", 0)            # 0 if setup failed before params were set
    tokens = runner.tokens_per_step() if params else 0
    obs = diagnose(cap, params=params, tokens=tokens, gpu=args.gpu)

    print(obs.summary)
    print(json.dumps(obs.__dict__, indent=2, default=str))


if __name__ == "__main__":
    main()
