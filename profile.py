"""Profile one config and print the diagnostic — the local tool.

  python profile.py            # torch fixture runner on CPU (smoke)

Ties the pieces together: capture (clean time + profiled trace) → diagnose (roofline + trace
→ Observation). Roofline/trace/diagnostic are torch-free; only capture + the runner need torch.
`--gpu` is the roofline denominator (the hardware you're optimizing for); on a CPU smoke run
MFU is meaningless, it just exercises the pipeline.
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

    clean, trace_path = capture(runner, config)
    obs = diagnose(clean, trace_path,
                   params=runner.params, tokens=runner.tokens_per_step(), gpu=args.gpu)

    print(obs.summary)
    print(json.dumps(obs.__dict__, indent=2, default=str))


if __name__ == "__main__":
    main()
