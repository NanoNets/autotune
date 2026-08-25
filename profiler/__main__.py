"""Profile one config and print the diagnostic — the local tool.

  python -m profiler --device cpu --gpu A10G                         # torch fixture runner (smoke)
  python -m profiler --framework hf --device cuda --gpu A10G \       # real HF model, real data
      --model Qwen/Qwen2.5-0.5B --dataset Salesforce/wikitext --name wikitext-2-raw-v1 --dtype bfloat16

(Named __main__.py so it runs as `python -m profiler` — a top-level `profile.py` would shadow
Python's stdlib `profile` module and break torch's imports.)

capture (clean time + memory + profiled trace) -> diagnose (roofline + trace -> Observation).
Diagnostic / roofline / trace are torch-free; only capture + the runner need torch. `--gpu` is
a REQUIRED hardware label for the record; the MFU denominator is MEASURED on the device, so any
hardware works and no table can misprice it. On CPU MFU and memory are meaningless; it just
exercises the pipeline.
"""
import argparse
import json

from profiler.capture import capture
from profiler.diagnostic import diagnose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--framework", choices=["torch", "hf"], default="torch",
                    help="torch = random-data MLP fixture; hf = real transformers causal-LM")
    ap.add_argument("--device", default="cpu")       # cpu / cuda / mps
    ap.add_argument("--gpu", required=True,           # a LABEL for the record, e.g. A10G. The
                    help="hardware label for the record, e.g. A10G")  # denominator is measured
    ap.add_argument("--dtype", default=None, help="run dtype (hf); the peak is measured at this")
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--model", help="HF hub id or local path (required for --framework hf)")
    ap.add_argument("--dataset", help="HF hub id or local path (required for --framework hf)")
    ap.add_argument("--name", default=None, help="dataset config/subset (hf)")
    args = ap.parse_args()

    config = {"device": args.device, "micro_batch": args.micro_batch, "seq": args.seq}
    if args.dtype:
        config["dtype"] = args.dtype

    if args.framework == "hf":
        if not (args.model and args.dataset):
            ap.error("--framework hf requires --model and --dataset")
        from frameworks.hf import HFRunner
        runner = HFRunner()
        config.update({"model": args.model, "dataset": args.dataset, "name": args.name,
                       "seq_len": args.seq})
    else:
        from frameworks.torch import TorchRunner
        runner = TorchRunner()

    cap = capture(runner, config)
    params = getattr(runner, "params", 0)            # 0 if setup failed before params were set
    tokens = runner.tokens_per_step() if params else 0
    obs = diagnose(cap, params=params, tokens=tokens, gpu=args.gpu)

    print(obs.summary)
    print(json.dumps(obs.__dict__, indent=2, default=str))


if __name__ == "__main__":
    main()
