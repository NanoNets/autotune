"""CLI: python -m frame --model <hub-id|path> --dataset <hub-id|path> --gpu A10G --max-len 2048

Prints the one-line summary, any notes, and the full Profile JSON. Needs transformers + datasets.

`--max-len` is required: it is the human's declared application sequence length, and it is what
the OOM ceiling is measured at. Without it there is no worst case to measure against.

`--ceiling` measures that ceiling using the toy `frameworks.torch` runner — a SMOKE PATH that
proves the search runs end to end, not a number to tune against. A ceiling that means anything
needs the real training step: call `frame.profile.frame(..., runner=<real runner>)`.
"""
import argparse
import dataclasses
import json

from frame.profile import frame


def main():
    ap = argparse.ArgumentParser(prog="frame")
    ap.add_argument("--model", required=True, help="HF hub id or local path")
    ap.add_argument("--dataset", required=True, help="HF hub id or local path")
    ap.add_argument("--gpu", required=True, help="a label for the record (any hardware; not validated)")
    ap.add_argument("--max-len", type=int, required=True,
                    help="the application's sequence length; the OOM ceiling is measured at it")
    ap.add_argument("--dtype", default="bfloat16", help="run compute dtype; the peak is measured at this")
    ap.add_argument("--device", default="cuda", help="device to measure the peak on")
    ap.add_argument("--framework", default="hf", help="which runner's knob surface to report")
    ap.add_argument("--effective-batch", type=int, default=16,
                    help="human target the loop factorizes; also caps the ceiling search")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--split", default="train")
    ap.add_argument("--sample", type=int, default=1000)
    ap.add_argument("--name", default=None, help="dataset config, e.g. wikitext-2-raw-v1")
    ap.add_argument("--ceiling", action="store_true",
                    help="measure the OOM ceiling with the TOY runner (smoke path, not a real number)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the synthetic batches")
    a = ap.parse_args()

    runner = None
    if a.ceiling:
        from frameworks.torch import TorchRunner
        runner = TorchRunner()
        print(f"frame: measuring the memory ceiling at max_len={a.max_len} "
              f"(synthetic, seed {a.seed}, TOY runner)")

    def on_trial(t):
        peak = f"peak {t['peak_gb']}GB" if t["peak_gb"] is not None else "peak n/a"
        print(f"  mb={t['micro_batch']:<5} {t['status']:<5} {peak}")

    p = frame(a.model, a.dataset, a.gpu, a.max_len, dtype=a.dtype, device=a.device,
              framework=a.framework, effective_batch=a.effective_batch,
              text_field=a.text_field, split=a.split, sample=a.sample, name=a.name,
              runner=runner, seed=a.seed, on_trial=on_trial)
    print(p.summary)
    for n in p.notes:
        print(f"NOTE: {n}")
    print(json.dumps(dataclasses.asdict(p), indent=2, default=str))


if __name__ == "__main__":
    main()
