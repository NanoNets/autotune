"""CLI: python -m frame --model <hub-id|path> --dataset <hub-id|path> --gpu A10G

Prints the one-line summary + the full Profile JSON. Needs transformers + datasets.
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
    ap.add_argument("--dtype", default="bfloat16", help="run compute dtype; the peak is measured at this")
    ap.add_argument("--device", default="cuda", help="device to measure the peak on")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--split", default="train")
    ap.add_argument("--sample", type=int, default=1000)
    ap.add_argument("--name", default=None, help="dataset config, e.g. wikitext-2-raw-v1")
    a = ap.parse_args()

    p = frame(a.model, a.dataset, a.gpu, dtype=a.dtype, device=a.device,
              text_field=a.text_field, split=a.split, sample=a.sample, name=a.name)
    print(p.summary)
    print(json.dumps(dataclasses.asdict(p), indent=2, default=str))


if __name__ == "__main__":
    main()
