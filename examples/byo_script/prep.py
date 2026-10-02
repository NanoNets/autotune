"""A separate preprocessing step: tokenize a JSON-lines file once, save the ids (`.pt`). The
pre-tokenized case: `trainer_pretok.py` reads only the output. For a ceiling trial the agent runs
this same step on Frame's `ceiling_rows.jsonl`.

  python prep.py --in rows.jsonl --out rows.pt [--model HuggingFaceTB/SmolLM2-360M]
"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _data import tokenizer_for  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
    a = ap.parse_args()
    tok, _ = tokenizer_for(a.model)
    with open(a.src) as f:
        texts = [json.loads(line)["text"] for line in f if line.strip()]
    torch.save([tok(t, add_special_tokens=False)["input_ids"] for t in texts], a.out)
    print(f"wrote {len(texts)} rows to {a.out}")


if __name__ == "__main__":
    main()
