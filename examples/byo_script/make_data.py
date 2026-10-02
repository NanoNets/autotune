"""Write `data/train.jsonl` (wikitext-2 train, non-empty rows) for the examples that read a fixed
dataset file. Run once on the box: `python make_data.py`. The file isn't tracked."""
import json
import os

from datasets import load_dataset

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "train.jsonl")
os.makedirs(os.path.dirname(out), exist_ok=True)
rows = [t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"]
        if t.strip()][:4096]
with open(out, "w") as f:
    for t in rows:
        f.write(json.dumps({"text": t}) + "\n")
print(f"wrote {len(rows)} rows to {out}")
