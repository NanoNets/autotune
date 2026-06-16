#!/usr/bin/env python3
"""
Quick token length distribution check for the autotune dataset.

Run on the VM after activating the swift-env:
  source ~/envs/swift-env/bin/activate
  cd ~/Desktop/auto_tune
  python check_token_lengths.py

Output: token length stats + recommendation for max_length.
Note: counts text tokens only — add ~256-1024 per image for Qwen3.5 vision tokens.
"""

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer

MODEL_ID   = "Qwen/Qwen3.5-0.8B"
DATASET_ID = "nikhitrivedi1/nn_ocr4"
SPLIT      = "train"
SUBSET     = "v0"

# ── Load tokenizer ────────────────────────────────────────────────────────────

print(f"Loading tokenizer: {MODEL_ID} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

# ── Load dataset ──────────────────────────────────────────────────────────────

print(f"Loading dataset:   {DATASET_ID} (subset={SUBSET}) ...")
ds = load_dataset(DATASET_ID, SUBSET, split=SPLIT)
print(f"  {len(ds):,} samples | columns: {ds.column_names}\n")

# Show the first sample so you can verify the structure looks right
print("── First sample (structure check) " + "─" * 44)
sample0 = ds[0]
for k, v in sample0.items():
    preview = str(v)[:120] + ("..." if len(str(v)) > 120 else "")
    print(f"  {k}: {preview}")
print()

# ── Extract text per sample ───────────────────────────────────────────────────

def extract_text(sample: dict) -> str:
    """
    Pull all text content out of a sample regardless of schema.
    Handles ms-swift conversation formats and plain text columns.
    """
    # ms-swift messages format: [{"role": ..., "content": ...}, ...]
    if "messages" in sample:
        parts = []
        for msg in sample["messages"]:
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):          # multimodal content list
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
        return " ".join(parts)

    # sharegpt / alpaca conversation format
    if "conversations" in sample:
        return " ".join(
            str(t.get("value", t.get("content", "")))
            for t in sample["conversations"]
            if isinstance(t, dict)
        )

    # alpaca-style flat fields
    if "input" in sample or "output" in sample:
        return " ".join(filter(None, [
            str(sample.get("instruction", "")),
            str(sample.get("input", "")),
            str(sample.get("output", "")),
        ]))

    # plain text column
    if "text" in sample:
        return str(sample["text"])

    # fallback: concatenate all string values
    return " ".join(str(v) for v in sample.values() if isinstance(v, str))


# ── Tokenize ──────────────────────────────────────────────────────────────────

print("Tokenizing ...")
lengths = []
empty = 0
for sample in ds:
    text = extract_text(sample).strip()
    if not text:
        empty += 1
        continue
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    lengths.append(len(ids))

if not lengths:
    print("ERROR: could not extract text from any sample.")
    print("Check the 'First sample' output above and update extract_text().")
    raise SystemExit(1)

if empty:
    print(f"  Warning: {empty} samples had no extractable text and were skipped.\n")

lengths = np.array(lengths)

# ── Report ────────────────────────────────────────────────────────────────────

print("── Token length distribution (text only) " + "─" * 37)
stats = [
    ("samples",  f"{len(lengths):,}"),
    ("min",      f"{lengths.min():,}"),
    ("mean",     f"{lengths.mean():,.0f}"),
    ("median",   f"{np.median(lengths):,.0f}"),
    ("p75",      f"{np.percentile(lengths, 75):,.0f}"),
    ("p90",      f"{np.percentile(lengths, 90):,.0f}"),
    ("p95",      f"{np.percentile(lengths, 95):,.0f}"),
    ("p99",      f"{np.percentile(lengths, 99):,.0f}"),
    ("max",      f"{lengths.max():,}"),
]
for label, value in stats:
    print(f"  {label:<8} {value:>10}")

print()
print("── Samples exceeding common max_length thresholds " + "─" * 28)
for threshold in [2048, 4096, 8192, 12288, 15000]:
    n = (lengths > threshold).sum()
    pct = 100 * n / len(lengths)
    bar = "█" * int(pct / 2)
    print(f"  > {threshold:>6,}  {n:>6,} samples  ({pct:5.1f}%)  {bar}")

print()
p99 = int(np.percentile(lengths, 99))
# round up to nearest 512, add headroom for image tokens
image_token_budget = 1024   # conservative estimate for Qwen3.5 per image
suggested = ((p99 + image_token_budget + 511) // 512) * 512
print("── Recommendation " + "─" * 60)
print(f"  p99 text tokens  : {p99:,}")
print(f"  + image budget   : ~{image_token_budget:,}  (Qwen3.5 vision tokens per image, varies by resolution)")
print(f"  suggested max_length: {suggested:,}  (p99 + image budget, rounded to nearest 512)")
print()
print("  If most samples are shorter than your current max_length=15000,")
print("  lowering it will significantly raise your OOM ceiling for full SFT.")
