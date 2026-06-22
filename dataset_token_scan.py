#!/usr/bin/env python3
"""
True multimodal sequence-length scan for the autotune dataset.

Unlike check_token_lengths.py (text tokens only + a flat image guess), this counts
the *real* per-sample length the model sees: image tokens + text tokens. For an OCR
vision model the image tokens usually dominate and vary per sample (they scale with
page resolution), so they decide which documents blow past max_length.

Efficiency: image tokens come from Qwen's smart_resize formula applied to each image's
(W,H) read lazily from the file header (no full decode/resize), and the formula is
pinned to the model's actual processor config. Text tokens come from a fast tokenizer
pass. No model forward, no per-sample AutoProcessor call.

Run on the VM/Modal after activating the swift env:
  python dataset_token_scan.py --out token_scan.json

Output: stdout summary + a structured JSON artifact the loop / Step-2 cut-analysis read.
"""

import argparse, io, json, math, sys
from collections import Counter

import numpy as np

# ── Defaults (mirror the loop's fixed config) ─────────────────────────────────
MODEL_ID   = "Qwen/Qwen3.5-0.8B"
DATASET_ID = "nikhitrivedi1/nn_ocr4"
SPLIT      = "train"
SUBSET     = "v0"
MAX_LENGTH = 15000          # the current cap we're considering cutting to 8000


# ── Image token math (Qwen smart_resize) ──────────────────────────────────────

def smart_resize(h, w, factor, min_pixels, max_pixels):
    """Qwen's resize: round to a multiple of `factor`, clamp area to [min,max] pixels."""
    h = max(h, factor)
    w = max(w, factor)
    h_bar = round(h / factor) * factor
    w_bar = round(w / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = math.floor(h / beta / factor) * factor
        w_bar = math.floor(w / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return max(h_bar, factor), max(w_bar, factor)


def image_tokens(w, h, factor, merge, min_pixels, max_pixels):
    """Merged vision tokens for one image of native size (w,h). +2 for vision start/end."""
    h_bar, w_bar = smart_resize(h, w, factor, min_pixels, max_pixels)
    grid = (h_bar // factor) * (w_bar // factor)   # already divided by merge**2
    return grid + 2


def read_processor(model_id):
    """Pin (factor, merge, min_pixels, max_pixels, tokenizer) to the model's real config."""
    from transformers import AutoProcessor, AutoTokenizer
    factor, merge = 28, 2
    min_pixels, max_pixels = 56 * 56, 28 * 28 * 1280     # documented Qwen-VL fallbacks
    tokenizer = None
    try:
        proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        ip = getattr(proc, "image_processor", None)
        tokenizer = getattr(proc, "tokenizer", None)
        if ip is not None:
            patch = getattr(ip, "patch_size", 14)
            merge = getattr(ip, "merge_size", merge)
            factor = patch * merge
            size = getattr(ip, "size", {}) or {}
            min_pixels = getattr(ip, "min_pixels", size.get("min_pixels", min_pixels))
            max_pixels = getattr(ip, "max_pixels", size.get("max_pixels", max_pixels))
            print(f"  processor: factor={factor} merge={merge} "
                  f"min_pixels={min_pixels:,} max_pixels={max_pixels:,}")
    except Exception as e:
        print(f"  WARNING: AutoProcessor unavailable ({e}); using fallback image config "
              f"and AutoTokenizer. Image-token estimates may be off — pass --min-pixels/"
              f"--max-pixels to match swift if you know them.")
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return factor, merge, int(min_pixels), int(max_pixels), tokenizer


# ── Per-sample extraction (schema-flexible) ───────────────────────────────────

def extract_text(sample):
    """Pull all text content out of a sample across common ms-swift / chat schemas."""
    if "messages" in sample:
        parts = []
        for msg in sample["messages"]:
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
        return " ".join(parts)
    if "conversations" in sample:
        return " ".join(str(t.get("value", t.get("content", "")))
                        for t in sample["conversations"] if isinstance(t, dict))
    if "query" in sample or "response" in sample:
        return " ".join(filter(None, [str(sample.get("query", "")),
                                       str(sample.get("response", ""))]))
    if "text" in sample:
        return str(sample["text"])
    return " ".join(str(v) for v in sample.values() if isinstance(v, str))


def image_sizes(sample, img_cols):
    """Yield (w,h) for every image in a sample, reading dims lazily (no full decode)."""
    sizes = []
    for col in img_cols:
        vals = sample.get(col)
        if vals is None:
            continue
        for v in (vals if isinstance(vals, list) else [vals]):
            wh = _dims(v)
            if wh:
                sizes.append(wh)
    return sizes


def _dims(v):
    """Get (w,h) from a decode=False dict, a path/url string, or a PIL image."""
    from PIL import Image
    try:
        if isinstance(v, dict):                       # cast_column(Image(decode=False))
            if v.get("bytes"):
                return Image.open(io.BytesIO(v["bytes"])).size   # header only
            if v.get("path"):
                return Image.open(v["path"]).size
        elif isinstance(v, str):
            return Image.open(v).size
        elif hasattr(v, "size"):                       # already a PIL image
            return v.size
    except Exception:
        return None
    return None


def find_image_columns(ds):
    """Detect Image-typed columns and cast them to decode=False for cheap dim reads."""
    from datasets import Image as HFImage
    from datasets import Sequence
    img_cols = []
    for name, feat in ds.features.items():
        is_img = isinstance(feat, HFImage) or (
            isinstance(feat, Sequence) and isinstance(getattr(feat, "feature", None), HFImage))
        if is_img:
            img_cols.append(name)
    for name in img_cols:
        try:
            inner = ds.features[name]
            new = HFImage(decode=False)
            ds = ds.cast_column(name, Sequence(new) if isinstance(inner, Sequence) else new)
        except Exception:
            pass
    # common ms-swift column not typed as Image
    if "images" not in img_cols and "images" in ds.column_names:
        img_cols.append("images")
    return ds, img_cols


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--dataset", default=DATASET_ID)
    ap.add_argument("--subset", default=SUBSET)
    ap.add_argument("--split", default=SPLIT)
    ap.add_argument("--max-samples", type=int, default=None, help="cap for a quick look")
    ap.add_argument("--cut", type=int, default=8000, help="candidate max_length to evaluate")
    ap.add_argument("--current", type=int, default=MAX_LENGTH, help="current max_length")
    ap.add_argument("--min-pixels", type=int, default=None, help="override processor min_pixels")
    ap.add_argument("--max-pixels", type=int, default=None, help="override processor max_pixels")
    ap.add_argument("--out", default="token_scan.json")
    args = ap.parse_args()

    from datasets import load_dataset

    print(f"Loading processor/tokenizer: {args.model} ...")
    factor, merge, min_px, max_px, tokenizer = read_processor(args.model)
    if args.min_pixels is not None: min_px = args.min_pixels
    if args.max_pixels is not None: max_px = args.max_pixels

    print(f"Loading dataset: {args.dataset} (subset={args.subset}, split={args.split}) ...")
    ds = load_dataset(args.dataset, args.subset, split=args.split)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    ds, img_cols = find_image_columns(ds)
    print(f"  {len(ds):,} samples | columns: {ds.column_names} | image columns: {img_cols}\n")

    print("── First sample (structure check) " + "─" * 44)
    for k, v in ds[0].items():
        print(f"  {k}: {str(v)[:100]}")
    print()

    print("Scanning ...")
    records, no_image = [], 0
    for i, sample in enumerate(ds):
        text = extract_text(sample).strip()
        t_tok = len(tokenizer(text, add_special_tokens=False)["input_ids"]) if text else 0
        sizes = image_sizes(sample, img_cols)
        if not sizes:
            no_image += 1
        i_tok = sum(image_tokens(w, h, factor, merge, min_px, max_px) for (w, h) in sizes)
        records.append({"idx": i, "image_tok": i_tok, "text_tok": t_tok,
                        "total": i_tok + t_tok,
                        "dominant": "image" if i_tok >= t_tok else "text"})
        if (i + 1) % 500 == 0:
            print(f"  {i + 1:,}/{len(ds):,}")

    if not records:
        print("ERROR: no samples scanned."); sys.exit(1)
    if no_image:
        print(f"  Note: {no_image:,} samples had no detectable image "
              f"(text-only, or schema not recognised — check the first-sample dump).")

    totals = np.array([r["total"] for r in records])
    img_arr = np.array([r["image_tok"] for r in records])
    txt_arr = np.array([r["text_tok"] for r in records])

    # ── Report ────────────────────────────────────────────────────────────────
    def pct(a, p): return int(np.percentile(a, p))
    print("\n── Total length (image + text tokens) " + "─" * 40)
    for label, fn in [("min", np.min), ("mean", np.mean), ("median", np.median),
                      ("p90", lambda a: np.percentile(a, 90)),
                      ("p95", lambda a: np.percentile(a, 95)),
                      ("p99", lambda a: np.percentile(a, 99)), ("max", np.max)]:
        print(f"  {label:<7} total={fn(totals):>8,.0f}   "
              f"image={fn(img_arr):>7,.0f}   text={fn(txt_arr):>7,.0f}")

    print("\n── Samples exceeding thresholds (TRUE length) " + "─" * 32)
    for thr in sorted({2048, 4096, args.cut, 12288, args.current}):
        n = int((totals > thr).sum()); p = 100 * n / len(totals)
        print(f"  > {thr:>6,}  {n:>6,} samples  ({p:5.1f}%)  {'█' * int(p / 2)}")

    # ── Step-2 cut analysis: what gets DROPPED (truncation_strategy=delete) ──────
    # The probe runs swift with --truncation_strategy delete (probe_runner.py), so an over-length
    # sample is removed WHOLE — not truncated. Account in whole-document terms, not tail overflow.
    dropped = [r for r in records if r["total"] > args.cut]
    tokens_removed = sum(r["total"] for r in dropped)          # full docs, not just the overflow
    dom = Counter(r["dominant"] for r in dropped)
    text_dropped = [r for r in dropped if r["dominant"] == "text"]
    print(f"\n── If max_length {args.current:,} → {args.cut:,}: what gets DROPPED "
          f"(--truncation_strategy delete) " + "─" * 4)
    print(f"  {len(dropped):,} of {len(totals):,} docs ({100*len(dropped)/len(totals):.1f}%) "
          f"exceed {args.cut:,} → removed ENTIRELY (whole doc, not truncated).")
    print(f"  {tokens_removed:,} total tokens removed from training.")
    print(f"  driver: {dict(dom)}  → text-dominated drops = lost labels/supervision (risky); "
          f"image-dominated drops = lost pages (often re-includable by lowering max_pixels).")
    print(f"  {len(text_dropped):,} of the dropped docs are text-dominated — the risky ones.")

    # ── Artifact ────────────────────────────────────────────────────────────────
    over_cut = sorted(dropped, key=lambda r: -r["total"])
    artifact = {
        "model": args.model, "dataset": args.dataset, "subset": args.subset,
        "n_samples": len(totals),
        "image_config": {"factor": factor, "merge": merge,
                         "min_pixels": min_px, "max_pixels": max_px},
        "percentiles": {p: pct(totals, p) for p in (50, 90, 95, 99, 100)},
        "thresholds": {str(t): int((totals > t).sum())
                       for t in (4096, args.cut, args.current)},
        "cut": {"current": args.current, "candidate": args.cut,
                "truncation_strategy": "delete",
                "n_dropped": len(dropped),
                "pct_dropped": round(100 * len(dropped) / len(totals), 2),
                "tokens_removed_total": int(tokens_removed),
                "dominant_breakdown": dict(dom),
                "n_text_dominated_dropped": len(text_dropped)},
        "over_cut": over_cut,   # per-doc detail for the dropped docs
    }
    with open(args.out, "w") as f:
        json.dump(artifact, f, indent=2)
    print(f"\nWrote {args.out}  ({len(over_cut):,} over-cut docs recorded for review).")


if __name__ == "__main__":
    main()
