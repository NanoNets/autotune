"""The image half of a VLM's worst case, for the memory ceiling.

`max_len` bounds the text; for a VLM the vision tower's memory is set by the images too, and once the
vision tower trains a text-only ceiling reads 2x too high (Qwen2-VL-2B on an A10: text rows fit
micro-batch 2, rows with one 672x672 image fit only 1; `docs/verification/final_boss_report.md`,
VLM theme). So the human declares the largest image the job sees, `--max-image-size WxH`, the way
they declare `max_len`, and the ceiling rows carry that many images at that size.

- **Size** is W x H in pixels, read from the training code's image cap (never chosen here). Frame
  writes one image that size and the script's own processor resizes it as it always would, so Frame
  needs no per-family resize rules; the trial then checks what reached the model.
- **Images per sample:** the training code's own limit when it has one (`--max-images`), else the
  most images in any scanned row of the dataset (`scan_images`).
- **Expected image tokens per row** come from the model's processor on that image
  (`image_tokens`), and every ceiling trial checks the script fed exactly that many
  (`profiler/memtrial.py`). Fewer means the script's processor shrank or dropped images: the ceiling
  would describe smaller images than the job's.
"""
import json
import os

PLACEHOLDER = "{IMAGE}"


def parse_size(text):
    """'672x672' -> (672, 672)."""
    try:
        w, h = (int(x) for x in str(text).lower().split("x"))
    except ValueError:
        raise ValueError(f"image size must be WxH in pixels, e.g. 672x672; got {text!r}") from None
    if w < 1 or h < 1:
        raise ValueError(f"image size must be positive, got {text!r}")
    return w, h


def write_image(path, size):
    """A noise image of `size` (W, H) at `path`. Memory depends on the image's shape, not its
    content; noise just keeps an encoder from special-casing a blank one."""
    import numpy as np
    from PIL import Image
    w, h = size
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    Image.fromarray(np.random.RandomState(0).randint(0, 255, (h, w, 3), dtype=np.uint8)).save(path)
    return path


def fill_images(obj, path, n):
    """Put the image path where the row template says `{IMAGE}`. A list item holding the
    placeholder (an `images` list, or a chat message's `{"type": "image", ...}` part) is repeated
    `n` times, so a row carries the declared images per sample."""
    if isinstance(obj, str):
        return obj.replace(PLACEHOLDER, path)
    if isinstance(obj, list):
        out = []
        for x in obj:
            times = n if PLACEHOLDER in json.dumps(x) else 1
            out += [fill_images(x, path, n) for _ in range(times)]
        return out
    if isinstance(obj, dict):
        return {k: fill_images(v, path, n) for k, v in obj.items()}
    return obj


def _count(value):
    if value is None:
        return 0
    if isinstance(value, (list, tuple)):
        return sum(1 for v in value if v is not None)
    return 1


def _image_field(features):
    """The column holding images: an `Image` feature or a list of them, else one named
    image / images."""
    from datasets import Image, Sequence
    for name, f in (features or {}).items():
        inner = f.feature if isinstance(f, Sequence) else f[0] if isinstance(f, list) and f else f
        if isinstance(f, Image) or isinstance(inner, Image):
            return name
    for name in ("images", "image"):
        if name in (features or {}):
            return name
    return None


def scan_images(dataset, name=None, split="train", image_field=None, max_scan=None):
    """Images per row over the split (or its first `max_scan` rows), without decoding them.
    Returns {"field", "rows", "max", "rows_with_images"}; field None when no image column is
    found. Needs `datasets`."""
    from datasets import Image, Sequence, load_dataset
    ds = load_dataset(dataset, name, split=split, streaming=True)
    field = image_field or _image_field(ds.features)
    if field is None:
        return {"field": None, "rows": 0, "max": None, "rows_with_images": 0}
    f = (ds.features or {}).get(field)
    if isinstance(f, Image):                           # count, don't decode
        ds = ds.cast_column(field, Image(decode=False))
    elif isinstance(f, Sequence) and isinstance(f.feature, Image):
        ds = ds.cast_column(field, Sequence(Image(decode=False)))
    elif isinstance(f, list) and f and isinstance(f[0], Image):
        ds = ds.cast_column(field, [Image(decode=False)])
    rows = most = with_images = 0
    for i, row in enumerate(ds.select_columns([field])):
        if max_scan is not None and i >= max_scan:
            break
        c = _count(row.get(field))
        rows, most, with_images = rows + 1, max(most, c), with_images + (c > 0)
    return {"field": field, "rows": rows, "max": most, "rows_with_images": with_images}


def image_tokens(model, size, n):
    """(image tokens per row, image token id): what the model's processor makes of `n` images of
    `size`, at its default settings. (None, None) when the processor can't say (none, or it
    needs a library that isn't installed); the trial then can't check the image tokens."""
    try:
        from PIL import Image
        from transformers import AutoProcessor
        proc = AutoProcessor.from_pretrained(model)
        token = getattr(proc, "image_token", None)
        tid = proc.tokenizer.convert_tokens_to_ids(token) if token else None
        if tid is None:
            return None, None
        img = Image.new("RGB", size)
        for images in ([img] * n, [[img] * n]):        # flat (Qwen2-VL, LLaVA) or per-sample (Idefics3)
            try:
                enc = proc(text=[token * n], images=images, return_tensors="pt")
                return int((enc["input_ids"] == tid).sum()), int(tid)
            except Exception:
                continue
    except Exception:
        pass
    return None, None
