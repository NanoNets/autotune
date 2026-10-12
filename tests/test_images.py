"""A VLM's image worst case for the memory ceiling (frame/images.py, profiler/memtrial.py): the
rows carry the declared max images, and every trial checks they reached the model at that size.
Text-only rows read the ceiling 2x too high once the vision tower
trains (Qwen2-VL-2B, A10)."""
import json
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame import ceiling as C  # noqa: E402
from frame.ceiling_rows import write_ceiling_rows  # noqa: E402
from frame.images import fill_images, parse_size, write_image  # noqa: E402
from profiler.memtrial import MemoryTrial, TrialDone  # noqa: E402

IMG = 99                                               # the image token id in these tests


def test_size_is_w_by_h():
    assert parse_size("672x448") == (672, 448)
    with pytest.raises(ValueError):
        parse_size("672")


def test_each_image_placeholder_in_a_list_is_repeated_per_image():
    t = {"images": ["{IMAGE}"], "messages": [{"role": "user", "content": [
        {"type": "image", "image": "{IMAGE}"}, {"type": "text", "text": "{TEXT}"}]}]}
    row = fill_images(t, "img.png", 3)
    assert row["images"] == ["img.png"] * 3
    content = row["messages"][0]["content"]
    assert [c["type"] for c in content] == ["image"] * 3 + ["text"]
    assert content[3]["text"] == "{TEXT}"                       # text is ceiling_rows' to fill


class _Tok:
    """A whitespace tokenizer over a tiny vocab: enough for write_ceiling_rows."""
    all_special_ids = []

    def __len__(self):
        return 50

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(f"w{i}" for i in ids)

    def encode(self, text, add_special_tokens=False):
        return text.split()


def test_image_rows_need_an_image_placeholder_and_carry_n_images(tmp_path):
    img = write_image(str(tmp_path / "ceiling_image_32x16.png"), (32, 16))
    from PIL import Image
    assert Image.open(img).size == (32, 16)
    with pytest.raises(ValueError, match="IMAGE"):
        write_ceiling_rows(str(tmp_path / "r.jsonl"), _Tok(), 8, 2, image=(img, 2))
    path = str(tmp_path / "rows.jsonl")
    write_ceiling_rows(path, _Tok(), 8, 2, template={"images": ["{IMAGE}"], "text": "{TEXT}"},
                       image=(img, 2))
    rows = [json.loads(line) for line in open(path)]
    assert all(r["images"] == [img, img] and len(r["text"].split()) > 8 for r in rows)


def _trial(rows_image_tokens, expect=4, steps=2):
    """MemoryTrial on a tiny model fed [2, 8] ids whose rows hold the given image-token counts."""
    trial = MemoryTrial(device="cpu", steps=steps, expect_shape=(2, 8),
                        expect_image_tokens=expect, image_token_id=IMG)
    trial.install()
    model = torch.nn.Sequential(torch.nn.Embedding(100, 4), torch.nn.Linear(4, 100))
    opt = torch.optim.AdamW(model.parameters())
    ids = torch.ones(2, 8, dtype=torch.long)
    for r, n in enumerate(rows_image_tokens):
        ids[r, :n] = IMG
    try:
        for _ in range(steps + 1):
            model(ids).logsumexp(-1).mean().backward()
            opt.step()
    except TrialDone:
        pass
    trial.uninstall()
    return trial.result("ok")


def test_a_trial_passes_when_every_row_carries_the_maxed_images():
    t = _trial([4, 4])
    assert t["status"] == "ok" and t["image_tokens_seen"] == [4]


def test_a_trial_fails_when_the_script_shrank_or_dropped_images():
    t = _trial([4, 2])                                          # one row's images shrank
    assert t["status"] == "error" and "image tokens, not 4" in t["error"]


def test_another_image_size_or_count_is_another_ceiling():
    cfg = {"script": "train.py", "memory_knobs": {"gradient_checkpointing": False},
           "dtype": "bfloat16", "max_len": 1024, "effective_batch": 16,
           "max_image_size": [672, 672], "max_images": 1}
    rec = {"seq_len": 1024, "effective_batch": 16, "config": C.stamp(cfg)}
    assert not C.is_stale(rec, cfg)
    assert C.is_stale(rec, {**cfg, "max_image_size": [448, 448]})
    assert C.is_stale(rec, {**cfg, "max_images": 2})


def test_frame_refuses_images_that_leave_no_room_for_text():
    from frame.dataset import distribution
    from frame.hardware import hardware
    from frame.model import internals
    from frame.profile import build
    cfg = dict(model_type="llama", num_hidden_layers=2, hidden_size=64, num_attention_heads=4,
               vocab_size=256, intermediate_size=128, max_position_embeddings=4096, hidden_act="silu")
    hw = hardware("A10", peak_flops=1e14, mem_gb=24.0, dtype="bfloat16")
    vlm = {**internals(cfg), "modality": "multimodal"}
    images = {"max_image_size": [672, 672], "max_images": 2, "image_tokens_per_row": 1152}
    with pytest.raises(ValueError, match="more than max_len"):
        build(vlm, distribution([256]), hw, {}, dtype="bfloat16", max_len=1024, images=images)
    p = build(vlm, distribution([256]), hw, {}, dtype="bfloat16", max_len=2048, images=images)
    assert p.images == images and any("carry 2 image(s) of 672x672" in n for n in p.notes)
