"""PaLM's attention term (Chowdhery et al. 2022, arXiv:2204.02311, App. B): 12 * L * H * Q per
attended (query, key) pair. Which layers attend over what comes off the config
(profiler/attention.py); the pairs per step come from the batch (profiler/attach.py)."""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from profiler.attention import spec  # noqa: E402
from profiler.roofline import attention_flops  # noqa: E402

LLAMA = dict(num_hidden_layers=4, num_attention_heads=8, hidden_size=512)


def test_a_dense_model_has_one_full_attention_part():
    parts, skipped = spec(LLAMA)
    assert parts == [{"part": "lm_full", "layers": 4, "hq": 512, "window": None, "backward": True}]
    assert skipped == []


def test_layer_types_split_full_and_sliding_and_leave_out_linear_attention():
    cfg = {**LLAMA, "head_dim": 32, "sliding_window": 128,
           "layer_types": ["sliding_attention", "full_attention", "linear_attention", "sliding_attention"]}
    parts, skipped = spec(cfg)
    assert {p["part"]: (p["layers"], p["hq"], p["window"]) for p in parts} == {
        "lm_full": (1, 256, None), "lm_window": (2, 256, 128)}
    assert "linear_attention" in skipped[0]                    # Qwen3.5-style: left out, named


def test_a_disabled_sliding_window_is_full_attention():
    parts, _ = spec({**LLAMA, "sliding_window": 4096, "use_sliding_window": False})   # Qwen2
    assert [p["part"] for p in parts] == ["lm_full"]


def test_a_vision_tower_attends_on_its_own_width_and_can_be_forward_only():
    cfg = {"text_config": LLAMA, "vision_config": {"depth": 32, "embed_dim": 1280, "hidden_size": 1536}}
    parts, _ = spec(cfg, backward=True, vision_backward=False)
    vision = next(p for p in parts if p["part"] == "vision")
    assert (vision["layers"], vision["hq"], vision["backward"]) == (32, 1280, False)


def test_windowed_vision_blocks_are_left_out_not_guessed():
    cfg = {"text_config": LLAMA, "vision_config": {"depth": 32, "hidden_size": 1280,
                                                   "fullatt_block_indexes": [7, 15, 23, 31]}}
    parts, skipped = spec(cfg)
    assert next(p for p in parts if p["part"] == "vision")["layers"] == 4
    assert "28 windowed blocks" in skipped[0]


def test_the_term_is_palms_12_LHQ_per_pair_or_4_forward_only():
    parts = [{"part": "lm_full", "layers": 2, "hq": 10, "window": None, "backward": True},
             {"part": "vision", "layers": 3, "hq": 5, "window": None, "backward": False}]
    assert attention_flops(parts, {"lm_full": 100, "vision": 7}) == 12 * 2 * 10 * 100 + 4 * 3 * 5 * 7


# ---- pairs per step, through the attached capture --------------------------------------------

from profiler.attach import Attach  # noqa: E402

WARMUP, CLEAN = 2, 3


class _Model(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.config = cfg
        self.emb, self.head = torch.nn.Embedding(50, 16), torch.nn.Linear(16, 50)

    def forward(self, input_ids, **kw):
        return self.head(self.emb(input_ids))


def _pairs(kw, cfg=None):
    torch.manual_seed(0)
    att = Attach(device="cpu", warmup=WARMUP, clean_steps=CLEAN, peak_flops=1.0, peak_mem_bw_gb_s=1.0)
    att.install()
    model = _Model(cfg or LLAMA)
    opt = torch.optim.AdamW(model.parameters())
    for _ in range(WARMUP + CLEAN + 2):
        model(**kw).logsumexp(-1).mean().backward()
        opt.step()
        opt.zero_grad()
    return att.step_attention_pairs


def test_unpacked_rows_attend_over_their_padded_width():
    assert _pairs({"input_ids": torch.randint(0, 50, (4, 10))}) == [{"lm_full": 4 * 10 * 10}] * CLEAN


def test_a_packed_row_attends_within_each_document():
    pos = torch.cat([torch.arange(6), torch.arange(4), torch.arange(10)])[None]   # docs 6, 4, 10
    got = _pairs({"input_ids": torch.randint(0, 50, (1, 20)), "position_ids": pos})
    assert got == [{"lm_full": 36 + 16 + 100}] * CLEAN                    # not 20^2 = 400


def test_sliding_window_layers_cap_the_attended_length():
    cfg = {**LLAMA, "sliding_window": 4}
    got = _pairs({"input_ids": torch.randint(0, 50, (2, 10))}, cfg)
    assert got == [{"lm_window": 2 * 10 * 4}] * CLEAN


def test_a_vision_tower_attends_within_each_image():
    """Qwen-VL: `image_grid_thw` rows give each image's patches (t*h*w); pairs = sum of patches^2."""
    from types import SimpleNamespace as NS
    cfg = NS(text_config=NS(**LLAMA), vision_config=NS(depth=2, embed_dim=16, patch_size=4))

    class VLM(_Model):
        def __init__(self):
            super().__init__(cfg)
            self.visual = torch.nn.Linear(16, 16)

        def forward(self, input_ids, pixel_values, image_grid_thw):
            return self.head(self.emb(input_ids) + self.visual(pixel_values).mean())

    torch.manual_seed(0)
    att = Attach(device="cpu", warmup=WARMUP, clean_steps=CLEAN, peak_flops=1.0, peak_mem_bw_gb_s=1.0)
    att.install()
    model = VLM()
    opt = torch.optim.AdamW(model.parameters())
    grid = torch.tensor([[1, 4, 6], [1, 2, 2]])                    # two images: 24 and 4 patches
    for _ in range(WARMUP + CLEAN + 2):
        model(input_ids=torch.randint(0, 50, (2, 8)), pixel_values=torch.randn(28, 16),
              image_grid_thw=grid).logsumexp(-1).mean().backward()
        opt.step()
        opt.zero_grad()
    assert att.step_attention_pairs == [{"lm_full": 2 * 64, "vision": 24 ** 2 + 4 ** 2}] * CLEAN
    assert att.step_patches == [28] * CLEAN
