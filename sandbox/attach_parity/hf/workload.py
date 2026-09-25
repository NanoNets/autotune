"""The one workload every experiment-4 arm runs: a small Llama built from config (random init,
no download) on random token ids. Shared so L, T and R build the identical model and data.

Config keys: hidden, intermediate, layers, heads, vocab, seq, micro_batch, grad_accum, samples,
workers, seed.
"""
import torch
from torch.utils.data import Dataset

DEFAULTS = {"hidden": 1024, "intermediate": 2816, "layers": 8, "heads": 16, "vocab": 32000,
            "seq": 512, "micro_batch": 8, "grad_accum": 4, "samples": 1024, "workers": 0,
            "seed": 0}


def resolve(cfg):
    return {**DEFAULTS, **cfg}


def build_model(cfg):
    from transformers import LlamaConfig, LlamaForCausalLM
    c = resolve(cfg)
    torch.manual_seed(c["seed"])
    conf = LlamaConfig(hidden_size=c["hidden"], intermediate_size=c["intermediate"],
                       num_hidden_layers=c["layers"], num_attention_heads=c["heads"],
                       num_key_value_heads=c["heads"], vocab_size=c["vocab"],
                       max_position_embeddings=c["seq"], attn_implementation="sdpa")
    return LlamaForCausalLM(conf)


class RandomTokens(Dataset):
    """Fixed random ids; labels = input_ids (causal LM); no padding, so the mask is all ones."""

    def __init__(self, cfg):
        c = resolve(cfg)
        g = torch.Generator().manual_seed(c["seed"])
        self.ids = torch.randint(0, c["vocab"], (c["samples"], c["seq"]), generator=g)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        x = self.ids[i]
        return {"input_ids": x, "attention_mask": torch.ones_like(x), "labels": x.clone()}


def truth(cfg, model):
    c = resolve(cfg)
    return {"params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "tokens": c["grad_accum"] * c["micro_batch"] * c["seq"],
            "grad_accum": c["grad_accum"]}
