"""Shared by the example scripts: seeded random token ids, so the examples need no download.

A real script would load its dataset here; the examples keep data out of the way so what they
exercise is how a script exposes its settings.
"""
import torch
from torch.utils.data import Dataset


class RandomTokens(Dataset):
    def __init__(self, vocab, seq_len, n=2048, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.ids = torch.randint(0, vocab, (n, seq_len), generator=g)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        x = self.ids[i]
        return {"input_ids": x, "attention_mask": torch.ones_like(x), "labels": x.clone()}
