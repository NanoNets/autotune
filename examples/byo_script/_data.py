"""Shared by the example scripts: where their training rows come from.

- `RandomTokens`: seeded random token ids, every row exactly `seq_len`, so the examples run with no
  download. The default.
- `JsonlText`: a JSON-lines file of `{"text": ...}` rows, tokenized and truncated at `seq_len`,
  batched by `pad_collate` (pad to the batch's longest). How a real script reads a dataset path,
  and what a ceiling trial points at (Frame's `ceiling_rows.jsonl`).
- `hub_rows`: a hub dataset by id, for the "no path setting at all" case.
"""
import json

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


class TokenRows(Dataset):
    """Rows of token ids (lists), truncated at `seq_len`."""
    def __init__(self, rows, seq_len):
        self.rows = [r[:seq_len] for r in rows if r]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return {"input_ids": self.rows[i]}


def JsonlText(path, tokenizer, seq_len, text_field="text"):
    with open(path) as f:
        texts = [json.loads(line)[text_field] for line in f if line.strip()]
    return TokenRows([tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts], seq_len)


def hub_rows(name, config, tokenizer, seq_len, n=4096, text_field="text"):
    from datasets import load_dataset
    texts = [t for t in load_dataset(name, config, split="train")[text_field] if t.strip()][:n]
    return TokenRows([tokenizer(t, add_special_tokens=False)["input_ids"] for t in texts], seq_len)


def pad_collate(pad_id):
    """Pad each batch to its own longest row; labels ignore the padding."""
    def collate(batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        att = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            x = torch.as_tensor(b["input_ids"], dtype=torch.long)
            ids[i, :len(x)], att[i, :len(x)] = x, 1
        return {"input_ids": ids, "attention_mask": att, "labels": ids.masked_fill(att == 0, -100)}
    return collate


def tokenizer_for(model_name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    return tok, (tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
