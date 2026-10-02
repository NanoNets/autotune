"""A plain PyTorch training loop (no Trainer): argparse flags, a torch DataLoader, grad accumulation.

  python torch_flags.py --max-steps 100 --micro-batch 8 --grad-accum 4
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _data import RandomTokens  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--attn", default="sdpa")
    a = ap.parse_args()
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 attn_implementation=a.attn).cuda()
    loader = DataLoader(RandomTokens(model.config.vocab_size, a.seq_len), batch_size=a.micro_batch,
                        shuffle=True, num_workers=a.workers, pin_memory=True,
                        generator=torch.Generator().manual_seed(42))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    it = iter(loader)
    for step in range(1, a.max_steps + 1):
        for _ in range(a.grad_accum):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(loader)
                batch = next(it)
            batch = {k: v.cuda(non_blocking=True) for k, v in batch.items()}
            (model(**batch).loss / a.grad_accum).backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step % 10 == 0:
            print(f"step {step}")


if __name__ == "__main__":
    main()
