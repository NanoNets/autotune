"""A plain PyTorch LoRA fine-tune (no Trainer): `torch_flags.py` with the base model frozen and PEFT
adapters on every attention / MLP projection. The optimizer gets the adapters only.

  python torch_lora.py --max-steps 100 --micro-batch 8 --grad-accum 4 [--lora-r 16] [--data-path rows.jsonl]

Defaults match BYO experiment 7: r=16, alpha 32, dropout 0.05,
seq 512, micro_batch 8, grad_accum 4, random tokens. Needs `peft`.
"""
import argparse
import os
import sys

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _data import JsonlText, RandomTokens, pad_collate, tokenizer_for  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-360M")
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--data-path", default=None, help="JSON lines of {\"text\": ...}; unset -> random")
    a = ap.parse_args()
    torch.manual_seed(42)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16,
                                                 attn_implementation=a.attn)
    model = get_peft_model(model, LoraConfig(r=a.lora_r, lora_alpha=a.lora_alpha,
                                             lora_dropout=a.lora_dropout, bias="none",
                                             target_modules=TARGETS, task_type="CAUSAL_LM")).cuda()
    if a.data_path:
        tok, pad = tokenizer_for(a.model)
        data, collate = JsonlText(a.data_path, tok, a.seq_len), pad_collate(pad)
    else:
        data, collate = RandomTokens(model.config.vocab_size, a.seq_len), None
    loader = DataLoader(data, batch_size=a.micro_batch, shuffle=True, num_workers=a.workers,
                        pin_memory=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(42))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
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
