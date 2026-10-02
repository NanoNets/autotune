"""HF Trainer reading PRE-TOKENIZED rows (`prep.py`'s `.pt`) by flag.

  python prep.py --in data/train.jsonl --out data/train.pt
  python trainer_pretok.py --output_dir /tmp/out --max_steps 100 --data_path data/train.pt
"""
import os
import sys
from dataclasses import dataclass, field

import torch
from transformers import AutoModelForCausalLM, HfArgumentParser, Trainer, TrainingArguments

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _data import TokenRows, pad_collate, tokenizer_for  # noqa: E402


@dataclass
class ModelArgs:
    model_name: str = field(default="HuggingFaceTB/SmolLM2-360M")
    seq_len: int = field(default=512)
    data_path: str = field(default="data/train.pt")


def main():
    margs, targs = HfArgumentParser((ModelArgs, TrainingArguments)).parse_args_into_dataclasses()
    model = AutoModelForCausalLM.from_pretrained(margs.model_name, attn_implementation="sdpa")
    _, pad = tokenizer_for(margs.model_name)
    data = TokenRows(torch.load(margs.data_path), margs.seq_len)
    Trainer(model=model, args=targs, train_dataset=data, data_collator=pad_collate(pad)).train()


if __name__ == "__main__":
    main()
