"""HF Trainer, every setting a command-line flag (HfArgumentParser). The common, easy case.

  python trainer_flags.py --output_dir /tmp/out --max_steps 100 --per_device_train_batch_size 8 \
      [--data_path rows.jsonl]

The data source is a flag too: `--data_path` (JSON lines of {"text": ...}); unset -> random tokens.
"""
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from transformers import AutoModelForCausalLM, HfArgumentParser, Trainer, TrainingArguments

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _data import JsonlText, RandomTokens, pad_collate, tokenizer_for  # noqa: E402


@dataclass
class ModelArgs:
    model_name: str = field(default="HuggingFaceTB/SmolLM2-360M")
    attn_implementation: str = field(default="sdpa")
    seq_len: int = field(default=512)
    data_path: Optional[str] = field(default=None)


def main():
    margs, targs = HfArgumentParser((ModelArgs, TrainingArguments)).parse_args_into_dataclasses()
    model = AutoModelForCausalLM.from_pretrained(margs.model_name,
                                                 attn_implementation=margs.attn_implementation)
    if margs.data_path:
        tok, pad = tokenizer_for(margs.model_name)
        data, collate = JsonlText(margs.data_path, tok, margs.seq_len), pad_collate(pad)
    else:
        data, collate = RandomTokens(model.config.vocab_size, margs.seq_len), None
    Trainer(model=model, args=targs, train_dataset=data, data_collator=collate).train()


if __name__ == "__main__":
    main()
