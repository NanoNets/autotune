"""A BUGGY script: it accepts `--data_path` but trains on its fixed file anyway. A ceiling trial
pointed at the ceiling rows must FAIL here (the batches aren't [micro_batch, max_len]), not
report a ceiling measured on the wrong rows. A test fixture, not a pattern to copy.

  python make_data.py            # once: writes data/train.jsonl
  python trainer_ignores_data.py --output_dir /tmp/out --max_steps 100 --data_path rows.jsonl
"""
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

from transformers import AutoModelForCausalLM, HfArgumentParser, Trainer, TrainingArguments

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _data import JsonlText, pad_collate, tokenizer_for  # noqa: E402

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "train.jsonl")


@dataclass
class ModelArgs:
    model_name: str = field(default="HuggingFaceTB/SmolLM2-360M")
    seq_len: int = field(default=512)
    data_path: Optional[str] = field(default=None)        # parsed, then (wrongly) unused


def main():
    margs, targs = HfArgumentParser((ModelArgs, TrainingArguments)).parse_args_into_dataclasses()
    model = AutoModelForCausalLM.from_pretrained(margs.model_name, attn_implementation="sdpa")
    tok, pad = tokenizer_for(margs.model_name)
    data = JsonlText(DATA_PATH, tok, margs.seq_len)
    Trainer(model=model, args=targs, train_dataset=data, data_collator=pad_collate(pad)).train()


if __name__ == "__main__":
    main()
