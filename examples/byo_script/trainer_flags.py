"""HF Trainer, every setting a command-line flag (HfArgumentParser). The common, easy case.

  python trainer_flags.py --output_dir /tmp/out --max_steps 100 --per_device_train_batch_size 8
"""
import os
import sys
from dataclasses import dataclass, field

from transformers import AutoModelForCausalLM, HfArgumentParser, Trainer, TrainingArguments

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _data import RandomTokens  # noqa: E402


@dataclass
class ModelArgs:
    model_name: str = field(default="HuggingFaceTB/SmolLM2-360M")
    attn_implementation: str = field(default="sdpa")
    seq_len: int = field(default=512)


def main():
    margs, targs = HfArgumentParser((ModelArgs, TrainingArguments)).parse_args_into_dataclasses()
    model = AutoModelForCausalLM.from_pretrained(margs.model_name,
                                                 attn_implementation=margs.attn_implementation)
    data = RandomTokens(model.config.vocab_size, margs.seq_len)
    Trainer(model=model, args=targs, train_dataset=data).train()


if __name__ == "__main__":
    main()
