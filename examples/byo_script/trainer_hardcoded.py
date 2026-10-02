"""HF Trainer with every speed setting HARDCODED in TrainingArguments. Only the step budget and
the output directory are flags. The settings can't be tuned without editing this file.

  python trainer_hardcoded.py --max_steps 100
"""
import argparse
import os
import sys

from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _data import RandomTokens  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_steps", type=int, default=100)
    ap.add_argument("--output_dir", default="/tmp/byo_trainer_hardcoded")
    a = ap.parse_args()
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M",
                                                 attn_implementation="sdpa")
    data = RandomTokens(model.config.vocab_size, 512)
    args = TrainingArguments(
        output_dir=a.output_dir,
        max_steps=a.max_steps,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=4,
        gradient_checkpointing=False,
        learning_rate=1e-4,
        bf16=True,
        optim="adamw_torch",
        seed=42,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        disable_tqdm=True,
    )
    Trainer(model=model, args=args, train_dataset=data).train()


if __name__ == "__main__":
    main()
