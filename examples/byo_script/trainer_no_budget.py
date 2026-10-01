"""HF Trainer whose settings are flags, but with NO way to cap the run: it trains for a fixed 3
epochs over 2048 rows. A probe can't stop it after ~15 steps without an edit.

  python trainer_no_budget.py --per_device_train_batch_size 8 --gradient_accumulation_steps 4
"""
import argparse
import os
import sys

from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _data import RandomTokens  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_device_train_batch_size", type=int, default=8)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=4)
    ap.add_argument("--gradient_checkpointing", action="store_true")
    a = ap.parse_args()
    model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M",
                                                 attn_implementation="sdpa")
    data = RandomTokens(model.config.vocab_size, 512)
    args = TrainingArguments(
        output_dir="/tmp/byo_trainer_no_budget", num_train_epochs=3,
        per_device_train_batch_size=a.per_device_train_batch_size,
        gradient_accumulation_steps=a.gradient_accumulation_steps,
        gradient_checkpointing=a.gradient_checkpointing,
        learning_rate=1e-4, bf16=True, optim="adamw_torch", seed=42,
        logging_steps=10, save_strategy="no", report_to="none", disable_tqdm=True)
    Trainer(model=model, args=args, train_dataset=data).train()


if __name__ == "__main__":
    main()
