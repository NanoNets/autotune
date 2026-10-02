"""HF Trainer, settings in a YAML file whose path is HARDCODED (no flag to point it elsewhere).
Only the step budget is a flag. A tool can't swap in a different config without editing this file.

  python trainer_yaml_fixed_path.py --max_steps 100
"""
import argparse
import os
import sys

import yaml
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _data import RandomTokens  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_steps", type=int, default=100)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(os.path.join(HERE, "train.yaml")))      # fixed path
    cfg["training"]["max_steps"] = a.max_steps
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"],
                                                 attn_implementation=cfg.get("attn_implementation", "sdpa"))
    data = RandomTokens(model.config.vocab_size, 512)
    Trainer(model=model, args=TrainingArguments(**cfg["training"]), train_dataset=data).train()


if __name__ == "__main__":
    main()
