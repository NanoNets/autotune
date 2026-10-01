"""HF Trainer, settings in YAML files passed by flag: `--config` (training) and optionally
`--data-config` (data). The config-file case.

  python trainer_yaml.py --config train.yaml [--data-config data.yaml]
"""
import argparse
import os
import sys

import yaml
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _data import RandomTokens  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-config", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    data_cfg = yaml.safe_load(open(a.data_config)) if a.data_config else {"seq_len": 512}
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"],
                                                 attn_implementation=cfg.get("attn_implementation", "sdpa"))
    data = RandomTokens(model.config.vocab_size, data_cfg["seq_len"])
    Trainer(model=model, args=TrainingArguments(**cfg["training"]), train_dataset=data).train()


if __name__ == "__main__":
    main()
