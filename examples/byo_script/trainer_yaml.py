"""HF Trainer, settings in YAML files passed by flag: `--config` (training) and optionally
`--data-config` (data). The config-file case.

  python trainer_yaml.py --config train.yaml [--data-config data.yaml]

The data source is a config key: `path` in the data config (JSON lines of {"text": ...}); unset ->
random tokens.
"""
import argparse
import os
import sys

import yaml
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _data import JsonlText, RandomTokens, pad_collate, tokenizer_for  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--data-config", default=None)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    data_cfg = yaml.safe_load(open(a.data_config)) if a.data_config else {"seq_len": 512}
    model = AutoModelForCausalLM.from_pretrained(cfg["model_name"],
                                                 attn_implementation=cfg.get("attn_implementation", "sdpa"))
    if data_cfg.get("path"):
        tok, pad = tokenizer_for(cfg["model_name"])
        data, collate = JsonlText(data_cfg["path"], tok, data_cfg["seq_len"]), pad_collate(pad)
    else:
        data, collate = RandomTokens(model.config.vocab_size, data_cfg["seq_len"]), None
    Trainer(model=model, args=TrainingArguments(**cfg["training"]), train_dataset=data,
            data_collator=collate).train()


if __name__ == "__main__":
    main()
