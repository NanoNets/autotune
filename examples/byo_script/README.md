# examples/byo_script/

Example **user training scripts** for bring-your-own-script mode (`program.md`, "Script
mode"). Each one is standalone, with no autotune imports, and shows one way a script
can expose (or fail to expose) its settings. They're also the fixtures for the workflow
verification in `docs/verification/byo_script.md`.

They train `HuggingFaceTB/SmolLM2-360M`. The scripts in the first table train by default on seeded
random token ids (`_data.py`), so they need no dataset download; the ones with a data setting can
read a JSON-lines file instead, which is also how a **ceiling trial** points them at Frame's
`ceiling_rows.jsonl` (`frame/README.md`, "User scripts").

| Script | Settings live in | What the agent does (`program.md`, "Script mode") |
|---|---|---|
| `trainer_flags.py` | CLI flags (`HfArgumentParser`) | tunes flags via `script_args` |
| `trainer_yaml.py` + `train.yaml` (+ `data.yaml`) | YAML files passed by flag (`--config`, `--data-config`) | per-probe config copies under `runs/<run>/probes/` |
| `trainer_yaml_fixed_path.py` + `train.yaml` | a YAML at a **hardcoded path** | asks permission to expose the path as a flag (on a copy) |
| `trainer_hardcoded.py` | **hardcoded** in `TrainingArguments(...)` | asks: move them to a config file / expose as flags / leave untuned |
| `trainer_no_budget.py` | flags, but **no step budget** | asks permission to add one (on a copy), or stops |
| `torch_flags.py` | `argparse` flags, plain PyTorch loop | tunes flags via `script_args` |
| `torch_lora.py` | `argparse` flags, plain PyTorch loop, **LoRA** (PEFT, base frozen; needs `peft`) | tunes flags via `script_args`; LoRA rank / alpha / targets are off-limits |

Where the **data source** lives decides how a ceiling trial reaches it:

| Script | Data source | A ceiling trial points at the rows through |
|---|---|---|
| `trainer_flags.py`, `torch_flags.py`, `torch_lora.py` | a flag (`--data_path` / `--data-path`) | `script_args` |
| `trainer_yaml.py` + `data.yaml` | a config key (`path`) | a per-trial config copy, `runs/<run>/probes/ceiling_mbN.yaml` |
| `trainer_hardcoded_data.py` | a **hardcoded** path (`data/train.jsonl`) | the Step 2b question: move it to a flag or config, on a copy |
| `trainer_hub_data.py` | a **hub id** only (`load_dataset("Salesforce/wikitext", ...)`) | treated as hardcoded: Step 2b |
| `trainer_pretok.py` (+ `prep.py`) | **pre-tokenized** `.pt`, by flag | the agent runs `prep.py` on the ceiling rows, then `script_args` |
| `trainer_ignores_data.py` | a flag it ignores (a bug, on purpose) | the trial **fails** on the batch shape: the guard for exactly this |

The data-source scripts in the second table never use random tokens: `trainer_hardcoded_data.py`
and `trainer_ignores_data.py` read `data/train.jsonl`, `trainer_hub_data.py` loads wikitext from the
hub, and `trainer_pretok.py` reads `prep.py`'s `.pt`. `make_data.py` writes `data/train.jsonl`
(wikitext-2, untracked) for the scripts that read a fixed file.

Run one under autotune:

```
/run-loop my_run model=HuggingFaceTB/SmolLM2-360M host=user@gpu-box gpu=A10 \
    script=examples/byo_script/trainer_yaml.py script_configs=examples/byo_script/train.yaml
```
