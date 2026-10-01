# examples/byo_script/

Example **user training scripts** for bring-your-own-script mode (`README.md`, "Bring your own
training script"). Each one is standalone, with no autotune imports, and shows one way a script
can expose (or fail to expose) its settings. They're also the fixtures for the workflow
verification in `docs/verification/byo_script.md`.

They train `HuggingFaceTB/SmolLM2-360M` on seeded random token ids (`_data.py`), so they need no
dataset download. A real script loads its own data.

| Script | Settings live in | What the agent does (`program.md`, "Script mode") |
|---|---|---|
| `trainer_flags.py` | CLI flags (`HfArgumentParser`) | tunes flags via `script_args` |
| `trainer_yaml.py` + `train.yaml` (+ `data.yaml`) | YAML files passed by flag (`--config`, `--data-config`) | per-probe config copies under `runs/<run>/probes/` |
| `trainer_yaml_fixed_path.py` + `train.yaml` | a YAML at a **hardcoded path** | asks permission to expose the path as a flag (on a copy) |
| `trainer_hardcoded.py` | **hardcoded** in `TrainingArguments(...)` | asks: move them to a config file / expose as flags / leave untuned |
| `trainer_no_budget.py` | flags, but **no step budget** | asks permission to add one (on a copy), or stops |
| `torch_flags.py` | `argparse` flags, plain PyTorch loop | tunes flags via `script_args` |

Run one under autotune:

```
/run-loop my_run model=HuggingFaceTB/SmolLM2-360M host=user@gpu-box gpu=A10 \
    script=examples/byo_script/trainer_yaml.py script_configs=examples/byo_script/train.yaml
```
