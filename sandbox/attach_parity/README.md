# attach_parity — sandbox

Isolated experiment: does hook-based capture (`attach.py`) record the same information as the
`record_function`-labelled runner, and are its measurements within the noise floor?
Context and design: `bring_your_own_script.md` §2 at the repo root.

**Fully isolated.** Nothing here imports the live `profiler/`, `frameworks/` or `frame/`. The
baseline is a frozen copy in `vendor/` (source commit in `vendor/__init__.py`), and `run_arm.py`
fails if any live module gets imported. Iterate on `attach.py` freely; the baseline can't move.

| File | Role |
|---|---|
| `vendor/` | frozen baseline: capture, diagnostic, trace, roofline, synthetic, torch runner |
| `bare_runner.py` | `vendor/torch_runner.py` with every `record_function` removed ("user's script") |
| `attach.py` | the hook-driven capture under test |
| `loader_runner.py` | experiment 2 workload: real `DataLoader` + grad accum, labelled (A) and bare (C) |
| `train.py` | the same workload as a standalone user script: argparse, no autotune imports (arm L) |
| `launcher.py` | runs a script with Attach installed first, same process; writes the Capture JSON |
| `hf/` | experiment 4: HF Trainer. `requirements-hf.txt` + `versions.py` (exact pins, checked every run), `workload.py` (shared Llama + data), `train_hf.py` (the user's Trainer script), `raw_runner.py` (arm R), `run_hf_arm.py`, `hf_parity.py` |
| `run_arm.py` | one arm, fresh process, prints a JSON record |
| `parity.py` | driver: freezes ceilings, alternates A/C for N rounds, reports both checks |

Arms: **A** `capture()` drives the labelled runner · **C** hooks installed in-process around the
bare runner · **L** `launcher.py -- train.py`, a separate process, result read back from JSON.

```bash
# experiment 1 (A vs C)
python sandbox/attach_parity/parity.py --device cuda --rounds 5 \
    --config '{"hidden":2048,"layers":8,"micro_batch":32,"seq":512}'
# experiments 2-3 (A vs C vs L)
python sandbox/attach_parity/parity.py --device cuda --runner loader --arms C,L --rounds 5 \
    --config '{"hidden":2048,"layers":8,"micro_batch":8,"seq":512,"grad_accum":4,"batches":64,"workers":0,"collate_ms":20}'
```

The launcher on its own (what the agent would run on a user's script):

```bash
python sandbox/attach_parity/launcher.py --out cap.json --device cuda -- train.py --config '{...}' --steps 16
```

Experiment 4 (HF Trainer) needs its own venv with the exact pins:

```bash
python3 -m venv --system-site-packages venv_hf          # on top of the box's CUDA torch 2.7.0
venv_hf/bin/pip install -r sandbox/attach_parity/hf/requirements-hf.txt
venv_hf/bin/python sandbox/attach_parity/hf/hf_parity.py --rounds 5
```

`results_*.json` are the raw records behind `bring_your_own_script.md` experiments 1-4; the
`trace_path`s inside them point at the A10 box and are not kept.
