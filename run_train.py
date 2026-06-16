#!/usr/bin/env python3
"""Autotune probe (managed by Claude Code agentic loop)."""
import sys
sys.path.insert(0, '/home/paperspace/repos/ms-swift')
import argparse, hashlib, itertools, json, os, subprocess
from pathlib import Path
import yaml

BASE_CONFIG = Path('configs/sft_base.yaml')
COMPLETED_RUNS_FILE = Path('completed_runs/runs.json')

GRID = {
    'model': ['Qwen/Qwen3.5-0.8B'],
    'per_device_train_batch_size': [16],
    'gradient_accumulation_steps': [4],
    'dataset_num_proc': [8],
    'dataloader_num_workers': [2],
    'dataloader_prefetch_factor': [2],
    'gradient_checkpointing': [True],
    'padding_free': [True],
    'dataloader_persistent_workers': [True],
}

CUDA_VISIBLE_DEVICES = '0'

def run_id(ov):
    return hashlib.md5(json.dumps(ov, sort_keys=True).encode()).hexdigest()[:8]

def build_args(overrides):
    with open(BASE_CONFIG) as f:
        base = yaml.safe_load(f) or {}
    merged = {**base, **overrides}
    args = []
    for k, v in merged.items():
        if isinstance(v, list):
            if v:
                args += [f'--{k}'] + [str(x) for x in v]
        elif isinstance(v, bool):
            args += [f'--{k}', str(v).lower()]
        elif v is not None:
            args += [f'--{k}', str(v)]
    return args

def load_state():
    COMPLETED_RUNS_FILE.parent.mkdir(parents=True, exist_ok=True)
    if COMPLETED_RUNS_FILE.exists():
        return json.loads(COMPLETED_RUNS_FILE.read_text())
    return {}

def save_state(s):
    COMPLETED_RUNS_FILE.write_text(json.dumps(s, indent=2))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--reset', action='store_true')
    parser.add_argument('--max-steps', type=int, default=None)
    args = parser.parse_args()
    if args.reset and COMPLETED_RUNS_FILE.exists():
        COMPLETED_RUNS_FILE.unlink()
    keys = list(GRID.keys())
    combos = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]
    state = load_state()
    pending = [c for c in combos if run_id(c) not in state]
    for ov in pending:
        rid = run_id(ov)
        extra = {'output_dir': f'output/autotune_{rid}', 'report_to': 'none'}
        if args.max_steps:
            extra['max_steps'] = args.max_steps
        cmd = ['swift', 'sft'] + build_args({**ov, **extra})
        env = {**os.environ, 'CUDA_VISIBLE_DEVICES': CUDA_VISIBLE_DEVICES}
        print(f'PROBE: {ov}', flush=True)
        rc = subprocess.run(cmd, env=env).returncode
        if rc != 0:
            sys.exit(rc)
        state[rid] = {'status': 'ok'}
        save_state(state)

if __name__ == '__main__':
    main()
