"""A user's training script, written the way a user would: top-level argparse, no autotune
imports, no annotations. The workload is `loader_runner.BareLoaderRunner`'s, inlined, so arm L
(this script under launcher.py) can be compared with arms A and C op-for-op.

  python train.py --config '{"device": "cuda", "grad_accum": 4, ...}' --steps 20
"""
import argparse
import json
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def make_collate(cost_ms):
    def collate(items):
        x = torch.stack([it[0] for it in items])
        if cost_ms:
            end = time.perf_counter() + cost_ms / 1e3
            while time.perf_counter() < end:
                pass
        return x
    return collate


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="{}")
    ap.add_argument("--steps", type=int, default=20, help="optimizer steps (a probe's max_steps)")
    args = ap.parse_args()
    cfg = json.loads(args.config)

    device = cfg.get("device", "cpu")
    d, layers = cfg.get("hidden", 256), cfg.get("layers", 4)
    bs, seq, accum = cfg.get("micro_batch", 8), cfg.get("seq", 128), cfg.get("grad_accum", 1)
    workers = cfg.get("workers", 0)

    model = nn.Sequential(*[nn.Linear(d, d) for _ in range(layers)]).to(device)
    g = torch.Generator().manual_seed(cfg.get("seed", 0))
    data = TensorDataset(torch.randn(bs * cfg.get("batches", 64), seq, d, generator=g))
    loader = DataLoader(data, batch_size=bs, shuffle=True, drop_last=True, num_workers=workers,
                        persistent_workers=workers > 0, pin_memory=device == "cuda",
                        collate_fn=make_collate(cfg.get("collate_ms", 0)))
    opt = torch.optim.AdamW(model.parameters())

    it = iter(loader)
    for _ in range(args.steps):
        for _ in range(accum):
            try:
                x = next(it)
            except StopIteration:
                it = iter(loader)
                x = next(it)
            x = x.to(device, non_blocking=True)
            loss = model(x).square().mean() / accum
            loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    print(f"train.py: finished {args.steps} steps")


if __name__ == "__main__":
    main()
