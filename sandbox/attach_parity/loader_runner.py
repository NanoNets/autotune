"""Experiment 2 runners: a real torch DataLoader + gradient accumulation.

Same MLP as `vendor/torch_runner.py`, but batches come from a `DataLoader` (optionally with
workers and a CPU-side collate cost, so there is real data-wait to measure) and one `step()` is
`grad_accum` micro-batches of fwd/bwd followed by one optimizer step.

  LabelledLoaderRunner  arm A: phases wrapped in `record_function`, the way our runners do it
  BareLoaderRunner      arm C: the same loop with no annotations at all ("the user's script")

Both share `_Base` so the model, data and step order are identical; only the labels differ.
"""
import time

import torch
import torch.nn as nn
from torch.profiler import record_function
from torch.utils.data import DataLoader, TensorDataset


def _collate(cost_ms):
    """Stack the samples, then burn `cost_ms` of CPU: a stand-in for tokenization/augmentation."""
    def collate(items):
        x = torch.stack([it[0] for it in items])
        if cost_ms:
            end = time.perf_counter() + cost_ms / 1e3
            while time.perf_counter() < end:           # busy-wait: CPU work, not a sleep
                pass
        return x
    return collate


class _Base:
    def setup(self, config):
        self.device = config.get("device", "cpu")
        d = config.get("hidden", 256)
        layers = config.get("layers", 4)
        self.bs = config.get("micro_batch", 8)
        self.seq = config.get("seq", 128)
        self.grad_accum = config.get("grad_accum", 1)
        self.model = nn.Sequential(*[nn.Linear(d, d) for _ in range(layers)]).to(self.device)

        g = torch.Generator().manual_seed(config.get("seed", 0))
        n = self.bs * config.get("batches", 64)
        data = TensorDataset(torch.randn(n, self.seq, d, generator=g))
        workers = config.get("workers", 0)
        self.loader = DataLoader(data, batch_size=self.bs, shuffle=True, drop_last=True,
                                 num_workers=workers, persistent_workers=workers > 0,
                                 pin_memory=self.device == "cuda",
                                 collate_fn=_collate(config.get("collate_ms", 0)))
        self._it = iter(self.loader)

        self.opt = torch.optim.AdamW(self.model.parameters())
        self.params = sum(p.numel() for p in self.model.parameters())

    def _next(self):
        try:
            return next(self._it)
        except StopIteration:                          # new epoch; persistent workers survive it
            self._it = iter(self.loader)
            return next(self._it)

    def tokens_per_step(self):
        return self.grad_accum * self.bs * self.seq

    def teardown(self):
        self.model = self.opt = self.loader = self._it = None


class LabelledLoaderRunner(_Base):
    def step(self):
        for _ in range(self.grad_accum):
            with record_function("dataloader"):
                x = self._next()
            x = x.to(self.device, non_blocking=True)
            with record_function("forward"):
                loss = self.model(x).square().mean() / self.grad_accum
            with record_function("backward"):
                loss.backward()
        with record_function("optimizer"):
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)


class BareLoaderRunner(_Base):
    def step(self):
        for _ in range(self.grad_accum):
            x = self._next()
            x = x.to(self.device, non_blocking=True)
            loss = self.model(x).square().mean() / self.grad_accum
            loss.backward()
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
