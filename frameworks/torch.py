"""Minimal raw-torch runner — the first framework adapter.

A runner exposes `setup(config)` + `step()`, wrapping each phase in `record_function` so the
trace parser can attribute time. This one is a tiny MLP over random data: a real fwd/bwd/opt
step (enough to exercise the whole capture→trace→diagnose pipeline) that runs on CPU/MPS/CUDA.
Swap in `frameworks/swift.py` later to profile a real fine-tune; the profiler is unchanged.
"""
import torch
import torch.nn as nn
from torch.profiler import record_function


class TorchRunner:
    def setup(self, config):
        self.device = config.get("device", "cpu")
        d = config.get("hidden", 256)
        layers = config.get("layers", 4)
        self.bs = config.get("micro_batch", 8)
        self.seq = config.get("seq", 128)
        self.model = nn.Sequential(*[nn.Linear(d, d) for _ in range(layers)]).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters())
        self.params = sum(p.numel() for p in self.model.parameters())
        self._batch = torch.randn(self.bs, self.seq, d, device=self.device)  # data stand-in

    def _next_batch(self):
        with record_function("dataloader"):
            return self._batch                      # real runner fetches from a DataLoader here

    def step(self):
        x = self._next_batch()
        with record_function("forward"):
            loss = self.model(x).square().mean()
        with record_function("backward"):
            loss.backward()
        with record_function("optimizer"):
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)

    def tokens_per_step(self):
        return self.bs * self.seq
