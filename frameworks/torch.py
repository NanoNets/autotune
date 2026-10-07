"""Minimal raw-torch runner — the first framework adapter.

A runner exposes `setup(config)` + `step()`, wrapping each phase in `record_function` so the
trace parser can attribute time. This one is a tiny MLP over random data: a real fwd/bwd/opt
step (enough to exercise the whole capture->trace->diagnose pipeline) that runs on CPU/MPS/CUDA.
Swap in `frameworks/swift.py` later to profile a real fine-tune; the profiler is unchanged.

Two data sources. `data="synthetic"` takes a worst-case batch from `frame.synthetic` — every
sequence exactly `max_len`, through an embedding so token ids actually flow — which is what a
ceiling search drives. Anything else is the random-float stand-in. `teardown()` exists so a
ceiling search can free a trial's allocations before the next one starts.
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
        self.seq = config.get("max_len") or config.get("seq", 128)
        self.model = nn.Sequential(*[nn.Linear(d, d) for _ in range(layers)]).to(self.device)
        self.embed, self._batch = None, None

        if config.get("data") == "synthetic":
            from frame.synthetic import random_batch        # worst case, built not sampled
            vocab = config.get("vocab", 32000)
            self.embed = nn.Embedding(vocab, d).to(self.device)
            self._batch = random_batch(vocab, self.bs, self.seq, self.device,
                                       seed=config.get("seed", 0))["input_ids"]
        else:
            self._batch = torch.randn(self.bs, self.seq, d, device=self.device)  # data stand-in

        params = list(self.model.parameters())
        if self.embed is not None:
            params += list(self.embed.parameters())
        self.opt = torch.optim.AdamW(params)
        self.params = sum(p.numel() for p in params)

    def _next_batch(self):
        with record_function("dataloader"):
            return self._batch                      # real runner fetches from a DataLoader here

    def step(self):
        x = self._next_batch()
        with record_function("forward"):
            if self.embed is not None:              # synthetic: ids -> activations
                x = self.embed(x)
            loss = self.model(x).square().mean()
        with record_function("backward"):
            loss.backward()
        with record_function("optimizer"):
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)

    def teardown(self):
        """Free this config's allocations, so a ceiling search's next trial starts clean."""
        self.model = self.opt = self.embed = self._batch = None

    def tokens_per_step(self):
        return self.bs * self.seq

    def last_step_samples(self):
        return self.bs
