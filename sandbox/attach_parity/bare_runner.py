"""`frameworks/torch.py` with every `record_function` removed — the "user's script" arm.

Identical model, data and step order; only the annotations are gone. Attach has to recover them.
"""
import torch
import torch.nn as nn


class BareTorchRunner:
    def setup(self, config):
        self.device = config.get("device", "cpu")
        d = config.get("hidden", 256)
        layers = config.get("layers", 4)
        self.bs = config.get("micro_batch", 8)
        self.seq = config.get("max_len") or config.get("seq", 128)
        self.model = nn.Sequential(*[nn.Linear(d, d) for _ in range(layers)]).to(self.device)
        self.embed, self._batch = None, None

        if config.get("data") == "synthetic":
            from vendor.synthetic import random_batch
            vocab = config.get("vocab", 32000)
            self.embed = nn.Embedding(vocab, d).to(self.device)
            self._batch = random_batch(vocab, self.bs, self.seq, self.device,
                                       seed=config.get("seed", 0))["input_ids"]
        else:
            self._batch = torch.randn(self.bs, self.seq, d, device=self.device)

        params = list(self.model.parameters())
        if self.embed is not None:
            params += list(self.embed.parameters())
        self.opt = torch.optim.AdamW(params)
        self.params = sum(p.numel() for p in params)

    def step(self):
        x = self._batch
        if self.embed is not None:
            x = self.embed(x)
        loss = self.model(x).square().mean()
        loss.backward()
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)

    def tokens_per_step(self):
        return self.bs * self.seq
