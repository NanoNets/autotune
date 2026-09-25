"""Arm R: the same Llama + data in OUR kind of loop: a raw, `record_function`-labelled runner
that `capture()` drives. Mirrors what Trainer does per step where it changes the kernels:
bf16 autocast, AdamW over Trainer's two param groups (decay / no-decay), grad clip at 1.0,
DataLoader with pinned memory. Informational: shows what Trainer adds on top of a bare loop.
"""
import torch
from torch.profiler import record_function
from torch.utils.data import DataLoader

import workload


class RawHFRunner:
    def setup(self, config):
        self.device = config.get("device", "cuda")
        self.cfg = workload.resolve(config)
        self.model = workload.build_model(self.cfg).to(self.device)
        self.model.train()
        from transformers import Trainer
        decay = set(Trainer.get_decay_parameter_names(None, self.model))   # doesn't use self
        named = [(n, p) for n, p in self.model.named_parameters() if p.requires_grad]
        groups = [{"params": [p for n, p in named if n in decay], "weight_decay": 0.0},
                  {"params": [p for n, p in named if n not in decay], "weight_decay": 0.0}]
        self.opt = torch.optim.AdamW(groups, lr=1e-4)
        self.loader = DataLoader(workload.RandomTokens(self.cfg), batch_size=self.cfg["micro_batch"],
                                 shuffle=True, drop_last=True, pin_memory=True,
                                 num_workers=self.cfg["workers"])
        self._it = iter(self.loader)
        self.params = sum(p.numel() for _, p in named)

    def _next(self):
        try:
            return next(self._it)
        except StopIteration:
            self._it = iter(self.loader)
            return next(self._it)

    def step(self):
        accum = self.cfg["grad_accum"]
        for _ in range(accum):
            with record_function("dataloader"):
                batch = self._next()
            batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
            with record_function("forward"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = self.model(**batch).loss / accum
            with record_function("backward"):
                loss.backward()
        with record_function("optimizer"):
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)

    def tokens_per_step(self):
        return self.cfg["grad_accum"] * self.cfg["micro_batch"] * self.cfg["seq"]
