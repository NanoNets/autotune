"""A user's HF Trainer script: build model + dataset, TrainingArguments, trainer.train().
No autotune imports, no annotations. Run directly, or under launcher.py (arm L).

  python train_hf.py --config '{...}' --max-steps 16 [--time-out steps.json --warmup 3 --clean 10]

`--time-out` is arm T only: a TrainerCallback times each optimizer step (CUDA sync, then the
clock, at `on_step_end`) with no hooks installed: an independent measurement of this same run.

Probe settings: logging, eval and checkpointing are off, so no step pays for them (a real
probe would set these the same way, alongside `max_steps`).
"""
import argparse
import json
import os
import sys
import tempfile
import time

import torch
from transformers import Trainer, TrainerCallback, TrainingArguments

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import versions                                                  # noqa: E402
import workload                                                  # noqa: E402


class StepTimer(TrainerCallback):
    """Times optimizer steps with no hooks. `mode` sets WHERE the CUDA sync happens:

      end  sync at `on_step_end`: after lr_scheduler.step(), zero_grad and Trainer bookkeeping
      opt  sync at `on_optimizer_step`: right after optimizer.step(), the same point the
           hooks (attach.py) sync at. Also records the host time from that sync to
           `on_step_end`: the host work this placement makes wait for the GPU.
      two  sync only at the start and end of the window: 2 syncs for `clean` steps, the least
           perturbing timing. One mean, no per-step times.
    """

    def __init__(self, warmup, clean, mode="end"):
        self.warmup, self.clean, self.mode = warmup, clean, mode
        self.times, self.last, self.host_after_sync, self._t_sync = [], None, [], None

    def _tick(self, n):                                          # n = steps completed so far
        torch.cuda.synchronize()
        now = time.perf_counter()
        if n == self.warmup:
            self.last = now
        elif self.warmup < n <= self.warmup + self.clean:
            self.times.append(now - self.last)
            self.last = now
        return now

    def on_optimizer_step(self, args, state, control, **kw):
        if self.mode == "opt":                                   # fires before global_step += 1
            self._t_sync = self._tick(state.global_step + 1)

    def on_step_end(self, args, state, control, **kw):
        n = state.global_step
        if self.mode == "end":
            self._tick(n)
        elif self.mode == "opt":
            if self._t_sync is not None and self.warmup < n <= self.warmup + self.clean:
                self.host_after_sync.append(time.perf_counter() - self._t_sync)
        elif self.mode == "two" and n in (self.warmup, self.warmup + self.clean):
            torch.cuda.synchronize()
            now = time.perf_counter()
            if n == self.warmup:
                self.last = now
            else:
                self.times = [(now - self.last) / self.clean] * self.clean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="{}")
    ap.add_argument("--max-steps", type=int, default=16)
    ap.add_argument("--time-out", default=None)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--clean", type=int, default=10)
    ap.add_argument("--time-mode", choices=["end", "opt", "two"], default="end")
    a = ap.parse_args()
    got = versions.check()
    cfg = workload.resolve(json.loads(a.config))

    model = workload.build_model(cfg)
    args = TrainingArguments(
        output_dir=tempfile.mkdtemp(prefix="hf_probe_"),
        max_steps=a.max_steps,
        per_device_train_batch_size=cfg["micro_batch"],
        gradient_accumulation_steps=cfg["grad_accum"],
        bf16=True,
        optim="adamw_torch",                                     # explicit: the default depends on torch
        learning_rate=1e-4,
        dataloader_num_workers=cfg["workers"],
        logging_strategy="no", save_strategy="no", eval_strategy="no",
        report_to="none", disable_tqdm=True, seed=cfg["seed"],
    )
    timer = StepTimer(a.warmup, a.clean, a.time_mode) if a.time_out else None
    trainer = Trainer(model=model, args=args, train_dataset=workload.RandomTokens(cfg),
                      callbacks=[timer] if timer else None)
    trainer.train()

    if timer:
        with open(a.time_out, "w") as f:
            json.dump({"times": timer.times, "mode": timer.mode,
                       "host_after_sync": timer.host_after_sync, "versions": got,
                       "truth": workload.truth(cfg, model), "loaded": workload.describe(model)}, f)
    print(f"train_hf.py: {trainer.state.global_step} steps")


if __name__ == "__main__":
    main()
