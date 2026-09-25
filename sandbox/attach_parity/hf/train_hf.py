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
    def __init__(self, warmup, clean):
        self.warmup, self.clean, self.times, self.last = warmup, clean, [], None

    def on_step_end(self, args, state, control, **kw):
        torch.cuda.synchronize()
        now = time.perf_counter()
        n = state.global_step                                    # steps completed so far
        if n == self.warmup:
            self.last = now
        elif self.warmup < n <= self.warmup + self.clean:
            self.times.append(now - self.last)
            self.last = now


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="{}")
    ap.add_argument("--max-steps", type=int, default=16)
    ap.add_argument("--time-out", default=None)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--clean", type=int, default=10)
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
    timer = StepTimer(a.warmup, a.clean) if a.time_out else None
    trainer = Trainer(model=model, args=args, train_dataset=workload.RandomTokens(cfg),
                      callbacks=[timer] if timer else None)
    trainer.train()

    if timer:
        with open(a.time_out, "w") as f:
            json.dump({"times": timer.times, "versions": got,
                       "truth": workload.truth(cfg, model)}, f)
    print(f"train_hf.py: {trainer.state.global_step} steps")


if __name__ == "__main__":
    main()
