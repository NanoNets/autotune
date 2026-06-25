# Autotune search space (v1)

A **lever** is any `swift sft` CLI arg that changes **throughput or memory but not the trained model**
— same weights, same output, just faster or fitting in memory. The loop may set any such arg (the full
swift surface — see the docs). **The only rule: never set a knob that affects the model's output or
quality.** That's the agent's judgment, not code-enforced.

ms-swift `swift sft` parameter reference:
<https://swift.readthedocs.io/en/latest/Instruction/Command-line-parameters.html>

## How to choose

- **`effective_batch` is a fixed input** — the loop only picks the `bs × ga` factorization that hits it.
  `per_device_train_batch_size` is the primary lever; `ga` is derived (`effective_batch / (bs ×
  world_size)`), never set.
- **Pick by regime** the probe reveals: memory-bound / OOM → use less memory; compute-bound → cut
  wasted or slow compute; data-bound → feed the GPU faster.
- **Off-limits = anything that changes the model's output or quality** (optimization, regularization,
  precision, the sequence cap, what's trained). Setting one makes throughput numbers incomparable and
  changes the model — so don't. `max_length` is one of these: it's a fixed **input**, not a lever.

`attn_impl` is also a fixed **input** (config.json), not a lever — a per-model/hardware choice
(default `flash_attn`) that also selects the Modal image, so it's set once, not searched.

The harness fixes the measurement plumbing (step logging, token counting, truncation, pinned/persistent
dataloader) on every probe — those aren't levers.
