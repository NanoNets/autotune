# frame/

**Frame is the "know what you're dealing with" step.** It runs **once** at the start of a tuning
run and produces one small JSON file — the **Profile** — that describes the job before any tuning
begins. Think of it as the scouting report the agent reads before it starts trying to make training
faster.

You give it three things — **which model, which dataset, which GPU** — and it answers five questions
about them.

---

## The five questions it answers

1. **What's the model?** Reads the model's config and works out its real shape: layers, hidden size,
   parameter count, attention type (MHA / GQA / MoE), context length. The parameter count is computed
   with math — **no multi-gigabyte weight download** — and lands within 1% of the published sizes.

2. **What's the data?** Tokenizes a sample of the dataset and reports how long the sequences are
   (median, 90th/99th percentile, max). That shape drives batch-size and packing decisions later.

3. **What's the hardware?** **Measures** how fast the GPU actually is (peak FLOP/s) and its memory
   bandwidth, right there on the card — not from a spec sheet. This measured peak is the yardstick
   everything else is judged against ("are we using the GPU well?").

4. **Will it fit?** Does the memory math — weights + gradients + optimizer state — and whether that
   leaves room to actually train on this card.

5. **What's installed?** Records library versions, so the loop knows which speed levers even exist
   (no flash-attention installed → that knob is off the table).

---

## How to run it

```bash
python -m frame --model <hub-id|path> --dataset <hub-id|path> --gpu A10G \
                [--dtype bfloat16] [--device cuda] \
                [--name <dataset-config>] [--split train] [--sample 1000]
```

Prints a one-line summary, then the full Profile JSON. Example summary:

```
Qwen2ForCausalLM (gqa, bfloat16) | 0.5B params | A10G 24.0GB x1 | peak 121 TFLOP/s (measured) | full-FT model-state 7.36GB -> model state fits (headroom 16.6GB) | 200 samples, p50 16/p99 317 tok
```

**Inputs.** `model` and `dataset` are each an **HF hub id or a local path**. `gpu` is just a **label**
for the record — any hardware works, because the peak is measured, not looked up. `dtype` is the
precision the model runs in (the peak is measured at this). The durable place to put these is
**`run.toml`** (copy from the committed `run.toml.example`; it's gitignored). Secrets go in `.env`.

---

## What comes out: the Profile

One compact JSON object — small on purpose, so it can sit in the agent's context every iteration.

```json
{
  "model":    { "arch": "Qwen2ForCausalLM", "params": 493961216, "attention": "gqa", "layers": 24, ... },
  "dataset":  { "n_samples": 200, "total_tokens": 12426, "p50": 16, "p99": 317, ... },
  "hardware": { "gpu": "A10G", "peak_flops": 1.21e14, "peak_dtype": "bfloat16", "mem_gb": 24.0, ... },
  "versions": { "torch": "2.4", "transformers": "5.15.1", "flash_attn": null },
  "memory":   { "weights_gb": 0.92, "grads_gb": 0.92, "optimizer_gb": 3.68, "model_state_gb": 7.36 },
  "default_config": { "dtype": "bfloat16", "param_bytes": 2, "optimizer": "adamw" },
  "flops_per_token": 2963767296.0,
  "activation_headroom_gb": 16.6,
  "model_state_fits": true,
  "summary": "..."
}
```

- **model** — the model's shape + analytical parameter count.
- **dataset** — how long the sequences are, and how many tokens the sample held.
- **hardware** — the **measured** peak FLOP/s (the MFU yardstick) + bandwidth + card memory.
- **memory** — where the memory goes; pairs with the profiler's *measured* peak to reveal activations.
- **flops_per_token / activation_headroom_gb / model_state_fits** — the roofline math and the fit check.
  (`model_state_fits` is weights+grads+optimizer only; `activation_headroom_gb` — what's left for
  activations — is the number that actually decides whether a config runs.)

---

## How it fits into the loop

The whole run is a scientific-method loop:

```
FRAME (once) ──▶ [ HYPOTHESIZE ──▶ TEST ──▶ OBSERVE ──▶ LEARN ] repeat ──▶ OUTPUT best config
```

Frame is the **FRAME** step — it runs a single time and hands the agent the Profile. From then on, the
agent reads two kinds of things each iteration:

```
agent context each iteration =
    Profile        ← from frame, CONSTANT (the scouting report, never changes)
  + Observation    ← from the profiler, NEW every probe (how the last config actually ran)
  + History        ← grows each iteration (what's been tried and learned)
  + Method + Budget
```

So there are **two measurement components**, and they split cleanly:

| | Frame → **Profile** | Profiler → **Observation** |
|---|---|---|
| When | once, at the start | every probe |
| Answers | *what are we working with* | *how did that config run* |
| In context | constant background | the newest result |

Frame gives the agent solid ground to reason from; the profiler tells it what happened when it tried
something. The MFU number the profiler reports uses the **peak Frame measured** — which is why Frame
measures it, and why it does so only once (below).

---

## The one thing it's careful about

**Measure the GPU's peak speed once, reuse it for every experiment.** On cloud GPUs the measured peak
swings ~28% run-to-run, while the step-time noise is ~1%. If each experiment measured its own peak, the
comparisons *between* experiments would be mostly measurement noise. So Frame measures once, at the run
dtype, and that same number is threaded to every probe. (A peak measured at the wrong precision — fp32
vs bf16 — is off by ~10x, so Frame ties the peak to the run's dtype and refuses a mismatch.)

## What it deliberately does *not* do

- **No hardcoded "safe to tune" rules.** Frame lists facts; the loop's quality check decides what's safe.
- **No training, no picking the config.** It sets the table; the agent does the tuning.

---

## Layout

Each piece has a **pure** part (just math/parsing — testable on bare `python3`, no GPU) and, where
needed, a **shell** that hits the hub or the card.

| File | Does |
|---|---|
| `model.py` | HF config → model shape + analytical params |
| `dataset.py` | tokenize a sample → sequence-length distribution |
| `hardware.py` | assemble hardware facts; `measure()` runs the on-device peak/bandwidth benchmark |
| `memory.py` | analytical weights/grads/optimizer breakdown |
| `profile.py` | assemble the `Profile` (`build` = pure; `frame` = live end-to-end) |
| `__main__.py` | the `python -m frame` CLI |

The only profiler dependency is a runtime call to `profiler.capture` (peak/bandwidth/card facts) on the
live path — Frame no longer imports anything static from the profiler.

---

## Status

Built and **unit-tested** (24 tests, bare `python3`, no GPU), and **live-validated end to end on an
A10G** — real `Qwen/Qwen2.5-0.5B` config, real `wikitext` scan, measured card, then the Profile's
frozen ceilings handed to a real probe, which reused them verbatim:

```
Qwen2ForCausalLM (gqa, bfloat16) | 0.5B params | A10G 22.1GB x1 | peak 89 TFLOP/s (measured)
 | full-FT model-state 7.36GB -> model state fits (headroom 14.7GB) | 200 samples, p50 16/p99 317 tok
```

**Single-GPU** for now; multi-GPU, the framework's tunable-knob list, and reading `run.toml` /
writing `runs/<id>/profile.json` are deferred.
