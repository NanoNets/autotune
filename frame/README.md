# frame/

**Frame is the "know what you're dealing with" step.** It runs **once** at the start of a tuning
run and produces one small JSON file — the **Profile** — that describes the job before any tuning
begins. Think of it as the scouting report the agent reads before it starts trying to make training
faster.

You give it four things — **which model, which dataset, which GPU, and the `max_len` your
application needs** — and it answers six questions about them.

---

## Setup checklist (`program.md` Step 2)

Get the model and dataset verified and understood before touching any knob — a thing you can't
load is a setup failure, not an experiment result.

**Model** (named in `run.toml`):

- **Confirm it's retrievable.** If it's a HuggingFace repo id, confirm you can actually pull its
  config/weights (valid `HF_TOKEN`, correct repo id/revision, gated access resolved). If it's a
  local checkpoint path, confirm it exists and actually loads.
- **Build a real architecture picture**, not just a name: layers, hidden size, attention heads vs.
  kv_heads (GQA/MQA changes memory/parallelism math), intermediate size, vocab size, MoE
  experts/top_k if applicable, attention type, param count, modality, dtype. This is the `model`
  field of the `Profile` contract (`loop/contracts.md`) and it directly determines which levers in
  `loop/throughput.md` even apply — e.g. `expert_parallel` is meaningless on a dense model, GQA
  changes the memory/throughput tradeoff of recompute and batch knobs.
- **Where to get it, in order:**
  1. `knowledge.md` and `knowledge/beliefs.md` (see `knowledge/README.md`) — someone may have
     already profiled this exact model (or its architecture family) on similar hardware. Don't
     re-derive what's already known. This is the *architecture* lookup; the full-configuration
     match (model · dataset · hardware · dtype · `effective_batch` · `max_len`) happens in the same
     phase at `loop/loop.md` Frame §3, and a hit there is replayed as experiment 2.
  2. The model's own config (`config.json`/model card on the hub, or the local checkpoint's
     config) — the ground truth for the numeric fields.
  3. Websearch — the model card, paper, or announcement, especially for a newer or less
     documented model. Don't guess at architecture details from the name alone.
- Whatever websearch turns up, write it into `runs/<run_name>/research.md` as you find it (see
  `runs/README.md`) — that's the landing zone, not `knowledge.md` directly. If this is a new model
  this repo hasn't seen before, the architecture summary you build here is exactly the kind of
  thing worth promoting into a durable `knowledge.md` entry (or belief) at Step 5's end-of-run
  review, so the next run against it starts from an answer, not from scratch.

**`max_len`** (named in `run.toml`, REQUIRED):

- It is the human's declaration about their application, not a knob — off-limits, like
  `effective_batch` (`loop/throughput.md`). Never change it to make something fit.
- Don't confuse the three lengths. `max_len` is the human's; `model.context_len` is the
  architecture's hard cap (Frame refuses `max_len` above it); `dataset.max_len`/`p99` is what the
  data happens to contain. Only the first is the yardstick for the ceiling.

**Dataset**:

- **Local path** — confirm it exists, is readable, and matches the format the training script
  expects (schema, modality, on-disk layout). Don't assume; open a sample and check.
- **HuggingFace dataset** — confirm you can actually pull it (valid `HF_TOKEN` in `.env`, correct
  repo id/revision, gated-access resolved).

---

## The six questions it answers

1. **What's the model?** Reads the model's config and works out its real shape: layers, hidden size,
   parameter count, attention type (MHA / GQA / MoE), context length. The parameter count is computed
   with math — **no multi-gigabyte weight download** — and lands within 1% of the published sizes.

2. **What's the data?** Draws a uniform random pool over the whole split (20× the sample, blank
   rows skipped), tokenizes it, and reports how long the sequences are (median, 90th/99th
   percentile, max). That shape drives batch-size and packing decisions later. From the same pool it
   draws the **probe sample**: `sample` rows at evenly spaced length quantiles, so the rows every
   probe times carry the corpus's length distribution, tail included. With `--sample-out` it writes
   them to `runs/<run>/sample.jsonl`, and every probe loads that file (`sample_file`). Frame's
   statistics and the probes' rows are then one population (the Measurement axiom,
   `loop/contracts.md`). It is not a head slice: the first N rows of an ordered file aren't a sample
   of it, and on wikitext that alone moved throughput 1.9x.

3. **What's the hardware?** **Measures** how fast the GPU actually is (peak FLOP/s) and its memory
   bandwidth, right there on the card — not from a spec sheet. This measured peak is the yardstick
   everything else is judged against ("are we using the GPU well?").

4. **Will it fit, on paper?** Does the memory math — weights + gradients + optimizer state — and
   whether that leaves room to actually train on this card. Analytical; activations are not in it.

5. **What's installed?** Records library versions, so the loop knows which speed levers even exist
   (no flash-attention installed → that knob is off the table).

6. **How much actually fits?** **Measures** the largest `micro_batch` that runs at `max_len` — the
   OOM ceiling. This is the one that decides whether a config runs, and it is measured, not
   derived. See below.

---

## The memory ceiling — measure it, don't hope for it

**Memory is decided by the largest batch a run will ever see, not the average one.** On wikitext
(p50 16 tok, p99 317) a batch drawn from real data almost never holds a 2048-token sequence, so a
memory reading taken from one is a random variable: a config passes ten clean steps at Frame and
OOMs three hours into training, when a long batch finally arrives. That is not a noisy number —
it's a wrong recommendation.

So the worst case is **manufactured, not sampled**: every sequence exactly `max_len`, attention
mask all ones, ids drawn uniformly from the vocab and seeded (`frame/synthetic.py`). Activation
memory is a function of shape, not content — which is why this is sound, and why it needs no
dataset, no tokenizer and no dataloader. Then `frame/ceiling.py` brackets the boundary by doubling
and bisects it: **~2·log₂(n) trials of a few seconds each**, not a linear walk.

```
frame: measuring the memory ceiling at max_len=2048 (synthetic, seed 0)
  mb=1   ok    peak  8.9GB
  mb=2   ok    peak 10.4GB
  mb=4   ok    peak 13.4GB
  mb=8   ok    peak 19.5GB
  mb=16  oom   reached 22.0GB
  mb=12  oom   reached 22.1GB
  mb=10  ok    peak 21.4GB
  mb=11  ok    peak 21.9GB      ← ceiling
```

**What this buys the loop:** `micro_batch` stops being an unbounded search dimension. With
`effective_batch = 16`, micro_batch must divide 16, so the ceiling of 11 leaves `{1, 2, 4, 8}` —
and `16` is eliminated *before* the loop spends a 5-minute probe watching it crash.

**Read the record, don't just read the number:**

| Field | Use it for |
|---|---|
| `micro_batch_max` | the bound on the `micro_batch` knob. `0` → nothing fits (see below) |
| `capped` | `true` → the search hit `effective_batch`; the real ceiling is higher, memory isn't binding |
| `oom_at` | the smallest micro_batch that failed |
| `frag_gb` / `card_frac` | how close to the edge. Fixed-shape probing does NOT reproduce the fragmentation varying real lengths cause — at a high `card_frac`, keep a margin below the ceiling rather than running at it |
| `config` | **the knobs it was measured under.** This is how a stale ceiling is detected |

### Rules

- **Dynamic padding is safe IFF `micro_batch <= ceiling`.** Every real batch is at most `max_len`
  wide, so a config that fits the synthetic worst case fits every batch the dataset can produce.
  That is the whole guarantee — and it is what makes `padding="longest"` usable at all.
- **NEVER raise `micro_batch` on a probe's `headroom_gb`.** Under dynamic padding a probe's peak
  reflects the rows it happened to draw, not the dataset's longest, so its headroom is an artifact
  of the sample. On `test_a10G` the scan saw max 520 tok against `max_len=2048`: no probe in 23
  experiments ran the `[8, 2048]` batch the shipped config meets in training — only the ceiling did.
  The ceiling bounds `micro_batch`; an observed peak never does.
- **`dtype` is not a memory lever.** It is in the staleness list because a human changing precision
  invalidates the measurement. IF nothing fits, escalate to the human with the numbers — recompute,
  offload, sharding and 8-bit optimizer states are yours; precision is theirs.
- **The ceiling is NOT a run constant.** `recompute`, `attn_kernel`, `optimizer`, `dtype`,
  sharding and **`compile`** all move it. `compile` is not an obvious member of that list but was
  measured moving a ceiling from 2 to 4: Inductor fuses the cross-entropy and never materializes
  the fp32 logits copies, cutting peak memory 5.4GB
  (`knowledge/compile-cuts-memory-not-just-time.md`). Before trusting a ceiling, compare the
  Config against the record's `config` (`frame.ceiling.is_stale`). Moved → **re-measure**, don't
  extrapolate. Turning on `recompute`
  can easily take a ceiling of 11 to 32; the search space is meant to widen on that evidence.
- **Measure it with the REAL training step.** A ceiling measured on a different workload describes
  that workload, not this job. `--ceiling` uses `--framework`'s real runner (`frameworks/hf.py`
  loads real weights and runs a real fwd/bwd/opt); `frame.profile.frame(..., runner=...)` takes one
  directly. Constructing a runner is never implicit — no `runner`, no ceiling.
- **The bound lands on the knob.** `build()` stamps `micro_batch.max` from the measured ceiling
  (`_bound_knobs`). A framework's `knobs()` can only declare `min: 1` — it takes no arguments and
  cannot know a per-card measurement — so without this the SearchSpace would advertise an unbounded
  int and the bound would hold only if the agent remembered to cross-reference.
- **`micro_batch=1` OOMs → stop and say so.** That is a memory regime, not a throughput one: the
  first moves are recompute / offload / sharding, never batch knobs. Frame puts it in `notes`.
- **Synthetic probes NEVER enter `results.tsv`.** Memory feasibility is measured on the worst case;
  throughput is measured on the real distribution. Two questions, two ledgers
  (`runs/README.md`).
- **`max_len` ≫ dataset `p99` is a flag, not an error.** Frame notes it: most of every unpacked
  batch is padding, so `packing` is likely a large win — or `max_len` is set higher than this data
  needs. Raise it with the human at `program.md` Step 3; never change `max_len` yourself.

---

## How to run it

```bash
python -m frame --model <hub-id|path> --dataset <hub-id|path> --gpu A10G --max-len 2048 \
                [--dtype bfloat16] [--device cuda] \
                [--name <dataset-config>] [--split train] [--sample 1000] \
                [--sample-out runs/<run>/sample.jsonl] [--sample-seed 0] [--max-scan N] \
                [--ceiling --effective-batch 16 --seed 0]
```

Prints a one-line summary, then any `NOTE:` lines, then the full Profile JSON. Example summary:

```
Qwen2ForCausalLM (gqa, bfloat16) | 0.5B params | A10G 24.0GB x1 | peak 121 TFLOP/s (measured) | full-FT model-state 7.36GB -> model state fits (headroom 16.6GB) | ceiling @2048: micro_batch<=11 (measured) | 200 samples, p50 16/p99 317 tok
```

`--max-len` is **required** — without it there is no worst case to measure against. `--ceiling`
runs the search with `--framework`'s real runner, so the number is one to tune against; it costs
~2·log₂(n) short trials. Omit it and the Profile carries `ceiling: null`.

`--sample` is rounded down to a multiple of `--effective-batch` (1000 → 992 at 16), so one pass over
the probe sample is whole optimizer steps. The pool streams the whole split by default: seconds for
a fine-tuning set, since only the pool is tokenized. `--max-scan` caps the rows streamed for a corpus
too large to read end to end. The pool is then random over that prefix only, and the sample file's
`meta.max_scan` records it.

**Inputs.** `model` and `dataset` are each an **HF hub id or a local path**. `gpu` is just a **label**
for the record — any hardware works, because the peak is measured, not looked up. `dtype` is the
precision the model runs in (the peak is measured at this). `max_len` is the application's sequence
length — human-owned, off-limits, and the length the ceiling is measured at. The durable place to put these is
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
  "max_len": 2048,
  "ceiling":  { "micro_batch_max": 11, "seq_len": 2048, "oom_at": 12, "capped": false,
                "peak_gb": 21.9, "frag_gb": 0.4, "card_frac": 0.93,
                "config": { "dtype": "bfloat16", "optimizer": "adamw" }, "trials": [ ... ] },
  "notes":    [ ],
  "summary": "..."
}
```

- **model** — the model's shape + analytical parameter count.
- **dataset** — how long the sequences are, and how many tokens the sample held.
- **hardware** — the **measured** peak FLOP/s (the MFU yardstick) + bandwidth + card memory.
- **memory** — where the memory goes; pairs with the profiler's *measured* peak to reveal activations.
- **flops_per_token / activation_headroom_gb / model_state_fits** — the roofline math and the
  analytical fit check (`model_state_fits` is weights+grads+optimizer only).
- **max_len / ceiling** — the human's declared sequence length and the **measured** micro_batch
  ceiling at it. This is what actually decides whether a config runs; `activation_headroom_gb` only
  says how much room there was to work with. `null` ceiling means it was not measured — it never
  means "fine".
- **notes** — things the agent must read before tuning (memory regime, padding-heavy `max_len`, a
  ceiling sitting at the edge of the card). Facts and consequences, never a verdict.

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
| `dataset.py` | the random pool → sequence-length distribution, and the length-stratified probe sample (`draw_sample`, `write_sample`, `read_sample`) |
| `hardware.py` | assemble hardware facts; `measure()` runs the on-device peak/bandwidth benchmark |
| `memory.py` | analytical weights/grads/optimizer breakdown |
| `synthetic.py` | worst-case batch at `max_len` — every sequence full length, seeded |
| | *consumed by `frameworks/hf.py` (`data="synthetic"`) and `frameworks/torch.py`* |
| `ceiling.py` | `search` = the pure double-then-bisect ladder; `find_ceiling` = the live search |
| `profile.py` | assemble the `Profile` (`build` = pure; `frame` = live end-to-end) |
| `__main__.py` | the `python -m frame` CLI |

The only profiler dependency is a runtime call to `profiler.capture` (peak/bandwidth/card facts) on the
live path — Frame no longer imports anything static from the profiler.

---

## Status

Built and **unit-tested** (bare `python3`, no GPU — the ceiling ladder is pure, and the live
search is walked on CPU with a simulated OOM), and **live-validated end to end on an A10G** — real `Qwen/Qwen2.5-0.5B` config, real `wikitext` scan, measured card, then the Profile's
frozen ceilings handed to a real probe, which reused them verbatim:

```
Qwen2ForCausalLM (gqa, bfloat16) | 0.5B params | A10G 22.1GB x1 | peak 89 TFLOP/s (measured)
 | full-FT model-state 7.36GB -> model state fits (headroom 14.7GB) | 200 samples, p50 16/p99 317 tok
```

**Single-GPU** for now; multi-GPU and reading `run.toml` / writing `runs/<id>/profile.json` are
deferred. The ceiling is built, unit-tested, and wired to the real `frameworks/hf.py` runner; it
still wants a **GPU validation run** — every trial's memory field is `None` off-GPU, so only a real
card exercises the bisection against actual OOMs. A second ceiling at the dataset's `p99` — whose
gap from the `max_len` ceiling estimates what `packing` is worth — is one extra
`find_ceiling(..., seq_len=p99)` call, not yet wired into `frame()`.
