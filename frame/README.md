# frame/

The **framing** step. Runs **once** at the start of a run and produces the **Profile** — the
static picture of *what we're working with* (this model, this dataset, this GPU) that the agent
reads before it tunes anything. No training step, no weight download — the parameter count is
computed analytically from the config. The one on-device step is **measuring the GPU's peak
FLOP/s once** (the MFU denominator), which is then threaded to every probe.

Frame is one of two things that feed the agent's context. The other is the profiler's
**Observation** (recomputed every probe). The Profile is the constant one: computed once, then
held in context across every loop iteration as the grounding the agent never has to re-derive.

```
run.toml ──▶ frame (ONCE) ──▶ Profile (JSON)
                                  │
   agent context each iteration = Profile (constant) + History (grows) + Observation (newest probe) + Method + Budget
```

---

## Run it

```bash
python -m frame --model <hub-id|path> --dataset <hub-id|path> --gpu A10G \
                [--dtype bfloat16] [--device cuda] \
                [--name <dataset-config>] [--split train] [--text-field text] [--sample 1000]
```

Prints the one-line summary to stdout, then the full Profile JSON. Needs `transformers` + `datasets`,
and torch + a GPU (to measure the peak). The pure extractors underneath need none of it — see *Layout*.

Example (summary line, illustrative):

```bash
python -m frame --model Qwen/Qwen2.5-0.5B --dataset Salesforce/wikitext --name wikitext-2-raw-v1 --gpu A10G --sample 200
# Qwen2ForCausalLM (gqa, bfloat16) | 0.5B params | A10G 24.0GB x1 | peak 121 TFLOP/s (measured) | full-FT model-state 7.36GB -> fits (headroom 16.6GB) | 200 samples, p50 16/p99 317 tok
```

### The run spec (`run.toml`)

The durable, declarative input for a run lives in **`run.toml`** (gitignored; copy it from the
committed **`run.toml.example`**). It's the single source of truth the whole run reads from — Frame
for the Profile, the loop for the run inputs. Secrets (`HF_TOKEN`, Modal/SSH auth) stay in `.env`,
never here.

```toml
model   = "Qwen/Qwen2.5-0.5B"     # HF hub id or local path
dataset = "Salesforce/wikitext"   # HF hub id or local path
name    = "wikitext-2-raw-v1"     # dataset config/subset; omit if none
gpu     = "A10G"                  # a free-form label for the record (any hardware; peak is measured)

[scan]
text_field = "text"
split      = "train"
sample     = 1000
```

*(Wiring `frame` to read `run.toml` directly and write `runs/<id>/profile.json` is the next step —
right now it takes the same values as CLI flags and prints to stdout.)*

---

## Input contract

Every input is a plain string/int. `model` and `dataset` are the same idea — an **HF hub id or a
local path**, whatever `transformers`/`datasets` can resolve. `gpu` is **not** a link and **not**
validated: it's just a label for the record — any hardware works, because the peak is measured.

| Field | Format | Example |
|---|---|---|
| `model` | HF hub id ∣ local path | `"Qwen/Qwen2.5-0.5B"`, `"/models/ckpt"` |
| `dataset` | HF hub id ∣ local path | `"Salesforce/wikitext"`, `"/data/train.jsonl"` |
| `name` | dataset config ∣ `None` | `"wikitext-2-raw-v1"` |
| `gpu` | free-form label | `"A10G"`, `"H100-SXM"`, anything |
| `dtype` | torch dtype name | `"bfloat16"` (peak is measured at this) |
| `device` | torch device | `"cuda"` |
| `text_field` / `split` / `sample` | str / str / int | `"text"` / `"train"` / `1000` |

Measuring the peak is the first thing `frame()` does, so a run with no usable device fails **fast**,
before the model/dataset download.

---

## Output contract: the Profile

One compact JSON object (a `Profile` dataclass in Python). Small enough to inject **verbatim** into
every iteration's context. Five nested objects plus a few scalars and the summary line:

```json
{
  "model":    { "arch": "...", "params": 493961216, "attention": "gqa", ... },
  "dataset":  { "n_samples": 200, "total_tokens": 12426, "p50": 16, "p99": 317, ... },
  "hardware": { "gpu": "A10G", "count": 1, "peak_flops": 1.21e14, "peak_dtype": "bfloat16", "mem_gb": 24.0, "device_name": "NVIDIA A10", "num_sms": 72 },
  "versions": { "torch": "2.4", "transformers": "5.15.1", "flash_attn": null },
  "memory":   { "weights_gb": 0.92, "grads_gb": 0.92, "optimizer_gb": 3.68, "master_gb": 1.84, "model_state_gb": 7.36 },
  "default_config": { "dtype": "bfloat16", "param_bytes": 2, "optimizer": "adamw" },
  "flops_per_token": 2963767296.0,
  "activation_headroom_gb": 16.6,
  "model_state_fits": true,
  "summary": "Qwen2ForCausalLM (gqa, bf16) | 0.5B params | ..."
}
```

Every field has one of five **provenance classes**:

| Class | Meaning | Fields |
|---|---|---|
| **Declared** | read straight from the HF config | `layers, hidden, heads, kv_heads, head_dim, vocab, intermediate, context_len, dtype, moe, experts, experts_per_tok` |
| **Derived** | computed from declared fields | `params, params_total, attention, gated_mlp, modality`; `memory.*`, `flops_per_token`, `activation_headroom_gb`, `model_state_fits` |
| **Measured on-device** | run a GEMM benchmark / read the card | `hardware.peak_flops, mem_gb, device_name, num_sms` |
| **Introspected** | from the running environment | `versions.*` |
| **Sampled** | measured from the actual data | `dataset.*` |

### model
Normalized internals across arch variants (field names differ — llama/qwen/mixtral/gpt). `attention`
is derived (`kv_heads==heads`→mha, `==1`→mqa, else gqa). **`params`** is analytical — GQA, gated-MLP,
MoE-active, and tied-embedding aware — the N for the roofline; `params_total` differs only for MoE.
Matches published sizes to <1% (Llama-2-7B → 6.738B, Llama-3-8B → 8.03B, Mixtral → 12.9B active /
46.7B total, Qwen2.5-0.5B → 0.494B). No weights downloaded.

### dataset
Length distribution over the first `sample` rows, tokenized with the model's tokenizer:
`n_samples, total_tokens, mean_len, p50/p90/p99, max_len` (nearest-rank percentiles). Note
`total_tokens` is over the *sample*, not the whole corpus.

### hardware
`peak_flops` is **measured** — a large-GEMM benchmark (`profiler.capture.measure_peak_flops`) at the
run `dtype`, the real MFU denominator, measured **once** and threaded to every probe. No datasheet
table, so any chip works (`gpu` is just a label). `mem_gb`/`device_name`/`num_sms` are read from the
card. **Single-GPU** (`count: 1`).

### memory
Analytical model-state = weights + grads + optimizer + fp32 master (GiB, to match the profiler's units).
A **function of the knobs**, not a fixed assumption: the run `dtype` (→ `param_bytes`) and `optimizer`
are inputs, so the numbers move when the agent changes them. Standard ZeRO accounting — bf16 + AdamW
mixed precision = 16 bytes/param. Pairs with the profiler's *measured* peak-memory to recover
activations (measured − model_state). Recorded for a default full-FT config in `default_config`.

### roofline
`flops_per_token` = 6N. The per-step floor is `flops_per_token · tokens_per_step / peak_flops` — the
loop computes it per probe (tokens/step is a config, not framing) against the measured peak.
`model_state_fits` answers whether the **model state alone** (weights + grads + optimizer) fits.
It does not count activations, so `activation_headroom_gb` — what's left for them — is the number
that actually decides whether a config runs.

---

## Layout

Mirrors the profiler's **pure / impure split** — pure extractors are unit-testable on bare `python3`
(no transformers/datasets/GPU); thin shells hit the hub.

| File | Pure | Shell |
|---|---|---|
| `model.py` | `internals(config)` → normalized dict + analytical params | `load(model)` = AutoConfig |
| `dataset.py` | `distribution(lengths)` → percentiles + volume | `scan(dataset, tokenizer, …)` = streaming tokenize |
| `hardware.py` | `hardware(...)` = assemble from measured numbers; `versions()` | `measure(device, dtype)` = on-device peak + memory |
| `memory.py` | `training_memory(params, …)` = ZeRO breakdown | — |
| `profile.py` | `build(...)` = assemble the `Profile` | `frame(model, dataset, gpu, …)` = live end-to-end |
| `__main__.py` | — | the CLI |

The only profiler coupling is a runtime call to `profiler.capture.measure_peak_flops` on the live
path (`measure`). Frame no longer imports `roofline.py` at all — no static shared-file dependency.

---

## Design notes

- **No fixed, human-owned off-limits list.** The tunable surface is whatever the **framework** exposes
  (a `frameworks/` adapter's `knobs()`, deferred), and quality is protected empirically by the loop's
  **quality gate**, not a blocklist Frame writes in advance. The only fixed things are the problem
  *inputs* — model, dataset, hardware — which are exactly what the Profile captures.
- **Measure the peak *once*, thread it everywhere.** On cloud GPUs the measured peak swings ~28%
  run-to-run while the step-time noise floor is ~1%. If each probe measured its own peak, probe-to-probe
  MFU differences would be mostly measurement noise and the search would compare garbage. So Frame
  measures once and the loop passes that number to every `capture(..., peak_flops=...)`.
- **The peak is measured at the run `dtype`.** A fp32 model against a bf16 peak reads MFU wrong by ~10x,
  so the run `dtype` is carried in the Profile and passed through as `config["dtype"]`.
- **`model.dtype` is the config's *declared* dtype, not the run dtype.** The run `dtype` (a knob) is a
  separate input and drives both the memory model and the peak measurement.
- **`params` is analytical, not counted.** A formula (no weight download) — an intentional tradeoff;
  accurate to <1% against published sizes.

---

## Status

Extractors + assembler built and **unit-tested** (22 tests, bare `python3`, no GPU). The model +
dataset path was live-validated end to end under the previous API (the tied-embedding and
transformers-5.x `dtype` bugs it caught are fixed and regression-tested). The **measured-peak path**
(`measure` → `measure_peak_flops`) is new and **not yet re-validated live** — it needs the updated
profiler in the worktree (frame branched off an earlier profiler commit), so it verifies once frame
rebases onto the new profiler.

**Deferred:** multi-GPU topology (single-GPU for now, like distributed in the profiler); the framework
`knobs()` surface (belongs in `frameworks/`); reading `run.toml` + writing `runs/<id>/profile.json`
(the loop seam); true corpus volume (streaming gives sample tokens, not a full row count);
published-comparable MFU (a datasheet peak passthrough — dropped for now).
