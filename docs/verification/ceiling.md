# Verification: the memory ceiling redesign (`design_improvements.md` §7)

> Private paper trail for the PR that makes ceiling trials memory-only, searches only divisors of
> `effective_batch`, re-measures from the previous ceiling on memory-knob changes, and gives user
> scripts a ceiling. In the style of `docs/verification/byo_script.md`; remove before
> open-sourcing. Run 2026-10-02.

## Method

- **Box:** 1× NVIDIA A10 (23.5 GB usable), torch 2.7.0 / CUDA 12.8 (system), Python 3.10.12.
- **Probe venv** `~/autotune/.venv_at`: transformers 5.15.1, huggingface-hub 1.28.0, tokenizers
  0.22.2, safetensors 0.8.0, accelerate 1.14.0, peft 0.20.0, datasets 5.0.1, numpy 1.24.4,
  pillow 12.3.0, networkx 3.4.2, PyYAML 5.4.1 (system). `flash_attn` not installed (R10 needs that).
- **Driver:** Frame ran on the box (`python -m frame`). Every ceiling trial went through
  `backends.run` with the ssh backend from a laptop, exactly as `python -m frame.ceiling` calls it.
- **Known values to compare against**, from runs already on record on the same card type:
  - `test_a10G` (Qwen2.5-0.5B, seq 2048, `effective_batch` 16): ceilings 2, then 4 with compile,
    then 8 with checkpointing added (`runs/test_a10G/ceiling.tsv`).
  - SmolLM2-360M through the example HF Trainer script at seq 512, `effective_batch` 32:
    micro-batch 16 fits, 32 OOMs, with or without compile (`experience/my_byo_yaml.md`,
    `docs/verification/byo_script.md` row 14).
- **Trials:** 3 optimizer steps each unless stated. Raw records in `runs/ceil_*` (gitignored):
  `profile.json`, `ceiling.tsv`, every candidate config and its log.

## Results

**All 22 rows pass.**

### Our runner (Qwen2.5-0.5B, wikitext-2, seq 2048, `effective_batch` 16)

| # | Case | Result | Trials | Pass |
|---|---|---|---|---|
| R1 | Fresh ceiling at Frame | 1, 2 fit, 4 OOMs → **2**. Peaks 8.98 / 15.028 GB allocated, 10.271 / 16.93 GB reserved: identical to `test_a10G`'s record | 3 | ✅ |
| R2 | Trial length | peak at micro-batch 2: 1 step 13.18 GB, 2 / 3 / 5 steps 15.03 GB. Under compile at micro-batch 4: 1 step 15.93, 2 / 3 / 5 steps 17.79 GB. One step reads 1.9 GB low (AdamW's state, first allocated at step 2); 2, 3 and 5 are identical | 8 | ✅ |
| R3 | Full stamp | `{dtype: bfloat16, gradient_checkpointing: false, attn_implementation: sdpa, torch_compile: false, chunked_ce: 0}`: defaults included, where the old stamp held only `dtype` | 0 | ✅ |
| R4 | Re-measure: compile on | stale; from 2: 4 fits, 8 OOMs → **4** (matches 2 → 4) | 2 | ✅ |
| R5 | Re-measure: + checkpointing | from 4: 8 fits, 16 OOMs → **8**. Peak 17.385 GB vs 17.386 on record | 2 | ✅ |
| R6 | Re-measure: eager attention | from 2: 4 OOMs, 2 OOMs, 1 fits → **1** | 3 | ✅ |
| R7 | Non-memory change (micro-batch, padding, length grouping, workers) | "ceiling still valid", nothing run | 0 | ✅ |
| R8 | Nothing fits (seq 16384) | micro-batch 1 OOMs → `fits: false`; through Frame, the memory-regime note is shown | 1 | ✅ |
| R9 | Capped (SmolLM2-360M, seq 128) | every divisor fits → **16+**, `capped: true` | 5 | ✅ |
| R10 | A crash that isn't an OOM (`flash_attention_2`, library absent) | search stops: "failed with an error, not an OOM"; the Profile is untouched | 1 | ✅ |

### User scripts (SmolLM2-360M, HF Trainer unless stated, seq 512, `effective_batch` 32)

Frame wrote `ceiling_rows.jsonl`: 96 rows, each 654-690 tokens with the model's tokenizer, so the
scripts' truncation cut every one to 512.

| # | Case | Result | Trials | Pass |
|---|---|---|---|---|
| S1 | Data source is a flag (`trainer_flags.py --data_path`) | **16**; 32 OOMs. Every trial that ran saw exactly `[N, 512]` | 6 | ✅ |
| S2 | A config key (`trainer_yaml.py`, `data.yaml` `path`), per-trial `ceiling_mbN.yaml` copies, `sync = false` | **16**; the per-trial copies were new files and still reached the box; the user's `train.yaml` / `data.yaml` checksums unchanged | 6 | ✅ |
| S3 | A hardcoded path (`trainer_hardcoded_data.py`); the user says "move it to a config" | copy `runs/ceil_s3/train.py` (12-line diff) reads `path` from a config, same default. Equivalence over 5 logged steps: steps 1-4 identical, step 5 3.061 vs 3.062; the original re-run gave 3.060, so that's run-to-run noise (max relative diff 3.3e-4, under the 1e-3 rule). Then **16** through the copy; the user's script untouched | 6 | ✅ |
| S4 | A hub id only (`trainer_hub_data.py`); the user declines the edit | no ceiling, no trials; said at Step 3, micro-batch kept at the baseline's unless the user OKs more | 0 | ✅ (procedure) |
| S5 | A script that ignores its `--data_path` (`trainer_ignores_data.py`) | trial fails: "script fed [1, 8], not [1, 512]" | 1 | ✅ |
| S6 | The script truncates at 1024, `run.toml` says 512 | trial fails: "script fed [1, 674], not [1, 512]". Re-Framed at 1024 → **8** (micro-batch 8 at 1024 peaks at 16.656 GB, the same as 16 at 512: the same tokens) | 1 + 5 | ✅ |
| S7 | A plain PyTorch loop (`torch_flags.py --data-path`) | **16** | 6 | ✅ |
| S8 | Pre-tokenized: the user's own `prep.py` run on the rows, then `trainer_pretok.py` | **16** | 6 | ✅ |
| S9 | Re-measure: compile on (via the config template) | from 16: 32 OOMs at a 20.4 GB peak (the same 20.4 GB on record), 16 fits → stays **16** | 2 | ✅ |
| S10 | Re-measure: checkpointing on | from 16: 32 fits → **32**, capped | 1 | ✅ |

### End to end (short supervised loops)

**E1, our runner**, `runs/ceil_e1`: the ceiling is re-checked before every probe, and re-measured
only when a memory knob moves.

| Exp | Change | Ceiling check | micro-batch | samples/s | peak |
|---|---|---|---|---|---|
| 1 | baseline (shipped defaults) | Frame: 2 | 1 | 4.46 | 10.3 GB |
| 2 | compile | stale → 2 → 4 (2 trials) | 4 | 6.27 | 18.7 GB |
| 3 | padding longest + length grouping | still valid (0 trials) | 4 | 68.66 | 6.6 GB |
| 4 | + checkpointing | stale → 4 → 8 (2 trials) | 8 | 71.26 | 6.4 GB |

Exp 3 is the case the ceiling exists for: under dynamic padding the probe peaked at 6.6 GB, which
looks like room for far more, but the bound at full `max_len` is still 4.

**E2, a user script** (`trainer_yaml.py`), `runs/ceil_e2`:

| Exp | Change | Ceiling check | samples/s | peak |
|---|---|---|---|---|
| 1 | baseline (micro 8 × accum 4) | 16 (S2's) | 26.77 | 10.1 GB |
| 2 | micro 16 × accum 2, no question asked (within the ceiling) | still valid (0 trials) | 27.59 | 17.4 GB |
| 3 | + compile | stale → stays 16 (2 trials) | 35.06 | 12.8 GB |

Micro-batch 32 was never proposed. This reproduces `my_byo_yaml`: compile +27% here, +31% there,
and the same 17.36 → 12.75 GB peak drop. In both loops the trials went to `ceiling.tsv` and only
probes to `results.tsv`.

## Bugs the A10 found, fixed in this PR (with tests)

1. **HF Trainer calls `model(**batch)`.** The one-shot global hook that finds the model can't see
   kwargs, so the first call read as "no token tensor" and every trial failed. Now that call is
   checked only when its input is positional; every later call goes through the model's own hook
   (`test_a_model_called_with_keyword_inputs_is_checked`).
2. **An OOM in step 1's forward** happened before any shape was readable, and was turned into an
   `error`. It now counts as "doesn't fit", marked `shape_checked: false`: an unchecked OOM can only
   make the ceiling too low, and the smaller trials of the same search already checked the rows
   reach the model (`test_an_oom_in_the_first_kwargs_forward_still_counts_as_does_not_fit`).
3. **`--fresh` didn't force a measurement** when the existing ceiling wasn't stale. It now implies
   `--force`.

## Deviations, recorded rather than smoothed over

- **S5, first attempt:** the script's fixed data file had been written only on the box, and the
  next probe's `rsync --delete` removed it, so the trial failed with `FileNotFoundError` (a harness
  error). Copied back to the laptop's checkout (gitignored) so the sync keeps it, then re-run.
- **E1, first attempt:** the driver's probe configs left out `dataset`, so the four probes failed
  before running (the ceiling re-measures between them ran correctly). Those rows are in
  `runs/ceil_e1/discarded/`; the Profile was reset to Frame's ceiling and E1 re-run.
- **S2, first attempt:** run on a copy of S1's Profile, whose ceiling had the same stamp, so the
  command correctly said "still valid" and ran nothing (that is bug 3's context). Re-run fresh.
- **R6 took 3 trials,** not the 1-2 the matrix estimated: a drop of one step is "next up, previous,
  then down" by design.

## Limits of this verification

- **One card type** (A10), two small models, one framework family (HF Trainer) plus one plain
  PyTorch loop. Multi-GPU is out of scope.
- **The agent was this session, following `program.md` by hand.** The user's answers at S3 / S4
  were the matrix's.
- **E1 and E2 are 3-4 probes each:** enough to show re-measures land before memory-knob probes and
  the bound is respected, not a tuning session.
- **Fragmentation over a long run** isn't reproduced by 3-step fixed-shape trials
  (`design_improvements.md` §5); `card_frac` still drives the margin note.
