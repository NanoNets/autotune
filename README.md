# autotune



The idea: give an AI agent your training job and a GPU, and let it find the fastest configuration
overnight. It profiles a real training step, works out where the time goes, changes one setting,
measures again on the GPU, keeps or discards, and repeats. You wake up to a log of experiments and
a faster config. The run above took Qwen2.5-0.5B on an A10 from 4.3 to 108.6 samples/s.

The agent only touches settings that change **speed** (micro-batch, attention kernel, compile,
gradient checkpointing, padding, dataloader). Learning rate, precision, effective batch and sequence
length are off-limits, so a faster config is never a worse model.

## How it works

1. **Frame:** size up the job once: the model, the data, the GPU's real peak, and the largest
   micro-batch that fits.
2. **Baseline:** run your config as-is.
3. **Loop:** pick one change, test it on the GPU, keep it if it's faster, repeat.
4. **Learn:** record the run and update the shared beliefs for next time.

## Quick start

**Requirements:** a single NVIDIA GPU (local, over SSH, or on Modal), Python 3.11+,
[uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/NanoNets/autotune && cd autotune
uv sync
uv run pytest
```



## Running the agent

Open the repo in [Claude Code](https://claude.com/claude-code). The bundled skill, [`.claude/skills/run-loop/SKILL.md`](.claude/skills/run-loop/SKILL.md), takes the whole run in one command:

```
/run-loop <run_name>
    model=<hf_id>                    # e.g. Qwen/Qwen2.5-0.5B
    dataset=<hf_id>[:<config>]       # e.g. Salesforce/wikitext:wikitext-2-raw-v1
    max_len=<N>                      # sequence length, never changed
    effective_batch=<N>              # batch size, never changed
    host=<user>@<ip> | local         # where experiments run
    gpu=<label>                      # e.g. A10
    [script=<path>]                  # optional: your own training script
    [script_configs=<path>,...]      # optional: the config files it loads
    [resume | review | smoke]        # optional: continue / write up / CPU dry run
```


| Argument                      | Meaning                                                                                      |
| ----------------------------- | -------------------------------------------------------------------------------------------- |
| `model`, `dataset`            | Hugging Face ids (`:config` picks a dataset subset)                                          |
| `max_len`, `effective_batch`  | your training's sequence length and batch size; never changed                                |
| `host` / `local`              | where experiments run: a GPU box over SSH, or this machine                                   |
| `gpu`                         | a label for the record (the peak is measured on the card)                                    |
| `script`, `script_configs`    | tune your own training script (and the config files it loads) instead of the built-in runner |
| `resume` / `review` / `smoke` | continue a stopped run / write up a finished one / a CPU dry run                             |


For example:

```
/run-loop my_run model=Qwen/Qwen2.5-0.5B dataset=Salesforce/wikitext:wikitext-2-raw-v1 \
          max_len=2048 effective_batch=16 host=ubuntu@<gpu-box> gpu=A10
```

It sizes up the job, shows you a summary to confirm, then runs unattended until you stop it.
Results land in `runs/my_run/`. Your own script (HF Trainer, TRL, a plain PyTorch loop) runs
unchanged; see [`examples/byo_script/`](examples/byo_script/).

No Claude Code? Fill in `run.toml` and point any coding agent at `program.md`.

## Our contributions

- **Measured, not assumed.** The card's peak FLOP/s and bandwidth are benchmarked on the device,
so MFU is right on any GPU. Every comparison uses the same data rows and the same instrument, and
impossible readings are thrown out.
- **A memory ceiling that's measured.** Real batches rarely hit the maximum length, so a config can
pass every test and crash hours into training. autotune runs worst-case batches up front, and
re-checks whenever a memory setting changes.
- **A memory that compounds.**

```
    run N                          shared memory                        run N+1
 ┌─────────┐  writes a record   ┌──────────────────┐   run before?    ┌─────────┐
 │  frame  │ ─────────────────▶ │ experience/      │ ───────────────▶ │  frame  │
 │  probe  │                    │ indexed record   │  replay a match  │  probe  │
 │  learn  │                    │ of every run     │                  │  learn  │
 │         │  confirms, splits  ├──────────────────┤ priors for each  │         │
 │         │  or deletes claims │ beliefs/         │    hypothesis    │         │
 │         │ ─────────────────▶ │ evolving claims  │ ───────────────▶ │         │
 └─────────┘                    └──────────────────┘                  └─────────┘
```



## Project structure

```
program.md     agent instructions
frame/         sizes up the job, measures the memory ceiling
profiler/      measures each experiment
backends/      where experiments run: local, ssh, modal
frameworks/    the built-in Hugging Face runner
beliefs/       shared beliefs from past runs
experience/    one record per past run
examples/      example training scripts to tune
loop/          design notes and data contracts
```



## Platform support

Single NVIDIA GPU for now. Multi-GPU and a learned model that ranks candidates are next.