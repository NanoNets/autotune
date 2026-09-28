# Bring your own training script

> **Developer notes. Not part of the agent's prompt hierarchy.**
> `program.md` does not route here and the loop never reads or writes this file. It is where we
> work out the architecture for running the loop on a user's own training script. Nothing here is
> built yet, and every claim is marked by whether it has been verified.

>
> **`sandbox/attach_parity/` is a local, untracked harness** (gitignored). The experiments below ran
> there; paths into it describe where the code and raw results live on the machine that ran them.
---

## 1. The problem

**Users want to run the auto-research loop on their own training script** (an ms-swift `swift sft`
command, a TRL/axolotl/LLaMA-Factory config, a hand-written PyTorch loop) instead of on our
`frameworks/hf.py` adapter. How do we make that work?

Today the loop can only tune what it builds itself. A **runner** (`frameworks/README.md`) is the
contract between the loop and a training run:

| Member | What the loop gets from it |
|---|---|
| `setup(config)` | autotune **builds** the model, optimizer and data from a config. This is how knobs take effect |
| `step()` | autotune **calls** exactly one optimizer step, with phases wrapped in `record_function` |
| `tokens_per_step()`, `.params` | the D and N of the roofline (FLOPs, MFU) |

A user's script breaks all three:

1. **Autotune doesn't own the loop.** `capture()` (`profiler/capture.py:246-273`) drives the run:
   warmup steps, then N clean timed steps, then one profiled step. `swift sft` runs
   `Trainer.train()`, and the loop is hidden inside it. There is no `step()` to call.
2. **Autotune doesn't build the model.** Knobs are applied in `setup(config)`. For a user's script,
   the knobs are the script's own flags (`--per_device_train_batch_size`, `--gradient_checkpointing`,
   `--attn_impl`, `--torch_compile`, …). Their names, which ones exist, and how they interact are
   different in every framework.
3. **Nobody reports the roofline inputs.** Parameter count and tokens per step have to be recovered
   from the run.

This splits into two sub-problems that can be solved separately:

- **Measurement:** getting an Observation out of a run autotune doesn't control. That's §2.
- **Actuation:** turning autotune knobs into a framework's flags and launching a short probe run.
  Working stance: the user points at the YAML their script reads, or the agent writes one. The
  agent already knows each framework's names for these settings, so we don't build a name-mapping
  layer. Not designed further yet.

The rest of the loop is framework-independent and stays as it is: the search, beliefs,
`diagnostic.py`, and Frame's analytical memory model (it reads the model config, not the script).

**Constraint:** whatever we build must not change what the user's script learns. The same rule as
`loop/throughput.md`: speed levers only.

---

## 2. How portable is the profiler?

### What "the profiler" depends on

"The profiler is a torch library byproduct" is only half true. The Observation's 41 fields come
from four kinds of source, and they aren't equally portable:

| Tier | Source | Fields | Needs from the run |
|---|---|---|---|
| **A. Device level** | `torch.profiler` + CUPTI inside the profile window; `torch.cuda` memory stats; psutil process tree | kernels, `gpu_busy_s`, gaps, occupancy, `provenance`, `device`, memory, CPU, `trace_path` (~30 fields) | **Only that we run inside the training process.** CUPTI records every kernel the process launches, whoever launched it |
| **B. Step boundary** | `capture()` calling `runner.step()` | `step_time_s`, `step_time_cv`, `profiled_step_s`, and where the profile window starts and ends | a signal for "one optimizer step just finished" |
| **C. Phase labels** | `record_function("dataloader"/"forward"/"backward"/"optimizer")`; `diagnostic.py:19` hardcodes the names | `phase_times_s`, `gpu_phase_times_s`, `gpu_phase_coverage`, `data_wait_s/frac` | annotations with the names we expect |
| **D. Roofline inputs** | `.params`, `tokens_per_step()` | `flops`, `mfu`, `roofline_floor_s` | parameter count; non-pad tokens per optimizer step |

`peak_flops` and `peak_mem_bw_gb_s` are **not** tied to the run. They're measured per device and
then frozen. They should be measured in a separate process before the user's run, because once
the model is loaded the GPU memory is taken.

**Tier A works for any in-process PyTorch run as it is.** The portability question is really about
B, C and D.

### Recovering B, C, D without touching the user's script

Torch has process-global hooks that fire whatever framework is driving the loop. They can be
installed from a `sitecustomize.py` on `PYTHONPATH`, which is also inherited by the processes that
torchrun/accelerate spawn.

"Source-checked" means read in the installed library source (torch 2.13.0, transformers 5.15.1),
not run on a GPU.

| Need | Mechanism | Fidelity vs `hf.py` | Status |
|---|---|---|---|
| B: step boundary | `torch.optim.optimizer.register_optimizer_step_post_hook`. One optimizer step = one step, which handles grad accumulation correctly | Window also includes the Trainer's per-step extras (grad clip, LR scheduler, logging). Probably more realistic, but won't match `hf.py` exactly | **source-checked**: the global post-hook runs inside torch's `profile_hook_step` wrapper (`torch/optim/optimizer.py:508-541`). See the step-boundary caveats below |
| C: `optimizer` | torch emits `Optimizer.step#<Cls>.step` by itself, from the same wrapper | same | **verified**: already in our traces (`profiler/README.md` example) |
| C: `dataloader` | torch emits `enumerate(DataLoader)#<IterCls>.__next__` on every `__next__` (`torch/utils/data/dataloader.py:697,721`) | same, once `diagnostic.py` accepts it as an alias. **HF Trainer shape differs:** it pre-fetches the whole grad-accum window of batches up front (`get_batch_samples`, `trainer.py:2124`), so data-wait is one lump at the start of the step, not spread between micro-batches | **source-checked** (torch). Not checked under accelerate's `DataLoaderShard` wrapper |
| C: `backward` | wrap `torch.Tensor.backward` / `torch.autograd.backward` in `record_function` | same semantics as `hf.py`. **Not** module backward hooks: those are fragile with in-place ops and reentrant checkpointing, and backward runs on autograd's thread | **source-checked** for HF: `Trainer.training_step` → `accelerator.backward(loss)` (`trainer.py:1961`). `Tensor.backward` is a Python method that calls `torch.autograd.backward` (`torch/_tensor.py:566`), so it can be patched |
| C: `forward` | global `register_module_forward_pre_hook` + forward hook, with a depth counter so only the outermost module opens the label | same, except under `torch.compile` (below) | **source-checked**. But see the overhead rule below |
| D: `params` | outermost module seen by the forward hook, first call | exact | no |
| D: tokens | ~~`attention_mask.sum()` in the global forward pre-hook~~ **Doesn't work:** global forward pre-hooks receive **positional args only** (`torch/nn/modules/module.py:217-235`), and HF models are called as `model(**inputs)`. Instead: on the first global call, find the outermost module and register a per-module `register_forward_pre_hook(..., with_kwargs=True)` on it. Or count from the batch the dataloader returns | measured, not assumed. Matches `hf.py`'s rule | **source-checked** (the constraint) |

HF Trainer also computes its own `floating_point_ops(inputs)` (6ND, `trainer.py:3928`) and
optionally counts input tokens (`_track_num_input_tokens`, `trainer.py:2502`). Those are a
useful cross-check for Tier D, but only inside the HF family.

**Step-boundary caveats** (from reading `optimizer.py`):

- **The hook only exists if `Optimizer.__init__` ran.** torch wraps `step` in
  `_patch_step_function()` from `__init__` (`optimizer.py:387`). Wrappers that subclass `Optimizer`
  without calling `super().__init__` (I believe accelerate's `AcceleratedOptimizer` is one; not
  checked, it isn't installed) don't fire the hook themselves. The inner `AdamW` does. So the
  step boundary comes from whichever **real** torch optimizer's `step()` runs.
- **More than one optimizer means more than one firing per step.** Setups that combine Muon and
  AdamW, or keep separate optimizers per parameter group, fire the hook N times per step. It also
  fires twice if both a wrapper and its inner optimizer are hooked. Fix: count only one designated
  optimizer (the first one seen), or de-duplicate within a step.
- **Skipped steps are invisible.** Under fp16 `GradScaler`, an overflow step skips the inner
  `optimizer.step()`, so the hook never fires and two steps merge into one window. bf16 has no
  scaler, so this doesn't happen there. Trainer knows about skipped steps
  (`accelerator.optimizer_step_was_skipped`, `trainer.py:1791`); a pure torch hook doesn't.
- **Where eval/save lands:** Trainer runs `_maybe_log_save_evaluate` right after `on_step_end`
  (`trainer.py:1800`). With a post-hook-to-post-hook window, that work lands in the *next* step's
  window.

**Two rules the design has to follow:**

- **Hooks must be *removed*, not just made no-ops, during the clean window.** `Module.__call__`
  has a fast path that skips all hook handling when no hooks are registered. **Any** global module
  hook turns that fast path off for **every** module in the model (`module.py:1788,1805`). That
  overhead lands on every module call, and `step_time_s` would measure our instrumentation. Only
  the optimizer-step hook (one call per step) and the sync run during clean steps. The module
  hooks are registered just before the profiled step and removed right after.
- **The profiled step must not land on an eval or checkpoint step.** Those are large outliers.
  The median of the clean steps can absorb one; the single profiled step can't.

### Portability by framework

Confidence: **high** means it follows directly from how the framework is built. **Unverified**
means it's a reasonable expectation that hasn't been checked.

| Framework | Tier A | B (optimizer hook) | C (labels) | Notes |
|---|---|---|---|---|
| Hand-written PyTorch loop | ✅ | ✅ high | ✅ | the base case. Global hooks don't care who wrote the loop |
| HF `Trainer` (and ms-swift, TRL, LLaMA-Factory, axolotl, all built on it) | ✅ | ✅ high | ✅ | also offers `TrainerCallback` (`on_step_begin/end`, `control.should_training_stop`) as a cleaner option for this family only |
| HF Accelerate custom loop | ✅ | ✅ high | ✅ | `accelerator.backward` ends up calling `Tensor.backward` |
| Lightning | ✅ | ✅ high | ✅ | wraps `torch.optim` optimizers |
| Unsloth | ✅ | ✅ | ⚠️ | HF Trainer underneath, but heavy kernel patching and compile. See the `torch.compile` row |
| `torch.compile`'d model (any framework) | ✅ | ✅ | ⚠️ | global module hooks can cause graph breaks or recompiles, which changes the thing we measure. Fallback: step boundary plus torch's own labels, no `forward`/`backward` split |
| DeepSpeed (ZeRO) | ✅ | ⚠️ unverified | ✅ | engine wraps the optimizer. Expected to reach the inner `torch.optim` `step()`, not checked |
| FSDP | ✅ | ✅ probably | ✅ | multi-rank: profile rank 0 only, or one file per rank |
| Megatron-LM / NeMo | ✅ | ⚠️ unverified | ⚠️ | own optimizer wrapper and pipeline schedule. The step boundary may need Megatron's own iteration hook |
| JAX / non-torch | ❌ | ❌ | ❌ | out of scope. Needs a different profiler entirely |

### What this means

- **Kernel-level insight carries over completely** to any in-process PyTorch run. The hooks don't
  collect the data; they only decide when the profile window opens and add labels.
- The portable layer is **torch, not HF Trainer.** Building on global torch hooks covers the whole
  HF family and more. The `TrainerCallback` is an optional nicety.
- The real risks are **`torch.compile`** (phase labels), **wrapped optimizers** (DeepSpeed,
  Megatron step boundary), and **multi-rank** runs.

### Experiment that would settle §2

Run one config two ways on a GPU: through today's `hf.py` runner, and as a plain HF `Trainer`
script with the global hooks attached. Compare the Observations:

| Field group | Expected |
|---|---|
| Tier A (kernels, `gpu_busy_s`, occupancy, memory) | equal within noise |
| `step_time_s` | slightly higher under Trainer (per-step extras) |
| `gpu_phase_times_s` fwd/bwd/opt split | roughly equal |
| `flops`, `mfu` | equal if tokens/params recovery is right |

If it passes, repeat on `swift sft`, then with `torch_compile` on, then under DeepSpeed.

### Experiment 1: raw PyTorch loop, hooks vs `record_function` (2026-09-24)

**Setup.** The sandbox is `sandbox/attach_parity/`. It is fully isolated: the baseline is a
frozen copy of the profiler at `bdc33d6`, and nothing imports the live `profiler/`.
- **Arm A:** `capture()` drives `frameworks/torch.py`, which has `record_function` labels.
- **Arm C:** the same runner with every `record_function` removed. Its own loop runs freely, and
  `attach.py` hooks find the step boundaries and add the labels.
- Arms alternate A C A C, each in a fresh process, 5 rounds. The ceilings are measured once and
  shared by both arms.
- Hardware: 1× A10, torch 2.7.0, fp32, 8×2048 MLP, micro-batch 32, seq 512, with both float input
  and `data="synthetic"` (embedding + token ids).
- **Tolerance:** the noise floor, meaning the median `step_time_cv` across both arms (1.35–1.44%).

**Result: no information loss in what the trace records, and every headline number is within the
noise floor.**

| Check | float input | synthetic ids |
|---|---|---|
| CPU ops recorded (events / names) | 584 / 50 in both, zero count mismatches | 639 / 60, zero |
| GPU kernels recorded (events / names) | 71 / 19 in both, zero count mismatches | 128 / 33, zero |
| Observation fields populated in A but empty in C | none | none |
| params / tokens recovered by hooks | exact (33,570,816 / 16,384) | exact (99,106,816 / 16,384) |
| `step_time_s` Δ | +0.7% | +0.2% |
| `gpu_busy_s` Δ | −1.0% | −0.3% |
| `mfu` Δ | −0.7% | −0.2% |
| GPU `forward` / `backward` / `Optimizer.step` Δ | −1.7% / −0.2% / −0.1% | −1.3% / −0.4% / +0.1% |

**What differs, and why:**
- **Two labels have no hook equivalent.** A's `optimizer` label wraps `step` + `zero_grad`; C only
  has torch's own `Optimizer.step#AdamW.step` and `zero_grad`, which A also has nested inside its
  label. Compared like-for-like, GPU time matches to 0.1%. A's `dataloader` label has nothing to
  map to in C, because `torch.py` reads a resident tensor rather than a `DataLoader`. The hook
  mechanism can only pick up torch's `enumerate(DataLoader)` label, which appears only when a real
  `DataLoader` is used. That path is still untested.
- **The loss computation is not labeled.** In C, `forward` covers only module calls, so the
  `.square().mean()` loss ops, which run outside any module, fall inside `step` but in no phase:
  0.84 ms, about 0.3% of GPU time. That accounts for the GPU `forward` −1.7% flag in the float run
  (tolerance 1.44%, which the synthetic run passes). In HF models the loss is usually computed
  inside `model.forward` when `labels` are passed. A loss computed outside the model, such as a
  custom `compute_loss` or chunked CE, would be unlabeled the same way.
- **Host-dispatch fields fail the tolerance, but they're noisy in both arms.** Examples: host
  `Optimizer.step` +2–5% and host `forward`/`backward` −1 to −4%. These are single-sample, sub-ms
  host timings, and their within-arm spread across rounds (2–15%) is larger than the A-vs-C
  difference. The step-time noise floor is the wrong tolerance for them. They agree within their
  own run-to-run noise.
- **A bug was found and fixed during the run.** The first version called `cudaDeviceSynchronize`
  before closing the `step` annotation, so host `step` read 249 ms instead of about 5 ms. It is now
  closed before the sync, as in `capture()`.
- **Not new:** synthetic MFU reads 1.94 in **both** arms. 6ND counts the 65M embedding parameters
  as matmul FLOPs, which the embedding lookup doesn't actually do. This is a roofline issue shared
  by both arms, not a hook issue.

**Not covered yet:** a real `DataLoader`, grad accumulation, HF Trainer / accelerate, `torch.compile`,
bf16/AMP, and multiple optimizers. CPU runs on the Mac passed the information check, but its timing
noise (up to 39% spread between rounds) is too large to judge timing.

### Experiment 2: real `DataLoader` + gradient accumulation (2026-09-24)

**Setup.** Same sandbox, rig, tolerance and alternation as experiment 1.
- New runner pair in `sandbox/attach_parity/loader_runner.py`: batches come from a torch
  `DataLoader` (pinned, shuffled), and one step = `grad_accum=4` micro-batches of fwd/bwd, then one
  `AdamW` step.
- Model: 8×2048 MLP, micro-batch 8, seq 512, so 16,384 tokens per optimizer step, the same as
  experiment 1.
- A busy-wait in `collate_fn` stands in for CPU preprocessing. Three configs:

| Config | Workers | Collate cost / batch | What it exercises |
|---|---|---|---|
| a | 0 | 0 | grad accum alone; data-wait is only the batch stack (~7%) |
| b | 0 | 20 ms | **data-bound**: host data-wait ~35% of the step |
| c | 2 | 20 ms | workers hide the same cost: data-wait ~0.1% |

**First run found a real information loss.** The trace was fine (torch's
`enumerate(DataLoader)` label matched A's `dataloader` label). But C's **Observation** reported
`data_wait_s = 0.0` and "data-wait 0%" in config b, where the truth was 36%. `diagnostic.py` reads
only a label literally named `dataloader`. The "fields populated" check missed it because `0.0`
counts as populated.
- **Fix, in the sandbox:** the hooks now also emit a `dataloader` label around
  `_BaseDataLoaderIter.__next__`, for the profiled step only. The diagnostic is unchanged.
- **New check:** the harness now also flags any Observation number that is > 0 in A but exactly
  0 in C.

**Result after the fix: no information loss, and every headline number is within the noise
floor (0.83–1.07%).**

| | a | b (data-bound) | c (workers) |
|---|---|---|---|
| CPU ops / kernels recorded | identical (1810 / 264 events) | identical | identical (1702 / 264) |
| Observation numbers wrongly 0 in C | none | none | none |
| forwards / backwards per step seen by hooks | 4 / 4 (grad_accum = 4) | 4 / 4 | 4 / 4 |
| tokens / params recovered | exact | exact | exact |
| `step_time_s` Δ | −0.1% | +0.1% | +0.1% |
| `gpu_busy_s` Δ | +0.4% | −0.2% | −0.1% |
| `mfu` Δ | +0.1% | −0.1% | −0.1% |
| `data_wait_s` A → C | 17.9 → 17.9 ms (+0.3%) | 100.5 → 99.4 ms (−1.2%) | 0.40 → 0.33 ms |

**What the remaining flags are:**
- **`data_wait_s` −1.2% in b is run-to-run variance, not the hooks.** Across rounds it swings 48%
  within each arm, and torch's own `enumerate(DataLoader)` label, identical code in both arms,
  differs by the same −1.1%.
- **In c, data-wait is ~0.1% of the step,** so a percentage difference is meaningless. A's
  0.07 ms extra is its own `record_function` overhead, 4 calls per step.
- **Unlabeled GPU time.** A has 5.3 ms of GPU time in no phase: the pinned host→device copies
  (`x.to(device)` sits outside every label in both arms). C has 0.88 ms more: the loss ops, as in
  experiment 1, now × 4 micro-batches. This is the same label-scope difference as before. Nothing
  new is lost.
- GPU `forward` −1.1 to −1.2% is again the loss ops excluded from C's `forward`.

**What this settles:**
- Grad accumulation needs no special support. The optimizer-step boundary contains all N
  micro-batches automatically, and the count as it actually ran is observable from the hooks.
- Data-wait is observable at the torch level, including the difference between a data-bound
  loader and one whose workers hide the cost. This holds only for a torch `DataLoader`.

**Open, with the fix still in the sandbox only:**
- **Decide where the `dataloader` name should come from:** the hooks emitting it (current fix),
  or `diagnostic.py` accepting torch's `enumerate(DataLoader)` label as an alias.
- **Consider labeling the H2D copy and the loss.** They are unlabeled in both arms today, so
  it is not a regression, but a user's script may have more such work.

### Experiment 3: the launcher, single GPU (2026-09-24)

**The workflow this tests:** the user points the agent at a `train.py`, and the agent runs it
through a launcher instead of running it directly:

```bash
python launcher.py --out cap.json --device cuda [--warmup 3 --clean 10 --peak-flops … --peak-bw …] \
    -- train.py <the script's own args>
```

**How the launcher works** (`sandbox/attach_parity/launcher.py`):
1. It installs Attach's hooks.
2. It sets `sys.argv` to the script's own arguments and puts the script's directory on `sys.path`.
3. It runs the script in the same process with `runpy.run_path(script, run_name="__main__")`, so
   the script behaves exactly as if started with `python train.py`.
4. The Capture is written to `--out` (atomically) the moment the profiled step ends. The script
   then runs to its own end, so a probe gives it a small step budget.
5. If the script ends before the profiled step, `--out` gets `status: "error"` with the step count
   it reached.

Multi-GPU is out of scope for now.

**Setup.**
- **Arm L:** `sandbox/attach_parity/train.py` is experiment 2's workload written as an ordinary
  user script: top-level argparse, an inline loop, no autotune imports, no annotations. It runs
  in its own process under the launcher, and the result is read back from the JSON.
- It is compared with A and C on experiment 2's three configs, 5 rounds each, alternating A C L.

**Result: L matches the baseline as closely as C does, so the launcher adds no information loss
and no measurable overhead.**

| A vs L | a | b (data-bound) | c (workers) |
|---|---|---|---|
| CPU ops / kernels recorded | identical (1810 / 264) | identical | identical (1702 / 264) |
| Observation numbers wrongly 0 | none | none | none |
| forwards / backwards per step, tokens, params | 4 / 4, exact, exact | same | same |
| `step_time_s` Δ | +0.3% | +0.4% | +0.1% |
| `gpu_busy_s` Δ | +0.3% | +0.0% | −0.4% |
| `mfu` Δ | −0.3% | −0.4% | −0.1% |
| `data_wait_s` Δ | −0.1% | −0.2% | −7.2% of 0.36 ms (≈0.1% of the step) |

The flags are the same ones as in experiment 2, with the same causes:
- GPU `forward` −0.9 to −1.1%, and +0.9 ms of unlabeled GPU time: the loss ops sit outside the
  hooks' `forward` label.
- Data-wait in c is too small for a percentage to mean anything.

**Still open for the launcher:**
- **Installed commands** like `swift sft`, which are not a `.py` file. The launcher would need to
  run the module behind the command.
- **Probe length.** Once the Capture is written, the script keeps running until its own end.
  Should the launcher stop it early, or leave that to `max_steps` in the probe's YAML?
- **Crash behavior.** A script that calls `os._exit`, or a process that gets killed (for
  example by the OOM killer), skips the launcher's `finally` and leaves no `--out` file. The loop
  should treat a missing file as an error.
- **Update 2026-09-25: failures are classified,** using the statuses `capture()` uses. The
  exception is re-raised after `--out` is written, so the exit code and the traceback are
  unchanged.
  - A CUDA OOM → `oom`, with the clean steps that ran and memory at failure.
  - An OOM after the profiled step → `oom` (it fit for one step, not for the run).
  - Any other exception → `error: <Type>: <first line>`. After the profiled step, the
    measurement is kept and a `script_error` is added.
  - `sys.exit(0)` is a normal end.

  Tested: `tests/test_launcher.py` (8 CPU cases), plus a real OOM on the A10 (SmolLM2-360M at
  micro-batch 64 → `oom`, 21.65 / 22.06 GB).

### Experiment 4: an HF Trainer script through the launcher, single GPU (2026-09-24)

**Exact versions** (`sandbox/attach_parity/hf/requirements-hf.txt`; `hf/versions.py` checks all
of them at the start of every arm, and a mismatch fails the run):

| Package | Version | Why this version |
|---|---|---|
| Python | 3.10.12 | the box's |
| torch | 2.7.0, CUDA 12.8 | the box's **system** package, not pip-installed. transformers 5.15.1 requires ≥2.5 |
| transformers | 5.15.1 | the repo's `uv.lock`, and the version the §2 source checks were read against |
| huggingface-hub / tokenizers / safetensors | 1.28.0 / 0.22.2 / 0.8.0 | the repo's `uv.lock` |
| accelerate | 1.14.0 | not in `uv.lock`. The release current when transformers 5.15.1 shipped (2026-08-19; 1.15.0 came 2026-09-09). Satisfies transformers' `accelerate>=1.1.0` |
| numpy | 1.24.4 | three constraints, all hit on the box (below) |
| scipy | 1.8.0 | the box's system package, imported by Trainer. Checked, not installed |

**numpy took three tries, and each failure is a real compatibility fact:**
- **1.21.5** (the box's system numpy): importing `transformers.Trainer` fails with
  `TypeError: 'type' object is not subscriptable` (`data_collator.py:974` subscripts
  `np.ndarray[...]`). transformers 5.15.1 declares `numpy>=1.17`, but it actually needs ≥1.22.
- **2.2.6** (last numpy for Python 3.10; `uv.lock`'s 2.4.6 / 2.5.2 need ≥3.11): the system torch
  and scipy are built against the numpy 1.x ABI. torch's numpy bridge fails
  (`_ARRAY_API not found`), and scipy fails (`numpy.dtype size changed`).
- **1.26.4:** works, but system scipy 1.8.0 warns that it supports numpy `>=1.17.3,<1.25.0`.
- **1.24.4:** satisfies all three constraints, with a clean import (`-W error::UserWarning`).

**Accelerate 1.14.0 source checks** (the installed files on the box):
- `AcceleratedOptimizer` subclasses `torch.optim.Optimizer` but never calls `super().__init__()`
  (`optimizer.py:55`), so torch never hooks the wrapper's `step`. Without a grad scaler (bf16
  has none) it calls the inner `AdamW.step` (`optimizer.py:179`), so the boundary fires once.
- `DataLoaderShard.__iter__` drives the base torch `DataLoader`'s iterator (`data_loader.py:582`),
  so the `__next__` hook fires. `send_to_device` runs outside that call.
- `Accelerator.backward` → `loss.backward()` without a scaler (`accelerator.py:2850`).
- `model.forward` is replaced by an autocast + `convert_outputs_to_fp32` wrapper
  (`accelerator.py:1824`). Module hooks still see the call.

**Setup** (`sandbox/attach_parity/hf/`):
- **Workload** (`workload.py`, shared by every arm): a Llama built from config, 168.3M params
  (hidden 1024, 8 layers, 16 heads, vocab 32000), random init, no download. Random token ids, seq
  512, micro-batch 8, `grad_accum` 4, bf16, SDPA.
- `train_hf.py` is an ordinary Trainer script: `TrainingArguments(max_steps=16, bf16=True,
  optim="adamw_torch", logging/save/eval off, report_to="none")`, then `trainer.train()`.
- **L:** `launcher.py -- train_hf.py`, hooks on.
- **T:** the same script with no hooks, timed by a `TrainerCallback` (CUDA sync + clock at
  `on_step_end`). This is an independent measurement of the same Trainer run, used for the
  pass/fail timing check.
- **R:** our labelled raw loop on the same model and data, driven by `capture()`. It mirrors
  Trainer where that changes kernels: autocast, Trainer's decay/no-decay param groups, clip at
  1.0, pinned DataLoader. Informational only.
- 5 rounds, R L T alternating. bf16 ceilings are measured once and frozen.

**Pass/fail: all pass.**

| Check | Result |
|---|---|
| Versions exactly the pins, every arm, every round | ✅ |
| Labels in L's trace | `step` 1, `forward` 4, `backward` 4, `dataloader` 4, `Optimizer.step#AdamW.step` 1 (none missing) |
| Boundary fires once per optimizer step | ✅ 4 forwards / 4 backwards per step = `grad_accum` 4. A double firing would show 2 |
| tokens / params recovered | exact: 16,384 / 168,313,856 |
| Observation fields empty or wrongly 0 vs R | none |
| **L vs T `step_time_s`** (tolerance = noise floor 0.31%) | 0.5311 vs 0.5304 s, **+0.1%**. The hooks don't move Trainer's step time |

**Informational: what Trainer adds on top of a bare loop (L vs R).**
- **Step time is almost the same:** 0.5311 vs 0.5303 s, +0.2%. MFU 35.4% in both.
- **Gradient clipping is unlabeled under Trainer.** Trainer clips before `optimizer.step()`, so
  the clip's 4.2 ms of GPU work (`multi_tensor_apply` norm kernels, 0.8% of the step) falls in
  no phase. In R it's inside our `optimizer` label. This is the one new label gap Trainer
  introduces: the phase map under-reports "optimizer" work by the clip.
- **Trainer syncs the host every micro-batch.** Its loss NaN/Inf check (`aten::isnan` /
  `aten::isinf`, Trainer only) calls `.item()` per micro-batch: 12 device-to-host copies per step
  vs R's 4. That changes **where** the host waits: R's host runs ahead and blocks inside
  `forward` (317 ms host `forward`), while Trainer's host blocks inside `step` between phases
  (51 ms host `forward`). Host phase times are therefore not comparable across the two loops.
  GPU phase times are: `forward` GPU is equal (+0.0%).
- **Other kernel and op differences are workload differences, not measurement error.** Examples:
  Trainer's `num_items_in_batch` loss normalization (`aten::ne`, extra `DivBackward`), and
  accelerate's fp32 output conversion. They explain GPU `backward` −1.6% and `gpu_busy_s` −1.0%.
  R mirrors Trainer only approximately.
- **Data-wait is 1.2 ms (L) vs 3.4 ms (R),** both under 1% of the step. Different collate paths
  (Trainer's `default_data_collator` + `DataLoaderShard`'s one-batch lookahead vs torch's default
  collate).

**What this settles, for a single-GPU HF Trainer script launched by the agent:**
- The launcher + hooks produce a complete Observation.
- They find the step boundary through accelerate's optimizer wrapper.
- They see Trainer's `DataLoaderShard` and grad accumulation.
- They don't perturb Trainer's step time beyond the noise floor.

**Candidates for the live profiler, from this experiment:**
- Label `clip_grad_norm_`: wrap `torch.nn.utils.clip_grad_norm_` the same way `backward` is.
- Treat host phase times as loop-specific. Rank on GPU phase times.

### Experiment 4b: the `from_pretrained` pathway (2026-09-24)

**What it tests:** experiment 4 built its Llama from a config. Real scripts load a checkpoint:
`AutoModelForCausalLM.from_pretrained(name)`. Same harness, same pins, same three arms, but the
shared workload now loads a hub checkpoint.

**Checkpoints** (ungated, pinned to exact commits in `hf/workload.py`; a different revision
fails the run):

| Model | Revision | Class | Why |
|---|---|---|---|
| `HuggingFaceTB/SmolLM2-360M` | `f8027fd0eaeea54caa13c31d31b9fdc459c38b49` | `LlamaForCausalLM` | same class as experiment 4, so only the **loading pathway** changes |
| `Qwen/Qwen2.5-0.5B` | `060db6499f32faf8b98477b0a26969ef7d8b9987` | `Qwen2ForCausalLM` | a **different class**, which tests the hooks' class-agnosticism |

Both are downloaded once before the timed rounds (`snapshot_download` at the pinned revision), and
every arm then runs with `HF_HUB_OFFLINE=1`. Loading time never lands in a timed window.

**Finding: `from_pretrained` with no `dtype` trains pure bf16 in transformers 5.15.1.** With
`dtype` omitted, it defaults to `"auto"` (`modeling_utils.py:4136-4137`), which is the
checkpoint's own dtype. Both checkpoints declare `bfloat16`, and both loaded as bf16 in every arm
and round. A typical Trainer script (`bf16=True`, no `dtype`) therefore trains bf16 weights with
bf16 AdamW state, while the config-built model in experiment 4 trained fp32 weights under
autocast. That changes memory, kernels and numerics, so **the loop should treat the loaded
dtype as a run parameter it reports, not an assumption.**

**Pass/fail.**

| Check | SmolLM2-360M | Qwen2.5-0.5B |
|---|---|---|
| Versions exactly the pins, every arm, every round | ✅ | ✅ |
| Model as loaded, identical in every arm | Llama, bf16, tied embeddings, SDPA | Qwen2, bf16, tied embeddings, SDPA |
| Labels (`step`/`forward`/`backward`/`dataloader`/`Optimizer.step`) | 1 / 4 / 4 / 4 / 1 | 1 / 4 / 4 / 4 / 1 |
| forwards / backwards per step (grad_accum 4) | 4 / 4 | 4 / 4 |
| params recovered (the tied weight counted once) | exact: 361,821,120 | exact: 494,032,768 |
| tokens recovered | exact: 16,384 | exact: 16,384 |
| Observation fields empty or wrongly 0 vs R | none | none |
| **L vs T `step_time_s`** | 1.3151 vs 1.3126 s, **+0.2%**; noise floor 0.15%: **over** | 1.5545 vs 1.5518 s, **+0.2%**; noise floor 0.18%: **within** |

**The timing result, stated precisely.** Paired round by round, L (hooks) is slower than T (the
hook-free `TrainerCallback`) in 9 of 10 rounds:

| Round | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| SmolLM2 L − T | −1.4 ms | +1.5 | +2.5 | +2.1 | +2.6 |
| Qwen2.5 L − T | +1.2 ms | −0.2 | +4.0 | +2.0 | +3.1 |

So there is a small **systematic** overhead of about 2 ms per step (≈0.15–0.2%). It is below the
round-to-round spread of either arm (0.3–0.9%), which is the noise the loop actually faces when
it compares probes across processes. It is at or just above the within-run noise floor, which is
unusually tight here (0.15–0.18%) because these steps are long (1.3–1.6 s). The config-built
model in experiment 4 (0.53 s steps) showed +0.1% within a 0.31% floor. Experiment 4 likely has
the same ~1 ms overhead, just hidden inside its larger floor.

**Likely cause (a hypothesis, not yet tested): where the per-step sync happens.** Both arms sync
the GPU once per step to time it, but at different points. L syncs inside the optimizer-step
wrapper, *before* Trainer runs `lr_scheduler.step()` and `zero_grad()`. T syncs at
`on_step_end`, *after* them. L's placement makes that host work wait for the GPU instead of
overlapping with the tail of the optimizer kernels. Note that `capture()` also syncs every step,
so no arm measures a fully unperturbed run.
- **Candidate fix:** time with CUDA events recorded at each boundary (`torch.cuda.Event`), and
  sync once at the end of the clean window instead of every step. That would remove the per-step
  serialization for every arm and every framework, and would be a small change to `attach.py`.

**Informational (L vs R): what Trainer adds, now with real checkpoints.**
- **Step time:** +0.8% (SmolLM2) and +2.3% (Qwen2.5) over the raw loop. Experiment 4 was +0.2%.
- **Qwen2.5's GPU `forward` is +5.6% under Trainer; SmolLM2's is +0.0%.** The likely cause is
  accelerate's `convert_outputs_to_fp32` (`accelerator.py:1824`) casting the logits to fp32
  after every forward. The cast scales with vocabulary: 151,936 for Qwen2.5 vs 49,152 for
  SmolLM2, per micro-batch of 8 × 512 tokens. Inferred from the numbers, not traced kernel by
  kernel.
- **Same Trainer-only ops as experiment 4:** `isnan`/`isinf` (the per-micro-batch loss check),
  `ne` (the `num_items_in_batch` count), and `abs`. The raw loop has nothing Trainer lacks.

**What this settles:**
- The `from_pretrained` pathway needs nothing extra. The hooks see a hub checkpoint exactly as
  they see a config-built model.
- Tied embeddings are counted correctly.
- The hooks worked unchanged on a second model class (`Qwen2ForCausalLM`).

**What it opens:**
- The ~2 ms/step overhead, and whether event-based timing removes it.
- Treat the loaded dtype as a reported run parameter.

### Experiment 5 (in progress): patch-free capture (2026-09-25)

**Goal:** no monkeypatching, so the approach is acceptable in an open-source repo. See
`profiler_metric_coverage.md` §0 for the mechanism-by-mechanism replacement.

`attach.py` now uses only public APIs:
- the optimizer post-hook;
- `torch.profiler(record_shapes=True)` plus one `record_function("step")` of our own;
- `torch.cuda`;
- psutil.

Phases, tokens and the micro-batch count are recovered from the trace torch recorded
(`derive.py`). The three monkeypatches and all module hooks are gone, and `nopatch.py` fails any
run that replaces a torch function or leaves a global module hook registered.

**CPU results** (1 round each, Mac, timing not meaningful):
- **What was recorded:** identical CPU-op sets vs the labelled baseline on the float MLP, the
  token-id MLP, and the DataLoader + grad-accum workload (C and L arms).
- **What was recovered:**
  - 4 / 4 micro-batches;
  - tokens exact through the embedding path (`aten::embedding` indices) and the first-linear path;
  - params exact;
  - torch's `Optimizer.step` label for the profiled step still present, even though the window
    now closes inside it. Whether its duration is complete is not yet checked.
- **One gap, now out of scope:** on a runner with no DataLoader, data-wait reads 0.0 instead of
  null. That's the frozen `diagnose()`'s default. Only our in-memory smoke-test fixture
  (`frameworks/torch.py`) hits it.

**Scope decision (2026-09-25): a torch `DataLoader` is assumed.** HF Trainer, TRL, ms-swift,
axolotl, LLaMA-Factory and Lightning all build one. Scripts that fetch batches without one
(nanoGPT-style `get_batch()` over a memmap, DALI, iterating Ray Data directly) are a **future
improvement**. For them, data-wait can't be measured at the torch level; the GPU-idle signals
(`gap_frac`, `max_gap_s`) still show a stall. The one-line `diagnose()` change (null when no
DataLoader span exists) goes with that future work.

**GPU results** (1× A10 at a new host, same image; 5 rounds each; the monkeypatch guard active
on every C / L run; raw records in `results_np_*.json`, logs in `logs_np/`):

| Experiment | Recorded vs baseline | Recovered | Derived GPU fwd / bwd vs labels | L (or C) vs reference step time |
|---|---|---|---|---|
| 1 raw loop, float / ids | identical kernels + ops | params, tokens exact (first-linear / embedding) | +0.8 / +0.6%; −0.2 / +0.4% | +0.5% / +0.1% vs A ✅ |
| 2+3 DataLoader + accum, 3 configs × (C, L) | identical | 4 / 4, tokens, params exact | −0.5 to +0.1% / −0.2 to +0.9% | −0.2 to +0.3% vs A ✅ |
| 4 Trainer, config-built Llama | complete, all labels | 4 / 4, exact | +0.3% / −1.2% (vs raw loop R) | **+0.2% vs T**, floor 0.31% ✅ |
| 4b SmolLM2-360M | complete | 4 / 4, exact, tied weight once | +0.3% / −0.2% (vs R) | **+0.4% vs T**, floor 0.16% ❌ |
| 4b Qwen2.5-0.5B | complete | 4 / 4, exact, tied weight once | +5.6% / −0.5% (vs R: Trainer's fp32 logits, as before) | **+0.4% vs T**, floor 0.17% ❌ |

**What improved over the patched version:** the derived `forward` includes the loss ops, as our
runners' label does. Unlabeled GPU time now matches the baseline exactly (5.31 vs 5.31 ms in
experiments 2–3); the patched version was +16.5% there.

**Data-wait flags:** experiment 1 reads 0.0 (no DataLoader, out of scope). Config c (≈0.3 ms,
~0.1% of the step) is too small for a percentage. HF's 1.4 vs 3.4 ms is Trainer's collate path vs
the raw loop's, the same as experiment 4.

**Host phase times:** derived host `forward` reads about +24% vs the label in experiments 2–3.
The derived span runs up to the first autograd node, so it includes the hand-off into
`backward()`. Host phase times were already documented as loop-specific; rank on GPU times.

**Found, not explained: the patch-free version adds ~4 ms/step over the patched one on long HF
steps.** Throwaway A/B on the same box (SmolLM2-360M, 5 rounds; T, then new, then old each
round):

| | L − T per round (ms/step) | median |
|---|---|---|
| new (patch-free) | +9.5, +7.4, +5.7, +4.5, +6.0 | **+6.0** |
| old (patched, commit `51c8c4e`) | +10.6, +5.9, +2.0, −0.8, +0.8 | **+2.0** |

Paired, new − old = −1.1, +1.5, +3.7, +5.3, +5.2 ms. Both versions sync at essentially the same
point, right after the inner `AdamW.step`. The differences are:
- the new one syncs *inside* torch's `Optimizer.step` wrapper;
- it keeps the global post-hook registered, instead of removing it after the first step.

Neither obviously costs milliseconds, so the cause is open. The **CUDA-event timing** fix (record
an event at each boundary, sync once at the end of the clean window) would remove the per-step
sync from the clean window entirely. That would make the question moot, and would likely take
the old ~2 ms with it.

**Follow-up: the ~4 ms was mostly measurement design (A/A/B control, 2026-09-25).** 10 rounds on
SmolLM2-360M, arms new / new / old (identical code in the two "new" arms), **order shuffled each
round**. Raw records: `sandbox/attach_parity/results_np_aa_ab_smol.json`.
- **Position effect:** within a round, the 2nd and 3rd runs were +0.7 / +1.6 ms slower than the
  1st, likely GPU warm-up.
- **Session drift:** 1.270 s in round 1 up to ~1.291 s.
- The first A/B used a fixed order (T, new, old) and 5 rounds, which confounded both effects
  with the arm.

Fitting arm + position + round (least squares) gives:

| | all 10 rounds | round 1 dropped (first run of the session, cool GPU) |
|---|---|---|
| **new − old** (both new arms averaged), median / mean | +1.8 ± 0.8 / +1.6 ± 0.8 ms | **+1.7 ± 0.5 / +1.4 ± 0.5 ms** |
| A/A: new_a − new_b (identical code) | −1.4 ± 0.9 / −1.8 ± 0.9 ms | −0.5 ± 0.6 / −0.9 ± 0.5 ms |

**Reading:**
- The residual new − old is about **+1.5 ms/step, ≈0.12% of the step**, roughly 3σ once round 1
  is dropped. The A/A control sits within about 1.5σ of zero. So it's probably real, but about a
  third of the first estimate.
- It is far below the between-launch spread (1.9% max − min across all 30 runs; 0.5–0.9% round to
  round in experiment 4b). A constant instrument offset also cancels out of rankings when every
  probe uses the same method.
- Neither version is ground truth: both sync every step at nearly the same point. Settling
  "which is more correct" would need the D1–D3 ablation (sync inside vs outside torch's
  `Optimizer.step` wrapper; hook kept vs removed; extra wrapper frame) plus a less-perturbing
  reference, such as a `TrainerCallback` syncing only twice, N steps apart. **Parked:** not worth
  an hour of GPU for a 0.1% effect.

**Verdict:** the patch-free capture is **complete and correct** everywhere tested: kernels, fields,
grad accum, tokens, params, and GPU phase split. Its step time is within the noise floor
everywhere except the long pretrained runs, where it read ~+0.4% vs T. After the order-shuffled
control, the part attributable to the rewrite is ≈+1.5 ms/step (≈0.12%) vs the patched version,
far below the between-launch spread.

### Experiment 6: what the capture's timing gap actually is (2026-09-25)

**Question:** the capture (L) read ~+0.4% slower than a plain `TrainerCallback` (T) on the long
`from_pretrained` runs. **What is being measured differently, and which reading is right?**

**Hypothesis:** where the per-step CUDA sync happens.
- L syncs right after `optimizer.step()`, so Trainer's host bookkeeping before `on_step_end`
  (`lr_scheduler.step()`, `zero_grad`, callbacks) runs while the GPU sits idle.
- T syncs at `on_step_end`, so that same bookkeeping overlaps with the GPU finishing the
  optimizer kernels.

**Design** (`sandbox/attach_parity/hf/sync_placement.py`): the same Trainer script, SmolLM2-360M,
8 rounds, arm order shuffled every round, fitted for round + position.

| Arm | Timing |
|---|---|
| L | the capture (`attach.py`): sync after the inner `AdamW.step` |
| T_opt | `TrainerCallback`, sync at `on_optimizer_step`: the **same point** as L, no hooks |
| T_end | `TrainerCallback`, sync at `on_step_end` (the T of experiments 4 / 4b) |
| GT | `TrainerCallback` syncing only **twice**, 10 steps apart: the least perturbed reference |

T_opt also records the host time from its sync to `on_step_end`: the work L's placement
leaves exposed. That's a direct measurement, not an inference. Raw:
`results_np_e6_sync_smol.json`.

**Results** (round 1 dropped: the first launch of the session ran on a cold GPU, T_end 1.2698 s
vs ~1.30 s for everything after):

| Arm | vs GT, mean | vs GT, median |
|---|---|---|
| L | **+3.3 ± 1.5 ms** | **+3.6 ± 1.5 ms** |
| T_opt | +1.6 ± 1.5 | +1.9 ± 1.4 |
| T_end | +1.3 ± 1.3 | +1.2 ± 1.3 |

The exposed host work (T_opt: from the sync to `on_step_end`) is **1.18 ms/step**, 10th–90th
percentile 1.15–1.22 ms, over 80 steps.

**Attribution:**
- **Sync placement is a real mechanism, bounded at 1.18 ms/step** (measured). It is about half
  of L − T_end (+2.0 to +2.4 ms).
- **The remaining ~1 ms is not attributed.** It appears as L − T_opt (+1.6 ± 1.3 ms: same sync
  point, capture vs callback). That's within noise.
- This design **cannot resolve ~1 ms**: between-launch residual sd is ~2.3 ms, giving ±1.3–1.5 ms
  per contrast. Settling the remainder needs several times more rounds or a longer clean window.

**Which is more correct:**
- T_end-style timing is statistically indistinguishable from GT.
- **The capture reads ~0.25% high** (≈3 ms on a 1.3 s step, ~2σ).
- Both the patched and the patch-free capture sync at essentially this same early point, so the
  bias predates the rewrite. (The rewrite itself adds ≈1.5 ms at most; experiment 5 follow-up.)

**Fix, and what it predicts:** time the clean window with **CUDA events**. Record a
`torch.cuda.Event` at each optimizer-step boundary, with no host sync, and synchronize once at
the end of the window. The host bookkeeping then overlaps with GPU work exactly as in an
untouched run, so the placement term disappears by construction. It's still public API and
patch-free. **Prediction:** the capture moves to within noise of GT. If a gap remains, it's
the unattributed ~1 ms, and that becomes the next question.

**Why this matters for the loop:** a ~0.25% constant bias cancels in rankings when every probe
uses the same method. It matters only when a capture number is compared with an external one
(the user's own logs, or a different instrument).

### Experiment 7: LoRA (2026-09-25)

**Setup:**
- Experiment 4b's harness, unchanged, on the pinned SmolLM2-360M and Qwen2.5-0.5B checkpoints.
- A peft LoRA adapter: r=16, α=32, dropout 0.05, on `q/k/v/o/gate/up/down_proj`.
- `peft==0.20.0`: the release current when transformers 5.15.1 shipped. It doesn't pin
  transformers, so compatibility was verified by running, and imports are clean with warnings as
  errors.
- 5 rounds; arms R / L / T. Sandbox measurement (patch-free), guard active.
- Raw: `results_np_e7_lora_{smol,qwen}.json`, `logs_np/e7_lora_*.log`.

| | SmolLM2-360M + LoRA | Qwen2.5-0.5B + LoRA |
|---|---|---|
| Loaded as | `PeftModelForCausalLM` over `LlamaForCausalLM`, bf16, tied | over `Qwen2ForCausalLM`, bf16, tied |
| Labels (`step` / fwd / bwd / dataloader / `Optimizer.step`) | 1 / 4 / 4 / 4 / 1 | 1 / 4 / 4 / 4 / 1 |
| forwards / backwards per step, tokens | 4 / 4, 16,384 exact | 4 / 4, 16,384 exact |
| params reported (the optimizer's: trainable only) | 8,683,520 of 370,504,640 (2.34%) | 8,798,208 of 502,830,976 (1.75%) |
| Observation fields empty or wrongly 0 vs R | none | none |
| **MFU reported → LoRA-corrected** | **0.53% → 15.3% (28.8× low)** | **0.51% → 19.7% (38.4× low)** |
| L vs T step time | +0.3% (≈5.3 ms), floor 0.12% | +0.3% (≈5.0 ms), floor 0.11% |

The corrected MFU uses FLOPs/step ≈ (4·N_total + 2·N_trainable)·D: forward plus
backward-through-activations on every weight, and weight gradients only for trainable ones.
That's an approximation (it ignores, e.g., that no gradient flows into the frozen embedding).

**Findings:**
1. **Sandbox measurement is complete and correct under PEFT.** The step boundary fires through an
   optimizer that holds only adapter weights, and labels, micro-batches and tokens are all exact.
2. **MFU is unusable under LoRA:** 29–38× too low, because N comes from the optimizer. This
   settles `integration_plan.md` D2: it must be fixed before user scripts go live. The model has
   to be counted in full, and the FLOP formula has to separate total from trainable.
3. **The L − T timing gap is larger under LoRA** (≈5 ms vs 2–3 ms for full fine-tuning in
   experiment 6), with the same sign. **Unattributed:**
   - The sync-placement mechanism would predict a *smaller* gap, since the LoRA optimizer's GPU
     work is 1.6 ms vs 28 ms.
   - This harness runs a fixed arm order with T last, which would shrink the gap, not grow it.
   - It's worth one order-shuffled run. At 0.3%, it cancels in rankings.
4. **Context (LoRA is not a knob; it changes what's learned):**
   - At this scale, LoRA steps are slower than full fine-tuning (SmolLM2: 1.82 s vs 1.30 s in
     experiment 4b). The adapter adds matmuls and dropout on every projection, more than the ~26
     ms of optimizer work it saves.
   - Under Trainer, a LoRA step is 3.5–4.0% slower than the raw loop (R), with GPU busy
     −0.2 / +1.3%, i.e. more GPU idle per step.

## Integration: draft recommendation (2026-09-25, not decided)

**Nomenclature:**
- **OG** = the existing workflow: `profiler/capture.py: capture(runner, config)` driving our own
  runners (`frameworks/hf.py`, `frameworks/torch.py`), with `record_function` labels and
  in-process OOM recovery for the ceiling.
- **sandbox** = the new workflow in `sandbox/attach_parity/`: `launcher.py` running the user's
  script, plus the patch-free `attach.py` / `derive.py` / `nopatch.py`.

In those terms: **sandbox becomes the only measurement engine; OG's runners stay as training
loops driven through it.**

From a delegated architecture review; its two load-bearing claims were checked in code.

**Recommended: one measurement engine, two drivers.** `Attach` + `derive` becomes the only thing
that measures:
- **User scripts** run through the launcher.
- **Our runners** go through a `capture(runner, config)` rewritten as a thin driver: install
  `Attach`, loop `runner.step()` until done, catch OOM in-process.
- **The runners stay as training loops:** the default trainee, the CPU test fixtures, in-process
  ceiling trials. They no longer define metric semantics, so there is one set of semantics for
  the §12 metric reference.

Rejected alternatives:
- (A) attached only: loses fixtures, and breaks `frame --ceiling` and `backends/local`.
- (B) two peer paths: two semantics, permanent parity drift.
- (C) runners as test-only fixtures: tests validate a path production never runs.

**Verified corrections:**
- In-process ceiling trials also reload the model on every trial: `frame/ceiling.py:116` →
  `profiler/capture.py:246` (`runner.setup`) → `frameworks/hf.py:152` (`from_pretrained`). So
  per-launch trials for user scripts cost only process startup and CUDA init on top.
- **Regression to fix before promotion:** `Attach` counts tokens from the single profiled step.
  `frameworks/hf.py:324-340` uses the MEAN over all steps, because under `padding="longest"` a
  single step's count gave 7490 vs 9449 tok/s for identical configs. All experiments here used
  fixed shapes, so none caught it.

**Other gaps flagged:**
- `augment` must skip phase names already present, or a labelled runner's trace double-counts in
  `trace.phase_times_us`.
- Tokens under `torch.compile` may be lost if Inductor fuses away `aten::embedding` (inferred).
- `_MEMORY_KNOBS` can't see a user framework's flag names, so ceiling staleness would go undetected.

**Draft migration steps:**
1. Split `profiler/capture.py` into `ceilings.py` / `window.py`, no behaviour change.
2. Promote `attach` / `derive` / `nopatch` with their tests.
3. Declared tokens/params override derived; labels win; tokens averaged over the clean window.
4. `capture()` over `Attach`, with the legacy one kept. GPU parity on `HFRunner`, shuffled, with
   `padding=longest`, gradient checkpointing, compile, chunked CE.
5. Promote the launcher; add a script mode to `backends.local.run`.
6. `find_ceiling(trial)` with runner and script trials; shape check; re-measure from the
   previous ceiling.
7. Docs.
8. Delete legacy only after GPU parity AND a short full loop reproduce a prior run within the
   between-launch floor.

**Open questions:**
- runner labels: keep or drop;
- tokens under dynamic padding for scripts: shapes over the whole window, or rank on samples/s;
- total params under LoRA via public APIs;
- CUDA-event timing before or after the migration (it re-baselines `knowledge.md` numbers);
- ceiling staleness for user flag names;
- compile / gradient checkpointing as blockers or documented limitations;
- `hf.py` also runnable as a script;
- data-wait null vs 0.0;
- whether to keep `vendor/`.
