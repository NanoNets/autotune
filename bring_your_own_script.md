# Bring your own training script

> **Developer notes. Not part of the agent's prompt hierarchy.**
> `program.md` does not route here and the loop never reads or writes this file. It is where we
> work out the architecture for running the loop on a user's own training script. Nothing here is
> built yet, and every claim is marked by whether it has been verified.

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
