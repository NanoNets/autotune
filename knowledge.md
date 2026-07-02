# Knowledge base — what we've learned across runs

The loop's **durable, cross-run memory** (above per-run `search_state.json`). The agent READS this at
run start and CURATES it at run end (see `program.md`). It is curated ground-truth: writes are gated
and rare — only confirmed findings, each tagged with the context it holds in.

**Entry schema** (one `##` per claim, grouped by topic):

- **Claim** — the finding, with magnitude.
- **Context** — *required.* The setup it was learned in (model, GPU, max_length, data shape, regime).
  An entry without context is unusable — you can't tell if it applies.
- **Why** — the mechanism, when known.
- **Scope** — `invariant` (a mechanism/direction that holds broadly; only magnitude is context-specific)
  or `contextual` (true in that regime; a HYPOTHESIS to re-test elsewhere).
- **Evidence** — run date + probe(s).

**How to use it (summary; rules in `program.md`):** invariant → apply as a prior. contextual +
context matches → strong prior. contextual + context differs → don't assume, test it as a hypothesis.

---

## gradient_checkpointing trades compute for activation memory
- **Claim:** GC cuts peak activation memory (recomputes activations in backward) at ~some throughput cost.
- **Context:** general transformer/VLM training.
- **Why:** activations aren't kept for backward; they're recomputed.
- **Scope:** invariant — the direction is a law; the magnitude (and whether it's worth it) is context-specific.
- **Evidence:** mechanism; seen 2026-06-25.

## At 70k max_length, GC is mandatory and bs caps at 2 (Qwen3.5-0.8B, H100×1)
- **Claim:** GC off OOMs even at bs=1 (~46 GB activations at 70k); bs=2 fits (70/80 GB), bs=4 OOMs (~140 GB).
- **Context:** Qwen3.5-0.8B, H100 80GB ×1, max_length 70000, image-heavy OCR (256 worst-case largest samples).
- **Why:** activation memory scales with sequence length; 70k windows are huge.
- **Scope:** contextual — re-derive the ceiling at a different max_length / GPU memory / model.
- **Evidence:** 2026-06-25, probes across the bs sweep.

## padding_free hurt throughput (−42%) on long uniform image-heavy samples
- **Claim:** padding_free was −42% vs padded.
- **Context:** Qwen3.5-0.8B, H100×1, max_length 70k, near-uniform long sample lengths (little padding present).
- **Why:** with little padding to recover, the repack overhead isn't offset by saved compute.
- **Scope:** contextual — re-test when sample lengths vary widely (lots of padding to recover); likely flips positive.
- **Evidence:** 2026-06-25.

## packing OOMs even at bs=1 at 70k max_length (H100×1) — SUPERSEDED
- **Claim:** packing fills the 70k window → ~4× activations → OOM even at bs=1.
- **Context:** Qwen3.5-0.8B, H100 80GB ×1, max_length 70000.
- **Scope:** superseded — `packing` is now excluded as a lever entirely (it repacks multiple samples
  into one block, changing what counts as a "sample" per step and breaking the effective_batch
  invariant; see program.md Guardrails). This memory-ceiling finding no longer matters.
- **Evidence:** 2026-06-25.

## liger_kernel and torch_compile don't help Qwen3.5
- **Claim:** liger_kernel = no-op (no kernels for Qwen3.5 linear-attn); torch_compile errors (model not dynamo-traceable).
- **Context:** Qwen3.5 architecture (linear-attention VLM).
- **Scope:** contextual to Qwen3.5. NOTE: liger was *importable* yet had no kernels — the capability scan
  can't catch this (present-but-no-kernel); it needed a probe.
- **Evidence:** 2026-06-25.

## Qwen3.5 linear-attn ran the slow PyTorch fallback (causal_conv1d missing)
- **Claim:** every probe ran linear-attn on the slow fallback ("fast path is not available"); installing
  causal_conv1d + flash-linear-attention likely beats the ~24% won by tuning.
- **Context:** Qwen3.5-0.8B on the Paperspace VM swift-env (causal_conv1d not installed).
- **Scope:** contextual to this VM env — check the capability scan's `libs.causal_conv1d` before assuming.
- **Evidence:** 2026-06-25 (load log); out-of-band fix, needs VM-owner approval.

## "workers don't help" here is SUSPECT — page-cached data + too-short probe
- **Claim:** workers=8 read −20%, but the worst-case set is page-cached (27 unique images, 490 GB RAM) and
  that probe was only 7 steps, so worker spin-up wasn't amortized. Treat as unresolved, not "data isn't the wall."
- **Context:** Qwen3.5-0.8B, H100×1, 70k, the 256-sample worst-case set; regime read compute-bound.
- **Scope:** contextual + low-confidence — RE-TEST with `--max-steps 20 --warmup 5` before concluding.
- **Evidence:** 2026-06-25 (single 7-step probe).
