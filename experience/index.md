# Experience — index

One entry per run, oldest first. Read **once, at Frame** (`loop/loop.md` §3), to answer "has this
configuration been run before?": match this run's Profile against each `Context:` line below, then
open **only** the entries that match or nearly match. The full write-up (winning config, findings,
dead levers, open hypotheses) lives in the entry's file. See `experience/README.md`.

Each index line copies its entry's `Context:` line **verbatim**, because that line is the match key.

- [test_a10G](test_a10G.md) — 2026-09-15
  Context: `Qwen/Qwen2.5-0.5B` · `wikitext-2-raw-v1` · 1x NVIDIA A10 22.1GB (CC 8.6), bf16, effective_batch 16, max_len 2048, ssh backend
  Baseline: 4.338 samples/s → Best: 108.59 samples/s (+2403%, 25.0x)
- [smollm2_1.7b_a100_wikitext](smollm2_1.7b_a100_wikitext.md) — 2026-09-21
  Context: `HuggingFaceTB/SmolLM2-1.7B-Instruct` · `wikitext-2-raw-v1` · 1x NVIDIA A100-SXM4-40GB (CC 8.0, 39.5GB), bf16, effective_batch 16, max_len 1024, ssh backend
  Baseline: 9.38 samples/s → Best: 102.77 samples/s (+995%, 10.96x)
- [qwen3.5_2b_a100_wikitext](qwen3.5_2b_a100_wikitext.md) — 2026-09-21
  Context: `Qwen/Qwen3.5-2B` · `wikitext-2-raw-v1` · 1x NVIDIA A100-SXM4-40GB (CC 8.0, 39.5GB), bf16, effective_batch 16, max_len 1024, ssh backend
  Baseline: 2.154 samples/s → Best: 36.006 samples/s (+1571%, 16.72x)
