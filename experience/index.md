# Experience — index

One entry per run, oldest first. Read **once, at Frame** (`loop/loop.md` §3), to answer "has this
configuration been run before?": match this run's Profile against each `Context:` line below, then
open **only** the entries that match or nearly match. The full write-up (winning config, findings,
dead levers, open hypotheses) lives in the entry's file. See `experience/README.md`.

Each index line copies its entry's `Context:` line **verbatim**, because that line is the match key.

- [a10_qwen2.5-0.5b_wikitext](a10_qwen2.5-0.5b_wikitext.md) — 2026-10-08
  Context: `Qwen/Qwen2.5-0.5B` · `wikitext-2-raw-v1` · 1x NVIDIA A10 22.06GB (CC 8.6) · bf16 (checkpoint's, none set) · effective_batch 64 · max_len 2048
  Baseline: 4.158 samples/s → Best: 88.120 samples/s (+2019%, 21.2x)
- [a10_smollm2-1.7b_wikitext](a10_smollm2-1.7b_wikitext.md) — 2026-10-08
  Context: `HuggingFaceTB/SmolLM2-1.7B` · `wikitext-2-raw-v1` · 1x NVIDIA A10 22.06GB (CC 8.6) · bf16 (checkpoint's, none set) · effective_batch 16 · max_len 1024
  Baseline: 3.677 samples/s → Best: 35.991 samples/s (+879%, 9.79x)
- [a10_qwen2.5-1.5b_cosmopedia](a10_qwen2.5-1.5b_cosmopedia.md) — 2026-10-08
  Context: `Qwen/Qwen2.5-1.5B` · `HuggingFaceTB/cosmopedia-100k` · 1x NVIDIA A10 22.06GB (CC 8.6) · bf16 (checkpoint's, none set) · effective_batch 16 · max_len 2048
  Baseline: 1.493 samples/s → Best: 7.052 samples/s (+372%, 4.72x)
- [a10_smollm2-360m_lora_script](a10_smollm2-360m_lora_script.md) — 2026-10-09
  Context: `HuggingFaceTB/SmolLM2-360M` LoRA r16 (7 projections, plain-PyTorch user script `examples/byo_script/torch_lora.py`) · `wikitext-2-raw-v1` · 1x NVIDIA A10 22.06GB (CC 8.6) · bf16 (script sets it; adapters fp32 via peft) · effective_batch 16 · max_len 1024
  Baseline: 27.74 samples/s → Best: 116.43 samples/s (+320%, 4.20x)
