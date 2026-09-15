# Beliefs &mdash; index

*(empty — the agent appends one line per belief file as it learns)*

- [ce-dominates-activation-memory-large-vocab] — on small models with large vocabs, the logits/CE term
  (not attention) dominates activation memory at long `max_len`; recompute cannot touch it.
- [tf32-inert-under-bf16] — `tf32` has no effect on a pure-bf16 graph: no fp32 matmul exists to accelerate.
- [cudagraphs-incompatible-with-grad-accum] — any cudagraph-enabling compile mode breaks a runner that
  invokes the model multiple times per optimizer step.
- [padding-regime-flips-attention-verdict] — `flash_attention_2` beats `sdpa` only while batches carry
  heavy padding; once padding is removed at the loader, its unpad/repad overhead makes it a net loss.
