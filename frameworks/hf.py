"""Real HuggingFace causal-LM runner — the framework adapter behind a real probe.

Implements the profiler's runner interface (`setup(config)` / `step()` / `tokens_per_step()` +
a `.params` attribute) for an actual `transformers` model trained with a raw PyTorch loop, so the
profiler can wrap each phase in `record_function` and time a genuine fwd/bwd/opt. Unlike frame
(config-only), this loads real weights and runs real data.

`knobs()` reports the tunable surface — the levers that change SPEED, not the learned model — so
the loop knows what it may turn. Heavy imports (torch / transformers / datasets) are lazy, so this
module plus `knobs()` / `_resolve()` import on bare python3.

Two data sources. The default loads real rows and is what a throughput probe runs on.
`data="synthetic"` takes a worst-case batch from `frame.synthetic` instead — every sequence exactly
`seq_len`, no padding, no dataset touched — which is what an OOM ceiling search drives
(`frame/ceiling.py`). Memory is decided by the longest batch a run will ever see, and real rows
supply one only by luck. `teardown()` frees a trial's weights before the next one starts.
"""

_DEFAULTS = {
    "device": "cuda",
    "dtype": "bfloat16",               # run PARAMETER (off-limits precision), not a searchable knob
    "seq_len": 1024,                   # run PARAMETER (off-limits max_length), not a searchable knob
    "effective_batch": 16,             # human-owned target; the loop factorizes it, never changes it
    "micro_batch": 1,                  # the free batch knob; grad_accum is derived to hold effective_batch
    "gradient_checkpointing": False,
    "attn_implementation": "sdpa",     # always available in torch 2.x; flash_attention_2 needs the lib
    "torch_compile": False,
    "tf32": False,                     # TF32 matmuls on Ampere+ (speeds matmul, doesn't change the model)
    "sample": 256,                     # rows pulled into the in-memory loader
    "name": None,                      # dataset config/subset
    "split": "train",
    "text_field": "text",
    "lr": 1e-5,                        # doesn't affect step timing; just to construct the optimizer
}


def knobs():
    """The tunable surface this framework exposes — SAFE speed levers only (per loop/throughput.md:
    change speed, not the learned model). Deliberately NOT here: `dtype` (precision) and `seq_len`
    (max_length) are run PARAMETERS the human sets, off-limits to search; `grad_accum` isn't free
    either — it's derived to hold `effective_batch`. Availability (e.g. flash-attn) is gated by
    frame's `versions()`. Parallelism / offload / packing / fused kernels: deferred (see README)."""
    return {
        "micro_batch":            {"type": "int",    "min": 1},   # grad_accum = effective_batch // micro_batch
        "gradient_checkpointing": {"type": "bool"},
        "attn_implementation":    {"type": "choice", "options": ["eager", "sdpa", "flash_attention_2"]},
        "torch_compile":          {"type": "bool"},
        "tf32":                   {"type": "bool"},
    }


def _usable_texts(rows, text_field, sample):
    """Non-empty texts from a row iterable, capped at `sample`.

    Skips blank/whitespace rows: an all-padding row tokenizes to all pad tokens, whose labels are
    all -100, whose causal-LM loss is 0/0 = NaN — which backprops into every param and silently
    poisons every later step. A missing field reads as "" and is skipped too.
    """
    texts = []
    for row in rows:
        t = row.get(text_field, "")
        if t and t.strip():
            texts.append(t)
            if len(texts) >= sample:
                break
    return texts


def _resolve(config):
    """Fill defaults + derived values. Pure — no torch/transformers, so it's unit-testable."""
    c = {**_DEFAULTS, **config}
    if c.get("max_len"):                                # canonical name (loop/throughput.md) ->
        c["seq_len"] = c["max_len"]                     # this framework's local one
    synthetic = c.get("data") == "synthetic"
    required = ("model",) if synthetic else ("model", "dataset")   # synthetic reads no dataset
    for req in required:
        if not c.get(req):
            raise ValueError(f"config needs a {req!r} (HF hub id or local path)")
    eb, mb = c["effective_batch"], c["micro_batch"]
    if eb % mb != 0:                                    # the loop only factorizes effective_batch,
        raise ValueError(                              # never changes it (it's the quality boundary)
            f"effective_batch ({eb}) must be divisible by micro_batch ({mb})")
    c["grad_accum"] = eb // mb                          # derived to hold the human's effective_batch
    if not synthetic and c["sample"] < mb:              # synthetic builds its batch, never loads rows
        raise ValueError(f"sample ({c['sample']}) < micro_batch ({mb}): loader would be empty")
    c["tokens_per_step"] = eb * c["seq_len"]            # = micro_batch * grad_accum * seq_len
    return c


class HFRunner:
    """A runner the profiler drives: setup(config) -> repeated step() -> tokens_per_step()."""

    def setup(self, config):
        import torch
        from torch.profiler import record_function
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.cfg = _resolve(config)
        self.device = self.cfg["device"]
        self._rf = record_function
        dtype = getattr(torch, self.cfg["dtype"])
        if self.cfg["tf32"]:                            # Ampere+ TF32 matmul path
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        tok = AutoTokenizer.from_pretrained(self.cfg["model"], trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            self.cfg["model"], dtype=dtype,                 # transformers 5.x: torch_dtype is deprecated
            attn_implementation=self.cfg["attn_implementation"],
            trust_remote_code=True).to(self.device)
        self.params = sum(p.numel() for p in self.model.parameters())   # before compile wrapping
        if self.cfg["gradient_checkpointing"]:
            self.model.gradient_checkpointing_enable()
        self.model.train()
        if self.cfg["torch_compile"]:
            self.model = torch.compile(self.model)

        self.opt = torch.optim.AdamW(self.model.parameters(), lr=self.cfg["lr"])
        if self.cfg.get("data") == "synthetic":
            self._loader = self._it = None
            self._batch = self._synthetic_batch(tok)    # one worst-case batch, reused every step
        else:
            self._batch = None
            self._loader = self._build_loader(tok)
            self._it = iter(self._loader)

    def _build_loader(self, tok):
        import torch
        from datasets import load_dataset

        c = self.cfg
        rows = load_dataset(c["dataset"], c["name"], split=c["split"], streaming=True)
        texts = _usable_texts(rows, c["text_field"], c["sample"])
        if len(texts) < c["micro_batch"]:                  # else drop_last empties the loader ->
            raise ValueError(                              # StopIteration out of step()
                f"only {len(texts)} usable rows from {c['dataset']} (split={c['split']}, "
                f"field={c['text_field']!r}); need at least micro_batch={c['micro_batch']}")
        enc = tok(texts, truncation=True, max_length=c["seq_len"], padding="max_length")
        input_ids = torch.tensor(enc["input_ids"])
        attention = torch.tensor(enc["attention_mask"])
        labels = input_ids.masked_fill(attention == 0, -100)   # don't train on padding
        dset = torch.utils.data.TensorDataset(input_ids, attention, labels)
        return torch.utils.data.DataLoader(dset, batch_size=c["micro_batch"],
                                           shuffle=True, drop_last=True)

    def _synthetic_batch(self, tok):
        """One worst-case batch: every row exactly seq_len, zero padding, on the device.

        Labels are the ids themselves — nothing is masked to -100, because there IS no padding.
        That is the worst case on purpose: loss runs over every position, so the [B, S, vocab]
        logits and the cross-entropy over them are at full size, and those are usually where the
        peak actually lands.

        vocab: the model's embedding width (`frame`'s `model["vocab"]`), falling back to the
        tokenizer's. An id past it is a device-side assert, not the OOM being measured.
        """
        from frame.synthetic import random_batch
        c = self.cfg
        b = random_batch(c.get("vocab") or len(tok), c["micro_batch"], c["seq_len"],
                         self.device, seed=c.get("seed", 0))
        return b["input_ids"], b["attention_mask"], b["input_ids"].clone()

    def _next(self):
        if self._batch is not None:                        # synthetic: one fixed worst-case batch
            return self._batch
        try:
            return next(self._it)
        except StopIteration:                              # cycle: the profiler runs more steps than one pass
            self._it = iter(self._loader)
            return next(self._it)

    def step(self):
        c = self.cfg
        self.opt.zero_grad(set_to_none=True)
        for _ in range(c["grad_accum"]):
            with self._rf("dataloader"):
                input_ids, attention, labels = self._next()
            input_ids = input_ids.to(self.device)
            attention = attention.to(self.device)
            labels = labels.to(self.device)
            with self._rf("forward"):
                loss = self.model(input_ids=input_ids, attention_mask=attention,
                                  labels=labels).loss / c["grad_accum"]
            with self._rf("backward"):
                loss.backward()
        with self._rf("optimizer"):
            self.opt.step()

    def teardown(self):
        """Free this trial's weights/optimizer/data, so a ceiling search's next trial starts on a
        clean card. Without it each trial measures the previous trial's leftovers."""
        self.model = self.opt = self._loader = self._it = self._batch = None

    def tokens_per_step(self):
        return self.cfg["tokens_per_step"]
