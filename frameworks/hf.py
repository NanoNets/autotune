"""Real HuggingFace causal-LM runner — the framework adapter behind a real probe.

Implements the profiler's runner interface (`setup(config)` / `step()` / `tokens_per_step()` +
a `.params` attribute) for an actual `transformers` model trained with a raw PyTorch loop, so the
profiler can wrap each phase in `record_function` and time a genuine fwd/bwd/opt. Unlike frame
(config-only), this loads real weights and runs real data.

`knobs()` reports the tunable surface — the levers that change SPEED, not the learned model — so
the loop knows what it may turn. Heavy imports (torch / transformers / datasets) are lazy, so this
module plus `knobs()` / `_resolve()` import on bare python3.
"""

_DEFAULTS = {
    "device": "cuda",
    "dtype": "bfloat16",
    "micro_batch": 1,
    "grad_accum": 1,
    "seq_len": 1024,
    "gradient_checkpointing": False,
    "attn_implementation": "sdpa",     # always available in torch 2.x; flash_attention_2 needs the lib
    "torch_compile": False,
    "sample": 256,                     # rows pulled into the in-memory loader
    "name": None,                      # dataset config/subset
    "split": "train",
    "text_field": "text",
    "lr": 1e-5,                        # doesn't affect step timing; just to construct the optimizer
}


def knobs():
    """The tunable surface this framework exposes — speed levers only (nothing that changes the
    learned model). The loop turns these; availability (e.g. flash-attn installed) is gated by
    frame's `versions()`."""
    return {
        "micro_batch":            {"type": "int",    "min": 1},
        "grad_accum":             {"type": "int",    "min": 1},
        "seq_len":                {"type": "int",    "min": 1},
        "dtype":                  {"type": "choice", "options": ["bfloat16", "float16", "float32"]},
        "gradient_checkpointing": {"type": "bool"},
        "attn_implementation":    {"type": "choice", "options": ["eager", "sdpa", "flash_attention_2"]},
        "torch_compile":          {"type": "bool"},
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
    for req in ("model", "dataset"):
        if not c.get(req):
            raise ValueError(f"config needs a {req!r} (HF hub id or local path)")
    if c["sample"] < c["micro_batch"]:
        raise ValueError(f"sample ({c['sample']}) < micro_batch ({c['micro_batch']}): loader would be empty")
    # one optimizer step covers grad_accum micro-batches of micro_batch x seq_len tokens
    c["tokens_per_step"] = c["micro_batch"] * c["seq_len"] * c["grad_accum"]
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

    def _next(self):
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

    def tokens_per_step(self):
        return self.cfg["tokens_per_step"]
