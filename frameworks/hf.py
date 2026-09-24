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
    "torch_compile": False,             # False | True (default mode) | a torch.compile mode string
    "tf32": False,                     # TF32 matmuls on Ampere+ (speeds matmul, doesn't change the model)
    "fused_optimizer": False,          # torch.optim.AdamW(fused=True): one fused kernel per param
                                       # group instead of the foreach multi_tensor_apply chain.
                                       # SAME AdamW algorithm -- a kernel swap, not an algorithm
                                       # change, so it is a speed knob (loop/throughput.md lists
                                       # "fused optimizer" under Kernels). Matters most when the
                                       # optimizer is a large share of the step: its cost is per
                                       # PARAMETER, so it does not shrink with the batch.
    "chunked_ce": 0,                   # 0 = off (the model computes its own loss). N > 0 = compute
                                       # cross-entropy in N chunks over the token axis, each chunk
                                       # checkpointed so its [chunk, vocab] logits are freed after
                                       # the forward and recomputed in the backward. SAME loss --
                                       # a sum over the same terms, in the same order, divided by
                                       # the same count -- so it is a memory/kernel lever, not a
                                       # model change (loop/throughput.md lists "chunked
                                       # cross-entropy" under Kernels). Pays only when
                                       # vocab x seq is large enough for the logits to dominate.
    "padding": "max_length",           # "max_length" (every batch seq_len) | "longest" (pad to the
                                       # batch's own longest, rounded up to pad_multiple)
    "pad_multiple": 64,                # bucket dynamic lengths so torch.compile sees few shapes
    "length_grouped": False,           # batch similar-length rows together, so pad-to-longest wastes
                                       # less; only meaningful with padding="longest"
    "sample_file": None,               # the probe sample Frame drew (frame.dataset.draw_sample):
                                       # length-stratified from a random pool. Every probe of a
                                       # run should load the same file. None -> the head fallback
    "sample": 256,                     # head fallback only: the FIRST `sample` usable rows, trimmed
                                       # DOWN to a multiple of effective_batch. Not representative
                                       # of an ordered file; population records it as "head"
    "loader_seed": 0,                  # seeds every shuffle, so a config sees the same order twice
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
        "torch_compile":          {"type": "choice",
                                   "options": [False, "default", "reduce-overhead",
                                               "max-autotune", "max-autotune-no-cudagraphs"]},
        "tf32":                   {"type": "bool"},
        "fused_optimizer":        {"type": "bool"},   # AdamW(fused=True); CUDA + float params only
        "padding":                {"type": "choice", "options": ["max_length", "longest"]},
        "length_grouped":         {"type": "bool"},   # pairs with padding="longest"
        "chunked_ce":             {"type": "int",    "min": 0},   # 0 = off; N = chunks
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


def _whole_steps(texts, effective_batch):
    """Trim `texts` down to a multiple of `effective_batch`, keeping file order.

    A pass over the sample must be a whole number of optimizer steps, or the pass boundary falls
    mid-step and no clean window can cover every row the same number of times. Trimming from the
    END keeps the kept rows identical across configs: every config drops the same tail.
    """
    return texts[:len(texts) - len(texts) % effective_batch]


def _reference_indices(lengths, n):
    """The `n` rows nearest the median length, ordered by length: the fixed reference batch.

    Chosen by rule, not by loader position, so `clean_steps`, warmup and order can't move it.
    """
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], i))
    start = max(0, min(len(order) // 2 - n // 2, len(order) - n))
    return order[start:start + n]


def _rows_hash(rows):
    """Short content hash of a list of token-id sequences: identifies a population of rows."""
    import hashlib
    import sys
    from array import array
    h = hashlib.sha1()
    for r in rows:
        h.update(len(r).to_bytes(4, "little"))
        ids = array("i", r)                             # 4-byte ints, no per-token Python loop
        if sys.byteorder == "big":
            ids.byteswap()                              # hash little-endian bytes on any host
        h.update(ids.tobytes())
    return h.hexdigest()[:16]


def _sample_file_texts(c):
    """The texts of a Frame-drawn probe sample, checked against this probe's config.

    A sample drawn from a different dataset, split or field is a different population, so a
    mismatch is an error rather than a silently wrong measurement. Returns (texts, meta).
    """
    from frame.dataset import read_sample
    meta, rows = read_sample(c["sample_file"])
    for key in ("dataset", "name", "split", "text_field"):
        if meta.get(key) != c.get(key):
            raise ValueError(f"sample_file {c['sample_file']!r} was drawn with {key}="
                             f"{meta.get(key)!r}, but this probe has {key}={c.get(key)!r}")
    return [r["text"] for r in rows], meta


def _length_grouped(rows, micro_batch, seed=0):
    """Reorder rows so each consecutive `micro_batch` chunk holds similar lengths.

    Pad-to-longest costs `micro_batch * max_len_in_batch`, so mixing a 10-token row with a
    500-token one pads the short one to 500. Sorting by length collapses that waste; shuffling the
    resulting CHUNKS (not the rows) keeps batch order random so training doesn't see all the short
    sequences first.
    """
    import random
    order = sorted(range(len(rows)), key=lambda i: rows[i].numel())
    chunks = [order[i:i + micro_batch] for i in range(0, len(order), micro_batch)]
    random.Random(seed).shuffle(chunks)
    return [rows[i] for chunk in chunks for i in chunk]


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
    if not synthetic and not c["sample_file"] and c["sample"] < eb:   # synthetic loads no rows
        raise ValueError(f"sample ({c['sample']}) < effective_batch ({eb}): "   # a pass must hold
                         "no whole optimizer step fits in the sample")          # one whole step
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
        compile_mode = self.cfg["torch_compile"]
        if compile_mode:                                # True -> inductor default; str -> that mode
            self.model = torch.compile(
                self.model, **({} if compile_mode is True else {"mode": compile_mode}))

        self.opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.cfg["lr"],
            **({"fused": True} if self.cfg["fused_optimizer"] else {}))
        self._reference = self._population = self.steps_per_pass = None
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
        if c["sample_file"]:                               # the rows Frame drew: every probe of the
            texts, self._sample_meta = _sample_file_texts(c)   # run times exactly these
        else:                                              # head fallback: first N usable rows
            rows = load_dataset(c["dataset"], c["name"], split=c["split"], streaming=True)
            texts = _usable_texts(rows, c["text_field"], c["sample"])
            self._sample_meta = {"method": "head"}
        if len(texts) < c["effective_batch"]:              # a pass must hold at least one step, or
            raise ValueError(                              # drop_last empties the loader
                f"only {len(texts)} usable rows from {c['dataset']} (split={c['split']}, "
                f"field={c['text_field']!r}); need at least effective_batch={c['effective_batch']}")
        texts = _whole_steps(texts, c["effective_batch"])
        # Every row is used exactly once per pass: micro_batch divides effective_batch, which
        # divides the trimmed sample, so drop_last drops nothing and a pass is whole steps.
        self.steps_per_pass = len(texts) // c["effective_batch"]
        seeded = torch.Generator().manual_seed(c["loader_seed"])   # same order, run to run
        mb = c["micro_batch"]
        if c["padding"] == "longest":
            # Pad each batch to its OWN longest row instead of to seq_len. Identical arithmetic --
            # pad positions are masked out of attention and set to -100 in labels, so they
            # contribute nothing to the loss or the gradients -- it just stops spending compute on
            # them. seq_len still caps via truncation, so max_len is respected, not changed.
            enc = tok(texts, truncation=True, max_length=c["seq_len"])
            rows_ = [torch.tensor(i) for i in enc["input_ids"]]
            collate = lambda b: self._pad_batch(b, tok.pad_token_id)   # noqa: E731
            ref = _reference_indices([r.numel() for r in rows_], c["effective_batch"])
            self._set_population(enc["input_ids"], ref, torch.utils.data.DataLoader(
                [rows_[i] for i in ref], batch_size=mb, shuffle=False, collate_fn=collate))
            if c["length_grouped"]:
                # Sort by length so each batch's rows are near-equal and pad-to-longest wastes
                # little. Batch ORDER is still shuffled, so the epoch is not length-ordered.
                rows_ = _length_grouped(rows_, mb, seed=c["loader_seed"])
                return torch.utils.data.DataLoader(
                    rows_, batch_size=mb, shuffle=False, drop_last=True, collate_fn=collate)
            return torch.utils.data.DataLoader(
                rows_, batch_size=mb, shuffle=True, drop_last=True, generator=seeded,
                collate_fn=collate)

        enc = tok(texts, truncation=True, max_length=c["seq_len"], padding="max_length")
        input_ids = torch.tensor(enc["input_ids"])
        attention = torch.tensor(enc["attention_mask"])
        labels = input_ids.masked_fill(attention == 0, -100)   # don't train on padding
        real = [[t for t, m in zip(ids, mask) if m]              # by mask, not by slicing: some
                for ids, mask in zip(enc["input_ids"], enc["attention_mask"])]   # tokenizers pad left
        dset = torch.utils.data.TensorDataset(input_ids, attention, labels)
        ref = _reference_indices([len(r) for r in real], c["effective_batch"])
        self._set_population(real, ref, torch.utils.data.DataLoader(
            torch.utils.data.Subset(dset, ref), batch_size=mb, shuffle=False))
        return torch.utils.data.DataLoader(dset, batch_size=mb, shuffle=True, drop_last=True,
                                           generator=seeded)

    def _set_population(self, real_rows, ref, ref_loader):
        """Record which rows the clean window covers and which batch the trace will describe.

        ref_loader serves the reference rows through the same collate as the clean window, so the
        traced step's `dataloader` phase does real fetch-and-pad work. `batch_key` changes exactly
        when its split or padding does.
        """
        import hashlib
        self._reference = ref_loader
        shapes = [list(b[0].shape) for b in ref_loader]
        rows_hash = _rows_hash([real_rows[i] for i in ref])
        self._population = {
            "sample_method": self._sample_meta.get("method"),   # length-stratified | head
            "sample_file": self.cfg["sample_file"],
            "sample_rows": len(real_rows),
            "sample_hash": _rows_hash(real_rows),       # same rows in the clean window?
            "steps_per_pass": self.steps_per_pass,
            "loader_seed": self.cfg["loader_seed"],
            "reference": {
                "row_ids": list(ref),
                "rows_hash": rows_hash,                 # same rows in the trace?
                "real_tokens": sum(len(real_rows[i]) for i in ref),
                "batch_shapes": shapes,
                "batch_key": hashlib.sha1(f"{rows_hash}{shapes}".encode()).hexdigest()[:16],
            },
        }

    def _pad_batch(self, rows, pad_id):
        """Pad a batch to its longest row, rounded up to `pad_multiple`.

        The rounding is what keeps torch.compile viable: raw longest-lengths would present a new
        shape almost every batch and trigger a recompile each time, so lengths are bucketed.
        """
        import torch
        m = self.cfg["pad_multiple"]
        longest = max(r.numel() for r in rows)
        width = min(self.cfg["seq_len"], -(-longest // m) * m)      # round up, never past seq_len
        ids = torch.full((len(rows), width), pad_id, dtype=torch.long)
        att = torch.zeros((len(rows), width), dtype=torch.long)
        for i, r in enumerate(rows):
            n = min(r.numel(), width)
            ids[i, :n], att[i, :n] = r[:n], 1
        return ids, att, ids.masked_fill(att == 0, -100)

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

    @property
    def has_reference(self):
        """True when the traced step runs a fixed reference batch distinct from the loader's."""
        return getattr(self, "_reference", None) is not None

    def start_pass(self):
        """Restart the loader at a pass boundary, so a clean window of whole passes covers every
        row the same number of times whatever warmup consumed. No-op for synthetic data."""
        if self._loader is not None:
            self._it = iter(self._loader)

    def population(self):
        """Which rows this runner's clean window covers and which batch its trace describes
        (see `_set_population`). None for synthetic data: one fixed batch, nothing to identify."""
        return self._population

    def last_step_tokens(self):
        """Tokens the most recent `step()` pushed through the model (padded width x rows)."""
        return self._tokens

    def last_step_samples(self):
        """Rows the most recent `step()` pushed through the model."""
        return self._samples

    def step(self):
        self._tokens = self._samples = 0                   # measured, not assumed (see tokens_per_step)
        self._steps = getattr(self, "_steps", 0) + 1
        self._train_step(self._next, count=True)

    def reference_step(self):
        """One training step on the fixed reference batch (what the profiler traces), kept out of
        the token counts, which describe the clean window only."""
        if self._reference is None:
            return self.step()
        batches = iter(self._reference)                    # fetch + collate run inside the
        self._train_step(lambda: next(batches), count=False)   # `dataloader` annotation

    def _train_step(self, next_batch, count):
        c = self.cfg
        self.opt.zero_grad(set_to_none=True)
        for _ in range(c["grad_accum"]):
            with self._rf("dataloader"):
                input_ids, attention, labels = next_batch()
            if count:
                self._tokens += input_ids.numel()
                self._samples += input_ids.shape[0]
                self._tokens_total = getattr(self, "_tokens_total", 0) + input_ids.numel()
            input_ids = input_ids.to(self.device)
            attention = attention.to(self.device)
            labels = labels.to(self.device)
            with self._rf("forward"):
                loss = self._loss(input_ids, attention, labels) / c["grad_accum"]
            with self._rf("backward"):
                loss.backward()
        with self._rf("optimizer"):
            self.opt.step()

    def _loss(self, input_ids, attention, labels):
        """The config's loss path: HF's own, or the chunked cross-entropy when `chunked_ce` is on."""
        c = self.cfg
        if c["chunked_ce"]:
            return self._chunked_loss(input_ids, attention, labels, c["chunked_ce"])
        return self.model(input_ids=input_ids, attention_mask=attention, labels=labels).loss

    def reference_fingerprint(self):
        """Loss and gradient norm on the fixed reference batch, on the weights as loaded.

        The config's numerical fingerprint, taken once before any step trains the model: one
        forward + backward over the reference rows (`_reference_indices`, batched and padded the
        way this config batches them), the loss weighted to the mean over real tokens, no
        optimizer step, gradients discarded. Same rows for every config in a run, so two configs
        that compute the same function return the same pair, to bf16 kernel-reordering noise. A
        grossly wrong forward moves the loss; a grossly wrong BACKWARD moves the gradient norm
        while the loss stays put -- fla / causal_conv1d ship their own backward kernels. A SMOKE
        CHECK, not verification: two means over thousands of tokens miss edge-case errors
        (loop/throughput.md). Runs the training graph (grad on, `train()` mode), so torch.compile
        traces it once. None for synthetic data.
        """
        import torch
        if self._reference is None:
            return None
        batches = list(self._reference)                 # collate once; the loader is re-iterable
        counts = [int((b[2][:, 1:] != -100).sum()) for b in batches]   # what HF's mean counts
        total = sum(counts)
        if not total:
            return None
        loss_sum = 0.0
        self.opt.zero_grad(set_to_none=True)
        devices = [torch.cuda.current_device()] if self.device == "cuda" else []
        with torch.random.fork_rng(devices=devices):    # dropout, if any, draws the same mask --
            torch.manual_seed(0)                        #   without touching the loader's RNG
            for (ids, att, labels), n in zip(batches, counts):
                ids, att, labels = ids.to(self.device), att.to(self.device), labels.to(self.device)
                loss = self._loss(ids, att, labels)
                loss_sum += loss.item() * n
                (loss * (n / total)).backward()        # grads sum to the token-mean's gradient
        grads = [p.grad for p in self.model.parameters() if p.grad is not None]
        norm = torch.nn.utils.get_total_norm(grads).item()   # one sync, not one per tensor
        self.opt.zero_grad(set_to_none=True)            # leave the model exactly as loaded
        return {"loss": loss_sum / total, "grad_norm": norm}

    def _chunked_loss(self, input_ids, attention, labels, chunks):
        """Causal-LM cross-entropy computed in `chunks` slices of the token axis.

        The full-logits path materializes `[batch, seq, vocab]` twice over -- once in bf16 from the
        lm_head and again as the fp32 copy cross_entropy upcasts to. At vocab 248320 that is the
        single largest activation in the step and it is what caps the OOM ceiling, so slicing it is
        the lever. Each slice runs under `checkpoint`, so its logits are freed once the slice's loss
        is computed and rebuilt during the backward: peak logits memory becomes one chunk's worth
        instead of the whole batch's.

        Arithmetically identical to the unchunked loss, which is why it is a speed lever and not a
        model change: the same shift, the same ignore_index mask, a sum of the same per-token terms
        divided by the same token count. Only the fp32 summation ORDER differs, which is the same
        class of reassociation any fused kernel performs.
        """
        import torch
        from torch.utils.checkpoint import checkpoint
        hidden = self.model.model(input_ids=input_ids,
                                  attention_mask=attention).last_hidden_state
        h = hidden[:, :-1, :].reshape(-1, hidden.size(-1))      # predict t+1 from t
        y = labels[:, 1:].reshape(-1)
        keep = y != -100                                        # drop padding before chunking, so
        h, y = h[keep], y[keep]                                 # chunks are equal in REAL tokens
        n = y.numel()
        if n == 0:                                              # an all-padding batch: keep the
            return hidden.sum() * 0.0                           # graph connected, contribute zero
        head = self.model.lm_head
        size = -(-n // chunks)                                  # ceil, so exactly <= `chunks` slices

        def slice_loss(hs, ys):
            return torch.nn.functional.cross_entropy(head(hs).float(), ys, reduction="sum")

        total = None
        for i in range(0, n, size):
            part = checkpoint(slice_loss, h[i:i + size], y[i:i + size], use_reentrant=False)
            total = part if total is None else total + part
        return total / n                                        # mean over real tokens, as HF does

    def teardown(self):
        """Free this trial's weights/optimizer/data, so a ceiling search's next trial starts on a
        clean card. Without it each trial measures the previous trial's leftovers."""
        self.model = self.opt = self._loader = self._it = self._batch = self._reference = None

    def tokens_per_step(self):
        """Tokens actually pushed through the model per optimizer step.

        MEASURED rather than assumed, because under `padding="longest"` the batch width varies and
        `effective_batch * seq_len` would overstate it -- inflating both tokens/s and MFU exactly
        where the lever is supposed to be judged. Under `padding="max_length"` the measured value
        equals that product, so this is a no-op there.

        It is the MEAN over every step run, not the last step's count. Rates divide by the mean
        step time over many steps, so pairing it with a single step's token count mismatches
        numerator and denominator: two runs of an identical config measured 7490 vs 9449 tokens/s while their
        samples/s agreed to 0.9%. The mean makes the ratio meaningful again.
        """
        steps = getattr(self, "_steps", 0)
        if steps:
            return getattr(self, "_tokens_total", 0) / steps
        return self.cfg["tokens_per_step"]
