"""The data file a user's script reads during a ceiling trial: `runs/<run>/ceiling_rows.jsonl`.

A user's script has no `data="synthetic"` switch, so the worst-case batch reaches it as DATA: rows
that each encode to more than `max_len` tokens with the model's tokenizer, so the script's own
truncation cuts every one to exactly `max_len`, and its own template, collator and DataLoader build
the batch in exactly the format the model sees. The agent points the script's data setting at the
file (a flag, a config key, or an edited copy when it's hardcoded; `program.md` Step 2a), and every
trial checks the batch really was `[micro_batch, max_len]` (`profiler/memtrial.py`).

Activation memory is a function of SHAPE, not content, so the text is random tokens decoded to a
string. Special tokens are excluded: a stray EOS or chat-template marker could make a collator
split or end a row early.

Formats:
- `text` (default): `{<text_field>: "..."}` per line, or the agent's one-row `template` with the
  long text in place of the string `"{TEXT}"` (chat `messages`, prompt / completion, extra
  columns). Plain JSON lines, no header: `load_dataset("json", data_files=...)` reads it.
- `ids`: `{"input_ids": [...]}`, exactly `max_len` ids, for scripts that read pre-tokenized rows.
"""
import json
import random

PLACEHOLDER = "{TEXT}"


def _fill(obj, text):
    if isinstance(obj, str):
        return obj.replace(PLACEHOLDER, text)
    if isinstance(obj, list):
        return [_fill(x, text) for x in obj]
    if isinstance(obj, dict):
        return {k: _fill(v, text) for k, v in obj.items()}
    return obj


def _ordinary_ids(tokenizer):
    special = set(getattr(tokenizer, "all_special_ids", None) or [])
    size = len(tokenizer)
    return [i for i in range(size) if i not in special]


def long_text(tokenizer, max_len, rng, pool, margin=None):
    """A string that encodes to at least `max_len + margin` tokens (no special tokens added)."""
    margin = margin if margin is not None else max(16, max_len // 8)
    want, ids = max_len + margin, []
    text = ""
    for _ in range(8):                                 # decode/encode isn't an exact round trip
        ids += [rng.choice(pool) for _ in range(want - len(ids) + margin)]
        text = tokenizer.decode(ids, skip_special_tokens=True)
        if len(tokenizer.encode(text, add_special_tokens=False)) >= want:
            return text
    raise ValueError(f"couldn't build a row encoding to {want} tokens")


def write_ceiling_rows(path, tokenizer, max_len, n_rows, text_field="text", template=None,
                       fmt="text", seed=0):
    """Write `n_rows` over-length rows to `path`. `template`: a dict (or JSON string) holding
    `"{TEXT}"` where the long text goes. Returns the number of rows written."""
    if fmt not in ("text", "ids"):
        raise ValueError(f"fmt must be 'text' or 'ids', got {fmt!r}")
    if isinstance(template, str):
        template = json.loads(template)
    if template is not None and PLACEHOLDER not in json.dumps(template):
        raise ValueError(f"the row template needs a {PLACEHOLDER!r} where the long text goes")
    rng, pool = random.Random(seed), _ordinary_ids(tokenizer)
    with open(path, "w") as f:
        for _ in range(n_rows):
            if fmt == "ids":
                row = {"input_ids": [rng.choice(pool) for _ in range(max_len)]}
            else:
                text = long_text(tokenizer, max_len, rng, pool)
                row = _fill(template, text) if template is not None else {text_field: text}
            f.write(json.dumps(row) + "\n")
    return n_rows
