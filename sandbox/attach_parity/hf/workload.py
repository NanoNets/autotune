"""The one workload every experiment-4 arm runs, on random token ids. Shared so L, T and R build
the identical model and data.

Two ways to get the model:
  - built from config (4a): a small Llama, random init, no download.
  - `from_pretrained` (4b): set `model` (hub id) + `revision` (exact commit). Called the way a
    user typically calls it: no `dtype` argument. In transformers 5.15.1 that means
    dtype="auto" (modeling_utils.py:4136), i.e. the CHECKPOINT's dtype, so a bf16 checkpoint
    trains pure bf16, unlike the config-built model's fp32 weights under autocast.

LoRA (experiment 7): set `lora` to {"r", "alpha", "dropout"}; the adapter goes on every attention
and MLP projection (`LORA_TARGETS`) via peft, after the base model is built.

Config keys: model, revision, lora, hidden, intermediate, layers, heads, vocab, seq, micro_batch,
grad_accum, samples, workers, seed. With `model` set, the architecture keys are ignored and
`vocab` comes from the checkpoint's config.
"""
import torch
from torch.utils.data import Dataset

PRETRAINED = {                                    # hub id -> exact commit, checked at load time
    "HuggingFaceTB/SmolLM2-360M": "f8027fd0eaeea54caa13c31d31b9fdc459c38b49",
    "Qwen/Qwen2.5-0.5B": "060db6499f32faf8b98477b0a26969ef7d8b9987",
}

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

DEFAULTS = {"model": None, "revision": None, "lora": None, "hidden": 1024, "intermediate": 2816, "layers": 8, "heads": 16, "vocab": 32000,
            "seq": 512, "micro_batch": 8, "grad_accum": 4, "samples": 1024, "workers": 0,
            "seed": 0}


def resolve(cfg):
    c = {**DEFAULTS, **cfg}
    if c["model"]:
        pinned = PRETRAINED.get(c["model"])
        if pinned is None or c["revision"] not in (None, pinned):
            raise SystemExit(f"{c['model']}@{c['revision']} is not a pinned checkpoint: {PRETRAINED}")
        c["revision"] = pinned
        from transformers import AutoConfig
        c["vocab"] = AutoConfig.from_pretrained(c["model"], revision=c["revision"]).vocab_size
    return c


def build_model(cfg):
    c = resolve(cfg)
    torch.manual_seed(c["seed"])
    if c["model"]:                                 # the user's usual path: no dtype argument
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(c["model"], revision=c["revision"],
                                                     attn_implementation="sdpa")
        return _with_lora(model, c["lora"])
    from transformers import LlamaConfig, LlamaForCausalLM
    conf = LlamaConfig(hidden_size=c["hidden"], intermediate_size=c["intermediate"],
                       num_hidden_layers=c["layers"], num_attention_heads=c["heads"],
                       num_key_value_heads=c["heads"], vocab_size=c["vocab"],
                       max_position_embeddings=c["seq"], attn_implementation="sdpa")
    return LlamaForCausalLM(conf)


class RandomTokens(Dataset):
    """Fixed random ids; labels = input_ids (causal LM); no padding, so the mask is all ones."""

    def __init__(self, cfg):
        c = resolve(cfg)
        g = torch.Generator().manual_seed(c["seed"])
        self.ids = torch.randint(0, c["vocab"], (c["samples"], c["seq"]), generator=g)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        x = self.ids[i]
        return {"input_ids": x, "attention_mask": torch.ones_like(x), "labels": x.clone()}


def _with_lora(model, lora):
    if not lora:
        return model
    from peft import LoraConfig, get_peft_model
    return get_peft_model(model, LoraConfig(r=lora["r"], lora_alpha=lora["alpha"],
                                            lora_dropout=lora["dropout"], bias="none",
                                            target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))


def truth(cfg, model):
    """`model.parameters()` yields a tied weight once, the same as the optimizer holds it."""
    c = resolve(cfg)
    return {"params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in model.parameters()),
            "tokens": c["grad_accum"] * c["micro_batch"] * c["seq"],
            "grad_accum": c["grad_accum"]}


def describe(model):
    """What actually loaded: class, dtype, tied embeddings, and the revision on the config."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    return {"class": type(model).__name__, "base_class": type(base).__name__,
            "dtype": str(base.dtype).replace("torch.", ""),
            "tied_embeddings": base.get_input_embeddings().weight is base.get_output_embeddings().weight,
            "attn_implementation": base.config._attn_implementation,
            "trainable_params": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "total_params": sum(p.numel() for p in model.parameters())}
