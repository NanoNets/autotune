"""Run ONE arm in a fresh process and print a JSON record.

  A  baseline: capture() drives frameworks/torch.py (record_function labels)
  C  attach:   the bare runner's loop runs on its own; Attach's hooks find steps + labels

Each arm gets its own process so the allocator, class-level optimizer patching and module
hooks can't leak between arms.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)                                          # the sandbox, and nothing else

from vendor.diagnostic import diagnose
from vendor.trace import load_trace


def fingerprint(trace_path):
    """What got recorded, independent of timing: annotation names, CPU-op and kernel counts."""
    t = load_trace(trace_path)
    ann, cpu_ops, kern = {}, {}, {}
    for e in t.get("traceEvents", []):
        cat, name = e.get("cat"), e.get("name")
        if cat == "user_annotation":
            ann[name] = ann.get(name, 0) + 1
        elif cat == "cpu_op":
            cpu_ops[name] = cpu_ops.get(name, 0) + 1
        elif cat in ("kernel", "gpu_memcpy", "gpu_memset"):
            kern[name] = kern.get(name, 0) + 1
    return {"annotations": ann, "cpu_ops": cpu_ops, "kernels": kern}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["A", "C", "L"], required=True)
    ap.add_argument("--runner", choices=["mlp", "loader"], default="mlp",
                    help="mlp = experiment 1 (vendored torch runner); loader = experiment 2")
    ap.add_argument("--config", required=True, help="JSON runner config")
    ap.add_argument("--peak-flops", type=float, required=True)
    ap.add_argument("--peak-bw", type=float, required=True)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--clean", type=int, default=10)
    a = ap.parse_args()
    cfg = json.loads(a.config)
    dev = cfg.get("device", "cpu")

    if a.arm == "A":
        from vendor.capture import capture
        if a.runner == "mlp":
            from vendor.torch_runner import TorchRunner as Runner
        else:
            from loader_runner import LabelledLoaderRunner as Runner
        r = Runner()
        cap = capture(r, cfg, warmup=a.warmup, clean_steps=a.clean,
                      peak_flops=a.peak_flops, peak_mem_bw_gb_s=a.peak_bw)
        params, tokens = r.params, r.tokens_per_step()
        truth = {"params": r.params, "tokens": r.tokens_per_step(),
                 "grad_accum": getattr(r, "grad_accum", 1)}
        micro = None
    elif a.arm == "L":
        import subprocess
        import tempfile
        from vendor.capture import Capture
        if a.runner != "loader":
            raise SystemExit("arm L runs train.py, which is the loader workload")
        out = os.path.join(tempfile.mkdtemp(), "cap.json")
        steps = a.warmup + a.clean + 3                  # a probe's max_steps: a little slack
        cmd = [sys.executable, os.path.join(HERE, "launcher.py"), "--out", out, "--device", dev,
               "--warmup", str(a.warmup), "--clean", str(a.clean),
               "--peak-flops", str(a.peak_flops), "--peak-bw", str(a.peak_bw)]
        if cfg.get("dtype"):
            cmd += ["--dtype", cfg["dtype"]]
        cmd += ["--", os.path.join(HERE, "train.py"), "--config", a.config, "--steps", str(steps)]
        p = subprocess.run(cmd, capture_output=True, text=True, cwd=HERE)
        if p.returncode != 0 or not os.path.exists(out):
            raise SystemExit(f"launcher failed ({p.returncode}):\n{p.stderr[-3000:]}")
        got = json.load(open(out))
        cap = Capture(**got["capture"])
        if cap.status != "ok":
            raise SystemExit(f"launcher capture: {cap.error}")
        params, tokens, micro = got["params"], got["tokens"], got["micro_batches"]
        d, layers = cfg.get("hidden", 256), cfg.get("layers", 4)
        truth = {"params": layers * (d * d + d),
                 "tokens": cfg.get("grad_accum", 1) * cfg.get("micro_batch", 8) * cfg.get("seq", 128),
                 "grad_accum": cfg.get("grad_accum", 1)}
        r = None
    else:
        from attach import Attach
        if a.runner == "mlp":
            from bare_runner import BareTorchRunner
        else:
            from loader_runner import BareLoaderRunner as BareTorchRunner
        att = Attach(device=dev, warmup=a.warmup, clean_steps=a.clean, dtype=cfg.get("dtype"),
                     peak_flops=a.peak_flops, peak_mem_bw_gb_s=a.peak_bw)
        att.install()
        r = BareTorchRunner()
        r.setup(cfg)
        guard = a.warmup + a.clean + 5                  # the user's loop: runs until we're done
        for _ in range(guard):
            if att.done:
                break
            r.step()
        cap = att.result()
        if cap is None:
            print(json.dumps({"arm": "C", "error": "attach never finished"}))
            return
        params, tokens = att.params, att.tokens_per_step()
        truth = {"params": r.params, "tokens": r.tokens_per_step(),
                 "grad_accum": getattr(r, "grad_accum", 1)}
        micro = att.micro_batches

    obs = diagnose(cap, params=params, tokens=tokens, gpu="sandbox")
    if r is not None and hasattr(r, "teardown"):                          # stop persistent DataLoader workers
        r.teardown()
    rec = {"arm": a.arm, "obs": obs.__dict__,
           "recovered": {"params": params, "tokens": tokens, "micro_batches": micro},
           "truth": truth, "step_times_s": cap.step_times_s,
           "fingerprint": fingerprint(cap.trace) if cap.status == "ok" else None}
    leaked = sorted(m for m in sys.modules if m.split(".")[0] in ("profiler", "frameworks", "frame"))
    if leaked:                                          # isolation is the point: fail loudly
        raise SystemExit(f"live repo modules imported: {leaked}")
    print(json.dumps(rec, default=str))


if __name__ == "__main__":
    main()
