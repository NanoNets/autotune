"""CLI: python -m frame --model <hub-id|path> --dataset <hub-id|path> --gpu A10G --max-len 2048

Prints the one-line summary, any notes, and the full Profile JSON. Needs transformers + datasets.

`--max-len` is required: it is the human's declared application sequence length, and it is what
the OOM ceiling is measured at. Without it there is no worst case to measure against.

`--ceiling` measures that ceiling with `--framework`'s real runner — it loads real weights and
runs a real fwd/bwd/opt on worst-case batches, so the number is one to tune against. It costs
~2*log2(n) short trials. Without the flag the Profile carries `ceiling: null`, and the summary
says so rather than implying a ceiling was checked.
"""
import argparse
import dataclasses
import json

from frame.profile import frame


def main():
    ap = argparse.ArgumentParser(prog="frame")
    ap.add_argument("--model", required=True, help="HF hub id or local path")
    ap.add_argument("--dataset", required=True, help="HF hub id or local path")
    ap.add_argument("--gpu", required=True, help="a label for the record (any hardware; not validated)")
    ap.add_argument("--max-len", type=int, required=True,
                    help="the application's sequence length; the OOM ceiling is measured at it")
    ap.add_argument("--dtype", default="bfloat16", help="run compute dtype; the peak is measured at this")
    ap.add_argument("--device", default="cuda", help="device to measure the peak on")
    ap.add_argument("--framework", default="hf", help="which runner's knob surface to report")
    ap.add_argument("--effective-batch", type=int, default=16,
                    help="human target the loop factorizes; also caps the ceiling search")
    ap.add_argument("--text-field", default="text")
    ap.add_argument("--split", default="train")
    ap.add_argument("--sample", type=int, default=1000,
                    help="probe sample size (rounded down to a multiple of --effective-batch)")
    ap.add_argument("--sample-out", default=None,
                    help="write the drawn probe sample here, e.g. runs/<run>/sample.jsonl; "
                         "every probe then loads it via sample_file")
    ap.add_argument("--sample-seed", type=int, default=0, help="seed for the sample's random pool")
    ap.add_argument("--max-scan", type=int, default=None,
                    help="cap the rows streamed for the pool (very large corpora); default: all")
    ap.add_argument("--name", default=None, help="dataset config, e.g. wikitext-2-raw-v1")
    ap.add_argument("--ceiling", action="store_true",
                    help="measure the OOM ceiling at --max-len using --framework's real runner")
    ap.add_argument("--seed", type=int, default=0, help="seed for the synthetic batches")
    a = ap.parse_args()

    runner = None
    if a.ceiling:
        from frame.profile import framework_runner
        runner = framework_runner(a.framework)()
        print(f"frame: measuring the memory ceiling at max_len={a.max_len} "
              f"(synthetic, seed {a.seed}, framework={a.framework})")

    def on_trial(t):
        peak = f"peak {t['peak_gb']}GB" if t["peak_gb"] is not None else "peak n/a"
        print(f"  mb={t['micro_batch']:<5} {t['status']:<5} {peak}")

    p = frame(a.model, a.dataset, a.gpu, a.max_len, dtype=a.dtype, device=a.device,
              framework=a.framework, effective_batch=a.effective_batch,
              text_field=a.text_field, split=a.split, sample=a.sample, name=a.name,
              runner=runner, seed=a.seed, on_trial=on_trial, sample_path=a.sample_out,
              sample_seed=a.sample_seed, max_scan=a.max_scan)
    if a.sample_out:
        print(f"frame: probe sample written to {a.sample_out} (pass it to probes as sample_file)")
    print(p.summary)
    for n in p.notes:
        print(f"NOTE: {n}")
    print(json.dumps(dataclasses.asdict(p), indent=2, default=str))


if __name__ == "__main__":
    main()
