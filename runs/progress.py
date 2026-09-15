"""Progress plot for one run — the experiment ladder and the running best.

Reads `runs/<run>/results.tsv` and draws what the loop actually did: every probe as a point, and
the monotone best-so-far as a staircase. The staircase is the real story — it only steps on an
experiment that beat everything before it, so the flat stretches show where the search was spending
probes without buying anything.

    python runs/progress.py [run_name] [-o out.png]

Single light surface on purpose: this is a static PNG, so there is no theme to switch and the
two-hue palette is validated against a light ground.
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.lines import Line2D                   # noqa: E402

# Two categorical hues + a reserved status red. Validated: CVD dE 29.5 (protan) / 26.8 (tritan),
# normal-vision 33.8, all >= 3:1 on a light surface. Controls are separated by MARKER, not by a
# third hue — a grey slot reads as no identity at all.
RECORD = "#2f5fd0"      # a probe that beat every probe before it
RAN = "#c8761f"         # ran fine, did not take the lead
FAILED = "#b23b3b"      # crash / timeout — status, never reused as a series
INK = "#161a22"
INK_2 = "#4a5263"
INK_3 = "#8a93a3"
GRID = "#e2e6ec"

# What each record-setting probe actually changed. Keyed by experiment number.
LEVERS = {
    3:  "torch_compile",
    6:  "flash_attention_2",
    9:  "padding=longest",
    11: "back to sdpa",
    15: "length grouping",
    18: "pad_multiple 64→8",
}


def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f, delimiter="\t"):
            concl = (r.get("conclusion") or "").upper()
            rows.append({
                "n": int(r["experiment_number"]),
                "sps": float(r["samples/s"]) if r["samples/s"] else None,
                "mem": float(r["memory"].replace("GB", "")) if r["memory"] else None,
                "ok": r["status"] == "ok",
                "control": "CONTROL" in concl,          # a repeat/isolation probe, not a proposal
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="test_a10G")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    rows = load(os.path.join(here, a.run, "results.tsv"))
    out = a.out or os.path.join(here, a.run, "progress.png")

    baseline = rows[0]["sps"]
    best, records = 0.0, []                            # records: the staircase steps
    for r in rows:
        r["record"] = bool(r["sps"] and r["sps"] > best)
        if r["record"]:
            best = r["sps"]
            records.append(r)
        r["best_so_far"] = best

    fig, (ax, bx) = plt.subplots(
        2, 1, figsize=(11.5, 8.2), sharex=True,
        gridspec_kw={"height_ratios": [3.1, 1], "hspace": 0.12})
    fig.patch.set_facecolor("white")

    # ---- panel A: throughput ------------------------------------------------
    ax.set_facecolor("white")
    xs = [r["n"] for r in rows]

    ax.step([r["n"] for r in rows], [r["best_so_far"] for r in rows], where="post",
            color=RECORD, lw=2, zorder=3, solid_joinstyle="miter")
    ax.axhline(baseline, color=INK_3, lw=1, ls=(0, (5, 4)), zorder=1)
    ax.text(23.6, baseline * 1.1, f"baseline {baseline:.2f}", va="bottom", ha="right",
            fontsize=8.5, color=INK_3, family="monospace")

    for r in rows:
        if not r["ok"]:
            ax.plot(r["n"], 3.1, marker="x", ms=7, mew=2, color=FAILED, zorder=4)
        elif r["record"]:
            ax.plot(r["n"], r["sps"], marker="o", ms=9, color=RECORD,
                    mec="white", mew=1.6, zorder=5)
        elif r["control"]:                              # hollow = a control, not a new idea
            ax.plot(r["n"], r["sps"], marker="o", ms=7.5, mfc="white",
                    mec=RAN, mew=1.8, zorder=4)
        else:
            ax.plot(r["n"], r["sps"], marker="o", ms=7.5, color=RAN,
                    mec="white", mew=1.2, zorder=4)

    for r in records:
        if r["n"] in LEVERS:
            ax.annotate(f"{LEVERS[r['n']]}\n{r['sps']:.1f}",
                        xy=(r["n"], r["sps"]), xytext=(-6, 15), textcoords="offset points",
                        fontsize=8.6, color=INK, ha="right" if r["n"] > 8 else "left",
                        linespacing=1.35,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, lw=0.8))

    ax.set_yscale("log")
    ax.set_ylim(2.6, 260)
    ax.set_yticks([4, 10, 25, 50, 100])
    ax.set_yticklabels(["4", "10", "25", "50", "100"])
    ax.set_ylabel("samples / second   (log)", fontsize=10, color=INK_2, labelpad=9)
    ax.grid(axis="y", color=GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)

    # The two regime boundaries — where the binding constraint changed, not just the number.
    for x, label in ((8.5, "dynamic padding → launch-bound"),):
        ax.axvline(x, color=INK_3, lw=1, ls=(0, (2, 3)), zorder=1)
        ax.text(x + 0.25, 190, label, fontsize=8.4, color=INK_2, family="monospace")
    ax.text(8.25, 190, "compute-bound", fontsize=8.4, color=INK_2,
            family="monospace", ha="right")

    ax.set_title(
        f"autotune · {a.run} — {baseline:.2f} → {best:.1f} samples/s  ({best / baseline:.1f}×)",
        fontsize=13.5, color=INK, pad=16, loc="left", fontweight="bold")
    ax.text(0, 1.012, "Qwen2.5-0.5B · wikitext-2 · 1× NVIDIA A10 · effective_batch 16 · max_len 2048",
            transform=ax.transAxes, fontsize=8.8, color=INK_3, family="monospace")

    ax.legend(handles=[
        Line2D([], [], marker="o", ls="none", ms=8, color=RECORD, mec="white",
               label="new best (staircase steps)"),
        Line2D([], [], marker="o", ls="none", ms=7, color=RAN, mec="white",
               label="ran, did not take the lead"),
        Line2D([], [], marker="o", ls="none", ms=7, mfc="white", mec=RAN, mew=1.8,
               label="control / repeat"),
        Line2D([], [], marker="x", ls="none", ms=7, mew=2, color=FAILED,
               label="crash or timeout"),
    ], loc="upper left", frameon=True, framealpha=1, edgecolor=GRID,
        fontsize=8.6, labelcolor=INK_2, borderpad=0.7)

    # ---- panel B: memory ----------------------------------------------------
    # Its own panel, its own axis — never a second y-scale on panel A.
    bx.set_facecolor("white")
    mem_x = [r["n"] for r in rows if r["mem"]]
    mem_y = [r["mem"] for r in rows if r["mem"]]
    bx.bar(mem_x, mem_y, width=0.62, color=RAN, alpha=0.55, zorder=2)
    bx.axhline(22.1, color=FAILED, lw=1, ls=(0, (4, 3)), zorder=1)
    bx.text(1, 22.9, "card limit 22.1 GB", fontsize=8, color=FAILED, family="monospace")

    bx.set_ylim(0, 27)
    bx.set_yticks([0, 10, 20])
    bx.set_ylabel("peak memory\nGB", fontsize=9.5, color=INK_2, labelpad=9)
    bx.set_xlabel("experiment", fontsize=10, color=INK_2, labelpad=7)
    bx.set_xticks(xs)
    bx.set_xlim(0.3, 23.9)
    bx.grid(axis="y", color=GRID, lw=0.9, zorder=0)
    bx.set_axisbelow(True)

    for axis in (ax, bx):
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            axis.spines[side].set_color(GRID)
        axis.tick_params(colors=INK_3, labelsize=8.8, length=3)

    fig.subplots_adjust(left=0.085, right=0.975, top=0.885, bottom=0.085)
    fig.savefig(out, dpi=200, facecolor="white")
    print(f"wrote {out}")
    print(f"baseline {baseline:.3f} -> best {best:.3f} samples/s ({best / baseline:.1f}x), "
          f"{len(records)} records over {len(rows)} experiments")


if __name__ == "__main__":
    main()
