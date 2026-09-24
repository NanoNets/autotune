"""Draw the probe data-partition figure: which rows of one probe's sample each step uses.

The numbers are the Qwen3.5 A100 run's settings (sample=1000, effective_batch=16, micro_batch=4,
warmup=3, clean_steps=30), worked through Frame's draw (`frame.dataset.draw_sample`) and
`profiler/capture.py`'s step order. Illustrative, not read from a run. Regenerate both themes with:

    python profiler/img/partition_fig.py
"""
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#8b8a85", rule="#dddcd6",
                  neutral="#c9c8c1", trimmed="#ecebe6",
                  clean="#2a78d6", warm="#eb6834", ref="#1baf7a"),
    "dark": dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", muted="#8f8e86", rule="#3a3a37",
                 neutral="#55544f", trimmed="#2c2c2a",
                 clean="#3987e5", warm="#d95926", ref="#199e70"),
}

SAMPLE, KEPT, WARM, REF = 992, 992, 48, 16   # sample=1000 is rounded down to 992 at the draw
POOL = 20 * KEPT
GAP = 2.0          # rows of surface gap between adjacent fills (~2px at this scale)


def seg(ax, x0, width, y, h, color, t, hatch=None, edge=None, fill=True):
    ax.add_patch(FancyBboxPatch((x0, y), width, h, boxstyle="round,pad=0,rounding_size=3",
                                mutation_aspect=h / 30, linewidth=1.2 if edge else 0,
                                facecolor=color if fill else "none",
                                edgecolor=edge or color, hatch=hatch))


def draw(mode, out):
    t = THEMES[mode]
    plt.rcParams.update({"font.family": "DejaVu Sans", "hatch.color": t["muted"], "hatch.linewidth": 0.8})
    fig, ax = plt.subplots(figsize=(12, 5.6), dpi=160)
    fig.patch.set_facecolor(t["surface"]); ax.set_facecolor(t["surface"])

    lanes = [  # (y, label, sublabel)
        (5, "Probe sample", "drawn once by Frame"),
        (4, "5 · clean window", "profiler off"),
        (3, "2 · warmup", "discarded"),
        (2, "3 · 7 · 8 · reference batch", "warm · timed · traced"),
        (1, "9 · population", "hashed, not run"),
    ]
    h = 0.52
    for y, label, sub in lanes:
        ax.text(-18, y + 0.08, label, ha="right", va="center", fontsize=11, color=t["ink"], weight="bold")
        ax.text(-18, y - 0.2, sub, ha="right", va="center", fontsize=9, color=t["ink2"])

    # Probe sample: 992 rows, length-stratified from a random pool over the whole split
    seg(ax, 0, KEPT, 5 - h / 2, h, t["neutral"], t)
    ax.text(KEPT / 2, 5, f"992 rows  ·  length-stratified from a {POOL:,}-row random pool  ·  "
            "62 steps × 16 per pass", ha="center", va="center", fontsize=10, color=t["ink"])
    ax.text(KEPT, 5.42, "sample=1000 rounded down to a multiple of 16", ha="right", va="bottom",
            fontsize=9, color=t["ink2"])

    # Clean window: every kept row, once
    seg(ax, 0, KEPT - GAP / 2, 4 - h / 2, h, t["clean"], t)
    ax.text(KEPT / 2, 4, "all 992 sampled rows, each timed exactly once  ·  clean_steps 30 → 62",
            ha="center", va="center", fontsize=10, color="#ffffff", weight="bold")

    # Warmup: 48 rows
    seg(ax, 0, WARM, 3 - h / 2, h, t["warm"], t)
    ax.text(WARM + 12, 3, "48 rows (4.8%)  ·  3 steps × 16  ·  whichever rows the loader yields first",
            ha="left", va="center", fontsize=10, color=t["ink"])

    # Reference: 16 rows
    seg(ax, 0, REF, 2 - h / 2, h, t["ref"], t)
    ax.text(REF + 12, 2, "16 rows (1.6%)  ·  nearest the sample's median length  ·  the same 16 rows run 3 times",
            ha="left", va="center", fontsize=10, color=t["ink"])

    # Population: two hashed sets, outline only
    seg(ax, 0, KEPT - GAP / 2, 1 - h / 2, h, t["surface"], t, edge=t["muted"], fill=True)
    seg(ax, 0, REF, 1 - h / 2, h, t["ref"], t)
    ax.text(REF + 12, 1, "sample_hash over the 992 rows  ·  batch_key over the 16 reference rows + their shapes",
            ha="left", va="center", fontsize=10, color=t["ink"])

    # Axis: recessive, rows of the loaded sample
    ax.set_xlim(-328, SAMPLE + 10); ax.set_ylim(0.2, 6.35)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["rule"]); ax.spines["bottom"].set_bounds(0, SAMPLE)
    ax.set_yticks([])
    ax.set_xticks([0, 248, 496, 744, 992])
    ax.tick_params(axis="x", colors=t["muted"], labelsize=9, length=3)
    ax.set_xlabel("rows of the probe sample (width is proportional to row count)", color=t["ink2"], fontsize=9.5)

    fig.text(0.02, 0.955, "One probe: which rows each step uses", fontsize=15, weight="bold", color=t["ink"])
    fig.text(0.02, 0.905, "Example: Qwen run settings  sample=1000 · effective_batch=16 · micro_batch=4 · "
             "warmup=3 · clean_steps=30", fontsize=10, color=t["ink2"])
    fig.text(0.02, 0.06, "A pass is one pass over the 992-row probe sample, not the corpus. Steps 1, 4, 6 and 10 "
             "read no rows.", fontsize=9, color=t["ink2"])
    fig.text(0.02, 0.025, "Warmup and reference rows are drawn grouped at the left for scale; in the sample "
             "they are spread out, and they are also part of the clean window.", fontsize=9, color=t["ink2"])
    fig.subplots_adjust(left=0.02, right=0.99, top=0.86, bottom=0.2)
    fig.savefig(out, facecolor=t["surface"])


if __name__ == "__main__":
    import os
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
    draw("light", f"{d}/probe_data_partition.png")
    draw("dark", f"{d}/probe_data_partition_dark.png")
