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
import json
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

# Staircase steps are labelled with the knobs that actually changed, DERIVED per run by diffing
# each record against the one before it. This used to be a dict keyed by experiment number, which
# was silently wrong for every run but the one it was written for -- it labelled another run's
# experiment 3 "torch_compile" when that run had changed micro_batch. A per-run `annotations.json`
# can still add the hand-written regime notes no diff can infer:
#   {"subtitle": "...", "regimes": [[8.5, "dynamic padding -> launch-bound"], ...]}


def _knobs(hypothesis):
    """The `[k=v k=v ...]` snapshot the probe harness prefixes to each hypothesis, as a dict.

    Older runs have no prefix, in which case there is nothing to diff and the step goes unlabelled
    -- which is the honest outcome, and strictly better than captioning it with a guess.
    """
    if not hypothesis.startswith("[") or "]" not in hypothesis:
        return {}
    out = {}
    for tok in hypothesis[1:hypothesis.index("]")].split():
        if "=" in tok:
            k, v = tok.split("=", 1)
            out[k] = v
    return out


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
                # A reading the run rejected (e.g. MFU > 100%) is not allowed to set a record. It is
                # still drawn -- a discarded measurement is part of the history -- but the staircase
                # must not climb onto a number that did not apply. Any status but `ok` is one: an
                # `invalid` Observation keeps its samples/s, and must not become the headline.
                "unreliable": "UNRELIABLE" in concl or r["status"] != "ok",
                "hyp": r.get("hypothesis") or "",
                "knobs": _knobs(r.get("hypothesis") or ""),
            })
    return rows


def mark_records(rows, basis=None):
    """Mark each row's `superseded` / `record` / `best_so_far`; return (baseline, records).

    Only an `ok`, not-UNRELIABLE, not-superseded row can be the baseline or set a record.
    `basis` is annotations.json's `records_from`."""
    for r in rows:
        r["superseded"] = bool(basis and r["n"] < basis and not r["unreliable"]
                               and r["n"] != rows[0]["n"])

    eligible = [r for r in rows if r["sps"] and not r["unreliable"] and not r["superseded"]]
    baseline = eligible[0]["sps"] if eligible else rows[0]["sps"]
    best, records = 0.0, []                            # records: the staircase steps
    for r in rows:
        r["record"] = bool(r["sps"] and r["sps"] > best
                           and not r["unreliable"] and not r["superseded"])
        if r["record"]:
            best = r["sps"]
            records.append(r)
        r["best_so_far"] = best
    return baseline, records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", default="test_a10G")
    ap.add_argument("-o", "--out", default=None)
    a = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    rows = load(os.path.join(here, a.run, "results.tsv"))
    out = a.out or os.path.join(here, a.run, "progress.png")

    profile = {}
    prof_path = os.path.join(here, a.run, "profile.json")
    if os.path.exists(prof_path):
        profile = json.load(open(prof_path))
    notes = {}
    ann_path = os.path.join(here, a.run, "annotations.json")
    if os.path.exists(ann_path):
        notes = json.load(open(ann_path))

    # A run whose MEASUREMENT changed mid-flight cannot share one staircase: comparing a probe
    # scored one way against a probe scored another ranks the methodology, not the config. When
    # annotations.json declares `records_from`, earlier probes are still drawn -- they are the real
    # history of the search -- but they cannot set a record, and the ratio is computed on the
    # comparable basis only.
    baseline, records = mark_records(rows, notes.get("records_from"))
    best = rows[-1]["best_so_far"]

    # Everything the figure says about the hardware and the job comes from the run's own Profile,
    # so a plot can no longer describe a different run's card than the one it was measured on.
    hw, mdl = profile.get("hardware", {}), profile.get("model", {})
    card_gb = hw.get("mem_gb") or (max((r["mem"] or 0) for r in rows) * 1.25) or 24.0
    subtitle = notes.get("subtitle") or " · ".join(str(x) for x in [
        (profile.get("model", {}).get("arch") or a.run),
        f"{mdl['params'] / 1e9:.1f}B params" if mdl.get("params") else None,
        hw.get("device_name"), f"effective_batch {profile['effective_batch']}"
        if profile.get("effective_batch") else None,
        f"max_len {profile['max_len']}" if profile.get("max_len") else None] if x)

    ok = [r["sps"] for r in rows if r["sps"]]
    y_lo, y_hi = min(min(ok), baseline) * 0.55, max(ok) * 1.55

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
    x_hi = max(xs) + 0.9
    ax.text(x_hi - 0.3, baseline * 1.1, f"baseline {baseline:.2f}", va="bottom", ha="right",
            fontsize=8.5, color=INK_3, family="monospace")

    for r in rows:
        if not r["ok"]:
            ax.plot(r["n"], y_lo * 1.5, marker="x", ms=7, mew=2, color=FAILED, zorder=4)
        elif r["unreliable"]:                           # drawn, but never on the staircase
            ax.plot(r["n"], r["sps"], marker="s", ms=6.5, mfc="white",
                    mec=FAILED, mew=1.6, zorder=4)
        elif r["superseded"]:                           # measured on a basis since corrected
            ax.plot(r["n"], r["sps"], marker="^", ms=6.5, mfc="white",
                    mec=INK_3, mew=1.4, zorder=3)
        elif r["record"]:
            ax.plot(r["n"], r["sps"], marker="o", ms=9, color=RECORD,
                    mec="white", mew=1.6, zorder=5)
        elif r["control"]:                              # hollow = a control, not a new idea
            ax.plot(r["n"], r["sps"], marker="o", ms=7.5, mfc="white",
                    mec=RAN, mew=1.8, zorder=4)
        else:
            ax.plot(r["n"], r["sps"], marker="o", ms=7.5, color=RAN,
                    mec="white", mew=1.2, zorder=4)

    # A run whose ledger predates the knob snapshot has nothing to diff, so annotations.json can
    # supply the captions directly: {"levers": {"3": "torch_compile", ...}}.
    levers = {int(k): v for k, v in notes.get("levers", {}).items()}
    mid = (min(xs) + max(xs)) / 2
    for i, r in enumerate(records):
        if r["n"] in levers:
            ax.annotate(f"{levers[r['n']]}\n{r['sps']:.1f}",
                        xy=(r["n"], r["sps"]), xytext=(-6, 15), textcoords="offset points",
                        fontsize=8.6, color=INK, ha="right" if r["n"] > mid else "left",
                        linespacing=1.35,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, lw=0.8))
            continue
        if i == 0:                                      # the baseline diffs against nothing
            continue
        prev = records[i - 1]["knobs"]
        changed = [f"{k}={v}" for k, v in r["knobs"].items() if prev.get(k) != v]
        if not changed or len(changed) > 2:              # a big jump is a config dump, not a label
            continue
        ax.annotate("\n".join(changed) + f"\n{r['sps']:.1f}",
                    xy=(r["n"], r["sps"]), xytext=(-6, 15), textcoords="offset points",
                    fontsize=8.0, color=INK, ha="right" if r["n"] > mid else "left",
                    linespacing=1.35,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID, lw=0.8))

    ax.set_yscale("log")
    ax.set_ylim(y_lo, y_hi)
    ticks = [t for t in (1, 2, 4, 10, 25, 50, 100, 250, 500) if y_lo < t < y_hi]
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(t) for t in ticks])
    ax.set_ylabel("samples / second   (log)", fontsize=10, color=INK_2, labelpad=9)
    ax.grid(axis="y", color=GRID, lw=0.9, zorder=0)
    ax.set_axisbelow(True)

    # Regime boundaries — where the binding constraint changed, not just the number. These cannot
    # be inferred from a knob diff, so they come from the run's own annotations.json or not at all.
    for x, label in notes.get("regimes", []):
        ax.axvline(x, color=INK_3, lw=1, ls=(0, (2, 3)), zorder=1)
        right = x > (min(xs) + max(xs)) / 2
        ax.text(x + (-0.35 if right else 0.35), y_lo * 1.02, label,
                fontsize=8.4, color=INK_2, family="monospace",
                ha="right" if right else "left", va="bottom")

    ax.set_title(
        f"autotune · {a.run} — {baseline:.2f} → {best:.1f} samples/s  ({best / baseline:.1f}×)",
        fontsize=13.5, color=INK, pad=16, loc="left", fontweight="bold")
    ax.text(0, 1.012, subtitle, transform=ax.transAxes,
            fontsize=8.8, color=INK_3, family="monospace")

    ax.legend(handles=[
        Line2D([], [], marker="o", ls="none", ms=8, color=RECORD, mec="white",
               label="new best (staircase steps)"),
        Line2D([], [], marker="o", ls="none", ms=7, color=RAN, mec="white",
               label="ran, did not take the lead"),
        Line2D([], [], marker="o", ls="none", ms=7, mfc="white", mec=RAN, mew=1.8,
               label="control / repeat"),
        Line2D([], [], marker="s", ls="none", ms=6, mfc="white", mec=FAILED, mew=1.6,
               label="discarded (measurement did not apply)"),
        Line2D([], [], marker="^", ls="none", ms=6, mfc="white", mec=INK_3, mew=1.4,
               label="superseded measurement basis"),
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
    bx.axhline(card_gb, color=FAILED, lw=1, ls=(0, (4, 3)), zorder=1)
    bx.text(min(xs), card_gb * 1.035, f"card limit {card_gb:.1f} GB", fontsize=8,
            color=FAILED, family="monospace")

    bx.set_ylim(0, card_gb * 1.22)
    bx.set_yticks([t for t in (0, 10, 20, 30, 40, 60, 80) if t <= card_gb * 1.1])
    bx.set_ylabel("peak memory\nGB", fontsize=9.5, color=INK_2, labelpad=9)
    bx.set_xlabel("experiment", fontsize=10, color=INK_2, labelpad=7)
    bx.set_xticks(xs if len(xs) <= 25 else [n for n in xs if n % 2 or n == min(xs)])
    bx.set_xlim(min(xs) - 0.7, x_hi)
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
    n_sup = sum(r["superseded"] for r in rows)
    n_bad = sum(r["unreliable"] for r in rows)
    print(f"baseline {baseline:.3f} -> best {best:.3f} samples/s ({best / baseline:.1f}x), "
          f"{len(records)} records over {len(rows)} experiments"
          + (f" ({n_sup} on a superseded basis, {n_bad} discarded, excluded from the staircase)"
             if n_sup or n_bad else ""))


if __name__ == "__main__":
    main()
