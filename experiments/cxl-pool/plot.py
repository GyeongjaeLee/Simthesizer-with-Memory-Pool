#!/usr/bin/env python3
"""Figures from session-rate sweeps; sweeps of one scenario merge into one line.

ttft:   offered call QPS per worker against TTFT statistics; a dashed line
        marks the first saturated rate.
parts:  mean TTFT split into its parts at low, middle and highest complete load.
reuse:  share of prompt tokens restored from cached KV instead of prefilled.
queue:  waiting requests per worker over time at each scenario's highest rate
        (mean, and the range across workers).
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
STATS = {"mean": "Mean", "p50": "p50", "p90": "p90", "p99": "p99"}
TTFT_MAX = {"mean": 10, "p50": 10, "p90": 100, "p99": 100}  # seconds
PARTS = {"prefill": "Prefill", "kv_load": "KV reload", "admission": "HBM admission", "queue": "Queue"}
QPS = "Offered call QPS per worker"
PANEL = (3.0, 2.3)


def load(paths):
    """Scenario label -> (summary, rows sorted by load); each row keeps its run directory."""
    merged = {}
    for path in paths:
        summary = json.loads((path / "summary.json").read_text())
        for row in summary["rows"]:
            rate = [r for r in (row["sessions_per_s"], row["sessions_per_s_per_worker"])
                    if (path / f"rate-{r:g}").exists()]
            row["dir"] = path / f"rate-{rate[0]:g}" if rate else None
        label = Path(summary["scenario"]).stem
        merged.setdefault(label, (summary, []))[1].extend(summary["rows"])
    for _, rows in merged.values():
        rows.sort(key=lambda r: r["offered_call_qps_per_worker"])
    return merged


def axes(count, **options):
    figure, grid = plt.subplots(1, count, figsize=(PANEL[0] * count, PANEL[1]), squeeze=False, **options)
    for ax in grid[0]:
        ax.grid(True, axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    return figure, grid[0]


def complete(rows):
    return [r for r in rows if r["status"] == "complete"]


def line(ax, color, label, rows, value):
    rows = complete(rows)
    ax.plot([r["offered_call_qps_per_worker"] for r in rows], [value(r) for r in rows], color=color,
            linewidth=1.5, marker="o", markersize=4, markeredgecolor="white", markeredgewidth=0.8,
            label=label)


def saturation(ax, color, rows):
    for r in rows:
        if r["status"] == "saturated":
            ax.axvline(r["offered_call_qps_per_worker"], color=color, linewidth=1, linestyle=(0, (3, 2)))


def ttft(sweeps, stats):
    figure, grid = axes(len(stats))
    for ax, stat in zip(grid, stats):
        for color, (label, (_, rows)) in zip(SERIES, sweeps.items()):
            line(ax, color, label, rows, lambda r: r[f"ttft_{stat}_s"])
            saturation(ax, color, rows)
        ax.set(title=STATS[stat], xlabel=QPS, xlim=(0, None), ylim=(0, TTFT_MAX[stat]))
    grid[0].set_ylabel("TTFT (s)")
    grid[0].legend(frameon=False)
    return figure


def parts(sweeps):
    figure, grid = axes(len(sweeps), sharey=True)
    for ax, (label, (_, rows)) in zip(grid, sweeps.items()):
        rows = complete(rows)
        picked = sorted({0, len(rows) // 2, len(rows) - 1}) if rows else []
        bottom = [0.0] * len(picked)
        for color, (part, name) in zip(SERIES, PARTS.items()):
            values = [rows[i].get(f"ttft_{part}_mean_s", 0.0) for i in picked]
            ax.bar(range(len(picked)), values, 0.6, bottom=bottom, color=color, edgecolor="white",
                   linewidth=0.8, label=name)
            bottom = [b + v for b, v in zip(bottom, values)]
        ax.set_xticks(range(len(picked)), [f"{rows[i]['offered_call_qps_per_worker']:.3g}" for i in picked])
        ax.set(title=label, xlabel=QPS)
    grid[0].set(ylabel="Mean TTFT (s)", ylim=(0, TTFT_MAX["mean"]))
    handles, names = grid[0].get_legend_handles_labels()
    grid[-1].legend(handles[::-1], names[::-1], frameon=False, loc="upper left", bbox_to_anchor=(1, 1))
    return figure


def reuse(sweeps):
    figure, (ax,) = axes(1)
    for color, (label, (_, rows)) in zip(SERIES, sweeps.items()):
        line(ax, color, label, rows, lambda r: 100 * r["prefix_reuse"])
        saturation(ax, color, rows)
    ax.set(xlabel=QPS, ylabel="Reused prompt tokens (%)", xlim=(0, None), ylim=(0, 100))
    ax.legend(frameon=False)
    return figure


def waiting(trace):
    """Times in hours and waiting requests per worker at each sampled state."""
    times, counts = [], []
    with trace.open() as stream:
        for text in stream:
            state = json.loads(text)
            times.append(state["time_s"] / 3600)
            counts.append([w["scheduler"]["waiting_requests"] for w in state["state"]["instances"]])
    return times, counts


def queue(sweeps):
    figure, grid = axes(len(sweeps), sharey=True)
    for ax, color, (label, (_, rows)) in zip(grid, SERIES, sweeps.items()):
        row = next(r for r in reversed(rows) if r["dir"])
        times, counts = waiting(row["dir"] / "trace.jsonl")
        ax.fill_between(times, [min(c) for c in counts], [max(c) for c in counts], color=color,
                        alpha=0.2, linewidth=0)
        ax.plot(times, [sum(c) / len(c) for c in counts], color=color, linewidth=1.2)
        ax.set(title=f"{label}, {row['offered_call_qps_per_worker']:.3g} QPS/worker", xlabel="Time (h)",
               xlim=(0, None), ylim=(0, None))
    grid[0].set_ylabel("Waiting requests per worker")
    return figure


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("sweeps", type=Path, nargs="+", help="sweep output directories")
    parser.add_argument("--figures", nargs="+", choices=["ttft", "parts", "reuse", "queue"],
                        default=["ttft", "parts", "reuse", "queue"])
    parser.add_argument("--stats", nargs="+", choices=STATS, default=list(STATS), help="ttft panels")
    parser.add_argument("--format", default="pdf", help="pdf, svg or png")
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    args = parser.parse_args()
    sweeps = load(args.sweeps)
    if len(sweeps) > len(SERIES):
        raise ValueError(f"at most {len(SERIES)} scenarios")
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "text.color": INK,
                         "axes.labelcolor": INK, "axes.edgecolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "legend.fontsize": 7, "pdf.fonttype": 42})
    draw = {"ttft": lambda: ttft(sweeps, args.stats), "parts": lambda: parts(sweeps),
            "reuse": lambda: reuse(sweeps), "queue": lambda: queue(sweeps)}
    args.out.mkdir(parents=True, exist_ok=True)
    for name in args.figures:
        figure = draw[name]()
        figure.tight_layout()
        figure.savefig(args.out / f"{name}.{args.format}", dpi=300)
        plt.close(figure)


if __name__ == "__main__":
    main()
