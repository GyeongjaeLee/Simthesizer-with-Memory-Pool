#!/usr/bin/env python3
"""Compare the H200 MiniMax-M2.7 TP4 table with live FPM iteration truth.

Independent re-implementation of the `forward_pass` runner lookup, applied to
each observed iteration's scheduled batch. Zero-time heartbeats and the first
iteration after an idle step are excluded: their wall time spans idle waiting.
Run with `uv run --with pyarrow --with pandas experiments/fpm-validation/iterations.py`.
"""

import argparse
import bisect
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "output/fpm-validation/source/data/MiniMaxAI--MiniMax-M2.7"
TABLE = ROOT / "examples/data/h200/minimax_m2_7-tp4.json"
TRUTH = DATA / "h200-sxm/vllm/0.25.1/pure-tp4/measurements/iterations-01.parquet"


def interpolate(x, x0, x1, y0, y1):
    if x1 == x0:
        return y0
    if x <= x1:
        return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return y1 + max(0.0, y1 - y0) * (x - x1) / (x1 - x0)


class Surface:
    def __init__(self, rows):
        curves = {}
        for outer, inner, y in rows:
            curves.setdefault(outer, {}).setdefault(inner, []).append(y)
        self.keys = sorted(curves)
        self.curves = [sorted((x, np.mean(ys)) for x, ys in curves[k].items()) for k in self.keys]

    @staticmethod
    def curve(points, x):
        xs = [p[0] for p in points]
        if x <= xs[0] or len(xs) == 1:
            return points[0][1]
        i = min(bisect.bisect_left(xs, x), len(xs) - 1)
        return interpolate(x, *xs[i - 1:i + 1], points[i - 1][1], points[i][1])

    def at(self, outer, inner):
        if outer <= self.keys[0] or len(self.keys) == 1:
            return self.curve(self.curves[0], inner)
        i = min(bisect.bisect_left(self.keys, outer), len(self.keys) - 1)
        return interpolate(outer, self.keys[i - 1], self.keys[i],
                           self.curve(self.curves[i - 1], inner), self.curve(self.curves[i], inner))


class Table:
    def __init__(self, path):
        t = json.loads(Path(path).read_text())
        self.decode = Surface((b, k, y) for b, k, y in t["decode"])
        self.prefill = Surface((tk, tk * (tk / (2 * b) + k / b), y) for b, tk, k, y in t["prefill"])

    def iteration(self, npf, spt, spkv, nd, sdkv):
        """Balanced projection of FPM aggregates; one-token prefills are decodes."""
        if npf == 0 or spt == npf:
            return self.decode.at(nd + npf, sdkv + spkv)
        attention = spt * (spt / (2 * npf) + spkv / npf) + sdkv + nd / 2
        return self.prefill.at(spt + nd, attention)


def iterations(path):
    rows = []
    for ranks in pd.read_parquet(path).rank_measurements:
        assert len(ranks) == 1, "multi-rank grouping is not supported"
        r = ranks[0]
        s = r["scheduled_requests"]
        rows.append((r["counter_id"], r["wall_time"] * 1e3, s["num_prefill_requests"], s["sum_prefill_tokens"],
                     s["sum_prefill_kv_tokens"], s["num_decode_requests"], s["sum_decode_kv_tokens"]))
    x = pd.DataFrame(rows, columns=["ctr", "wall_ms", "npf", "spt", "spkv", "nd", "sdkv"]).sort_values("ctr")
    busy = (x.wall_ms > 0) & (x.npf + x.nd > 0)
    x["after_idle"] = ~busy.shift(1, fill_value=False)
    return x[busy].copy()


def classify(x):
    one = (x.npf > 0) & (x.spt == x.npf)
    return np.where(x.npf == 0, "decode", np.where(one, "one_token_prefill", np.where(x.nd == 0, "prefill", "mixed")))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    summary = []
    tab = Table(TABLE)
    x = iterations(TRUTH)
    x["pred_ms"] = [tab.iteration(*r) for r in x[["npf", "spt", "spkv", "nd", "sdkv"]].itertuples(index=False)]
    x["cls"] = classify(x)
    x.to_parquet(args.out / "iterations.parquet")
    for cls, g in x.groupby("cls"):
        idle = g[g.after_idle]
        g = g[~g.after_idle]
        ape = (g.pred_ms - g.wall_ms).abs() / g.wall_ms * 100
        summary.append({
            "class": cls, "n": len(g), "after_idle_excluded": len(idle),
            "mape": ape.mean(), "mdape": ape.median(), "p90_ape": ape.quantile(0.9),
            "wape": (g.pred_ms - g.wall_ms).abs().sum() / g.wall_ms.sum() * 100,
            "bias": (g.pred_ms.sum() / g.wall_ms.sum() - 1) * 100,
            "median_truth_ms": g.wall_ms.median(),
        })
    s = pd.DataFrame(summary)
    s.to_json(args.out / "summary.json", orient="records", indent=1)
    print(s.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
