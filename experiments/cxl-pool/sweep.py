#!/usr/bin/env python3
"""Sweep FreeInference session arrival rates on one scenario.

Sessions keep their recorded timing and arrive as a Poisson process until the
horizon T; nodes that cannot start before T are dropped. Calls released in
[W, T) are measured, after the warm-up W. W and T are fixed for every rate so
that rates compare over the same window. Session rate is the controlled input,
for the whole system or, with --per-worker, per worker. Offered call QPS counts
calls whose start in [W, T) would hold if every call took no time; call QPS
counts those actually released there.

As in BookSim, rates run in ascending order and a saturated rate ends the
sweep: a run stops once the mean first-token wait (TTFT minus prefill) of
calls completed in a window exceeds --saturation-wait in consecutive windows.
"""

import argparse
import csv
import fcntl
import json
import math
import re
import statistics
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import results  # noqa: E402

IDENTITY = ["--token-mode", "proxy-prefix", "--target-model", "MiniMaxAI/MiniMax-M2.7@d494266a",
            "--kv-layout", "tp4-fp8-block16", "--sharing-scope", "pool"]
HOUR = 3600.0
PARTS = ("kv_load", "admission", "queue", "prefill")


def workload(index, out, rate, horizon, seed, selection_seed, block_tokens):
    """Generate a workload into `out` unless one is there. Workloads depend only
    on these arguments, so scenarios may share them; a lock keeps concurrent
    sweeps from generating one twice."""
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.parent / f"{out.name}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (out / "manifest.json").exists():
            return
        subprocess.run([sys.executable, ROOT / "scripts/freeinference.py", "generate", "--index", index,
                        "--out", out, "--arrivals", "poisson", "--sessions-per-second", str(rate),
                        "--seed", str(seed), "--selection-seed", str(selection_seed),
                        "--until", str(horizon), "--invalid", "exclude",
                        "--prefix-block-tokens", str(block_tokens), *IDENTITY],
                       check=True, stdout=subprocess.DEVNULL)


def offered_qps(manifest, warmup, horizon):
    """Calls per second whose zero-latency start falls in [W, T)."""
    minutes = json.loads(manifest.read_text())["offered_llm_calls_per_minute"]
    return sum(minutes[int(warmup // 60):math.ceil(horizon / 60)]) / (horizon - warmup)


def pool_state(trace, warmup, horizon):
    """Transfer GB by kind between the first snapshots at or after W and T, and
    host and pool occupancy at T. A run that ends before T uses its last state."""
    stamp = re.compile(rb'^\{"time_s":([^,]+),')
    first, last = {}, None
    with trace.open("rb") as stream:
        for line in stream:
            time = float(stamp.match(line).group(1))
            for bound in (warmup, horizon):
                if bound not in first and time >= bound:
                    first[bound] = line
            last = line
            if horizon in first:
                break
    start, end = (json.loads(first.get(b, last))["state"]["cxl_pool"] for b in (warmup, horizon))
    row = {f"{kind}_GB": (nbytes - start["transfer_totals"].get(kind, (0, 0))[1]) / 1e9
           for kind, (_, nbytes) in end["transfer_totals"].items()}
    row["host_GB_at_T"] = sum(h["block_bytes"] for h in end["hosts"]) / 1e9
    row["pool_GB_at_T"] = sum(d["occupied_or_reserved_bytes"] for d in end["pool_usage"]) / 1e9
    return row


def summarize(path, trace, warmup, horizon):
    """Latency of calls released in [W, T); pool traffic and occupancy from the trace."""
    run = results.load_run(str(path))
    samples = results.measurements(run)
    measured = {(rid, index) for rid, row in run.requests.items()
                for index, (start, node) in enumerate(zip(results.timeline(row)[0], row["sub_requests"]))
                if node["kind"] == "llm" and warmup <= start < horizon}
    row = {"sessions": len(run.requests), "measured_calls": len(measured),
           "call_qps": len(measured) / (horizon - warmup)}
    for metric in ("ttft", "tpot"):
        values = {key: v for key, v in samples[metric].items() if key in measured}
        for stat, value in results.summarize(values, [50, 90, 99]).items():
            row[f"{metric}_{stat}_s"] = value
    calls = [run.requests[rid]["sub_requests"][index] for rid, index in measured]
    parts = [call.get("ttft_parts") for call in calls]
    for part in PARTS:
        values = sorted(p[part] for p in parts if p)
        if values:
            row[f"ttft_{part}_mean_s"] = statistics.fmean(values)
            row[f"ttft_{part}_p99_s"] = results.percentile(values, 99)
    row["prompt_tokens"] = sum(call["input_tokens"] for call in calls)
    row["reused_tokens"] = sum(call.get("reused_tokens", 0) for call in calls)
    row["prefix_reuse"] = row["reused_tokens"] / max(row["prompt_tokens"], 1)
    row.update(pool_state(trace, warmup, horizon))
    return row


def write_summary(args, workers, warmup, horizon, rows):
    """Rewritten after every rate, so a sweep that fails keeps finished rates."""
    (args.out / "summary.json").write_text(json.dumps(
        {"scenario": str(args.scenario), "workers": workers, "warmup_s": warmup, "horizon_s": horizon,
         "seed": args.seed, "selection_seed": args.selection_seed,
         "saturation": {"wait_s": args.saturation_wait, "period_s": args.saturation_period,
                        "windows": args.saturation_windows},
         "rows": rows}, indent=1))
    with open(args.out / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=ROOT / "output/freeinference/all-index")
    parser.add_argument("--rates", type=float, nargs="+", required=True, help="sessions per second")
    parser.add_argument("--per-worker", action="store_true",
                        help="read rates per worker and multiply by the scenario's worker count")
    parser.add_argument("--warmup-hours", type=float, default=24, help="W: calls before it are not measured")
    parser.add_argument("--horizon-hours", type=float, default=48, help="T: arrivals and measurement end")
    parser.add_argument("--seed", type=int, default=42, help="arrival gaps")
    parser.add_argument("--selection-seed", type=int, default=0, help="session order")
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/simthesizer")
    parser.add_argument("--trace", action="store_true",
                        help="write full engine traces for kv_metrics.py (large); "
                             "otherwise states are kept every --sample-interval and at W and T")
    parser.add_argument("--sample-interval", type=float, default=60.0, help="trace sampling period in s")
    parser.add_argument("--saturation-wait", type=float, default=150.0,
                        help="mean first-token wait in s that marks a window unstable; 0 disables")
    parser.add_argument("--saturation-period", type=float, default=600.0, help="window in simulated s")
    parser.add_argument("--saturation-windows", type=int, default=2,
                        help="consecutive unstable windows that saturate a run")
    parser.add_argument("--workloads", type=Path,
                        help="share workloads here across sweeps with the same index; "
                             "otherwise each rate keeps its own")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    warmup, horizon = args.warmup_hours * HOUR, args.horizon_hours * HOUR
    if not 0 <= warmup < horizon or not math.isfinite(horizon):
        raise ValueError("need 0 <= warm-up < horizon")
    system = tomllib.loads(args.scenario.read_text())["system"]
    workers, block_tokens = len(system["instances"]), system["block_tokens"]
    scale = workers if args.per_worker else 1
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    rates = sorted(args.rates)
    for rate in rates:
        case = args.out / f"rate-{rate:g}"
        source = (args.workloads / f"r{rate * scale:g}-t{horizon:g}-s{args.seed}-sel{args.selection_seed}-b{block_tokens}"
                  if args.workloads else case / "workload")
        workload(args.index, source, rate * scale, horizon, args.seed, args.selection_seed, block_tokens)
        case.mkdir(exist_ok=True)
        command = [args.binary, "--config", args.scenario, "--workload", source / "workload.jsonl",
                   "--output-jsonl", case / "results.jsonl", "--output-engine-trace", case / "trace.jsonl"]
        if not args.trace:
            samples = sorted({*(i * args.sample_interval for i in range(int(horizon // args.sample_interval))),
                              warmup, horizon})
            command += ["--trace-times", *map(str, samples)]
        if args.saturation_wait > 0:
            command += ["--saturation-wait", str(args.saturation_wait),
                        "--saturation-period", str(args.saturation_period),
                        "--saturation-windows", str(args.saturation_windows)]
        status = json.loads(subprocess.run(command, check=True, stdout=subprocess.PIPE,
                                           text=True).stdout.splitlines()[-1])
        offered = offered_qps(source / "manifest.json", warmup, horizon)
        row = {"sessions_per_s": rate * scale, "sessions_per_s_per_worker": rate * scale / workers,
               "offered_call_qps": offered, "offered_call_qps_per_worker": offered / workers,
               "status": status["status"]}
        if status["status"] == "saturated":
            row |= {"saturated_at_s": status["time_s"], "saturated_wait_s": status["mean_wait_s"]}
        else:
            row |= summarize(case / "results.jsonl", case / "trace.jsonl", warmup, horizon)
        rows.append(row)
        write_summary(args, workers, warmup, horizon, rows)
        print(json.dumps(row), flush=True)
        if status["status"] == "saturated":
            skipped = [r for r in rates if r > rate]
            if skipped:
                print(f"saturated at {rate:g}; skipping {skipped}", file=sys.stderr)
            break


if __name__ == "__main__":
    main()
