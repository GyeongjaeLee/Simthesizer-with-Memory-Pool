#!/usr/bin/env python3
"""Sweep FreeInference session arrival rates on one scenario.

Session rate is the controlled input; LLM-call QPS is measured, because each
session's calls wait for earlier calls and tools. Each rate gets a Poisson
proxy-prefix workload, a simulation, and one summary row.
"""

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import results  # noqa: E402

IDENTITY = ["--token-mode", "proxy-prefix", "--target-model", "MiniMaxAI/MiniMax-M2.7@d494266a",
            "--kv-layout", "tp4-fp8-block16", "--sharing-scope", "pool"]


def workload(index, out, rate, sessions, seed):
    subprocess.run([sys.executable, ROOT / "scripts/freeinference.py", "generate", "--index", index,
                    "--out", out, "--arrivals", "poisson", "--sessions-per-second", str(rate),
                    "--seed", str(seed), "--limit", str(sessions), "--invalid", "exclude", *IDENTITY],
                   check=True, stdout=subprocess.DEVNULL)


def summarize(path, trace):
    run = results.load_run(str(path))
    samples = results.measurements(run)
    starts = sorted(
        start for row in run.requests.values()
        for start, node in zip(results.timeline(row)[0], row["sub_requests"]) if node["kind"] == "llm")
    span = starts[-1] - starts[0]
    row = {"llm_calls": len(starts), "call_qps": len(starts) / span if span else math.nan,
           "makespan_s": samples["makespan"][()]}
    for metric in ("ttft", "tpot"):
        for stat, value in results.summarize(samples[metric], [50, 90, 99]).items():
            row[f"{metric}_{stat}_s"] = value
    if trace:
        state = json.loads(trace.read_text().splitlines()[-1])["state"]["cxl_pool"]
        for kind, (_, nbytes) in state["transfer_totals"].items():
            row[f"{kind}_GB"] = nbytes / 1e9
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", type=Path, required=True)
    parser.add_argument("--index", type=Path, default=ROOT / "output/freeinference/all-index")
    parser.add_argument("--rates", type=float, nargs="+", required=True, help="sessions per second")
    parser.add_argument("--sessions", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--binary", type=Path, default=ROOT / "target/release/simthesizer")
    parser.add_argument("--trace", action="store_true",
                        help="write engine traces for kv_metrics.py and transfer totals (large)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for rate in args.rates:
        case = args.out / f"rate-{rate:g}"
        workload(args.index, case / "workload", rate, args.sessions, args.seed)
        command = [args.binary, "--config", args.scenario, "--workload", case / "workload/workload.jsonl",
                   "--output-jsonl", case / "results.jsonl"]
        if args.trace:
            command += ["--output-engine-trace", case / "trace.jsonl"]
        subprocess.run(command, check=True)
        rows.append({"sessions_per_s": rate, **summarize(case / "results.jsonl",
                                                          case / "trace.jsonl" if args.trace else None)})
        print(json.dumps(rows[-1]), flush=True)
    (args.out / "summary.json").write_text(json.dumps(
        {"scenario": str(args.scenario), "sessions": args.sessions, "seed": args.seed, "rows": rows}, indent=1))
    with open(args.out / "summary.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
