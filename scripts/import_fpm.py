#!/usr/bin/env python3
"""Normalize one AISim forward-pass (FPM) Parquet into a `forward_pass` table.

Run with `uv run --with pyarrow scripts/import_fpm.py ...`.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path

# Columns that identify one deployment; every row must agree.
IDENTITY = (
    "model_path", "system", "backend", "backend_version", "weight_quantization",
    "gemm_quant_mode", "moe_quant_mode", "fmha_quant_mode", "comm_quant_mode",
    "kv_cache_dtype", "parallel_strategy", "tp", "pp", "dp", "moe_tp", "moe_ep",
    "cp", "moe_backend", "attention_backend", "enable_wideep", "enable_eplb",
)
COORDINATES = "iteration_totals_balanced_v1"


def normalize(rows, metadata, parquet_sha256, source_url):
    if metadata.get("coordinate_system") != COORDINATES:
        raise ValueError(f"unsupported coordinate system {metadata.get('coordinate_system')}")
    if metadata.get("parquet_sha256") not in (None, parquet_sha256):
        raise ValueError("metadata does not describe this Parquet")
    identity = {key: rows[0][key] for key in IDENTITY}
    decode, prefill, keys = [], [], set()
    for index, row in enumerate(rows):
        if any(row[key] != identity[key] for key in IDENTITY):
            raise ValueError(f"row {index}: mixed deployment identity")
        if row["partition_policy"] != "balanced_v1":
            raise ValueError(f"row {index}: unbalanced partition {row['partition_policy']}")
        batch, tokens, kv = (int(row[k]) for k in (
            "batch_size", "total_prefill_tokens", "total_kv_read_tokens"))
        latency = float(row["latency_ms"])
        if batch < 1 or kv < 0 or not math.isfinite(latency) or latency <= 0:
            raise ValueError(f"row {index}: invalid coordinates or latency")
        key = (row["workload_kind"], batch, tokens, kv)
        if key in keys:
            raise ValueError(f"row {index}: duplicate coordinates {key}")
        keys.add(key)
        if row["workload_kind"] == "decode" and tokens == 0:
            decode.append([batch, kv, latency])
        elif row["workload_kind"] == "prefill" and tokens >= batch:
            prefill.append([batch, tokens, kv, latency])
        else:
            raise ValueError(f"row {index}: unsupported workload {key}")
    if not decode or not prefill:
        raise ValueError("table needs decode and prefill measurements")
    if identity["pp"] != 1 or identity["cp"] != 1 or identity["dp"] != 1:
        raise ValueError("only DP1/PP1/CP1 deployments are supported")
    return {
        "format_version": 1,
        "latency_unit": "ms",
        "deployment": {**identity, "tp": int(identity["tp"]), "ep": int(identity["moe_ep"])},
        "source": {
            "url": source_url,
            "sha256": parquet_sha256,
            "rows": len(rows),
            **{k: metadata.get(k) for k in (
                "coordinate_system", "measurement_policy", "measurement_repeats",
                "warmup_repeats", "source", "source_parquet_sha256",
                "source_revision", "aisim_commit", "collected_by")},
        },
        "decode": sorted(decode),
        "prefill": sorted(prefill),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parquet", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--source-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import pyarrow.parquet as pq

    content = args.parquet.read_bytes()
    table = normalize(
        pq.read_table(args.parquet).to_pylist(),
        json.loads(args.metadata.read_text()),
        hashlib.sha256(content).hexdigest(),
        args.source_url,
    )
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(table, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
