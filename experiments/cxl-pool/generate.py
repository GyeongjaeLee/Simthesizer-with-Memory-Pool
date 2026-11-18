#!/usr/bin/env python3
"""Generate a cxl_pool scenario: platform × pool design × switch count.

Every worker is MiniMax-M2.7 TP4 on the measured H200 forward-pass table; a
projected platform divides it by a declared `compute_speedup`. Each worker has
one x16 CXL adapter, devices attach by x8, and every switch spends all 256 lanes.
See docs/docs/getting-started/cxl-experiments.md for the rationale of each value.
"""

import argparse
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TABLE = ROOT / "examples/data/h200/minimax_m2_7-tp4.json"
RANKS = 4
LANES = 256
EFFICIENCY = 0.8  # effective / raw link rate
GIB = 1 << 30

# Raw per-lane rates per direction: Gen5 32 GT/s with 128b/130b, Gen6 64 GT/s.
GEN5_LANE = 32e9 * 128 / 130 / 8
GEN6_LANE = 64e9 / 8

PLATFORMS = {
    # Measured compute. HBM from nvidia-smi (143,771 MiB); PCIe Gen5 x16 GPU
    # links; 4 × ConnectX-7 400 Gb/s and one Xeon socket of DDR5 per worker.
    "h200": {
        "compute_speedup": 1.0,
        "hbm_bytes": 150_754_820_096,
        "lane": GEN5_LANE * EFFICIENCY,
        "gpu_link": 16 * GEN5_LANE * EFFICIENCY,
        "nic": 4 * 50e9 * EFFICIENCY,
        "dram": 307e9 * EFFICIENCY,
        "device_bytes": 256e9,  # Samsung CMM-D 2.0
        "device_rate": 36e9,
    },
    # Vera Rubin NVL72 projection: 288 GB HBM4, NVLink-C2C 1.8 TB/s per Vera
    # shared by two GPUs, ConnectX-9 1.6 Tb/s per GPU, two Vera CPUs (1.2 TB/s
    # LPDDR5X each) per TP4 worker, CXL 3.1 on PCIe Gen6.
    "vr-nvl72": {
        "compute_speedup": 3.0,
        "hbm_bytes": 288_000_000_000,
        "lane": GEN6_LANE * EFFICIENCY,
        "gpu_link": 900e9 * EFFICIENCY,
        "nic": 4 * 200e9 * EFFICIENCY,
        "dram": 2 * 1.2e12 * EFFICIENCY,
        "device_bytes": 1e12,  # Samsung CMM-D 3.1
        "device_rate": 72e9,
    },
}

HOP_LATENCY = 0.25e-6  # per switch traversal
UNBOUNDED = 1e13  # junction edges whose rate is set elsewhere


def small(switches):
    """One switch: 8 workers × x16 + 16 devices × x8."""
    if switches != 1:
        raise ValueError("small uses one switch")
    return [8], [16], {}


def uniform(switches):
    """Compute switches (8 workers each) fully connected to memory switches
    (16 devices each): every worker reaches every device through two switches."""
    if switches < 2 or switches % 2 or LANES % switches:
        raise ValueError("uniform needs an even switch count dividing 256")
    half = switches // 2
    lanes = 128 // half
    edges = {(c, half + m): lanes for c in range(half) for m in range(half)}
    return [8] * half + [0] * half, [0] * half + [16] * half, edges


def ring(switches):
    if switches < 2:
        raise ValueError("ring needs at least two switches")
    return {(i, (i + 1) % switches): 64 for i in range(switches)}


def torus(switches):
    rows = max(r for r in range(2, int(math.isqrt(switches)) + 1) if switches % r == 0)
    cols = switches // rows
    edges = {}
    for r in range(rows):
        for c in range(cols):
            here = r * cols + c
            for there in (((r + 1) % rows) * cols + c, r * cols + (c + 1) % cols):
                edges[(here, there)] = edges.get((here, there), 0) + 32
    return edges


# Non-uniform designs: each switch hosts 4 workers and 8 devices and spends its
# other 128 lanes on the edges a builder returns. Add a topology by adding a
# function from switch count to {(switch, switch): lanes}.
NON_UNIFORM = {"ring": ring, "torus": torus}


def design(name, switches):
    if name == "small":
        return small(switches)
    if name == "uniform":
        return uniform(switches)
    edges = {}
    for (a, b), lanes in NON_UNIFORM[name](switches).items():
        key = (min(a, b), max(a, b))
        edges[key] = edges.get(key, 0) + lanes
    return [4] * switches, [8] * switches, edges


def build(platform, design_name, switches, compute_speedup, routing, staging_gib, out, block_tokens=256):
    p = PLATFORMS[platform]
    workers_at, devices_at, edges = design(design_name, switches)
    endpoints = [{"id": "fabric", "kind": "fabric"}]
    links, bindings, used = [], [], [0] * switches

    def link(id, a, b, rate, width=None, latency=0.0, both=True):
        row = {"id": id, "a": a, "b": b, "a_port": f"{id}:a", "b_port": f"{id}:b",
               "bandwidth_ab": rate, "latency": latency}
        if both:
            row["bandwidth_ba"] = rate
        if width:
            row["width"] = width
        links.append(row)

    for s in range(switches):
        endpoints.append({"id": f"S{s}", "kind": "switch", "lane_budget": LANES})
    worker = device = 0
    for s in range(switches):
        for _ in range(workers_at[s]):
            n = f"w{worker}"
            endpoints += [{"id": f"{n}-{k}", "kind": kind, "node": n} for k, kind in
                          [("dram", "dram"), ("nic", "nic"), ("adapter", "adapter"), ("in", "junction")]]
            link(f"{n}-cxl", f"{n}-adapter", f"S{s}", 16 * p["lane"], 16, HOP_LATENCY)
            link(f"{n}-dram-cxl", f"{n}-dram", f"{n}-adapter", p["dram"], both=False)
            link(f"{n}-cxl-in", f"{n}-adapter", f"{n}-in", UNBOUNDED, both=False)
            link(f"{n}-dram-nic", f"{n}-dram", f"{n}-nic", p["dram"])
            link(f"{n}-net", f"{n}-nic", "fabric", p["nic"])
            link(f"{n}-net-in", f"{n}-nic", f"{n}-in", UNBOUNDED, both=False)
            for r in range(RANKS):
                gpu = f"{n}-r{r}-gpu"
                endpoints.append({"id": gpu, "kind": "gpu", "node": n})
                link(f"{gpu}-in", f"{n}-in", gpu, p["gpu_link"], both=False)
                link(f"{gpu}-out", gpu, f"{n}-nic", p["gpu_link"], both=False)
                bindings.append({"worker": worker, "rank": r, "node": n, "adapters": [f"{n}-adapter"],
                                 "dram": f"{n}-dram", "nic": f"{n}-nic", "gpu": gpu})
            used[s] += 16
            worker += 1
        for _ in range(devices_at[s]):
            d = f"D{device}"
            endpoints.append({"id": d, "kind": "memory", "capacity_bytes": int(p["device_bytes"]),
                              "read_bandwidth": p["device_rate"], "write_bandwidth": p["device_rate"]})
            link(f"{d}-cxl", f"S{s}", d, 8 * p["lane"], 8)
            used[s] += 8
            device += 1
    for (a, b), lanes in sorted(edges.items()):
        for i in range(math.ceil(lanes / 16)):
            width = min(16, lanes - 16 * i)
            link(f"S{a}-S{b}-{i}", f"S{a}", f"S{b}", width * p["lane"], width, HOP_LATENCY)
        used[a] += lanes
        used[b] += lanes
    if used != [LANES] * switches:
        raise ValueError(f"switch lanes must all be used: {used}")

    table = os.path.relpath(TABLE, out.parent)
    instance = {
        "kind": "single_instance",
        "model": {"kind": "minimax_m2", "hidden_size": 3072, "intermediate_size": 1536,
                  "num_attention_heads": 48, "num_hidden_layers": 62, "num_key_value_heads": 8,
                  "num_local_experts": 256, "num_experts_per_tok": 8, "vocab_size": 200064,
                  "head_dim": 128, "rotary_dim": 64, "kv_dtype": "fp8"},
        "model_runner": {"kind": "forward_pass", "tp": RANKS, "table": table,
                         "compute_speedup": compute_speedup or p["compute_speedup"]},
        "scheduler": {"kind": "chunked_prefill", "block_size": 16, "max_batch": 1024,
                      "max_num_batched_tokens": 8192, "device_mem": p["hbm_bytes"],
                      "device_mem_utilization": 0.92, "reserved_device_bytes": 4 * GIB,
                      "enable_prefix_caching": False},
    }
    system = {
        "kind": "cxl_pool", "routing": routing, "block_tokens": block_tokens,
        "host_capacity_bytes": {f"w{w}": staging_gib * GIB for w in range(worker)},
        "placement_devices": [f"D{d}" for d in range(device)],
        "stripe_blocks": 1, "max_outstanding_reads": 256,
        "topology": {"endpoints": endpoints, "links": links, "workers": bindings,
                     "routing": "equal_shortest"},
        "instances": [instance] * worker,
    }
    return system, {"platform": platform, "design": design_name, "switches": switches,
                    "workers": worker, "gpus": worker * RANKS, "devices": device}


def toml(system, header):
    def value(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float, str)):
            return json.dumps(v)
        if isinstance(v, list):
            return "[" + ", ".join(value(x) for x in v) + "]"
        return "{ " + ", ".join(f"{json.dumps(k)} = {value(x)}" for k, x in v.items()) + " }"

    lines = [f"# {header}", "[system]"]
    lines += [f"{k} = {value(v)}" for k, v in system.items() if k not in ("topology", "instances")]
    lines.append("\n[system.topology]")
    for k, v in system["topology"].items():
        if isinstance(v, list):
            lines.append(f"{k} = [\n" + ",\n".join("  " + value(x) for x in v) + "\n]")
        else:
            lines.append(f"{k} = {value(v)}")
    for instance in system["instances"]:
        lines.append("\n[[system.instances]]")
        lines += [f"{k} = {value(v)}" for k, v in instance.items()]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--platform", choices=PLATFORMS, default="h200")
    parser.add_argument("--design", choices=["small", "uniform", *NON_UNIFORM], required=True)
    parser.add_argument("--switches", type=int, default=1)
    parser.add_argument("--compute-speedup", type=float,
                        help="override the platform's projected speedup over the H200 table")
    parser.add_argument("--routing", choices=["vllm_dp", "session_hash"], default="vllm_dp")
    parser.add_argument("--host-staging-gib", type=int, default=128,
                        help="host DRAM KV staging per worker")
    parser.add_argument("--block-tokens", type=int, default=256,
                        help="offload chunk (vLLM kv_connector_extra_config.block_size); "
                             "16 matches the GPU block but multiplies transfers 16×")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.block_tokens % 16:
        raise ValueError("block tokens must be a multiple of the 16-token GPU block")
    system, summary = build(args.platform, args.design, args.switches, args.compute_speedup,
                            args.routing, args.host_staging_gib, args.out, args.block_tokens)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(toml(system, json.dumps(summary)))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
