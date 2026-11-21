#!/usr/bin/env python3
"""Generate a cxl_pool scenario: platform × model × pool design.

Every worker is one TP model instance whose forward passes come from a measured
table; a projected platform divides them by a declared `compute_speedup`. Each
worker has one x16 CXL adapter, devices attach by x8, and every switch spends
all 256 lanes. `none` keeps the workers and drops the pool, so host DRAM is the
last KV tier. See docs/docs/getting-started/cxl-experiments.md.
"""

import argparse
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LANES = 256
EFFICIENCY = 0.8  # effective / raw link rate
GIB = 1 << 30

# Raw per-lane rates per direction: Gen5 32 GT/s with 128b/130b, Gen6 64 GT/s.
GEN5_LANE = 32e9 * 128 / 130 / 8
GEN6_LANE = 64e9 / 8

PLATFORMS = {
    # Vera Rubin NVL72 projection: 288 GB HBM4; NVLink-C2C 1.8 TB/s per Vera
    # counting both directions (65 TB/s per rack), i.e. 450 GB/s per GPU each
    # way; ConnectX-9 1.6 Tb/s per GPU; one compute tray per TP4 worker with
    # two Vera CPUs (1.5 TB and 1.2 TB/s LPDDR5X each); CXL 3.1 on PCIe Gen6.
    "vr-nvl72": {
        "compute_speedup": 3.0,
        "hbm_bytes": 288_000_000_000,
        "lane": GEN6_LANE * EFFICIENCY,
        "gpu_link": 450e9 * EFFICIENCY,
        "nic": 4 * 200e9 * EFFICIENCY,
        "dram": 2 * 1.2e12 * EFFICIENCY,
        "host_dram_bytes": 2 * 1.5e12,
        "device_bytes": 1e12,  # Samsung CMM-D 3.1
        "device_rate": 72e9,
    },
    # Measured compute. HBM from nvidia-smi (143,771 MiB); PCIe Gen5 x16 GPU
    # links; 4 × ConnectX-7 400 Gb/s, one Xeon socket of DDR5 and half of a
    # DGX H200's 2 TB per worker.
    "h200": {
        "compute_speedup": 1.0,
        "hbm_bytes": 150_754_820_096,
        "lane": GEN5_LANE * EFFICIENCY,
        "gpu_link": 16 * GEN5_LANE * EFFICIENCY,
        "nic": 4 * 50e9 * EFFICIENCY,
        "dram": 307e9 * EFFICIENCY,
        "host_dram_bytes": 1e12,
        "device_bytes": 256e9,  # Samsung CMM-D 2.0
        "device_rate": 36e9,
    },
}

# A model needs its architecture and a forward-pass table measured on H200 at
# its TP size (scripts/import_fpm.py).
MODELS = {
    "minimax-m2.7": {
        "tp": 4,
        "table": ROOT / "examples/data/h200/minimax_m2_7-tp4.json",
        "model": {"kind": "minimax_m2", "hidden_size": 3072, "intermediate_size": 1536,
                  "num_attention_heads": 48, "num_hidden_layers": 62, "num_key_value_heads": 8,
                  "num_local_experts": 256, "num_experts_per_tok": 8, "vocab_size": 200064,
                  "head_dim": 128, "rotary_dim": 64, "kv_dtype": "fp8"},
    },
}

HOP_LATENCY = 0.25e-6  # per switch traversal
UNBOUNDED = 1e13  # junction edges whose rate is set elsewhere


def small(switches, isl):
    """One switch: 8 workers × x16 + 16 devices × x8."""
    if switches != 1:
        raise ValueError("small uses one switch")
    return [8], [16], {}


def uniform(switches, isl):
    """Compute switches fully connected to memory switches, `isl` lanes each:
    every worker reaches every device through two switches."""
    half = switches // 2
    if switches < 2 or switches % 2 or isl % half or (LANES - isl) % 16:
        raise ValueError("uniform needs an even switch count, ISL lanes divisible by "
                         "half of it, and the remaining lanes divisible by 16")
    edges = {(c, half + m): isl // half for c in range(half) for m in range(half)}
    return [(LANES - isl) // 16] * half + [0] * half, [0] * half + [(LANES - isl) // 8] * half, edges


def ring(switches, isl):
    if switches < 2 or isl % 2:
        raise ValueError("ring needs at least two switches and even ISL lanes")
    return {(i, (i + 1) % switches): isl // 2 for i in range(switches)}


def torus(switches, isl):
    if isl % 4:
        raise ValueError("torus needs ISL lanes divisible by 4")
    rows = max((r for r in range(2, int(math.isqrt(switches)) + 1) if switches % r == 0), default=None)
    if rows is None:
        raise ValueError("torus needs a switch count with a factor of at least 2")
    cols = switches // rows
    edges = {}
    for r in range(rows):
        for c in range(cols):
            here = r * cols + c
            for there in (((r + 1) % rows) * cols + c, r * cols + (c + 1) % cols):
                edges[(here, there)] = edges.get((here, there), 0) + isl // 4
    return edges


# Non-uniform designs: each switch spends `isl` lanes on the edges a builder
# returns and the rest on workers and twice as many devices. Add a topology by
# adding a function from (switch count, ISL lanes) to {(switch, switch): lanes}.
NON_UNIFORM = {"ring": ring, "torus": torus}
DESIGNS = ["none", "small", "uniform", *NON_UNIFORM]


def global_lanes(split):
    """Inter-switch lanes of a LOCAL:GLOBAL split of a switch's lanes, e.g. 3:1 -> 64."""
    try:
        local, remote = (int(part) for part in split.split(":"))
    except ValueError:
        raise ValueError(f"split must be LOCAL:GLOBAL, got {split!r}") from None
    if local <= 0 or remote <= 0 or LANES * remote % (local + remote):
        raise ValueError(f"split {split} needs positive parts dividing {LANES} lanes evenly")
    return LANES * remote // (local + remote)


def design(name, switches, isl):
    if name == "small":
        return small(switches, isl)
    if name == "uniform":
        return uniform(switches, isl)
    if (LANES - isl) % 32:
        raise ValueError("non-uniform switches need local lanes divisible by 32: "
                         "an x16 worker and two x8 devices")
    edges = {}
    for (a, b), lanes in NON_UNIFORM[name](switches, isl).items():
        key = (min(a, b), max(a, b))
        edges[key] = edges.get(key, 0) + lanes
    workers = (LANES - isl) // 32
    return [workers] * switches, [2 * workers] * switches, edges


def build(platform, model, design_name, switches, isl, workers, compute_speedup, routing,
          host_kv_bytes, out, block_tokens=256, beta=1.0, inflight_loads=None):
    p, m = PLATFORMS[platform], MODELS[model]
    ranks = m["tp"]
    if design_name == "none":
        switches, workers_at, devices_at, edges = 0, [workers], [0], {}
    else:
        workers_at, devices_at, edges = design(design_name, switches, isl)
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
    for s in range(len(workers_at)):
        for _ in range(workers_at[s]):
            n = f"w{worker}"
            # Host DRAM serves all its traffic at its own rate. Every GPU
            # reaches DRAM, the NICs and the CXL root port through its C2C link;
            # NICs and the root port hang off Vera's PCIe.
            endpoints += [{"id": f"{n}-dram", "kind": "dram", "node": n,
                           "read_bandwidth": p["dram"], "write_bandwidth": p["dram"]},
                          {"id": f"{n}-c2c", "kind": "junction", "node": n},
                          {"id": f"{n}-nic", "kind": "nic", "node": n}]
            link(f"{n}-dram-c2c", f"{n}-dram", f"{n}-c2c", UNBOUNDED)
            link(f"{n}-dram-nic", f"{n}-dram", f"{n}-nic", UNBOUNDED)
            link(f"{n}-nic-c2c", f"{n}-nic", f"{n}-c2c", UNBOUNDED)
            link(f"{n}-net", f"{n}-nic", "fabric", p["nic"])
            if switches:
                endpoints.append({"id": f"{n}-cxl", "kind": "adapter", "node": n})
                link(f"{n}-dram-cxl", f"{n}-dram", f"{n}-cxl", UNBOUNDED, both=False)
                link(f"{n}-cxl-c2c", f"{n}-cxl", f"{n}-c2c", UNBOUNDED, both=False)
                link(f"{n}-cxl-S{s}", f"{n}-cxl", f"S{s}", 16 * p["lane"], 16, HOP_LATENCY)
                used[s] += 16
            for r in range(ranks):
                gpu = f"{n}-r{r}-gpu"
                endpoints.append({"id": gpu, "kind": "gpu", "node": n})
                link(f"{gpu}-c2c", f"{n}-c2c", gpu, p["gpu_link"])
                bindings.append({"worker": worker, "rank": r, "node": n,
                                 "adapters": [f"{n}-cxl"] if switches else [],
                                 "dram": f"{n}-dram", "nic": f"{n}-nic", "gpu": gpu})
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

    instance = {
        "kind": "single_instance",
        "model": m["model"],
        "model_runner": {"kind": "forward_pass", "tp": ranks,
                         "table": os.path.relpath(m["table"], out.parent),
                         "compute_speedup": compute_speedup or p["compute_speedup"]},
        "scheduler": {"kind": "chunked_prefill", "block_size": 16, "max_batch": 1024,
                      "max_num_batched_tokens": 8192, "device_mem": p["hbm_bytes"],
                      "device_mem_utilization": 0.92, "reserved_device_bytes": 4 * GIB,
                      "enable_prefix_caching": False},
    }
    if inflight_loads:
        instance["scheduler"]["max_num_inflight_kv_loads"] = inflight_loads
    system = {
        "kind": "cxl_pool", "routing": routing, "affinity_beta": beta, "block_tokens": block_tokens,
        "host_capacity_bytes": {f"w{w}": int(host_kv_bytes) for w in range(worker)},
        "placement_devices": [f"D{d}" for d in range(device)],
        "stripe_blocks": 1, "max_outstanding_reads": 256,
        "topology": {"endpoints": endpoints, "links": links, "workers": bindings,
                     "routing": "equal_shortest"},
        "instances": [instance] * worker,
    }
    return system, {"platform": platform, "model": model, "design": design_name,
                    "switches": switches, "isl_lanes": isl if switches > 1 else 0,
                    "workers": worker, "gpus": worker * ranks, "devices": device,
                    "host_kv_gib": round(host_kv_bytes / GIB, 1)}


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
    parser.add_argument("--platform", choices=PLATFORMS, default="vr-nvl72")
    parser.add_argument("--model", choices=MODELS, default="minimax-m2.7")
    parser.add_argument("--design", choices=DESIGNS, required=True)
    parser.add_argument("--switches", type=int, default=1)
    parser.add_argument("--local-global", default="1:1", metavar="LOCAL:GLOBAL",
                        help="split of each switch's lanes between workers and devices and "
                             "other switches, e.g. 3:1 (uniform, ring, torus)")
    parser.add_argument("--workers", type=int, default=8, help="worker count for --design none")
    parser.add_argument("--compute-speedup", type=float,
                        help="override the platform's projected speedup over the H200 table")
    parser.add_argument("--routing", choices=["least_loaded", "session", "cache_affinity"], default="least_loaded")
    parser.add_argument("--affinity-beta", type=float, default=1.0,
                        help="cache_affinity weight on relative in-flight load (production-stack default 1.0)")
    parser.add_argument("--host-kv-fraction", type=float, default=0.5,
                        help="share of the worker's host DRAM given to KV; the rest serves "
                             "the OS, runtime, pinned buffers and tools")
    parser.add_argument("--host-kv-gib", type=float,
                        help="host KV capacity per worker, overriding --host-kv-fraction")
    parser.add_argument("--block-tokens", type=int, default=256,
                        help="offload chunk (vLLM kv_connector_extra_config.block_size); "
                             "16 matches the GPU block but multiplies transfers 16×")
    parser.add_argument("--max-num-inflight-kv-loads", type=int,
                        help="vLLM's limit on requests holding KV for a load before they run; "
                             "loads then also start without a free slot or token budget "
                             "(default: none, as vLLM)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.block_tokens % 16:
        raise ValueError("block tokens must be a multiple of the 16-token GPU block")
    if not 0 < args.host_kv_fraction <= 1:
        raise ValueError("host KV fraction must be in (0, 1]")
    host_kv = (args.host_kv_gib * GIB if args.host_kv_gib is not None
               else args.host_kv_fraction * PLATFORMS[args.platform]["host_dram_bytes"])
    system, summary = build(args.platform, args.model, args.design, args.switches,
                            global_lanes(args.local_global),
                            args.workers, args.compute_speedup, args.routing,
                            host_kv, args.out, args.block_tokens, args.affinity_beta,
                            args.max_num_inflight_kv_loads)
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(toml(system, json.dumps(summary)))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
