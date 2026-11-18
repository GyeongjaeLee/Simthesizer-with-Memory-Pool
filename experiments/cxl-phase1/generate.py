#!/usr/bin/env python3
"""Generate explicit CXL topology examples with synthetic service rates."""
import argparse
import json
from pathlib import Path

FAMILIES = {
    "single": (1, 8, 16, 0, 0),
    "balanced": (8, 4, 8, 4, 2),
    "fewer": (8, 4, 8, 1, 2),
    "dense": (8, 6, 12, 4, 1),
    "fixture": (2, 1, 4, 0, 0),
}


def topology(family, lane_rate=4e9, ingress_rate=64e9):
    """All families use the same explicit inventory and deterministic routing."""
    leaves, nodes_per_leaf, devices_per_leaf, spines, parallel = FAMILIES[family]
    endpoints, links, workers = [], [], []

    def endpoint(id, kind, **kwargs):
        endpoints.append({"id": id, "kind": kind, **kwargs})

    def link(id, a, b, bandwidth, width=None, duplex=False):
        row = {"id": id, "a": a, "b": b, "a_port": f"{id}:a",
               "b_port": f"{id}:b", "bandwidth_ab": bandwidth, "latency": 0.0}
        if duplex or width is not None:
            row["bandwidth_ba"] = bandwidth
        if width is not None:
            row["width"] = width
        links.append(row)

    endpoint("compute", "fabric")
    for leaf in range(leaves):
        leaf_id = f"leaf{leaf}"
        endpoint(leaf_id, "switch", lane_budget=256)
        for j in range(nodes_per_leaf):
            n = leaf * nodes_per_leaf + j
            node = f"node{n}"
            for suffix, kind in [("adapter", "adapter"), ("dram", "dram"),
                                 ("nic", "nic"), ("ingress", "junction"), ("gpu", "gpu")]:
                endpoint(f"{node}-{suffix}", kind, node=node)
            link(f"cxl-node{n}", f"{node}-adapter", leaf_id, 16 * lane_rate, 16)
            link(f"dram-cxl{n}", f"{node}-dram", f"{node}-adapter", 1e12)
            link(f"adapter-gpu{n}", f"{node}-adapter", f"{node}-ingress", 1e12)
            link(f"dram-nic{n}", f"{node}-dram", f"{node}-nic", 1e12)
            link(f"compute-node{n}", f"{node}-nic", "compute", 64e9, duplex=True)
            link(f"nic-gpu{n}", f"{node}-nic", f"{node}-ingress", 1e12)
            link(f"gpu-ingress{n}", f"{node}-ingress", f"{node}-gpu", ingress_rate)
            link(f"gpu-egress{n}", f"{node}-gpu", f"{node}-nic", 64e9)
            workers.append({"worker": n, "node": node, "adapter": f"{node}-adapter",
                            "dram": f"{node}-dram", "nic": f"{node}-nic",
                            "gpu": f"{node}-gpu"})
        for j in range(devices_per_leaf):
            d = leaf * devices_per_leaf + j
            endpoint(f"D{d}", "memory", capacity_bytes=64_000_000_000,
                     read_bandwidth=1e12, write_bandwidth=1e12)
            link(f"cxl-D{d}", leaf_id, f"D{d}", 8 * lane_rate, 8)
    for spine in range(spines):
        endpoint(f"spine{spine}", "switch", lane_budget=256)
        for leaf in range(leaves):
            for p in range(parallel):
                link(f"uplink-{leaf}-{spine}-{p}", f"leaf{leaf}", f"spine{spine}",
                     16 * lane_rate, 16)
    if family == "fixture":
        link("leaf-boundary", "leaf0", "leaf1", 16 * lane_rate, 16)
    return {"endpoints": endpoints, "links": links, "workers": workers,
            "required_routes": [], "routing": "equal_shortest"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("output/cxl-topologies"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for family in FAMILIES:
        path = args.out / f"topology-{family}.json"
        path.write_text(json.dumps(topology(family), indent=2) + "\n")
