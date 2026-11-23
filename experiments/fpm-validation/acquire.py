#!/usr/bin/env python3
"""Fetch the pinned H200 MiniMax-M2.7 TP4 FPM table and live iterations."""

import hashlib
import urllib.request
from pathlib import Path

REVISION = "dbfcdd9eb62ea733efbc3f0cbd78277f2960061a"
BASE = (f"https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/resolve/{REVISION}"
        "/data/MiniMaxAI--MiniMax-M2.7/")
OUT = Path(__file__).resolve().parents[2] / "output/fpm-validation/source/data/MiniMaxAI--MiniMax-M2.7"
FILES = [
    ("h200-sxm/vllm/0.25.1/pure-tp4/fpm/fpm_forward_perf.metadata.json",
     "aab7bb66f7005db897ec4086ea468da74f89a38935883ed9e5189b6a635096dc"),
    ("h200-sxm/vllm/0.25.1/pure-tp4/fpm/fpm_forward_perf.parquet",
     "c5415e92b228026d4f62b916243999aa76dbf108205804fdd95a6dbc34e26892"),
    ("h200-sxm/vllm/0.25.1/pure-tp4/measurements/iterations-01.parquet",
     "f17818a77f496edabd28025d2dc8e5e1a752be1a35ef7fc5adfb9e92a7cd3432"),
]


def main():
    for path, sha256 in FILES:
        target = OUT / path
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(BASE + path, target)
        digest = hashlib.sha256(target.read_bytes()).hexdigest()
        if digest != sha256:
            raise ValueError(f"{path}: SHA256 {digest} != {sha256}")
    print(f"verified {len(FILES)} files under {OUT}")


if __name__ == "__main__":
    main()
