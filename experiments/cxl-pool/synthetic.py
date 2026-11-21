#!/usr/bin/env python3
"""Synthetic agent sessions for pool stress tests.

Every session shares one system prompt, then alternates LLM calls with
exponential tool runs; each call's prompt extends the previous one, so its KV
prefix can be reused. Sessions arrive as a Poisson process.
"""

import argparse
import json
import random
from pathlib import Path

NAMESPACE = "synthetic|MiniMaxAI/MiniMax-M2.7@d494266a|tp4-fp8-block16|pool"
BLOCK = 16


def sessions(count, rate, turns, system, growth, output, tool, seed):
    rng = random.Random(seed)
    shared = list(range(1, system // BLOCK + 1))
    next_id, arrival = len(shared) + 1, 0.0
    for session in range(count):
        arrival += rng.expovariate(rate)
        ids, nodes, tokens = list(shared), [], system
        for turn in range(turns):
            added = tokens // BLOCK - len(ids)
            # Each call extends the previous call's identity.
            prefix = {"namespace": NAMESPACE, "block_tokens": BLOCK, "block_ids": list(ids)}
            if turn:
                prefix = {**prefix, "parent": len(nodes) - 2, "shared": len(ids),
                          "block_ids": list(range(next_id, next_id + added))}
            ids += range(next_id, next_id + added)
            next_id += added
            nodes.append({"kind": "llm", "input_tokens": tokens, "output_tokens": output,
                          "session_id": str(session), "kv_prefix": prefix})
            if turn + 1 < turns:
                nodes.append({"kind": "tool_call", "input_tokens": 0, "output_tokens": 0,
                              "duration": rng.expovariate(1 / tool), "session_id": str(session)})
            tokens += growth
        for index, node in enumerate(nodes):
            node["next"] = [index + 1] if index + 1 < len(nodes) else []
        yield {"request_id": session, "arrival_time": arrival, "initial": [0], "sub_requests": nodes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sessions", type=int, default=64)
    parser.add_argument("--sessions-per-second", type=float, default=1.0)
    parser.add_argument("--turns", type=int, default=10, help="LLM calls per session")
    parser.add_argument("--system-tokens", type=int, default=4096, help="prompt shared by all sessions")
    parser.add_argument("--growth-tokens", type=int, default=2048, help="prompt added per call")
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--tool-seconds", type=float, default=2.0, help="mean tool run")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as stream:
        for row in sessions(args.sessions, args.sessions_per_second, args.turns, args.system_tokens,
                            args.growth_tokens, args.output_tokens, args.tool_seconds, args.seed):
            stream.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
