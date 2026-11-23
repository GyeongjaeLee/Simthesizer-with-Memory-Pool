"""Fetch, index and replay FreeInference sessions (Python 3.14+)."""

import argparse
from array import array
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import contextlib
from compression import zstd
import hashlib
import io
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
import urllib.request

import orjson

DATASET = "harvardMadsys/freeinference_agentic_trace"
REVISION = "bb26cf72c9c20f2b05f765d88548156a74d3c253"
WEEK = "2026-08-30_2026-09-05"
HUB = "https://huggingface.co"
PROXY_BLOCK_TOKENS = 16


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_bytes(orjson.dumps(value, option=orjson.OPT_INDENT_2) + b"\n")


def json_line(value):
    return orjson.dumps(value).decode() + "\n"


def get_json(url):
    with urllib.request.urlopen(url, timeout=60) as response:
        return json.load(response)


def fresh(path):
    path.mkdir(parents=True, exist_ok=False)


def download(url, destination, size=None, expected_sha=None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        partial = destination.with_name(destination.name + ".partial")
        with urllib.request.urlopen(url, timeout=60) as response, partial.open("wb") as out:
            shutil.copyfileobj(response, out, length=1024 * 1024)
        if size is not None and partial.stat().st_size != size:
            raise ValueError(f"download size mismatch: {destination}")
        if expected_sha is not None and sha256(partial) != expected_sha:
            raise ValueError(f"download hash mismatch: {destination}")
        partial.rename(destination)
    digest = sha256(destination)
    if size is not None and destination.stat().st_size != size:
        raise ValueError(f"cached size mismatch: {destination}")
    if expected_sha is not None and digest != expected_sha:
        raise ValueError(f"cached hash mismatch: {destination}")
    return {"url": url, "bytes": destination.stat().st_size, "sha256": digest}


def fetch(args):
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("revision must be a pinned 40-character commit")
    if args.all_weeks:
        entries = get_json(f"{HUB}/api/datasets/{DATASET}/tree/{args.revision}/release?limit=1000")
        args.weeks = sorted(e["path"].split("/")[-1] for e in entries if e["type"] == "directory")
    args.out.mkdir(parents=True, exist_ok=True)
    receipt_path = args.out / "source.json"
    if receipt_path.exists():
        old = json.loads(receipt_path.read_text())
        if old["revision"] != args.revision or old["weeks"] != args.weeks:
            raise ValueError("use another directory for a different source selection")
    base = f"{HUB}/datasets/{DATASET}/resolve/{args.revision}/"
    files = {"README.md": download(base + "README.md", args.out / "README.md")}
    for week in args.weeks:
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}", week):
            raise ValueError("invalid week")
        url = f"{HUB}/api/datasets/{DATASET}/tree/{args.revision}/release/{week}?recursive=true&limit=1000"
        entries = get_json(url)
        chosen = [e for e in entries if e["type"] == "file" and
                  (e["path"].endswith("/manifest.json") or
                   "/traces/" in e["path"] and e["path"].endswith(".jsonl.zst"))]
        if not chosen:
            raise ValueError(f"no trace files in {week}")
        for entry in sorted(chosen, key=lambda e: e["path"]):
            path = entry["path"]
            if not path.startswith(f"release/{week}/") or ".." in Path(path).parts:
                raise ValueError("invalid remote path")
            expected = entry.get("lfs", {}).get("oid")
            files[path] = download(base + path, args.out / path, entry["size"], expected)
            print(f"{path}: {files[path]['bytes']} bytes", flush=True)
    write_json(receipt_path, {"dataset": DATASET, "revision": args.revision,
                             "license": "CC BY 4.0", "weeks": args.weeks,
                             "files": files})


def projection(obj):
    """Keep an object's replayed fields; a call keeps its block IDs and their count."""
    if "block_ids" in obj:
        obj["proxy_block_count"] = len(obj["block_ids"])
    if "tool_calls" in obj and "provider_tokens" in obj:
        keys = ("step", "turn_index", "step_trigger", "start_ms", "end_ms", "model",
                "provider_tokens", "proxy_block_count", "ttft_ms", "tool_calls", "block_ids")
        return {k: obj[k] for k in keys if k in obj}
    if "call_index" in obj and "subagent" in obj:
        return {k: obj[k] for k in ("call_index", "latency_ms", "subagent") if k in obj}
    return obj


def project(value):
    """Apply `projection` to every object bottom-up, as a JSON object hook does.
    Source lists hold only objects or only scalars, such as block IDs."""
    if isinstance(value, dict):
        return projection({key: project(item) for key, item in value.items()})
    if isinstance(value, list) and value and isinstance(value[0], (dict, list)):
        return [project(item) for item in value]
    return value


def issues(session):
    found = set()
    requests = session.get("requests", [])
    if not requests:
        return ["empty_session"]
    for i, request in enumerate(requests):
        for key in ("start_ms", "end_ms"):
            if type(request.get(key)) is not int:
                found.add("missing_request_clock")
        if "missing_request_clock" not in found:
            if request["end_ms"] < request["start_ms"]:
                found.add("negative_service")
            if i and request["start_ms"] < requests[i - 1]["end_ms"]:
                found.add("overlapping_calls")
        tokens = request.get("provider_tokens") or {}
        for field in ("prompt", "completion"):
            if type(tokens.get(field)) is not int or tokens[field] <= 0:
                found.add(f"nonpositive_or_missing_{field}")
        if all(type(tokens.get(k)) is int for k in ("prompt", "completion")):
            if tokens["prompt"] + tokens["completion"] > 2**32 - 1:
                found.add("token_overflow")
        if request.get("step_trigger") not in ("user", "tool"):
            found.add("unknown_trigger")
        for tool in request.get("tool_calls", []):
            child = tool.get("subagent")
            if child is not None:
                found.update(issues(child))
                if (type(child.get("start_ms")) is int and
                        type(request.get("start_ms")) is int and
                        child["start_ms"] < request["start_ms"]):
                    found.add("subagent_precedes_parent")
    if type(session.get("start_ms")) is not int or session["start_ms"] != requests[0].get("start_ms"):
        found.add("session_start_mismatch")
    return sorted(found)


def all_sessions(session):
    yield session
    for request in session["requests"]:
        for tool in request.get("tool_calls", []):
            child = tool.get("subagent")
            if child is not None:
                yield from all_sessions(child)


def all_calls(session):
    for child in all_sessions(session):
        yield from child["requests"]


def replay_calls(session):
    """Calls in replay order: each call, then its subagents' calls."""
    for request in session["requests"]:
        yield request
        for tool in request.get("tool_calls", []):
            if tool.get("subagent") is not None:
                yield from replay_calls(tool["subagent"])


def valid_ids(blocks):
    return isinstance(blocks, list) and all(type(b) is int and 0 <= b < 2**64 for b in blocks)


def common_prefix(a, b):
    """Length of the longest common prefix, comparing slices."""
    if b[:len(a)] == a:
        return len(a)
    low, high = 0, min(len(a), len(b))
    while low < high:
        middle = (low + high + 1) // 2
        if a[:middle] == b[:middle]:
            low = middle
        else:
            high = middle - 1
    return low


def extract_deltas(session):
    """Remove each call's original block IDs and return, in replay order,
    [shared, new IDs] against the previous call of the same agent, or None
    for a missing or invalid list."""
    deltas = {}
    for thread in all_sessions(session):
        previous = []
        for request in thread["requests"]:
            blocks = request.pop("block_ids", None)
            if valid_ids(blocks):
                shared = common_prefix(previous, blocks)
                deltas[id(request)] = [shared, blocks[shared:]]
                previous = blocks
            else:
                deltas[id(request)], previous = None, []
    return [deltas[id(request)] for request in replay_calls(session)]


def attach_deltas(session, deltas):
    for request, delta in zip(replay_calls(session), deltas, strict=True):
        request["block_delta"] = delta


def index_file(task):
    """Verify and project one compressed source file, read once: its sessions
    with their block deltas, its integrity record and its counts."""
    root, week, path, compressed_sha, expected = task
    data = (root / path).read_bytes()
    if hashlib.sha256(data).hexdigest() != compressed_sha:
        raise ValueError(f"compressed source changed: {path}")
    digest, size, lines, rows, counts = hashlib.sha256(), 0, 0, [], Counter()
    with zstd.open(io.BytesIO(data), "rb") as stream:
        for lines, raw in enumerate(stream, 1):
            digest.update(raw)
            size += len(raw)
            session = project(orjson.loads(raw))
            if session.get("week") != week:
                raise ValueError("session week differs from source path")
            session["source"] = {"file": path, "line": lines}
            session["replay_issues"] = issues(session)
            # Block IDs are kept once, as deltas between consecutive calls.
            rows.append((session, json_line(extract_deltas(session))))
            counts["sessions"] += 1
            counts["top_level_llm_calls"] += len(session["requests"])
            counts["all_llm_calls"] += sum(1 for _ in all_calls(session))
            counts["nested_sessions"] += sum(1 for _ in all_sessions(session)) - 1
            counts.update(session["replay_issues"])
    actual = {"bytes": size, "lines": lines, "sha256": digest.hexdigest()}
    if actual != expected:
        raise ValueError(f"publisher integrity mismatch: {path}: {actual}")
    return path, actual, rows, counts


def index(args):
    source = json.loads((args.source / "source.json").read_text())
    fresh(args.out)
    tasks = []
    for week in source["weeks"]:
        manifest_path = f"release/{week}/manifest.json"
        if sha256(args.source / manifest_path) != source["files"][manifest_path]["sha256"]:
            raise ValueError("publisher manifest changed")
        manifest = json.loads((args.source / manifest_path).read_text())
        for relative, expected in sorted(manifest["files"].items()):
            if not re.fullmatch(r"traces/\d{4}-\d{2}-\d{2}\.jsonl", relative):
                raise ValueError("invalid manifest trace path")
            path = f"release/{week}/{relative}.zst"
            tasks.append((args.source, week, path, source["files"][path]["sha256"], expected))
    seen, counts, checked = set(), Counter(), {}
    catalog, blocks = args.out / "sessions.jsonl", args.out / "blocks.jsonl"
    # Files are processed in parallel and written in source order.
    with (ProcessPoolExecutor(getattr(args, "jobs", None)) as pool,
          catalog.open("w") as out, blocks.open("w") as block_out):
        for path, actual, rows, file_counts in pool.map(index_file, tasks):
            for session, deltas in rows:
                key = (session["week"], session["session_id"])
                if key in seen:
                    raise ValueError(f"duplicate session {key}")
                seen.add(key)
                session["blocks_offset"] = block_out.tell()
                block_out.write(deltas)
                out.write(json_line(session))
            counts.update(file_counts)
            checked[path] = actual
            print(f"verified {path}: {actual['lines']} sessions", flush=True)
    write_json(args.out / "index.json", {"source": str(args.source.resolve()),
               "source_sha256": sha256(args.source / "source.json"),
               "dataset": source["dataset"], "revision": source["revision"],
               "license": source["license"], "counts": counts, "verified": checked,
               "sessions_sha256": sha256(catalog), "blocks_sha256": sha256(blocks)})


def arrivals(sessions, mode, rate, seed, until=None):
    """One arrival per session, or with `until`, every arrival before it."""
    if mode == "original":
        if rate is not None:
            raise ValueError("original arrivals do not take sessions-per-second")
        origin = sessions[0]["start_ms"]
        times = [(s["start_ms"] - origin) / 1000 for s in sessions]
        return times if until is None else [t for t in times if t < until]
    if rate is None or not math.isfinite(rate) or rate <= 0:
        raise ValueError("sessions-per-second must be finite and positive")
    count = len(sessions) if until is None else math.inf
    if mode == "regular":
        times, i = [], 0
        while i < count and (until is None or i / rate < until):
            times.append(i / rate)
            i += 1
        return times
    if mode != "poisson":
        raise ValueError("unknown arrival mode")
    rng = random.Random(seed)
    times = [0.0]
    while len(times) < count:
        arrival = times[-1] + rng.expovariate(rate)
        if until is not None and arrival >= until:
            break
        times.append(arrival)
    return times


def earliest_starts(row):
    """Each node's start if every LLM call took no time: the session's
    offered schedule, before any serving delay."""
    nodes = row["sub_requests"]
    degrees = [0] * len(nodes)
    for node in nodes:
        for child in node["next"]:
            degrees[child] += 1
    starts = [row["arrival_time"]] * len(nodes)
    ready = list(row["initial"])
    while ready:
        index = ready.pop()
        node = nodes[index]
        starts[index] += node.get("interval", 0)
        end = starts[index] + (node["duration"] if node["kind"] == "tool_call" else 0)
        for child in node["next"]:
            starts[child] = max(starts[child], end)
            degrees[child] -= 1
            if degrees[child] == 0:
                ready.append(child)
    return starts


def truncate(row, until):
    """Drop the nodes that cannot start before `until`. LLM calls count as
    instantaneous, so every node that might start in time is kept, and the kept
    nodes are closed under predecessors. Returns the row and old→new indices."""
    nodes = row["sub_requests"]
    starts = earliest_starts(row)
    kept = {old: new for new, old in enumerate(i for i in range(len(nodes)) if starts[i] < until)}
    sub_requests = [{**nodes[old], "next": [kept[c] for c in nodes[old]["next"] if c in kept]}
                    for old in kept]
    for node in sub_requests:  # an identity parent precedes its call, so it is kept
        if "parent" in node.get("kv_prefix", {}):
            node["kv_prefix"] = {**node["kv_prefix"], "parent": kept[node["kv_prefix"]["parent"]]}
    return {**row, "initial": [kept[i] for i in row["initial"] if i in kept],
            "sub_requests": sub_requests}, kept


def identity(parts):
    return json.dumps(parts, separators=(",", ":"), ensure_ascii=True)


def condense(blocks, group):
    """One ID per `group` consecutive proxy blocks, hashing their IDs; a final
    partial group is dropped, as the pool drops it. Prefixes share a condensed
    ID exactly when they share its original IDs."""
    data = array("Q", blocks)
    if sys.byteorder == "big":
        data.byteswap()
    data, width = memoryview(data).cast("B"), 8 * group  # little-endian u64 IDs
    return [int.from_bytes(hashlib.blake2b(data[at:at + width], digest_size=8).digest(), "little")
            for at in range(0, len(data) - width + 1, width)]


def native(session, request_id, arrival, *, token_mode="provider-count",
           dataset=DATASET, revision=REVISION, target_model=None, kv_layout=None,
           sharing_scope="shared", copy=0, prefix_block_tokens=PROXY_BLOCK_TOKENS):
    """`copy` > 0 replays a session again with its own identities, so copies
    never share KV with the original. Proxy-prefix calls carry `block_delta`
    (see `extract_deltas`); each identity lists only the condensed IDs after
    those it shares with the previous call of its agent, its `parent`."""
    replay = [f"copy{copy}"] if copy else []
    if token_mode not in ("provider-count", "proxy-prefix"):
        raise ValueError("unknown token mode")
    if token_mode == "proxy-prefix" and not all((target_model, kv_layout, sharing_scope)):
        raise ValueError("proxy-prefix requires target-model, kv-layout and sharing-scope")
    problems = issues(session)
    if problems:
        raise ValueError(f"unsupported session {session['session_id']}: {problems}")
    nodes, mapping, namespaces = [], [], {}

    def marker(parents, at_ms, session_id, as_tool=False):
        delay = (at_ms - max((clock for _, clock in parents), default=session["start_ms"])) / 1000
        if delay < 0:
            raise ValueError("negative dependency delay")
        idx = len(nodes)
        nodes.append({"kind": "tool_call", "input_tokens": 0, "output_tokens": 0,
                      "duration": delay if as_tool else 0,
                      "interval": 0 if as_tool else delay, "session_id": session_id, "next": []})
        for parent, _ in parents:
            nodes[parent]["next"].append(idx)
        return idx

    seen = set()

    def build(current, launch_parents, path):
        if current["session_id"] in seen:
            raise ValueError("duplicate nested session identity")
        seen.add(current["session_id"])
        session_id = identity([dataset, revision, session["week"], current["session_id"], *replay])
        previous, children, terminals = None, [], []
        blocks, parent = [], None  # this agent's last original IDs and (node, namespace)
        for position, request in enumerate(current["requests"]):
            parents = launch_parents if previous is None else [previous]
            # Timing-derived join: a child finishing before the next parent call
            # gates it. A child overlapping that call remains an asynchronous branch.
            if previous is not None:
                parents += [child for child in children if child[1] <= request["start_ms"]]
            gate = marker(parents, request["start_ms"], session_id,
                          previous is not None and request["step_trigger"] == "tool")
            llm = len(nodes)
            nodes[gate]["next"].append(llm)
            provider = {key: request["provider_tokens"][key] for key in ("prompt", "completion")}
            node = {"kind": "llm", "input_tokens": provider["prompt"],
                    "output_tokens": provider["completion"], "provider_tokens": provider,
                    "session_id": session_id,
                    "model": request["model"], "next": []}
            if token_mode == "proxy-prefix":
                delta = request.get("block_delta")
                # The index validated each ID; None marks a missing or invalid list.
                if delta is None or not 0 <= delta[0] <= len(blocks):
                    raise ValueError("proxy-prefix requires original unsigned 64-bit block IDs")
                del blocks[delta[0]:]
                blocks.extend(delta[1])
                node["input_tokens"] = PROXY_BLOCK_TOKENS * len(blocks)
                if node["input_tokens"] + node["output_tokens"] >= 2**32:
                    raise ValueError("proxy-prefix token count exceeds u32")
                if request["model"] not in namespaces:
                    namespaces[request["model"]] = identity([dataset, revision, session["week"], request["model"],
                                                             target_model, kv_layout, sharing_scope, *replay])
                namespace = namespaces[request["model"]]
                group = prefix_block_tokens // PROXY_BLOCK_TOKENS
                # Condensed IDs before the first differing original group are the parent's.
                shared = delta[0] // group if parent and parent[1] == namespace else 0
                tail = blocks[shared * group:]
                prefix = {"namespace": namespace, "block_tokens": prefix_block_tokens}
                if shared:
                    prefix |= {"parent": parent[0], "shared": shared}
                prefix["block_ids"] = tail if group == 1 else condense(tail, group)
                node["kv_prefix"] = prefix
                parent = (llm, namespace)
            nodes.append(node)
            location = f"{path}/requests/{position}"
            mapping.append({"subrequest_index": llm, "source_step": request["step"],
                            "source_path": location, "session_id": current["session_id"],
                            "source_start_ms": request["start_ms"],
                            "source_end_ms": request["end_ms"]})
            children = []
            for tool_index, tool in enumerate(request.get("tool_calls", [])):
                child = tool.get("subagent")
                if child is None:
                    continue
                # Mid-response launches use the parent's release marker, not a
                # fabricated token index. Later launches follow LLM completion.
                anchor = ((gate, request["start_ms"]) if child["start_ms"] < request["end_ms"]
                          else (llm, request["end_ms"]))
                children.append(build(child, [anchor], f"{location}/tool_calls/{tool_index}/subagent"))
            terminals.extend(children)
            previous = (llm, request["end_ms"])
        terminals.append(previous)
        completion_ms = max(clock for _, clock in terminals)
        return marker(terminals, completion_ms, session_id), completion_ms

    build(session, [], "")
    return {"request_id": request_id, "arrival_time": arrival, "initial": [0],
            "sub_requests": nodes}, mapping


def generate(args):
    metadata = json.loads((args.index / "index.json").read_text())
    catalog = args.index / "sessions.jsonl"
    if sha256(catalog) != metadata["sessions_sha256"]:
        raise ValueError("session index changed")
    token_mode = getattr(args, "token_mode", "provider-count")
    options = {"token_mode": token_mode, "dataset": metadata["dataset"], "revision": metadata["revision"],
               "target_model": getattr(args, "target_model", None),
               "kv_layout": getattr(args, "kv_layout", None),
               "sharing_scope": getattr(args, "sharing_scope", "shared")}
    prefix_block_tokens = getattr(args, "prefix_block_tokens", PROXY_BLOCK_TOKENS)
    if prefix_block_tokens % PROXY_BLOCK_TOKENS:
        raise ValueError("prefix block tokens must be a multiple of the proxy block")
    if token_mode not in ("provider-count", "proxy-prefix"):
        raise ValueError("unknown token mode")
    if token_mode == "proxy-prefix" and not all(options[k] for k in ("target_model", "kv_layout", "sharing_scope")):
        raise ValueError("proxy-prefix requires target-model, kv-layout and sharing-scope")
    blocks = args.index / "blocks.jsonl"
    if token_mode == "proxy-prefix" and ("blocks_sha256" not in metadata or sha256(blocks) != metadata["blocks_sha256"]):
        raise ValueError("index lacks matching block deltas; rebuild it with this version")
    selected, excluded, seen_ids = [], [], set()
    with catalog.open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            session = orjson.loads(line)
            seen_ids.add(session["session_id"])
            reasons = list(session["replay_issues"])
            if args.session_ids and session["session_id"] not in args.session_ids:
                reasons.append("session_ids")
            requests = list(all_calls(session))
            if args.max_calls is not None and len(requests) > args.max_calls:
                reasons.append("max_calls")
            for option, field in (("max_input_tokens", "prompt"), ("max_output_tokens", "completion")):
                limit = getattr(args, option)
                values = ([PROXY_BLOCK_TOKENS * r.get("proxy_block_count", 0) for r in requests]
                          if token_mode == "proxy-prefix" and field == "prompt" else
                          [(r.get("provider_tokens") or {}).get(field) for r in requests])
                if limit is not None and any(type(v) is int and v > limit for v in values):
                    reasons.append(option)
            if reasons:
                excluded.append({"week": session["week"], "session_id": session["session_id"],
                                 "reasons": reasons})
            else:
                selected.append({"offset": offset, "blocks_offset": session.get("blocks_offset"),
                                 "start_ms": session["start_ms"],
                                 "week": session["week"], "session_id": session["session_id"],
                                 "source": session["source"]})
    if args.session_ids and set(args.session_ids) - seen_ids:
        raise ValueError("requested session ID absent from index")
    if args.invalid == "error" and any(
            "session_ids" not in row["reasons"] and
            set(row["reasons"]) - {"max_calls", "max_input_tokens", "max_output_tokens"}
            for row in excluded):
        raise ValueError("unsupported sessions present; inspect index or explicitly use --invalid exclude")
    selected.sort(key=lambda s: (s["start_ms"], s["week"], s["session_id"]))
    selection_seed = getattr(args, "selection_seed", None)
    until = getattr(args, "until", None)
    if selection_seed is not None:
        if args.arrivals == "original":
            raise ValueError("original arrivals keep source order; omit selection-seed")
        random.Random(selection_seed).shuffle(selected)
    if until is not None and not (math.isfinite(until) and until > 0):
        raise ValueError("until must be finite and positive")
    if until is not None and args.limit is not None:
        raise ValueError("until sets the session count; omit limit")
    if args.limit is not None:
        excluded.extend({"week": s["week"], "session_id": s["session_id"], "reasons": ["limit"]}
                        for s in selected[args.limit:])
        selected = selected[:args.limit]
    if not selected:
        raise ValueError("no eligible sessions")
    times = arrivals(selected, args.arrivals, args.sessions_per_second, args.seed, until)
    # Past the eligible sessions, replay them in the same order as copies.
    picks = [(selected[i % len(selected)], i // len(selected)) for i in range(len(times))]
    selected = selected[:len(times)]
    fresh(args.out)
    mappings, call_count, nested_count, offered = [], 0, 0, Counter()
    with (catalog.open("rb") as source,
          blocks.open("rb") if token_mode == "proxy-prefix" else contextlib.nullcontext() as deltas,
          (args.out / "workload.jsonl").open("w") as stream):
        for request_id, ((selection, copy), arrival) in enumerate(zip(picks, times, strict=True)):
            source.seek(selection["offset"])
            session = orjson.loads(source.readline())
            if deltas:
                deltas.seek(selection["blocks_offset"])
                attach_deltas(session, orjson.loads(deltas.readline()))
            row, calls = native(session, request_id, arrival, copy=copy,
                                prefix_block_tokens=prefix_block_tokens, **options)
            for node, start in zip(row["sub_requests"], earliest_starts(row)):
                if node["kind"] == "llm" and (until is None or start < until):
                    offered[int(start // 60)] += 1
            if until is not None:
                row, kept = truncate(row, until)
                calls = [{**call, "subrequest_index": kept[call["subrequest_index"]]}
                         for call in calls if call["subrequest_index"] in kept]
            stream.write(json_line(row))
            call_count += len(calls)
            nested_count += sum(1 for _ in all_sessions(session)) - 1
            mappings.append({"request_id": request_id, "week": session["week"],
                             "session_id": session["session_id"], "copy": copy,
                             "source": session["source"],
                             "source_start_ms": session["start_ms"],
                             "calls": [{key: call[key] for key in
                                       ("subrequest_index", "source_path", "session_id")} for call in calls]})
    manifest = {"index": str(args.index.resolve()), "index_sha256": sha256(args.index / "index.json"),
                "arrivals": args.arrivals, "sessions_per_second": args.sessions_per_second,
                "seed": args.seed, "selection_seed": selection_seed, "until": until,
                "first_arrival": times[0], "last_arrival": times[-1],
                "filters": {k: getattr(args, k) for k in ("invalid", "limit", "max_calls",
                            "max_input_tokens", "max_output_tokens", "session_ids")},
                "sessions": len(picks), "copies": picks[-1][1] + 1, "llm_calls": call_count,
                "nested_sessions": nested_count,
                "offered_llm_calls_per_minute": [offered[m] for m in range(max(offered, default=-1) + 1)],
                "token_mode": token_mode,
                "identity": {key: options[key] for key in
                             ("dataset", "revision", "target_model", "kv_layout", "sharing_scope")},
                "fidelity": {
                    "proxy_tokenizer": "o200k_base", "proxy_block_tokens": PROXY_BLOCK_TOKENS,
                    "prefix_block_tokens": prefix_block_tokens,
                    "omitted_partial_proxy_tokens_per_prompt": [0, PROXY_BLOCK_TOKENS - 1],
                    "provider_prompt_preserved": True,
                    "completion_length": "Provider count approximation; not proxy-tokenizer equivalent.",
                    "unrecoverable_reuse": ["Dropped partial final blocks.",
                                           "Generated decode KV to subsequent input blocks."],
                    "decode_to_input_identity": False},
                "selected": mappings, "excluded": excluded,
                "workload_sha256": sha256(args.out / "workload.jsonl"),
                "semantics": ("Provider counts without prefix identity. " if token_mode == "provider-count" else
                              "Complete proxy blocks define input length and identity; native target model determines KV bytes. ") +
                             "Provider counts remain provenance; source model does not choose compute costs. Mid-response subagent launches use observed delay from parent release. Children completing before the next parent call gate it; overlapping children drain asynchronously. Tool/user residual gaps follow simulated dependencies."}
    write_json(args.out / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("sessions", "llm_calls", "first_arrival", "last_arrival")}))


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    command = commands.add_parser("fetch")
    command.add_argument("--revision", default=REVISION)
    selection = command.add_mutually_exclusive_group()
    selection.add_argument("--weeks", nargs="+", default=[WEEK])
    selection.add_argument("--all-weeks", action="store_true")
    command.add_argument("--out", type=Path, required=True)
    command.set_defaults(run=fetch)
    command = commands.add_parser("index")
    command.add_argument("--source", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--jobs", type=positive_int, default=os.process_cpu_count(),
                         help="source files processed at once")
    command.set_defaults(run=index)
    command = commands.add_parser("generate")
    command.add_argument("--index", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--arrivals", choices=["original", "regular", "poisson"], required=True)
    command.add_argument("--sessions-per-second", type=float)
    command.add_argument("--seed", type=int, default=0, help="arrival gaps")
    command.add_argument("--selection-seed", type=int,
                         help="shuffle eligible sessions; default keeps source start order")
    command.add_argument("--until", type=float,
                         help="arrivals until this many seconds; drop nodes that cannot start before it")
    command.add_argument("--invalid", choices=["error", "exclude"], default="error")
    command.add_argument("--session-ids", nargs="+")
    command.add_argument("--token-mode", choices=["provider-count", "proxy-prefix"], default="provider-count")
    command.add_argument("--target-model", help="Target model/revision identity required by proxy-prefix")
    command.add_argument("--kv-layout", help="Target KV layout/dtype identity required by proxy-prefix")
    command.add_argument("--sharing-scope", default="shared", help="Prefix-cache sharing namespace")
    command.add_argument("--prefix-block-tokens", type=positive_int, default=256,
                         help="one prefix ID per this many tokens (a multiple of 16); match the "
                              "pool's block_tokens, or pool reuse misses")
    for option in ("limit", "max-calls", "max-input-tokens", "max-output-tokens"):
        command.add_argument("--" + option, type=positive_int)
    command.set_defaults(run=generate)
    args = parser.parse_args()
    try:
        args.run(args)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
