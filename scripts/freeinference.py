"""Fetch, index and replay FreeInference sessions (Python 3.14+)."""

import argparse
from collections import Counter
from compression import zstd
import hashlib
import json
import math
from pathlib import Path
import random
import re
import shutil
import tempfile
import urllib.request

DATASET = "harvardMadsys/freeinference_agentic_trace"
REVISION = "bb26cf72c9c20f2b05f765d88548156a74d3c253"
WEEK = "2026-08-30_2026-09-05"
HUB = "https://huggingface.co"
PROXY_BLOCK_TOKENS = 16


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with path.open("w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def json_line(value):
    return json.dumps(value, separators=(",", ":"), allow_nan=False) + "\n"


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


def projection(obj, keep_blocks=False):
    """Release large block arrays as soon as their request object is parsed."""
    if "block_ids" in obj:
        obj["proxy_block_count"] = len(obj["block_ids"])
        if not keep_blocks:
            del obj["block_ids"]
    if "tool_calls" in obj and "provider_tokens" in obj:
        keys = ("step", "turn_index", "step_trigger", "start_ms", "end_ms", "model",
                "provider_tokens", "proxy_block_count", "ttft_ms", "tool_calls", "block_ids")
        return {k: obj[k] for k in keys if k in obj}
    if "call_index" in obj and "subagent" in obj:
        return {k: obj[k] for k in ("call_index", "latency_ms", "subagent") if k in obj}
    return obj


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


def index(args):
    source = json.loads((args.source / "source.json").read_text())
    fresh(args.out)
    seen, counts, checked = set(), Counter(), {}
    catalog = args.out / "sessions.jsonl"
    with catalog.open("w") as out:
        for week in source["weeks"]:
            manifest_path = f"release/{week}/manifest.json"
            if sha256(args.source / manifest_path) != source["files"][manifest_path]["sha256"]:
                raise ValueError("publisher manifest changed")
            manifest = json.loads((args.source / manifest_path).read_text())
            for relative, expected in sorted(manifest["files"].items()):
                if not re.fullmatch(r"traces/\d{4}-\d{2}-\d{2}\.jsonl", relative):
                    raise ValueError("invalid manifest trace path")
                path = f"release/{week}/{relative}.zst"
                if sha256(args.source / path) != source["files"][path]["sha256"]:
                    raise ValueError(f"compressed source changed: {path}")
                digest, size, lines = hashlib.sha256(), 0, 0
                with zstd.open(args.source / path, "rb") as stream:
                    for lines, raw in enumerate(stream, 1):
                        digest.update(raw)
                        size += len(raw)
                        session = json.loads(raw, object_hook=projection)
                        key = (week, session["session_id"])
                        if key in seen:
                            raise ValueError(f"duplicate session {key}")
                        if session.get("week") != week:
                            raise ValueError("session week differs from source path")
                        seen.add(key)
                        session["source"] = {"file": path, "line": lines}
                        session["replay_issues"] = issues(session)
                        out.write(json_line(session))
                        counts["sessions"] += 1
                        counts["top_level_llm_calls"] += len(session["requests"])
                        counts["all_llm_calls"] += sum(1 for _ in all_calls(session))
                        counts["nested_sessions"] += sum(1 for _ in all_sessions(session)) - 1
                        counts.update(session["replay_issues"])
                actual = {"bytes": size, "lines": lines, "sha256": digest.hexdigest()}
                if actual != expected:
                    raise ValueError(f"publisher integrity mismatch: {path}: {actual}")
                checked[path] = actual
                print(f"verified {path}: {lines} sessions", flush=True)
    write_json(args.out / "index.json", {"source": str(args.source.resolve()),
               "source_sha256": sha256(args.source / "source.json"),
               "dataset": source["dataset"], "revision": source["revision"],
               "license": source["license"], "counts": counts, "verified": checked,
               "sessions_sha256": sha256(catalog)})


def arrivals(sessions, mode, rate, seed):
    if mode == "original":
        if rate is not None:
            raise ValueError("original arrivals do not take sessions-per-second")
        origin = sessions[0]["start_ms"]
        return [(s["start_ms"] - origin) / 1000 for s in sessions]
    if rate is None or not math.isfinite(rate) or rate <= 0:
        raise ValueError("sessions-per-second must be finite and positive")
    if mode == "regular":
        return [i / rate for i in range(len(sessions))]
    if mode != "poisson":
        raise ValueError("unknown arrival mode")
    rng = random.Random(seed)
    times = [0.0]
    for _ in sessions[1:]:
        times.append(times[-1] + rng.expovariate(rate))
    return times


def identity(parts):
    return json.dumps(parts, separators=(",", ":"), ensure_ascii=True)


def native(session, request_id, arrival, *, token_mode="provider-count",
           dataset=DATASET, revision=REVISION, target_model=None, kv_layout=None,
           sharing_scope="shared"):
    if token_mode not in ("provider-count", "proxy-prefix"):
        raise ValueError("unknown token mode")
    if token_mode == "proxy-prefix" and not all((target_model, kv_layout, sharing_scope)):
        raise ValueError("proxy-prefix requires target-model, kv-layout and sharing-scope")
    problems = issues(session)
    if problems:
        raise ValueError(f"unsupported session {session['session_id']}: {problems}")
    nodes, mapping = [], []

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
        session_id = identity([dataset, revision, session["week"], current["session_id"]])
        previous, children, terminals = None, [], []
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
                blocks = request.get("block_ids")
                if not isinstance(blocks, list) or any(
                        type(block) is not int or not 0 <= block < 2**64 for block in blocks):
                    raise ValueError("proxy-prefix requires original unsigned 64-bit block IDs")
                node["input_tokens"] = PROXY_BLOCK_TOKENS * len(blocks)
                if node["input_tokens"] + node["output_tokens"] >= 2**32:
                    raise ValueError("proxy-prefix token count exceeds u32")
                node["kv_prefix"] = {
                    "namespace": identity([dataset, revision, session["week"], request["model"],
                                           target_model, kv_layout, sharing_scope]),
                    "block_tokens": PROXY_BLOCK_TOKENS, "block_ids": blocks}
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


def recover_blocks(metadata, selected, spool):
    """Recover selected original lists once per file; keep one session in memory."""
    root = Path(metadata["source"])
    receipt = root / "source.json"
    if sha256(receipt) != metadata["source_sha256"]:
        raise ValueError("indexed source receipt changed")
    source = json.loads(receipt.read_text())
    if any(source[field] != metadata[field] for field in ("dataset", "revision")):
        raise ValueError("indexed source identity changed")
    files = {}
    for selection in selected:
        location = selection["source"]
        files.setdefault(location["file"], {})[location["line"]] = selection
    for relative, wanted in sorted(files.items()):
        if relative not in source["files"] or ".." in Path(relative).parts or Path(relative).is_absolute():
            raise ValueError("invalid indexed source path")
        if sha256(root / relative) != source["files"][relative]["sha256"]:
            raise ValueError(f"compressed source changed: {relative}")
        with zstd.open(root / relative, "rb") as stream:
            for line, raw in enumerate(stream, 1):
                selection = wanted.pop(line, None)
                if selection is not None:
                    session = json.loads(raw, object_hook=lambda obj: projection(obj, keep_blocks=True))
                    if any(session[field] != selection[field] for field in ("week", "session_id", "start_ms")):
                        raise ValueError("indexed session differs from original source")
                    session["source"] = selection["source"]
                    selection["block_offset"] = spool.tell()
                    spool.write(json_line(session).encode())
                if not wanted:
                    break
        if wanted:
            raise ValueError(f"indexed source lines missing: {relative}")


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
    if token_mode not in ("provider-count", "proxy-prefix"):
        raise ValueError("unknown token mode")
    if token_mode == "proxy-prefix" and not all(options[k] for k in ("target_model", "kv_layout", "sharing_scope")):
        raise ValueError("proxy-prefix requires target-model, kv-layout and sharing-scope")
    selected, excluded, seen_ids = [], [], set()
    with catalog.open("rb") as stream:
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line:
                break
            session = json.loads(line)
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
                selected.append({"offset": offset, "start_ms": session["start_ms"],
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
    if args.limit is not None:
        excluded.extend({"week": s["week"], "session_id": s["session_id"], "reasons": ["limit"]}
                        for s in selected[args.limit:])
        selected = selected[:args.limit]
    if not selected:
        raise ValueError("no eligible sessions")
    times = arrivals(selected, args.arrivals, args.sessions_per_second, args.seed)
    fresh(args.out)
    mappings, call_count, nested_count = [], 0, 0
    with tempfile.TemporaryFile() as spool, catalog.open("rb") as source, (args.out / "workload.jsonl").open("w") as stream:
        if token_mode == "proxy-prefix":
            recover_blocks(metadata, selected, spool)
        for request_id, (selection, arrival) in enumerate(zip(selected, times, strict=True)):
            if token_mode == "proxy-prefix":
                spool.seek(selection["block_offset"])
                session = json.loads(spool.readline())
            else:
                source.seek(selection["offset"])
                session = json.loads(source.readline())
            row, calls = native(session, request_id, arrival, **options)
            stream.write(json_line(row))
            call_count += len(calls)
            nested_count += sum(1 for _ in all_sessions(session)) - 1
            mappings.append({"request_id": request_id, "week": session["week"],
                             "session_id": session["session_id"], "source": session["source"],
                             "source_start_ms": session["start_ms"],
                             "calls": [{key: call[key] for key in
                                       ("subrequest_index", "source_path", "session_id")} for call in calls]})
    manifest = {"index": str(args.index.resolve()), "index_sha256": sha256(args.index / "index.json"),
                "arrivals": args.arrivals, "sessions_per_second": args.sessions_per_second,
                "seed": args.seed, "first_arrival": times[0], "last_arrival": times[-1],
                "filters": {k: getattr(args, k) for k in ("invalid", "limit", "max_calls",
                            "max_input_tokens", "max_output_tokens", "session_ids")},
                "sessions": len(selected), "llm_calls": call_count,
                "nested_sessions": nested_count,
                "token_mode": token_mode,
                "identity": {key: options[key] for key in
                             ("dataset", "revision", "target_model", "kv_layout", "sharing_scope")},
                "fidelity": {
                    "proxy_tokenizer": "o200k_base", "proxy_block_tokens": PROXY_BLOCK_TOKENS,
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
    command.set_defaults(run=index)
    command = commands.add_parser("generate")
    command.add_argument("--index", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--arrivals", choices=["original", "regular", "poisson"], required=True)
    command.add_argument("--sessions-per-second", type=float)
    command.add_argument("--seed", type=int, default=0)
    command.add_argument("--invalid", choices=["error", "exclude"], default="error")
    command.add_argument("--session-ids", nargs="+")
    command.add_argument("--token-mode", choices=["provider-count", "proxy-prefix"], default="provider-count")
    command.add_argument("--target-model", help="Target model/revision identity required by proxy-prefix")
    command.add_argument("--kv-layout", help="Target KV layout/dtype identity required by proxy-prefix")
    command.add_argument("--sharing-scope", default="shared", help="Prefix-cache sharing namespace")
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
