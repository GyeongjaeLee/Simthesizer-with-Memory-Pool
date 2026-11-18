import argparse
import copy
from compression import zstd
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import freeinference as fi
from results import parse_request, timeline


def call(start, end, step=0, trigger="user", tools=None):
    return {"step": step, "turn_index": 0, "step_trigger": trigger,
            "start_ms": start * 1000, "end_ms": end * 1000, "model": "source-model",
            "provider_tokens": {"prompt": 32, "completion": 2},
            "block_ids": [7, 8], "tool_calls": tools or []}


def session(name, requests, **extra):
    return {"week": fi.WEEK, "session_id": name,
            "start_ms": requests[0]["start_ms"], "end_ms": requests[-1]["end_ms"],
            "requests": requests, **extra}


def nested(child_start=4, child_end=12):
    child = session("child", [call(child_start, child_end)])
    return session("parent", [call(0, 10, tools=[{"subagent": child, "latency_ms": 5000}]),
                              call(15, 17, 1, "tool")])


def reference_result(source, arrival=0):
    row, mapping = fi.native(source, 0, arrival)
    for entry in mapping:
        node = row["sub_requests"][entry["subrequest_index"]]
        duration = (entry["source_end_ms"] - entry["source_start_ms"]) / 1000
        node["token_latency"] = [duration] + [0] * (node["output_tokens"] - 1)
    return parse_request(row), mapping


class SessionConversionTests(unittest.TestCase):
    def assert_source_clock(self, source):
        row, mapping = reference_result(source, arrival=7)
        starts, ends = timeline(row)
        for entry in mapping:
            i = entry["subrequest_index"]
            self.assertAlmostEqual(starts[i], 7 + (entry["source_start_ms"] - source["start_ms"]) / 1000)
            self.assertAlmostEqual(ends[i], 7 + (entry["source_end_ms"] - source["start_ms"]) / 1000)
        return row, mapping, starts, ends

    def test_tool_gap_is_one_wall_interval_and_user_gap_is_preserved(self):
        source = session("s", [call(100, 102, tools=[{"latency_ms": 3000}] * 4),
                               call(105, 107, 1, "tool"), call(117, 120, 2)])
        original = copy.deepcopy(source)
        row, mapping, starts, _ = self.assert_source_clock(source)
        self.assertEqual([starts[x["subrequest_index"]] for x in mapping], [7, 12, 24])
        self.assertEqual(sum(n.get("duration", 0) for n in row["sub_requests"]), 3)
        self.assertEqual(source, original)

    def test_mid_response_fork_join_has_no_double_charged_gap(self):
        row, mapping, starts, ends = self.assert_source_clock(nested())
        self.assertEqual([starts[x["subrequest_index"]] for x in mapping], [7, 11, 22])
        self.assertEqual(max(ends), 24)
        child = row["sub_requests"][mapping[1]["subrequest_index"]]
        child["token_latency"] = [20, 0]
        starts, _ = timeline(row)
        self.assertEqual(starts[mapping[2]["subrequest_index"]], 34)  # 7 + 4 + 20 + 3

    def test_overlapping_child_does_not_serialize_next_parent_call(self):
        row, mapping, _, _ = self.assert_source_clock(nested(child_end=20))
        row["sub_requests"][mapping[1]["subrequest_index"]]["token_latency"] = [40, 0]
        starts, ends = timeline(row)
        self.assertEqual(starts[mapping[2]["subrequest_index"]], 22)
        self.assertEqual(max(ends), 51)  # root 7 + launch 4 + child 40

    def test_tools_and_terminal_join_keep_source_session_membership(self):
        row, mapping, _, ends = self.assert_source_clock(nested(child_end=20))
        parent = row["sub_requests"][mapping[0]["subrequest_index"]]["session_id"]
        child = row["sub_requests"][mapping[1]["subrequest_index"]]["session_id"]
        self.assertNotEqual(parent, child)
        self.assertTrue(all(node["session_id"] in (parent, child) for node in row["sub_requests"]))
        terminal = row["sub_requests"][-1]
        self.assertEqual((terminal["kind"], terminal["session_id"], terminal["next"]),
                         ("tool_call", parent, []))
        self.assertEqual(ends[-1], 27)  # Parent lifetime drains the overlapping child.
        self.assertLess(ends[mapping[-1]["subrequest_index"]], ends[-1])

    def test_after_response_launch_and_recursive_children(self):
        source = nested(12, 14)
        child = source["requests"][0]["tool_calls"][0]["subagent"]
        child["requests"][0]["tool_calls"] = [
            {"subagent": session("grandchild", [call(13, 14)])}]
        _, mapping, _, _ = self.assert_source_clock(source)
        self.assertEqual(len(mapping), 4)

    def test_parallel_children_join_latest_completion(self):
        source = nested()
        source["requests"][0]["tool_calls"].append(
            {"subagent": session("second", [call(6, 13)])})
        row, mapping, _, _ = self.assert_source_clock(source)
        self.assertEqual(len(mapping), 4)
        self.assertEqual(sum(n.get("duration", 0) for n in row["sub_requests"]), 2)

    def test_invalid_sessions_are_never_silently_repaired(self):
        for change in (lambda s: s["requests"][0]["provider_tokens"].update(completion=0),
                       lambda s: s["requests"][1].update(start_ms=9000),
                       lambda s: s["requests"][1].update(step_trigger="unknown")):
            source = nested()
            change(source)
            with self.assertRaises(ValueError):
                fi.native(source, 0, 0)

    def test_duplicate_child_identity_is_rejected(self):
        source = nested()
        source["requests"][0]["tool_calls"] *= 2
        with self.assertRaisesRegex(ValueError, "duplicate nested"):
            fi.native(source, 0, 0)

    def test_proxy_ids_are_not_model_tokens(self):
        source = json.loads(json.dumps(nested()), object_hook=fi.projection)
        self.assertEqual(source["requests"][0]["proxy_block_count"], 2)
        row, _ = fi.native(source, 0, 0)
        self.assertTrue(all("input_token_ids" not in n for n in row["sub_requests"]))
        self.assertEqual(row["sub_requests"][1]["input_tokens"], 32)


class PrefixTests(unittest.TestCase):
    def proxy(self, source, **options):
        return fi.native(source, 0, 0, token_mode="proxy-prefix",
                         target_model="target@revision", kv_layout="fp16:kv_heads=8:dim=128", **options)

    def test_provider_count_keeps_provenance_without_cache_identity(self):
        source = nested()
        source["requests"][0]["provider_tokens"]["prompt"] = 37
        row, mapping = fi.native(source, 0, 0)
        node = row["sub_requests"][mapping[0]["subrequest_index"]]
        self.assertEqual(node["input_tokens"], 37)
        self.assertEqual(node["provider_tokens"], {"prompt": 37, "completion": 2})
        self.assertTrue(all("kv_prefix" not in n for n in row["sub_requests"]))
        self.assertEqual(json.loads(node["session_id"]), [fi.DATASET, fi.REVISION, fi.WEEK, "parent"])

    def test_proxy_uses_complete_original_blocks_and_no_decode_identity(self):
        source = nested()
        source["requests"][0]["provider_tokens"]["prompt"] = 37
        row, mapping = self.proxy(source)
        node = row["sub_requests"][mapping[0]["subrequest_index"]]
        self.assertEqual((node["input_tokens"], node["output_tokens"]), (32, 2))
        self.assertEqual(node["provider_tokens"]["prompt"], 37)
        self.assertEqual(node["kv_prefix"]["block_ids"], [7, 8])
        self.assertEqual(node["kv_prefix"]["block_tokens"], 16)
        self.assertTrue(all("input_token_ids" not in n and "output_token_ids" not in n
                            for n in row["sub_requests"]))
        parent, child, parent_next = [row["sub_requests"][m["subrequest_index"]] for m in mapping]
        self.assertEqual(parent["session_id"], parent_next["session_id"])
        self.assertNotEqual(parent["session_id"], child["session_id"])
        self.assertEqual(parent["kv_prefix"], child["kv_prefix"])

    def test_namespace_dimensions_are_independent_of_session_membership(self):
        source = session("s", [call(0, 1)])
        base, _ = self.proxy(source)
        original = base["sub_requests"][1]["kv_prefix"]["namespace"]
        for key in ("dataset", "revision", "sharing_scope"):
            row, _ = self.proxy(source, **{key: "changed"})
            self.assertNotEqual(row["sub_requests"][1]["kv_prefix"]["namespace"], original)
        for key, value in (("week", "other-week"), ("session_id", "another-session")):
            changed = copy.deepcopy(source)
            changed[key] = value
            row, _ = self.proxy(changed)
            self.assertEqual(row["sub_requests"][1]["kv_prefix"]["namespace"] == original,
                             key == "session_id")
        changed = copy.deepcopy(source)
        changed["requests"][0]["model"] = "another-model"
        row, _ = self.proxy(changed)
        self.assertNotEqual(row["sub_requests"][1]["kv_prefix"]["namespace"], original)
        for key in ("target_model", "kv_layout"):
            options = {"target_model": "target@revision", "kv_layout": "fp16:kv_heads=8:dim=128", key: "changed"}
            row, _ = fi.native(source, 0, 0, token_mode="proxy-prefix", **options)
            self.assertNotEqual(row["sub_requests"][1]["kv_prefix"]["namespace"], original)

    def test_missing_invalid_or_partial_id_lists_are_never_reconstructed(self):
        source = session("s", [call(0, 1)])
        for blocks in (None, [True], [-1], [2**64], ["1"]):
            source["requests"][0]["block_ids"] = blocks
            with self.subTest(blocks=blocks), self.assertRaisesRegex(ValueError, "original unsigned"):
                self.proxy(source)
        source["requests"][0].pop("block_ids")
        source["requests"][0]["proxy_block_count"] = 2
        with self.assertRaises(ValueError):
            self.proxy(source)
        source["requests"][0]["block_ids"] = []
        row, _ = self.proxy(source)
        self.assertEqual(row["sub_requests"][1]["input_tokens"], 0)

    def test_modes_preserve_nested_edges_and_source_clock(self):
        source = nested(child_end=20)
        plain, _ = fi.native(source, 0, 7)
        proxy, _ = self.proxy(source)
        for left, right in zip(plain["sub_requests"], proxy["sub_requests"], strict=True):
            for field in ("kind", "next", "interval", "duration", "output_tokens"):
                self.assertEqual(left.get(field), right.get(field))
        for options in ({}, {"target_model": "target"}, {"kv_layout": "layout"}):
            with self.assertRaisesRegex(ValueError, "requires target"):
                fi.native(source, 0, 0, token_mode="proxy-prefix", **options)


class ArrivalTests(unittest.TestCase):
    def test_original_and_regular_session_clock(self):
        sessions = [{"start_ms": 1000}, {"start_ms": 2500}, {"start_ms": 9000}]
        self.assertEqual(fi.arrivals(sessions, "original", None, 0), [0, 1.5, 8])
        self.assertEqual(fi.arrivals(sessions, "regular", 2, 0), [0, .5, 1])
        with self.assertRaises(ValueError):
            fi.arrivals(sessions, "original", 2, 0)

    def test_poisson_reproducibility_rate_scaling_and_distribution(self):
        sessions = [{}] * 100001
        times = fi.arrivals(sessions, "poisson", 2, 31)
        self.assertEqual(times, fi.arrivals(sessions, "poisson", 2, 31))
        self.assertNotEqual(times[:10], fi.arrivals(sessions[:10], "poisson", 2, 32))
        fast = fi.arrivals(sessions, "poisson", 4, 31)
        self.assertEqual(times, [2 * t for t in fast])
        gaps = [b - a for a, b in zip(times, times[1:])]
        # Exponential mean=1/r, variance=1/r². These bounds exceed 5 SE
        # at n=100000 and distinguish a uniform/constant substitute.
        self.assertAlmostEqual(statistics.mean(gaps), .5, delta=.01)
        self.assertAlmostEqual(statistics.pvariance(gaps), .25, delta=.0125)

    def test_invalid_rate(self):
        for rate in (None, 0, -1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                fi.arrivals([{}, {}], "regular", rate, 0)


class SourceTests(unittest.TestCase):
    def test_all_weeks_fetch_selects_raw_files_and_pins_revision(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def listing(url):
                self.assertIn(fi.REVISION, url)
                if "recursive=true" not in url:
                    return [{"type": "directory", "path": f"release/{fi.WEEK}"}]
                return [{"type": "file", "path": f"release/{fi.WEEK}/{p}", "size": 1}
                        for p in ("manifest.json", "traces/day.jsonl.zst", "processed/requests.parquet")]
            def downloaded(url, destination, *args):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"x")
                return {"url": url, "bytes": 1, "sha256": fi.sha256(destination)}
            args = argparse.Namespace(out=root, weeks=[], all_weeks=True, revision=fi.REVISION)
            with patch.object(fi, "get_json", side_effect=listing), patch.object(fi, "download", side_effect=downloaded):
                fi.fetch(args)
            receipt = json.loads((root / "source.json").read_text())
            self.assertEqual(receipt["weeks"], [fi.WEEK])
            self.assertEqual(len(receipt["files"]), 3)
            self.assertFalse(any("parquet" in p for p in receipt["files"]))

    def create_source(self, root, rows, wrong_digest=False):
        relative = "traces/2026-08-30.jsonl"
        path = f"release/{fi.WEEK}/{relative}.zst"
        dest = root / path
        dest.parent.mkdir(parents=True)
        raw = "".join(fi.json_line(row) for row in rows).encode()
        with zstd.open(dest, "wb") as out:
            out.write(raw)
        manifest_path = f"release/{fi.WEEK}/manifest.json"
        fi.write_json(root / manifest_path, {"files": {relative: {
            "bytes": len(raw), "lines": len(rows),
            "sha256": "bad" if wrong_digest else hashlib.sha256(raw).hexdigest()}}})
        fi.write_json(root / "source.json", {"weeks": [fi.WEEK], "dataset": fi.DATASET,
                      "revision": fi.REVISION, "license": "CC BY 4.0",
                      "files": {p: {"sha256": fi.sha256(root / p)} for p in (path, manifest_path)}})

    def test_integrity_and_explicit_whole_session_exclusion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bad = session("bad", [call(0, 1)])
            bad["requests"][0]["provider_tokens"]["completion"] = 0
            self.create_source(root / "raw", [nested(), bad])
            fi.index(argparse.Namespace(source=root / "raw", out=root / "index"))
            args = argparse.Namespace(index=root / "index", out=root / "generated",
                    arrivals="regular", sessions_per_second=2, seed=0, invalid="error",
                    limit=None, max_calls=None, max_input_tokens=None, max_output_tokens=None,
                    session_ids=None)
            with self.assertRaises(ValueError):
                fi.generate(args)
            args.invalid = "exclude"
            fi.generate(args)
            manifest = json.loads((args.out / "manifest.json").read_text())
            self.assertEqual(manifest["sessions"], 1)
            self.assertEqual(manifest["llm_calls"], 3)
            self.assertEqual(manifest["nested_sessions"], 1)
            self.assertEqual(manifest["excluded"][0]["session_id"], "bad")
            with self.assertRaises(FileExistsError):
                fi.generate(args)

    def test_publisher_hash_mismatch_has_no_complete_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.create_source(root / "raw", [nested()], wrong_digest=True)
            with self.assertRaisesRegex(ValueError, "publisher integrity"):
                fi.index(argparse.Namespace(source=root / "raw", out=root / "index"))
            self.assertFalse((root / "index/index.json").exists())

    def test_selection_seeks_whole_sessions_in_time_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            late = session("late", [call(100, 102)])
            late["requests"][0]["provider_tokens"]["prompt"] = 64
            self.create_source(root / "raw", [late, nested()])
            fi.index(argparse.Namespace(source=root / "raw", out=root / "index"))
            args = argparse.Namespace(index=root / "index", out=root / "generated",
                    arrivals="regular", sessions_per_second=2, seed=0, invalid="error",
                    limit=None, max_calls=None, max_input_tokens=None, max_output_tokens=None,
                    session_ids=None)
            fi.generate(args)
            rows = [json.loads(line) for line in (args.out / "workload.jsonl").read_text().splitlines()]
            manifest = json.loads((args.out / "manifest.json").read_text())
            self.assertEqual([r["arrival_time"] for r in rows], [0, .5])
            self.assertEqual([s["session_id"] for s in manifest["selected"]], ["parent", "late"])
            self.assertEqual(rows[1]["sub_requests"][1]["input_tokens"], 64)
            self.assertEqual(manifest["selected"][0]["source"]["line"], 2)

    def test_corrupt_cached_download_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace"
            path.write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "cached hash"):
                fi.download("unused", path, expected_sha="bad")

    def test_proxy_recovers_selected_original_blocks_once_and_keeps_arrival_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            late = session("late", [call(100, 102)])
            late["requests"][0]["block_ids"] = [2**64 - 1]
            self.create_source(root / "raw", [late, nested()])
            fi.index(argparse.Namespace(source=root / "raw", out=root / "index"))
            self.assertNotIn('"block_ids"', (root / "index/sessions.jsonl").read_text())
            args = argparse.Namespace(index=root / "index", out=root / "proxy", arrivals="regular",
                    sessions_per_second=2, seed=0, invalid="error", limit=None, max_calls=None,
                    max_input_tokens=None, max_output_tokens=None, session_ids=None,
                    token_mode="proxy-prefix", target_model="target", kv_layout="layout", sharing_scope="shared")
            with patch.object(fi.zstd, "open", wraps=fi.zstd.open) as opened:
                fi.generate(args)
            self.assertEqual(opened.call_count, 1)
            rows = [json.loads(line) for line in (args.out / "workload.jsonl").read_text().splitlines()]
            self.assertEqual([r["arrival_time"] for r in rows], [0, .5])
            self.assertEqual(rows[1]["sub_requests"][1]["kv_prefix"]["block_ids"], [2**64 - 1])
            self.assertEqual(rows[1]["sub_requests"][1]["input_tokens"], 16)
            manifest = json.loads((args.out / "manifest.json").read_text())
            self.assertEqual(manifest["fidelity"]["omitted_partial_proxy_tokens_per_prompt"], [0, 15])
            self.assertFalse(manifest["fidelity"]["decode_to_input_identity"])
            self.assertEqual(manifest["nested_sessions"], 1)
            args.out = root / "filtered"
            args.max_input_tokens = 16
            fi.generate(args)
            manifest = json.loads((args.out / "manifest.json").read_text())
            self.assertEqual([s["session_id"] for s in manifest["selected"]], ["late"])

    def test_proxy_rejects_changed_original_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.create_source(root / "raw", [nested()])
            fi.index(argparse.Namespace(source=root / "raw", out=root / "index"))
            metadata = json.loads((root / "index/index.json").read_text())
            source = json.loads((root / "index/sessions.jsonl").read_text())
            selection = {key: source[key] for key in ("source", "week", "session_id", "start_ms")}
            (root / "raw" / source["source"]["file"]).write_bytes(b"changed")
            with tempfile.TemporaryFile() as spool, self.assertRaisesRegex(ValueError, "compressed source changed"):
                fi.recover_blocks(metadata, [selection], spool)


if __name__ == "__main__":
    unittest.main()
