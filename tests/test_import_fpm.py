"""FPM normalization checks on in-memory rows; Parquet reading is not exercised."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location("import_fpm", Path(__file__).parents[1] / "scripts/import_fpm.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

METADATA = {"coordinate_system": "iteration_totals_balanced_v1", "parquet_sha256": "abc"}


def row(kind, batch, tokens, kv, latency, **overrides):
    identity = {key: "x" for key in MODULE.IDENTITY}
    identity.update(tp=4, moe_ep=4, moe_tp=1, pp=1, dp=1, cp=1)
    return {**identity, "workload_kind": kind, "batch_size": batch, "total_prefill_tokens": tokens,
            "total_kv_read_tokens": kv, "latency_ms": latency, "partition_policy": "balanced_v1", **overrides}


class FpmImportTests(unittest.TestCase):
    def test_rows_split_into_decode_and_prefill_with_identity(self):
        table = MODULE.normalize([row("prefill", 2, 512, 0, 9.0), row("decode", 4, 0, 400, 2.5)],
                                 METADATA, "abc", "url")
        self.assertEqual(table["decode"], [[4, 400, 2.5]])
        self.assertEqual(table["prefill"], [[2, 512, 0, 9.0]])
        self.assertEqual((table["deployment"]["tp"], table["deployment"]["ep"]), (4, 4))
        self.assertEqual(table["source"]["sha256"], "abc")

    def test_inconsistent_or_unsupported_rows_fail(self):
        good = [row("prefill", 1, 8, 0, 1.0), row("decode", 1, 0, 8, 1.0)]
        cases = {
            "mixed deployment": good + [row("decode", 2, 0, 8, 1.0, tp=8)],
            "duplicate coordinates": good + [row("decode", 1, 0, 8, 2.0)],
            "unbalanced": good + [row("decode", 2, 0, 8, 1.0, partition_policy="skewed")],
            "unsupported workload": good + [row("mixed", 2, 8, 8, 1.0)],
            "invalid": good + [row("decode", 2, 0, 8, 0.0)],
        }
        for message, rows in cases.items():
            with self.assertRaisesRegex(ValueError, message):
                MODULE.normalize(rows, METADATA, "abc", "url")
        with self.assertRaisesRegex(ValueError, "coordinate system"):
            MODULE.normalize(good, {"coordinate_system": "other"}, "abc", "url")
        with self.assertRaisesRegex(ValueError, "does not describe"):
            MODULE.normalize(good, METADATA, "other", "url")


if __name__ == "__main__":
    unittest.main()
