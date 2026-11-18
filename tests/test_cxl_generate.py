"""Pool designs share resources fairly and emit loadable scenarios."""
import importlib.util
from pathlib import Path
import tomllib
import unittest

SPEC = importlib.util.spec_from_file_location("generate", Path(__file__).parents[1] / "experiments/cxl-pool/generate.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
OUT = Path("/tmp/scenario.toml")


class CxlGenerateTests(unittest.TestCase):
    def test_large_designs_match_workers_devices_and_full_lanes(self):
        for switches in (8, 16):
            shapes = {name: MODULE.build("h200", name, switches, None, "vllm_dp", 128, OUT)[1]
                      for name in ("uniform", "ring", "torus")}
            counts = {(s["workers"], s["devices"]) for s in shapes.values()}
            self.assertEqual(counts, {(4 * switches, 8 * switches)})
        self.assertEqual(MODULE.build("h200", "small", 1, None, "vllm_dp", 128, OUT)[1]["workers"], 8)
        with self.assertRaisesRegex(ValueError, "even switch count"):
            MODULE.design("uniform", 3)

    def test_torus_merges_wraparound_neighbors(self):
        # 2×4: the size-2 dimension reaches one neighbor twice.
        edges = MODULE.design("torus", 8)[2]
        self.assertEqual(edges[(0, 4)], 64)
        self.assertEqual(edges[(0, 1)], 32)

    def test_projection_uses_gen6_rates_and_scenario_parses(self):
        system, _ = MODULE.build("vr-nvl72", "ring", 4, 2.0, "session_hash", 128, OUT)
        scenario = tomllib.loads(MODULE.toml(system, "test"))["system"]
        adapter = next(l for l in scenario["topology"]["links"] if l["id"] == "w0-cxl")
        self.assertAlmostEqual(adapter["bandwidth_ab"], 16 * 8e9 * 0.8)
        self.assertEqual(scenario["routing"], "session_hash")
        self.assertEqual(scenario["instances"][0]["model_runner"]["compute_speedup"], 2.0)
        self.assertEqual(len(scenario["instances"]), 16)


if __name__ == "__main__":
    unittest.main()
