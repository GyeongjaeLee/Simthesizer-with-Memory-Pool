"""Pool designs share resources fairly and emit loadable scenarios; synthetic sessions reuse prefixes."""
import importlib.util
from pathlib import Path
import tomllib
import unittest


def load(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / f"experiments/cxl-pool/{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load("generate")
SYNTHETIC = load("synthetic")
OUT = Path("/tmp/scenario.toml")


def build(design, switches=1, isl=128, workers=8, platform="vr-nvl72", speedup=None, routing="least_loaded"):
    return MODULE.build(platform, "minimax-m2.7", design, switches, isl, workers, speedup, routing,
                        128 * MODULE.GIB, OUT)


class CxlGenerateTests(unittest.TestCase):
    def test_inflight_kv_loads_reach_the_scheduler_only_when_set(self):
        def scheduler(**options):
            return MODULE.build("vr-nvl72", "minimax-m2.7", "small", 1, 128, 8, None, "least_loaded",
                                128 * MODULE.GIB, OUT, **options)[0]["instances"][0]["scheduler"]
        self.assertNotIn("max_num_inflight_kv_loads", scheduler())
        self.assertEqual(scheduler(inflight_loads=4)["max_num_inflight_kv_loads"], 4)

    def test_large_designs_match_workers_devices_and_full_lanes(self):
        for switches in (8, 16):
            counts = {(s["workers"], s["devices"]) for s in
                      (build(name, switches)[1] for name in ("uniform", "ring", "torus"))}
            self.assertEqual(counts, {(4 * switches, 8 * switches)})
        self.assertEqual(build("small")[1]["workers"], 8)
        with self.assertRaisesRegex(ValueError, "even switch count"):
            MODULE.design("uniform", 3, 128)

    def test_local_global_split_trades_endpoints_for_inter_switch_bandwidth(self):
        self.assertEqual([MODULE.global_lanes(s) for s in ("2:2", "1:1", "3:1", "1:3")], [128, 128, 64, 192])
        for split in ("1:2", "4:0", "3", "a:b"):
            with self.assertRaises(ValueError):
                MODULE.global_lanes(split)
        # 3:1 leaves 192 local lanes: 6 workers and 12 devices per switch.
        _, summary = build("torus", 8, isl=MODULE.global_lanes("3:1"))
        self.assertEqual((summary["workers"], summary["devices"]), (48, 96))
        self.assertEqual(MODULE.design("ring", 4, 192)[2][(0, 1)], 96)
        with self.assertRaisesRegex(ValueError, "divisible by 32"):
            MODULE.design("ring", 4, 100)

    def test_torus_merges_wraparound_neighbors(self):
        # 2×4: the size-2 dimension reaches one neighbor twice.
        edges = MODULE.design("torus", 8, 128)[2]
        self.assertEqual(edges[(0, 4)], 64)
        self.assertEqual(edges[(0, 1)], 32)

    def test_none_keeps_workers_without_pool_devices(self):
        system, summary = build("none", workers=4)
        scenario = tomllib.loads(MODULE.toml(system, "test"))["system"]
        self.assertEqual((summary["workers"], summary["devices"]), (4, 0))
        self.assertEqual(scenario["placement_devices"], [])
        self.assertFalse(any(e["kind"] in ("switch", "adapter") for e in scenario["topology"]["endpoints"]))

    def test_platforms_set_rates_and_scenario_parses(self):
        system, _ = build("ring", 4, speedup=2.0, routing="session")
        scenario = tomllib.loads(MODULE.toml(system, "test"))["system"]
        adapter = next(l for l in scenario["topology"]["links"] if l["id"] == "w0-cxl-S0")
        self.assertAlmostEqual(adapter["bandwidth_ab"], 16 * 8e9 * 0.8)
        self.assertEqual(scenario["routing"], "session")
        self.assertEqual(scenario["instances"][0]["model_runner"]["compute_speedup"], 2.0)
        self.assertEqual(len(scenario["instances"]), 16)
        h200 = build("small", platform="h200")[0]["instances"][0]["model_runner"]
        self.assertEqual(h200["compute_speedup"], 1.0)
        self.assertEqual(build("small", routing="cache_affinity")[0]["affinity_beta"], 1.0)

    def test_synthetic_sessions_share_a_prompt_and_extend_their_own(self):
        rows = list(SYNTHETIC.sessions(2, 1.0, 3, 64, 32, 8, 1.0, 0))
        calls = [[n for n in row["sub_requests"] if n["kind"] == "llm"] for row in rows]

        def full(row, index):
            prefix = row["sub_requests"][index]["kv_prefix"]
            head = full(row, prefix["parent"])[:prefix["shared"]] if "parent" in prefix else []
            return head + prefix["block_ids"]

        ids = [[full(row, i) for i, n in enumerate(row["sub_requests"]) if n["kind"] == "llm"]
               for row in rows]
        self.assertEqual(calls[0][2]["kv_prefix"]["parent"], 2)
        self.assertEqual(ids[0][0], ids[1][0])
        self.assertEqual(ids[0][2][:len(ids[0][1])], ids[0][1])
        self.assertNotEqual(ids[0][1][-1], ids[1][1][-1])
        self.assertEqual([c["input_tokens"] for c in calls[0]], [64, 96, 128])
        self.assertEqual(rows[0]["sub_requests"][-1]["next"], [])


if __name__ == "__main__":
    unittest.main()
