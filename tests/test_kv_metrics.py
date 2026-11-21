import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from kv_metrics import integrate


def scheduler(running=False, admission=0, preparation=0, queued=0):
    return {
        "iteration_active": running,
        "hbm_admission_waiting_requests": admission,
        "kv_preparation_waiting_requests": preparation,
        "scheduler_waiting_requests": queued,
    }


def row(time, **state):
    return {"time_s": time, "state": state}


class KvMetricsTests(unittest.TestCase):
    def test_concurrent_waits_are_not_gpu_idle_or_elapsed_time(self):
        result = integrate([
            row(0, scheduler=scheduler(True, admission=1, preparation=1, queued=1)),
            row(2, scheduler=scheduler()),
            row(5, scheduler=scheduler()),
        ])
        totals = result["totals"]
        self.assertEqual(totals["hbm_admission_wait_request_s"], 2)
        self.assertEqual(totals["kv_preparation_wait_request_s"], 2)
        self.assertEqual(totals["scheduler_wait_request_s"], 2)
        self.assertEqual(totals["waiting_request_s"], 6)
        self.assertEqual(totals["any_request_waiting_s"], 2)
        self.assertEqual(totals["all_execution_domains_idle_s"], 3)
        self.assertEqual(result["execution_by_scope"]["/scheduler"], {
            "iteration_active_s": 2, "gpu_idle_s": 3,
        })

    def test_transitions_partition_one_calls_wait(self):
        result = integrate([
            row(0, scheduler=scheduler(admission=1)),
            row(2, scheduler=scheduler(preparation=1)),
            row(5, scheduler=scheduler(queued=1)),
            row(6, scheduler=scheduler(True)),
            row(10, scheduler=scheduler()),
        ])
        self.assertEqual(result["totals"]["waiting_request_s"], 6)
        self.assertEqual(result["totals"]["any_request_waiting_s"], 6)
        self.assertEqual(result["totals"]["all_execution_domains_idle_s"], 6)

    def test_overlapping_workers_use_a_union_for_global_idle(self):
        result = integrate([
            row(0, instances=[{"scheduler": scheduler(True)}, {"scheduler": scheduler()}]),
            row(2, instances=[{"scheduler": scheduler()}, {"scheduler": scheduler(True)}]),
            row(4, instances=[{"scheduler": scheduler()}, {"scheduler": scheduler()}]),
            row(5, instances=[{"scheduler": scheduler()}, {"scheduler": scheduler()}]),
        ])
        self.assertEqual(result["totals"]["all_execution_domains_idle_s"], 1)
        self.assertEqual(result["execution_by_scope"]["/instances/0/scheduler"]["gpu_idle_s"], 3)
        self.assertEqual(result["execution_by_scope"]["/instances/1/scheduler"]["gpu_idle_s"], 3)

    def test_cxl_waits_come_only_from_worker_schedulers(self):
        # Gauges a parent exported before calls enqueued on release are ignored.
        initial = {
            "cxl_pool": {"kv_preparation_waiting_requests": 1, "events": [{"scheduler": {}}]},
            "instances": [{"scheduler": scheduler(True, preparation=1, admission=1)}],
        }
        final = copy.deepcopy(initial)
        final["instances"][0]["scheduler"] = scheduler()
        result = integrate([row(0, **initial), row(2, **final)])
        self.assertEqual(result["totals"]["kv_preparation_wait_request_s"], 2)
        self.assertEqual(result["totals"]["hbm_admission_wait_request_s"], 2)
        self.assertEqual(list(result["waits_by_scope"]), ["/instances/0/scheduler"])

    def test_joint_ep_execution_covers_idle_attention_rank(self):
        result = integrate([
            row(0, iteration_active=True, schedulers=[scheduler(True), scheduler()]),
            row(3, iteration_active=False, schedulers=[scheduler(), scheduler()]),
        ])
        self.assertEqual(result["execution_by_scope"], {
            "/": {"iteration_active_s": 3, "gpu_idle_s": 0},
        })
        self.assertEqual(len(result["waits_by_scope"]), 2)

    def test_rejects_missing_or_invalid_gauges_and_ambiguous_ep(self):
        for state in [
            {"scheduler": {"running_requests": 1}},
            {"scheduler": scheduler(admission=-1)},
            {"scheduler": scheduler(running=1)},
            {"scheduler": scheduler(queued=0.5)},
            {"schedulers": [scheduler(), scheduler()]},
        ]:
            with self.subTest(state=state), self.assertRaises(ValueError):
                integrate([row(0, **state)])

    def test_validates_clock_and_stable_scopes(self):
        bad_traces = [
            [],
            [row(1, scheduler=scheduler())],
            [row(float("nan"), scheduler=scheduler())],
            [row(0, scheduler=scheduler()), row(0, scheduler=scheduler())],
            [row(0, scheduler=scheduler()), row(1, instances=[{"scheduler": scheduler()}])],
        ]
        for trace in bad_traces:
            with self.subTest(trace=trace), self.assertRaises(ValueError):
                integrate(trace)
        empty_run = integrate([row(0, scheduler=scheduler())])
        self.assertEqual(empty_run["horizon_s"], 0)
        self.assertEqual(empty_run["totals"]["all_execution_domains_idle_s"], 0)

    def test_cli_reads_native_jsonl(self):
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "trace.jsonl"
            output = Path(directory) / "metrics.json"
            trace.write_text("\n".join(json.dumps(item) for item in [
                row(0, scheduler=scheduler()), row(3, scheduler=scheduler()),
            ]))
            subprocess.run([
                sys.executable, str(ROOT / "scripts" / "kv_metrics.py"), str(trace),
                "--output", str(output),
            ], check=True)
            self.assertEqual(json.loads(output.read_text())["totals"]["all_execution_domains_idle_s"], 3)


if __name__ == "__main__":
    unittest.main()
