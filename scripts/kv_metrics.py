"""Integrate disjoint KV waits and serving-device idle from native event traces."""

import argparse
import json
import math
import sys
from pathlib import Path


WAIT_FIELDS = {
    "hbm_admission_waiting_requests": "hbm_admission_wait_request_s",
    "kv_preparation_waiting_requests": "kv_preparation_wait_request_s",
    "scheduler_waiting_requests": "scheduler_wait_request_s",
}
CHILD_ARRAYS = ("instances", "prefill_instances", "decode_instances")


def nonnegative(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and non-negative")
    return value


def count(value, name):
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def active(value, name):
    if type(value) is not bool:
        raise ValueError(f"{name} must be a boolean")
    return value


def snapshot(state):
    """Extract component gauges without walking cache contents or event histories."""
    waits = {}
    execution = {}

    def scheduler(value, path, execution_covered):
        if not isinstance(value, dict):
            raise ValueError(f"{path} must be an object")
        waits[path] = {
            metric: count(value.get(field), f"{path}/{field}")
            for field, metric in WAIT_FIELDS.items()
        }
        running = active(value.get("iteration_active"), f"{path}/iteration_active")
        if not execution_covered:
            execution[path] = running

    def system(value, path=""):
        if not isinstance(value, dict):
            raise ValueError(f"{path or '/'} must be an object")
        covered = "iteration_active" in value
        if covered:
            execution[path or "/"] = active(
                value["iteration_active"], f"{path}/iteration_active"
            )
        if "scheduler" in value:
            scheduler(value["scheduler"], f"{path}/scheduler", covered)
        if "schedulers" in value:
            children = value["schedulers"]
            if not isinstance(children, list):
                raise ValueError(f"{path}/schedulers must be an array")
            if len(children) > 1 and not covered:
                raise ValueError(f"{path or '/'} needs joint iteration_active for EP")
            for index, child in enumerate(children):
                scheduler(child, f"{path}/schedulers/{index}", covered)
        if "cxl_pool" in value:
            pool = value["cxl_pool"]
            pool_path = f"{path}/cxl_pool"
            if not isinstance(pool, dict):
                raise ValueError(f"{pool_path} must be an object")
            # Parent gauges contain only calls not yet enqueued in a child.
            # Resumed calls appear in the child's KV-preparation gauge instead.
            waits[pool_path] = {
                metric: (
                    0
                    if field == "scheduler_waiting_requests"
                    else count(pool.get(field), f"{pool_path}/{field}")
                )
                for field, metric in WAIT_FIELDS.items()
            }
        for key in CHILD_ARRAYS:
            if key not in value:
                continue
            children = value[key]
            if not isinstance(children, list):
                raise ValueError(f"{path}/{key} must be an array")
            for index, child in enumerate(children):
                system(child, f"{path}/{key}/{index}")

    system(state)
    if not execution:
        raise ValueError("trace contains no observable serving execution domain")
    return waits, execution


def integrate(rows):
    """Hold each exact event snapshot until the next timestamp; never extrapolate."""
    previous = None
    waits_by_scope = {}
    execution_by_scope = {}
    totals = {metric: 0.0 for metric in WAIT_FIELDS.values()}
    totals.update(any_request_waiting_s=0.0, all_execution_domains_idle_s=0.0)
    end = 0.0
    frames = 0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("trace row must be an object")
        now = nonnegative(row.get("time_s"), "time_s")
        current = snapshot(row.get("state"))
        if previous is None:
            if now != 0:
                raise ValueError("native trace must begin at time zero")
            waits_by_scope = {
                path: {metric: 0.0 for metric in WAIT_FIELDS.values()}
                for path in current[0]
            }
            execution_by_scope = {
                path: {"iteration_active_s": 0.0, "gpu_idle_s": 0.0}
                for path in current[1]
            }
        else:
            if now <= end:
                raise ValueError("trace timestamps must increase strictly")
            if current[0].keys() != previous[0].keys() or current[1].keys() != previous[1].keys():
                raise ValueError("observed component scopes changed during the trace")
            duration = now - end
            any_wait = False
            for path, counts in previous[0].items():
                for metric, requests in counts.items():
                    elapsed = requests * duration
                    waits_by_scope[path][metric] += elapsed
                    totals[metric] += elapsed
                    any_wait |= requests > 0
            if any_wait:
                totals["any_request_waiting_s"] += duration
            for path, running in previous[1].items():
                metric = "iteration_active_s" if running else "gpu_idle_s"
                execution_by_scope[path][metric] += duration
            if not any(previous[1].values()):
                totals["all_execution_domains_idle_s"] += duration
        previous = current
        end = now
        frames += 1
    if previous is None:
        raise ValueError("trace is empty")
    totals["waiting_request_s"] = sum(totals[metric] for metric in WAIT_FIELDS.values())
    return {
        "schema_version": 1,
        "horizon_s": end,
        "frames": frames,
        "definitions": {
            "gpu_idle_s": "No active serving iteration in this execution domain; not per-kernel SM idle. An iteration includes its collectives. TP and joint EP each form one domain.",
            "wait_request_s": "Integral of mutually exclusive per-call wait phases; summed across calls, so it may exceed elapsed time.",
            "any_request_waiting_s": "Union of intervals with at least one waiting call, irrespective of its phase.",
            "all_execution_domains_idle_s": "Union of intervals with no active serving iteration in any domain; independent of request waits.",
        },
        "waits_by_scope": waits_by_scope,
        "execution_by_scope": execution_by_scope,
        "totals": totals,
    }


def load_rows(path):
    with Path(path).open() as source:
        for line, text in enumerate(source, 1):
            if text.strip():
                try:
                    yield json.loads(text)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{line}: {error}") from error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.dumps(integrate(load_rows(args.trace)), indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(report)
    else:
        sys.stdout.write(report)


if __name__ == "__main__":
    main()
