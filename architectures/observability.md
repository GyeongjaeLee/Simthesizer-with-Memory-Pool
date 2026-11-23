# Observability

Components own observation meaning. `Module::observe` returns a read-only JSON
value; `null` means unavailable. Systems compose child snapshots using their
configuration field names and array positions. A new field or component needs
no engine metric registry or new trace type.

`Engine::run_with_observer` calls a consumer after every event at the same time
has run, including newly created zero-duration descendants. It also reports
time zero for delayed or empty workloads. The callback receives the engine
time and a shared system reference. It adds no graph nodes or modeled cost.
Ordinary `run` uses the same event loop with an empty callback.

The CLI's `--output-engine-trace PATH` writes JSONL:

```json
{"time_s":0.0,"state":{"scheduler":{"running_requests":1,"waiting_requests":0,"generation_tokens_total":0}}}
```

Each row contains a complete snapshot. Unchanged intermediate snapshots are
omitted; the first and final engine times remain. Writing retains one snapshot
in memory and uses a temporary file. The file is published only after a
successful simulation and flush; existing trace files are never replaced.

## Scheduler signals

`chunked_prefill` exposes:

| Signal | Meaning |
|---|---|
| `running_requests` | Admitted, unfinished LLM sub-requests, including those retaining state outside the current batch |
| `waiting_requests` | Enqueued sub-requests without admission, including preempted requests |
| `generation_tokens_total` | Committed token emissions since scheduler creation |
| `hbm_allocated_bytes` | Physical HBM occupancy, including weights, private allocation, cached blocks and externally owned reservations |
| `prefix_cache_bytes` | Resident attention KV and recurrent checkpoints, including request-local opaque cache |
| `preemptions_total` | Running requests returned to waiting after allocation failure |
| `hbm_admission_waiting_requests` | Calls waiting for HBM capacity, including the FCFS suffix blocked behind them |
| `kv_preparation_waiting_requests` | Enqueued calls blocked on external KV preparation |
| `scheduler_waiting_requests` | Other enqueued calls outside the current batch |
| `iteration_active` | A serving iteration is executing |

An admission set keyed by completion node owns membership; its size gives the
running count. Waiting is queue length minus that size. Admission, preemption,
and completion update the set. Token generation increments at the emission
site, including the final prefill chunk. Cached input and scheduled work do not
increment it. Routing load snapshots retain their existing batch-based meaning.

Wait gauges partition enqueued calls: an external HBM wait takes precedence
over external KV preparation, which takes precedence over admission behind it
and scheduler waiting. A call awaiting its KV lookup turn is a scheduler wait.
CXL calls enqueue on release, so worker schedulers own all their wait phases and
the parent exports none. Their preparation equals `ttft_parts.kv_load`; admission
also counts calls queued behind an HBM wait, which `ttft_parts` counts as queue.
Durations on individual CXL events can span multiple phases; use the gauges for
disjoint totals.

`python scripts/kv_metrics.py TRACE.jsonl --output METRICS.json` integrates
snapshots until the final trace timestamp. Wait totals are request-seconds;
concurrent requests can make them exceed elapsed time. `any_request_waiting_s`
instead measures the union of waiting intervals. Neither quantity is GPU idle.

`gpu_idle_s` measures intervals with no active serving iteration. An iteration
includes its internal compute and collectives; this is serving-device idle,
not per-kernel SM idle. A TP worker is one execution domain. Joint EP exposes
system-level `iteration_active`, so an empty attention rank participating in
expert execution does not become an idle GPU. Global idle is the union of
intervals when every execution domain is idle, independent of waiting calls.
The analyzer rejects missing phase gauges, ambiguous EP occupancy, changing
component scopes, and invalid clocks. It never extrapolates beyond the trace.

## Composition and comparison

Single-instance snapshots contain `scheduler`; EP contains `schedulers` in
configured order. `multi_instance` contains `instances`; P/D contains
`prefill_instances` and `decode_instances`. These paths compose recursively and
identify physical scheduler state without cross-instance aggregation. P/D
counts the first output token in the prefill pool and the remaining emissions
in the decode pool. Native capture supports the system's ordinary request DAGs.

`tools/validate.py run-sim` converts these snapshots into comparison bundles.
The adapter selects the declared scheduler signals and retains configuration
paths as metric dimensions. It samples states with a hold rule and preserves
counters; only a drained run may extend idle to a predeclared horizon. Unknown
scheduler semantics or missing observations fail capture. Additional native
fields remain in the raw trace until a comparison mapping defines their meaning.

The bundle runner currently accepts fresh, independent single-LLM roots with
fixed arrivals. This restriction belongs to the comparison contract. Engine
observation itself supports nested systems and dependent requests. A reference
with different component identities needs an explicit correspondence before
pairing. See the [tool guide](../.agents/tools/validation.md) for commands.
