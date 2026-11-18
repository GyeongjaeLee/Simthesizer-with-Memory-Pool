# Scheduling Architecture

## Purpose

A `Scheduler` owns the ready LLM sub-requests of one memory domain: which run
in the next iteration, how many tokens each processes, and what KV each holds.
It is the only component that tracks token progress; runners see only the
batch shape. A `Router` owns whole-request placement across instances.

## Contract

- Initialization supplies weight bytes, per-request state bytes, per-token KV
  bytes, and the parallel strategy.
- A ready sub-request is enqueued with its completion node, which the scheduler
  carries back opaquely.
- Scheduling returns one batch of `(q_len, kv_len, lm_head_len)`, or nothing
  while an iteration is in flight or no work is ready.
- Completing an iteration returns finished sub-requests with their
  `token_latency`.
- A load snapshot exposes waiting and running requests and tokens.

## Chunked Prefill

`chunked_prefill` follows ordinary synchronous vLLM V1 FCFS semantics, pinned to
`b7e43db256a69b8c2917e5ef4cf8073a086187eb`
([cache manager](https://github.com/vllm-project/vllm/blob/b7e43db256a69b8c2917e5ef4cf8073a086187eb/vllm/v1/core/single_type_kv_cache_manager.py),
[store trigger](https://github.com/vllm-project/vllm/blob/b7e43db256a69b8c2917e5ef4cf8073a086187eb/vllm/distributed/kv_transfer/kv_connector/v1/offloading/store_triggers/on_compute.py)).

- Running requests get work before new admissions. Waiting requests remain
  FCFS; an unsatisfied admission blocks the suffix. Batch and token budgets
  bound each iteration. Preemption suppresses new admissions in that iteration.
- Usable memory per GPU is `device_mem × device_mem_utilization` minus
  `reserved_device_bytes` (activation and graph workspace). Weights, cached
  blocks, private block-rounded KV, recurrent states and external reservations
  all consume it.
- Admission normally checks whether the complete currently known input could
  fit, including reclaimable cache capacity. Physical allocation covers only
  current work and configured lookahead. Unknown future output is never reserved.
  Setting `scheduler_reserve_full_isl = false` disables only the full-input gate.
- Allocation reclaims only enough eligible cache entries. If running work still
  cannot fit, FCFS preempts the newest running request and retries. Victims retain
  known token identities, recheck surviving prefixes, and prefill the missing
  suffix when resumed. Transfer-protected storage remains unavailable until safe.
- A token is emitted when an iteration brings KV up to the known tokens,
  including the final prefill chunk. The sampled token has no KV until the next
  iteration.
- Completion releases execution references. Cache contents survive until pressure
  requires reclamation; a successor reuses only proven available prefix identity.
- Capacity waits permit other in-flight work to finish. A request that cannot
  make progress even alone fails instead of leaving an undrained simulation.
- `max_batch` and `max_num_batched_tokens` are per rank and scale with `ep`.

## Prefix Cache

- One cache per scheduler. Cross-worker/tier sharing requires the explicit
  external directory below.
- Content identity, request/session membership, and physical location are
  separate. Native token IDs or namespaced proxy block IDs establish a continuous
  prefix chain. Model/layout compatibility and sharing scope constrain reuse;
  equal session IDs or token counts do not establish a hit.
- Unknown computed content receives request-local opaque identity. It can survive
  completion or support the same invocation's resumption, but cannot prove a
  later invocation's input. Proxy input IDs never identify generated decode KV.
- Requests retain block hashes through preemption and extend them as traced
  tokens become relevant. Hash identity does not imply resident or computed KV.
- Requests reference the blocks they use. Unreferenced blocks are kept in LRU
  order and evicted only under memory pressure. Releasing a prefix makes its
  later blocks evictable first. There is no TTL or completed-session priority.
- After each iteration, a request publishes its newly completed full blocks,
  including committed decode blocks. Only known identities can be shared across
  invocations. Publication and allocation work follow block boundaries.
- A cache hit leaves at least one token to recompute for logits.
- Cached blocks count against device memory.

## Recurrent State

Hybrid prefix reuse also requires a state checkpoint at the selected boundary.
The supported sparse policy retains the last block boundary strictly before
the prompt ends. Prefill splits there to materialize the state, following
vLLM's fallback when internal checkpoint export is unavailable. Attention-only
matches beyond a retained state cannot skip recurrent computation.

Checkpoint storage and its references are independent of attention blocks.
The next computation consumes a checkpoint as a copy source; its execution pin
then ends. Later mutable state is charged separately. In aligned mode, crossing
a positional block keeps old and new live state until the computation finishes;
updates within that block use one live state. Lookahead reserves attention
capacity without inventing future recurrent copies. With prefix caching disabled,
one live state per request remains sufficient.

This scope models prompt replay checkpoints and live-state ownership for existing
models. It does not assume arbitrary internal state exports, dense snapshots at
every attention block, speculative recurrent execution, or junction checkpoints
that the simulator did not materialize.

## External KV

For an externally owned KV tier, the parent keeps resident and incoming HBM
bytes charged to scheduler admission. Explicit declarations identify restored
progress and state backed by those bytes; the scheduler charges private growth
and live state without counting the same prefix twice. Computed full blocks
move into directory ownership, and preemption releases execution pins.
This mode disables the worker's private cache; the parent directory is the sole
residency authority. Capacity and transfer readiness feed separate wait phases.

## Routing

- A router picks one destination from per-instance load snapshots. It never
  splits a request or touches scheduler state.
- `round_robin` ignores load. `moe_load_balancing` picks the lowest queued plus
  running tokens, then requests, rotating ties.
- Placement runs at simulated arrival after parent dependencies. System load
  includes scheduler work and assigned initial LLM calls awaiting enqueue;
  reservations move into scheduler ownership on enqueue. This makes same-time
  arrivals account for earlier assignments. Tool calls contribute no device load.

## Evolution Rules

- Model memory by allocating and reclaiming real blocks, not with capacity or
  hit-rate factors.
- Tool calls never reach a scheduler.
- Cache state never crosses a scheduler boundary; a shared cache tier needs an
  explicit owner above the schedulers.
