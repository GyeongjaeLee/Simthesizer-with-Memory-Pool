# CXL pool lifecycle

`cxl_pool` composes `single_instance` workers, each one KV domain of all its
GPUs; GPU-level state and imbalance are not modeled. The parent owns the request DAG, block identities, physical replicas,
and transfer dependencies. Worker schedulers own execution and incremental
private KV allocation. Tools retain their native DAG timing.

An ordered prefix identity is independent of session membership, execution
references, and storage locations. Native `kv_prefix` namespaces identify source
content and sharing scope; token traces identify input and traced output in one
pool-wide scope. Physical model/layout fingerprints check compatibility.
Equal token counts, session IDs, or DAG edges never establish content identity.
Computed blocks without supplied identity receive request-local keys, created at
compute completion. Every reuse starts at the longest available continuous prefix
of the scheduler's known tokens but the last, so a resumed call also reuses and
reserves its generated tokens; missing suffixes are prefetched or recomputed.
The final sampled token has no KV.

Released calls without an explicit binding choose a worker by `routing`.
`least_loaded` (default) follows vLLM's data-parallel load balancer (`DPLBAsyncMPClient`,
commit `ce72586f`): a 100 ms snapshot of waiting (queued or preparing KV) and
running calls plus KV usage, where unreferenced cached blocks count as free;
score `max(in-flight, waiting + running)`, plus `6 · waiting · max(0, usage − 0.5)`;
a rotating scan start; and a local waiting increment per choice. `session`
pins each session to a stable hash, as in vLLM production-stack session routing,
and falls back to `least_loaded` without a session. `cache_affinity` follows production-stack's
`LoadAwareRouter` (commit `2f879ef8`): matched-prefix fraction minus β times
relative in-flight load. As in LMCache's controller lookup, the prompt's blocks
are walked until one is in no host DRAM, and a worker's match ends at the last
block of that run in its own host DRAM; with no match anywhere it routes by session.

Explicit `calls` bindings remain diagnostic fixtures. `blocks` specifies a full
ordered prefix, `completed_blocks` supplies computed identities, and
`seed_host_blocks` declares initial host copies without an invented transfer.
Inconsistent prefix order is rejected. Pool blocks may group complete source identity blocks; smaller or
incommensurate blocks conservatively miss. Payload sizes are the worker's KV
bytes; explicit proxy byte sizes remain available for controlled fixtures. A
block is one directory entry holding every GPU's shard: position, session
membership, copies, LRU position, pending transfer and pins.

Outside HBM a block has one copy: its computing worker's host DRAM, a CXL
device after a spill, or none after eviction. A call gathers each block from its
own worker's HBM or from that copy, local or remote; other workers' HBM is no source.
Reuse ends at the first block found nowhere, because vLLM keeps one computed-token
count per request. As in vLLM, a released call joins its worker's queue at once
and loads its prefix only at its turn, in queue order, while a running slot and
token budget remain (or always, up to `max_num_inflight_kv_loads`), and only if
HBM can then hold the prefix and the rest of the prompt; the scheduler holds those bytes until
admission, and a call that cannot fit holds nothing and blocks later ones.
Preempted calls reload at their next turn. Other calls run while a prefix loads. A reused block whose copy was evicted during the call is
stored again at its end.

Full-block compute notifications transfer HBM accounting from the scheduler to
the directory and initiate asynchronous HBM-to-host stores. This follows the
full-chunk, `ON_COMPUTE` trigger in vLLM V1's OffloadingConnector, pinned to commit
`b7e43db256a69b8c2917e5ef4cf8073a086187eb`. Stores preserve the HBM source. Reloads
preserve backing copies. A valid host or CXL copy suppresses duplicate stores;
a reserved destination is never a cache hit. Pending stores and transfers protect
only their source until publication; other copies stay reclaimable. Request completion releases execution references;
the last node in each source session removes membership without deleting cached
data, independently of sibling sessions. Scoped tool and end nodes preserve trailing
session lifetimes. Native DAGs containing unscoped tools conservatively retain
session membership until the containing DAG completes.

Hybrid models also require a recurrent-state checkpoint. The default sparse
boundary is the last scheduler block boundary strictly before the prompt ends.
The directory caches that state separately from attention blocks, using the
model's actual per-request state bytes. Reuse requires both a continuous attention
prefix and a matching checkpoint. The state follows the same transfer and backing
rules, and its execution pin ends after the next compute consumes it. Pure
recurrent models cache only state bytes. A checkpoint boundary not representable
by full pool blocks conservatively misses; no token-only hybrid hit is inferred.

The HBM → host → CXL hierarchy is a **Simthesizer extension**, not vLLM's default
hierarchy. HBM reclamation is separate from storage and occurs only when the next
allocation needs capacity. Each store maintains free extents and a reclaimable
LRU index. A reservation may use multiple disjoint extents; artificial contiguous
address requirements never prevent an otherwise feasible allocation. Reload
admission also accounts for the worker scheduler's current private reservations. A consumer pins only
its own worker's HBM copy, so other workers' idle copies stay reclaimable; the last last-tier
source awaiting a reload also remains pinned.
Independent backing copies can be reclaimed safely. A sole pinned host source
may still spill because publication precedes source release. Host pressure spills reclaimable host copies to CXL;
CXL pressure deletes reclaimable cache copies in tier-wide LRU order. Placement
tries free compatible devices before eviction and skips devices too small for
the payload. A payload that fits no CXL device is rejected explicitly. With no
placement devices, host DRAM is the last tier: host pressure deletes reclaimable
host copies in LRU order, as vLLM CPU offloading does. There is no completed
session priority or TTL. Per-host store queues stop at capacity dependencies
without repeatedly scanning their backlog.

Reservations consume destination capacity from transfer issue. Completion publishes
the destination; only host-to-CXL spill releases its source. HBM and lower-tier
copies can coexist after completion, and immutable HBM sources can serve concurrent
cross-worker copies. Preemption releases execution pins, allowing HBM reclamation;
resume restores the surviving continuous prefix before scheduling the suffix.

Every transfer uses the parent's physical flow network and charges the worker's
block payload. `placement_devices`, `stripe_blocks`, and
`max_outstanding_reads` control CXL placement (a whole block per device) and
reload concurrency (blocks in flight per call). Host pressure
fixtures reserve another session's host bytes only after spill completion frees
space. Configured routes must support both HBM-to-host stores and reloads.
Collectives retain worker-local fabrics; shared collective/CXL contention and
predictive tool-latency policies are outside this model.

Default observations contain aggregate block/transfer counters and tier
occupancy; wait gauges come from the worker schedulers, where released calls wait.
`detailed_observations = true` records transfer/event history and additionally
serializes the directory, physical extents, pins, membership, reservations,
and topology. Default mode retains no event or transfer history.
Detailed snapshots cost time proportional to
retained state per observation and are intended for bounded diagnostics. GPU idle is derived independently from worker execution state;
request wait durations are not labeled GPU idle. Reference-guided CPU fixtures
validate lifecycle and capacity invariants, without claiming measured CXL accuracy.
