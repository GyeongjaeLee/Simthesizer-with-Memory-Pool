# Cost Model Architecture

## Purpose

Simulators price graph nodes. A compute simulator returns the latency of one
layer for one batch. A network simulator returns the duration of point-to-point
transfers under contention. Each instance belongs to one owner, a runner or a
disaggregated system, which initializes it.

## Table Lookup Compute

`perf_dir` names a `.jsonl` profile for one model and machine. The loader requires
format version 1, a complete run, and a matching TP degree. `profile_variant`
selects a variant when several exist. Rank 0 supplies the timings.

A `components` map connects model layer names to lists of profile component
names; omitted names match directly. Multiple components or instances combine
by invocation count within each sample, then take the median. Microseconds
convert to seconds at this boundary. An uncalled component contributes no sample;
a called component with zero GPU work contributes zero latency.

Tables are built on first use, when the model's layer type determines how exact
request shapes project into lookup keys:

| Layer | `key_0` | `key_1` |
|---|---|---|
| token-wise (embedding, norm, GEMM, RoPE, activation) | 0 | total query tokens |
| attention, Mamba | Σ `kv_len` | Σ `q_len · (q_len/2 + kv_len)` |
| LM head, sampler | 0 | sampled rows |

Samples that project to the same keys average. Interpolation is piecewise linear,
clamped below the smallest sample and extrapolated above the largest. Attention
holds compute/memory constant across nonzero KV rows; zero-KV prefill uses the
raw compute axis. Multi-token batches exclude decode-only samples from inner
interpolation because their kernels can differ.

One-token attention samples also build a table indexed by batch size, maximum KV,
and total KV. Its interpolation preserves mean/maximum KV. Without such samples,
attention uses the two-key table. Prefill uses compute/KV keys; kernel tile
metadata is not part of the current profile format.

Optional `cudagraph_capture_sizes` pads token-wise dense lookup to the next
captured size. Attention, LM head, sampler, and measured shapes remain unpadded.
Batches beyond the largest captured size remain unpadded. Missing components
fail the run. Forced MoE routing is not yet supported by this loader.

## Profiling

[`scripts/profile.py`](profiling.md) emits exact-shape JSONL measurements under
`examples/data/<device>/`. Each scenario names its profile explicitly; model and
hardware must match. The simulator does not load CSV profiles.

## Forward-Pass Tables

`scripts/import_fpm.py` converts one deployment's FPM Parquet
(`iteration_totals_balanced_v1`) into JSON: decode rows (requests, total KV,
latency) and prefill rows (requests, new tokens, cached KV, latency) of balanced
batches. The `forward_pass` runner rejects a table measured at another TP or
with expert parallelism.

An iteration whose requests all compute one token uses decode (requests, Σ kv).
Any other batch uses prefill (Σ q, Σ q·(q/2 + kv)); measured rows project to the
same coordinates. One axis carries token-wise work and the other causal
attention, so unequal lengths, more requests than measured and decode tokens in
mixed batches need no separate rule. Lookups interpolate linearly and clamp
below. Above a curve's last sample they continue at the least-squares slope of
its upper half (at least the last two samples), never negative: adjacent samples
from different batch compositions sit at nearly equal coordinates, so their
slope is noise. Extrapolated and mixed lookups are counted. `compute_speedup` divides every lookup for a declared projection to an
unmeasured platform; it is a hypothesis, never fitted to serving results, and is
reported in observations. Tables measured by a serving engine include its graphs, fusion and
collectives, but not idle time or KV transfers.

## Flow Network

- Owners register devices and directed links (bandwidth, latency) during
  initialization. Multi-hop routes prefer the widest bottleneck, then the
  lowest latency, then the fewest hops, and are cached until the topology
  changes.
- A flow transmits at its rate, then waits its route's total latency.
- Rates are weighted max-min fair, as under work-conserving link arbitration:
  links saturate in order of their per-weight share, fixing their flows, and
  capacity a flow cannot use because of another bottleneck goes to the rest.
  Flows with the same route and weight always share one rate, so they form a
  class: rate filling, progress and the next event cost one step per class.
  A class orders its flows by the class progress at which each finishes.
- The simulator keeps its own clock, advanced on every poll, and answers each
  pending poll with the delay to that transfer's next flow event: a flow
  ending transmission, which changes rates, or arriving. After each change
  only the owner of the next event, and transfers that just arrived, are
  polled earlier; the owner's poll advances every flow, so every event time is
  visited and completions are exact. A transfer polled too early answers
  again. A new transfer's first poll is deferred, so transfers starting
  together share one rate computation.
- Every owner has its own instance, so flows of different owners never contend.

The opt-in CXL topology compiles physical ports and directional service edges
into one shared flow instance, so all traffic routed over it contends. Explicit
route waypoints keep traffic on its configured fabric; equal-hop path striping
uses parallel links. See [CXL topology](cxl-topology.md).

## Evolution Rules

- Price compute from profiles; extend the profile rather than adding a formula
  fallback.
- `link_bw` is measured effective bandwidth, not the theoretical peak.
- A new cost model is a new simulator `kind`; graph nodes stay unchanged.
