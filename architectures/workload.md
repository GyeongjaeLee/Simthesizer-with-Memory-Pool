# Workload and Result Architecture

## Purpose

Workload JSONL is the only simulator input and the contract shared by every
runner. Result JSONL is the shared output. Both hold one request DAG per row.

## Request DAG

- A row is one request: `request_id`, `arrival_time`, `initial`, and non-empty
  `sub_requests`.
- A sub-request is an `llm` or a `tool_call` node; `next` indices link them.
- `initial` lists the roots explicitly and must equal the indegree-zero set
  implied by `next`. Root activation is a validated input fact, not inferred.
- `interval` delays a node's start after all predecessors finish, modeling
  think time and invocation overhead.
- A `tool_call` occupies wall-clock time (`duration`) and no device.
- Unknown fields are rejected; legacy flat rows are not accepted.
- Cycles and invalid indices are rejected. Arrival times, intervals, and tool
  durations must be finite and nonnegative. LLM input plus output must fit `u32`.

## Context Authority

Each `llm` sub-request is self-contained. `input_tokens` is the full prompt of
that step, and optional `input_token_ids` / `output_token_ids` are that step's
own token trace. Runtime never derives context or KV lineage from predecessor
completions. Prefix identity comes from token IDs or an explicit `kv_prefix`
(`namespace`, `block_tokens`, ordered unsigned 64-bit `block_ids`), never both.
A `kv_prefix` may name an earlier LLM sub-request of the same row as `parent`;
its full list is then the parent's first `shared` IDs followed by `block_ids`.
This only compresses the row: identity never depends on execution, and both
calls keep their namespace and block size.
The namespace identifies compatible model/layout and sharing scope; reuse
requires a continuous matching prefix. Request-level token counts do not exist.

Optional `session_id` on LLM and tool nodes records ownership, independently of
prefix identity and physical storage. Tools and terminal joins keep session
membership alive through their completion. Optional `provider_tokens` preserves
original `prompt` and `completion` counts without overriding native work sizes.

Optional seeded progress (`known_tokens`, `kv_tokens`) is set together on an
`llm` sub-request so that a system can start a synthetic child mid-generation,
as disaggregated decode does. Committed workloads omit it, which means
`known_tokens = input_tokens` and `kv_tokens = 0`.

`model` records provenance and never affects simulation.

## Results

A result row repeats the request's DAG, arrival, token counts, intervals, and
tool durations, and adds one `token_latency` array per `llm` sub-request. A
scheduled call also reports `ttft_parts` (`kv_load`, `admission`, `queue`,
`prefill`), which sum to its first latency; a pool call also reports
`reused_tokens`, the prompt tokens restored from cached KV.
Entry 0 is the time from entering the scheduler to the first token; later
entries are inter-token latencies. Keeping both in one array prevents them from
drifting apart.

`scripts/results.py` reconstructs absolute token times by walking the DAG from
`arrival_time`. A node starts after every predecessor completes, plus its own
`interval`; tool durations also advance the timeline. Rows are matched by
`request_id`, LLM calls by sub-request index, and ITLs by output-token index.

## Boundaries

- The Rust types are the schema authority. Python analysis validates the fields
  it reads and requires every LLM's latency count to equal `output_tokens`.
- Raw agent traces are converted to canonical rows offline. Runtime accepts
  only canonical rows.

## FreeInference ingestion

`scripts/freeinference.py` fetches pinned compressed sources, verifies and
indexes sessions, and generates native DAGs. Arrival mode selects original
session timestamps, regular session/s, or seeded Poisson session/s. Sessions
are taken in source start order or a seeded shuffle. A horizon replaces the
session count: sessions arrive until it, nodes that cannot start before it
even with instantaneous LLM calls are dropped, and sessions beyond the eligible
set replay as copies with their own session and KV identities.

Default `provider-count` mode uses provider prompt/completion counts and emits
no prefix identity. `proxy-prefix` uses original complete 16-token
`o200k_base` block lists; input length is 16 times their count. The index
reads each verified raw file once, in parallel, storing each call as the IDs
after those shared with the previous call of its agent; generation reads only
the index. Each identity names that previous call as its `parent`. Its namespace contains dataset revision, week, source model,
explicit target model/KV layout and sharing scope. Session ownership remains
separate. The configured native model supplies KV bytes and compute costs.

Both modes retain provider counts. Provider completion is only a generation
length approximation; it is not a measured proxy-tokenizer count. Partial final
proxy blocks (0–15 tokens) are absent, and output-to-next-input identity cannot
be recovered. The converter never synthesizes those identities. Rows, the index
and runtime memory grow with distinct blocks, not with repeated prefixes.

Nested sessions become branches. Mid-response launches use observed delay
from parent release; later launches follow parent completion. A child that
finishes before the next recorded parent call gates it, with only the residual
gap charged. Other children drain asynchronously. These timing-derived
dependencies approximate streaming tool dispatch; they are not token-level
observations. Conversion never supplies model profiles or CXL placement.
See `experiments/freeinference-load/experiments.md` for selection and fidelity.

## Evolution Rules

- Convert a new trace source offline; never teach runtime a second format.
- A field that changes simulation belongs on the sub-request it affects.
- Change the schema together with both languages, bundled workloads, and
  analysis.
