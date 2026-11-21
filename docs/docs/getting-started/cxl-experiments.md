---
title: CXL Memory Pool Experiments
description: How to configure, run and analyze agent-workload KV experiments on CXL memory pools, how results are made trustworthy, and how the pool is modeled.
---
# CXL memory pool experiments

This page answers one question: **when many agent sessions share GPU workers,
how does a CXL memory pool for KV cache change latency, and how much does its
design matter?** You choose a platform, a model, a pool design and a load; the
simulator returns per-token latencies and the KV traffic behind them. Everything
runs on a CPU from the repository root.

A run reads two files and writes one or two:

```text
scenario.toml   workers, pool, topology ┐
                                        ├─ simthesizer ─→ results.jsonl  latency of every token
workload.jsonl  agent sessions          ┘                 trace.jsonl    state over time (optional)
```

## Quick start

```sh
cargo build --release
uv sync --locked              # Python 3.14 and orjson for the scripts below

# Eight Vera Rubin workers on one CXL switch
uv run experiments/cxl-pool/generate.py --design small --out output/pool/small.toml

# FreeInference agent traces, once: a 420 MB download, then an index that
# reads each file once in parallel (5 min on 40 cores; --jobs)
uv run scripts/freeinference.py fetch --all-weeks --out output/freeinference/all-source
uv run scripts/freeinference.py index --source output/freeinference/all-source \
  --out output/freeinference/all-index

# Latency at 0.015 sessions/s per worker over a short window
uv run experiments/cxl-pool/sweep.py --scenario output/pool/small.toml --per-worker \
  --rates 0.015 --warmup-hours 1 --horizon-hours 2 --out output/pool/sweep-small
cat output/pool/sweep-small/summary.csv
```

## 1. Running experiments

### 1.1 Scenario: platform × model × pool design

`experiments/cxl-pool/generate.py` writes a scenario. A **worker** is one model
instance on four GPUs (TP4) with its host DRAM, NICs and one x16 CXL root port. A
**device** is a CXL memory expander on an x8 port. Every switch has 256 lanes
and uses all of them.

| `--design` | Switches | Workers | Devices | Shape |
|---|---|---|---|---|
| `none` | 0 | `--workers` (8) | 0 | No pool: KV lives in HBM and host DRAM only. The baseline. |
| `small` | 1 | 8 | 16 | One switch: 8 × x16 workers and 16 × x8 devices. |
| `uniform` | N, even | 4N | 8N | Worker switches each linked to every device switch; reads cross two. |
| `ring` | N | 4N | 8N | Each switch hosts workers and devices and links to 2 neighbors. |
| `torus` | N | 4N | 8N | Near-square 2-D torus; each switch links to 4 neighbors. |

Counts assume the default 1:1 local:global split (128 lanes each). At equal N, the
large designs have the same workers, devices, switches and lanes and differ
only in how switches connect. N = 16 gives 64 workers (256 GPUs).

**Local:global split.** `--local-global A:B` divides each switch's 256 lanes
between local links (workers and devices) and global links (other switches) in
the ratio A:B, so 2:2 is 1:1 and 3:1 gives 192 local and 64 global lanes. More
global lanes mean more bandwidth between switches and fewer endpoints per switch.
With G = 256·B/(A + B) global lanes:

| Design | Endpoints per switch | Lanes per switch-to-switch edge |
|---|---|---|
| `ring`, `torus` | (256 − G)/32 workers and twice as many devices | ring G/2, torus G/4 |
| `uniform` | compute: (256 − G)/16 workers; memory: (256 − G)/8 devices | G/(N/2) per compute–memory pair |

For example, `--design torus --switches 8 --local-global 3:1` gives 6 workers and
12 devices per switch (48 and 96 in total) with x16 to each neighbor. Edges
wider than x16 become parallel x16 links. G must be a whole number that each
design can divide, e.g. 1:2 is rejected.

**Options.**

| Option | Default | Meaning |
|---|---|---|
| `--platform` | `vr-nvl72` | `vr-nvl72` (projected Vera Rubin) or `h200` (measured); §2.2 |
| `--model` | `minimax-m2.7` | model and its compute table; see §2.1 |
| `--switches`, `--local-global`, `--workers` | 1, 1:1, 8 | size as above; `--workers` applies to `none` |
| `--compute-speedup` | platform's (Rubin 3) | divide the H200 compute table by this |
| `--routing` | `least_loaded` | worker choice: `least_loaded`, `cache_affinity` or `session` (§3.2) |
| `--affinity-beta` | 1.0 | `cache_affinity` weight on relative load (β) |
| `--host-kv-fraction` | 0.5 | share of host DRAM (§2.2) for KV; the rest serves the OS and tools |
| `--host-kv-gib` | | host KV per worker, overriding the fraction; small stresses the pool |
| `--block-tokens` | 256 | KV transfer chunk; 16 matches the GPU block but makes 16× more transfers |
| `--max-num-inflight-kv-loads` | off | loads start early; at most this many calls per worker hold KV (§3.3) |

```sh
G="uv run experiments/cxl-pool/generate.py"
$G --design small                                    --out output/pool/small.toml
$G --design uniform --switches 16 --local-global 1:1 --out output/pool/uniform-16.toml
$G --design torus --switches 16 --local-global 1:1   --out output/pool/torus-16.toml
$G --design torus --switches 16 --local-global 3:1   --out output/pool/torus-16-3to1.toml
$G --design none --workers 64                        --out output/pool/none-64.toml
```

The generator prints the resulting counts. Three registries in `generate.py`
extend it: add a topology to `NON_UNIFORM` (a function from switch count and
global lanes to `{(switch, switch): lanes}`), a platform to `PLATFORMS`, or a model
to `MODELS` (its architecture plus a compute table measured on H200).

### 1.2 Workload: agent sessions

[FreeInference](https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace)
records real coding-agent sessions: LLM calls separated by tool runs and user
think time. The index covers 16 weeks: 11,532 replayable sessions (470 more
have replay issues), 42 calls per session at the median (209 at p90) and
122k prompt tokens per call at the median.

`scripts/freeinference.py generate` turns sessions into a workload:

- **Which sessions.** Eligible sessions are ordered by their original start
  time, or shuffled with `--selection-seed`; `--limit N` takes the first N.
  `--session-ids` picks sessions by ID; `--max-calls` and `--max-input-tokens`
  filter; `--invalid exclude` drops sessions with replay issues.
- **When they arrive.** `--arrivals poisson --sessions-per-second R --seed S`
  draws exponential gaps; this seed affects only the gaps. `regular` spaces
  sessions evenly; `original` keeps the recorded start times.
- **Until when.** `--until T` (seconds) replaces `--limit`: sessions keep
  arriving until T, and calls and tools that cannot start before T, even if
  every LLM call took no time, are dropped. No call that could start before T
  is lost. Past the last eligible session, sessions replay in the same order
  as copies with their own session and KV identities.
- **KV identity.** `--token-mode proxy-prefix`, with the target model and KV
  layout, gives every call ordered prefix block IDs, so a later call can reuse
  an earlier call's KV. The pool needs this. The trace has one ID per 16
  tokens; `--prefix-block-tokens 256` (default) hashes each 16 into one, so
  prefixes share an ID exactly when they share its 16. Match the pool's
  `--block-tokens`: results are identical, and the workload and its memory
  shrink about 6×. Each call lists only the IDs after those it shares with
  the previous call of its agent (`parent`, `shared`), so a workload grows
  with distinct blocks, not repeated prompts.

Generation reads only the index, which stores every call's blocks once; an
index from an older version is rejected, so rebuild it. A 6 h workload at 3.84
sessions/s (82,600 sessions) takes 15 min and 7 GB.

```sh
# 0.06 sessions/s per worker for 64 workers, arriving for 6 h
uv run scripts/freeinference.py generate --index output/freeinference/all-index \
  --out output/pool/workloads/r3.84-t21600-s42-sel0-b256 --arrivals poisson \
  --sessions-per-second 3.84 --seed 42 --selection-seed 0 --until 21600 --invalid exclude \
  --prefix-block-tokens 256 --token-mode proxy-prefix --target-model MiniMaxAI/MiniMax-M2.7@d494266a \
  --kv-layout tp4-fp8-block16 --sharing-scope pool
```

**Synthetic sessions** stress the pool with uniform load.
`experiments/cxl-pool/synthetic.py` writes sessions that share a 4,096-token
system prompt and make 10 calls each, every prompt 2,048 tokens longer than
the last, with 256 output tokens and 2 s mean tool runs between calls. They
arrive as a Poisson process (`--sessions`, `--sessions-per-second`, `--seed`;
the other sizes are options too). Run them with the one-run command of §1.3.

**Session rate is the knob; call QPS is the outcome.** Sessions keep their
recorded tool and user think time, as in a datacenter where many sessions
overlap while each waits for its user. Each call waits for the previous call
and its tools, so calls per second follow from how fast the system serves
them. A session lasts 2.73 h on average (0.39 h median, 38 h p99) and makes 96
calls, so at session rate λ the steady state holds about λ × 2.73 h sessions
and λ × 96 calls/s: λ = 0.1/s gives about 980 sessions and 9.6 calls/s.

**Warm-up W and horizon T.** Starting empty, the system approaches that steady
state slowly because long sessions accumulate:

| Time since start | Call rate | Sessions holding KV |
|---|---|---|
| 6 h | 80% | 49% |
| 12 h | 86% | 66% |
| 24 h | 94% | 83% |
| 48 h | 98% | 94% |

Measure only calls released in [W, T) and keep W and T equal for every rate,
so all points cover the same window. W = 24 h and T = 48 h are the defaults;
the pool question needs the right column near steady state, because waiting
sessions keep their KV. Shorter windows, such as W = 2 h and T = 6 h in the
examples here, suit exploration. Calls released just
before T run with fewer newcomers behind them, a boundary effect of seconds in
a window of hours. At λ × T beyond 11,532 sessions, copies appear.

### 1.3 Commands

One run:

```sh
target/release/simthesizer --config output/pool/torus-16.toml \
  --workload output/pool/workloads/r3.84-t21600-s42-sel0-b256/workload.jsonl \
  --output-jsonl output/pool/torus-16.jsonl \
  --output-engine-trace output/pool/torus-16.trace.jsonl   # optional, large
```

A session-rate sweep generates one workload per rate, runs it and summarizes.
Three scenarios at four rates per worker, sharing workloads:

```sh
for s in small uniform-16 torus-16; do
  uv run experiments/cxl-pool/sweep.py --scenario output/pool/$s.toml --per-worker \
    --rates 0.015 0.03 0.045 0.06 --warmup-hours 2 --horizon-hours 6 \
    --workloads output/pool/workloads --out output/pool/sweep-$s &
done; wait
```

A sweep runs its rates one after another; for more parallelism run one sweep
per rate, each with its own `--out`. A 64-worker run at 0.06 per worker holds
25 GB or more.

Each rate generates sessions arriving until T (`--until`), runs them and
measures calls released in [W, T). Defaults are `--warmup-hours 24`,
`--horizon-hours 48`, `--seed 42` (arrival gaps), `--selection-seed 0`
(session order) and `--index output/freeinference/all-index`. Every rate draws
sessions in the same order, so a higher rate adds sessions to a lower rate's
set. Two scenarios swept with the same arguments compare call by call.
Each run keeps the states every minute (`--sample-interval`) and at W and T
(`--trace-times`); `--trace` keeps every state instead. `--per-worker` reads the rates per worker, so scenarios
with different worker counts sweep the same load per worker. `--workloads DIR`
keeps each workload there under its rate, T, seeds and block size, so sweeps
of other scenarios reuse it instead of generating it again.

**Saturation.** As BookSim does for injection rates, a sweep runs its rates in
ascending order and stops at the first saturated one. A run is saturated when
the mean first-token wait (TTFT minus prefill: KV load, HBM admission and
queue) of the calls completed in a 10-minute window of simulated time exceeds
150 s in two consecutive windows, so TTFT up to 150 s is still measured. The simulator then stops without results and
prints `{"status": "saturated", "time_s", "mean_wait_s"}`; a finished run prints
`{"status": "complete"}`. Prefill is excluded because a long prompt takes
seconds even on an idle worker, while the wait is near zero until queues grow;
past capacity it grows with simulated time, so the run's remaining hours would
only confirm it. Set the rule with `--saturation-wait`, `--saturation-period`
and `--saturation-windows` on `sweep.py` or the simulator; `--saturation-wait 0`
disables it in a sweep.

**Figures.** `experiments/cxl-pool/plot.py` draws four figures from sweeps,
one line or panel per scenario (sweeps of one scenario merge):

| Figure | Shows |
|---|---|
| `ttft` | TTFT mean, p50, p90 and p99 against offered call QPS per worker |
| `parts` | mean TTFT parts at the lowest, middle and highest complete load |
| `reuse` | `prefix_reuse` against offered call QPS per worker |
| `queue` | waiting requests per worker over time at the highest rate run |

`ttft` shows mean and p50 up to 10 s and p90 and p99 up to 100 s; a dashed
line marks the first saturated rate. `parts` splits TTFT into queue, HBM
admission, KV reload and prefill. `queue` shows the mean and the range across
workers.

```sh
uv run experiments/cxl-pool/plot.py output/pool/sweep-small output/pool/sweep-uniform-16 \
  output/pool/sweep-torus-16 \
  --out output/pool/figures            # ttft.pdf parts.pdf reuse.pdf queue.pdf
```

Offered call QPS counts calls whose start would fall in [W, T) if every call
took no time, so it is the same input for every scenario.

### 1.4 Outputs and metrics

| File | Content | Tools |
|---|---|---|
| `results.jsonl` | every call's token latencies, TTFT parts and reuse | `scripts/compare.py`, `draw.py`, `results.py` |
| `summary.csv` (sweep) | per rate: status, load, latency, reuse and traffic | `plot.py` |
| `trace.jsonl` | component state over time | `scripts/kv_metrics.py` |

- `results.jsonl` holds each session's DAG plus, per LLM call, `token_latency`
  (TTFT, then inter-token latencies), `ttft_parts` and `reused_tokens` (prompt
  tokens restored from cached KV instead of prefilled). Compare runs with
  `scripts/compare.py a=X b=Y`.
- `summary.csv` has one row per rate: `status` (`complete` or `saturated`,
  with `saturated_at_s` and `saturated_wait_s`) and `offered_call_qps`, also per
  worker. Complete runs add `sessions`, `measured_calls` (released in [W, T)),
  `call_qps` (measured calls / (T − W)), their TTFT and TPOT mean/p50/p90/p99,
  the mean and p99 of each TTFT part, `prompt_tokens`, `reused_tokens` and their
  ratio `prefix_reuse`, transfer GB by kind in the window, and host and pool
  occupancy at T.
- `trace.jsonl` holds `{"time_s", "state"}` after each event time, or with
  `--trace-times t1 t2 …` the first state at or after each time.
  `scripts/kv_metrics.py TRACE --output M.json` integrates its waits.

**TTFT** runs from a call's release (after its tools) to its first token.
`ttft_parts` splits it into parts that sum to it:

| Part | Time spent |
|---|---|
| `kv_load` | reading the reused prefix into HBM, from the call's turn |
| `admission` | waiting at its turn for HBM to hold the prefix and the prompt |
| `queue` | the rest of the wait in the worker's queue before its first iteration |
| `prefill` | from that iteration to the first token, shared iterations included |

`state.cxl_pool.transfer_totals` in the trace counts and sizes every KV move:

| Kind | Path | Meaning |
|---|---|---|
| `store` | HBM → host DRAM | a computed block is offloaded |
| `spill` | host DRAM → CXL device | host is full; its least recently used copy moves to the pool |
| `cxl_read` | CXL device → GPU | reload from the pool |
| `host_read` | local host DRAM → GPU | reload from the worker's own host |
| `rdma_read` | remote host DRAM → GPU | reload from another worker's host over RDMA |

`state.cxl_pool` also holds per-device `pool_usage`, per-rank `hbm_usage`, host
occupancy and the number of calls waiting for KV or HBM. `state.instances[w]`
holds each worker's scheduler counters and how many compute lookups were
interpolated or extrapolated (`lookups`). `detailed_observations = true` in the
scenario adds every transfer, the block directory and per-link bytes; use it on
small runs only.

**When does the pool matter?** Only when reused KV outgrows HBM plus host
DRAM. With the default 1.5 TB of host KV per Rubin worker, light loads never
spill, and `none` and `small` give identical latencies. With `--host-kv-gib 2`
and 256 synthetic 10-call sessions, `none` discards blocks and stores 2,183 GB
as it recomputes them (TTFT mean 58 ms, p99 194 ms), while `small` spills
584 GB to the pool and stores 600 GB (30 ms, 51 ms).

### 1.5 Experiments to run

| Question | Vary | Keep fixed |
|---|---|---|
| Where does each design saturate? | `--rates`: double until saturation, then fill between the last two | scenario, sessions, seed |
| Pool vs no pool | `--design none --workers W` vs a pool with W workers | workload |
| Uniform vs non-uniform | `--design` at equal `--switches` | everything else |
| How much ISL bandwidth is enough? | `--local-global` | design, switches |
| Does the conclusion survive the Rubin compute band? | `--compute-speedup 2`, `3`, `4` | scenario |
| Does cache affinity matter? | `--routing least_loaded` vs `cache_affinity` | scenario |
| How much host DRAM does the pool replace? | `--host-kv-gib` | scenario |
| Does chunk size change the answer? | `--block-tokens 256` vs `16` | scenario |
| Does loading ahead of a free slot help? | `--max-num-inflight-kv-loads` vs off | scenario |

### 1.6 Cost

Measured on one CPU core:

| Run | Wall time |
|---|---|
| Setup, 16 or 32 switches | 0.2 s, 0.5 s |
| 20 FreeInference sessions on H200 `small` | 5 s |
| 200 FreeInference sessions (17,381 calls) on H200 `small` | 3 min, 2.6 GB |
| Sweep of 0.005, 0.01, 0.02 sessions/s with T = 2 h on Rubin `small` | 5 min |
| 256 synthetic 10-call sessions on `small`, 16 switches | 2 s, 7 s |
| 512 synthetic sessions on 32 switches | 25 s |
| 1,024 synthetic sessions on `small`, overloaded (TTFT p99 25 s) | 57 s |
| 64 synthetic sessions with 16-token chunks | 9 s (0.6 s with 256) |

Cost follows model iterations and KV transfers, not calls. FreeInference costs
about 5× more per call than synthetic sessions: its sessions are sparse in time,
so decode runs nearly one request per iteration (374k iterations for 1,148
calls, against 69k for 2,560 synthetic calls), and its long prompts move 3.6×
more blocks per call. Overload and small chunks add concurrent transfers.
Memory is dominated by the block directory: about 2 KB per stored block per
GPU, or 260 MB per TB of KV held, so it is bounded by the scenario's HBM, host
and pool capacity (about 9 GB for `small`, 72 GB for 64 workers when full).
Traces write a full snapshot per event time; enable them for analysis only.

## 2. Fidelity: what is measured and what is assumed

The rule is the Simthesizer paper's: costs come from mechanisms with explicit
parameters, measured where possible, and every unmeasured value is a declared
assumption with a source ([Simthesizer](https://arxiv.org/html/2608.24650v1)).

### 2.1 Compute: a measured table

Workers do not model kernels. Each forward pass costs what vLLM measured for
the same batch shape on real GPUs.

- **Source.** NVIDIA's AISim forward-pass dataset
  ([nvidia/aisimulate-fpm-dataset](https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset),
  revision `dbfcdd9e`): MiniMax-M2.7 FP8 on 4 × H200, pure TP4, vLLM 0.25.1.
  `scripts/import_fpm.py` converts it to `examples/data/h200/minimax_m2_7-tp4.json`.
- **Lookup.** Decode batches index by (requests, total KV tokens); prefill by
  (query tokens, attention work); mixed batches fold decode tokens into the
  prefill terms. Values are interpolated and clamped below the grid. Above
  it, each curve continues at the least-squares slope of its upper half,
  never negative; the trace counts each case. Predicting each curve's upper
  40% from the rest, this errs 14% (MAPE), against 106% for the slope of the
  last two samples, which sit at nearly equal coordinates.
- **Context length.** The table reaches MiniMax-M2.7's 204,800-token context,
  but the simulator enforces no context limit: we study the serving system, not
  model quality. Longer prompts (28% of FreeInference calls, up to 1.26 M
  tokens) continue the last measured slope in the coordinates above, which is
  linear in KV for decode and quadratic in prompt length for prefill attention.
  HBM capacity still applies; every FreeInference prompt fits one worker.
  In a 1-hour window at 0.005 sessions/s on Rubin `small`, calls over 200k
  tokens had TTFT p50 0.09 s and p90 0.23 s; `--max-input-tokens 204800` in
  `generate` keeps a workload within the measured range.
- **Validation.** The same deployment's live engine stream, recorded separately
  from the table, gives 148,541 iterations. The table predicts them within 2.5%
  (prefill), 4.6% (decode) and 11.9% (mixed) MAPE; mixed batches run about 9%
  fast because the table has no mixed rows. Weight bytes match the checkpoint,
  and modeled HBM capacity (2.42 M KV tokens per worker) matches the live peak
  (2.35 M). See `experiments/fpm-validation/guide.md`.

### 2.2 Platforms: H200 measured, Vera Rubin projected

Rubin has no public measurements, so it is **reference-guided**: the H200
table's shape is kept and its latencies are divided by a declared
`compute_speedup`. The anchor is B200, measured with the same model and vLLM:
1.7× faster than H200 on decode and 1.4× on prefill, against spec ratios of
1.6× (HBM bandwidth) and 2.3× (FP8). Spec scaling misses by 14% on decode and
prefill and by 46% on short queries over long cached KV, where B200 is slower
than H200, most likely from vLLM's attention kernel choice on Blackwell. Rubin
has 4.6× H200's HBM bandwidth; 2–4× is a conservative band and 3 the default.
Report conclusions that hold across the band, not absolute Rubin latencies.

Everything else comes from specifications. Link rates are raw rate × 0.8;
pinned PCIe Gen5 copies reach about 87% of raw
([report](https://manishklach.github.io/writings/the-pcie-tax-why-bypassing-the-host-doubles-moe-throughput.html)).

| Per worker (TP4) | `vr-nvl72` (default) | `h200` | Source |
|---|---|---|---|
| Compute | H200 table ÷ 3 | measured | §2.1 |
| HBM per GPU | 288 GB → 6.4 M KV tokens | 143,771 MiB → 2.42 M | [Vera Rubin](https://tomshardware.com/pc-components/gpus/nvidias-vera-rubin-platform-in-depth-inside-nvidias-most-complex-ai-and-hpc-platform-to-date), [DGX H200](https://www.nvidia.com/en-in/data-center/dgx-h200) |
| GPU ↔ host | NVLink-C2C 450 GB/s each way → 360 | PCIe Gen5 x16: 50 GB/s | [VR NVL72](https://www.nvidia.com/en-us/data-center/vera-rubin-nvl72/) |
| Host DRAM | 2 Vera CPUs × 1.5 TB, 1.2 TB/s LPDDR5X | 1 TB, one DDR5 socket: 307 GB/s | [Vera](https://storagereview.com/review/nvidia-vera-cpu-review), [NVL72](https://hashrateindex.com/blog/nvidia-vera-rubin-nvl72-specs-breakdown/) |
| Network | 4 × ConnectX-9 1.6 Tb/s: 640 GB/s | 4 × ConnectX-7 400 Gb/s: 160 GB/s | |
| CXL lane | Gen6 64 GT/s: 6.4 GB/s (x16 102, x8 51) | Gen5 32 GT/s: 3.15 GB/s (x16 50, x8 25) | |
| CXL device | 1 TB, 72 GB/s | 256 GB, 36 GB/s | [Samsung CMM-D 3.1 / 2.0](https://semiconductor.samsung.com/cxl-memory/cmm-d/) |
| Switch | 256 lanes, 0.25 µs per hop | same | [XConn XC50256](https://www.servethehome.com/xconn-xc50256-cxl-2-0-switch-chip-shown/), [FMS 2024](https://files.futurememorystorage.com/proceedings/2024/20240807_CXLT-202-1_Jiang.pdf) |

NVLink-C2C's 1.8 TB/s per Vera counts both directions for its 2 GPUs (65 TB/s
per rack). A Rubin worker's host DRAM is one compute tray's; an H200 worker's
is half a DGX H200's 2 TB.

CXL device latency (1.35–3× DDR5, [MICRO '23](https://arxiv.org/pdf/2303.15375))
is small beside a block copy and is not modeled. All designs assume CXL 3.x
port-based routing, which permits multi-level and non-tree fabrics
([CXL 3.0](https://computeexpresslink.org/wp-content/uploads/2023/12/CXL_3.0_white-paper_FINAL.pdf)).

### 2.3 Serving behavior: pinned to vLLM

The worker scheduler, the KV offload trigger and the cross-worker load
balancer follow specific vLLM source revisions (§3). Session routing follows
vLLM production-stack.

### 2.4 Network model: exact for its assumptions

Transfers share links by weighted max-min fairness (§3.4). Unit tests check
completion times against hand-derived cases, including a new flow that slows
others and one that speeds another up. Event handling was checked against
re-polling every transfer at every network event: on a 1,024-session overload,
completions agree to 10⁻¹³ s.

### 2.5 Not validated

- No CXL fabric was measured; rates and capacities are derated specifications.
- Rubin compute scales the H200 table uniformly; kernel effects such as the
  B200 short-query case fall inside the 2–4× band, not in the model.
- Compute beyond the 204,800-token measured context is extrapolated.
- Each worker is its own host; workers sharing a host's DRAM are not coupled.
- Host DRAM reads and writes each get the full DRAM rate.
- KV moves HBM → host → CXL; direct GPU-to-CXL writes are not modeled.
- Locality-aware KV placement and disaggregated prefill/decode on the pool are
  not implemented.

## 3. How it is built

### 3.1 Worker

Each worker is a `single_instance` system: the `minimax_m2` model sizes weights
and KV (31,744 bytes per token per GPU), the `forward_pass` runner prices
iterations from the table of §2.1, and the `chunked_prefill` scheduler follows
vLLM V1:

- Running requests go first; waiting requests are admitted first-come,
  first-served while they fit.
- Each iteration holds up to 1,024 requests and 8,192 tokens; long prompts
  are split into chunks.
- KV gets 92% of HBM minus 4 GiB of workspace and the weights, in 16-token
  blocks.
- When KV runs out, the newest running request is preempted and later resumes
  from its surviving prefix.
- The scheduler's own prefix cache is off; the pool owns KV reuse.

Details: `architectures/scheduling.md`.

### 3.2 Choosing a worker per call

Each LLM call picks a worker when it is released.

- **`least_loaded`** (default) reproduces vLLM's data-parallel load balancer
  ([`DPLBAsyncMPClient`, commit `ce72586f`](https://github.com/vllm-project/vllm/blob/ce72586f8aded8dafa7d4c745a5cef3198070e3e/vllm/v1/engine/core_client.py)).
  Every 100 ms it snapshots each worker's waiting calls (queued or loading KV),
  running calls and KV usage, and picks the lowest
  `max(in-flight, waiting + running) + 6 · waiting · max(0, usage − 0.5)`.
  The scan starts at a rotating worker, and each choice counts as one more
  waiting call until the next snapshot.
- **`session`** sends every call of a session to the same worker by a stable
  hash of its ID, as vLLM production-stack's session routing does. Calls
  without a session use `least_loaded`.
- **`cache_affinity`** reproduces production-stack's `LoadAwareRouter`
  ([`routing_logic.py`, commit `2f879ef8`](https://github.com/vllm-project/production-stack/blob/2f879ef8a7c21f975ac0b2675267b7ba681b2cfa/src/vllm_router/routers/routing_logic.py)),
  which weighs KV-cache hits against load. Worker *i* scores

  ```text
  score_i = min(matched_i, P) / P − β · (L_i − L̄) / max(1, L̄)
  ```

  | Term | Meaning |
  |---|---|
  | P | the call's prompt tokens |
  | L_i, L̄ | calls in prefill or decode on worker *i* at routing, and their mean |
  | β | `--affinity-beta`, default 1: a full hit is worth load 100% above the mean |
  | matched_i | how far into the prompt worker *i*'s host DRAM reaches |

  The floor of 1 on L̄ keeps a nearly idle fleet from amplifying one call.
  `matched` follows LMCache's controller lookup
  ([`kv_controller.py`, commit `3f7c2017`](https://github.com/LMCache/LMCache/blob/3f7c2017b1d52ed7c1f00fee9a57936080725dac/lmcache/inprocess/v1/cache_controller/controllers/kv_controller.py)).
  It walks the prompt's blocks from the start until a block is in no host
  DRAM, and gives each host the end of the **last** block it holds in that run,
  not the number of blocks it holds. With blocks 1–2 in A's DRAM and 3–4 in
  B's, A matches 2 blocks and B 4, so B wins at equal load. The pool and HBM
  belong to no LMCache instance and never count. With β = 1, a worker holding
  the whole prompt beats a cold worker at mean load until it carries twice
  the mean load. The first highest score wins; when no host holds the first
  block, the call falls back to `session`.

### 3.3 KV in the pool

- **Identity.** A prompt is a chain of 256-token blocks. A block's ID hashes
  its tokens and the previous block's ID, so two calls share a block only if
  their prompts agree up to it. Each block also records which sessions and
  running calls use it.
- **Store.** A fully computed block is copied from HBM to the worker's host
  DRAM, as vLLM's offloading connector does (`ON_COMPUTE` trigger, commit
  `b7e43db2`), unless it already has an offload copy: a block recomputed
  while a copy survives is not written again.
- **Spill.** When host DRAM is full, its least recently used copies move to
  CXL devices, round-robin over devices with space. When the pool is full, its
  least recently used blocks are deleted. With `none`, host DRAM is the last
  tier and evicts instead.
- **One offload copy.** Outside HBM, a block lives in exactly one place: its
  worker's host DRAM, after a spill a CXL device, or nowhere once evicted.
- **Reload.** A call gathers its prompt's blocks from wherever each one is:
  its own GPU's HBM (no transfer), its own host DRAM (`host_read`), another
  worker's host DRAM (`rdma_read`) or a CXL device (`cxl_read`). One call may
  mix all four; another GPU's HBM is never a source. Each GPU keeps at most
  256 reads in flight. The call starts once its blocks have arrived and HBM
  can hold them.
- **Turn.** As in vLLM, a released call joins its worker's queue at once and
  looks up and loads its prefix only at its turn. Each scheduling step serves
  running calls first, then calls whose prefix has loaded, then looks up
  waiting calls in queue order while a running slot (`max_batch`) and token
  budget (`max_num_batched_tokens`, 8192) remain; a load takes neither. A
  prefix loads only if HBM can then hold it and the rest of the prompt (vLLM's
  full-prompt check); the scheduler holds those bytes until the call is
  admitted. A call that cannot fit holds nothing and keeps its turn, so later
  calls wait. Other calls keep running while a prefix loads, and a preempted
  call loads its surviving prefix again at its next turn.
- **Early loads.** By default, as in vLLM, no load starts while running calls
  take every slot or the whole token budget, such as during a long prefill.
  `--max-num-inflight-kv-loads N` sets vLLM's option of that name: loads then
  start regardless, so a prefix can be ready when a slot frees, while at most N
  calls per worker hold KV before they run (loading, or loaded and waiting).
- **Contiguity.** Reuse ends at the first block found nowhere; later blocks are
  prefilled again even if cached. vLLM keeps one computed-token count per
  request and its KV connectors report one hit count, so a reused prefix must
  run from the start; LMCache's lookup likewise stops at the first miss. On
  1,749 FreeInference calls on `small`, no call had cached blocks after a miss
  at default capacities. With a 32 GB pool, spilling 12 TB per hour, those
  blocks were 1.6% of prefix blocks under `least_loaded` and 0.02% under
  `cache_affinity`.
- **Rewrite.** When a call ends, a reused block whose offload copy was evicted
  meanwhile is stored again; one that survived is not.
- **Protection.** A copy that is being read or awaited is never deleted.

Details: `architectures/cxl-pool.md`.

### 3.4 Topology and contention

Each compute tray is one worker: four GPUs, the host DRAM of its two Vera CPUs,
NICs and one CXL root port (the adapter). NICs and the root port attach to
Vera's PCIe, so every GPU reaches DRAM, the NICs and the pool through its
NVLink-C2C link
([Vera tray topology](https://docs.nvidia.com/dccpu/vera-mn-perf-tuning-guide/system.html#compute-tray-topology)).
Rates sit on these links and endpoints:

| Resource | Rate on Rubin | Carries |
|---|---|---|
| GPU ↔ C2C | 360 GB/s each way, per GPU | everything entering or leaving a GPU |
| host DRAM | 1.92 TB/s read, 1.92 TB/s write | every host read and write |
| NIC ↔ network | 640 GB/s | RDMA between workers |
| root port ↔ switch | CXL x16 | all traffic between a worker and the pool |
| switch ↔ device, device | CXL x8; 72 GB/s read and write | one device's reads and writes |
| switch ↔ switch | ISL lanes | traffic between switches |

Links between DRAM, C2C, NIC and root port inside a tray are unbounded, so
transfers take these paths:

| Transfer | Path |
|---|---|
| `store` | GPU → C2C → DRAM |
| `host_read` | DRAM → C2C → GPU |
| `rdma_read` | remote DRAM → remote NIC → network → NIC → C2C → GPU |
| `spill` | DRAM → root port → switches → device |
| `cxl_read` | device → switches → root port → C2C → GPU |

Per Rubin tray, reads therefore top out at 1.44 TB/s from local DRAM (four
C2C links), 640 GB/s over RDMA (the NICs) and 102 GB/s from the pool (the x16
root port; 51 GB/s per device). Fixed per-transfer costs such as RDMA setup are
not modeled: a block is 8 MB per GPU, and 256 reads overlap.

- **Routes.** A transfer passes its required stops (both NICs for RDMA, the
  worker's root port for the pool) and otherwise takes the fewest-hop paths;
  pool traffic stays on the CXL switches even where a host relay is shorter;
  when several tie, its bytes split evenly across them. Parallel x16 links between two switches act
  as one trunk.
- **Sharing.** Transfers on a link share it by weighted max-min fairness: the
  most contended link fixes the rate of its flows first, and capacity a flow
  cannot use elsewhere goes to the others. A hot device, root port or trunk thus
  limits only the flows that cross it.
- **Events.** Rates are recomputed whenever a transfer starts (more
  contention) or a flow finishes sending (less). Each transfer is next checked
  at its own next event. A recomputation that moves a transfer's event earlier
  checks it earlier; one that moves it later is caught by the early check,
  which reschedules it. Transfers starting at the same instant share one
  recomputation.

Details: `architectures/cxl-topology.md` and `architectures/cost-models.md`.
