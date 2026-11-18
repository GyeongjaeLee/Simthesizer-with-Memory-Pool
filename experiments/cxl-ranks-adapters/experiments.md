# CXL example

`examples/h200-tp4-shared.toml` configures four H200 MiniMax-M2.7 TP4 workers
sharing CXL adapters. Workers use the validated forward-pass table of
[`4xh200--minimax_m2_7--realistic`](../../examples/scenarios/4xh200--minimax_m2_7--realistic/simthesizer.toml).
The workload supplies explicit prefix identities, seeded host KV and memory
pressure; topology rates and capacities are illustrative, not measured.

```sh
cargo run --release -- \
  --config experiments/cxl-ranks-adapters/examples/h200-tp4-shared.toml \
  --workload experiments/cxl-ranks-adapters/examples/h200-tp4-shared.jsonl \
  --output-jsonl output/cxl-example/results.jsonl \
  --output-engine-trace output/cxl-example/trace.jsonl

python3 experiments/cxl-phase1/generate.py --out output/cxl-topologies
```

The generator emits explicit single-switch and multi-switch inventories.
Replace illustrative service rates before using these examples for hardware studies.
Native `cxl_pool` and `cxl_topology` tests cover ownership, capacity and contention.
See [KV storage](../../architectures/cxl-pool.md) and
[topology](../../architectures/cxl-topology.md) for component contracts.
