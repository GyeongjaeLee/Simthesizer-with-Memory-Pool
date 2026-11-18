# CXL example

`examples/tp2-shared.toml` configures four TP2 workers sharing CXL adapters.
The workload supplies explicit prefix identities, seeded host KV and memory
pressure. Component costs are synthetic, not hardware measurements.

```sh
cargo run --release -- \
  --config experiments/cxl-ranks-adapters/examples/tp2-shared.toml \
  --workload experiments/cxl-ranks-adapters/examples/tp2-shared.jsonl \
  --output-jsonl output/cxl-example/results.jsonl \
  --output-engine-trace output/cxl-example/trace.jsonl

python3 experiments/cxl-phase1/generate.py --out output/cxl-topologies
```

The generator emits explicit single-switch and multi-switch inventories.
Replace synthetic service rates before using these examples for hardware studies.
Native `cxl_pool` and `cxl_topology` tests cover ownership, capacity and contention.
See [KV storage](../../architectures/cxl-pool.md) and
[topology](../../architectures/cxl-topology.md) for component contracts.
