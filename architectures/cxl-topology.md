# Configurable CXL topology

`Topology` declares physical endpoints, ports, directed service budgets and
worker bindings. The CXL system compiles it into one flow simulator; spill,
reload and background graph nodes reference that same service state.

Physical links have a width and two independently configured directions.
Each attached switch counts the width once, within its maximum 256 lanes.
Adapters have one x16 port and a node identity; a node may have several.
Memory devices have one x8 port, explicit capacity, and separate read/write
service rates. Each worker selects an ordered adapter list on its node,
nonempty whenever the topology has memory devices; several workers may share adapters. Physical ports count once,
regardless of binding count. Duplicate ports, invalid bindings,
missing routes and excess physical lanes fail initialization. Traffic
oversubscription remains valid.

Links without a width represent explicit service/path resources. Direct DMA
fixtures use DRAM→adapter for spill, adapter→GPU ingress for CXL reads, and
DRAM→NIC→compute fabric→NIC→GPU ingress for remote DRAM. Shared ingress is
an explicit edge; GPU destination identity imposes no implicit cap. Device
payload endpoints have mandatory read/write service edges before their
physical ports; host DRAM endpoints may declare the same edges, so all host
traffic shares one memory rate. Independent link directions and fabrics remain independent.

Worker bindings constrain spill through the source adapter, CXL reload
through the destination adapter, and cross-node DRAM/HBM through both NICs.
A route segment between two CXL endpoints (switches, adapters, memory devices)
uses only links among them, so pool traffic never relays through another host's
adapter and NIC.
Each block transfer selects
`adapters[logical_block_index % adapters.len()]`; list order is
authoritative. Immutable route handles refer to the same physical links, so
shared DRAM/GPU endpoints do not overwrite another worker's selection or
duplicate bandwidth. Same-node DRAM follows its configured local path.
`required_routes` checks named endpoints; optional `via` overrides must retain
the selected adapter and cannot name another adapter for an owned transfer.
Unowned endpoint traffic uses the first declared binding's default route,
unless explicitly overridden.

A worker's GPUs form one `gpu` endpoint whose links carry their summed rates.
Legacy `adapter = "a0"` remains accepted. For example:

```toml
[[system.topology.workers]]
worker = 0
node = "node0"
adapters = ["a0", "a1"]
dram = "dram0"
nic = "nic0"
gpu = "gpu1"
```

`equal_shortest` divides each transfer's integer bytes equally among all
shortest-hop paths between consecutive waypoints, in link declaration order;
earlier paths receive remainder bytes. Each stripe has 1/N fairness weight,
so a striped transfer claims one flow's share of a shared link. Every stripe shares the physical link state.
Network striping does not change whole-block storage placement. `single`
retains widest-bottleneck, latency, then hop routing. Existing flow scenarios
default to their original single-path behavior.

Snapshots expose inventory, endpoint IDs, switch lanes, directed inter-switch
cut capacities, owned route handles and selected adapters, chosen paths and
their bottlenecks, and per-link transmitted bytes, busy seconds and active/peak
stripe counts. Adapter incoming/outgoing payload counters read its sole
physical port once. Per-link bytes count every hop; a unique leaf-boundary
payload oracle must select its crossing once.
Rates are effective bytes/second; latency and busy time are seconds.

This is a fluid model with propagation after payload transmission, independent
read/write budgets and weighted max-min link shares.
Equal path striping is validated for symmetric paths and balanced demand;
it does not claim optimal allocation on unequal paths, packet behavior or
measured CXL latency fidelity. Counts and widths remain configuration data.

Native `cxl_topology` tests cover lane ledgers, parallel links, endpoint limits,
worker/adapter bindings and route isolation.
[`experiments/cxl-pool/generate.py`](../experiments/cxl-pool/generate.py) emits
pool-free, single-switch, uniform and non-uniform scenarios.

Parallel physical links between two switches form one flow trunk per direction
with their summed rate, while lanes stay counted per port. Equal striping loads
members equally, so this matches per-port flows without enumerating every
member combination along a path; observations report each member's share.
