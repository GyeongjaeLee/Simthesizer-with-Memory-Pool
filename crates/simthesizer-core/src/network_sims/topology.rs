use std::collections::{BTreeMap, HashMap, HashSet};

use anyhow::{anyhow, ensure, Context, Result};
use serde::{Deserialize, Serialize};

use super::flow::{FlowNetworkSimulator, RouteInfo};
use crate::{types::NetworkDeviceID, Module, NetworkSimulator};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EndpointKind {
    Switch,
    Adapter,
    Memory,
    Dram,
    Nic,
    Fabric,
    Gpu,
    Junction,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Endpoint {
    pub id: String,
    pub kind: EndpointKind,
    pub node: Option<String>,
    pub lane_budget: Option<u32>,
    pub capacity_bytes: Option<u64>,
    pub read_bandwidth: Option<f64>,
    pub write_bandwidth: Option<f64>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Link {
    pub id: String,
    pub a: String,
    pub b: String,
    pub a_port: String,
    pub b_port: String,
    pub width: Option<u32>,
    pub bandwidth_ab: f64,
    pub bandwidth_ba: Option<f64>,
    pub latency: f64,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct WorkerBinding {
    pub worker: usize,
    #[serde(default)]
    pub rank: usize,
    pub node: String,
    #[serde(alias = "adapter", deserialize_with = "deserialize_adapters")]
    pub adapters: Vec<String>,
    pub dram: String,
    pub nic: String,
    pub gpu: String,
}

fn deserialize_adapters<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<Vec<String>, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Selection {
        One(String),
        Many(Vec<String>),
    }
    Ok(match Selection::deserialize(deserializer)? {
        Selection::One(adapter) => vec![adapter],
        Selection::Many(adapters) => adapters,
    })
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RouteRequirement {
    pub src: String,
    pub dst: String,
    #[serde(default)]
    pub via: Vec<String>,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Routing {
    #[default]
    EqualShortest,
    Single,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Topology {
    pub endpoints: Vec<Endpoint>,
    pub links: Vec<Link>,
    #[serde(default)]
    pub workers: Vec<WorkerBinding>,
    #[serde(default)]
    pub required_routes: Vec<RouteRequirement>,
    #[serde(default)]
    pub routing: Routing,
}

pub struct TopologyNetwork {
    pub flow: FlowNetworkSimulator,
    topology: Topology,
    ids: BTreeMap<String, NetworkDeviceID>,
    lanes: BTreeMap<String, u32>,
    cuts: BTreeMap<String, DirectedCut>,
    adapter_links: BTreeMap<String, (usize, usize)>,
    transfer_routes: Vec<TransferRoute>,
}

#[derive(PartialEq, Serialize)]
struct TransferRoute {
    src: String,
    dst: String,
    source_owner: Option<(usize, usize)>,
    destination_owner: Option<(usize, usize)>,
    selected_adapter: Option<String>,
    via: Vec<String>,
    network_src: NetworkDeviceID,
    network_dst: NetworkDeviceID,
}

#[derive(Default, Serialize)]
struct DirectedCut {
    outgoing_bytes_per_second: f64,
    incoming_bytes_per_second: f64,
    physical_lanes: u32,
}

impl Topology {
    pub fn build(&self) -> Result<TopologyNetwork> {
        let mut flow = FlowNetworkSimulator::default();
        flow.equal_path_striping = matches!(self.routing, Routing::EqualShortest);
        let mut ids = BTreeMap::new();
        let mut ports = HashMap::new();
        let mut endpoints = HashMap::new();
        let mut lanes = BTreeMap::new();
        let mut cuts = BTreeMap::<String, DirectedCut>::new();
        let mut adapter_links = BTreeMap::new();
        for endpoint in &self.endpoints {
            ensure!(!endpoint.id.is_empty(), "empty endpoint name");
            ensure!(
                endpoints.insert(endpoint.id.as_str(), endpoint).is_none(),
                "duplicate endpoint {}",
                endpoint.id
            );
            let id = flow.add_device();
            ids.insert(endpoint.id.clone(), id);
            if endpoint.kind == EndpointKind::Switch {
                let budget = endpoint.lane_budget.unwrap_or(256);
                ensure!(
                    budget > 0 && budget <= 256,
                    "switch {} lane budget must be 1..256",
                    endpoint.id
                );
                lanes.insert(endpoint.id.clone(), 0u32);
                cuts.insert(endpoint.id.clone(), DirectedCut::default());
            }
            if endpoint.kind == EndpointKind::Memory {
                ensure!(
                    endpoint.capacity_bytes.is_some_and(|bytes| bytes > 0),
                    "memory {} needs positive capacity_bytes",
                    endpoint.id
                );
                let read = endpoint
                    .read_bandwidth
                    .ok_or_else(|| anyhow!("memory {} needs read_bandwidth", endpoint.id))?;
                let write = endpoint
                    .write_bandwidth
                    .ok_or_else(|| anyhow!("memory {} needs write_bandwidth", endpoint.id))?;
                let port = flow.add_device();
                ports.insert(endpoint.id.clone(), port);
                flow.add_named_link(id, port, read, 0.0, format!("{}:read", endpoint.id))?;
                flow.add_named_link(port, id, write, 0.0, format!("{}:write", endpoint.id))?;
            } else {
                ensure!(
                    endpoint.read_bandwidth.is_none() && endpoint.write_bandwidth.is_none(),
                    "read/write rates are memory service inputs; use explicit links for {}",
                    endpoint.id
                );
                ports.insert(endpoint.id.clone(), id);
            }
        }
        let mut physical_ports = HashSet::new();
        let mut physical_widths = HashMap::<&str, Vec<u32>>::new();
        let mut link_names = HashSet::new();
        for link in &self.links {
            ensure!(
                !link.id.is_empty() && link_names.insert(&link.id),
                "empty or duplicate link {}",
                link.id
            );
            let a = endpoints
                .get(link.a.as_str())
                .ok_or_else(|| anyhow!("link {} has invalid endpoint {}", link.id, link.a))?;
            let b = endpoints
                .get(link.b.as_str())
                .ok_or_else(|| anyhow!("link {} has invalid endpoint {}", link.id, link.b))?;
            ensure!(a.id != b.id, "link {} joins an endpoint to itself", link.id);
            if let Some(width) = link.width {
                ensure!(
                    width > 0 && width <= 16,
                    "link {} physical width must be 1..16",
                    link.id
                );
                ensure!(
                    link.bandwidth_ba.is_some(),
                    "physical link {} needs both directions",
                    link.id
                );
                ensure!(
                    a.kind == EndpointKind::Switch || b.kind == EndpointKind::Switch,
                    "physical link {} must attach a switch",
                    link.id
                );
                for (endpoint, port) in [(*a, &link.a_port), (*b, &link.b_port)] {
                    ensure!(
                        !port.is_empty()
                            && physical_ports.insert((endpoint.id.as_str(), port.as_str())),
                        "duplicate or empty physical port {}:{}",
                        endpoint.id,
                        port
                    );
                    ensure!(
                        matches!(
                            endpoint.kind,
                            EndpointKind::Switch | EndpointKind::Adapter | EndpointKind::Memory
                        ),
                        "invalid physical endpoint {}",
                        endpoint.id
                    );
                    physical_widths.entry(&endpoint.id).or_default().push(width);
                    if let Some(used) = lanes.get_mut(&endpoint.id) {
                        *used = used
                            .checked_add(width)
                            .context("switch lane count overflow")?;
                        ensure!(
                            *used <= endpoint.lane_budget.unwrap_or(256),
                            "switch {} uses {} lanes over budget {}",
                            endpoint.id,
                            used,
                            endpoint.lane_budget.unwrap_or(256)
                        );
                    }
                }
                if a.kind == EndpointKind::Switch && b.kind == EndpointKind::Switch {
                    for (id, outgoing, incoming) in [
                        (&a.id, link.bandwidth_ab, link.bandwidth_ba.unwrap()),
                        (&b.id, link.bandwidth_ba.unwrap(), link.bandwidth_ab),
                    ] {
                        let cut = cuts.get_mut(id).unwrap();
                        cut.outgoing_bytes_per_second += outgoing;
                        cut.incoming_bytes_per_second += incoming;
                        cut.physical_lanes += width;
                    }
                }
            }
            let ab = flow.add_named_link(
                ports[&link.a],
                ports[&link.b],
                link.bandwidth_ab,
                link.latency,
                format!("{}:ab", link.id),
            )?;
            if let Some(bandwidth) = link.bandwidth_ba {
                let ba = flow.add_named_link(
                    ports[&link.b],
                    ports[&link.a],
                    bandwidth,
                    link.latency,
                    format!("{}:ba", link.id),
                )?;
                if link.width.is_some() {
                    if a.kind == EndpointKind::Adapter {
                        adapter_links.insert(a.id.clone(), (ab, ba));
                    }
                    if b.kind == EndpointKind::Adapter {
                        adapter_links.insert(b.id.clone(), (ba, ab));
                    }
                }
            }
        }
        let mut adapters = HashMap::new();
        for endpoint in &self.endpoints {
            match endpoint.kind {
                EndpointKind::Adapter | EndpointKind::Memory => {
                    let width = if endpoint.kind == EndpointKind::Adapter {
                        16
                    } else {
                        8
                    };
                    ensure!(
                        physical_widths.get(endpoint.id.as_str()).map(Vec::as_slice)
                            == Some(&[width]),
                        "{} needs exactly one physical x{} port",
                        endpoint.id,
                        width
                    );
                    if endpoint.kind == EndpointKind::Adapter {
                        let node = endpoint
                            .node
                            .as_ref()
                            .ok_or_else(|| anyhow!("adapter {} needs node", endpoint.id))?;
                        adapters.insert(&endpoint.id, node);
                    }
                }
                _ => {}
            }
        }
        let mut workers = HashSet::new();
        for worker in &self.workers {
            ensure!(
                workers.insert((worker.worker, worker.rank)),
                "duplicate worker {} rank {}",
                worker.worker,
                worker.rank
            );
            ensure!(
                !worker.adapters.is_empty(),
                "worker {} rank {} needs at least one adapter",
                worker.worker,
                worker.rank
            );
            let mut selected = HashSet::new();
            for adapter in &worker.adapters {
                ensure!(
                    selected.insert(adapter),
                    "worker {} rank {} repeats adapter {}",
                    worker.worker,
                    worker.rank,
                    adapter
                );
                ensure!(
                    adapters.get(adapter).copied() == Some(&worker.node),
                    "worker {} rank {} node/adapter binding mismatch",
                    worker.worker,
                    worker.rank
                );
            }
            for (name, kind) in [
                (&worker.dram, EndpointKind::Dram),
                (&worker.nic, EndpointKind::Nic),
                (&worker.gpu, EndpointKind::Gpu),
            ] {
                ensure!(
                    endpoints
                        .get(name.as_str())
                        .is_some_and(|endpoint| endpoint.kind == kind),
                    "worker {} has invalid {:?} endpoint {}",
                    worker.worker,
                    kind,
                    name
                );
            }
        }
        let mut network = TopologyNetwork {
            flow,
            topology: self.clone(),
            ids,
            lanes,
            cuts,
            adapter_links,
            transfer_routes: Vec::new(),
        };
        // A declared worker must be able to spill and reload every pool device.
        let mut defaults = HashSet::new();
        for worker in &self.workers {
            for device in network.memory_ids() {
                for adapter in &worker.adapters {
                    for (src, dst) in [(&worker.dram, &device), (&device, &worker.gpu)] {
                        let via = vec![network.endpoint(adapter)?];
                        network
                            .flow
                            .check_waypoints(network.endpoint(src)?, network.endpoint(dst)?, &via)
                            .with_context(|| {
                                format!(
                                    "worker {} rank {} route {src} -> {dst} via {adapter}",
                                    worker.worker, worker.rank
                                )
                            })?;
                    }
                }
                for (src, dst) in [(&worker.dram, &device), (&device, &worker.gpu)] {
                    if defaults.insert((src.clone(), dst.clone())) {
                        network.set_route(src, dst, &[&worker.adapters[0]])?;
                    }
                }
            }
            for destination in &self.workers {
                let via = if worker.node == destination.node {
                    Vec::new()
                } else {
                    vec![worker.nic.as_str(), destination.nic.as_str()]
                };
                if defaults.insert((worker.dram.clone(), destination.gpu.clone())) {
                    network
                        .set_route(&worker.dram, &destination.gpu, &via)
                        .with_context(|| {
                            format!(
                                "worker {} DRAM route to worker {}",
                                worker.worker, destination.worker
                            )
                        })?;
                }
            }
        }
        for route in &self.required_routes {
            if route.via.is_empty() {
                network
                    .flow
                    .check_route(network.endpoint(&route.src)?, network.endpoint(&route.dst)?)
            } else {
                network.set_route(
                    &route.src,
                    &route.dst,
                    &route.via.iter().map(String::as_str).collect::<Vec<_>>(),
                )
            }
            .with_context(|| format!("required route {} -> {}", route.src, route.dst))?;
        }
        Ok(network)
    }
}

impl TopologyNetwork {
    fn set_route(&mut self, src: &str, dst: &str, via: &[&str]) -> Result<()> {
        let waypoints = via
            .iter()
            .map(|name| self.endpoint(name))
            .collect::<Result<Vec<_>>>()?;
        self.flow
            .set_waypoints(self.endpoint(src)?, self.endpoint(dst)?, waypoints)
    }

    pub fn endpoint(&self, name: &str) -> Result<NetworkDeviceID> {
        self.ids
            .get(name)
            .copied()
            .ok_or_else(|| anyhow!("unknown topology endpoint {name}"))
    }

    pub fn worker_rank(&self, worker: usize, rank: usize) -> Result<&WorkerBinding> {
        self.topology
            .workers
            .iter()
            .find(|binding| binding.worker == worker && binding.rank == rank)
            .ok_or_else(|| anyhow!("unknown topology worker {worker} rank {rank}"))
    }

    pub fn transfer_route(
        &mut self,
        src: &str,
        dst: &str,
        source_owner: Option<(usize, usize)>,
        destination_owner: Option<(usize, usize)>,
        block_index: u64,
    ) -> Result<(NetworkDeviceID, NetworkDeviceID)> {
        let source = source_owner
            .map(|(worker, rank)| self.worker_rank(worker, rank))
            .transpose()?;
        let destination = destination_owner
            .map(|(worker, rank)| self.worker_rank(worker, rank))
            .transpose()?;
        let is_memory = |name: &str| {
            self.topology
                .endpoints
                .iter()
                .any(|e| e.id == name && e.kind == EndpointKind::Memory)
        };
        let adapter_owner = if is_memory(src) {
            destination
        } else if is_memory(dst) {
            source
        } else {
            None
        };
        let selected_adapter = adapter_owner.map(|owner| {
            let index = ((block_index % owner.adapters.len() as u64) as usize
                + owner.rank % owner.adapters.len())
                % owner.adapters.len();
            owner.adapters[index].clone()
        });
        let mut via = if let Some(adapter) = &selected_adapter {
            vec![adapter.clone()]
        } else if let (Some(source), Some(destination)) = (source, destination) {
            if source.node != destination.node {
                vec![source.nic.clone(), destination.nic.clone()]
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };
        if let Some(route) = self
            .topology
            .required_routes
            .iter()
            .rev()
            .find(|route| route.src == src && route.dst == dst && !route.via.is_empty())
        {
            if let Some(adapter) = &selected_adapter {
                ensure!(
                    route.via.contains(adapter),
                    "required route {src} -> {dst} bypasses selected adapter {adapter}"
                );
                ensure!(
                    route
                        .via
                        .iter()
                        .filter(|name| self.adapter_links.contains_key(*name))
                        .all(|name| name == adapter),
                    "required route {src} -> {dst} selects another adapter"
                );
            }
            via = route.via.clone();
        }
        let waypoints = via
            .iter()
            .map(|name| self.endpoint(name))
            .collect::<Result<Vec<_>>>()?;
        let (network_src, network_dst) =
            self.flow
                .bind_route(self.endpoint(src)?, self.endpoint(dst)?, waypoints)?;
        let route = TransferRoute {
            src: src.into(),
            dst: dst.into(),
            source_owner,
            destination_owner,
            selected_adapter,
            via,
            network_src,
            network_dst,
        };
        if !self.transfer_routes.contains(&route) {
            self.transfer_routes.push(route);
        }
        Ok((network_src, network_dst))
    }

    pub fn memory_ids(&self) -> Vec<String> {
        self.topology
            .endpoints
            .iter()
            .filter(|endpoint| endpoint.kind == EndpointKind::Memory)
            .map(|endpoint| endpoint.id.clone())
            .collect()
    }

    pub fn memory_capacity(&self, name: &str) -> Result<u64> {
        self.topology
            .endpoints
            .iter()
            .find(|endpoint| endpoint.id == name && endpoint.kind == EndpointKind::Memory)
            .and_then(|endpoint| endpoint.capacity_bytes)
            .ok_or_else(|| anyhow!("unknown memory endpoint {name}"))
    }

    pub fn route_info(&mut self, src: &str, dst: &str) -> Result<Vec<RouteInfo>> {
        self.flow
            .route_info(self.endpoint(src)?, self.endpoint(dst)?)
    }

    pub fn observe(&self) -> serde_json::Value {
        let flow = self.flow.observe();
        let adapters = self
            .adapter_links
            .iter()
            .map(|(name, &(outgoing, incoming))| {
                (
                    name.clone(),
                    serde_json::json!({
                        "outgoing_link": outgoing, "incoming_link": incoming,
                        "outgoing_bytes": flow["links"][outgoing]["bytes"],
                        "incoming_bytes": flow["links"][incoming]["bytes"],
                    }),
                )
            })
            .collect::<BTreeMap<_, _>>();
        serde_json::json!({"inventory": self.topology, "endpoint_ids": self.ids,
            "lanes": self.lanes, "cuts": self.cuts, "adapters": adapters,
            "transfer_routes": self.transfer_routes, "flow": flow})
    }
}
