use std::any::Any;
use std::cmp::Ordering;
use std::cmp::Reverse;
use std::collections::{BTreeSet, BinaryHeap, HashMap, HashSet, VecDeque};

use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};

use crate::module::Module;
use crate::network_sim::NetworkSimulator;
use crate::types::{NetworkDeviceID, NodeID, PollResult};

type FlowID = u64;

const ROUTE_COMPARE_EPSILON: f64 = 1e-12;
const RELATIVE_BYTES_EPSILON: f64 = 1e-12;
const ABSOLUTE_BYTES_EPSILON: f64 = 1e-9;
const RELATIVE_TIME_EPSILON: f64 = 1e-12;
const ABSOLUTE_TIME_EPSILON: f64 = 1e-15;

#[derive(Debug, Clone)]
struct DirectionalLink {
    src: NetworkDeviceID,
    dst: NetworkDeviceID,
    name: String,
    bandwidth: f64,
    latency: f64,
    /// Transmitting flows.
    flows: usize,
    bytes: f64,
    busy_seconds: f64,
    peak_flows: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct Route {
    link_indices: Vec<usize>,
    total_latency: f64,
}

/// An `f64` ordered by `total_cmp`, for heaps.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Due(f64);
impl Eq for Due {}
impl PartialOrd for Due {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Due {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// Transmitting flows on one route with one weight. Max-min fairness gives
/// them one rate, so they are priced and advanced together.
#[derive(Debug)]
struct FlowClass {
    route: Route,
    weight: f64,
    /// Rate of each member; valid while `rates_valid`.
    rate: f64,
    /// Bytes each member has received since the class last emptied.
    served: f64,
    /// Members by the `served` value at which they finish transmitting.
    members: BinaryHeap<Reverse<(Due, FlowID)>>,
}

#[derive(Debug)]
struct Flow {
    class: usize,
    total_bytes: f64,
    node: NodeID,
    /// Class `served` value at which transmission ends, then the arrival time
    /// of its last bit.
    due: f64,
    propagating: bool,
}

#[derive(Debug)]
struct Transfer {
    flows: Vec<FlowID>,
    /// Flows not yet arrived.
    remaining: usize,
    /// When the engine polls this transfer next.
    wake: f64,
}

/// Buffers reused across rate computations.
#[derive(Debug, Default)]
struct RateScratch {
    /// Directional link index -> active link index.
    link_of: Vec<usize>,
    /// Active link -> directional link index.
    active_links: Vec<usize>,
    capacity: Vec<f64>,
    demand: Vec<f64>,
    unfixed: Vec<usize>,
    /// Classes (positions in `active`) per active link.
    members: Vec<Vec<usize>>,
    fixed: Vec<bool>,
    saturated: Vec<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RouteInfo {
    pub links: Vec<usize>,
    pub bottleneck: f64,
    pub latency: f64,
}

#[derive(Debug, Clone)]
struct PathState {
    total_latency: f64,
    hops: usize,
    path_nodes: Vec<NetworkDeviceID>,
    link_indices: Vec<usize>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FlowNetworkSimulator {
    #[serde(default)]
    pub equal_path_striping: bool,
    #[serde(skip_deserializing, default)]
    next_device_id: NetworkDeviceID,

    #[serde(skip_deserializing, default)]
    next_flow_id: FlowID,

    #[serde(skip_deserializing, default)]
    time: f64,

    #[serde(skip_deserializing, default)]
    links: Vec<DirectionalLink>,

    #[serde(skip_deserializing, default)]
    outgoing_links: HashMap<NetworkDeviceID, Vec<usize>>,

    #[serde(skip_deserializing, default)]
    incoming_links: HashMap<NetworkDeviceID, Vec<usize>>,

    #[serde(skip_deserializing, default)]
    routes: HashMap<(NetworkDeviceID, NetworkDeviceID), Vec<Route>>,

    #[serde(skip_deserializing, default)]
    waypoints: HashMap<(NetworkDeviceID, NetworkDeviceID), Vec<NetworkDeviceID>>,

    /// A route segment between two domain devices uses only links inside it.
    #[serde(skip_deserializing, default)]
    domain: HashSet<NetworkDeviceID>,

    #[serde(skip_deserializing, default)]
    bound_routes:
        HashMap<(NetworkDeviceID, NetworkDeviceID), (NetworkDeviceID, Vec<NetworkDeviceID>)>,

    #[serde(skip_deserializing, default)]
    route_handles:
        HashMap<(NetworkDeviceID, NetworkDeviceID, Vec<NetworkDeviceID>), NetworkDeviceID>,

    #[serde(skip_deserializing, default)]
    flows: HashMap<FlowID, Flow>,

    #[serde(skip_deserializing, default)]
    classes: Vec<FlowClass>,

    #[serde(skip_deserializing, default)]
    class_index: HashMap<(Vec<usize>, u64), usize>,

    /// Each pair's classes for a transfer split over all its routes.
    #[serde(skip_deserializing, default)]
    pair_classes: HashMap<(NetworkDeviceID, NetworkDeviceID), Vec<usize>>,

    /// Classes with members, in index order.
    #[serde(skip_deserializing, default)]
    active: BTreeSet<usize>,

    /// Propagating flows by arrival time.
    #[serde(skip_deserializing, default)]
    propagation: BinaryHeap<Reverse<(Due, FlowID)>>,

    /// Transfers whose last flow arrived since the last wake-up check.
    #[serde(skip_deserializing, default)]
    arrived: Vec<NodeID>,

    /// Flow rates depend only on the active flow set.
    #[serde(skip_deserializing, default)]
    rates_valid: bool,

    /// Pending transfers by engine node.
    #[serde(skip_deserializing, default)]
    transfers: HashMap<NodeID, Transfer>,

    /// Polls brought forward since the engine last asked, and the time of the
    /// poll in progress, before which none may land.
    #[serde(skip_deserializing, default)]
    wakeups: Vec<(NodeID, f64)>,
    #[serde(skip_deserializing, default)]
    poll_time: f64,

    #[serde(skip_deserializing, default)]
    scratch: RateScratch,

    /// Next flow event after reconciling at `time`; `None` when flow state
    /// changed since. Polls at one timestamp share it.
    #[serde(skip_deserializing, default)]
    settled: Option<Option<f64>>,
}

crate::register_network_simulator!(FlowNetworkSimulator, "flow");

impl Module for FlowNetworkSimulator {
    fn observe(&self) -> serde_json::Value {
        let links = self
            .links
            .iter()
            .enumerate()
            .map(|(id, link)| {
                serde_json::json!({
                    "id": id, "name": link.name, "src": link.src, "dst": link.dst,
                    "bandwidth": link.bandwidth, "latency": link.latency,
                    "bytes": link.bytes, "busy_seconds": link.busy_seconds,
                    "active_flows": link.flows, "peak_flows": link.peak_flows,
                })
            })
            .collect::<Vec<_>>();
        let mut pairs = self.routes.keys().copied().collect::<Vec<_>>();
        pairs.sort_unstable();
        let routes = pairs.iter().map(|&(src, dst)| serde_json::json!({
            "src": src, "dst": dst,
            "paths": self.routes[&(src, dst)].iter().map(|route| self.describe_route(route)).collect::<Vec<_>>()
        })).collect::<Vec<_>>();
        serde_json::json!({"time": self.time, "links": links, "routes": routes,
            "active_subflows": self.flows.len()})
    }
}

impl FlowNetworkSimulator {
    fn route_for(&mut self, src: NetworkDeviceID, dst: NetworkDeviceID) -> Result<Vec<Route>> {
        if src == dst {
            return Ok(vec![Route {
                link_indices: Vec::new(),
                total_latency: 0.0,
            }]);
        }

        if let Some(route) = self.routes.get(&(src, dst)) {
            return Ok(route.clone());
        }

        let (physical_dst, via) = match self.bound_routes.get(&(src, dst)) {
            Some((physical_dst, via)) => (*physical_dst, via.as_slice()),
            None => (
                dst,
                self.waypoints
                    .get(&(src, dst))
                    .map(Vec::as_slice)
                    .unwrap_or(&[]),
            ),
        };
        let route = self.routes_via(src, physical_dst, via)?;
        self.routes.insert((src, dst), route.clone());
        Ok(route)
    }

    fn routes_via(
        &self,
        src: NetworkDeviceID,
        dst: NetworkDeviceID,
        via: &[NetworkDeviceID],
    ) -> Result<Vec<Route>> {
        let stops = std::iter::once(src)
            .chain(via.iter().copied())
            .chain(std::iter::once(dst))
            .collect::<Vec<_>>();
        let mut route = vec![Route {
            link_indices: Vec::new(),
            total_latency: 0.0,
        }];
        for segment in stops.windows(2) {
            if segment[0] == segment[1] {
                continue;
            }
            let choices = if self.equal_path_striping {
                self.find_equal_shortest_routes(segment[0], segment[1])?
            } else {
                vec![self.find_shortest_route(segment[0], segment[1])?]
            };
            route = route
                .iter()
                .flat_map(|prefix| {
                    choices.iter().map(move |suffix| Route {
                        link_indices: prefix
                            .link_indices
                            .iter()
                            .chain(&suffix.link_indices)
                            .copied()
                            .collect(),
                        total_latency: prefix.total_latency + suffix.total_latency,
                    })
                })
                .collect();
        }
        for path in &route {
            let mut seen = HashSet::new();
            if !path.link_indices.iter().all(|link| seen.insert(link)) {
                bail!("route waypoints repeat a directed link from {src} to {dst}");
            }
        }
        Ok(route)
    }

    /// Bind a route without changing other transfers between the physical endpoints.
    pub fn bind_route(
        &mut self,
        src: NetworkDeviceID,
        dst: NetworkDeviceID,
        via: Vec<NetworkDeviceID>,
    ) -> Result<(NetworkDeviceID, NetworkDeviceID)> {
        if src == dst && via.is_empty() {
            return Ok((src, dst));
        }
        let key = (src, dst, via.clone());
        if let Some(&handle) = self.route_handles.get(&key) {
            return Ok((src, handle));
        }
        let routes = self.routes_via(src, dst, &via)?;
        let handle = self.add_device();
        self.bound_routes.insert((src, handle), (dst, via));
        self.route_handles.insert(key, handle);
        self.routes.insert((src, handle), routes);
        Ok((src, handle))
    }

    fn describe_route(&self, route: &Route) -> RouteInfo {
        RouteInfo {
            links: route.link_indices.clone(),
            bottleneck: route
                .link_indices
                .iter()
                .map(|&i| self.links[i].bandwidth)
                .fold(f64::INFINITY, f64::min),
            latency: route.total_latency,
        }
    }

    pub fn route_info(
        &mut self,
        src: NetworkDeviceID,
        dst: NetworkDeviceID,
    ) -> Result<Vec<RouteInfo>> {
        Ok(self
            .route_for(src, dst)?
            .iter()
            .map(|route| self.describe_route(route))
            .collect())
    }

    /// Confines route segments between `devices` to links among them.
    pub fn confine(&mut self, devices: HashSet<NetworkDeviceID>) {
        self.domain = devices;
        self.routes.clear();
    }

    pub fn confined(&self, src: NetworkDeviceID, dst: NetworkDeviceID) -> bool {
        self.domain.contains(&src) && self.domain.contains(&dst)
    }

    /// Links at `node` from `adjacency`, only those inside the domain if `confined`.
    fn links_at<'a>(
        &'a self,
        adjacency: &'a HashMap<NetworkDeviceID, Vec<usize>>,
        node: NetworkDeviceID,
        confined: bool,
    ) -> impl DoubleEndedIterator<Item = usize> + 'a {
        adjacency
            .get(&node)
            .into_iter()
            .flatten()
            .copied()
            .filter(move |&index| {
                !confined || self.confined(self.links[index].src, self.links[index].dst)
            })
    }

    pub fn reachable(&self, src: NetworkDeviceID, confined: bool) -> HashSet<NetworkDeviceID> {
        let mut seen = HashSet::from([src]);
        let mut pending = VecDeque::from([src]);
        while let Some(node) = pending.pop_front() {
            for index in self.links_at(&self.outgoing_links, node, confined) {
                if seen.insert(self.links[index].dst) {
                    pending.push_back(self.links[index].dst);
                }
            }
        }
        seen
    }

    /// Route `src -> dst` through `via`; checked when first used.
    pub fn set_waypoints(
        &mut self,
        src: NetworkDeviceID,
        dst: NetworkDeviceID,
        via: Vec<NetworkDeviceID>,
    ) {
        self.waypoints.insert((src, dst), via);
        self.routes.remove(&(src, dst));
    }

    fn find_equal_shortest_routes(
        &self,
        src: NetworkDeviceID,
        dst: NetworkDeviceID,
    ) -> Result<Vec<Route>> {
        let confined = self.confined(src, dst);
        let mut distance = HashMap::from([(dst, 0usize)]);
        let mut pending = VecDeque::from([dst]);
        while let Some(node) = pending.pop_front() {
            let next_distance = distance[&node] + 1;
            for index in self.links_at(&self.incoming_links, node, confined) {
                let link = &self.links[index];
                if let std::collections::hash_map::Entry::Vacant(entry) = distance.entry(link.src) {
                    entry.insert(next_distance);
                    pending.push_back(link.src);
                }
            }
        }
        if !distance.contains_key(&src) {
            bail!("no directed route from {src} to {dst}");
        }
        let mut stack = vec![(src, Vec::new(), 0.0)];
        let mut routes = Vec::new();
        while let Some((node, path, latency)) = stack.pop() {
            if node == dst {
                routes.push(Route {
                    link_indices: path,
                    total_latency: latency,
                });
                continue;
            }
            for index in self.links_at(&self.outgoing_links, node, confined).rev() {
                let link = &self.links[index];
                if distance
                    .get(&link.dst)
                    .is_some_and(|&d| d + 1 == distance[&node])
                {
                    let mut next = path.clone();
                    next.push(index);
                    stack.push((link.dst, next, latency + link.latency));
                }
            }
        }
        Ok(routes)
    }

    fn find_shortest_route(&self, src: NetworkDeviceID, dst: NetworkDeviceID) -> Result<Route> {
        let confined = self.confined(src, dst);
        let bandwidth = self.widest_bandwidth(src, dst, confined)?;
        let mut frontier = HashMap::new();
        frontier.insert(
            src,
            PathState {
                total_latency: 0.0,
                hops: 0,
                path_nodes: vec![src],
                link_indices: Vec::new(),
            },
        );
        let mut settled = HashSet::new();

        while let Some(current_node) = self.next_frontier_node(&frontier, &settled) {
            if current_node == dst {
                break;
            }
            settled.insert(current_node);

            let current_path = frontier
                .get(&current_node)
                .cloned()
                .ok_or_else(|| anyhow!("path frontier disappeared during route search"))?;
            for link_index in self.links_at(&self.outgoing_links, current_node, confined) {
                let link = &self.links[link_index];
                if link.bandwidth < bandwidth || settled.contains(&link.dst) {
                    continue;
                }
                let next_path = PathState {
                    total_latency: current_path.total_latency + link.latency,
                    hops: current_path.hops + 1,
                    path_nodes: current_path
                        .path_nodes
                        .iter()
                        .copied()
                        .chain(std::iter::once(link.dst))
                        .collect(),
                    link_indices: current_path
                        .link_indices
                        .iter()
                        .copied()
                        .chain(std::iter::once(link_index))
                        .collect(),
                };

                let should_update = frontier
                    .get(&link.dst)
                    .map(|current_best| {
                        Self::compare_paths(&next_path, current_best) == Ordering::Less
                    })
                    .unwrap_or(true);
                if should_update {
                    frontier.insert(link.dst, next_path);
                }
            }
        }

        let path = frontier
            .get(&dst)
            .ok_or_else(|| anyhow!("network links must be configured before transfers start"))?;
        Ok(Route {
            link_indices: path.link_indices.clone(),
            total_latency: path.total_latency,
        })
    }

    fn widest_bandwidth(
        &self,
        src: NetworkDeviceID,
        dst: NetworkDeviceID,
        confined: bool,
    ) -> Result<f64> {
        let mut widths = HashMap::from([(src, f64::INFINITY)]);
        let mut settled = HashSet::new();
        loop {
            let Some((&node, &width)) = widths
                .iter()
                .filter(|(node, _)| !settled.contains(*node))
                .max_by(|(a, aw), (b, bw)| aw.total_cmp(bw).then_with(|| b.cmp(a)))
            else {
                bail!("network links must be configured before transfers start");
            };
            if node == dst {
                return Ok(width);
            }
            settled.insert(node);
            for index in self.links_at(&self.outgoing_links, node, confined) {
                let link = &self.links[index];
                let candidate = width.min(link.bandwidth);
                widths
                    .entry(link.dst)
                    .and_modify(|best| *best = best.max(candidate))
                    .or_insert(candidate);
            }
        }
    }

    fn next_frontier_node(
        &self,
        frontier: &HashMap<NetworkDeviceID, PathState>,
        settled: &HashSet<NetworkDeviceID>,
    ) -> Option<NetworkDeviceID> {
        frontier
            .iter()
            .filter(|(node, _)| !settled.contains(node))
            .min_by(|(_, left), (_, right)| Self::compare_paths(left, right))
            .map(|(&node, _)| node)
    }

    fn compare_paths(left: &PathState, right: &PathState) -> Ordering {
        match Self::compare_f64(left.total_latency, right.total_latency) {
            Ordering::Equal => left
                .hops
                .cmp(&right.hops)
                .then_with(|| left.path_nodes.cmp(&right.path_nodes)),
            ordering => ordering,
        }
    }

    fn compare_f64(left: f64, right: f64) -> Ordering {
        if (left - right).abs() <= ROUTE_COMPARE_EPSILON {
            Ordering::Equal
        } else {
            left.total_cmp(&right)
        }
    }

    fn bytes_tolerance(total_bytes: f64) -> f64 {
        ABSOLUTE_BYTES_EPSILON.max(total_bytes.abs().max(1.0) * RELATIVE_BYTES_EPSILON)
    }

    fn time_tolerance(total_time: f64) -> f64 {
        ABSOLUTE_TIME_EPSILON.max(total_time.abs().max(1.0) * RELATIVE_TIME_EPSILON)
    }

    fn advances_time(now: f64, delta: f64) -> bool {
        delta.is_finite() && delta > 0.0 && now + delta > now
    }

    fn class_for(&mut self, route: Route, weight: f64) -> usize {
        let key = (route.link_indices.clone(), weight.to_bits());
        let next = self.classes.len();
        let class = *self.class_index.entry(key).or_insert(next);
        if class == next {
            self.classes.push(FlowClass {
                route,
                weight,
                rate: 0.0,
                served: 0.0,
                members: BinaryHeap::new(),
            });
        }
        class
    }

    fn register(
        &mut self,
        node: NodeID,
        src: NetworkDeviceID,
        dst: NetworkDeviceID,
        bytes: u64,
    ) -> Result<()> {
        let classes = match self.pair_classes.get(&(src, dst)) {
            Some(classes) if (classes.len() as u64) <= bytes => classes.clone(),
            _ => {
                let routes = self.route_for(src, dst)?;
                let count = routes.len().min(bytes as usize);
                let classes = routes
                    .into_iter()
                    .take(count)
                    .map(|route| self.class_for(route, 1.0 / count as f64))
                    .collect::<Vec<_>>();
                if count == self.routes[&(src, dst)].len() {
                    self.pair_classes.insert((src, dst), classes.clone());
                }
                classes
            }
        };
        let count = classes.len() as u64;
        let mut flows = Vec::with_capacity(classes.len());
        for (index, &class) in classes.iter().enumerate() {
            let payload = bytes / count + u64::from((index as u64) < bytes % count);
            let flow_id = self.next_flow_id;
            self.next_flow_id += 1;
            let entry = &mut self.classes[class];
            let due = entry.served + payload as f64;
            entry.members.push(Reverse((Due(due), flow_id)));
            for &link in &entry.route.link_indices {
                let link = &mut self.links[link];
                link.flows += 1;
                link.peak_flows = link.peak_flows.max(link.flows);
            }
            self.active.insert(class);
            self.flows.insert(
                flow_id,
                Flow {
                    class,
                    total_bytes: payload as f64,
                    node,
                    due,
                    propagating: false,
                },
            );
            flows.push(flow_id);
        }
        self.rates_valid = false;
        self.settled = None;
        let remaining = flows.len();
        self.transfers.insert(
            node,
            Transfer {
                flows,
                remaining,
                wake: self.time,
            },
        );
        Ok(())
    }

    /// Weighted max-min fair rates by progressive filling: links that saturate
    /// first fix their classes; capacity a class cannot use elsewhere stays with
    /// the others, as in work-conserving link arbitration. Members of a class
    /// share every link and so always get equal rates. Rates depend only on
    /// the active flow set, so they are kept until it changes.
    fn update_rates(&mut self) {
        if self.rates_valid {
            return;
        }
        self.rates_valid = true;
        let RateScratch {
            link_of,
            active_links,
            capacity,
            demand,
            unfixed,
            members,
            fixed,
            saturated,
        } = &mut self.scratch;
        link_of.resize(self.links.len(), usize::MAX);
        for &link in active_links.iter() {
            link_of[link] = usize::MAX;
        }
        active_links.clear();
        capacity.clear();
        demand.clear();
        unfixed.clear();
        members.iter_mut().for_each(Vec::clear);
        fixed.clear();
        let active = self.active.iter().copied().collect::<Vec<_>>();
        for (position, &class) in active.iter().enumerate() {
            let class = &mut self.classes[class];
            class.rate = 0.0;
            fixed.push(false);
            let count = class.members.len() as f64;
            for &link in &class.route.link_indices {
                if link_of[link] == usize::MAX {
                    link_of[link] = active_links.len();
                    active_links.push(link);
                    capacity.push(self.links[link].bandwidth);
                    demand.push(0.0);
                    unfixed.push(0);
                    if members.len() < active_links.len() {
                        members.push(Vec::new());
                    }
                }
                let k = link_of[link];
                demand[k] += count * class.weight;
                unfixed[k] += 1;
                members[k].push(position);
            }
        }
        let mut remaining = active.len();
        while remaining > 0 {
            let share = |l: usize| capacity[l].max(0.0) / demand[l];
            let level = (0..capacity.len())
                .filter(|&l| unfixed[l] > 0)
                .map(share)
                .fold(f64::INFINITY, f64::min);
            saturated.clear();
            saturated.extend(
                (0..capacity.len()).filter(|&l| {
                    unfixed[l] > 0 && share(l) <= level * (1.0 + RELATIVE_BYTES_EPSILON)
                }),
            );
            for &l in saturated.iter() {
                for &position in &members[l] {
                    if fixed[position] {
                        continue;
                    }
                    fixed[position] = true;
                    remaining -= 1;
                    let class = &mut self.classes[active[position]];
                    class.rate = class.weight * level;
                    let count = class.members.len() as f64;
                    for &link in &class.route.link_indices {
                        let k = link_of[link];
                        capacity[k] -= count * class.rate;
                        demand[k] -= count * class.weight;
                        unfixed[k] -= 1;
                    }
                }
            }
        }
    }

    /// Ends the transmissions and propagations due at `time`, including ones
    /// whose remainder is within tolerance or cannot advance the clock.
    fn finish_due(&mut self) -> bool {
        let now = self.time;
        let mut changed = false;
        for class in self.active.iter().copied().collect::<Vec<_>>() {
            loop {
                let entry = &self.classes[class];
                let Some(&Reverse((Due(due), flow))) = entry.members.peek() else {
                    break;
                };
                let remaining = due - entry.served;
                let total = self.flows[&flow].total_bytes;
                if remaining > Self::bytes_tolerance(total)
                    && (entry.rate <= 0.0 || Self::advances_time(now, remaining / entry.rate))
                {
                    break;
                }
                self.classes[class].members.pop();
                self.end_transmission(flow);
                changed = true;
            }
            let entry = &mut self.classes[class];
            if entry.members.is_empty() {
                entry.served = 0.0;
                self.active.remove(&class);
            }
        }
        while let Some(&Reverse((Due(arrival), flow))) = self.propagation.peek() {
            let latency = self.classes[self.flows[&flow].class].route.total_latency;
            let remaining = arrival - now;
            if remaining > Self::time_tolerance(latency) && Self::advances_time(now, remaining) {
                break;
            }
            self.propagation.pop();
            self.arrive(flow);
            changed = true;
        }
        changed
    }

    fn end_transmission(&mut self, flow: FlowID) {
        let class = self.flows[&flow].class;
        for &link in &self.classes[class].route.link_indices {
            self.links[link].flows -= 1;
        }
        self.rates_valid = false;
        self.settled = None;
        let latency = self.classes[class].route.total_latency;
        if latency <= Self::time_tolerance(latency) || !Self::advances_time(self.time, latency) {
            self.arrive(flow);
            return;
        }
        let arrival = self.time + latency;
        let state = self.flows.get_mut(&flow).unwrap();
        state.due = arrival;
        state.propagating = true;
        self.propagation.push(Reverse((Due(arrival), flow)));
    }

    fn arrive(&mut self, flow: FlowID) {
        let node = self.flows.remove(&flow).unwrap().node;
        let transfer = self.transfers.get_mut(&node).unwrap();
        transfer.remaining -= 1;
        if transfer.remaining == 0 {
            self.arrived.push(node);
        }
    }

    /// Delay to a flow's next event: transmission ending, which changes rates,
    /// or its last bit arriving.
    fn flow_event(&self, flow: &Flow) -> Option<f64> {
        if flow.propagating {
            return Some(flow.due - self.time);
        }
        let class = &self.classes[flow.class];
        (class.rate > 0.0).then(|| (flow.due - class.served) / class.rate)
    }

    /// The next flow event and its flow.
    fn next_event(&self) -> Option<(f64, FlowID)> {
        let transmissions = self.active.iter().filter_map(|&class| {
            let class = &self.classes[class];
            let &Reverse((Due(due), flow)) = class.members.peek()?;
            (class.rate > 0.0).then(|| ((due - class.served) / class.rate, flow))
        });
        let arrival = self
            .propagation
            .peek()
            .map(|&Reverse((Due(arrival), flow))| (arrival - self.time, flow));
        transmissions
            .chain(arrival)
            .min_by(|a, b| a.0.total_cmp(&b.0))
    }

    fn schedule_wake(&mut self, node: NodeID, wake: f64) {
        let transfer = self.transfers.get_mut(&node).unwrap();
        if wake < transfer.wake {
            transfer.wake = wake;
            self.wakeups.push((node, wake));
        }
    }

    /// Reconcile flow state at `time` and return the delay to the next flow
    /// event. Only transfers that arrived and the owner of the next event are
    /// woken: every event time then gets a poll, which advances all flows.
    fn settle(&mut self) -> Option<f64> {
        if let Some(next) = self.settled {
            return next;
        }
        loop {
            self.update_rates();
            if !self.finish_due() {
                break;
            }
        }
        let next = self.next_event();
        self.settled = Some(next.map(|(delay, _)| delay));
        let floor = self.poll_time.max(self.time);
        for node in std::mem::take(&mut self.arrived) {
            self.schedule_wake(node, floor);
        }
        if let Some((delay, flow)) = next {
            let node = self.flows[&flow].node;
            self.schedule_wake(node, (self.time + delay).max(floor));
        }
        next.map(|(delay, _)| delay)
    }

    fn advance(&mut self, until_s: f64) {
        while self.time < until_s {
            let Some(next_event_delay) = self.settle() else {
                self.time = until_s;
                return;
            };
            let step_s = (until_s - self.time).min(next_event_delay);
            for &link in &self.scratch.active_links {
                self.links[link].busy_seconds += step_s;
            }
            for &class in &self.active {
                let class = &mut self.classes[class];
                let progress = class.rate * step_s;
                class.served += progress;
                let bytes = progress * class.members.len() as f64;
                for &link in &class.route.link_indices {
                    self.links[link].bytes += bytes;
                }
            }
            self.time += step_s;
            self.settled = None;
            self.settle();

            if step_s + f64::EPSILON < next_event_delay {
                break;
            }
        }
    }
}

impl NetworkSimulator for FlowNetworkSimulator {
    fn add_device(&mut self) -> NetworkDeviceID {
        let device_id = self.next_device_id;
        self.next_device_id += 1;
        device_id
    }

    fn add_link(
        &mut self,
        src: NetworkDeviceID,
        dst: NetworkDeviceID,
        bandwidth: f64,
        latency: f64,
    ) -> Result<()> {
        self.add_named_link(
            src,
            dst,
            bandwidth,
            latency,
            format!("{}", self.links.len()),
        )?;
        Ok(())
    }

    fn simulate(
        &mut self,
        node: NodeID,
        src: NetworkDeviceID,
        dst: NetworkDeviceID,
        bytes: u64,
        now: f64,
        state: &mut Option<Box<dyn Any>>,
    ) -> Result<PollResult> {
        if !now.is_finite() || now < self.time {
            bail!("network time must be finite and nondecreasing");
        }
        self.poll_time = now;
        self.advance(now);
        if state.is_none() {
            if bytes == 0 || src == dst {
                return Ok(PollResult::Complete { latency: 0.0 });
            }
            self.register(node, src, dst, bytes)?;
            *state = Some(Box::new(()));
            // Transfers starting together share one rate computation.
            return Ok(PollResult::Deferred);
        }
        self.settle();
        let transfer = self
            .transfers
            .get(&node)
            .ok_or_else(|| anyhow!("network node has no flow transfer"))?;
        if transfer.remaining == 0 {
            self.transfers.remove(&node);
            *state = None;
            return Ok(PollResult::Complete { latency: 0.0 });
        }
        // A transfer answers its own next flow event; rate changes bring the
        // next event's owner forward through `take_wakeups`.
        let delay = transfer
            .flows
            .iter()
            .filter_map(|flow| self.flows.get(flow))
            .filter_map(|flow| self.flow_event(flow))
            .reduce(f64::min)
            .ok_or_else(|| anyhow!("network flow has no active route bandwidth"))?;
        self.transfers.get_mut(&node).unwrap().wake = now + delay;
        Ok(PollResult::Pending { latency: delay })
    }

    fn take_wakeups(&mut self) -> Vec<(NodeID, f64)> {
        std::mem::take(&mut self.wakeups)
    }
}

impl FlowNetworkSimulator {
    pub fn add_named_link(
        &mut self,
        src: NetworkDeviceID,
        dst: NetworkDeviceID,
        bandwidth: f64,
        latency: f64,
        name: String,
    ) -> Result<usize> {
        if !bandwidth.is_finite() || bandwidth <= 0.0 {
            bail!("network link bandwidth must be finite and positive");
        }
        if !latency.is_finite() || latency < 0.0 {
            bail!("network link latency must be finite and non-negative");
        }
        self.rates_valid = false;
        self.settled = None;
        let link_index = self.links.len();
        self.links.push(DirectionalLink {
            src,
            dst,
            name,
            bandwidth,
            latency,
            flows: 0,
            bytes: 0.0,
            busy_seconds: 0.0,
            peak_flows: 0,
        });
        self.outgoing_links.entry(src).or_default().push(link_index);
        self.incoming_links.entry(dst).or_default().push(link_index);
        self.routes.clear();
        self.pair_classes.clear();
        Ok(link_index)
    }
}
