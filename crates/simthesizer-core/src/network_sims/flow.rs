use std::any::Any;
use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

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
    active_flows: BTreeMap<FlowID, *mut FlowTransferState>,
    bytes: f64,
    busy_seconds: f64,
    peak_flows: usize,
}

#[derive(Debug, Clone, PartialEq)]
struct Route {
    link_indices: Vec<usize>,
    total_latency: f64,
}

#[derive(Debug)]
struct FlowTransferState {
    flow_id: FlowID,
    route: Route,
    total_bytes: f64,
    remaining_bytes: f64,
    propagation_remaining: f64,
    weight: f64,
    /// Max-min fair rate; valid while `rates_valid`.
    rate: f64,
    /// Dense index while rates are computed.
    slot: usize,
}

/// Buffers reused across rate computations.
#[derive(Debug, Default)]
struct RateScratch {
    /// Directional link index -> active link index.
    link_of: Vec<usize>,
    capacity: Vec<f64>,
    demand: Vec<f64>,
    unfixed: Vec<usize>,
    /// Flow slots per active link, delimited by `starts`.
    members: Vec<usize>,
    starts: Vec<usize>,
    flows: Vec<*mut FlowTransferState>,
    fixed: Vec<bool>,
    saturated: Vec<usize>,
}

#[derive(Debug)]
struct TransferState {
    flows: Vec<Box<FlowTransferState>>,
    node: NodeID,
    /// When the engine polls this transfer next.
    wake: f64,
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
    flow_states: HashMap<FlowID, *mut FlowTransferState>,

    /// Flow rates depend only on the active flow set.
    #[serde(skip_deserializing, default)]
    rates_valid: bool,

    /// Rates changed since pending transfers were last checked for wake-ups.
    #[serde(skip_deserializing, default)]
    rates_changed: bool,

    /// Pending transfers by engine node.
    #[serde(skip_deserializing, default)]
    transfers: HashMap<NodeID, *mut TransferState>,

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
                    "active_flows": link.active_flows.len(), "peak_flows": link.peak_flows,
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
            "active_subflows": self.flow_states.len()})
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

    fn snap_bytes_to_zero(remaining_bytes: f64, total_bytes: f64) -> f64 {
        if remaining_bytes <= Self::bytes_tolerance(total_bytes) {
            0.0
        } else {
            remaining_bytes
        }
    }

    fn snap_time_to_zero(remaining_time: f64, total_time: f64) -> f64 {
        if remaining_time <= Self::time_tolerance(total_time) {
            0.0
        } else {
            remaining_time
        }
    }

    fn advances_time(now: f64, delta: f64) -> bool {
        delta.is_finite() && delta > 0.0 && now + delta > now
    }

    fn flows(&self) -> impl Iterator<Item = &FlowTransferState> + '_ {
        // NOTE: Safety: flow states are boxed in network node state and
        // unregistered before their node drops them.
        self.flow_states.values().map(|&flow| unsafe { &*flow })
    }

    fn flows_mut(&mut self) -> impl Iterator<Item = &mut FlowTransferState> + '_ {
        self.flow_states.values().map(|&flow| unsafe { &mut *flow })
    }

    fn downcast_state_mut(state: &mut Option<Box<dyn Any>>) -> Result<Option<&mut TransferState>> {
        match state.as_deref_mut() {
            Some(state) => state
                .downcast_mut::<TransferState>()
                .map(Some)
                .ok_or_else(|| anyhow!("network node has unexpected simulator state")),
            None => Ok(None),
        }
    }

    fn register(
        &mut self,
        node: NodeID,
        src: NetworkDeviceID,
        dst: NetworkDeviceID,
        bytes: u64,
        state: &mut Option<Box<dyn Any>>,
    ) -> Result<()> {
        let routes = self.route_for(src, dst)?;
        let count = routes.len().min(bytes as usize);
        let mut transfer = Box::new(TransferState {
            flows: Vec::with_capacity(count),
            node,
            wake: self.time,
        });
        for (index, route) in routes.into_iter().take(count).enumerate() {
            let payload = bytes / count as u64 + u64::from((index as u64) < bytes % count as u64);
            let flow_id = self.next_flow_id;
            self.next_flow_id += 1;
            let mut flow = Box::new(FlowTransferState {
                flow_id,
                route,
                total_bytes: payload as f64,
                remaining_bytes: payload as f64,
                propagation_remaining: 0.0,
                weight: 1.0 / count as f64,
                rate: 0.0,
                slot: 0,
            });
            let flow_ptr = (&mut *flow) as *mut FlowTransferState;
            for &link_index in &flow.route.link_indices {
                let link = &mut self.links[link_index];
                link.active_flows.insert(flow_id, flow_ptr);
                link.peak_flows = link.peak_flows.max(link.active_flows.len());
            }
            self.flow_states.insert(flow_id, flow_ptr);
            transfer.flows.push(flow);
        }
        self.rates_valid = false;
        self.settled = None;
        self.transfers.insert(node, &mut *transfer);
        *state = Some(transfer);
        Ok(())
    }

    /// Weighted max-min fair rates by progressive filling: links that saturate
    /// first fix their flows; capacity a flow cannot use elsewhere stays with
    /// the others, as in work-conserving link arbitration. Rates depend only on
    /// the active flow set, so they are kept until it changes.
    fn update_rates(&mut self) {
        if self.rates_valid {
            return;
        }
        self.rates_valid = true;
        self.rates_changed = true;
        for flow in self.flows_mut() {
            flow.rate = 0.0;
            flow.slot = usize::MAX;
        }
        let RateScratch {
            link_of,
            capacity,
            demand,
            unfixed,
            members,
            starts,
            flows,
            fixed,
            saturated,
        } = &mut self.scratch;
        link_of.resize(self.links.len(), usize::MAX);
        capacity.clear();
        demand.clear();
        unfixed.clear();
        members.clear();
        starts.clear();
        flows.clear();
        fixed.clear();
        for (index, link) in self.links.iter().enumerate() {
            if link.active_flows.is_empty() {
                continue;
            }
            link_of[index] = capacity.len();
            capacity.push(link.bandwidth);
            starts.push(members.len());
            let mut weight = 0.0;
            for &flow in link.active_flows.values() {
                let flow = unsafe { &mut *flow };
                if flow.slot == usize::MAX {
                    flow.slot = flows.len();
                    flows.push(flow);
                    fixed.push(false);
                }
                weight += flow.weight;
                members.push(flow.slot);
            }
            demand.push(weight);
            unfixed.push(link.active_flows.len());
        }
        starts.push(members.len());
        let mut remaining = flows.len();
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
                for &f in &members[starts[l]..starts[l + 1]] {
                    if fixed[f] {
                        continue;
                    }
                    fixed[f] = true;
                    remaining -= 1;
                    let flow = unsafe { &mut *flows[f] };
                    flow.rate = flow.weight * level;
                    for &link in &flow.route.link_indices {
                        let k = link_of[link];
                        capacity[k] -= flow.rate;
                        demand[k] -= flow.weight;
                        unfixed[k] -= 1;
                    }
                }
            }
        }
    }

    fn snap_small_residuals(&mut self) -> bool {
        let mut changed = false;
        for flow in self.flows_mut() {
            let bytes = Self::snap_bytes_to_zero(flow.remaining_bytes, flow.total_bytes);
            let time =
                Self::snap_time_to_zero(flow.propagation_remaining, flow.route.total_latency);
            changed |= bytes != flow.remaining_bytes || time != flow.propagation_remaining;
            flow.remaining_bytes = bytes;
            flow.propagation_remaining = time;
        }
        changed
    }

    fn snap_unobservable_progress(&mut self) -> bool {
        self.update_rates();
        let now = self.time;
        let mut changed = false;
        for flow in self.flows_mut() {
            if flow.remaining_bytes > 0.0 {
                if flow.rate > 0.0 && !Self::advances_time(now, flow.remaining_bytes / flow.rate) {
                    flow.remaining_bytes = 0.0;
                    changed = true;
                }
            } else if flow.propagation_remaining > 0.0
                && !Self::advances_time(now, flow.propagation_remaining)
            {
                flow.propagation_remaining = 0.0;
                changed = true;
            }
        }
        changed
    }

    fn reconcile_flow_states(&mut self) {
        loop {
            let mut changed = false;
            changed |= self.snap_small_residuals();
            changed |= self.snap_unobservable_progress();
            changed |= self.begin_propagation();
            changed |= self.snap_small_residuals();
            changed |= self.snap_unobservable_progress();
            if !changed {
                break;
            }
        }
    }

    /// Reconcile flow state at `time` and return the delay to the next flow event.
    fn settle(&mut self) -> Option<f64> {
        if let Some(next) = self.settled {
            return next;
        }
        self.reconcile_flow_states();
        self.update_rates();
        let next = self.flows().filter_map(Self::flow_event).reduce(f64::min);
        self.settled = Some(next);
        if std::mem::take(&mut self.rates_changed) {
            // A rate change can only advance a transfer's next event here; a
            // delayed one is polled early and answers again.
            let floor = self.poll_time.max(self.time);
            for (&node, &transfer) in &self.transfers {
                let transfer = unsafe { &mut *transfer };
                if let Some(delay) = Self::transfer_event(transfer) {
                    let wake = (self.time + delay).max(floor);
                    if wake < transfer.wake {
                        transfer.wake = wake;
                        self.wakeups.push((node, wake));
                    }
                }
            }
        }
        next
    }

    /// Delay to a flow's next event: transmission ending, which changes rates,
    /// or its last bit arriving.
    fn flow_event(flow: &FlowTransferState) -> Option<f64> {
        if flow.remaining_bytes > 0.0 {
            (flow.rate > 0.0).then(|| flow.remaining_bytes / flow.rate)
        } else {
            (flow.propagation_remaining > 0.0).then_some(flow.propagation_remaining)
        }
    }

    fn transfer_event(transfer: &TransferState) -> Option<f64> {
        transfer
            .flows
            .iter()
            .filter_map(|flow| Self::flow_event(flow))
            .reduce(f64::min)
    }

    fn release_flow_links(&mut self, flow_id: FlowID, route: &Route) {
        for &link_index in &route.link_indices {
            if self.links[link_index]
                .active_flows
                .remove(&flow_id)
                .is_some()
            {
                self.rates_valid = false;
                self.settled = None;
            }
        }
    }

    fn begin_propagation(&mut self) -> bool {
        let finished = self
            .flows()
            .filter(|flow| {
                flow.remaining_bytes == 0.0
                    && flow
                        .route
                        .link_indices
                        .iter()
                        .any(|&link| self.links[link].active_flows.contains_key(&flow.flow_id))
            })
            .map(|flow| (flow.flow_id, flow.route.clone()))
            .collect::<Vec<_>>();
        let changed = !finished.is_empty();
        for (flow_id, route) in finished {
            self.release_flow_links(flow_id, &route);
            unsafe { (*self.flow_states[&flow_id]).propagation_remaining = route.total_latency };
        }
        changed
    }

    fn advance(&mut self, until_s: f64) {
        while self.time < until_s {
            let Some(next_event_delay) = self.settle() else {
                self.time = until_s;
                return;
            };
            let step_s = (until_s - self.time).min(next_event_delay);
            for link in &mut self.links {
                if !link.active_flows.is_empty() {
                    link.busy_seconds += step_s;
                    link.bytes += link
                        .active_flows
                        .values()
                        .map(|&flow| unsafe { (*flow).rate } * step_s)
                        .sum::<f64>();
                }
            }
            for flow in self.flows_mut() {
                if flow.remaining_bytes > 0.0 {
                    flow.remaining_bytes = (flow.remaining_bytes - flow.rate * step_s).max(0.0);
                } else if flow.propagation_remaining > 0.0 {
                    flow.propagation_remaining = (flow.propagation_remaining - step_s).max(0.0);
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
            self.register(node, src, dst, bytes, state)?;
            // Transfers starting together share one rate computation.
            return Ok(PollResult::Deferred);
        }
        self.settle();
        let transfer = Self::downcast_state_mut(state)?.unwrap();
        if transfer
            .flows
            .iter()
            .all(|flow| flow.remaining_bytes == 0.0 && flow.propagation_remaining == 0.0)
        {
            for flow in &transfer.flows {
                self.release_flow_links(flow.flow_id, &flow.route);
                self.flow_states.remove(&flow.flow_id);
            }
            self.transfers.remove(&transfer.node);
            *state = None;
            return Ok(PollResult::Complete { latency: 0.0 });
        }
        // Each transfer is polled at its own next flow event; rate changes
        // bring other transfers forward through `take_wakeups`.
        let delay = Self::transfer_event(transfer)
            .ok_or_else(|| anyhow!("network flow has no active route bandwidth"))?;
        transfer.wake = now + delay;
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
            active_flows: BTreeMap::new(),
            bytes: 0.0,
            busy_seconds: 0.0,
            peak_flows: 0,
        });
        self.outgoing_links.entry(src).or_default().push(link_index);
        self.incoming_links.entry(dst).or_default().push(link_index);
        self.routes.clear();
        Ok(link_index)
    }
}
