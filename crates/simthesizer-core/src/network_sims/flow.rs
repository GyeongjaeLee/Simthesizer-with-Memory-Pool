use std::any::Any;
use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap, HashSet, VecDeque};

use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};

use crate::module::Module;
use crate::network_sim::NetworkSimulator;
use crate::types::{NetworkDeviceID, PollResult};

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
    active_flows: BTreeSet<FlowID>,
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
}

#[derive(Debug)]
struct TransferState {
    flows: Vec<Box<FlowTransferState>>,
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

    #[serde(skip_deserializing, default)]
    bound_routes:
        HashMap<(NetworkDeviceID, NetworkDeviceID), (NetworkDeviceID, Vec<NetworkDeviceID>)>,

    #[serde(skip_deserializing, default)]
    route_handles:
        HashMap<(NetworkDeviceID, NetworkDeviceID, Vec<NetworkDeviceID>), NetworkDeviceID>,

    #[serde(skip_deserializing, default)]
    flow_states: HashMap<FlowID, *mut FlowTransferState>,
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

    pub fn check_waypoints(
        &self,
        src: NetworkDeviceID,
        dst: NetworkDeviceID,
        via: &[NetworkDeviceID],
    ) -> Result<()> {
        self.routes_via(src, dst, via).map(|_| ())
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

    pub fn check_route(&self, src: NetworkDeviceID, dst: NetworkDeviceID) -> Result<()> {
        let mut seen = HashSet::from([src]);
        let mut pending = VecDeque::from([src]);
        while let Some(node) = pending.pop_front() {
            if node == dst {
                return Ok(());
            }
            for &index in self.outgoing_links.get(&node).into_iter().flatten() {
                if seen.insert(self.links[index].dst) {
                    pending.push_back(self.links[index].dst);
                }
            }
        }
        bail!("no directed route from {src} to {dst}")
    }

    pub fn set_waypoints(
        &mut self,
        src: NetworkDeviceID,
        dst: NetworkDeviceID,
        via: Vec<NetworkDeviceID>,
    ) -> Result<()> {
        let stops = std::iter::once(src)
            .chain(via.iter().copied())
            .chain(std::iter::once(dst))
            .collect::<Vec<_>>();
        for segment in stops.windows(2) {
            self.check_route(segment[0], segment[1])?;
        }
        self.waypoints.insert((src, dst), via);
        self.routes.remove(&(src, dst));
        Ok(())
    }

    fn find_equal_shortest_routes(
        &self,
        src: NetworkDeviceID,
        dst: NetworkDeviceID,
    ) -> Result<Vec<Route>> {
        let mut distance = HashMap::from([(dst, 0usize)]);
        let mut pending = VecDeque::from([dst]);
        while let Some(node) = pending.pop_front() {
            let next_distance = distance[&node] + 1;
            for &index in self.incoming_links.get(&node).into_iter().flatten() {
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
            for &index in self.outgoing_links.get(&node).into_iter().flatten().rev() {
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
        let bandwidth = self.widest_bandwidth(src, dst)?;
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
            let Some(outgoing) = self.outgoing_links.get(&current_node) else {
                continue;
            };

            for &link_index in outgoing {
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

    fn widest_bandwidth(&self, src: NetworkDeviceID, dst: NetworkDeviceID) -> Result<f64> {
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
            for &index in self.outgoing_links.get(&node).into_iter().flatten() {
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

    fn flow_ids(&self) -> Vec<FlowID> {
        self.flow_states.keys().copied().collect()
    }

    fn flow_ptr(&self, flow_id: FlowID) -> Result<*mut FlowTransferState> {
        self.flow_states
            .get(&flow_id)
            .copied()
            .ok_or_else(|| anyhow!("network flow {flow_id} is not registered"))
    }

    fn with_flow<R>(&self, flow_id: FlowID, f: impl FnOnce(&FlowTransferState) -> R) -> Result<R> {
        let ptr = self.flow_ptr(flow_id)?;
        Ok(unsafe { f(&*ptr) })
    }

    fn with_flow_mut<R>(
        &mut self,
        flow_id: FlowID,
        f: impl FnOnce(&mut FlowTransferState) -> R,
    ) -> Result<R> {
        let ptr = self.flow_ptr(flow_id)?;
        Ok(unsafe { f(&mut *ptr) })
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

    fn ensure_flow_state(
        &mut self,
        src: NetworkDeviceID,
        dst: NetworkDeviceID,
        bytes: u64,
        state: &mut Option<Box<dyn Any>>,
    ) -> Result<Vec<FlowID>> {
        if let Some(transfer) = Self::downcast_state_mut(state)? {
            return Ok(transfer.flows.iter().map(|flow| flow.flow_id).collect());
        }

        if bytes == 0 || src == dst {
            return Ok(Vec::new());
        }

        let routes = self.route_for(src, dst)?;
        let count = routes.len().min(bytes as usize);
        let mut transfer = TransferState {
            flows: Vec::with_capacity(count),
        };
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
            });
            let flow_ptr = (&mut *flow) as *mut FlowTransferState;
            for &link_index in &flow.route.link_indices {
                let link = &mut self.links[link_index];
                link.active_flows.insert(flow_id);
                link.peak_flows = link.peak_flows.max(link.active_flows.len());
            }
            self.flow_states.insert(flow_id, flow_ptr);
            transfer.flows.push(flow);
        }
        let ids = transfer.flows.iter().map(|flow| flow.flow_id).collect();
        *state = Some(Box::new(transfer));
        Ok(ids)
    }

    /// Weighted max-min fair rates by progressive filling: links that saturate
    /// first fix their flows; capacity a flow cannot use elsewhere stays with
    /// the others, as in work-conserving link arbitration.
    fn active_rates(&self) -> HashMap<FlowID, f64> {
        let mut capacity = HashMap::new();
        let mut flows = HashMap::<FlowID, (f64, Vec<usize>)>::new();
        for (index, link) in self.links.iter().enumerate() {
            if link.active_flows.is_empty() {
                continue;
            }
            capacity.insert(index, link.bandwidth);
            for &flow_id in &link.active_flows {
                let weight = unsafe { (*self.flow_states[&flow_id]).weight };
                flows
                    .entry(flow_id)
                    .or_insert_with(|| (weight, Vec::new()))
                    .1
                    .push(index);
            }
        }
        let mut rates = HashMap::with_capacity(flows.len());
        while !flows.is_empty() {
            let mut demand: HashMap<usize, f64> = HashMap::new();
            for (weight, links) in flows.values() {
                for &link in links {
                    *demand.entry(link).or_default() += weight;
                }
            }
            let level = demand
                .iter()
                .map(|(link, weight)| capacity[link].max(0.0) / weight)
                .fold(f64::INFINITY, f64::min);
            let saturated: HashSet<usize> = demand
                .iter()
                .filter(|(link, weight)| {
                    capacity[*link].max(0.0) / *weight <= level * (1.0 + RELATIVE_BYTES_EPSILON)
                })
                .map(|(&link, _)| link)
                .collect();
            let mut fixed: Vec<FlowID> = flows
                .iter()
                .filter(|(_, (_, links))| links.iter().any(|link| saturated.contains(link)))
                .map(|(&flow_id, _)| flow_id)
                .collect();
            fixed.sort_unstable();
            for flow_id in fixed {
                let (weight, links) = flows.remove(&flow_id).expect("fixed flow is active");
                let rate = weight * level;
                for link in links {
                    *capacity.get_mut(&link).expect("active link") -= rate;
                }
                rates.insert(flow_id, rate);
            }
        }
        rates
    }

    fn snap_small_residuals(&mut self) -> Result<bool> {
        let mut changed = false;
        for flow_id in self.flow_ids() {
            self.with_flow_mut(flow_id, |flow| {
                let snapped_bytes =
                    Self::snap_bytes_to_zero(flow.remaining_bytes, flow.total_bytes);
                if snapped_bytes != flow.remaining_bytes {
                    flow.remaining_bytes = snapped_bytes;
                    changed = true;
                }

                let snapped_time =
                    Self::snap_time_to_zero(flow.propagation_remaining, flow.route.total_latency);
                if snapped_time != flow.propagation_remaining {
                    flow.propagation_remaining = snapped_time;
                    changed = true;
                }
            })?;
        }
        Ok(changed)
    }

    fn snap_unobservable_progress(&mut self) -> Result<bool> {
        let rates = self.active_rates();
        let mut data_done = Vec::new();
        let mut prop_done = Vec::new();

        for flow_id in self.flow_ids() {
            self.with_flow(flow_id, |flow| {
                if flow.remaining_bytes > 0.0 {
                    let rate = rates.get(&flow_id).copied().unwrap_or(0.0);
                    if rate > 0.0 {
                        let delay = flow.remaining_bytes / rate;
                        if !Self::advances_time(self.time, delay) {
                            data_done.push(flow_id);
                        }
                    }
                } else if flow.propagation_remaining > 0.0
                    && !Self::advances_time(self.time, flow.propagation_remaining)
                {
                    prop_done.push(flow_id);
                }
            })?;
        }

        let changed = !data_done.is_empty() || !prop_done.is_empty();
        for flow_id in data_done {
            self.with_flow_mut(flow_id, |flow| {
                flow.remaining_bytes = 0.0;
            })?;
        }
        for flow_id in prop_done {
            self.with_flow_mut(flow_id, |flow| {
                flow.propagation_remaining = 0.0;
            })?;
        }
        Ok(changed)
    }

    fn reconcile_flow_states(&mut self) -> Result<()> {
        loop {
            let mut changed = false;
            changed |= self.snap_small_residuals()?;
            changed |= self.snap_unobservable_progress()?;
            changed |= self.begin_propagation()?;
            changed |= self.snap_small_residuals()?;
            changed |= self.snap_unobservable_progress()?;
            if !changed {
                break;
            }
        }
        Ok(())
    }

    fn next_event_delay(&self) -> Result<Option<f64>> {
        let rates = self.active_rates();
        let mut next_delay: Option<f64> = None;
        for flow_id in self.flow_ids() {
            self.with_flow(flow_id, |flow| {
                if flow.remaining_bytes > 0.0 {
                    let rate = *rates.get(&flow_id).unwrap_or(&0.0);
                    if rate > 0.0 {
                        let delay = flow.remaining_bytes / rate;
                        next_delay = Some(next_delay.map_or(delay, |current| current.min(delay)));
                    }
                } else if flow.propagation_remaining > 0.0 {
                    next_delay = Some(next_delay.map_or(flow.propagation_remaining, |current| {
                        current.min(flow.propagation_remaining)
                    }));
                }
            })?;
        }
        Ok(next_delay)
    }

    fn release_flow_links(&mut self, flow_id: FlowID, route: &Route) {
        for &link_index in &route.link_indices {
            self.links[link_index].active_flows.remove(&flow_id);
        }
    }

    fn begin_propagation(&mut self) -> Result<bool> {
        let mut finished_tx = Vec::new();
        for flow_id in self.flow_ids() {
            let still_active = self.with_flow(flow_id, |flow| {
                flow.route
                    .link_indices
                    .iter()
                    .any(|&link_index| self.links[link_index].active_flows.contains(&flow_id))
                    && flow.remaining_bytes == 0.0
            })?;
            if still_active {
                finished_tx.push(flow_id);
            }
        }

        let changed = !finished_tx.is_empty();
        for flow_id in finished_tx {
            let route = self.with_flow(flow_id, |flow| flow.route.clone())?;
            self.release_flow_links(flow_id, &route);
            self.with_flow_mut(flow_id, |flow| {
                flow.propagation_remaining = route.total_latency;
            })?;
        }
        Ok(changed)
    }

    fn advance(&mut self, until_s: f64) -> Result<()> {
        while self.time < until_s {
            self.reconcile_flow_states()?;
            let Some(next_event_delay) = self.next_event_delay()? else {
                self.time = until_s;
                return Ok(());
            };
            let step_s = (until_s - self.time).min(next_event_delay);
            let rates = self.active_rates();
            for link in &mut self.links {
                if !link.active_flows.is_empty() {
                    link.busy_seconds += step_s;
                    link.bytes += link
                        .active_flows
                        .iter()
                        .map(|id| rates[id] * step_s)
                        .sum::<f64>();
                }
            }
            for flow_id in self.flow_ids() {
                self.with_flow_mut(flow_id, |flow| {
                    if flow.remaining_bytes > 0.0 {
                        let rate = *rates.get(&flow_id).unwrap_or(&0.0);
                        flow.remaining_bytes = (flow.remaining_bytes - rate * step_s).max(0.0);
                    } else if flow.propagation_remaining > 0.0 {
                        flow.propagation_remaining = (flow.propagation_remaining - step_s).max(0.0);
                    }
                })?;
            }
            self.time += step_s;
            self.reconcile_flow_states()?;

            if step_s + f64::EPSILON < next_event_delay {
                break;
            }
        }
        Ok(())
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
        src: NetworkDeviceID,
        dst: NetworkDeviceID,
        bytes: u64,
        now: f64,
        state: &mut Option<Box<dyn Any>>,
    ) -> Result<PollResult> {
        if !now.is_finite() || now < self.time {
            bail!("network time must be finite and nondecreasing");
        }
        self.advance(now)?;
        self.reconcile_flow_states()?;

        let flow_ids = self.ensure_flow_state(src, dst, bytes, state)?;
        if flow_ids.is_empty() {
            return Ok(PollResult::Complete { latency: 0.0 });
        }

        let mut is_complete = true;
        for &flow_id in &flow_ids {
            is_complete &= self.with_flow(flow_id, |flow| {
                flow.remaining_bytes == 0.0 && flow.propagation_remaining == 0.0
            })?;
        }
        if is_complete {
            for flow_id in flow_ids {
                let route = self.with_flow(flow_id, |flow| flow.route.clone())?;
                self.release_flow_links(flow_id, &route);
                self.flow_states.remove(&flow_id);
            }
            *state = None;
            return Ok(PollResult::Complete { latency: 0.0 });
        }

        let delay_s = self
            .next_event_delay()?
            .ok_or_else(|| anyhow!("network flow has no active route bandwidth"))?;
        Ok(PollResult::Pending { latency: delay_s })
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
        let link_index = self.links.len();
        self.links.push(DirectionalLink {
            src,
            dst,
            name,
            bandwidth,
            latency,
            active_flows: BTreeSet::new(),
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
