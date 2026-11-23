use std::any::Any;
use std::collections::{BTreeMap, BTreeSet, HashMap, VecDeque};
use std::path::Path;
use std::rc::Rc;

use anyhow::{anyhow, ensure, Result};
use serde::{Deserialize, Serialize};

use crate::engine::Graph;
use crate::module::{LogicalHandler, Module};
use crate::network_sims::topology::{Topology, TopologyNetwork};
use crate::scheduler::KvMemoryInfo;
use crate::schedulers::prefix_cache::hash_block;
use crate::system::{System, SystemEvent};
use crate::types::{
    FastMap, InstanceLoad, NodeID, Request, RequestResult, ResultSubRequest, SubRequest,
};

fn one() -> usize {
    1
}

/// vLLM's DP coordinator publishes engine stats at most every 100 ms.
const STATS_INTERVAL_S: f64 = 0.1;

/// Worker choice for calls without an explicit binding.
#[derive(Clone, Copy, Debug, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum Routing {
    /// vLLM data-parallel load balancing (`DPLBAsyncMPClient`, commit
    /// ce72586f8aded8dafa7d4c745a5cef3198070e3e).
    #[default]
    LeastLoaded,
    /// Stable hash of the session ID, as in vLLM production-stack session
    /// routing; calls without a session fall back to `least_loaded`.
    Session,
    /// vLLM production-stack `LoadAwareRouter` (commit
    /// 2f879ef8a7c21f975ac0b2675267b7ba681b2cfa of `routing_logic.py`): KV-cache
    /// affinity weighted by in-flight load; see `cache_affinity_choice`.
    CacheAffinity,
}

fn default_beta() -> f64 {
    1.0
}

/// One frontend: load is the larger of this client's in-flight calls and the
/// snapshot's waiting + running; waiting is penalized from 50% KV usage. The
/// scan starts at a rotating index and the first strict minimum wins.
fn least_loaded_choice(snapshot: &[(f64, f64, f64)], inflight: &[u64], start: usize) -> usize {
    let mut best = (f64::INFINITY, 0);
    for i in 0..snapshot.len() {
        let index = (start + i) % snapshot.len();
        let (waiting, running, kv_usage) = snapshot[index];
        let mut score = (inflight[index] as f64).max(waiting + running);
        if waiting > 0.0 {
            score += waiting * 6.0 * (kv_usage - 0.5).max(0.0);
        }
        if score < best.0 {
            best = (score, index);
        }
    }
    best.1
}

/// A configured call placement; its block names are interned at init.
#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct CallConfig {
    request_id: u64,
    subrequest_index: usize,
    worker: usize,
    #[serde(default)]
    blocks: Vec<String>,
    #[serde(default)]
    completed_blocks: Vec<String>,
    #[serde(default)]
    seed_host_blocks: Vec<String>,
}

#[derive(Clone)]
struct CallBinding {
    automatic: bool,
    subrequest_index: usize,
    worker: usize,
    blocks: Vec<Name>,
    completed_blocks: Vec<Name>,
    seed_host_blocks: Vec<Name>,
}

/// Dense IDs for strings, in first-seen order; each string is stored once.
#[derive(Default)]
struct Interner {
    names: Vec<Rc<str>>,
    ids: HashMap<Rc<str>, u64>,
}
impl Interner {
    fn intern(&mut self, name: &str) -> u64 {
        if let Some(&id) = self.ids.get(name) {
            return id;
        }
        let id = self.names.len() as u64;
        let name: Rc<str> = name.into();
        self.names.push(name.clone());
        self.ids.insert(name, id);
        id
    }
}

/// A small set in an exactly sized slice. An emptied `BTreeSet` keeps its
/// node, and every block keeps its sets for its whole life.
#[derive(Debug, Default)]
struct SmallSet<T>(Box<[T]>);
impl<T: Copy + PartialEq> SmallSet<T> {
    fn insert(&mut self, value: T) {
        if !self.0.contains(&value) {
            self.0 = self.0.iter().copied().chain([value]).collect();
        }
    }
    fn retain(&mut self, keep: impl Fn(&T) -> bool) {
        if !self.0.iter().all(&keep) {
            self.0 = self.0.iter().copied().filter(|value| keep(value)).collect();
        }
    }
    fn iter(&self) -> std::slice::Iter<'_, T> {
        self.0.iter()
    }
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Pressure {
    request_id: u64,
    subrequest_index: usize,
    owner_request_id: u64,
    node: String,
    bytes: u64,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
struct Background {
    request_id: u64,
    subrequest_index: usize,
    src: String,
    dst: String,
    bytes: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CxlPoolSystem {
    instances: Vec<Box<dyn System>>,
    #[serde(default)]
    detailed_observations: bool,
    topology: Topology,
    block_tokens: u32,
    #[serde(default)]
    block_bytes: u64,
    host_capacity_bytes: BTreeMap<String, u64>,
    placement_devices: Vec<String>,
    #[serde(default = "one")]
    stripe_blocks: usize,
    #[serde(default = "one")]
    max_outstanding_reads: usize,
    #[serde(default)]
    routing: Routing,
    /// `cache_affinity` weight on relative load, in full cache hits per 100% above
    /// the fleet mean (production-stack `DEFAULT_LOADAWARE_BETA`).
    #[serde(default = "default_beta")]
    affinity_beta: f64,
    #[serde(default)]
    calls: Vec<CallConfig>,
    /// `calls` with interned names, indexed by `binding_index`.
    #[serde(skip)]
    configured: Vec<CallBinding>,
    #[serde(skip)]
    binding_index: BTreeMap<(u64, usize), usize>,
    /// Logical block names; see `Name`.
    #[serde(skip)]
    block_names: Interner,
    #[serde(skip)]
    session_names: Interner,
    /// Prefix namespaces, and each (namespace, chained hash)'s block name, so
    /// a block name is formatted and interned once.
    #[serde(skip)]
    namespaces: Interner,
    #[serde(skip)]
    prefix_names: FastMap<(u64, u64), Name>,
    #[serde(default)]
    pressures: Vec<Pressure>,
    #[serde(default)]
    background: Vec<Background>,
    #[serde(skip)]
    fabric: Option<TopologyNetwork>,
    #[serde(skip)]
    /// Each worker's KV domain.
    memory: Vec<KvMemoryInfo>,
    /// Hosts in `host_capacity_bytes` order; `Location::Host` indexes them.
    #[serde(skip)]
    host_names: Vec<String>,
    /// Distinct KV geometries, and each worker's index into them.
    #[serde(skip)]
    geometries: Vec<(u64, u64, u64)>,
    #[serde(skip)]
    worker_geometry: Vec<u32>,
    #[serde(skip)]
    active: BTreeMap<u64, Active>,
    #[serde(skip)]
    seen_requests: BTreeSet<u64>,
    #[serde(skip)]
    children: BTreeMap<u64, (u64, usize)>,
    #[serde(skip)]
    next_child: u64,
    #[serde(skip)]
    blocks: BTreeMap<Name, Block>,
    #[serde(skip)]
    next_block: u64,
    #[serde(skip)]
    placement_cursor: usize,
    #[serde(skip)]
    pressure_state: BTreeMap<(u64, usize), PressureState>,
    #[serde(skip)]
    transfers: Vec<Transfer>,
    #[serde(skip)]
    transfer_totals: BTreeMap<String, (u64, u64)>,
    #[serde(skip)]
    unfinished_transfers: usize,
    #[serde(skip)]
    results: Vec<RequestResult>,
    #[serde(skip)]
    first_token_waits: (u64, f64),
    #[serde(skip)]
    observations: Vec<serde_json::Value>,
    #[serde(skip)]
    host_bytes_freed: u64,
    #[serde(skip)]
    stores: FastMap<StoreId, Store>,
    #[serde(skip)]
    session_blocks: BTreeMap<(u64, Session), BTreeSet<Name>>,
    #[serde(skip)]
    child_placement: BTreeMap<u64, usize>,
    #[serde(skip)]
    pending_stores: BTreeMap<u32, VecDeque<Name>>,
    #[serde(skip)]
    queued_stores: BTreeSet<Name>,
    /// Per worker (waiting, running, KV usage) as of `lb_refreshed`.
    #[serde(skip)]
    lb_snapshot: Vec<(f64, f64, f64)>,
    #[serde(skip)]
    lb_refreshed: Option<f64>,
    #[serde(skip)]
    lb_start: usize,
    #[serde(skip)]
    routed: BTreeMap<(u64, usize), usize>,
    #[serde(skip)]
    inflight_calls: Vec<u64>,
}

crate::register_system!(CxlPoolSystem, "cxl_pool");

/// A copy's store and allocation handle. `node` indexes `host_names` and
/// `device` indexes `placement_devices`, so a copy owns no heap.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Location {
    Host {
        node: u32,
        worker: usize,
        offset: u64,
    },
    Pool {
        device: u32,
        offset: u64,
    },
    Hbm {
        worker: usize,
        offset: u64,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum StoreId {
    Host(u32),
    Pool(u32),
    Hbm(usize),
}
#[derive(Default)]
struct Store {
    free: BTreeMap<u64, u64>,
    /// Extents of fragmented allocations; a contiguous one is `(handle, bytes)`.
    allocations: BTreeMap<u64, Box<[(u64, u64)]>>,
    capacity: u64,
    used: u64,
    /// Reclaimable copies in LRU order. A touch only advances `Block::ordinal`;
    /// `CxlPoolSystem::oldest` re-keys stale copies once they reach the front.
    reclaimable: BTreeMap<u64, Name>,
    /// Each reclaimable copy's key in `reclaimable`.
    reclaimable_ordinals: FastMap<Name, u64>,
    reclaimable_bytes: u64,
    pending: BTreeSet<NodeID>,
    /// Usage changed since the owning worker last saw it.
    dirty: bool,
}
impl Store {
    fn new(capacity: u64) -> Self {
        Self {
            free: BTreeMap::from([(0, capacity)]),
            capacity,
            ..Self::default()
        }
    }
    fn available(&self, bytes: u64) -> Option<u64> {
        assert!(bytes > 0, "storage reservations must be nonzero");
        (self.capacity - self.used >= bytes).then(|| *self.free.first_key_value().unwrap().0)
    }
    fn reserve(&mut self, bytes: u64) -> Option<u64> {
        let handle = self.available(bytes)?;
        let mut remaining = bytes;
        let mut extents = Vec::new();
        while remaining > 0 {
            let (offset, size) = self.free.pop_first().unwrap();
            let taken = remaining.min(size);
            extents.push((offset, taken));
            if size > taken {
                self.free.insert(offset + taken, size - taken);
            }
            remaining -= taken;
        }
        self.used += bytes;
        self.dirty = true;
        if extents.len() > 1 {
            assert!(self
                .allocations
                .insert(handle, extents.into_boxed_slice())
                .is_none());
        }
        Some(handle)
    }
    fn extents(&self, handle: u64, bytes: u64) -> Box<[(u64, u64)]> {
        (self.allocations.get(&handle).cloned()).unwrap_or_else(|| Box::new([(handle, bytes)]))
    }
    fn release(&mut self, handle: u64, bytes: u64) {
        let extents = self
            .allocations
            .remove(&handle)
            .unwrap_or_else(|| Box::new([(handle, bytes)]));
        assert_eq!(extents.iter().map(|(_, size)| *size).sum::<u64>(), bytes);
        self.used -= bytes;
        self.dirty = true;
        for &(offset, size) in &extents {
            self.release_extent(offset, size);
        }
    }
    fn set_reclaimable(&mut self, key: &Name, ordinal: u64, bytes: u64, reclaimable: bool) {
        use std::collections::hash_map::Entry;
        match (self.reclaimable_ordinals.entry(*key), reclaimable) {
            (Entry::Vacant(entry), true) => {
                entry.insert(ordinal);
                self.reclaimable.insert(ordinal, *key);
                self.reclaimable_bytes += bytes;
            }
            (Entry::Occupied(entry), false) => {
                self.reclaimable.remove(&entry.remove());
                self.reclaimable_bytes -= bytes;
            }
            _ => return,
        }
        self.dirty = true;
    }
    fn release_extent(&mut self, mut offset: u64, mut bytes: u64) {
        if let Some((&previous, &size)) = self.free.range(..offset).next_back() {
            if previous + size == offset {
                self.free.remove(&previous);
                offset = previous;
                bytes += size;
            }
        }
        if let Some(size) = self.free.remove(&(offset + bytes)) {
            bytes += size;
        }
        self.free.insert(offset, bytes);
    }
}
impl Location {
    fn store(&self) -> StoreId {
        match self {
            Self::Host { node, .. } => StoreId::Host(*node),
            Self::Pool { device, .. } => StoreId::Pool(*device),
            Self::Hbm { worker, .. } => StoreId::Hbm(*worker),
        }
    }
    fn same_store(&self, other: &Self) -> bool {
        self.store() == other.store()
    }
    fn offset(&self) -> u64 {
        match self {
            Self::Host { offset, .. } | Self::Pool { offset, .. } | Self::Hbm { offset, .. } => {
                *offset
            }
        }
    }
    fn with_offset(self, value: u64) -> Self {
        match self {
            Self::Host { node, worker, .. } => Self::Host {
                node,
                worker,
                offset: value,
            },
            Self::Pool { device, .. } => Self::Pool {
                device,
                offset: value,
            },
            Self::Hbm { worker, .. } => Self::Hbm {
                worker,
                offset: value,
            },
        }
    }
}

/// Interned block name: a logical block's ID doubled, plus one for its
/// recurrent checkpoint. IDs only identify blocks; eviction order comes from
/// `Block::ordinal`.
type Name = u64;
type Session = u64;

/// A cached block of one worker's KV: every GPU's shard moves together.
struct Block {
    /// Prefix position.
    index: u64,
    sessions: SmallSet<(Session, u64)>,
    bytes: u64,
    /// Index into `geometries`.
    geometry: u32,
    location: Location,
    /// Exactly sized: a block rarely has more than two copies.
    replicas: Box<[Location]>,
    ordinal: u64,
    /// Boxed: only blocks in transfer have one.
    pending: Option<Box<(Location, NodeID)>>,
    /// Consuming child -> the worker whose HBM copy it uses.
    pins: SmallSet<(u64, usize)>,
}
impl Block {
    fn add_replica(&mut self, copy: Location) {
        self.replicas = self.replicas.iter().copied().chain([copy]).collect();
    }
    fn take_replica(&mut self, index: usize) -> Location {
        let copy = self.replicas[index];
        self.replicas = (self.replicas.iter().enumerate())
            .filter(|&(i, _)| i != index)
            .map(|(_, &copy)| copy)
            .collect();
        copy
    }
}
struct Active {
    request: Request,
    sink: Option<*mut dyn LogicalHandler>,
    completion: Vec<NodeID>,
    /// Boxed: most sub-requests are still pending.
    results: Vec<Option<Box<ResultSubRequest>>>,
    completed: Vec<bool>,
    calls: BTreeMap<usize, CallState>,
    /// Session of each sub-request.
    sessions: Vec<Session>,
    session_calls: BTreeMap<Session, usize>,
    conservative_sessions: bool,
}
struct CallState {
    binding: CallBinding,
    child: u64,
    released: f64,
    /// When the prefix first finished loading and the call became schedulable.
    ready: Option<f64>,
    /// Queued for its turn to look up its prefix.
    pending: bool,
    /// Fires once the prefix being loaded has arrived.
    loaded: Option<NodeID>,
    load_started: f64,
    /// Unissued reads, in prefix order.
    reads: VecDeque<Name>,
    /// Outstanding reads.
    inflight: usize,
    restored_blocks: usize,
    /// Tokens restored when the prefix first loaded.
    reused_tokens: Option<u32>,
    published_blocks: usize,
    /// Pinned blocks by prefix position.
    owned_keys: BTreeSet<(u64, Name)>,
    checkpoint_source: Option<(u32, Name)>,
    wait_since: f64,
    admission_wait: f64,
    preparation_wait: f64,
    admission_waiting: bool,
    admission_barrier: Option<NodeID>,
    preempted: bool,
}
struct PressureState {
    config: Pressure,
    offset: Option<u64>,
}
#[derive(Serialize)]
struct Transfer {
    kind: String,
    block: Option<String>,
    src_owner: Option<usize>,
    dst_owner: Option<usize>,
    route_src: u64,
    route_dst: u64,
    src: String,
    dst: String,
    bytes: u64,
    issue: f64,
    start: f64,
    completion: Option<f64>,
}

enum Event {
    AdmissionReleased,
    Arrival {
        request: u64,
        index: usize,
    },
    Release {
        request: u64,
        index: usize,
    },
    Complete {
        request: u64,
        index: usize,
    },
    Prepare {
        request: u64,
        index: usize,
    },
    Loaded {
        request: u64,
        index: usize,
    },
    MoveComplete {
        key: Name,
        transfer: Option<usize>,
        call: Option<(u64, usize)>,
        source: Location,
        release_source: bool,
    },
    Capacity {
        child: u64,
        required_bytes: u64,
        continuation: NodeID,
    },
    BackgroundComplete {
        transfer: Option<usize>,
    },
    ToolRun {
        duration: f64,
    },
}

impl CxlPoolSystem {
    fn active_calls(&self) -> impl Iterator<Item = &CallState> {
        self.children
            .values()
            .map(|(request, index)| &self.active[request].calls[index])
    }
    fn fabric(&self) -> &TopologyNetwork {
        self.fabric.as_ref().unwrap()
    }
    fn binding(&mut self, request: u64, index: usize) -> CallBinding {
        if let Some(&binding) = self.binding_index.get(&(request, index)) {
            let mut binding = self.configured[binding].clone();
            if binding.completed_blocks.is_empty() && binding.seed_host_blocks.is_empty() {
                binding.completed_blocks = binding.blocks.clone();
            }
            return binding;
        }
        let mut binding = CallBinding {
            automatic: true,
            subrequest_index: index,
            worker: 0,
            blocks: vec![],
            completed_blocks: vec![],
            seed_host_blocks: vec![],
        };
        let mut keys = Vec::new();
        let mut restorable = 0;
        let active = self.active.get(&request);
        if let Some(SubRequest::Llm {
            input_tokens,
            output_tokens,
            input_token_ids,
            output_token_ids,
            ..
        }) = active.and_then(|active| active.request.sub_requests.get(index))
        {
            let mut previous = 0u64;
            if let Some(prefix) = active.unwrap().request.resolved_kv_prefix(index) {
                use std::hash::{Hash, Hasher};
                let namespace = self.namespaces.intern(&prefix.namespace);
                let group = if self.block_tokens.is_multiple_of(prefix.block_tokens) {
                    (self.block_tokens / prefix.block_tokens) as usize
                } else {
                    prefix.block_ids.len() + 1
                };
                for ids in prefix.block_ids.chunks_exact(group) {
                    let mut hash = std::collections::hash_map::DefaultHasher::new();
                    previous.hash(&mut hash);
                    self.block_tokens.hash(&mut hash);
                    ids.hash(&mut hash);
                    previous = hash.finish();
                    keys.push((namespace, previous));
                }
            } else if let Some(input) = input_token_ids {
                // A token trace identifies its computed KV: the input and any
                // traced output but the last sampled token.
                let output = output_token_ids.as_deref().unwrap_or_default();
                let computed = (input_tokens + output_tokens.saturating_sub(1)) as usize;
                let tokens: Vec<u32> = input.iter().chain(output).take(computed).copied().collect();
                let namespace = self.namespaces.intern("tokens");
                for block in tokens.chunks_exact(self.block_tokens as usize) {
                    previous = hash_block(previous, block, &[]);
                    keys.push((namespace, previous));
                }
            }
            restorable = (input_tokens.saturating_sub(1) / self.block_tokens) as usize;
        }
        binding.completed_blocks = keys.into_iter().map(|key| self.prefix_name(key)).collect();
        binding.blocks = binding
            .completed_blocks
            .iter()
            .take(restorable)
            .copied()
            .collect();
        binding
    }
    fn prefix_name(&mut self, key: (u64, u64)) -> Name {
        if let Some(&name) = self.prefix_names.get(&key) {
            return name;
        }
        let namespace = &self.namespaces.names[key.0 as usize];
        let name = self.block_name_id(&format!("{namespace}:{:016x}", key.1));
        self.prefix_names.insert(key, name);
        name
    }
    fn block_name_id(&mut self, name: &str) -> Name {
        self.block_names.intern(name) << 1
    }
    fn block_name(&self, name: Name) -> String {
        let logical = &*self.block_names.names[(name >> 1) as usize];
        if name & 1 == 1 {
            format!("{logical}:recurrent-state")
        } else {
            logical.to_owned()
        }
    }
    /// vLLM counts queued requests, including those loading external KV, as
    /// waiting and admitted ones as running.
    fn load_snapshot(&self, worker: usize) -> Result<(f64, f64, f64)> {
        let state = self.instances[worker].observe();
        let (waiting, running) = match (
            state["scheduler"]["waiting_requests"].as_u64(),
            state["scheduler"]["running_requests"].as_u64(),
        ) {
            (Some(waiting), Some(running)) => (waiting, running),
            _ => {
                let load = self.instances[worker].instance_load();
                (load.waiting_requests as u64, load.running_requests as u64)
            }
        };
        // vLLM usage counts referenced blocks; unreferenced cached blocks are free.
        let capacity = self.memory[worker].capacity_bytes;
        let mut kv_usage = 0.0;
        if capacity > 0 {
            let private = capacity.saturating_sub(self.instances[worker].external_kv_budget()?);
            let store = &self.stores[&StoreId::Hbm(worker)];
            let pinned = store.used.saturating_sub(store.reclaimable_bytes);
            kv_usage = (private + pinned) as f64 / capacity as f64;
        }
        Ok((waiting as f64, running as f64, kv_usage))
    }
    /// production-stack `LoadAwareRouter`: worker `i` scores
    /// `min(matched_i, prompt) / prompt - beta * (inflight_i - mean) / max(1, mean)`.
    /// As in LMCache's controller `lookup` (commit
    /// 3f7c2017b1d52ed7c1f00fee9a57936080725dac), the prefix is walked block by block
    /// until a block has no copy in any host DRAM, and each host's `matched` is
    /// the end of the last block it holds in that run; the shared pool belongs
    /// to no instance. The first maximum wins; with no match anywhere, the
    /// router falls back to session routing.
    fn cache_affinity_choice(
        &self,
        request: u64,
        index: usize,
        blocks: &[Name],
    ) -> Result<Option<usize>> {
        let SubRequest::Llm { input_tokens, .. } =
            &self.active[&request].request.sub_requests[index]
        else {
            unreachable!()
        };
        let mut matched = HashMap::new();
        for (position, name) in blocks.iter().enumerate() {
            let Some(&Location::Host { node, .. }) =
                self.blocks.get(name).and_then(Self::offload_copy)
            else {
                break;
            };
            matched.insert(node, (position as u64 + 1) * u64::from(self.block_tokens));
        }
        if matched.is_empty() {
            return Ok(None);
        }
        let workers = self.instances.len();
        let mean = self.inflight_calls.iter().sum::<u64>() as f64 / workers as f64;
        let prompt = u64::from(*input_tokens);
        let mut best = (f64::NEG_INFINITY, 0);
        for worker in 0..workers {
            let node = self.host_of(worker)?;
            let hit = node
                .and_then(|node| matched.get(&node))
                .map_or(0.0, |&tokens: &u64| {
                    tokens.min(prompt) as f64 / prompt.max(1) as f64
                });
            let load = (self.inflight_calls[worker] as f64 - mean) / mean.max(1.0);
            let score = hit - self.affinity_beta * load;
            if score > best.0 {
                best = (score, worker);
            }
        }
        Ok(Some(best.1))
    }
    fn route(&mut self, request: u64, index: usize, blocks: &[Name], now: f64) -> Result<usize> {
        let workers = self.instances.len();
        let session = self.active[&request].request.sub_requests[index].session_id();
        let affine = match self.routing {
            Routing::CacheAffinity => self.cache_affinity_choice(request, index, blocks)?,
            _ => None,
        };
        let worker = match (self.routing, session, affine) {
            (_, _, Some(worker)) => worker,
            (Routing::Session | Routing::CacheAffinity, Some(session), _) => {
                use std::hash::{Hash, Hasher};
                let mut hash = std::collections::hash_map::DefaultHasher::new();
                session.hash(&mut hash);
                (hash.finish() % workers as u64) as usize
            }
            _ => {
                if self
                    .lb_refreshed
                    .is_none_or(|time| now - time >= STATS_INTERVAL_S)
                {
                    self.lb_snapshot = (0..workers)
                        .map(|worker| self.load_snapshot(worker))
                        .collect::<Result<_>>()?;
                    self.lb_refreshed = Some(now);
                }
                let worker =
                    least_loaded_choice(&self.lb_snapshot, &self.inflight_calls, self.lb_start);
                self.lb_snapshot[worker].0 += 1.0;
                self.lb_start = (self.lb_start + 1) % workers;
                worker
            }
        };
        self.inflight_calls[worker] += 1;
        self.routed.insert((request, index), worker);
        Ok(worker)
    }
    fn session(&self, request: u64, index: usize) -> (Session, u64) {
        (self.active[&request].sessions[index], request)
    }
    fn capacity(&self, location: &Location) -> Result<u64> {
        self.stores
            .get(&location.store())
            .map(|store| store.capacity)
            .ok_or_else(|| anyhow!("unknown KV store {location:?}"))
    }
    fn host_index(&self, node: &str) -> Option<u32> {
        self.host_names
            .binary_search_by(|name| name.as_str().cmp(node))
            .ok()
            .map(|index| index as u32)
    }
    /// The host DRAM of a worker's node, if it has KV capacity.
    fn host_of(&self, worker: usize) -> Result<Option<u32>> {
        Ok(self.host_index(&self.fabric().worker(worker)?.node))
    }
    fn used(&self, location: &Location) -> u64 {
        self.stores
            .get(&location.store())
            .map_or(0, |store| store.used)
    }
    fn locations(block: &Block) -> impl Iterator<Item = &Location> {
        std::iter::once(&block.location).chain(block.replicas.iter())
    }
    /// The block's one copy outside HBM, in some host DRAM or the pool.
    fn offload_copy(block: &Block) -> Option<&Location> {
        Self::locations(block).find(|p| !matches!(p, Location::Hbm { .. }))
    }
    /// A worker can use a block from its own HBM or from the block's offload
    /// copy, including one still being stored. Other workers' HBM is no source.
    fn usable(block: &Block, worker: usize) -> bool {
        Self::locations(block).any(|p| match p {
            Location::Hbm { worker: w, .. } => *w == worker,
            _ => true,
        }) || block
            .pending
            .as_deref()
            .is_some_and(|(target, _)| !matches!(target, Location::Hbm { .. }))
    }
    fn refresh(&mut self, key: &Name, touch: bool) {
        let Some(block) = self.blocks.get_mut(key) else {
            return;
        };
        if touch {
            block.ordinal = self.next_block;
            self.next_block += 1;
        }
        // A transfer reads the offload copy, or stores from the HBM location.
        let source = block
            .pending
            .is_some()
            .then(|| Self::offload_copy(block).unwrap_or(&block.location));
        // The last pinned copy in the last tier is a restore source.
        let last_pinned = !block.pins.is_empty() && block.replicas.is_empty();
        for (position, location) in Self::locations(block).enumerate() {
            let protected = source == Some(location)
                || match location {
                    // A pin protects its consumer's own copy; a queued store reads `location`.
                    Location::Hbm { worker, .. } => {
                        block.pins.iter().any(|&(_, copy)| copy == *worker)
                            || position == 0 && self.queued_stores.contains(key)
                    }
                    Location::Host { .. } => last_pinned && self.placement_devices.is_empty(),
                    Location::Pool { .. } => last_pinned,
                };
            self.stores
                .get_mut(&location.store())
                .unwrap()
                .set_reclaimable(key, block.ordinal, block.bytes, !protected);
        }
    }
    /// The store's least recently used reclaimable copy.
    fn oldest(&mut self, id: &StoreId) -> Option<(u64, Name)> {
        let store = self.stores.get_mut(id).unwrap();
        while let Some((&indexed, &key)) = store.reclaimable.first_key_value() {
            let ordinal = self.blocks[&key].ordinal;
            if ordinal == indexed {
                return Some((ordinal, key));
            }
            store.reclaimable.pop_first();
            store.reclaimable.insert(ordinal, key);
            store.reclaimable_ordinals.insert(key, ordinal);
        }
        None
    }
    fn discard_copy(&mut self, key: &Name, store: &StoreId, now: f64) {
        let block = self.blocks.get_mut(key).unwrap();
        let location = if block.location.store() == *store {
            if block.replicas.is_empty() {
                block.location
            } else {
                let replacement = block.take_replica(block.replicas.len() - 1);
                std::mem::replace(&mut block.location, replacement)
            }
        } else {
            let index = block
                .replicas
                .iter()
                .position(|p| p.store() == *store)
                .unwrap();
            block.take_replica(index)
        };
        let last = block.location == location && block.replicas.is_empty();
        let bytes = block.bytes;
        let storage = self.stores.get_mut(store).unwrap();
        storage.set_reclaimable(key, 0, bytes, false);
        storage.release(location.offset(), bytes);
        if last {
            let removed = self.blocks.remove(key).unwrap();
            for &(session, request) in removed.sessions.iter() {
                let membership = (request, session);
                if let Some(names) = self.session_blocks.get_mut(&membership) {
                    names.remove(key);
                    if names.is_empty() {
                        self.session_blocks.remove(&membership);
                    }
                }
            }
        } else {
            self.refresh(key, false);
        }
        if self.detailed_observations {
            let block = self.block_name(*key);
            self.observations.push(serde_json::json!({"event":"cache_reclaimed","block":block,"location":self.location_json(&location),"bytes":bytes,"time":now}));
        }
    }
    fn reclaim(&mut self, location: &Location, bytes_needed: u64, now: f64) -> Result<()> {
        let capacity = self.capacity(location)?;
        let id = location.store();
        while self.stores[&id].used.saturating_add(bytes_needed) > capacity {
            let Some((_, key)) = self.oldest(&id) else {
                break;
            };
            self.discard_copy(&key, &id, now);
        }
        Ok(())
    }
    /// A worker's attention block payload; zero for pure recurrent models.
    fn block_bytes(&self, worker: usize) -> Result<u64> {
        if self.block_bytes > 0 {
            return Ok(self.block_bytes);
        }
        let memory = &self.memory[worker];
        ensure!(
            memory.kv_bytes_per_token > 0 || memory.kv_bytes_per_request > 0,
            "model has no KV or recurrent state to restore"
        );
        memory
            .kv_bytes_per_token
            .checked_mul(u64::from(self.block_tokens))
            .ok_or_else(|| anyhow!("KV block size overflow"))
    }
    fn geometry(&self, worker: usize) -> u32 {
        self.worker_geometry[worker]
    }
    fn free_extent(&self, location: &Location, bytes: u64) -> Result<Option<u64>> {
        Ok(self
            .stores
            .get(&location.store())
            .and_then(|store| store.available(bytes)))
    }
    fn location_json(&self, location: &Location) -> serde_json::Value {
        match *location {
            Location::Host {
                node,
                worker,
                offset,
            } => serde_json::json!({"tier":"host","node":self.host_names[node as usize],
                "worker":worker,"offset":offset}),
            Location::Pool { device, offset } => serde_json::json!({"tier":"pool",
                "device":self.placement_devices[device as usize],"offset":offset}),
            Location::Hbm { worker, offset } => {
                serde_json::json!({"tier":"hbm","worker":worker,"offset":offset})
            }
        }
    }
    fn observe_location(&self, location: &Location, bytes: u64) -> serde_json::Value {
        let mut value = self.location_json(location);
        value["extents"] =
            serde_json::to_value(self.stores[&location.store()].extents(location.offset(), bytes))
                .unwrap();
        value
    }
    fn allocate(&mut self, location: Location, bytes: u64) -> Result<Location> {
        let offset = self
            .stores
            .get_mut(&location.store())
            .unwrap()
            .reserve(bytes)
            .ok_or_else(|| anyhow!("no free KV extent in {location:?}"))?;
        Ok(location.with_offset(offset))
    }
    fn owner(location: &Location) -> Option<usize> {
        match location {
            Location::Host { worker, .. } | Location::Hbm { worker, .. } => Some(*worker),
            Location::Pool { .. } => None,
        }
    }
    fn endpoint(&self, location: &Location) -> Result<String> {
        match location {
            Location::Host { worker, .. } => Ok(self.fabric().worker(*worker)?.dram.clone()),
            Location::Pool { device, .. } => Ok(self.placement_devices[*device as usize].clone()),
            Location::Hbm { worker, .. } => Ok(self.fabric().worker(*worker)?.gpu.clone()),
        }
    }
    /// Tell each worker whose HBM usage changed.
    fn sync_hbm(&mut self) -> Result<()> {
        for worker in 0..self.instances.len() {
            let store = self.stores.get_mut(&StoreId::Hbm(worker)).unwrap();
            if std::mem::take(&mut store.dirty) {
                let (used, reclaimable) = (store.used, store.reclaimable_bytes);
                self.instances[worker].set_external_kv_reclaimable_bytes(reclaimable)?;
                self.instances[worker].set_external_kv_bytes(used)?;
            }
        }
        Ok(())
    }
    fn pressure_requested(&self, node: &str) -> u64 {
        self.pressure_state
            .values()
            .filter(|p| p.config.node == node)
            .map(|p| p.config.bytes)
            .sum()
    }
    fn admit_pressure(&mut self, now: f64) -> Result<()> {
        let waiting = self
            .pressure_state
            .iter()
            .filter(|(_, p)| p.offset.is_none())
            .map(|(key, p)| (*key, p.config.clone()))
            .collect::<Vec<_>>();
        for (key, p) in waiting {
            // `init` checks that every pressure host has capacity.
            let node = self.host_index(&p.node).unwrap();
            if let Some(offset) = self.free_extent(
                &Location::Host {
                    node,
                    worker: 0,
                    offset: 0,
                },
                p.bytes,
            )? {
                self.stores
                    .get_mut(&StoreId::Host(node))
                    .unwrap()
                    .reserve(p.bytes)
                    .unwrap();
                self.pressure_state.get_mut(&key).unwrap().offset = Some(offset);
                if self.detailed_observations {
                    self.observations.push(serde_json::json!({"event":"pressure_admitted","owner_request_id":p.owner_request_id,"node":p.node,"bytes":p.bytes,"offset":offset,"time":now}));
                }
            }
        }
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn move_block(
        &mut self,
        key: &Name,
        source: Location,
        target: Location,
        kind: &str,
        now: f64,
        issued: f64,
        current: NodeID,
        call: Option<(u64, usize)>,
        graph: &mut Graph,
    ) -> Result<NodeID> {
        ensure!(
            self.blocks[key].pending.is_none(),
            "block {key:?} already moving"
        );
        let src = self.endpoint(&source)?;
        let dst = self.endpoint(&target)?;
        let bytes = self.blocks[key].bytes;
        let block_index = self.blocks[key].index;
        let (source_id, destination_id) = self.fabric.as_mut().unwrap().transfer_route(
            &src,
            &dst,
            Self::owner(&source),
            Self::owner(&target),
            block_index,
        )?;
        let module = self as *mut Self as *mut dyn LogicalHandler;
        self.unfinished_transfers += 1;
        let totals = self.transfer_totals.entry(kind.to_owned()).or_default();
        totals.0 += 1;
        totals.1 += bytes;
        let transfer = self.detailed_observations.then(|| {
            let index = self.transfers.len();
            self.transfers.push(Transfer {
                kind: kind.into(),
                block: Some(self.block_name(*key)),
                src_owner: Self::owner(&source),
                dst_owner: Self::owner(&target),
                route_src: source_id,
                route_dst: destination_id,
                src,
                dst,
                bytes,
                issue: issued,
                start: now,
                completion: None,
            });
            index
        });
        let network = graph.add_network_node(
            &mut self.fabric.as_mut().unwrap().flow,
            source_id,
            destination_id,
            bytes,
        );
        let source_store = source.store();
        let complete = graph.add_logical_node(
            module,
            Event::MoveComplete {
                key: *key,
                transfer,
                call,
                source,
                release_source: kind == "spill",
            },
        );
        graph.add_edge(current, network)?;
        graph.add_edge(network, complete)?;
        self.stores
            .get_mut(&target.store())
            .unwrap()
            .pending
            .insert(complete);
        self.stores
            .get_mut(&source_store)
            .unwrap()
            .pending
            .insert(complete);
        self.blocks.get_mut(key).unwrap().pending = Some(Box::new((target, complete)));
        self.refresh(key, false);
        self.sync_hbm()?;
        Ok(complete)
    }
    fn pool_target(&mut self, bytes: u64, now: f64) -> Result<Option<Location>> {
        let start = (self.placement_cursor / self.stripe_blocks) % self.placement_devices.len();
        let eligible = (0..self.placement_devices.len())
            .map(|step| Location::Pool {
                device: ((start + step) % self.placement_devices.len()) as u32,
                offset: 0,
            })
            .filter(|location| self.stores[&location.store()].capacity >= bytes)
            .collect::<Vec<_>>();
        ensure!(
            !eligible.is_empty(),
            "no CXL device can hold a {bytes}-byte KV block or recurrent checkpoint"
        );
        loop {
            if let Some(target) = eligible
                .iter()
                .find(|p| self.stores[&p.store()].available(bytes).is_some())
            {
                return Ok(Some(*target));
            }
            let oldest = (0..self.placement_devices.len())
                .filter_map(|device| {
                    let id = StoreId::Pool(device as u32);
                    self.oldest(&id).map(|(ordinal, key)| (ordinal, key, id))
                })
                .min_by_key(|(ordinal, _, _)| *ordinal);
            let Some((_, key, store)) = oldest else {
                return Ok(None);
            };
            self.discard_copy(&key, &store, now);
        }
    }
    fn spill_for(
        &mut self,
        node: u32,
        additional: u64,
        now: f64,
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<Vec<NodeID>> {
        let host = Location::Host {
            node,
            worker: 0,
            offset: 0,
        };
        let name = self.host_names[node as usize].clone();
        let node = name.as_str();
        let requested = self.pressure_requested(node);
        let capacity = self.capacity(&host)?;
        ensure!(requested <= capacity, "pressure exceeds host capacity");
        let reserved_pressure: u64 = self
            .pressure_state
            .values()
            .filter(|p| p.config.node == node && p.offset.is_some())
            .map(|p| p.config.bytes)
            .sum();
        let mut deficit = self
            .used(&host)
            .saturating_add(requested - reserved_pressure)
            .saturating_add(additional)
            .saturating_sub(capacity);
        let mut completions = Vec::new();
        while deficit > 0 {
            let Some((_, key)) = self.oldest(&host.store()) else {
                break;
            };
            let source = *Self::locations(&self.blocks[&key])
                .find(|p| p.same_store(&host))
                .unwrap();
            let bytes = self.blocks[&key].bytes;
            // Without a pool, host DRAM is the last tier, as in vLLM CPU offloading.
            if self.placement_devices.is_empty()
                || Self::locations(&self.blocks[&key]).any(|p| matches!(p, Location::Pool { .. }))
            {
                self.discard_copy(&key, &host.store(), now);
            } else {
                let Some(target) = self.pool_target(bytes, now)? else {
                    break;
                };
                let target = self.allocate(target, bytes)?;
                self.placement_cursor += 1;
                completions.push(self.move_block(
                    &key, source, target, "spill", now, now, current, None, graph,
                )?);
            }
            deficit = deficit.saturating_sub(bytes);
        }
        self.admit_pressure(now)?;
        Ok(completions)
    }
    fn spill(&mut self, node: u32, now: f64, current: NodeID, graph: &mut Graph) -> Result<()> {
        self.spill_for(node, 0, now, current, graph)?;
        Ok(())
    }
    fn checkpoint_key(logical: &Name) -> Name {
        logical | 1
    }
    fn own_key(&mut self, request: u64, index: usize, key: &Name) {
        let session = self.session(request, index);
        let state = self
            .active
            .get_mut(&request)
            .unwrap()
            .calls
            .get_mut(&index)
            .unwrap();
        let block = self.blocks.get_mut(key).unwrap();
        block.pins.insert((state.child, state.binding.worker));
        block.sessions.insert(session);
        state.owned_keys.insert((block.index, *key));
        self.session_blocks
            .entry((request, session.0))
            .or_default()
            .insert(*key);
        self.refresh(key, true);
    }
    fn release_execution(&mut self, owned_keys: &BTreeSet<(u64, Name)>, child: u64) {
        // Release suffixes first, as with the vLLM free-block queue.
        for (_, key) in owned_keys.iter().rev() {
            if let Some(block) = self.blocks.get_mut(key) {
                block.pins.retain(|&(pin, _)| pin != child);
            }
            self.refresh(key, true);
        }
    }
    fn queue_store(&mut self, key: Name) -> Result<()> {
        let block = &self.blocks[&key];
        if Self::locations(block)
            .any(|p| matches!(p, Location::Host { .. } | Location::Pool { .. }))
            || !self.queued_stores.insert(key)
        {
            return Ok(());
        }
        let Location::Hbm { worker, .. } = block.location else {
            return Ok(());
        };
        let node = self.host_of(worker)?;
        let node = node
            .filter(|&node| self.stores[&StoreId::Host(node)].capacity >= block.bytes)
            .ok_or_else(|| anyhow!("host capacity cannot hold one KV block"))?;
        self.pending_stores.entry(node).or_default().push_back(key);
        self.refresh(&key, false);
        Ok(())
    }
    fn store_completed(&mut self, now: f64, current: NodeID, graph: &mut Graph) -> Result<()> {
        // Visit each host queue until its first capacity dependency; never rescan its backlog.
        let hosts = self.pending_stores.keys().cloned().collect::<Vec<_>>();
        for node in hosts {
            while let Some(&key) = self.pending_stores[&node].front() {
                let Some(block) = self.blocks.get(&key) else {
                    self.pending_stores.get_mut(&node).unwrap().pop_front();
                    self.queued_stores.remove(&key);
                    continue;
                };
                if Self::locations(block)
                    .any(|p| matches!(p, Location::Host { .. } | Location::Pool { .. }))
                {
                    self.pending_stores.get_mut(&node).unwrap().pop_front();
                    self.queued_stores.remove(&key);
                    continue;
                }
                if block.pending.is_some() {
                    break;
                }
                let source = block.location;
                let Location::Hbm { worker, .. } = source else {
                    break;
                };
                let bytes = block.bytes;
                let target = Location::Host {
                    node,
                    worker,
                    offset: 0,
                };
                if self.free_extent(&target, bytes)?.is_none() {
                    self.spill_for(node, bytes, now, current, graph)?;
                }
                if self.free_extent(&target, bytes)?.is_none() {
                    break;
                }
                let target = self.allocate(target, bytes)?;
                self.move_block(
                    &key, source, target, "store", now, now, current, None, graph,
                )?;
                self.pending_stores.get_mut(&node).unwrap().pop_front();
                self.queued_stores.remove(&key);
            }
        }
        Ok(())
    }
    fn issue_read(
        &mut self,
        request: u64,
        index: usize,
        key: Name,
        now: f64,
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<()> {
        let call = &self.active[&request].calls[&index];
        let loaded = call.loaded.expect("reads belong to a load");
        let (worker, issued) = (call.binding.worker, call.load_started);
        let source = *Self::offload_copy(&self.blocks[&key])
            .ok_or_else(|| anyhow!("reload of block without an offload copy"))?;
        // Destination extents were reserved at prepare; replace the placeholder.
        let target = self.blocks.get_mut(&key).unwrap().pending.take().unwrap().0;
        ensure!(
            target.store() == StoreId::Hbm(worker),
            "invalid reload reservation"
        );
        let kind = match &source {
            Location::Pool { .. } => "cxl_read",
            Location::Host { node, .. } if Some(*node) == self.host_of(worker)? => "host_read",
            _ => "rdma_read",
        };
        let complete = self.move_block(
            &key,
            source,
            target,
            kind,
            now,
            issued,
            current,
            Some((request, index)),
            graph,
        )?;
        graph.add_edge(complete, loaded)?;
        Ok(())
    }
    fn wait_phase(&mut self, request: u64, index: usize, admission: bool, now: f64) {
        let state = self
            .active
            .get_mut(&request)
            .unwrap()
            .calls
            .get_mut(&index)
            .unwrap();
        if state.ready.is_none() {
            let elapsed = now - state.wait_since;
            if state.admission_waiting {
                state.admission_wait += elapsed;
            } else {
                state.preparation_wait += elapsed;
            }
            state.wait_since = now;
        }
        state.admission_waiting = admission;
    }
    fn prepare(
        &mut self,
        request: u64,
        index: usize,
        now: f64,
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<()> {
        let binding = self.active[&request].calls[&index].binding.clone();
        let worker = binding.worker;
        let child = self.active[&request].calls[&index].child;
        let module = self as *mut Self as *mut dyn LogicalHandler;
        if self.active[&request].calls[&index].ready.is_none() {
            let older = self
                .active_calls()
                .filter(|call| {
                    call.child < child
                        && call.binding.worker == binding.worker
                        && call.ready.is_none()
                })
                .filter_map(|call| call.admission_barrier.map(|barrier| (call.child, barrier)))
                .min_by_key(|(child, _)| *child);
            if let Some((_, barrier)) = older {
                self.wait_phase(request, index, true, now);
                let retry = graph.add_logical_node(module, Event::Prepare { request, index });
                graph.add_edge(current, retry)?;
                graph.add_edge(barrier, retry)?;
                // Younger calls keep waiting behind this one.
                if let Some(own) = self.active[&request].calls[&index].admission_barrier {
                    graph.add_edge(retry, own)?;
                }
                return Ok(());
            }
        }
        // As in vLLM, the lookup covers the known tokens but the last, whose
        // logits are needed; a resumed call knows its generated tokens too.
        let known = self.instances[binding.worker]
            .known_tokens(child)
            .ok_or_else(|| anyhow!("call {child} is not queued on its worker"))?;
        let lookup = (known.saturating_sub(1) / self.block_tokens) as usize;
        let block_bytes = self.block_bytes(worker)?;
        let geometry = self.geometry(worker);
        let mut restored = 0;
        for logical in binding.blocks.iter().take(lookup) {
            if block_bytes > 0 {
                let Some(block) = self
                    .blocks
                    .get(logical)
                    .filter(|block| Self::usable(block, worker))
                else {
                    break;
                };
                ensure!(
                    block.bytes == block_bytes && block.geometry == geometry,
                    "incompatible physical KV geometry for {}",
                    self.block_name(*logical)
                );
            }
            restored += 1;
        }
        let state_bytes = self.memory[worker].kv_bytes_per_request;
        if state_bytes > 0 {
            restored = (1..=restored)
                .rev()
                .find(|&count| {
                    self.blocks
                        .get(&Self::checkpoint_key(&binding.blocks[count - 1]))
                        .is_some_and(|block| {
                            Self::usable(block, worker)
                                && block.bytes == state_bytes
                                && block.geometry == geometry
                        })
                })
                .unwrap_or(0);
        }
        self.active
            .get_mut(&request)
            .unwrap()
            .calls
            .get_mut(&index)
            .unwrap()
            .restored_blocks = restored;
        let mut keys = Vec::new();
        if block_bytes > 0 {
            keys.extend_from_slice(&binding.blocks[..restored]);
        }
        if state_bytes > 0 && restored > 0 {
            let checkpoint = Self::checkpoint_key(&binding.blocks[restored - 1]);
            self.active
                .get_mut(&request)
                .unwrap()
                .calls
                .get_mut(&index)
                .unwrap()
                .checkpoint_source = Some((restored as u32 * self.block_tokens, checkpoint));
            keys.push(checkpoint);
        }
        let target = Location::Hbm { worker, offset: 0 };
        let resident = |block: &Block| Self::locations(block).any(|p| p.same_store(&target));
        let mut wait = BTreeSet::new();
        for key in &keys {
            let block = &self.blocks[key];
            if !resident(block) {
                if let Some((_, complete)) = block.pending.as_deref() {
                    wait.insert(*complete);
                }
            }
        }
        let module = self as *mut Self as *mut dyn LogicalHandler;
        if !wait.is_empty() {
            self.wait_phase(request, index, false, now);
            self.active
                .get_mut(&request)
                .unwrap()
                .calls
                .get_mut(&index)
                .unwrap()
                .admission_barrier = None;
            let retry = graph.add_logical_node(module, Event::Prepare { request, index });
            graph.add_edge(current, retry)?;
            for node in wait {
                graph.add_edge(node, retry)?;
            }
            return Ok(());
        }
        // Pin the complete consuming prefix and its recurrent checkpoint before reclaiming.
        for key in &keys {
            self.own_key(request, index, key);
        }
        let restored_tokens = restored as u32 * self.block_tokens;
        // As vLLM allocates a request's new tokens with its external prefix,
        // the prefix loads only if the call can then be admitted.
        self.instances[worker].reserve_external_admission(child, 0)?;
        let admission = self.instances[worker].external_admission_bytes(known, restored_tokens);
        let mut blocked = BTreeSet::new();
        let needed = keys
            .iter()
            .map(|key| &self.blocks[key])
            .filter(|block| !resident(block))
            .map(|block| block.bytes)
            .sum::<u64>()
            + admission;
        let budget = self.instances[worker].external_kv_budget()?;
        let private = self.capacity(&target)?.saturating_sub(budget);
        self.reclaim(&target, needed.saturating_add(private), now)?;
        if self.used(&target).saturating_add(needed) > budget {
            if let Some(capacity) = self.instances[worker].next_kv_capacity_event() {
                blocked.insert(capacity);
            } else {
                blocked.extend(self.stores[&target.store()].pending.iter().copied());
                for active in self.active.values() {
                    for (&index, other) in &active.calls {
                        if other.child == child || other.binding.worker != worker {
                            continue;
                        }
                        if let Some(loaded) = other.loaded {
                            blocked.insert(loaded);
                        } else if other.ready.is_some() && !other.preempted {
                            blocked.insert(active.completion[index]);
                        }
                    }
                }
            }
            ensure!(
                !blocked.is_empty(),
                "restored prefix cannot fit available HBM"
            );
        }
        if !blocked.is_empty() {
            // As in vLLM, a request that cannot be allocated holds no KV while
            // it waits; its prefix is pinned again when it retries.
            let state = self
                .active
                .get_mut(&request)
                .unwrap()
                .calls
                .get_mut(&index)
                .unwrap();
            let owned = std::mem::take(&mut state.owned_keys);
            self.release_execution(&owned, child);
            self.wait_phase(request, index, true, now);
            let retry = graph.add_logical_node(module, Event::Prepare { request, index });
            graph.add_edge(current, retry)?;
            if self.active[&request].calls[&index].ready.is_none() {
                let barrier = match self.active[&request].calls[&index].admission_barrier {
                    Some(barrier) => barrier,
                    None => graph.add_logical_node(module, Event::AdmissionReleased),
                };
                self.active
                    .get_mut(&request)
                    .unwrap()
                    .calls
                    .get_mut(&index)
                    .unwrap()
                    .admission_barrier = Some(barrier);
                graph.add_edge(retry, barrier)?;
            }
            for complete in blocked {
                graph.add_edge(complete, retry)?;
            }
            return Ok(());
        }
        self.wait_phase(request, index, false, now);
        self.active
            .get_mut(&request)
            .unwrap()
            .calls
            .get_mut(&index)
            .unwrap()
            .admission_barrier = None;
        self.instances[worker].reserve_external_admission(child, admission)?;
        self.instances[worker].set_external_admission_blocked(child, false)?;
        let missing = keys
            .into_iter()
            .filter(|key| !resident(&self.blocks[key]))
            .collect::<Vec<_>>();
        if missing.is_empty() {
            self.sync_hbm()?;
            return self.finish_load(request, index, now);
        }
        let loaded = graph.add_logical_node(module, Event::Loaded { request, index });
        graph.add_edge(current, loaded)?;
        let state = self
            .active
            .get_mut(&request)
            .unwrap()
            .calls
            .get_mut(&index)
            .unwrap();
        state.loaded = Some(loaded);
        state.load_started = now;
        for key in &missing {
            let bytes = self.blocks[key].bytes;
            self.reclaim(&target, bytes, now)?;
            let target = self.allocate(target, bytes)?;
            self.blocks.get_mut(key).unwrap().pending = Some(Box::new((target, loaded)));
            self.refresh(key, false);
        }
        self.sync_hbm()?;
        // Issue in prefix order up to the limit; each completion issues the next read.
        for key in missing {
            let state = self
                .active
                .get_mut(&request)
                .unwrap()
                .calls
                .get_mut(&index)
                .unwrap();
            if state.inflight < self.max_outstanding_reads {
                state.inflight += 1;
                self.issue_read(request, index, key, now, current, graph)?;
            } else {
                state.reads.push_back(key);
            }
        }
        Ok(())
    }
    /// The call's prefix is in HBM: it becomes schedulable with that prefix.
    fn finish_load(&mut self, request: u64, index: usize, now: f64) -> Result<()> {
        self.wait_phase(request, index, false, now);
        let state = self
            .active
            .get_mut(&request)
            .unwrap()
            .calls
            .get_mut(&index)
            .unwrap();
        ensure!(
            state.inflight == 0 && state.reads.is_empty(),
            "prefix finished loading with KV transfers outstanding"
        );
        state.ready.get_or_insert(now);
        state.loaded = None;
        let resumed = std::mem::take(&mut state.preempted);
        let (worker, child) = (state.binding.worker, state.child);
        let tokens = state.restored_blocks as u32 * self.block_tokens;
        state.reused_tokens.get_or_insert(tokens);
        let (released, preparation_wait, admission_wait) =
            (state.released, state.preparation_wait, state.admission_wait);
        self.instances[worker].restore_external_prefix(child, tokens)?;
        self.instances[worker].set_external_pending(child, false)?;
        self.instances[worker].set_external_blocked(child, false)?;
        if self.detailed_observations {
            let event = if resumed {
                "request_resumed"
            } else {
                "kv_ready"
            };
            self.observations.push(serde_json::json!({"event":event,"request_id":request,"subrequest_index":index,"worker":worker,"time":now,"released":released,"kv_preparation_wait":preparation_wait,"hbm_admission_wait":admission_wait,"restored_tokens":tokens}));
        }
        Ok(())
    }
    fn publish(
        &mut self,
        request: u64,
        index: usize,
        computed_tokens: u32,
        now: f64,
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<()> {
        let source = self.active[&request].calls[&index].checkpoint_source;
        if let Some((tokens, key)) = source.filter(|(tokens, _)| computed_tokens > *tokens) {
            let state = self
                .active
                .get_mut(&request)
                .unwrap()
                .calls
                .get_mut(&index)
                .unwrap();
            state.checkpoint_source = None;
            let child = state.child;
            let block = self.blocks.get_mut(&key).unwrap();
            block.pins.retain(|&(pin, _)| pin != child);
            state.owned_keys.remove(&(block.index, key));
            self.refresh(&key, true);
            if self.detailed_observations {
                self.observations.push(serde_json::json!({"event":"recurrent_source_released","request_id":request,"subrequest_index":index,"tokens":tokens,"time":now}));
            }
            self.sync_hbm()?;
        }
        let input_tokens = match &self.active[&request].request.sub_requests[index] {
            SubRequest::Llm { input_tokens, .. } => *input_tokens,
            _ => unreachable!(),
        };
        let state = self
            .active
            .get_mut(&request)
            .unwrap()
            .calls
            .get_mut(&index)
            .unwrap();
        let completed_count = (computed_tokens / self.block_tokens) as usize;
        if state.binding.automatic {
            while state.binding.completed_blocks.len() < completed_count {
                let position = state.binding.completed_blocks.len();
                let name = self
                    .block_names
                    .intern(&format!("private:{request}:{index}:{position}"));
                state.binding.completed_blocks.push(name << 1);
            }
        }
        let (worker, child) = (state.binding.worker, state.child);
        let first = state.published_blocks;
        let count = (computed_tokens / self.block_tokens) as usize;
        let count = count.min(state.binding.completed_blocks.len());
        let new_blocks = state.binding.completed_blocks[first..count].to_vec();
        let checkpoint_logical = count
            .checked_sub(1)
            .map(|position| state.binding.completed_blocks[position]);
        let scheduler_block = self.memory[worker].block_tokens;
        let checkpoint_tokens =
            (input_tokens.saturating_sub(1) / scheduler_block) * scheduler_block;
        let state_bytes = self.memory[worker].kv_bytes_per_request;
        let checkpoint = computed_tokens == checkpoint_tokens
            && checkpoint_tokens > 0
            && checkpoint_tokens.is_multiple_of(self.block_tokens)
            && count > 0
            && checkpoint_tokens / self.block_tokens <= count as u32
            && state_bytes > 0;
        if count <= first && !checkpoint {
            return Ok(());
        }
        // Transfer ownership before increasing external occupancy.
        self.instances[worker].adopt_external_prefix(child, (count as u32) * self.block_tokens)?;
        let block_bytes = self.block_bytes(worker)?;
        if block_bytes > 0 {
            for (relative, logical) in new_blocks.iter().enumerate() {
                let position = (first + relative) as u64;
                self.publish_block(
                    request,
                    index,
                    *logical,
                    position,
                    block_bytes,
                    "publish_computed",
                    now,
                )?;
            }
        }
        if checkpoint {
            self.instances[worker].adopt_external_state(child)?;
            let position = (checkpoint_tokens / self.block_tokens - 1) as u64;
            let key = Self::checkpoint_key(checkpoint_logical.as_ref().unwrap());
            self.publish_block(
                request,
                index,
                key,
                position,
                state_bytes,
                "publish_recurrent_checkpoint",
                now,
            )?;
            self.active
                .get_mut(&request)
                .unwrap()
                .calls
                .get_mut(&index)
                .unwrap()
                .checkpoint_source = Some((checkpoint_tokens, key));
        }
        self.active
            .get_mut(&request)
            .unwrap()
            .calls
            .get_mut(&index)
            .unwrap()
            .published_blocks = count;
        self.sync_hbm()?;
        self.store_completed(now, current, graph)
    }
    /// Adopts one computed block from the call's HBM, pins it and queues its store.
    #[allow(clippy::too_many_arguments)]
    fn publish_block(
        &mut self,
        request: u64,
        index: usize,
        key: Name,
        position: u64,
        bytes: u64,
        event: &str,
        now: f64,
    ) -> Result<()> {
        let worker = self.active[&request].calls[&index].binding.worker;
        let geometry = self.geometry(worker);
        let target = Location::Hbm { worker, offset: 0 };
        if let Some(block) = self.blocks.get(&key) {
            ensure!(
                block.bytes == bytes && block.geometry == geometry,
                "incompatible published KV geometry for {}",
                self.block_name(key)
            );
        }
        if !self
            .blocks
            .get(&key)
            .is_some_and(|block| Self::locations(block).any(|p| p.same_store(&target)))
        {
            self.reclaim(&target, bytes, now)?;
            let target = self.allocate(target, bytes)?;
            if let Some(block) = self.blocks.get_mut(&key) {
                block.add_replica(target);
            } else {
                let block = Block {
                    index: position,
                    sessions: SmallSet::default(),
                    bytes,
                    geometry,
                    location: target,
                    replicas: Box::default(),
                    ordinal: self.next_block,
                    pending: None,
                    pins: SmallSet::default(),
                };
                self.blocks.insert(key, block);
                self.next_block += 1;
            }
            if self.detailed_observations {
                let block = self.block_name(key);
                self.observations.push(serde_json::json!({"event":event,"block":block,"worker":worker,"time":now,"bytes":bytes}));
            }
        }
        self.own_key(request, index, &key);
        self.queue_store(key)
    }
    fn seed(&mut self, binding: &CallBinding, request: u64, now: f64) -> Result<()> {
        let worker = binding.worker;
        let bytes = self.block_bytes(worker)?;
        if bytes == 0 {
            return Ok(());
        }
        let node = self
            .host_of(worker)?
            .ok_or_else(|| anyhow!("worker {worker} has no host capacity"))?;
        for (index, &key) in binding.seed_host_blocks.iter().enumerate() {
            if self.blocks.contains_key(&key) {
                continue;
            }
            let location = self.allocate(
                Location::Host {
                    node,
                    worker,
                    offset: 0,
                },
                bytes,
            )?;
            let session = self.session(request, binding.subrequest_index);
            let mut block = Block {
                index: index as u64,
                sessions: SmallSet::default(),
                bytes,
                geometry: self.geometry(worker),
                location,
                replicas: Box::default(),
                ordinal: self.next_block,
                pending: None,
                pins: SmallSet::default(),
            };
            block.sessions.insert(session);
            self.blocks.insert(key, block);
            self.next_block += 1;
            self.session_blocks
                .entry((request, session.0))
                .or_default()
                .insert(key);
            self.refresh(&key, false);
            if self.detailed_observations {
                let block = self.block_name(key);
                self.observations.push(serde_json::json!({"event":"seed_host","block":block,"worker":worker,"time":now,"bytes":bytes}));
            }
        }
        Ok(())
    }
    fn ensure_capacity(
        &mut self,
        child: u64,
        required_bytes: u64,
        continuation: NodeID,
        now: f64,
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<()> {
        let worker = self.child_placement[&child];
        self.instances[worker].set_external_admission_blocked(child, false)?;
        let required_bytes = self.instances[worker]
            .external_kv_requirement()
            .map_or(required_bytes, |(_, bytes)| bytes);
        let mut wait = BTreeSet::new();
        let location = Location::Hbm { worker, offset: 0 };
        self.reclaim(&location, required_bytes, now)?;
        if self.used(&location).saturating_add(required_bytes) > self.capacity(&location)? {
            wait.extend(self.stores[&location.store()].pending.iter().copied());
        }
        self.sync_hbm()?;
        if !wait.is_empty() {
            self.instances[worker].set_external_admission_blocked(child, true)?;
            let module = self as *mut Self as *mut dyn LogicalHandler;
            let retry = graph.add_logical_node(
                module,
                Event::Capacity {
                    child,
                    required_bytes,
                    continuation,
                },
            );
            graph.add_edge(current, retry)?;
            for complete in wait {
                graph.add_edge(complete, retry)?;
            }
            graph.add_edge(retry, continuation)?;
            if self.detailed_observations {
                self.observations.push(serde_json::json!({"event":"hbm_admission_wait","child_request_id":child,"worker":worker,"time":now,"required_private_bytes":required_bytes}));
            }
        }
        Ok(())
    }
    fn close_session_call(&mut self, request: u64, index: usize, now: f64) {
        let session = self.session(request, index);
        let active = self.active.get_mut(&request).unwrap();
        let remaining = active.session_calls.get_mut(&session.0).unwrap();
        *remaining -= 1;
        if *remaining == 0 && !active.conservative_sessions {
            active.session_calls.remove(&session.0);
            self.release_session(request, session.0, now);
        }
    }
    fn release_session(&mut self, request: u64, session: Session, now: f64) {
        for name in self
            .session_blocks
            .remove(&(request, session))
            .unwrap_or_default()
        {
            if let Some(block) = self.blocks.get_mut(&name) {
                block
                    .sessions
                    .retain(|&member| member != (session, request));
            }
        }
        if self.detailed_observations {
            let session = &*self.session_names.names[session as usize];
            self.observations.push(serde_json::json!({"event":"session_references_released","request_id":request,"session_id":session,"time":now}));
        }
    }
    fn child_request(&self, request: u64, index: usize, child: u64, now: f64) -> Request {
        let mut sub = self.active[&request].request.sub_requests[index].clone();
        let restored = self.active[&request]
            .calls
            .get(&index)
            .map_or(0, |state| state.restored_blocks);
        if let SubRequest::Llm {
            known_tokens,
            kv_tokens,
            input_tokens,
            interval,
            next,
            kv_prefix,
            ..
        } = &mut sub
        {
            *known_tokens = Some(*input_tokens);
            *kv_tokens = Some(restored as u32 * self.block_tokens);
            *interval = None;
            next.clear();
            // The pool owns prefix identity; the worker only executes the call.
            *kv_prefix = None;
        }
        Request {
            request_id: child,
            arrival_time: now,
            initial: vec![0],
            sub_requests: vec![sub],
        }
    }
    fn finish(
        &mut self,
        request: u64,
        index: usize,
        now: f64,
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<()> {
        if let Some(pressure) = self.pressure_state.remove(&(request, index)) {
            if let Some(offset) = pressure.offset {
                let node = self.host_index(&pressure.config.node).unwrap();
                self.stores
                    .get_mut(&StoreId::Host(node))
                    .unwrap()
                    .release(offset, pressure.config.bytes);
            }
        }
        self.store_completed(now, current, graph)?;
        if let Some(worker) = self.routed.remove(&(request, index)) {
            self.inflight_calls[worker] -= 1;
        }
        let active = self.active.get_mut(&request).unwrap();
        ensure!(!active.completed[index], "duplicate subrequest completion");
        active.completed[index] = true;
        if active.completed.iter().any(|done| !*done) {
            return Ok(());
        }
        let active = self.active.remove(&request).unwrap();
        ensure!(
            active.session_calls.values().all(|count| *count == 0),
            "session completed with outstanding node membership"
        );
        let mut sessions = active.session_calls.keys().copied().collect::<Vec<_>>();
        sessions.sort_by_key(|&session| &self.session_names.names[session as usize]);
        for session in sessions {
            self.release_session(request, session, now);
        }
        let result = RequestResult {
            request_id: request,
            arrival_time: active.request.arrival_time,
            initial: active.request.initial.clone(),
            sub_requests: active.results.into_iter().map(|r| *r.unwrap()).collect(),
        };
        if self.detailed_observations {
            self.observations.push(
                serde_json::json!({"event":"session_complete","request_id":request,"time":now}),
            );
        }
        if let Some(sink) = active.sink {
            let event = graph.add_logical_node(
                sink,
                SystemEvent::RequestCompleted {
                    request: active.request,
                    result,
                },
            );
            graph.add_edge(current, event)?;
        } else {
            self.results.push(result);
        }
        Ok(())
    }
}

impl LogicalHandler for CxlPoolSystem {
    fn handle(
        &mut self,
        payload: Box<dyn Any>,
        now: f64,
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<f64> {
        let payload = match payload.downcast::<Event>() {
            Ok(event) => *event,
            Err(payload) => {
                let event = *payload
                    .downcast::<SystemEvent>()
                    .map_err(|_| anyhow!("unexpected cxl_pool payload"))?;
                let (child, result) = match event {
                    SystemEvent::KvProgress {
                        request_id,
                        computed_tokens,
                    } => {
                        if let Some(&(request, index)) = self.children.get(&request_id) {
                            self.publish(request, index, computed_tokens, now, current, graph)?;
                        }
                        return Ok(0.0);
                    }
                    SystemEvent::KvPreempted { request_id } => {
                        let (request, index) = self.children[&request_id];
                        let state = self
                            .active
                            .get_mut(&request)
                            .unwrap()
                            .calls
                            .get_mut(&index)
                            .unwrap();
                        state.preempted = true;
                        state.pending = true;
                        state.loaded = None;
                        state.published_blocks = 0;
                        state.checkpoint_source = None;
                        state.restored_blocks = 0;
                        state.binding.blocks = state.binding.completed_blocks.clone();
                        let owned_keys = std::mem::take(&mut state.owned_keys);
                        let worker = state.binding.worker;
                        self.release_execution(&owned_keys, request_id);
                        self.instances[worker].set_external_pending(request_id, true)?;
                        if self.detailed_observations {
                            self.observations.push(serde_json::json!({"event":"request_preempted","request_id":request,"subrequest_index":index,"time":now}));
                        }
                        return Ok(0.0);
                    }
                    SystemEvent::KvCapacity {
                        request_id,
                        required_bytes,
                        continuation,
                    } => {
                        self.ensure_capacity(
                            request_id,
                            required_bytes,
                            continuation,
                            now,
                            current,
                            graph,
                        )?;
                        return Ok(0.0);
                    }
                    SystemEvent::KvLookup { request: child } => {
                        let (request, index) = self.children[&child];
                        let worker = self.child_placement[&child];
                        let state = self
                            .active
                            .get_mut(&request)
                            .unwrap()
                            .calls
                            .get_mut(&index)
                            .unwrap();
                        // Schedules prepared at one time may look up a call twice.
                        if !std::mem::take(&mut state.pending) {
                            return Ok(0.0);
                        }
                        state.wait_since = now;
                        state.admission_waiting = false;
                        self.instances[worker].set_external_blocked(child, true)?;
                        self.prepare(request, index, now, current, graph)?;
                        // A call that cannot fit keeps its turn: later ones wait.
                        if self.active[&request].calls[&index].admission_waiting {
                            self.instances[worker].set_external_admission_blocked(child, true)?;
                        }
                        return Ok(0.0);
                    }
                    SystemEvent::RequestCompleted { request, result } => (request, result),
                };
                let (request, index) = self
                    .children
                    .remove(&child.request_id)
                    .ok_or_else(|| anyhow!("unknown child completion"))?;
                let state = self
                    .active
                    .get_mut(&request)
                    .unwrap()
                    .calls
                    .remove(&index)
                    .unwrap();
                self.release_execution(&state.owned_keys, child.request_id);
                // A reused prefix whose offload copy was evicted during the call
                // is written again; one that survived is not.
                for (index, key) in &state.owned_keys {
                    if *index < state.restored_blocks as u64 && self.blocks.contains_key(key) {
                        self.queue_store(*key)?;
                    }
                }
                self.child_placement.remove(&child.request_id);
                let mut child_result = result.sub_requests.into_iter().next().unwrap();
                if let ResultSubRequest::Llm {
                    next,
                    interval,
                    model,
                    kv_prefix,
                    ttft_parts,
                    reused_tokens,
                    ..
                } = &mut child_result
                {
                    if let Some(parts) = ttft_parts {
                        // The child queued from release: its queue time
                        // includes loading its prefix and waiting for HBM.
                        parts.kv_load = state.preparation_wait;
                        parts.admission = state.admission_wait;
                        parts.queue = (parts.queue - parts.kv_load - parts.admission).max(0.0);
                        self.first_token_waits.0 += 1;
                        self.first_token_waits.1 += parts.kv_load + parts.admission + parts.queue;
                    }
                    *reused_tokens = state.reused_tokens;
                    let original = &self.active[&request].request.sub_requests[index];
                    *next = original.next().to_vec();
                    *interval = original.interval();
                    *model = original.model().map(str::to_owned);
                    *kv_prefix = original.kv_prefix().cloned();
                }
                self.active.get_mut(&request).unwrap().results[index] =
                    Some(Box::new(child_result));
                self.seed(&state.binding, request, now)?;
                self.store_completed(now, current, graph)?;
                graph.add_edge(current, self.active[&request].completion[index])?;
                return Ok(0.0);
            }
        };
        let module = self as *mut Self as *mut dyn LogicalHandler;
        match payload {
            Event::Arrival { request, index } => {
                let detached = graph.pop_edges(current)?;
                ensure!(
                    detached == vec![self.active[&request].completion[index]],
                    "invalid call completion edge"
                );
                let release = graph.add_logical_node(module, Event::Release { request, index });
                graph.add_edge(current, release)?;
                Ok(self.active[&request].request.sub_requests[index]
                    .interval()
                    .unwrap_or(0.0))
            }
            Event::Release { request, index } => {
                for pressure in self
                    .pressures
                    .iter()
                    .filter(|p| p.request_id == request && p.subrequest_index == index)
                    .cloned()
                    .collect::<Vec<_>>()
                {
                    ensure!(
                        pressure.owner_request_id != request,
                        "pressure must identify another session"
                    );
                    ensure!(
                        self.seen_requests.contains(&pressure.owner_request_id),
                        "pressure owner is not a workload session"
                    );
                    let node = pressure.node.clone();
                    self.pressure_state.insert(
                        (request, index),
                        PressureState {
                            config: pressure,
                            offset: None,
                        },
                    );
                    self.spill(self.host_index(&node).unwrap(), now, current, graph)?;
                }
                for background in self
                    .background
                    .iter()
                    .filter(|b| b.request_id == request && b.subrequest_index == index)
                    .cloned()
                    .collect::<Vec<_>>()
                {
                    let src = self.fabric().endpoint(&background.src)?;
                    let dst = self.fabric().endpoint(&background.dst)?;
                    self.unfinished_transfers += 1;
                    let totals = self.transfer_totals.entry("background".into()).or_default();
                    totals.0 += 1;
                    totals.1 += background.bytes;
                    let transfer = self.detailed_observations.then(|| {
                        let index = self.transfers.len();
                        self.transfers.push(Transfer {
                            kind: "background".into(),
                            block: None,
                            src_owner: None,
                            dst_owner: None,
                            route_src: src,
                            route_dst: dst,
                            src: background.src,
                            dst: background.dst,
                            bytes: background.bytes,
                            issue: now,
                            start: now,
                            completion: None,
                        });
                        index
                    });
                    let net = graph.add_network_node(
                        &mut self.fabric.as_mut().unwrap().flow,
                        src,
                        dst,
                        background.bytes,
                    );
                    let done =
                        graph.add_logical_node(module, Event::BackgroundComplete { transfer });
                    graph.add_edge(current, net)?;
                    graph.add_edge(net, done)?;
                }
                let sub = self.active[&request].request.sub_requests[index].clone();
                match sub {
                    SubRequest::ToolCall {
                        input_tokens,
                        output_tokens,
                        next,
                        duration,
                        interval,
                        session_id,
                    } => {
                        self.active.get_mut(&request).unwrap().results[index] =
                            Some(Box::new(ResultSubRequest::ToolCall {
                                input_tokens,
                                output_tokens,
                                next,
                                duration: duration.unwrap_or(0.0),
                                interval,
                                session_id,
                            }));
                        let tool = graph.add_logical_node(
                            module,
                            Event::ToolRun {
                                duration: duration.unwrap_or(0.0),
                            },
                        );
                        graph.add_edge(current, tool)?;
                        graph.add_edge(tool, self.active[&request].completion[index])?;
                        Ok(0.0)
                    }
                    SubRequest::Llm { .. } => {
                        let mut binding = self.binding(request, index);
                        if binding.automatic {
                            let blocks = binding.blocks.clone();
                            binding.worker = self.route(request, index, &blocks, now)?;
                        }
                        let child = self.next_child;
                        self.next_child += 1;
                        let child_request = self.child_request(request, index, child, now);
                        let worker = binding.worker;
                        self.children.insert(child, (request, index));
                        self.child_placement.insert(child, worker);
                        self.active.get_mut(&request).unwrap().calls.insert(
                            index,
                            CallState {
                                binding,
                                child,
                                reads: VecDeque::new(),
                                inflight: 0,
                                released: now,
                                ready: None,
                                pending: true,
                                loaded: None,
                                load_started: now,
                                restored_blocks: 0,
                                reused_tokens: None,
                                published_blocks: 0,
                                owned_keys: BTreeSet::new(),
                                checkpoint_source: None,
                                wait_since: now,
                                admission_wait: 0.0,
                                preparation_wait: 0.0,
                                admission_waiting: false,
                                admission_barrier: None,
                                preempted: false,
                            },
                        );
                        // As in vLLM, the call waits in its worker's queue and
                        // looks up and loads its prefix at its turn.
                        self.instances[worker].set_external_pending(child, true)?;
                        self.instances[worker].add_request_arrival(
                            child_request,
                            graph,
                            Some(current),
                            Some(module),
                        )?;
                        Ok(0.0)
                    }
                }
            }
            Event::Prepare { request, index } => {
                self.prepare(request, index, now, current, graph)?;
                if let Some(state) = self.active[&request].calls.get(&index) {
                    if !state.admission_waiting {
                        let worker = state.binding.worker;
                        self.instances[worker].wake(current, graph)?;
                    }
                }
                Ok(0.0)
            }
            Event::Loaded { request, index } => {
                self.finish_load(request, index, now)?;
                let worker = self.active[&request].calls[&index].binding.worker;
                self.instances[worker].wake(current, graph)?;
                Ok(0.0)
            }
            Event::MoveComplete {
                key,
                transfer,
                call,
                source,
                release_source,
            } => {
                let block = self.blocks.get_mut(&key).unwrap();
                let (target, complete) = *block
                    .pending
                    .take()
                    .ok_or_else(|| anyhow!("completion missing pending block"))?;
                self.stores
                    .get_mut(&source.store())
                    .unwrap()
                    .pending
                    .remove(&complete);
                self.stores
                    .get_mut(&target.store())
                    .unwrap()
                    .pending
                    .remove(&complete);
                // An HBM copy or a spill's new home becomes the primary location.
                if matches!(target, Location::Hbm { .. })
                    || release_source && block.location == source
                {
                    let previous = std::mem::replace(&mut block.location, target);
                    block.add_replica(previous);
                } else {
                    block.add_replica(target);
                }
                ensure!(
                    Self::locations(block)
                        .filter(|p| !matches!(p, Location::Hbm { .. }))
                        .count()
                        <= 1 + usize::from(release_source),
                    "a block keeps one offload copy"
                );
                if release_source {
                    let bytes = block.bytes;
                    self.discard_copy(&key, &source.store(), now);
                    if matches!(source, Location::Host { .. }) {
                        self.host_bytes_freed += bytes;
                    }
                }
                self.refresh(&key, true);
                if let Some(transfer) = transfer {
                    self.transfers[transfer].completion = Some(now);
                }
                self.unfinished_transfers -= 1;
                self.sync_hbm()?;
                self.admit_pressure(now)?;
                if let Some((request, index)) = call {
                    let state = self
                        .active
                        .get_mut(&request)
                        .unwrap()
                        .calls
                        .get_mut(&index)
                        .unwrap();
                    match state.reads.pop_front() {
                        Some(next) => self.issue_read(request, index, next, now, current, graph)?,
                        None => state.inflight -= 1,
                    }
                }
                self.store_completed(now, current, graph)?;
                Ok(0.0)
            }
            Event::Capacity {
                child,
                required_bytes,
                continuation,
            } => {
                self.ensure_capacity(child, required_bytes, continuation, now, current, graph)?;
                Ok(0.0)
            }
            Event::BackgroundComplete { transfer } => {
                if let Some(transfer) = transfer {
                    self.transfers[transfer].completion = Some(now);
                }
                self.unfinished_transfers -= 1;
                Ok(0.0)
            }
            Event::AdmissionReleased => Ok(0.0),
            Event::ToolRun { duration } => Ok(duration),
            Event::Complete { request, index } => {
                self.close_session_call(request, index, now);
                self.finish(request, index, now, current, graph)?;
                Ok(0.0)
            }
        }
    }
}

impl Module for CxlPoolSystem {
    fn resolve_path(&mut self, base: &Path) -> Result<()> {
        for worker in &mut self.instances {
            worker.resolve_path(base)?;
        }
        Ok(())
    }
    fn observe(&self) -> serde_json::Value {
        let blocks = self.detailed_observations.then(|| {
            let mut blocks = self.blocks.iter().map(|(&name, b)| (self.block_name(name), name, b)).collect::<Vec<_>>();
            blocks.sort_by(|a, b| a.0.cmp(&b.0));
            blocks.into_iter().map(|(key,name,b)|{
                let mut sessions = b.sessions.iter().map(|&(session, request)| (&*self.session_names.names[session as usize], request)).collect::<Vec<_>>();
                let mut pins = b.pins.iter().map(|&(pin, _)| pin).collect::<Vec<_>>();
                pins.sort();
                sessions.sort();
                serde_json::json!({"key":key,"bytes":b.bytes,"kind":if name & 1 == 1 {"recurrent_state"} else {"attention"},"checkpoint_tokens":(name & 1 == 1).then_some((b.index + 1) * u64::from(self.block_tokens)),"location":self.observe_location(&b.location,b.bytes),"ready":true,"replicas":Self::locations(b).map(|p|self.observe_location(p,b.bytes)).collect::<Vec<_>>(),"sessions":sessions,"source_valid":true,"pending_ready":false,"pending":b.pending.as_deref().map(|(p,_)|self.observe_location(p,b.bytes)),"pins":pins})}).collect::<Vec<_>>()
        });
        let hosts=self.host_names.iter().enumerate().map(|(index,node)|serde_json::json!({"node":node,"capacity_bytes":self.host_capacity_bytes[node],"block_bytes":self.used(&Location::Host{node:index as u32,worker:0,offset:0}).saturating_sub(self.pressure_state.values().filter(|p|p.config.node==*node&&p.offset.is_some()).map(|p|p.config.bytes).sum::<u64>()),"pressure_requested_bytes":self.pressure_requested(node),"pressure_reserved_bytes":self.pressure_state.values().filter(|p|p.config.node==*node&&p.offset.is_some()).map(|p|p.config.bytes).sum::<u64>()})).collect::<Vec<_>>();
        let pools=self.placement_devices.iter().enumerate().map(|(index,device)|serde_json::json!({"device":device,"occupied_or_reserved_bytes":self.used(&Location::Pool{device:index as u32,offset:0})})).collect::<Vec<_>>();
        let hbm=(0..self.instances.len()).map(|worker|serde_json::json!({"worker":worker,"occupied_or_reserved_bytes":self.used(&Location::Hbm{worker,offset:0})})).collect::<Vec<_>>();
        serde_json::json!({"cxl_pool":{"hosts":hosts,"pool_usage":pools,"hbm_usage":hbm,"block_count":self.blocks.len(),"transfer_totals":self.transfer_totals,"detailed_observations":self.detailed_observations,"blocks":blocks,"transfers":self.detailed_observations.then_some(&self.transfers),"events":self.detailed_observations.then_some(&self.observations),"host_bytes_freed":self.host_bytes_freed,"unfinished_requests":self.active.len(),"unfinished_transfers":self.unfinished_transfers,"topology":self.detailed_observations.then(|| self.fabric.as_ref().map(TopologyNetwork::observe))},"instances":self.instances.iter().map(|s|s.observe()).collect::<Vec<_>>()})
    }
}
impl System for CxlPoolSystem {
    fn first_token_waits(&self) -> Option<(u64, f64)> {
        Some(self.first_token_waits)
    }
    fn init(&mut self) -> Result<()> {
        ensure!(!self.instances.is_empty(), "cxl_pool needs workers");
        self.inflight_calls = vec![0; self.instances.len()];
        ensure!(
            self.block_tokens > 0 && self.stripe_blocks > 0 && self.max_outstanding_reads > 0,
            "block size, stripe and read concurrency must be positive"
        );
        ensure!(
            self.placement_devices.iter().collect::<BTreeSet<_>>().len()
                == self.placement_devices.len(),
            "duplicate placement device"
        );
        self.fabric = Some(self.topology.build()?);
        for device in &self.placement_devices {
            self.fabric().memory_capacity(device)?;
        }
        for (worker, instance) in self.instances.iter_mut().enumerate() {
            instance.init()?;
            let memory = instance.configure_external_kv()?;
            ensure!(
                memory.block_tokens > 0,
                "scheduler KV block size must be positive"
            );
            self.fabric.as_ref().unwrap().worker(worker)?;
            self.memory.push(memory);
        }
        self.host_names = self.host_capacity_bytes.keys().cloned().collect();
        for (index, &capacity) in self.host_capacity_bytes.values().enumerate() {
            self.stores
                .insert(StoreId::Host(index as u32), Store::new(capacity));
        }
        for (index, device) in self.placement_devices.iter().enumerate() {
            self.stores.insert(
                StoreId::Pool(index as u32),
                Store::new(self.fabric().memory_capacity(device)?),
            );
        }
        for (worker, m) in self.memory.iter().enumerate() {
            let geometry = (
                m.kv_bytes_per_token,
                m.kv_bytes_per_request,
                m.compatibility,
            );
            let id = match self.geometries.iter().position(|g| *g == geometry) {
                Some(id) => id,
                None => {
                    self.geometries.push(geometry);
                    self.geometries.len() - 1
                }
            };
            self.worker_geometry.push(id as u32);
            self.stores
                .insert(StoreId::Hbm(worker), Store::new(m.capacity_bytes));
        }
        self.binding_index = self
            .calls
            .iter()
            .enumerate()
            .map(|(index, call)| ((call.request_id, call.subrequest_index), index))
            .collect();
        let mut prefix_identity = BTreeMap::new();
        for binding in &self.calls {
            ensure!(binding.worker < self.instances.len(), "invalid call worker");
            for prefix in [
                &binding.blocks,
                &binding.completed_blocks,
                &binding.seed_host_blocks,
            ] {
                for (index, key) in prefix.iter().enumerate() {
                    let identity = (
                        index,
                        index
                            .checked_sub(1)
                            .map(|previous| prefix[previous].as_str()),
                    );
                    if let Some(previous) = prefix_identity.insert(key.as_str(), identity) {
                        ensure!(
                            previous == identity,
                            "incompatible prefix order for block {key}"
                        );
                    }
                }
            }
            for key in binding
                .blocks
                .iter()
                .chain(&binding.completed_blocks)
                .chain(&binding.seed_host_blocks)
            {
                ensure!(
                    key.split_once(':')
                        .is_some_and(|(scope, id)| !scope.is_empty() && !id.is_empty()),
                    "prefix keys require an explicit compatibility scope: {key}"
                );
            }
        }
        ensure!(
            self.calls
                .iter()
                .map(|b| (b.request_id, b.subrequest_index))
                .collect::<BTreeSet<_>>()
                .len()
                == self.calls.len(),
            "duplicate call binding"
        );
        for call in &self.calls {
            let mut intern = |names: &[String]| {
                names
                    .iter()
                    .map(|name| self.block_names.intern(name) << 1)
                    .collect::<Vec<_>>()
            };
            self.configured.push(CallBinding {
                automatic: false,
                subrequest_index: call.subrequest_index,
                worker: call.worker,
                blocks: intern(&call.blocks),
                completed_blocks: intern(&call.completed_blocks),
                seed_host_blocks: intern(&call.seed_host_blocks),
            });
        }
        ensure!(
            self.pressures
                .iter()
                .map(|p| (p.request_id, p.subrequest_index))
                .collect::<BTreeSet<_>>()
                .len()
                == self.pressures.len(),
            "duplicate pressure trigger"
        );
        for p in &self.pressures {
            ensure!(p.bytes > 0, "pressure bytes must be positive");
            ensure!(
                self.host_capacity_bytes.contains_key(&p.node),
                "unknown pressure host"
            );
        }
        Ok(())
    }
    fn instance_load(&self) -> InstanceLoad {
        self.instances.iter().map(|s| s.instance_load()).sum()
    }
    fn add_request_arrival(
        &mut self,
        request: Request,
        graph: &mut Graph,
        parent: Option<NodeID>,
        sink: Option<*mut dyn LogicalHandler>,
    ) -> Result<()> {
        request.validate()?;
        let id = request.request_id;
        ensure!(
            self.seen_requests.insert(id),
            "duplicate cxl_pool request {id}"
        );
        for (index, sub) in request.sub_requests.iter().enumerate() {
            if let SubRequest::Llm {
                input_tokens,
                output_tokens,
                ..
            } = sub
            {
                let binding = self.binding(id, index);
                ensure!(
                    binding.blocks.iter().collect::<BTreeSet<_>>().len() == binding.blocks.len(),
                    "duplicate mandatory block"
                );
                ensure!(
                    (binding.blocks.len() as u64) * u64::from(self.block_tokens)
                        <= u64::from(input_tokens.saturating_sub(1)),
                    "restored prefix must leave an uncomputed logits token"
                );
                for keys in [&binding.completed_blocks, &binding.seed_host_blocks] {
                    ensure!(
                        (keys.len() as u64) * u64::from(self.block_tokens)
                            <= u64::from(input_tokens + output_tokens.saturating_sub(1)),
                        "published prefix exceeds computed tokens"
                    );
                }
            }
        }
        let module = self as *mut Self as *mut dyn LogicalHandler;
        let mut arrival = vec![];
        let mut complete = vec![];
        for index in 0..request.sub_requests.len() {
            arrival.push(graph.add_logical_node_at(
                module,
                Event::Arrival { request: id, index },
                request.arrival_time,
            ));
            complete.push(graph.add_logical_node(module, Event::Complete { request: id, index }));
        }
        for &index in &request.initial {
            if let Some(parent) = parent {
                graph.add_edge(parent, arrival[index])?;
            }
        }
        for (index, sub) in request.sub_requests.iter().enumerate() {
            graph.add_edge(arrival[index], complete[index])?;
            for &next in sub.next() {
                graph.add_edge(complete[index], arrival[next])?;
            }
        }
        let sessions = request
            .sub_requests
            .iter()
            .map(|sub| match sub.session_id() {
                Some(session) => self.session_names.intern(session),
                None => self.session_names.intern(&format!("request:{id}")),
            })
            .collect::<Vec<_>>();
        let mut session_calls = BTreeMap::new();
        for &session in &sessions {
            *session_calls.entry(session).or_default() += 1;
        }
        let conservative_sessions = request.sub_requests.iter().any(|sub| {
            matches!(
                sub,
                SubRequest::ToolCall {
                    session_id: None,
                    ..
                }
            )
        });
        self.active.insert(
            id,
            Active {
                results: vec![None; request.sub_requests.len()],
                completed: vec![false; request.sub_requests.len()],
                request,
                sink,
                completion: complete,
                calls: BTreeMap::new(),
                sessions,
                session_calls,
                conservative_sessions,
            },
        );
        Ok(())
    }
    fn into_results(self: Box<Self>) -> Result<Vec<RequestResult>> {
        ensure!(
            self.active.is_empty() && self.children.is_empty(),
            "cxl_pool finished with incomplete requests"
        );
        ensure!(
            self.blocks.values().all(|block| block.pending.is_none()
                && block.pins.is_empty()
                && block.sessions.is_empty()),
            "cxl_pool finished with moving, pinned, or session-owned blocks"
        );
        ensure!(
            self.session_blocks.is_empty(),
            "cxl_pool finished with session membership index entries"
        );
        ensure!(
            self.queued_stores.is_empty(),
            "cxl_pool finished with queued KV stores"
        );
        ensure!(
            self.unfinished_transfers == 0,
            "cxl_pool finished with incomplete transfers"
        );
        Ok(self.results)
    }
}

#[cfg(test)]
mod storage_tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn least_loaded_balances_bursts_and_avoids_kv_bound_queues() {
        // Empty snapshot: the in-flight floor and rotating start spread a burst.
        let empty = vec![(0.0, 0.0, 0.0); 2];
        assert_eq!(least_loaded_choice(&empty, &[0, 0], 0), 0);
        assert_eq!(least_loaded_choice(&empty, &[1, 0], 1), 1);
        assert_eq!(least_loaded_choice(&empty, &[1, 1], 0), 0);
        // 2 + 2 + 2·6·(1.0 − 0.5) = 10 loses to 3 + 2 at 40% usage.
        let pressured = vec![(2.0, 2.0, 1.0), (3.0, 2.0, 0.4)];
        assert_eq!(least_loaded_choice(&pressured, &[0, 0], 0), 1);
        // Without waiting, KV usage does not matter.
        let idle = vec![(0.0, 3.0, 1.0), (0.0, 4.0, 0.0)];
        assert_eq!(least_loaded_choice(&idle, &[0, 0], 1), 0);
    }

    #[test]
    fn fragmented_storage_uses_disjoint_extents_and_coalesces_release() {
        let mut store = Store::new(16);
        let handles = (0..4)
            .map(|_| store.reserve(4).unwrap())
            .collect::<Vec<_>>();
        store.release(handles[0], 4);
        store.release(handles[2], 4);
        let combined = store.reserve(8).unwrap();
        assert_eq!(*store.allocations[&combined], [(0, 4), (8, 4)]);
        // Contiguous allocations keep no extent record.
        assert_eq!(store.allocations.len(), 1);
        assert_eq!(store.used, 16);
        assert!(store.reserve(1).is_none());
        store.release(combined, 8);
        store.release(handles[1], 4);
        store.release(handles[3], 4);
        assert_eq!(store.free.into_iter().collect::<Vec<_>>(), vec![(0, 16)]);
    }
    fn directory_with_pool_source() -> (CxlPoolSystem, Name) {
        let mut system: CxlPoolSystem = serde_json::from_value(json!({
            "instances": [], "topology": {"endpoints": [{"id":"D", "kind":"memory", "capacity_bytes":4, "read_bandwidth":1e6,"write_bandwidth":1e6}, {"id":"S","kind":"switch","lane_budget":8}], "links":[{"id":"DS","a":"D","b":"S","a_port":"p","b_port":"p","width":8,"bandwidth_ab":1e6,"bandwidth_ba":1e6,"latency":0.0}]},
            "block_tokens":1, "host_capacity_bytes":{"host":4}, "placement_devices":["D"]
        })).unwrap();
        system.fabric = Some(system.topology.build().unwrap());
        system.host_names = vec!["host".into()];
        system.stores.insert(StoreId::Pool(0), Store::new(4));
        system.stores.insert(StoreId::Host(0), Store::new(4));
        let location = system
            .allocate(
                Location::Pool {
                    device: 0,
                    offset: 0,
                },
                4,
            )
            .unwrap();
        let key = system.block_name_id("fixture:restore-source");
        let session = system.session_names.intern("session");
        let mut block = Block {
            index: 0,
            sessions: SmallSet::default(),
            bytes: 4,
            geometry: 0,
            location,
            replicas: Box::default(),
            ordinal: 0,
            pending: None,
            pins: SmallSet::default(),
        };
        block.sessions.insert((session, 7));
        system.blocks.insert(key, block);
        system
            .session_blocks
            .insert((7, session), BTreeSet::from([key]));
        system.refresh(&key, false);
        (system, key)
    }

    #[test]
    fn last_copy_eviction_removes_session_membership_index() {
        let (mut system, key) = directory_with_pool_source();
        system.discard_copy(&key, &StoreId::Pool(0), 0.0);
        assert_eq!(system.blocks.len(), 0);
        assert!(system.session_blocks.is_empty());
        assert_eq!(system.stores[&StoreId::Pool(0)].used, 0);
        assert!(system.observations.is_empty());
    }

    #[test]
    fn pool_pressure_cannot_delete_last_pinned_restore_source() {
        let (mut system, key) = directory_with_pool_source();
        let host = system
            .allocate(
                Location::Host {
                    node: 0,
                    worker: 0,
                    offset: 0,
                },
                4,
            )
            .unwrap();
        let block = system.blocks.get_mut(&key).unwrap();
        block.add_replica(host);
        block.pins.insert((99, 0)); // Consumer awaits HBM capacity before issuing its reload.
        system.refresh(&key, false);
        assert_eq!(system.stores[&StoreId::Pool(0)].reclaimable_bytes, 4);
        system.discard_copy(&key, &StoreId::Host(0), 0.0);
        assert!(system.pool_target(4, 1.0).unwrap().is_none());
        assert!(system.blocks.contains_key(&key));
        system.blocks.get_mut(&key).unwrap().pins.retain(|_| false);
        system.refresh(&key, false);
        assert!(system.pool_target(4, 2.0).unwrap().is_some());
        assert_eq!(system.blocks.len(), 0);
    }

    #[test]
    fn last_pinned_host_copy_remains_eligible_for_safe_spill() {
        let (mut system, key) = directory_with_pool_source();
        let host = system
            .allocate(
                Location::Host {
                    node: 0,
                    worker: 0,
                    offset: 0,
                },
                4,
            )
            .unwrap();
        system.blocks.get_mut(&key).unwrap().add_replica(host);
        system.blocks.get_mut(&key).unwrap().pins.insert((99, 0));
        system.discard_copy(&key, &StoreId::Pool(0), 0.0);
        assert_eq!(system.stores[&StoreId::Host(0)].reclaimable_bytes, 4);
    }
}
