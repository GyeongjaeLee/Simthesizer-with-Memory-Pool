use std::collections::{HashMap, HashSet, LinkedList};
use std::hash::{Hash, Hasher};

use anyhow::{anyhow, bail, Result};
use serde::Deserialize;

use crate::module::Module;
use crate::scheduler::{KvMemoryInfo, Scheduler};
use crate::schedulers::prefix_cache::{hash_block, PrefixCache};
use crate::types::{
    InstanceLoad, NodeID, ParallelStrategy, ReadySubRequest, RequestState, SubRequestCompletion,
};

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ChunkedPrefillScheduler {
    block_size: u32,
    max_batch: usize,
    max_num_batched_tokens: u32,
    device_mem: u64,
    device_mem_utilization: f64,
    /// Per-GPU workspace (activations, graphs, communication buffers).
    #[serde(default)]
    reserved_device_bytes: u64,
    #[serde(default = "default_true")]
    enable_prefix_caching: bool,
    #[serde(default)]
    num_lookahead_tokens: u32,
    #[serde(default = "default_true")]
    scheduler_reserve_full_isl: bool,
    /// vLLM `max_num_inflight_kv_loads`: at most this many requests hold KV
    /// for an external load before they run, and loads may start without a
    /// free running slot or token budget. None: no cap; loads start only
    /// while both remain.
    #[serde(default)]
    max_num_inflight_kv_loads: Option<usize>,
    #[serde(skip)]
    preemptions_total: u64,
    /// Start of the iteration in flight.
    #[serde(skip)]
    iteration_start: f64,
    #[serde(skip)]
    progress: Vec<(u64, u32)>,
    #[serde(skip)]
    external_preemptions: Vec<u64>,
    #[serde(skip)]
    admission_waiting: HashSet<NodeID>,
    #[serde(skip)]
    external_blocked: HashSet<u64>,
    #[serde(skip)]
    external_admission: HashSet<u64>,
    #[serde(skip)]
    external_pending: HashSet<u64>,
    #[serde(skip)]
    next_cache_nonce: u64,
    #[serde(skip)]
    external_reclaim: Option<(u64, u64)>,

    #[serde(skip)]
    external_kv_enabled: bool,
    #[serde(skip)]
    external_kv_bytes: u64,
    #[serde(skip)]
    external_kv_reclaimable_bytes: u64,
    #[serde(skip)]
    external_prefixes: HashMap<u64, u32>,
    /// Admission bytes held for requests whose external prefix is loading or
    /// loaded but which are not yet admitted.
    #[serde(skip)]
    external_admissions: HashMap<u64, u64>,

    #[serde(skip_deserializing, default)]
    queue: LinkedList<RunningRequest>,

    #[serde(skip_deserializing, default)]
    running: bool,

    #[serde(skip_deserializing, default)]
    current_batch: Vec<RequestState>,

    #[serde(skip_deserializing, default)]
    admitted: HashSet<NodeID>,

    #[serde(skip_deserializing, default)]
    generation_tokens_total: u64,

    #[serde(skip_deserializing, default)]
    model_weight_bytes_per_device: u64,

    #[serde(skip_deserializing, default)]
    kv_bytes_per_token_per_device: u64,

    #[serde(skip_deserializing, default)]
    kv_bytes_per_request_per_device: u64,

    #[serde(skip_deserializing, default)]
    usable_device_mem: u64,

    #[serde(skip_deserializing, default)]
    prefix_cache: PrefixCache,

    #[serde(skip_deserializing, default)]
    effective_max_batch: usize,

    #[serde(skip_deserializing, default)]
    effective_max_num_batched_tokens: u32,
}

fn default_true() -> bool {
    true
}

crate::register_scheduler!(ChunkedPrefillScheduler, "chunked_prefill");

impl Module for ChunkedPrefillScheduler {
    fn observe(&self) -> serde_json::Value {
        // An external HBM wait outranks the KV lookup it holds its turn for.
        let waiting = self.queue.iter().skip(self.current_batch.len());
        let admission = waiting
            .clone()
            .filter(|r| {
                self.external_admission.contains(&r.request_id)
                    || !self.external_blocked.contains(&r.request_id)
                        && self.admission_waiting.contains(&r.completion_node_id)
            })
            .count();
        let kv_waiting = waiting
            .clone()
            .filter(|r| {
                self.external_blocked.contains(&r.request_id)
                    && !self.external_admission.contains(&r.request_id)
            })
            .count();

        serde_json::json!({
            "running_requests": self.admitted.len(),
            "waiting_requests": self.queue.len() - self.admitted.len(),
            "generation_tokens_total": self.generation_tokens_total,
            "hbm_allocated_bytes": self.hbm_bytes(),
            "prefix_cache_bytes": self.prefix_cache.bytes_used(self.block_size, self.kv_bytes_per_token_per_device),
            "preemptions_total": self.preemptions_total,
            "hbm_admission_waiting_requests": admission,
            "kv_preparation_waiting_requests": kv_waiting,
            "scheduler_waiting_requests": self.queue.len().saturating_sub(self.current_batch.len()).saturating_sub(admission + kv_waiting),
            "iteration_active": self.running,
        })
    }
}

#[derive(Debug, Clone)]
struct RunningRequest {
    request_id: u64,
    subrequest_index: usize,
    completion_node_id: NodeID,
    arrival_time: f64,
    input_tokens: u32,
    output_tokens: u32,

    known_tokens: u32,
    shared_kv_tokens: u32,
    private_kv_tokens: u32,
    external_kv_tokens: u32,
    last_time: f64,
    token_latency: Vec<f64>,
    first_scheduled: Option<f64>,

    input_token_ids: Option<Vec<u32>>,
    output_token_ids: Option<Vec<u32>>,
    block_hashes: Vec<u64>,
    kv_prefix: Option<crate::types::KvPrefix>,
    private_capacity_tokens: u64,
    cache_nonce: u64,
    state_copies: u64,
    align_state: bool,
    state_hash: Option<u64>,
    external_state_source: bool,
}

impl RunningRequest {
    fn new(request: ReadySubRequest) -> Self {
        Self {
            request_id: request.request_id,
            subrequest_index: request.subrequest_index,
            completion_node_id: request.completion_node_id,
            arrival_time: request.arrival_time,
            input_tokens: request.input_tokens,
            output_tokens: request.output_tokens,
            known_tokens: request.known_tokens,
            shared_kv_tokens: 0,
            private_kv_tokens: request.kv_tokens,
            external_kv_tokens: 0,
            last_time: request.arrival_time,
            token_latency: Vec::new(),
            first_scheduled: None,
            input_token_ids: request.input_token_ids,
            output_token_ids: request.output_token_ids,
            block_hashes: Vec::new(),
            kv_prefix: request.kv_prefix,
            private_capacity_tokens: u64::from(request.kv_tokens),
            cache_nonce: 0,
            state_copies: u64::from(request.kv_tokens > 0),
            align_state: false,
            state_hash: None,
            external_state_source: false,
        }
    }

    fn effective_kv_tokens(&self) -> u32 {
        self.shared_kv_tokens + self.private_kv_tokens
    }

    /// Tokens one step computes within a positive `budget`, ending at the
    /// replay checkpoint.
    fn next_chunk(&self, budget: u32, block_size: u32) -> u32 {
        let computed = self.effective_kv_tokens();
        let q_len = self.known_tokens.saturating_sub(computed).clamp(1, budget);
        let checkpoint = self.replay_checkpoint(block_size);
        if self.align_state && computed < checkpoint && computed + q_len > checkpoint {
            checkpoint - computed
        } else {
            q_len
        }
    }

    fn private_kv_bytes(
        &self,
        block_size: u32,
        kv_bytes_per_request_per_device: u64,
        kv_bytes_per_token_per_device: u64,
    ) -> u64 {
        let token_bytes = (u64::from(self.private_kv_tokens)
            .max(self.private_capacity_tokens)
            .div_ceil(u64::from(block_size))
            * u64::from(block_size))
        .saturating_sub(u64::from(self.external_kv_tokens))
            * kv_bytes_per_token_per_device;
        let request_bytes = self.state_copies * kv_bytes_per_request_per_device;
        token_bytes + request_bytes
    }

    fn private_kv_bytes_after(
        &self,
        q_len: u64,
        state_q_len: u64,
        block_size: u32,
        kv_bytes_per_request_per_device: u64,
        kv_bytes_per_token_per_device: u64,
    ) -> u64 {
        let total_kv_tokens = u64::from(self.private_kv_tokens) + q_len;
        let token_bytes = (total_kv_tokens.div_ceil(u64::from(block_size)) * u64::from(block_size))
            .saturating_sub(u64::from(self.external_kv_tokens))
            * kv_bytes_per_token_per_device;
        let request_bytes =
            self.state_copies_after(state_q_len, block_size) * kv_bytes_per_request_per_device;
        token_bytes + request_bytes
    }

    fn state_copies_after(&self, q_len: u64, block_size: u32) -> u64 {
        if !self.align_state {
            return 1;
        }
        let before = u64::from(self.effective_kv_tokens()).div_ceil(u64::from(block_size));
        let after = (u64::from(self.effective_kv_tokens()) + q_len).div_ceil(u64::from(block_size));
        1 + u64::from(self.state_copies > 0 && before > 0 && after > before)
    }

    fn replay_checkpoint(&self, block_size: u32) -> u32 {
        self.input_tokens.saturating_sub(1) / block_size * block_size
    }

    fn ensure_block_hashes(&mut self, block_size: u32, max_tokens: u32) {
        let bs = block_size as usize;
        let input = self.input_token_ids.as_deref().unwrap_or_default();
        let output = self.output_token_ids.as_deref().unwrap_or_default();
        for index in self.block_hashes.len()..(max_tokens / block_size) as usize {
            let parent = self.block_hashes.last().copied().unwrap_or(0);
            if let Some(prefix) = &self.kv_prefix {
                if block_size.is_multiple_of(prefix.block_tokens) {
                    let group = (block_size / prefix.block_tokens) as usize;
                    if let Some(chunk) = prefix.block_ids.get(index * group..(index + 1) * group) {
                        let mut hasher = std::collections::hash_map::DefaultHasher::new();
                        "proxy-prefix".hash(&mut hasher);
                        prefix.namespace.hash(&mut hasher);
                        block_size.hash(&mut hasher);
                        parent.hash(&mut hasher);
                        chunk.hash(&mut hasher);
                        self.block_hashes.push(hasher.finish());
                        continue;
                    }
                }
            } else if input.len() == self.input_tokens as usize
                && (index + 1) * bs <= input.len() + output.len()
            {
                let start = index * bs;
                let end = start + bs;
                let first = &input[start.min(input.len())..end.min(input.len())];
                let second =
                    &output[start.saturating_sub(input.len())..end.saturating_sub(input.len())];
                self.block_hashes.push(hash_block(parent, first, second));
                continue;
            }
            // Unknown content remains reusable only by this execution after preemption.
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            "opaque-kv".hash(&mut hasher);
            self.cache_nonce.hash(&mut hasher);
            parent.hash(&mut hasher);
            index.hash(&mut hasher);
            self.block_hashes.push(hasher.finish());
        }
    }

    fn longest_cache_hit(
        &mut self,
        prefix_cache: &mut PrefixCache,
        block_size: u32,
        recurrent: bool,
    ) -> u32 {
        if self.effective_kv_tokens() != 0 {
            return self.shared_kv_tokens;
        }

        // A cached prefix has KV, but no logits for the next token.
        let max_tokens = self.known_tokens.saturating_sub(1);
        self.ensure_block_hashes(block_size, max_tokens);
        let count = self
            .block_hashes
            .len()
            .min((max_tokens / block_size) as usize);
        let hit = prefix_cache.longest_hit(&self.block_hashes[..count]);
        let hit = if recurrent {
            prefix_cache.checkpoint_hit(&self.block_hashes, hit)
        } else {
            hit
        };
        hit as u32 * block_size
    }

    fn publish_shared_prefix(&mut self, prefix_cache: &mut PrefixCache, block_size: u32) {
        let total = self.effective_kv_tokens();
        self.ensure_block_hashes(block_size, total);
        let start = (self.shared_kv_tokens / block_size) as usize;
        let end = self.block_hashes.len().min((total / block_size) as usize);
        prefix_cache.insert(&self.block_hashes[start..end]);
        self.relock_shared_prefix(prefix_cache, block_size, end as u32 * block_size);
        self.private_capacity_tokens = self
            .private_capacity_tokens
            .saturating_sub(u64::from(self.shared_kv_tokens - start as u32 * block_size));
        self.private_kv_tokens = total - self.shared_kv_tokens;
    }

    fn relock_shared_prefix(
        &mut self,
        prefix_cache: &mut PrefixCache,
        block_size: u32,
        new_shared_kv_tokens: u32,
    ) {
        let old = (self.shared_kv_tokens / block_size) as usize;
        let new = (new_shared_kv_tokens / block_size) as usize;
        if new > old {
            prefix_cache.lock(&self.block_hashes[old..new]);
        } else if new < old {
            prefix_cache.unlock(&self.block_hashes[new..old]);
        }
        self.shared_kv_tokens = new_shared_kv_tokens;
    }

    fn release_state_source(&mut self, prefix_cache: &mut PrefixCache) {
        if let Some(hash) = self.state_hash.take() {
            prefix_cache.unlock(&[hash]);
        }
    }

    fn preempt(&mut self, prefix_cache: &mut PrefixCache, block_size: u32) {
        self.release_state_source(prefix_cache);
        self.external_state_source = false;
        self.relock_shared_prefix(prefix_cache, block_size, 0);
        self.private_kv_tokens = self.external_kv_tokens;
        self.private_capacity_tokens = u64::from(self.external_kv_tokens);
        self.state_copies = 0;
    }
}

impl ChunkedPrefillScheduler {
    fn private_bytes(&self, request: &RunningRequest) -> u64 {
        request.private_kv_bytes(
            self.block_size,
            self.kv_bytes_per_request_per_device,
            self.kv_bytes_per_token_per_device,
        )
    }

    fn private_bytes_after(&self, request: &RunningRequest, tokens: u64, state_tokens: u64) -> u64 {
        request.private_kv_bytes_after(
            tokens,
            state_tokens,
            self.block_size,
            self.kv_bytes_per_request_per_device,
            self.kv_bytes_per_token_per_device,
        )
    }

    fn hbm_bytes(&self) -> u64 {
        self.model_weight_bytes_per_device
            + self.external_kv_bytes
            + self
                .prefix_cache
                .bytes_used(self.block_size, self.kv_bytes_per_token_per_device)
            + self
                .queue
                .iter()
                .map(|r| self.private_bytes(r))
                .sum::<u64>()
    }

    fn cache_block_bytes(&self) -> u64 {
        u64::from(self.block_size) * self.kv_bytes_per_token_per_device
    }

    /// Waiting request whose external prefix is loading or loaded.
    fn holds_external_kv(&self, request: &RunningRequest) -> bool {
        self.external_admissions.contains_key(&request.request_id)
    }

    /// A running request waits for HBM; V1 likewise admits no waiting
    /// request after a preemption.
    fn running_wait(&self) -> bool {
        self.queue.iter().any(|r| {
            self.external_admission.contains(&r.request_id)
                && self.admitted.contains(&r.completion_node_id)
        })
    }
}

impl Scheduler for ChunkedPrefillScheduler {
    fn external_kv_budget(&self) -> Result<u64> {
        anyhow::ensure!(
            self.external_kv_enabled,
            "external KV ownership is not configured"
        );
        let private: u64 = self
            .queue
            .iter()
            .map(|request| self.private_bytes(request))
            .sum::<u64>()
            + self.external_admissions.values().sum::<u64>();
        Ok(self
            .usable_device_mem
            .saturating_sub(self.model_weight_bytes_per_device)
            .saturating_sub(private))
    }
    fn known_tokens(&self, request_id: u64) -> Option<u32> {
        self.queue
            .iter()
            .find(|r| r.request_id == request_id)
            .map(|r| r.known_tokens)
    }
    fn external_admission_bytes(&self, known_tokens: u32, restored_tokens: u32) -> u64 {
        // The admission check of `schedule` for a waiting request whose
        // private KV starts at its restored external prefix.
        let block = u64::from(self.block_size);
        let restored = u64::from(restored_tokens);
        let bytes = |tokens: u64| {
            ((restored + tokens).div_ceil(block) * block - restored)
                * self.kv_bytes_per_token_per_device
                + self.kv_bytes_per_request_per_device
        };
        let remaining = u64::from(known_tokens.saturating_sub(restored_tokens).max(1));
        let chunk = remaining.min(u64::from(self.effective_max_num_batched_tokens));
        let needed = bytes(chunk + u64::from(self.num_lookahead_tokens));
        if self.scheduler_reserve_full_isl {
            needed.max(bytes(remaining))
        } else {
            needed
        }
    }
    fn set_external_pending(&mut self, request_id: u64, pending: bool) {
        if pending {
            self.external_pending.insert(request_id);
        } else {
            self.external_pending.remove(&request_id);
        }
    }
    fn external_lookup(&self) -> Option<u64> {
        // V1 prices this step's running chunks, then admits requests holding
        // loaded KV. A lookup takes no slot or budget, but needs both left
        // unless loads may start without them, up to the in-flight limit.
        if self.running_wait() {
            return None;
        }
        let blocked = |r: &&RunningRequest| {
            self.external_blocked.contains(&r.request_id)
                || self.external_admission.contains(&r.request_id)
        };
        let (running, waiting): (Vec<_>, Vec<_>) = self
            .queue
            .iter()
            .partition(|r| self.admitted.contains(&r.completion_node_id));
        let mut slots = self.effective_max_batch.saturating_sub(running.len());
        let mut budget = self.effective_max_num_batched_tokens;
        let mut inflight = self.external_admissions.len();
        for request in &running {
            if budget > 0 && !blocked(request) {
                budget -= request.next_chunk(budget, self.block_size);
            }
        }
        for request in &waiting {
            if slots > 0 && budget > 0 && self.holds_external_kv(request) && !blocked(request) {
                slots -= 1;
                inflight -= 1;
                budget -= request.next_chunk(budget, self.block_size);
            }
        }
        let loads_only = slots == 0 || budget == 0;
        if self
            .max_num_inflight_kv_loads
            .map_or(loads_only, |limit| inflight >= limit)
        {
            return None;
        }
        waiting
            .into_iter()
            .take_while(|r| !self.external_admission.contains(&r.request_id))
            .find(|r| {
                self.external_pending.contains(&r.request_id)
                    && !self.external_blocked.contains(&r.request_id)
            })
            .map(|r| r.request_id)
    }
    fn reserve_external_admission(&mut self, request_id: u64, bytes: u64) {
        if bytes == 0 {
            self.external_admissions.remove(&request_id);
        } else {
            self.external_admissions.insert(request_id, bytes);
        }
    }
    fn set_external_kv_reclaimable_bytes(&mut self, bytes: u64) {
        self.external_kv_reclaimable_bytes = bytes;
    }
    fn set_external_admission_blocked(&mut self, request_id: u64, blocked: bool) {
        if blocked {
            self.external_admission.insert(request_id);
        } else {
            self.external_admission.remove(&request_id);
        }
    }
    fn set_external_blocked(&mut self, request_id: u64, blocked: bool) {
        if blocked {
            self.external_blocked.insert(request_id);
        } else {
            self.external_blocked.remove(&request_id);
        }
    }

    fn external_reclaim_pending(&self) -> bool {
        self.external_reclaim.is_some()
    }

    fn adopt_external_state(&mut self, request_id: u64) {
        if let Some(request) = self.queue.iter_mut().find(|r| r.request_id == request_id) {
            request.state_copies = 0;
            request.external_state_source = true;
        }
    }

    fn take_external_preemptions(&mut self) -> Vec<u64> {
        std::mem::take(&mut self.external_preemptions)
    }

    fn cancel_iteration(&mut self) {
        self.running = false;
        self.current_batch.clear();
    }

    fn restore_external_prefix(&mut self, request_id: u64, tokens: u32) -> Result<()> {
        if let Some(request) = self.queue.iter_mut().find(|r| r.request_id == request_id) {
            if request.effective_kv_tokens() == 0 {
                anyhow::ensure!(
                    tokens < request.known_tokens.max(1),
                    "restored prefix must leave logits work"
                );
                request.external_kv_tokens = tokens;
                request.private_kv_tokens = tokens;
                request.private_capacity_tokens = u64::from(tokens);
                request.state_copies = 0;
                request.external_state_source =
                    tokens > 0 && self.kv_bytes_per_request_per_device > 0;
            }
        }
        Ok(())
    }
    fn take_kv_progress(&mut self) -> Vec<(u64, u32)> {
        std::mem::take(&mut self.progress)
    }

    fn external_kv_requirement(&self) -> Option<(u64, u64)> {
        if !self.external_kv_enabled || self.running {
            return None;
        }
        if let Some(required) = self.external_reclaim {
            return Some(required);
        }
        self.queue
            .iter()
            .find(|r| {
                !self.external_blocked.contains(&r.request_id)
                    && !self.external_admission.contains(&r.request_id)
                    && !self.external_pending.contains(&r.request_id)
            })
            .map(|r| {
                let q = r
                    .known_tokens
                    .saturating_sub(r.effective_kv_tokens())
                    .max(1);
                let q = q.min(self.effective_max_num_batched_tokens);
                (
                    r.request_id,
                    self.private_bytes_after(
                        r,
                        u64::from(q) + u64::from(self.num_lookahead_tokens),
                        u64::from(q),
                    ),
                )
            })
            .filter(|(id, bytes)| {
                self.queue
                    .iter()
                    .find(|r| r.request_id == *id)
                    .is_some_and(|r| {
                        !self.admitted.contains(&r.completion_node_id)
                            || *bytes > self.private_bytes(r)
                    })
            })
    }

    fn adopt_external_prefix(&mut self, request_id: u64, tokens: u32) -> Result<()> {
        for request in &mut self.queue {
            if request.request_id == request_id {
                anyhow::ensure!(
                    tokens <= request.effective_kv_tokens(),
                    "published KV exceeds computed progress"
                );
                request.external_kv_tokens = tokens;
            }
        }
        Ok(())
    }

    fn configure_external_kv(&mut self) -> Result<KvMemoryInfo> {
        anyhow::ensure!(
            !self.enable_prefix_caching,
            "externally owned KV requires child prefix caching disabled"
        );
        self.external_kv_enabled = true;
        Ok(KvMemoryInfo {
            compatibility: 0,
            capacity_bytes: self
                .usable_device_mem
                .saturating_sub(self.model_weight_bytes_per_device),
            block_tokens: self.block_size,
            kv_bytes_per_token: self.kv_bytes_per_token_per_device,
            kv_bytes_per_request: self.kv_bytes_per_request_per_device,
        })
    }

    fn set_external_kv_bytes(&mut self, bytes: u64) -> Result<()> {
        anyhow::ensure!(
            self.external_kv_enabled,
            "external KV ownership is not configured"
        );
        let private_bytes: u64 = self
            .queue
            .iter()
            .map(|request| self.private_bytes(request))
            .sum();
        anyhow::ensure!(
            self.model_weight_bytes_per_device
                .saturating_add(bytes)
                .saturating_add(private_bytes)
                <= self.usable_device_mem,
            "external KV reservation exceeds available HBM: weights={}, external={bytes}, private={private_bytes}, capacity={}",
            self.model_weight_bytes_per_device, self.usable_device_mem
        );
        self.external_kv_bytes = bytes;
        Ok(())
    }

    fn reserve_external_prefix(&mut self, request_id: u64, tokens: u32) -> Result<()> {
        anyhow::ensure!(
            self.external_kv_enabled,
            "external KV ownership is not configured"
        );
        anyhow::ensure!(
            !self.external_prefixes.contains_key(&request_id),
            "duplicate external prefix reservation"
        );
        self.external_prefixes.insert(request_id, tokens);
        Ok(())
    }

    fn init(
        &mut self,
        model_weight_bytes_per_device: u64,
        kv_bytes_per_request_per_device: u64,
        kv_bytes_per_token_per_device: u64,
        parallel: ParallelStrategy,
    ) -> Result<()> {
        if self.block_size == 0 {
            bail!("block_size must be positive");
        }
        if self.max_num_inflight_kv_loads == Some(0) {
            bail!("max_num_inflight_kv_loads must be positive");
        }
        if !self.device_mem_utilization.is_finite()
            || self.device_mem_utilization <= 0.0
            || self.device_mem_utilization > 1.0
        {
            bail!(
                "device_mem_utilization must be in (0, 1], got {}",
                self.device_mem_utilization
            );
        }

        self.model_weight_bytes_per_device = model_weight_bytes_per_device;
        self.kv_bytes_per_request_per_device = kv_bytes_per_request_per_device;
        self.kv_bytes_per_token_per_device = kv_bytes_per_token_per_device;
        self.usable_device_mem = ((self.device_mem as f64 * self.device_mem_utilization).floor()
            as u64)
            .checked_sub(self.reserved_device_bytes)
            .ok_or_else(|| anyhow!("reserved_device_bytes exceeds usable device memory"))?;
        self.effective_max_batch = self
            .max_batch
            .checked_mul(parallel.ep as usize)
            .ok_or_else(|| anyhow!("effective max_batch overflowed"))?;
        self.effective_max_num_batched_tokens = self
            .max_num_batched_tokens
            .checked_mul(parallel.ep)
            .ok_or_else(|| anyhow!("effective max_num_batched_tokens overflowed"))?;
        if self.usable_device_mem == 0 {
            bail!(
                "device_mem_utilization leaves no usable device memory from device_mem={}",
                self.device_mem
            );
        }
        Ok(())
    }

    fn schedule(&mut self, now: f64) -> Result<Vec<RequestState>> {
        self.iteration_start = now;
        if self.running || self.queue.is_empty() {
            return Ok(Vec::new());
        }
        self.external_reclaim = None;
        self.admission_waiting.clear();
        // External capacity waits retain the FCFS admission barrier. Transfer
        // preparation can be skipped, but later new work cannot bypass HBM waits.
        let capacity_wait = self.queue.iter().any(|r| {
            self.external_admission.contains(&r.request_id)
                || self.external_pending.contains(&r.request_id)
        });
        let running_wait = self.running_wait();
        let mut capacity_barrier = running_wait;
        let mut capacity_suffix = HashSet::new();
        for request in &self.queue {
            capacity_barrier |= self.external_admission.contains(&request.request_id);
            // Requests holding loaded KV pass a waiting request's HBM wait.
            if capacity_barrier
                && !self.admitted.contains(&request.completion_node_id)
                && (running_wait || !self.holds_external_kv(request))
            {
                capacity_suffix.insert(request.completion_node_id);
            }
        }
        let order: HashMap<NodeID, usize> = self
            .queue
            .iter()
            .enumerate()
            .map(|(position, r)| (r.completion_node_id, position))
            .collect();
        let (mut queue, blocked): (Vec<_>, Vec<_>) =
            std::mem::take(&mut self.queue).into_iter().partition(|r| {
                !self.external_blocked.contains(&r.request_id)
                    && !self.external_admission.contains(&r.request_id)
            });
        if queue.is_empty() {
            self.queue = blocked.into_iter().collect();
            return Ok(Vec::new());
        }
        // V1 runs running requests, then those holding loaded KV, then the
        // waiting queue.
        queue.sort_by_key(|r| {
            !self.admitted.contains(&r.completion_node_id) as u8
                * (2 - self.holds_external_kv(r) as u8)
        });
        let mut preempted = Vec::new();
        let mut batch = Vec::new();
        let mut budget = self.effective_max_num_batched_tokens;
        let mut gpu_bytes = self.model_weight_bytes_per_device
            + self.external_kv_bytes
            + self
                .prefix_cache
                .bytes_used(self.block_size, self.kv_bytes_per_token_per_device)
            + queue
                .iter()
                .chain(&blocked)
                .map(|r| self.private_bytes(r))
                .sum::<u64>();
        let mut index = 0;
        while index < queue.len() && batch.len() < self.effective_max_batch && budget > 0 {
            let is_running = self.admitted.contains(&queue[index].completion_node_id);
            if !is_running && !preempted.is_empty() {
                break;
            }
            if !is_running && capacity_suffix.contains(&queue[index].completion_node_id) {
                self.admission_waiting
                    .extend(queue[index..].iter().map(|r| r.completion_node_id));
                break;
            }
            // A call awaiting its KV lookup turn holds the queue, not HBM.
            if !is_running && self.external_pending.contains(&queue[index].request_id) {
                break;
            }
            let request = &mut queue[index];
            let original_shared = request.shared_kv_tokens;
            let matched = if self.enable_prefix_caching {
                request.longest_cache_hit(
                    &mut self.prefix_cache,
                    self.block_size,
                    request.align_state,
                )
            } else {
                original_shared
            };
            let state_key =
                (request.align_state && matched > original_shared && request.state_hash.is_none())
                    .then(|| {
                        PrefixCache::checkpoint_key(
                            request.block_hashes[(matched / self.block_size - 1) as usize],
                        )
                    });
            let acquired_bytes = self.prefix_cache.reclaimable_bytes_for(
                &request.block_hashes[(original_shared / self.block_size) as usize
                    ..(matched / self.block_size) as usize],
                self.cache_block_bytes(),
            ) + state_key.map_or(0, |key| {
                self.prefix_cache
                    .reclaimable_bytes_for(&[key], self.cache_block_bytes())
            });
            // Price the prospective prefix without touching the free queue.
            request.shared_kv_tokens = matched;
            let q_len = request.next_chunk(budget, self.block_size);
            let old = self.private_bytes(request);
            let allocation = u64::from(q_len) + u64::from(self.num_lookahead_tokens);
            let needed = self.private_bytes_after(request, allocation, u64::from(q_len));
            let gate = if !is_running && self.scheduler_reserve_full_isl {
                self.private_bytes_after(
                    request,
                    u64::from(
                        request
                            .known_tokens
                            .saturating_sub(request.effective_kv_tokens())
                            .max(1),
                    ),
                    u64::from(q_len),
                )
            } else {
                needed
            };
            let available_cache = self
                .prefix_cache
                .reclaimable_bytes(self.cache_block_bytes())
                .saturating_sub(acquired_bytes)
                + self.external_kv_reclaimable_bytes;
            if !is_running
                && gpu_bytes
                    .saturating_sub(old)
                    .saturating_sub(available_cache)
                    .saturating_add(gate.max(needed))
                    > self.usable_device_mem
            {
                queue[index].shared_kv_tokens = original_shared;
                self.admission_waiting
                    .extend(queue[index..].iter().map(|r| r.completion_node_id));
                break;
            }
            let request = &mut queue[index];
            request.shared_kv_tokens = original_shared;
            request.relock_shared_prefix(&mut self.prefix_cache, self.block_size, matched);
            if let Some(key) = state_key {
                self.prefix_cache.lock(&[key]);
                request.state_hash = Some(key);
            }
            loop {
                if gpu_bytes.saturating_sub(old).saturating_add(needed) <= self.usable_device_mem {
                    break;
                }
                let before_cache = self
                    .prefix_cache
                    .bytes_used(self.block_size, self.kv_bytes_per_token_per_device);
                if self.prefix_cache.evict_one() {
                    let after_cache = self
                        .prefix_cache
                        .bytes_used(self.block_size, self.kv_bytes_per_token_per_device);
                    gpu_bytes = gpu_bytes.saturating_sub(before_cache - after_cache);
                    continue;
                }
                if self.external_kv_reclaimable_bytes > 0 {
                    self.external_reclaim = Some((
                        queue[index].request_id,
                        gpu_bytes
                            .saturating_sub(
                                self.model_weight_bytes_per_device + self.external_kv_bytes + old,
                            )
                            .saturating_add(needed),
                    ));
                    break;
                }
                if !is_running {
                    break;
                }
                let victim = (index..queue.len())
                    .rev()
                    .find(|&i| self.admitted.contains(&queue[i].completion_node_id));
                let Some(victim) = victim else {
                    break;
                };
                let mut request = queue.remove(victim);
                gpu_bytes = gpu_bytes.saturating_sub(self.private_bytes(&request));
                if self.external_kv_enabled {
                    self.external_preemptions.push(request.request_id);
                    request.external_kv_tokens = 0;
                }
                request.preempt(&mut self.prefix_cache, self.block_size);
                gpu_bytes += self.private_bytes(&request);
                self.admitted.remove(&request.completion_node_id);
                self.preemptions_total += 1;
                preempted.push(request);
                if victim == index {
                    break;
                }
            }
            if self.external_reclaim.is_some() {
                break;
            }
            if index >= queue.len()
                || !is_running
                    && gpu_bytes.saturating_sub(old).saturating_add(needed) > self.usable_device_mem
            {
                if index < queue.len() {
                    queue[index].relock_shared_prefix(
                        &mut self.prefix_cache,
                        self.block_size,
                        original_shared,
                    );
                    if original_shared == 0 {
                        queue[index].release_state_source(&mut self.prefix_cache);
                    }
                    self.admission_waiting
                        .extend(queue[index..].iter().map(|r| r.completion_node_id));
                }
                break;
            }
            if is_running && !self.admitted.contains(&queue[index].completion_node_id) {
                break;
            }
            // The current running request may itself have been preempted.
            if gpu_bytes.saturating_sub(old).saturating_add(needed) > self.usable_device_mem {
                break;
            }
            let request = &mut queue[index];
            gpu_bytes = gpu_bytes.saturating_sub(old) + needed;
            request.state_copies = request.state_copies_after(u64::from(q_len), self.block_size);
            request.private_capacity_tokens = u64::from(request.private_kv_tokens) + allocation;
            self.admitted.insert(request.completion_node_id);
            self.external_admissions.remove(&request.request_id);
            batch.push(RequestState {
                q_len,
                kv_len: request.effective_kv_tokens(),
                lm_head_len: 1,
            });
            budget -= q_len;
            index += 1;
        }
        // V1 prepends each victim to waiting. Running requests retain FCFS
        // order, and so do waiting ones, including those loading external KV
        // or waiting for HBM, which V1 keeps in place.
        let split = queue
            .iter()
            .position(|r| !self.admitted.contains(&r.completion_node_id))
            .unwrap_or(queue.len());
        let waiting = queue.split_off(split);
        let (blocked_running, blocked_waiting): (Vec<_>, Vec<_>) = blocked
            .into_iter()
            .partition(|r| self.admitted.contains(&r.completion_node_id));
        let mut waiting = blocked_waiting
            .into_iter()
            .chain(waiting)
            .collect::<Vec<_>>();
        waiting.sort_by_key(|r| order[&r.completion_node_id]);
        queue.extend(blocked_running);
        queue.extend(preempted.into_iter().rev());
        queue.extend(waiting);
        self.queue = queue.into_iter().collect();
        if !self.external_preemptions.is_empty() || self.external_reclaim.is_some() {
            return Ok(Vec::new());
        }
        if batch.is_empty() {
            if capacity_wait {
                return Ok(Vec::new());
            }
            bail!("request cannot fit currently known input and lookahead in available HBM");
        }
        self.running = true;
        self.current_batch = batch.clone();
        Ok(batch)
    }

    fn enqueue_sub_request(&mut self, request: ReadySubRequest) {
        let external = self
            .external_prefixes
            .remove(&request.request_id)
            .unwrap_or(0);
        assert!(
            external <= request.kv_tokens,
            "external prefix exceeds restored progress"
        );
        let mut running = RunningRequest::new(request);
        running.align_state = self.kv_bytes_per_request_per_device > 0
            && (self.enable_prefix_caching || self.external_kv_enabled);
        running.cache_nonce = self.next_cache_nonce;
        self.next_cache_nonce += 1;
        running.external_kv_tokens = external;
        if external > 0 && self.kv_bytes_per_request_per_device > 0 {
            running.state_copies = 0;
            running.external_state_source = true;
        }
        self.queue.push_back(running);
    }

    fn done_iteration(&mut self, now: f64) -> Vec<SubRequestCompletion> {
        self.progress.clear();
        let current_batch = std::mem::take(&mut self.current_batch);

        let mut tail = self.queue.split_off(current_batch.len());
        let started = self.iteration_start;
        for (request, scheduled) in self.queue.iter_mut().zip(&current_batch) {
            request.first_scheduled.get_or_insert(started);
            let previous_tokens = request.effective_kv_tokens();
            request.private_kv_tokens += scheduled.q_len;
            request.state_copies = 1;
            request.release_state_source(&mut self.prefix_cache);
            if request.known_tokens <= request.effective_kv_tokens()
                && request.known_tokens < request.input_tokens + request.output_tokens
            {
                let latency = if request.token_latency.is_empty() {
                    now - request.arrival_time
                } else {
                    now - request.last_time
                };
                request.token_latency.push(latency);
                self.generation_tokens_total += 1;
                request.last_time = now;
                request.known_tokens += 1;
            }

            if self.external_kv_enabled
                && (request.external_state_source
                    || previous_tokens / self.block_size
                        < request.effective_kv_tokens() / self.block_size)
            {
                self.progress
                    .push((request.request_id, request.effective_kv_tokens()));
                request.external_state_source = false;
            }
            if self.enable_prefix_caching {
                request.publish_shared_prefix(&mut self.prefix_cache, self.block_size);
                let checkpoint = request.replay_checkpoint(self.block_size);
                if self.kv_bytes_per_request_per_device > 0
                    && checkpoint > 0
                    && request.effective_kv_tokens() == checkpoint
                {
                    let key = self.prefix_cache.publish_checkpoint(
                        request.block_hashes[(checkpoint / self.block_size - 1) as usize],
                        self.kv_bytes_per_request_per_device,
                    );
                    self.prefix_cache.lock(&[key]);
                    request.state_hash = Some(key);
                    request.state_copies = 0;
                }
            }
        }

        let mut completed = Vec::new();
        let mut next_queue = LinkedList::new();
        while let Some(mut request) = self.queue.pop_front() {
            // Completion needs the whole prompt computed, even without outputs.
            let total_tokens = request.input_tokens + request.output_tokens;
            if request.known_tokens >= total_tokens
                && request.effective_kv_tokens() >= request.input_tokens
            {
                self.admitted.remove(&request.completion_node_id);
                request.release_state_source(&mut self.prefix_cache);
                if self.enable_prefix_caching {
                    request.relock_shared_prefix(&mut self.prefix_cache, self.block_size, 0);
                }
                completed.push(SubRequestCompletion {
                    request_id: request.request_id,
                    subrequest_index: request.subrequest_index,
                    completion_node_id: request.completion_node_id,
                    queue_wait: request.first_scheduled.map(|t| t - request.arrival_time),
                    token_latency: request.token_latency,
                });
            } else {
                next_queue.push_back(request);
            }
        }

        next_queue.append(&mut tail);
        self.queue = next_queue;
        self.running = false;
        completed
    }

    fn instance_load(&self) -> InstanceLoad {
        let running_requests = self.current_batch.len();
        let running_tokens = self
            .current_batch
            .iter()
            .map(|request| u64::from(request.q_len))
            .sum();
        let mut waiting_requests = 0usize;
        let mut waiting_tokens = 0u64;
        for request in self.queue.iter().skip(running_requests) {
            waiting_requests += 1;
            waiting_tokens += u64::from(
                request
                    .known_tokens
                    .saturating_sub(request.effective_kv_tokens())
                    .max(1),
            );
        }
        InstanceLoad {
            waiting_requests,
            running_requests,
            waiting_tokens,
            running_tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scheduler(budget: u32, prefix_caching: bool) -> ChunkedPrefillScheduler {
        let mut scheduler: ChunkedPrefillScheduler = toml::from_str(&format!(
            "block_size = 4\nmax_batch = 32\nmax_num_batched_tokens = {budget}\n\
             device_mem = 1048576\ndevice_mem_utilization = 1.0\n\
             enable_prefix_caching = {prefix_caching}"
        ))
        .unwrap();
        scheduler
            .init(
                0,
                0,
                1,
                ParallelStrategy {
                    tp: 1,
                    pp: 1,
                    ep: 1,
                },
            )
            .unwrap();
        scheduler
    }

    fn request(input: u32, output: u32) -> ReadySubRequest {
        ReadySubRequest {
            request_id: 0,
            subrequest_index: 0,
            completion_node_id: 0,
            arrival_time: 0.0,
            input_tokens: input,
            output_tokens: output,
            known_tokens: input,
            kv_tokens: 0,
            input_token_ids: Some((0..input).collect()),
            output_token_ids: None,
            kv_prefix: None,
        }
    }

    fn state(scheduler: &ChunkedPrefillScheduler, running: u64, waiting: u64, generated: u64) {
        assert_eq!(
            {
                let value = scheduler.observe();
                serde_json::json!({
                    "running_requests": value["running_requests"],
                    "waiting_requests": value["waiting_requests"],
                    "generation_tokens_total": value["generation_tokens_total"],
                })
            },
            serde_json::json!({
                "running_requests": running,
                "waiting_requests": waiting,
                "generation_tokens_total": generated,
            })
        );
    }

    #[test]
    fn reserved_workspace_reduces_usable_memory() {
        let mut scheduler = scheduler(8, false);
        scheduler.device_mem = 16;
        scheduler.reserved_device_bytes = 4;
        let parallel = ParallelStrategy {
            tp: 1,
            pp: 1,
            ep: 1,
        };
        scheduler.init(4, 0, 1, parallel).unwrap();
        assert_eq!(scheduler.usable_device_mem, 12);
        scheduler.reserved_device_bytes = 17;
        assert!(scheduler.init(4, 0, 1, parallel).is_err());
    }

    #[test]
    fn observations_follow_admission_and_emission_not_scheduled_work() {
        let mut scheduler = scheduler(4, false);
        scheduler.enqueue_sub_request(request(8, 2));
        state(&scheduler, 0, 1, 0);
        scheduler.schedule(0.0).unwrap();
        state(&scheduler, 1, 0, 0);
        scheduler.done_iteration(1.0);
        state(&scheduler, 1, 0, 0);
        scheduler.schedule(0.0).unwrap();
        scheduler.done_iteration(2.0);
        state(&scheduler, 1, 0, 1);
        scheduler.schedule(0.0).unwrap();
        scheduler.done_iteration(3.0);
        state(&scheduler, 0, 0, 2);
    }

    #[test]
    fn zero_output_completes_after_its_whole_prompt() {
        let mut scheduler = scheduler(4, false);
        scheduler.enqueue_sub_request(request(12, 0));
        for (step, q_len) in [4, 4, 4].into_iter().enumerate() {
            assert_eq!(scheduler.schedule(0.0).unwrap()[0].q_len, q_len);
            let completed = scheduler.done_iteration(step as f64 + 1.0);
            assert_eq!(completed.len(), usize::from(step == 2));
        }
        state(&scheduler, 0, 0, 0);
    }

    #[test]
    fn external_prefix_has_one_memory_owner_and_keeps_progress() {
        let mut scheduler = scheduler(4, false);
        scheduler.usable_device_mem = 12;
        assert_eq!(
            scheduler.configure_external_kv().unwrap().capacity_bytes,
            12
        );
        scheduler.set_external_kv_bytes(8).unwrap();
        scheduler.reserve_external_prefix(0, 8).unwrap();
        let mut seeded = request(12, 1);
        seeded.kv_tokens = 8;
        scheduler.enqueue_sub_request(seeded);
        let batch = scheduler.schedule(0.0).unwrap();
        assert_eq!((batch[0].kv_len, batch[0].q_len), (8, 4));
        assert!(scheduler.set_external_kv_bytes(9).is_err());
        assert_eq!(scheduler.done_iteration(1.0).len(), 1);
        scheduler.set_external_kv_bytes(12).unwrap();
        scheduler.enqueue_sub_request(request(4, 1));
        assert!(scheduler.schedule(0.0).is_err());
        scheduler.set_external_kv_bytes(8).unwrap();
        assert_eq!(scheduler.schedule(0.0).unwrap()[0].q_len, 4);
    }

    #[test]
    fn partial_external_block_charges_the_remaining_slots_before_admission() {
        // Five restored tokens plus one new token occupy two four-token blocks.
        // With one byte/token, seven bytes cannot admit it; eight can.
        for capacity in [7, 8] {
            let mut scheduler = scheduler(4, false);
            scheduler.usable_device_mem = capacity;
            scheduler.configure_external_kv().unwrap();
            scheduler.set_external_kv_bytes(5).unwrap();
            scheduler.reserve_external_prefix(0, 5).unwrap();
            let mut seeded = request(6, 1);
            seeded.kv_tokens = 5;
            scheduler.enqueue_sub_request(seeded);
            let batch = scheduler.schedule(0.0);
            if capacity == 7 {
                assert!(batch.is_err());
            } else {
                let batch = batch.unwrap();
                assert_eq!((batch[0].kv_len, batch[0].q_len), (5, 1));
                assert_eq!(scheduler.done_iteration(1.0).len(), 1);
            }
        }
    }

    #[test]
    fn external_prefix_survives_preemption_and_rejects_private_cache() {
        let mut running = RunningRequest::new(request(12, 3));
        running.private_kv_tokens = 12;
        running.external_kv_tokens = 8;
        running.preempt(&mut PrefixCache::default(), 4);
        assert_eq!(running.effective_kv_tokens(), 8);
        assert_eq!(running.private_kv_bytes(4, 0, 1), 0);
        assert_eq!(running.private_kv_bytes_after(4, 4, 4, 0, 1), 4);
        assert!(scheduler(4, true).configure_external_kv().is_err());
    }

    #[test]
    fn loaded_kv_runs_before_lookups_which_need_budget_or_a_limit() {
        for limit in [None, Some(1)] {
            let mut scheduler = scheduler(8, false);
            scheduler.max_num_inflight_kv_loads = limit;
            scheduler.configure_external_kv().unwrap();
            // Request 0 awaits its lookup; request 1 behind it holds loaded KV.
            for id in 0..2u64 {
                let mut request = request(8, 1);
                request.request_id = id;
                request.completion_node_id = id as _;
                scheduler.enqueue_sub_request(request);
            }
            scheduler.set_external_pending(0, true);
            scheduler.reserve_external_admission(1, 8);
            // Request 1 takes the whole budget; its admission frees its
            // in-flight place.
            assert_eq!(scheduler.external_lookup(), limit.map(|_| 0));
            let batch = scheduler.schedule(0.0).unwrap();
            assert_eq!((batch.len(), batch[0].q_len), (1, 8));
            assert!(scheduler.admitted.contains(&1));
        }
    }

    #[test]
    fn retained_admission_and_preemption_have_different_observations() {
        for memory in [12, 8] {
            let mut scheduler = scheduler(8, false);
            scheduler.usable_device_mem = memory;
            for id in 0..2 {
                let mut request = request(4, 3);
                request.completion_node_id = id;
                scheduler.enqueue_sub_request(request);
            }
            assert_eq!(scheduler.schedule(0.0).unwrap().len(), 2);
            scheduler.done_iteration(1.0);
            state(&scheduler, 2, 0, 2);
            scheduler.effective_max_num_batched_tokens = 1;
            assert_eq!(scheduler.schedule(0.0).unwrap().len(), 1);
            assert_eq!(scheduler.instance_load().running_requests, 1);
            if memory == 12 {
                state(&scheduler, 2, 0, 2);
            } else {
                state(&scheduler, 1, 1, 2);
            }
        }
    }

    #[test]
    fn seeded_kv_waits_for_admission_and_zero_output_emits_nothing() {
        let mut scheduler = scheduler(4, false);
        let mut seeded = request(8, 1);
        seeded.kv_tokens = 7;
        scheduler.enqueue_sub_request(seeded);
        state(&scheduler, 0, 1, 0);
        scheduler.schedule(0.0).unwrap();
        state(&scheduler, 1, 0, 0);
        scheduler.done_iteration(1.0);
        state(&scheduler, 0, 0, 1);
        scheduler.enqueue_sub_request(request(4, 0));
        scheduler.schedule(0.0).unwrap();
        scheduler.done_iteration(2.0);
        state(&scheduler, 0, 0, 1);
    }

    #[test]
    fn prefill_produces_the_first_token_without_an_extra_decode() {
        let mut scheduler = scheduler(16, false);
        scheduler.enqueue_sub_request(request(8, 1));
        let batch = scheduler.schedule(0.0).unwrap();
        assert_eq!(batch[0].q_len, 8);
        assert_eq!(batch[0].kv_len, 0);
        let results = scheduler.done_iteration(0.5);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].token_latency, vec![0.5]);
        assert!(scheduler.schedule(0.0).unwrap().is_empty());
    }

    #[test]
    fn chunked_prefill_emits_only_after_the_last_chunk() {
        let mut scheduler = scheduler(4, false);
        scheduler.enqueue_sub_request(request(8, 3));
        assert_eq!(scheduler.schedule(0.0).unwrap()[0].q_len, 4);
        assert!(scheduler.done_iteration(1.0).is_empty());
        assert!(scheduler.queue.front().unwrap().token_latency.is_empty());
        let batch = scheduler.schedule(0.0).unwrap();
        assert_eq!((batch[0].q_len, batch[0].kv_len), (4, 4));
        assert!(scheduler.done_iteration(2.0).is_empty());
        assert_eq!(scheduler.queue.front().unwrap().token_latency, vec![2.0]);
        for kv in 8..10 {
            let batch = scheduler.schedule(0.0).unwrap();
            assert_eq!((batch[0].q_len, batch[0].kv_len), (1, kv));
            let results = scheduler.done_iteration(f64::from(kv - 5));
            if kv == 9 {
                assert_eq!(results[0].token_latency, vec![2.0, 1.0, 1.0]);
            } else {
                assert!(results.is_empty());
            }
        }
        assert!(scheduler.schedule(0.0).unwrap().is_empty());
    }

    #[test]
    fn full_prefix_hit_recomputes_the_last_block_for_logits() {
        let mut scheduler = scheduler(16, true);
        scheduler.enqueue_sub_request(request(8, 1));
        scheduler.schedule(0.0).unwrap();
        assert_eq!(scheduler.done_iteration(1.0).len(), 1);
        scheduler.enqueue_sub_request(request(8, 1));
        let batch = scheduler.schedule(0.0).unwrap();
        assert_eq!((batch[0].q_len, batch[0].kv_len), (4, 4));
        assert_eq!(scheduler.done_iteration(2.0).len(), 1);
    }

    #[test]
    fn eviction_after_completion_keeps_the_earlier_prefix_reusable() {
        let mut scheduler = scheduler(16, true);
        scheduler.enqueue_sub_request(request(12, 1));
        scheduler.schedule(0.0).unwrap();
        assert_eq!(scheduler.done_iteration(1.0).len(), 1);
        assert!(scheduler.prefix_cache.evict_one());
        scheduler.enqueue_sub_request(request(12, 1));
        let batch = scheduler.schedule(0.0).unwrap();
        assert_eq!((batch[0].q_len, batch[0].kv_len), (4, 8));
        assert_eq!(scheduler.done_iteration(2.0).len(), 1);
    }

    #[test]
    fn hashing_future_blocks_does_not_publish_uncomputed_kv() {
        let mut scheduler = scheduler(4, true);
        scheduler.enqueue_sub_request(request(20, 1));
        scheduler.schedule(0.0).unwrap();
        scheduler.done_iteration(1.0);
        let request = scheduler.queue.front().unwrap();
        assert_eq!(request.block_hashes.len(), 4);
        assert_eq!(scheduler.prefix_cache.bytes_used(4, 1), 4);
        assert_eq!(scheduler.prefix_cache.longest_hit(&request.block_hashes), 1);
        assert!(!scheduler.prefix_cache.evict_one());
    }

    #[test]
    fn output_blocks_cross_the_prompt_boundary_and_publish_once_computed() {
        let mut scheduler = scheduler(16, true);
        let mut input = request(6, 7);
        input.output_token_ids = Some((6..13).collect());
        scheduler.enqueue_sub_request(input);
        for time in 1_u32..=7 {
            scheduler.schedule(0.0).unwrap();
            let completed = scheduler.done_iteration(f64::from(time));
            let computed = 6 + time - 1;
            assert_eq!(
                scheduler.prefix_cache.bytes_used(4, 1),
                u64::from(computed / 4 * 4)
            );
            if time < 7 {
                let request = scheduler.queue.front().unwrap();
                assert_eq!(request.block_hashes.len(), (computed / 4) as usize);
                assert!(!scheduler.prefix_cache.evict_one());
            } else {
                assert_eq!(completed.len(), 1);
            }
        }
        scheduler.enqueue_sub_request(request(12, 1));
        let batch = scheduler.schedule(0.0).unwrap();
        assert_eq!((batch[0].q_len, batch[0].kv_len), (4, 8));
    }

    #[test]
    fn preemption_keeps_hashes_but_rechecks_cache_residency() {
        let mut request = RunningRequest::new(request(12, 3));
        let mut cache = PrefixCache::default();
        request.private_kv_tokens = 12;
        request.publish_shared_prefix(&mut cache, 4);
        let hashes = request.block_hashes.clone();
        request.preempt(&mut cache, 4);
        assert_eq!(request.effective_kv_tokens(), 0);
        assert_eq!(request.block_hashes, hashes);
        assert!(cache.evict_one());
        assert_eq!(request.longest_cache_hit(&mut cache, 4, false), 8);
        request.relock_shared_prefix(&mut cache, 4, 8);
        request.private_kv_tokens = 4;
        request.publish_shared_prefix(&mut cache, 4);
        assert_eq!(request.block_hashes, hashes);
        assert!(!cache.evict_one());
        request.preempt(&mut cache, 4);
        for _ in 0..3 {
            assert!(cache.evict_one());
        }
        assert_eq!(request.longest_cache_hit(&mut cache, 4, false), 0);
    }

    #[test]
    fn failed_admission_releases_acquired_blocks_for_retry() {
        let mut scheduler = scheduler(16, true);
        scheduler.enqueue_sub_request(request(12, 1));
        scheduler.schedule(0.0).unwrap();
        scheduler.done_iteration(1.0);
        scheduler.usable_device_mem = 8;
        scheduler.enqueue_sub_request(request(12, 1));
        assert!(scheduler.schedule(0.0).is_err());
        assert_eq!(scheduler.queue.front().unwrap().shared_kv_tokens, 0);
        scheduler.usable_device_mem = 12;
        let batch = scheduler.schedule(0.0).unwrap();
        assert_eq!((batch[0].q_len, batch[0].kv_len), (4, 8));
        assert_eq!(scheduler.done_iteration(2.0).len(), 1);
        for _ in 0..3 {
            assert!(scheduler.prefix_cache.evict_one());
        }
    }

    #[test]
    fn missing_output_ids_retain_execution_scoped_opaque_blocks() {
        let mut request = RunningRequest::new(request(6, 8));
        let mut cache = PrefixCache::default();
        request.private_kv_tokens = 12;
        request.publish_shared_prefix(&mut cache, 4);
        assert_eq!(request.shared_kv_tokens, 12);
        assert_eq!(request.private_kv_tokens, 0);
        assert_eq!(request.block_hashes.len(), 3);
    }

    #[test]
    fn unknown_prompt_blocks_are_retained_only_when_caching_is_enabled() {
        for enabled in [false, true] {
            let mut scheduler = scheduler(4, enabled);
            let mut input = request(8, 1);
            if enabled {
                input.input_token_ids = None;
                input.output_token_ids = Some(vec![8]);
            }
            scheduler.enqueue_sub_request(input);
            scheduler.schedule(0.0).unwrap();
            scheduler.done_iteration(1.0);
            assert_eq!(
                scheduler.queue.front().unwrap().block_hashes.is_empty(),
                !enabled
            );
            assert_eq!(
                scheduler.prefix_cache.bytes_used(4, 1),
                if enabled { 4 } else { 0 }
            );
        }
    }

    #[test]
    fn zero_block_size_is_rejected() {
        let mut scheduler = scheduler(4, true);
        scheduler.block_size = 0;
        assert_eq!(
            scheduler
                .init(
                    0,
                    0,
                    1,
                    ParallelStrategy {
                        tp: 1,
                        pp: 1,
                        ep: 1
                    }
                )
                .unwrap_err()
                .to_string(),
            "block_size must be positive"
        );
    }
}
