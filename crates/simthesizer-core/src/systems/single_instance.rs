use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::path::Path;

use anyhow::{anyhow, bail, Result};
use serde::Deserialize;

use crate::engine::Graph;
use crate::model::Model;
use crate::model_runner::ModelRunner;
use crate::module::{LogicalHandler, Module};
use crate::scheduler::Scheduler;
use crate::system::{pending_llm_load, System, SystemEvent};
use crate::types::{
    InstanceLoad, NodeID, ReadySubRequest, Request, RequestResult, ResultSubRequest, SubRequest,
    SubRequestCompletion, TtftParts,
};

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SingleInstanceSystem {
    #[serde(skip)]
    kv_capacity_event: Option<NodeID>,
    model_runner: Box<dyn ModelRunner>,
    model: Box<dyn Model>,
    scheduler: Box<dyn Scheduler>,

    #[serde(skip_deserializing, default)]
    results: Vec<RequestResult>,

    #[serde(skip_deserializing, default)]
    pending_requests: HashMap<u64, Request>,

    #[serde(skip_deserializing, default)]
    request_sinks: HashMap<u64, *mut dyn LogicalHandler>,

    #[serde(skip_deserializing, default)]
    active_requests: HashMap<u64, ActiveRequest>,

    #[serde(skip_deserializing, default)]
    pending_completion_results: HashMap<NodeID, SubRequestCompletion>,

    #[serde(skip_deserializing, default)]
    first_token_waits: (u64, f64),
}

struct ActiveRequest {
    pending_roots: HashSet<usize>,
    request: Request,
    completed_results: Vec<Option<SubRequestCompletion>>,
}

impl ActiveRequest {
    fn new(request: Request) -> Self {
        Self {
            pending_roots: request.initial.iter().copied().collect(),
            completed_results: vec![None; request.sub_requests.len()],
            request,
        }
    }

    fn all_complete(&self) -> bool {
        self.completed_results.iter().all(Option::is_some)
    }
}

crate::register_system!(SingleInstanceSystem, "single_instance");

enum LogicalPayload {
    RequestArrival {
        request_id: u64,
    },
    SubRequestArrival {
        request_id: u64,
        subrequest_index: usize,
    },
    SubRequestComplete {
        request_id: u64,
        subrequest_index: usize,
    },
    LlmEnqueue {
        request_id: u64,
        subrequest_index: usize,
        completion_node_id: NodeID,
    },
    LlmCapacityReady,
    LlmPrepareSchedule,
    LlmSchedule,
    LlmBatchComplete,
    ToolCallRun {
        duration: f64,
    },
}

fn parse_payload(payload: Box<dyn Any>) -> Result<LogicalPayload> {
    payload
        .downcast::<LogicalPayload>()
        .map(|payload| *payload)
        .map_err(|_| anyhow!("single_instance received an unexpected logical payload"))
}

impl SingleInstanceSystem {
    fn detached_completion_node(&mut self, current: NodeID, graph: &mut Graph) -> Result<NodeID> {
        let detached = graph.pop_edges(current)?;
        match detached.as_slice() {
            [completion_node_id] => Ok(*completion_node_id),
            _ => bail!(
                "sub-request arrival node {current} detached {} children, expected exactly one completion node",
                detached.len()
            ),
        }
    }

    fn queue_completion(
        &mut self,
        completion_node_id: NodeID,
        completion: SubRequestCompletion,
    ) -> Result<()> {
        if completion.completion_node_id != completion_node_id {
            bail!(
                "completion node mismatch: payload carries {} but queued under {}",
                completion.completion_node_id,
                completion_node_id
            );
        }
        if self
            .pending_completion_results
            .insert(completion_node_id, completion)
            .is_some()
        {
            bail!("completion node {completion_node_id} was queued more than once");
        }
        Ok(())
    }

    fn finalize_completed_request(
        &mut self,
        request_id: u64,
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<()> {
        let Some(active) = self.active_requests.get(&request_id) else {
            bail!("request {request_id} disappeared before finalization");
        };
        if !active.all_complete() {
            return Ok(());
        }

        let active = self
            .active_requests
            .remove(&request_id)
            .ok_or_else(|| anyhow!("request {request_id} disappeared during finalization"))?;
        let request = active.request;
        let completed_results = active.completed_results;
        let sub_requests = request
            .sub_requests
            .iter()
            .enumerate()
            .map(|(index, sub_request)| {
                let completion = completed_results
                    .get(index)
                    .and_then(Option::as_ref)
                    .ok_or_else(|| {
                        anyhow!("request {request_id} missing completion for sub-request {index}")
                    })?;
                Ok(match sub_request {
                    SubRequest::Llm {
                        input_tokens,
                        output_tokens,
                        next,
                        interval,
                        model,
                        ..
                    } => ResultSubRequest::Llm {
                        input_tokens: *input_tokens,
                        output_tokens: *output_tokens,
                        next: next.clone(),
                        token_latency: completion.token_latency.clone(),
                        interval: *interval,
                        model: model.clone(),
                        session_id: sub_request.session_id().map(str::to_owned),
                        provider_tokens: sub_request.provider_tokens().cloned(),
                        kv_prefix: sub_request.kv_prefix().cloned(),
                        ttft_parts: TtftParts::scheduled(
                            completion.token_latency.first().copied(),
                            completion.queue_wait,
                        ),
                        reused_tokens: None,
                    },
                    SubRequest::ToolCall {
                        input_tokens,
                        output_tokens,
                        next,
                        duration,
                        interval,
                        session_id,
                    } => ResultSubRequest::ToolCall {
                        input_tokens: *input_tokens,
                        output_tokens: *output_tokens,
                        next: next.clone(),
                        duration: duration.unwrap_or(0.0),
                        interval: *interval,
                        session_id: session_id.clone(),
                    },
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let result = RequestResult {
            request_id,
            arrival_time: request.arrival_time,
            initial: request.initial.clone(),
            sub_requests,
        };
        if let Some(sink) = self.request_sinks.remove(&request_id) {
            let event =
                graph.add_logical_node(sink, SystemEvent::RequestCompleted { request, result });
            graph.add_edge(current, event)?;
        } else {
            self.results.push(result);
        }
        Ok(())
    }
}

impl LogicalHandler for SingleInstanceSystem {
    fn handle(
        &mut self,
        payload: Box<dyn Any>,
        now: f64,
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<f64> {
        let module = self as *mut Self as *mut dyn LogicalHandler;
        match parse_payload(payload)? {
            LogicalPayload::RequestArrival { request_id } => {
                let request = self.pending_requests.remove(&request_id).ok_or_else(|| {
                    anyhow!("single_instance is missing request {request_id} during arrival")
                })?;
                self.active_requests
                    .insert(request_id, ActiveRequest::new(request));
                Ok(0.0)
            }
            LogicalPayload::SubRequestArrival {
                request_id,
                subrequest_index,
            } => {
                let completion_node_id = self.detached_completion_node(current, graph)?;
                let sub_request = {
                    let active = self
                        .active_requests
                        .get(&request_id)
                        .ok_or_else(|| anyhow!("unknown active request {request_id}"))?;
                    active
                        .request
                        .sub_requests
                        .get(subrequest_index)
                        .ok_or_else(|| anyhow!("unknown subrequest index {subrequest_index}"))?
                        .clone()
                };

                let interval = sub_request.interval().unwrap_or(0.0);

                match sub_request {
                    SubRequest::Llm { .. } => {
                        let enqueue = graph.add_logical_node(
                            module,
                            LogicalPayload::LlmEnqueue {
                                request_id,
                                subrequest_index,
                                completion_node_id,
                            },
                        );
                        graph.add_edge(current, enqueue)?;
                    }
                    SubRequest::ToolCall { duration, .. } => {
                        self.queue_completion(
                            completion_node_id,
                            SubRequestCompletion {
                                request_id,
                                subrequest_index,
                                completion_node_id,
                                token_latency: Vec::new(),
                                queue_wait: None,
                            },
                        )?;
                        let run = graph.add_logical_node(
                            module,
                            LogicalPayload::ToolCallRun {
                                duration: duration.unwrap_or(0.0),
                            },
                        );
                        graph.add_edge(current, run)?;
                        graph.add_edge(run, completion_node_id)?;
                    }
                }

                Ok(interval)
            }
            LogicalPayload::SubRequestComplete {
                request_id,
                subrequest_index,
            } => {
                let completion = self
                    .pending_completion_results
                    .remove(&current)
                    .ok_or_else(|| anyhow!("completion node {current} has no queued result"))?;
                if completion.request_id != request_id
                    || completion.subrequest_index != subrequest_index
                    || completion.completion_node_id != current
                {
                    bail!("completion node {current} received mismatched completion payload");
                }

                if let Some(queue) = completion.queue_wait {
                    self.first_token_waits.0 += 1;
                    self.first_token_waits.1 += queue;
                }
                let active = self
                    .active_requests
                    .get_mut(&request_id)
                    .ok_or_else(|| anyhow!("completion for unknown request {request_id}"))?;
                if active.completed_results[subrequest_index]
                    .replace(completion)
                    .is_some()
                {
                    bail!(
                        "request {request_id} sub-request {subrequest_index} completed more than once"
                    );
                }

                self.finalize_completed_request(request_id, current, graph)?;
                Ok(0.0)
            }
            LogicalPayload::LlmEnqueue {
                request_id,
                subrequest_index,
                completion_node_id,
            } => {
                let (
                    input_tokens,
                    output_tokens,
                    input_token_ids,
                    output_token_ids,
                    known_tokens,
                    kv_tokens,
                    kv_prefix,
                ) = {
                    let active = self
                        .active_requests
                        .get_mut(&request_id)
                        .ok_or_else(|| anyhow!("unknown active request {request_id}"))?;
                    active.pending_roots.remove(&subrequest_index);
                    let sub_request = active
                        .request
                        .sub_requests
                        .get(subrequest_index)
                        .ok_or_else(|| anyhow!("unknown subrequest index {subrequest_index}"))?;
                    let SubRequest::Llm {
                        input_tokens,
                        output_tokens,
                        known_tokens,
                        kv_tokens,
                        input_token_ids,
                        output_token_ids,
                        ..
                    } = sub_request
                    else {
                        bail!("LlmEnqueue received non-Llm sub-request");
                    };
                    let known_tokens = known_tokens.unwrap_or(*input_tokens);
                    let kv_tokens = kv_tokens.unwrap_or(0);
                    (
                        *input_tokens,
                        *output_tokens,
                        input_token_ids.clone(),
                        output_token_ids.clone(),
                        known_tokens,
                        kv_tokens,
                        active.request.resolved_kv_prefix(subrequest_index),
                    )
                };
                self.scheduler.enqueue_sub_request(ReadySubRequest {
                    request_id,
                    subrequest_index,
                    completion_node_id,
                    arrival_time: now,
                    input_tokens,
                    output_tokens,
                    known_tokens,
                    kv_tokens,
                    input_token_ids,
                    output_token_ids,
                    kv_prefix,
                });
                let schedule = graph.add_logical_node(module, LogicalPayload::LlmPrepareSchedule);
                graph.add_edge(current, schedule)?;
                Ok(0.0)
            }
            LogicalPayload::LlmCapacityReady => {
                if self.kv_capacity_event == Some(current) {
                    self.kv_capacity_event = None;
                }
                Ok(0.0)
            }
            LogicalPayload::LlmPrepareSchedule => {
                if let Some(capacity) = self.kv_capacity_event {
                    let retry = graph.add_logical_node(module, LogicalPayload::LlmPrepareSchedule);
                    graph.add_edge(current, retry)?;
                    graph.add_edge(capacity, retry)?;
                    return Ok(0.0);
                }
                // A request at its turn first looks up its external prefix.
                if let Some(request) = self.scheduler.external_lookup() {
                    let sink = self.request_sinks[&request];
                    let lookup = graph.add_logical_node(sink, SystemEvent::KvLookup { request });
                    let retry = graph.add_logical_node(module, LogicalPayload::LlmPrepareSchedule);
                    graph.add_edge(current, lookup)?;
                    graph.add_edge(lookup, retry)?;
                    return Ok(0.0);
                }
                let schedule = graph.add_logical_node(module, LogicalPayload::LlmSchedule);
                let mut previous = current;
                if let Some((request_id, required_bytes)) = self.scheduler.external_kv_requirement()
                {
                    if let Some(&sink) = self.request_sinks.get(&request_id) {
                        let ranks =
                            (0..self.model_runner.parallel_strategy().tp as usize).collect();
                        let event = graph.add_logical_node(
                            sink,
                            SystemEvent::KvCapacity {
                                request_id,
                                ranks,
                                required_bytes,
                                continuation: schedule,
                            },
                        );
                        graph.add_edge(previous, event)?;
                        previous = event;
                    }
                }
                graph.add_edge(previous, schedule)?;
                Ok(0.0)
            }
            LogicalPayload::LlmSchedule => {
                let batch_requests = self.scheduler.schedule(now)?;
                let preemptions = self.scheduler.take_external_preemptions();
                if !preemptions.is_empty() || self.scheduler.external_reclaim_pending() {
                    self.scheduler.cancel_iteration();
                    let mut previous = current;
                    for request_id in preemptions {
                        if let Some(&sink) = self.request_sinks.get(&request_id) {
                            let event = graph
                                .add_logical_node(sink, SystemEvent::KvPreempted { request_id });
                            graph.add_edge(previous, event)?;
                            previous = event;
                        }
                    }
                    let retry = graph.add_logical_node(module, LogicalPayload::LlmPrepareSchedule);
                    graph.add_edge(previous, retry)?;
                    return Ok(0.0);
                }
                if batch_requests.is_empty() {
                    return Ok(0.0);
                }
                let last_node = self.model_runner.add_iteration(
                    self.model.as_ref(),
                    &batch_requests,
                    current,
                    graph,
                )?;

                let complete = graph.add_logical_node(module, LogicalPayload::LlmBatchComplete);
                graph.add_edge(last_node, complete)?;
                let capacity = graph.add_logical_node(module, LogicalPayload::LlmCapacityReady);
                graph.add_edge(complete, capacity)?;
                self.kv_capacity_event = Some(capacity);

                Ok(0.0)
            }
            LogicalPayload::LlmBatchComplete => {
                let completions = self.scheduler.done_iteration(now);
                let mut previous = current;
                for (request_id, computed_tokens) in self.scheduler.take_kv_progress() {
                    if let Some(&sink) = self.request_sinks.get(&request_id) {
                        let event = graph.add_logical_node(
                            sink,
                            SystemEvent::KvProgress {
                                request_id,
                                computed_tokens,
                            },
                        );
                        graph.add_edge(previous, event)?;
                        previous = event;
                    }
                }
                for completion in completions {
                    let completion_node_id = completion.completion_node_id;
                    self.queue_completion(completion_node_id, completion)?;
                    graph.add_edge(previous, completion_node_id)?;
                }

                let capacity = self.kv_capacity_event.expect("active batch capacity fence");
                if previous != current {
                    graph.add_edge(previous, capacity)?;
                }
                let schedule = graph.add_logical_node(module, LogicalPayload::LlmPrepareSchedule);
                graph.add_edge(capacity, schedule)?;
                Ok(0.0)
            }
            LogicalPayload::ToolCallRun { duration } => Ok(duration),
        }
    }
}

impl Module for SingleInstanceSystem {
    fn observe(&self) -> serde_json::Value {
        let mut value = serde_json::json!({"scheduler": self.scheduler.observe()});
        let runner = self.model_runner.observe();
        if !runner.is_null() {
            value["model_runner"] = runner;
        }
        value
    }

    fn resolve_path(&mut self, base_dir: &Path) -> Result<()> {
        self.model_runner.resolve_path(base_dir)?;
        self.model.resolve_path(base_dir)?;
        self.scheduler.resolve_path(base_dir)?;
        Ok(())
    }
}

impl System for SingleInstanceSystem {
    fn first_token_waits(&self) -> Option<(u64, f64)> {
        Some(self.first_token_waits)
    }
    fn next_kv_capacity_event(&self, rank: usize) -> Option<NodeID> {
        (rank < self.model_runner.parallel_strategy().tp as usize)
            .then_some(self.kv_capacity_event)
            .flatten()
    }
    fn external_kv_budget(&self, rank: usize) -> Result<u64> {
        anyhow::ensure!(
            rank < self.model_runner.parallel_strategy().tp as usize,
            "invalid external KV rank"
        );
        self.scheduler.external_kv_budget()
    }
    fn adopt_external_state(&mut self, request_id: u64) -> Result<()> {
        self.scheduler.adopt_external_state(request_id);
        Ok(())
    }
    fn set_external_admission_blocked(&mut self, request_id: u64, blocked: bool) -> Result<()> {
        self.scheduler
            .set_external_admission_blocked(request_id, blocked);
        Ok(())
    }
    fn set_external_kv_reclaimable_bytes(&mut self, bytes: &[u64]) -> Result<()> {
        self.scheduler
            .set_external_kv_reclaimable_bytes(bytes.iter().copied().min().unwrap_or(0));
        Ok(())
    }
    fn set_external_blocked(&mut self, request_id: u64, blocked: bool) -> Result<()> {
        self.scheduler.set_external_blocked(request_id, blocked);
        Ok(())
    }
    fn external_kv_requirement(&self, ranks: &[usize]) -> Option<(u64, u64)> {
        let _ = ranks;
        self.scheduler.external_kv_requirement()
    }
    fn known_tokens(&self, request_id: u64) -> Option<u32> {
        self.scheduler.known_tokens(request_id)
    }
    fn external_admission_bytes(
        &self,
        _rank: usize,
        known_tokens: u32,
        restored_tokens: u32,
    ) -> u64 {
        self.scheduler
            .external_admission_bytes(known_tokens, restored_tokens)
    }
    fn reserve_external_admission(
        &mut self,
        request_id: u64,
        _rank: usize,
        bytes: u64,
    ) -> Result<()> {
        self.scheduler.reserve_external_admission(request_id, bytes);
        Ok(())
    }
    fn set_external_pending(&mut self, request_id: u64, pending: bool) -> Result<()> {
        self.scheduler.set_external_pending(request_id, pending);
        Ok(())
    }
    fn wake(&mut self, after: NodeID, graph: &mut Graph) -> Result<()> {
        let module = self as *mut Self as *mut dyn LogicalHandler;
        let schedule = graph.add_logical_node(module, LogicalPayload::LlmPrepareSchedule);
        graph.add_edge(after, schedule)
    }
    fn restore_external_prefix(&mut self, request_id: u64, tokens: u32) -> Result<()> {
        self.scheduler.restore_external_prefix(request_id, tokens)
    }

    fn adopt_external_prefix(&mut self, request_id: u64, tokens: u32) -> Result<()> {
        self.scheduler.adopt_external_prefix(request_id, tokens)
    }

    fn configure_external_kv_ranks(&mut self) -> Result<Vec<crate::scheduler::KvMemoryInfo>> {
        let parallel = self.model_runner.parallel_strategy();
        anyhow::ensure!(
            parallel.tp > 0 && parallel.pp == 1,
            "external KV requires PP=1"
        );
        anyhow::ensure!(
            parallel.ep == 1,
            "single_instance EP has no rank-local KV owner; use moe_ep_multi_instance for external KV"
        );
        let mut memory = self.scheduler.configure_external_kv()?;
        memory.compatibility = self.model.kv_compatibility();
        memory.kv_bytes_per_token = self.model.kv_bytes_per_token_per_device();
        memory.kv_bytes_per_request = self.model.kv_bytes_per_request_per_device();
        Ok(vec![memory; parallel.tp as usize])
    }

    fn set_external_kv_rank_bytes(&mut self, bytes: &[u64]) -> Result<()> {
        let parallel = self.model_runner.parallel_strategy();
        anyhow::ensure!(
            parallel.ep == 1 && bytes.len() == parallel.tp as usize,
            "external KV reservation must name every TP rank"
        );
        self.scheduler
            .set_external_kv_bytes(bytes.iter().copied().max().unwrap_or(0))
    }

    fn reserve_external_kv_request(&mut self, _request: &Request) -> Result<Vec<usize>> {
        let parallel = self.model_runner.parallel_strategy();
        anyhow::ensure!(
            parallel.ep == 1,
            "external KV requires rank-local attention"
        );
        Ok((0..parallel.tp as usize).collect())
    }

    fn configure_external_kv(&mut self) -> Result<crate::scheduler::KvMemoryInfo> {
        let parallel = self.model_runner.parallel_strategy();
        anyhow::ensure!(
            parallel.tp == 1 && parallel.pp == 1 && parallel.ep == 1,
            "externally owned KV requires TP=PP=EP=1"
        );
        let mut memory = self.scheduler.configure_external_kv()?;
        memory.compatibility = self.model.kv_compatibility();
        Ok(memory)
    }

    fn set_external_kv_bytes(&mut self, bytes: u64) -> Result<()> {
        self.scheduler.set_external_kv_bytes(bytes)
    }

    fn reserve_external_prefix(&mut self, request_id: u64, tokens: u32) -> Result<()> {
        self.scheduler.reserve_external_prefix(request_id, tokens)
    }

    fn instance_load(&self) -> InstanceLoad {
        std::iter::once(self.scheduler.instance_load())
            .chain(
                self.pending_requests
                    .values()
                    .map(|request| pending_llm_load(request, request.initial.iter().copied())),
            )
            .chain(self.active_requests.values().map(|active| {
                pending_llm_load(&active.request, active.pending_roots.iter().copied())
            }))
            .sum()
    }

    fn init(&mut self) -> Result<()> {
        let parallel = self.model_runner.parallel_strategy();
        self.model.init(parallel)?;
        self.scheduler.init(
            self.model.scheduler_model_weight_bytes(),
            self.model.scheduler_kv_bytes_per_request(),
            self.model.scheduler_kv_bytes_per_token(),
            parallel,
        )?;
        self.model_runner.init(self.model.as_ref())
    }

    fn add_request_arrival(
        &mut self,
        request: Request,
        graph: &mut Graph,
        parent: Option<NodeID>,
        sink: Option<*mut dyn LogicalHandler>,
    ) -> Result<()> {
        request.validate()?;
        if self.pending_requests.contains_key(&request.request_id)
            || self.active_requests.contains_key(&request.request_id)
        {
            return Err(anyhow!(
                "single_instance received duplicate request id {}",
                request.request_id
            ));
        }

        let module = self as *mut Self as *mut dyn LogicalHandler;
        let request_id = request.request_id;
        let arrival_time = request.arrival_time;
        let initial = request.initial.clone();
        let successors = request
            .sub_requests
            .iter()
            .map(|sub_request| sub_request.next().to_vec())
            .collect::<Vec<_>>();
        self.pending_requests.insert(request_id, request);
        if let Some(sink) = sink {
            self.request_sinks.insert(request_id, sink);
        } else {
            self.request_sinks.remove(&request_id);
        }

        let request_arrival = graph.add_logical_node_at(
            module,
            LogicalPayload::RequestArrival { request_id },
            arrival_time,
        );
        if let Some(parent) = parent {
            graph.add_edge(parent, request_arrival)?;
        }

        let mut arrival_nodes = Vec::with_capacity(successors.len());
        let mut completion_nodes = Vec::with_capacity(successors.len());
        for subrequest_index in 0..successors.len() {
            arrival_nodes.push(graph.add_logical_node(
                module,
                LogicalPayload::SubRequestArrival {
                    request_id,
                    subrequest_index,
                },
            ));
            completion_nodes.push(graph.add_logical_node(
                module,
                LogicalPayload::SubRequestComplete {
                    request_id,
                    subrequest_index,
                },
            ));
        }

        for subrequest_index in initial {
            graph.add_edge(request_arrival, arrival_nodes[subrequest_index])?;
        }
        for (subrequest_index, next_indices) in successors.into_iter().enumerate() {
            graph.add_edge(
                arrival_nodes[subrequest_index],
                completion_nodes[subrequest_index],
            )?;
            for next_index in next_indices {
                graph.add_edge(
                    completion_nodes[subrequest_index],
                    arrival_nodes[next_index],
                )?;
            }
        }

        Ok(())
    }

    fn take_results(&mut self) -> Vec<RequestResult> {
        std::mem::take(&mut self.results)
    }

    fn into_results(self: Box<Self>) -> Result<Vec<RequestResult>> {
        if !self.pending_requests.is_empty() {
            bail!(
                "single_instance finished with {} requests still awaiting arrival",
                self.pending_requests.len()
            );
        }
        if !self.active_requests.is_empty() {
            bail!(
                "single_instance finished with {} active requests still incomplete",
                self.active_requests.len()
            );
        }
        if !self.pending_completion_results.is_empty() {
            bail!(
                "single_instance finished with {} queued completion results still pending",
                self.pending_completion_results.len()
            );
        }
        Ok(self.results)
    }
}
