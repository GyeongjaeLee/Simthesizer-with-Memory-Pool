use anyhow::{bail, ensure, Result};
use serde::{Deserialize, Serialize};

pub type NodeID = u64;
pub type NetworkDeviceID = u64;

/// Input-only identities supplied by a trace, never inferred from token counts.
/// The full list is the first `shared` IDs of an earlier sub-request's full
/// list (`parent`, in the same request) followed by `block_ids`.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct KvPrefix {
    pub namespace: String,
    pub block_tokens: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub parent: Option<usize>,
    #[serde(default, skip_serializing_if = "is_zero")]
    pub shared: usize,
    pub block_ids: Vec<u64>,
}

fn is_zero(value: &usize) -> bool {
    *value == 0
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ProviderTokens {
    pub prompt: u32,
    pub completion: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
pub enum Layer {
    Embedding {
        name: String,
        num_embeddings: u64,
        embedding_dim: u64,
    },
    LayerNorm {
        name: String,
        hidden_size: u64,
    },
    Gemm {
        name: String,
        m: u64,
        n: u64,
    },
    Rope {
        name: String,
        head_dim: u64,
        num_heads: u64,
    },
    Attention {
        name: String,
        num_q_heads: u64,
        num_kv_heads: u64,
        head_dim: u64,
    },
    Mamba {
        name: String,
        num_heads: u64,
        head_dim: u64,
        state_size: u64,
        conv_kernel_size: u64,
    },
    Act {
        name: String,
        width: u64,
    },
    Sampler {
        name: String,
        vocab_size: u64,
    },
    AllReduce {
        name: String,
        bytes_per_token: u64,
    },
    Moe {
        name: String,
        num_experts: u64,
        num_experts_per_token: u32,
        num_shared_experts_per_token: u32,
        dispatch_bytes_per_token: u64,
        combine_bytes_per_token: u64,
    },
}

impl Layer {
    pub fn name(&self) -> &str {
        match self {
            Self::Embedding { name, .. }
            | Self::LayerNorm { name, .. }
            | Self::Gemm { name, .. }
            | Self::Rope { name, .. }
            | Self::Attention { name, .. }
            | Self::Mamba { name, .. }
            | Self::Act { name, .. }
            | Self::Sampler { name, .. }
            | Self::AllReduce { name, .. }
            | Self::Moe { name, .. } => name,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub request_id: u64,
    pub arrival_time: f64,
    pub initial: Vec<usize>,
    pub sub_requests: Vec<SubRequest>,
}

impl Request {
    pub fn total_input_tokens(&self) -> u32 {
        self.sub_requests.iter().map(SubRequest::input_tokens).sum()
    }

    /// Sub-request `index`'s full prefix identity, following `parent` references.
    pub fn resolved_kv_prefix(&self, index: usize) -> Option<KvPrefix> {
        let mut chain = vec![self.sub_requests[index].kv_prefix()?];
        while let Some(parent) = chain.last().unwrap().parent {
            chain.push(self.sub_requests[parent].kv_prefix()?);
        }
        let mut block_ids = Vec::new();
        for prefix in chain.iter().rev() {
            block_ids.truncate(prefix.shared);
            block_ids.extend_from_slice(&prefix.block_ids);
        }
        Some(KvPrefix {
            namespace: chain[0].namespace.clone(),
            block_tokens: chain[0].block_tokens,
            parent: None,
            shared: 0,
            block_ids,
        })
    }

    pub fn total_output_tokens(&self) -> u32 {
        self.sub_requests
            .iter()
            .map(SubRequest::output_tokens)
            .sum()
    }

    pub fn validate(&self) -> Result<()> {
        ensure!(
            self.arrival_time.is_finite() && self.arrival_time >= 0.0,
            "arrival_time must be finite and nonnegative"
        );
        ensure!(
            !self.sub_requests.is_empty(),
            "requests must contain at least one sub-request"
        );
        // Full prefix identity lengths, known for earlier sub-requests.
        let mut prefix_lengths = vec![None; self.sub_requests.len()];
        for (subrequest_index, sub_request) in self.sub_requests.iter().enumerate() {
            ensure!(
                sub_request.session_id().is_none_or(|id| !id.is_empty()),
                "session_id must not be empty"
            );
            for (name, value) in [
                ("interval", sub_request.interval()),
                ("duration", sub_request.duration()),
            ] {
                if let Some(value) = value {
                    ensure!(
                        value.is_finite() && value >= 0.0,
                        "sub-request {subrequest_index} {name} must be finite and nonnegative"
                    );
                }
            }
            if let SubRequest::Llm {
                input_tokens,
                output_tokens,
                known_tokens,
                kv_tokens,
                kv_prefix,
                input_token_ids,
                output_token_ids,
                ..
            } = sub_request
            {
                let total_tokens = input_tokens.checked_add(*output_tokens).ok_or_else(|| {
                    anyhow::anyhow!("llm sub-request {subrequest_index} token count exceeds u32")
                })?;
                for (name, ids, count) in [
                    ("input_token_ids", input_token_ids, input_tokens),
                    ("output_token_ids", output_token_ids, output_tokens),
                ] {
                    ensure!(
                        ids.as_ref().is_none_or(|ids| ids.len() == *count as usize),
                        "{name} length differs from token count"
                    );
                }
                if let Some(prefix) = kv_prefix {
                    ensure!(
                        !prefix.namespace.is_empty(),
                        "KV namespace must not be empty"
                    );
                    ensure!(
                        prefix.block_tokens > 0,
                        "KV identity block size must be positive"
                    );
                    ensure!(
                        input_token_ids.is_none(),
                        "choose token IDs or proxy KV identities"
                    );
                    let shared = match prefix.parent {
                        Some(parent) => {
                            let length = (parent < subrequest_index)
                                .then(|| prefix_lengths[parent])
                                .flatten()
                                .ok_or_else(|| anyhow::anyhow!(
                                    "llm sub-request {subrequest_index} KV parent must be an earlier LLM call with identities"
                                ))?;
                            let parent_prefix = self.sub_requests[parent].kv_prefix().unwrap();
                            ensure!(
                                parent_prefix.namespace == prefix.namespace
                                    && parent_prefix.block_tokens == prefix.block_tokens,
                                "KV parent must share namespace and block size"
                            );
                            ensure!(
                                prefix.shared <= length,
                                "KV identities share more than the parent's"
                            );
                            prefix.shared
                        }
                        None => {
                            ensure!(prefix.shared == 0, "shared KV identities need a parent");
                            0
                        }
                    };
                    let length = shared + prefix.block_ids.len();
                    ensure!(
                        length as u64 * u64::from(prefix.block_tokens) <= u64::from(*input_tokens),
                        "KV identities exceed the input length"
                    );
                    prefix_lengths[subrequest_index] = Some(length);
                }
                match (known_tokens, kv_tokens) {
                    (Some(known_tokens), Some(kv_tokens)) => {
                        ensure!(
                            *known_tokens >= *input_tokens,
                            "llm sub-request {subrequest_index} has known_tokens smaller than input_tokens"
                        );
                        ensure!(
                            *known_tokens <= total_tokens,
                            "llm sub-request {subrequest_index} has known_tokens larger than input_tokens + output_tokens"
                        );
                        ensure!(
                            *kv_tokens <= *known_tokens,
                            "llm sub-request {subrequest_index} has kv_tokens larger than known_tokens"
                        );
                    }
                    (None, None) => {}
                    _ => {
                        bail!(
                            "llm sub-request {subrequest_index} must set both known_tokens and kv_tokens together"
                        );
                    }
                }
            }
        }
        validate_initial_indices(&self.initial, &self.sub_requests)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SubRequestKind {
    Llm,
    ToolCall,
}

#[derive(Debug, Clone, PartialEq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum SubRequest {
    Llm {
        input_tokens: u32,
        output_tokens: u32,
        known_tokens: Option<u32>,
        kv_tokens: Option<u32>,
        next: Vec<usize>,
        interval: Option<f64>,
        model: Option<String>,
        input_token_ids: Option<Vec<u32>>,
        output_token_ids: Option<Vec<u32>>,
        kv_prefix: Option<KvPrefix>,
        session_id: Option<String>,
        provider_tokens: Option<ProviderTokens>,
    },
    ToolCall {
        input_tokens: u32,
        output_tokens: u32,
        next: Vec<usize>,
        duration: Option<f64>,
        interval: Option<f64>,
        session_id: Option<String>,
    },
}

impl SubRequest {
    pub fn kv_prefix(&self) -> Option<&KvPrefix> {
        match self {
            Self::Llm { kv_prefix, .. } => kv_prefix.as_ref(),
            _ => None,
        }
    }

    pub fn session_id(&self) -> Option<&str> {
        match self {
            Self::Llm { session_id, .. } | Self::ToolCall { session_id, .. } => {
                session_id.as_deref()
            }
        }
    }

    pub fn provider_tokens(&self) -> Option<&ProviderTokens> {
        match self {
            Self::Llm {
                provider_tokens, ..
            } => provider_tokens.as_ref(),
            _ => None,
        }
    }

    pub fn kind(&self) -> SubRequestKind {
        match self {
            Self::Llm { .. } => SubRequestKind::Llm,
            Self::ToolCall { .. } => SubRequestKind::ToolCall,
        }
    }

    pub fn input_tokens(&self) -> u32 {
        match self {
            Self::Llm { input_tokens, .. } | Self::ToolCall { input_tokens, .. } => *input_tokens,
        }
    }

    pub fn output_tokens(&self) -> u32 {
        match self {
            Self::Llm { output_tokens, .. } | Self::ToolCall { output_tokens, .. } => {
                *output_tokens
            }
        }
    }

    pub fn next(&self) -> &[usize] {
        match self {
            Self::Llm { next, .. } | Self::ToolCall { next, .. } => next,
        }
    }

    pub fn duration(&self) -> Option<f64> {
        match self {
            Self::Llm { .. } => None,
            Self::ToolCall { duration, .. } => *duration,
        }
    }

    pub fn interval(&self) -> Option<f64> {
        match self {
            Self::Llm { interval, .. } | Self::ToolCall { interval, .. } => *interval,
        }
    }

    pub fn model(&self) -> Option<&str> {
        match self {
            Self::Llm { model, .. } => model.as_deref(),
            Self::ToolCall { .. } => None,
        }
    }

    pub fn input_token_ids(&self) -> Option<&[u32]> {
        match self {
            Self::Llm {
                input_token_ids, ..
            } => input_token_ids.as_deref(),
            Self::ToolCall { .. } => None,
        }
    }

    pub fn output_token_ids(&self) -> Option<&[u32]> {
        match self {
            Self::Llm {
                output_token_ids, ..
            } => output_token_ids.as_deref(),
            Self::ToolCall { .. } => None,
        }
    }

    pub fn known_tokens(&self) -> Option<u32> {
        match self {
            Self::Llm { known_tokens, .. } => *known_tokens,
            Self::ToolCall { .. } => None,
        }
    }

    pub fn kv_tokens(&self) -> Option<u32> {
        match self {
            Self::Llm { kv_tokens, .. } => *kv_tokens,
            Self::ToolCall { .. } => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ReadySubRequest {
    pub request_id: u64,
    pub subrequest_index: usize,
    pub completion_node_id: NodeID,
    pub arrival_time: f64,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub known_tokens: u32,
    pub kv_tokens: u32,
    pub input_token_ids: Option<Vec<u32>>,
    pub output_token_ids: Option<Vec<u32>>,
    pub kv_prefix: Option<KvPrefix>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SubRequestCompletion {
    pub request_id: u64,
    pub subrequest_index: usize,
    pub completion_node_id: NodeID,
    pub token_latency: Vec<f64>,
    /// Time from enqueue to the first iteration that included the call.
    pub queue_wait: Option<f64>,
}

/// Where an LLM call's time to first token went, in seconds; the parts sum
/// to the first `token_latency` entry.
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct TtftParts {
    /// Waiting for KV reloads.
    pub kv_load: f64,
    /// Waiting for HBM to hold the reloaded prefix.
    pub admission: f64,
    /// In the worker's queue before its first iteration.
    pub queue: f64,
    /// From the first iteration to the first token, including iterations
    /// shared with other calls.
    pub prefill: f64,
}

impl TtftParts {
    /// Scheduler-side parts of a call's first token latency.
    pub fn scheduled(ttft: Option<f64>, queue_wait: Option<f64>) -> Option<Self> {
        let (ttft, queue) = (ttft?, queue_wait?);
        Some(Self {
            kv_load: 0.0,
            admission: 0.0,
            queue,
            prefill: ttft - queue,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestState {
    pub q_len: u32,
    pub kv_len: u32,
    pub lm_head_len: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParallelStrategy {
    pub tp: u32,
    pub pp: u32,
    pub ep: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct InstanceLoad {
    pub waiting_requests: usize,
    pub running_requests: usize,
    pub waiting_tokens: u64,
    pub running_tokens: u64,
}

impl InstanceLoad {
    pub fn total_requests(self) -> usize {
        self.waiting_requests + self.running_requests
    }

    pub fn total_tokens(self) -> u64 {
        self.waiting_tokens + self.running_tokens
    }
}

impl std::iter::Sum for InstanceLoad {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::default(), |a, b| Self {
            waiting_requests: a.waiting_requests + b.waiting_requests,
            running_requests: a.running_requests + b.running_requests,
            waiting_tokens: a.waiting_tokens + b.waiting_tokens,
            running_tokens: a.running_tokens + b.running_tokens,
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ComputeContext {
    pub moe_activated_experts: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ResultSubRequest {
    Llm {
        input_tokens: u32,
        output_tokens: u32,
        next: Vec<usize>,
        token_latency: Vec<f64>,
        interval: Option<f64>,
        model: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        session_id: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        provider_tokens: Option<ProviderTokens>,
        #[serde(skip_serializing_if = "Option::is_none")]
        kv_prefix: Option<KvPrefix>,
        #[serde(skip_serializing_if = "Option::is_none")]
        ttft_parts: Option<TtftParts>,
        /// Prompt tokens restored from cached KV instead of prefilled.
        #[serde(skip_serializing_if = "Option::is_none")]
        reused_tokens: Option<u32>,
    },
    ToolCall {
        input_tokens: u32,
        output_tokens: u32,
        next: Vec<usize>,
        duration: f64,
        interval: Option<f64>,
        #[serde(skip_serializing_if = "Option::is_none")]
        session_id: Option<String>,
    },
}

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct RequestResult {
    pub request_id: u64,
    pub arrival_time: f64,
    pub initial: Vec<usize>,
    pub sub_requests: Vec<ResultSubRequest>,
}

fn validate_initial_indices(initial: &[usize], sub_requests: &[SubRequest]) -> Result<()> {
    let mut seen_initial = vec![false; sub_requests.len()];
    for &index in initial {
        if index >= sub_requests.len() {
            bail!(
                "initial sub-request index {index} is out of range for {} sub-requests",
                sub_requests.len()
            );
        }
        ensure!(
            !seen_initial[index],
            "initial sub-request index {index} is duplicated"
        );
        seen_initial[index] = true;
    }

    let mut indegree = vec![0usize; sub_requests.len()];
    for (subrequest_index, sub_request) in sub_requests.iter().enumerate() {
        for &next_index in sub_request.next() {
            if next_index >= sub_requests.len() {
                bail!("sub-request {subrequest_index} references invalid successor {next_index}");
            }
            indegree[next_index] += 1;
        }
    }

    let derived_initial = indegree
        .iter()
        .enumerate()
        .filter_map(|(index, &degree)| (degree == 0).then_some(index))
        .collect::<Vec<_>>();
    ensure!(
        initial == derived_initial,
        "initial sub-request indices {initial:?} do not match DAG roots {derived_initial:?}"
    );

    let mut ready = derived_initial;
    let mut visited = 0;
    while let Some(index) = ready.pop() {
        visited += 1;
        for &next in sub_requests[index].next() {
            indegree[next] -= 1;
            if indegree[next] == 0 {
                ready.push(next);
            }
        }
    }
    ensure!(
        visited == sub_requests.len(),
        "sub-request graph contains a cycle"
    );

    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PollResult {
    Pending {
        latency: f64,
    },
    Complete {
        latency: f64,
    },
    /// Poll again once every other event at this time has run.
    Deferred,
}
