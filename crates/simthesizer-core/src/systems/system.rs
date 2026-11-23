use anyhow::Result;

use crate::engine::Graph;
use crate::module::{LogicalHandler, Module};
use crate::types::{InstanceLoad, NodeID, Request, RequestResult, SubRequest};

pub enum SystemEvent {
    KvPreempted {
        request_id: u64,
    },
    KvProgress {
        request_id: u64,
        computed_tokens: u32,
    },
    KvCapacity {
        request_id: u64,
        ranks: Vec<usize>,
        required_bytes: u64,
        continuation: NodeID,
    },
    RequestCompleted {
        request: Request,
        result: RequestResult,
    },
    /// A pending request reached its turn: look up and start loading its
    /// external prefix.
    KvLookup {
        request: u64,
    },
}

pub trait System: Module + LogicalHandler {
    /// LLM calls completed so far and the sum of their first-token waits
    /// (TTFT minus prefill), or `None` if the system does not report them.
    fn first_token_waits(&self) -> Option<(u64, f64)> {
        None
    }

    fn next_kv_capacity_event(&self, _rank: usize) -> Option<NodeID> {
        None
    }

    fn external_kv_budget(&self, _rank: usize) -> Result<u64> {
        anyhow::bail!("system does not expose external KV capacity")
    }
    fn adopt_external_state(&mut self, _request_id: u64) -> Result<()> {
        Ok(())
    }
    fn set_external_admission_blocked(&mut self, _request_id: u64, _blocked: bool) -> Result<()> {
        Ok(())
    }
    fn set_external_kv_reclaimable_bytes(&mut self, _bytes: &[u64]) -> Result<()> {
        Ok(())
    }
    fn set_external_blocked(&mut self, _request_id: u64, _blocked: bool) -> Result<()> {
        Ok(())
    }
    fn external_kv_requirement(&self, _ranks: &[usize]) -> Option<(u64, u64)> {
        None
    }
    /// See `Scheduler::known_tokens`.
    fn known_tokens(&self, _request_id: u64) -> Option<u32> {
        None
    }
    /// See `Scheduler::external_admission_bytes`, for one rank.
    fn external_admission_bytes(
        &self,
        _rank: usize,
        _known_tokens: u32,
        _restored_tokens: u32,
    ) -> u64 {
        0
    }
    fn reserve_external_admission(
        &mut self,
        _request_id: u64,
        _rank: usize,
        _bytes: u64,
    ) -> Result<()> {
        Ok(())
    }
    fn set_external_pending(&mut self, _request_id: u64, _pending: bool) -> Result<()> {
        Ok(())
    }
    /// Schedules again after `after`, e.g. once an external prefix arrived.
    fn wake(&mut self, _after: NodeID, _graph: &mut Graph) -> Result<()> {
        Ok(())
    }
    fn restore_external_prefix(&mut self, _request_id: u64, _tokens: u32) -> Result<()> {
        anyhow::bail!("system does not support external KV restore")
    }
    fn adopt_external_prefix(&mut self, _request_id: u64, _tokens: u32) -> Result<()> {
        anyhow::bail!("system does not support external KV adoption")
    }
    fn configure_external_kv_ranks(&mut self) -> Result<Vec<crate::scheduler::KvMemoryInfo>> {
        Ok(vec![self.configure_external_kv()?])
    }

    fn set_external_kv_rank_bytes(&mut self, bytes: &[u64]) -> Result<()> {
        anyhow::ensure!(bytes.len() == 1, "system requires one external KV rank");
        self.set_external_kv_bytes(bytes[0])
    }

    fn reserve_external_kv_request(&mut self, _request: &Request) -> Result<Vec<usize>> {
        Ok(vec![0])
    }

    fn configure_external_kv(&mut self) -> Result<crate::scheduler::KvMemoryInfo> {
        anyhow::bail!("system does not support externally owned KV")
    }

    fn set_external_kv_bytes(&mut self, _bytes: u64) -> Result<()> {
        anyhow::bail!("system does not support externally owned KV")
    }

    fn reserve_external_prefix(&mut self, _request_id: u64, _tokens: u32) -> Result<()> {
        anyhow::bail!("system does not support externally owned KV")
    }

    fn init(&mut self) -> Result<()>;

    /// Scheduler work plus assigned initial LLM calls awaiting enqueue.
    fn instance_load(&self) -> InstanceLoad;

    fn add_request_arrival(
        &mut self,
        request: Request,
        graph: &mut Graph,
        parent: Option<NodeID>,
        sink: Option<*mut dyn LogicalHandler>,
    ) -> Result<()>;

    /// Moves out requests completed since the last call; `into_results`
    /// returns the rest.
    fn take_results(&mut self) -> Vec<RequestResult> {
        Vec::new()
    }

    fn into_results(self: Box<Self>) -> Result<Vec<RequestResult>>;
}

pub(super) fn pending_llm_load(
    request: &Request,
    indices: impl IntoIterator<Item = usize>,
) -> InstanceLoad {
    indices
        .into_iter()
        .filter_map(|index| match &request.sub_requests[index] {
            SubRequest::Llm {
                input_tokens,
                known_tokens,
                kv_tokens,
                ..
            } => Some(InstanceLoad {
                waiting_requests: 1,
                waiting_tokens: u64::from(
                    known_tokens
                        .unwrap_or(*input_tokens)
                        .saturating_sub(kv_tokens.unwrap_or(0))
                        .max(1),
                ),
                ..Default::default()
            }),
            _ => None,
        })
        .sum()
}

crate::declare_registry! {
    registration = SystemRegistration,
    trait = crate::system::System,
    deserialize_fn = deserialize_system_impl,
    registry_name = "system",
}

crate::impl_registry_deserialize! {
    trait = crate::system::System,
    deserialize_fn = crate::system::deserialize_system_impl,
}

#[macro_export]
macro_rules! register_system {
    ($ty:ty, $kind:expr $(,)?) => {
        $crate::submit_registration!(
            registration = $crate::system::SystemRegistration,
            trait = $crate::system::System,
            type = $ty,
            kind = $kind,
            registry_name = "system",
        );
    };
}
