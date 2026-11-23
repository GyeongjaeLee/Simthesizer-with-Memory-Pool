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

    fn next_kv_capacity_event(&self) -> Option<NodeID> {
        None
    }

    /// External KV is in worker bytes: a worker's GPUs form one KV domain.
    fn external_kv_budget(&self) -> Result<u64> {
        anyhow::bail!("system does not expose external KV capacity")
    }
    fn adopt_external_state(&mut self, _request_id: u64) -> Result<()> {
        Ok(())
    }
    fn set_external_admission_blocked(&mut self, _request_id: u64, _blocked: bool) -> Result<()> {
        Ok(())
    }
    fn set_external_kv_reclaimable_bytes(&mut self, _bytes: u64) -> Result<()> {
        Ok(())
    }
    fn set_external_blocked(&mut self, _request_id: u64, _blocked: bool) -> Result<()> {
        Ok(())
    }
    fn external_kv_requirement(&self) -> Option<(u64, u64)> {
        None
    }
    /// See `Scheduler::known_tokens`.
    fn known_tokens(&self, _request_id: u64) -> Option<u32> {
        None
    }
    /// See `Scheduler::external_admission_bytes`.
    fn external_admission_bytes(&self, _known_tokens: u32, _restored_tokens: u32) -> u64 {
        0
    }
    fn reserve_external_admission(&mut self, _request_id: u64, _bytes: u64) -> Result<()> {
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
