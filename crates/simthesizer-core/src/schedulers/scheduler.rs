use anyhow::Result;

use crate::module::Module;
use crate::types::{
    InstanceLoad, ParallelStrategy, ReadySubRequest, RequestState, SubRequestCompletion,
};

#[derive(Debug, Clone, Copy)]
pub struct KvMemoryInfo {
    pub compatibility: u64,
    pub capacity_bytes: u64,
    pub block_tokens: u32,
    pub kv_bytes_per_token: u64,
    pub kv_bytes_per_request: u64,
}

pub trait Scheduler: Module {
    fn external_kv_budget(&self) -> Result<u64> {
        anyhow::bail!("scheduler does not expose external KV capacity")
    }
    fn external_reclaim_pending(&self) -> bool {
        false
    }
    fn adopt_external_state(&mut self, _request_id: u64) {}
    fn set_external_admission_blocked(&mut self, _request_id: u64, _blocked: bool) {}
    fn set_external_kv_reclaimable_bytes(&mut self, _bytes: u64) {}
    fn set_external_blocked(&mut self, _request_id: u64, _blocked: bool) {}
    fn take_kv_progress(&mut self) -> Vec<(u64, u32)> {
        Vec::new()
    }
    fn take_external_preemptions(&mut self) -> Vec<u64> {
        Vec::new()
    }
    fn cancel_iteration(&mut self) {}
    fn restore_external_prefix(&mut self, _request_id: u64, _tokens: u32) -> Result<()> {
        anyhow::bail!("scheduler does not support external KV restore")
    }
    fn external_kv_requirement(&self) -> Option<(u64, u64)> {
        None
    }
    fn adopt_external_prefix(&mut self, _request_id: u64, _tokens: u32) -> Result<()> {
        anyhow::bail!("scheduler does not support external KV adoption")
    }

    fn configure_external_kv(&mut self) -> Result<KvMemoryInfo> {
        anyhow::bail!("scheduler does not support externally owned KV")
    }

    fn set_external_kv_bytes(&mut self, _bytes: u64) -> Result<()> {
        anyhow::bail!("scheduler does not support externally owned KV")
    }

    fn reserve_external_prefix(&mut self, _request_id: u64, _tokens: u32) -> Result<()> {
        anyhow::bail!("scheduler does not support externally owned KV")
    }

    fn init(
        &mut self,
        model_weight_bytes_per_device: u64,
        kv_bytes_per_request_per_device: u64,
        kv_bytes_per_token_per_device: u64,
        parallel: ParallelStrategy,
    ) -> Result<()> {
        let _ = (
            model_weight_bytes_per_device,
            kv_bytes_per_request_per_device,
            kv_bytes_per_token_per_device,
            parallel,
        );
        Ok(())
    }

    fn schedule(&mut self) -> Result<Vec<RequestState>>;

    fn enqueue_sub_request(&mut self, sub_request: ReadySubRequest);

    fn done_iteration(&mut self, now: f64) -> Vec<SubRequestCompletion>;

    fn instance_load(&self) -> InstanceLoad {
        InstanceLoad::default()
    }
}

crate::declare_registry! {
    registration = SchedulerRegistration,
    trait = crate::scheduler::Scheduler,
    deserialize_fn = deserialize_scheduler_impl,
    registry_name = "scheduler",
}

crate::impl_registry_deserialize! {
    trait = crate::scheduler::Scheduler,
    deserialize_fn = crate::scheduler::deserialize_scheduler_impl,
}

#[macro_export]
macro_rules! register_scheduler {
    ($ty:ty, $kind:expr $(,)?) => {
        $crate::submit_registration!(
            registration = $crate::scheduler::SchedulerRegistration,
            trait = $crate::scheduler::Scheduler,
            type = $ty,
            kind = $kind,
            registry_name = "scheduler",
        );
    };
}
