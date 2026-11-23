use std::any::Any;

use anyhow::Result;

use crate::module::Module;
use crate::types::{NetworkDeviceID, NodeID, PollResult};

pub trait NetworkSimulator: Module {
    fn add_device(&mut self) -> NetworkDeviceID;

    fn add_link(
        &mut self,
        src: NetworkDeviceID,
        dst: NetworkDeviceID,
        bandwidth: f64,
        latency: f64,
    ) -> Result<()>;

    fn simulate(
        &mut self,
        node: NodeID,
        src: NetworkDeviceID,
        dst: NetworkDeviceID,
        bytes: u64,
        now: f64,
        state: &mut Option<Box<dyn Any>>,
    ) -> Result<PollResult>;

    /// Pending nodes to poll earlier than answered, as `(node, time)`. The
    /// engine asks after every poll; a wrapper must forward it.
    fn take_wakeups(&mut self) -> Vec<(NodeID, f64)> {
        Vec::new()
    }
}

crate::declare_registry! {
    registration = NetworkSimulatorRegistration,
    trait = crate::network_sim::NetworkSimulator,
    deserialize_fn = deserialize_network_simulator_impl,
    registry_name = "network_sim",
}

crate::impl_registry_deserialize! {
    trait = crate::network_sim::NetworkSimulator,
    deserialize_fn = crate::network_sim::deserialize_network_simulator_impl,
}

#[macro_export]
macro_rules! register_network_simulator {
    ($ty:ty, $kind:expr $(,)?) => {
        $crate::submit_registration!(
            registration = $crate::network_sim::NetworkSimulatorRegistration,
            trait = $crate::network_sim::NetworkSimulator,
            type = $ty,
            kind = $kind,
            registry_name = "network_sim",
        );
    };
}
