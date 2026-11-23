use anyhow::Result;

use crate::module::Module;
use crate::types::{Layer, ParallelStrategy};

pub trait Model: Module {
    /// Concrete architecture/layout identity; checkpoint identity belongs to the input namespace.
    fn kv_compatibility(&self) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::any::type_name::<Self>().hash(&mut hasher);
        serde_json::to_string(&(
            self.pre_block_layers(),
            self.block_layers(),
            self.post_block_layers(),
            self.num_blocks(),
            self.weight_bytes_per_device(),
            self.kv_bytes_per_token_per_device(),
            self.kv_bytes_per_request_per_device(),
        ))
        .expect("model geometry serializes")
        .hash(&mut hasher);
        hasher.finish()
    }
    fn init(&mut self, parallel: ParallelStrategy) -> Result<()>;
    fn weight_bytes_per_device(&self) -> u64;
    fn kv_bytes_per_request_per_device(&self) -> u64 {
        0
    }
    fn kv_bytes_per_token_per_device(&self) -> u64;
    fn scheduler_model_weight_bytes(&self) -> u64 {
        self.weight_bytes_per_device()
    }
    fn scheduler_kv_bytes_per_request(&self) -> u64 {
        0
    }
    fn scheduler_kv_bytes_per_token(&self) -> u64 {
        self.kv_bytes_per_token_per_device()
    }
    fn pre_block_layers(&self) -> &[Layer];
    fn block_layers(&self) -> &[Layer];
    fn post_block_layers(&self) -> &[Layer];
    fn num_blocks(&self) -> u32;
}

crate::declare_registry! {
    registration = ModelRegistration,
    trait = crate::model::Model,
    deserialize_fn = deserialize_model_impl,
    registry_name = "model",
}

crate::impl_registry_deserialize! {
    trait = crate::model::Model,
    deserialize_fn = crate::model::deserialize_model_impl,
}

#[macro_export]
macro_rules! register_model {
    ($ty:ty, $kind:expr $(,)?) => {
        $crate::submit_registration!(
            registration = $crate::model::ModelRegistration,
            trait = $crate::model::Model,
            type = $ty,
            kind = $kind,
            registry_name = "model",
        );
    };
}
