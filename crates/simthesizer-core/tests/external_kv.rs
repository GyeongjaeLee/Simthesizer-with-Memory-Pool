mod common;

use common::{model, request, single};
use serde_json::{json, Value};
use simthesizer_core::{Engine, Graph, System};

fn ep() -> Value {
    let child = single();
    let mut runner = child["model_runner"].clone();
    runner["kind"] = json!("moe_ep");
    runner.as_object_mut().unwrap().remove("tp");
    runner["ep"] = json!(2);
    json!({"kind":"moe_ep_multi_instance", "model":model(),
        "schedulers":[child["scheduler"], child["scheduler"]], "model_runner":runner})
}

fn system(config: Value) -> Box<dyn System> {
    let mut system: Box<dyn System> = serde_json::from_value(config).unwrap();
    system.init().unwrap();
    system
}

#[test]
fn tp_worker_is_one_kv_domain_of_all_its_gpus() {
    // Two 8-element KV heads, K+V, BF16: 64 B/token at TP1.
    // TP2 shards heads; TP4 replicates one head on each GPU.
    // HBM is every GPU's free memory after its weight shard.
    for (tp, worker_bytes) in [(1, 64), (2, 64), (4, 128)] {
        let mut config = single();
        config["model_runner"]["tp"] = json!(tp);
        let mut worker = system(config);
        let memory = worker.configure_external_kv().unwrap();
        assert_eq!(memory.kv_bytes_per_token, worker_bytes);
        assert_eq!(memory.capacity_bytes % tp, 0);
        assert!(memory.capacity_bytes > (tp - 1) * 1048576);
        assert_eq!(worker.external_kv_budget().unwrap(), memory.capacity_bytes);
        assert!(worker
            .set_external_kv_bytes(memory.capacity_bytes + 1)
            .is_err());
    }
}

#[test]
fn full_worker_hbm_blocks_admission() {
    let mut config = single();
    config["model_runner"]["tp"] = json!(2);
    let mut worker = system(config);
    let memory = worker.configure_external_kv().unwrap();
    worker.set_external_kv_bytes(memory.capacity_bytes).unwrap();
    let mut graph = Graph::new();
    worker
        .add_request_arrival(request(0, 0.0, 1), &mut graph, None, None)
        .unwrap();
    let error = Engine::new(graph, worker).run().unwrap_err();
    assert!(error.to_string().contains("available HBM"));
}

#[test]
fn ep_systems_own_no_external_kv() {
    assert!(system(ep()).configure_external_kv().is_err());
    let mut config = single();
    config["model_runner"] = ep()["model_runner"].clone();
    let error = system(config).configure_external_kv().unwrap_err();
    assert!(error.to_string().contains("PP=EP=1"));
}
