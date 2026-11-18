mod common;

use common::{model, request, single, take_compute_calls};
use serde_json::{json, Value};
use simthesizer_core::types::{Layer, Request};
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
fn tp_geometry_preserves_shards_and_replicated_kv_heads() {
    // Two 8-element KV heads, K+V, BF16: 64 B/token at TP1.
    // TP2 shards heads; TP4 replicates one head on each rank.
    for (tp, per_rank, physical_total) in [(1, 64, 64), (2, 32, 64), (4, 32, 128)] {
        let mut config = single();
        config["model_runner"]["tp"] = json!(tp);
        let mut worker = system(config);
        let memory = worker.configure_external_kv_ranks().unwrap();
        assert_eq!(memory.len(), tp);
        assert!(memory
            .iter()
            .all(|rank| rank.kv_bytes_per_token == per_rank));
        assert_eq!(
            memory
                .iter()
                .map(|rank| rank.kv_bytes_per_token)
                .sum::<u64>(),
            physical_total
        );
        assert_eq!(
            worker
                .reserve_external_kv_request(&request(0, 0.0, 1))
                .unwrap(),
            (0..tp).collect::<Vec<_>>()
        );
        assert!(worker.set_external_kv_rank_bytes(&vec![0; tp + 1]).is_err());
        let mut bytes = vec![0; tp];
        bytes[tp - 1] = memory[tp - 1].capacity_bytes + 1;
        assert!(worker.set_external_kv_rank_bytes(&bytes).is_err());
    }
}

#[test]
fn tp_admission_reserves_the_most_occupied_rank() {
    let mut config = single();
    config["model_runner"]["tp"] = json!(2);
    let mut worker = system(config);
    let memory = worker.configure_external_kv_ranks().unwrap();
    worker
        .set_external_kv_rank_bytes(&[0, memory[1].capacity_bytes])
        .unwrap();
    let mut graph = Graph::new();
    worker
        .add_request_arrival(request(0, 0.0, 1), &mut graph, None, None)
        .unwrap();
    let error = Engine::new(graph, worker).run().unwrap_err();
    assert!(error.to_string().contains("available HBM"));
}

#[test]
fn ep_reservations_reach_load_balancing_and_reject_duplicates() {
    let mut config = ep();
    config["router"] = json!({"kind":"moe_load_balancing"});
    let mut worker = system(config);
    let memory = worker.configure_external_kv_ranks().unwrap();
    assert_eq!(memory.len(), 2);
    assert!(memory.iter().all(|rank| rank.kv_bytes_per_token == 64));
    let mut large = serde_json::to_value(request(0, 0.0, 1)).unwrap();
    large["sub_requests"][0]["input_tokens"] = json!(100);
    let large: Request = serde_json::from_value(large).unwrap();
    assert_eq!(worker.reserve_external_kv_request(&large).unwrap(), vec![0]);
    assert!(worker.reserve_external_kv_request(&large).is_err());
    assert_eq!(
        worker
            .reserve_external_kv_request(&request(1, 0.0, 1))
            .unwrap(),
        vec![1]
    );
    assert_eq!(
        worker
            .reserve_external_kv_request(&request(2, 0.0, 1))
            .unwrap(),
        vec![1]
    );
    assert_eq!(worker.instance_load().waiting_tokens, 108);
    let mut graph = Graph::new();
    worker
        .add_request_arrival(large, &mut graph, None, None)
        .unwrap();
    assert_eq!(worker.instance_load().waiting_requests, 3);
    assert_eq!(worker.instance_load().waiting_tokens, 108);
    assert!(worker.reserve_external_prefix(99, 4).is_err());
    assert!(worker.into_results().is_err());
}

#[test]
fn ep_keeps_reserved_attention_ranks_when_reloads_arrive_out_of_order() {
    take_compute_calls();
    let mut worker = system(ep());
    worker.configure_external_kv_ranks().unwrap();
    worker.set_external_kv_rank_bytes(&[256, 0]).unwrap();
    let first: Request = serde_json::from_value(json!({"request_id":0,"arrival_time":10.0,"initial":[0],
        "sub_requests":[{"kind":"llm","input_tokens":8,"output_tokens":1,"known_tokens":8,"kv_tokens":4,"next":[]}]})).unwrap();
    let second = request(1, 0.0, 3);
    assert_eq!(worker.reserve_external_kv_request(&first).unwrap(), vec![0]);
    assert_eq!(
        worker.reserve_external_kv_request(&second).unwrap(),
        vec![1]
    );
    worker.reserve_external_prefix(0, 4).unwrap();
    worker.reserve_external_prefix(1, 0).unwrap();
    let mut graph = Graph::new();
    worker
        .add_request_arrival(first, &mut graph, None, None)
        .unwrap();
    worker
        .add_request_arrival(second, &mut graph, None, None)
        .unwrap();
    let mut final_state = Value::Null;
    let results = Engine::new(graph, worker)
        .run_with_observer(|_, worker| {
            final_state = worker.observe();
            Ok(())
        })
        .unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(final_state["schedulers"][0]["generation_tokens_total"], 1);
    assert_eq!(final_state["schedulers"][1]["generation_tokens_total"], 3);
    assert!(take_compute_calls()
        .iter()
        .any(|call| matches!(call.layer, Layer::Attention { .. })
            && call
                .batch
                .iter()
                .any(|request| request.kv_len == 4 && request.q_len == 4)));
}

#[test]
fn ep_capacity_is_per_rank_and_idle_rank_does_not_block_attention() {
    let mut worker = system(ep());
    let memory = worker.configure_external_kv_ranks().unwrap();
    assert!(worker.set_external_kv_rank_bytes(&[0]).is_err());
    assert!(worker
        .set_external_kv_rank_bytes(&[0, memory[1].capacity_bytes + 1])
        .is_err());
    worker
        .set_external_kv_rank_bytes(&[memory[0].capacity_bytes, 0])
        .unwrap();
    let tool: Request = serde_json::from_value(json!({"request_id":0,"arrival_time":0.0,"initial":[0],
        "sub_requests":[{"kind":"tool_call","input_tokens":0,"output_tokens":0,"duration":0.0,"next":[]}]})).unwrap();
    let llm = request(1, 0.0, 1);
    assert_eq!(worker.reserve_external_kv_request(&tool).unwrap(), vec![0]);
    assert_eq!(worker.reserve_external_kv_request(&llm).unwrap(), vec![1]);
    worker.reserve_external_prefix(1, 0).unwrap();
    let mut graph = Graph::new();
    for request in [tool, llm] {
        worker
            .add_request_arrival(request, &mut graph, None, None)
            .unwrap();
    }
    assert_eq!(Engine::new(graph, worker).run().unwrap().len(), 2);
}

#[test]
fn aggregate_ep_explains_the_rank_local_configuration() {
    let mut config = single();
    config["model_runner"] = ep()["model_runner"].clone();
    let error = system(config).configure_external_kv_ranks().unwrap_err();
    assert!(error.to_string().contains("use moe_ep_multi_instance"));
}
