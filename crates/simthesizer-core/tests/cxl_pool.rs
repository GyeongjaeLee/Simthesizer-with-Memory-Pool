mod common;

use common::{close, single, take_compute_calls};
use serde_json::{json, Value};
use simthesizer_core::types::{Request, RequestResult, ResultSubRequest};
use simthesizer_core::{Engine, Graph, System};

fn topology() -> Value {
    let mut endpoints = vec![json!({"id":"fabric","kind":"fabric"})];
    let mut links = Vec::new();
    let mut workers = Vec::new();
    let mut link = |a: String, b: String, width: Option<u32>, reverse: bool| {
        let id = format!("{a}-{b}");
        let mut v = json!({"id":id,"a":a,"b":b,"a_port":format!("{id}:a"),"b_port":format!("{id}:b"),"bandwidth_ab":1e6,"latency":0.0});
        if let Some(width) = width {
            v["width"] = json!(width);
        }
        if reverse {
            v["bandwidth_ba"] = json!(1e6);
        }
        links.push(v);
    };
    for i in 0..2 {
        endpoints.push(json!({"id":format!("leaf{i}"),"kind":"switch","lane_budget":256}));
        endpoints
            .push(json!({"id":format!("ingress{i}"),"kind":"junction","node":format!("node{i}")}));
        for kind in ["adapter", "dram", "nic", "gpu"] {
            endpoints
                .push(json!({"id":format!("{kind}{i}"),"kind":kind,"node":format!("node{i}")}));
        }
        link(format!("adapter{i}"), format!("leaf{i}"), Some(16), true);
        link(format!("dram{i}"), format!("adapter{i}"), None, false);
        link(format!("adapter{i}"), format!("ingress{i}"), None, false);
        link(format!("ingress{i}"), format!("gpu{i}"), None, false);
        link(format!("dram{i}"), format!("nic{i}"), None, true);
        link(format!("nic{i}"), "fabric".into(), None, true);
        link(format!("nic{i}"), format!("ingress{i}"), None, false);
        link(format!("gpu{i}"), format!("nic{i}"), None, false);
        workers.push(json!({"worker":i,"node":format!("node{i}"),"adapter":format!("adapter{i}"),"dram":format!("dram{i}"),"nic":format!("nic{i}"),"gpu":format!("gpu{i}")}));
        for d in i * 4..i * 4 + 4 {
            endpoints.push(json!({"id":format!("D{d}"),"kind":"memory","capacity_bytes":64000000000u64,"read_bandwidth":1e9,"write_bandwidth":1e9}));
            link(format!("leaf{i}"), format!("D{d}"), Some(8), true);
        }
    }
    link("leaf0".into(), "leaf1".into(), Some(16), true);
    json!({"endpoints":endpoints,"links":links,"workers":workers})
}
fn keys(n: usize) -> Vec<String> {
    (0..n).map(|i| format!("fixture:B{i}")).collect()
}
fn config(devices: Vec<String>, pressure: bool) -> Value {
    let mut worker = single();
    worker["scheduler"]["block_size"] = json!(1);
    worker["scheduler"]["max_num_batched_tokens"] = json!(64);
    json!({"kind":"cxl_pool","instances":[worker.clone(),worker],"topology":topology(),"block_tokens":1,"block_bytes":64,"host_capacity_bytes":{"node0":1024,"node1":1024},"placement_devices":devices,"max_outstanding_reads":16,"calls":[{"request_id":0,"subrequest_index":0,"worker":0,"seed_host_blocks":keys(16)},{"request_id":0,"subrequest_index":2,"worker":1,"blocks":keys(16)}],"pressures":if pressure {json!([{"request_id":0,"subrequest_index":1,"owner_request_id":1,"node":"node0","bytes":512}])}else{json!([])}})
}
fn workload(tool: f64) -> Vec<Request> {
    vec![serde_json::from_value(json!({"request_id":0,"arrival_time":0.0,"initial":[0],"sub_requests":[{"kind":"llm","input_tokens":17,"output_tokens":1,"next":[1]},{"kind":"tool_call","input_tokens":0,"output_tokens":0,"duration":tool,"next":[2]},{"kind":"llm","input_tokens":17,"output_tokens":1,"next":[]}]})).unwrap(),serde_json::from_value(json!({"request_id":1,"arrival_time":0.0,"initial":[0],"sub_requests":[{"kind":"tool_call","input_tokens":0,"output_tokens":0,"duration":10.0,"next":[]}]})).unwrap()]
}
fn run(mut config: Value, requests: Vec<Request>) -> (Vec<RequestResult>, Value, Vec<Value>) {
    config["detailed_observations"] = json!(true);
    let mut system: Box<dyn System> = serde_json::from_value(config).unwrap();
    system.init().unwrap();
    let mut graph = Graph::new();
    for request in requests {
        system
            .add_request_arrival(request, &mut graph, None, None)
            .unwrap();
    }
    let mut snapshots = vec![];
    let result = Engine::new(graph, system)
        .run_with_observer(|time, state| {
            snapshots.push(json!({"time":time,"state":state.observe()}));
            Ok(())
        })
        .unwrap();
    let final_state = snapshots.last().unwrap()["state"]["cxl_pool"].clone();
    (result, final_state, snapshots)
}
fn bytes(state: &Value, kind: &str) -> u64 {
    state["transfers"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|t| t["kind"] == kind)
        .map(|t| t["bytes"].as_u64().unwrap())
        .sum()
}
#[test]
fn spill_publishes_inside_tool_then_reload_gates_native_llm() {
    take_compute_calls();
    let (results, state, snapshots) = run(
        config((0..8).map(|i| format!("D{i}")).collect(), true),
        workload(1.0),
    );
    assert_eq!(bytes(&state, "spill"), 512);
    assert_eq!(bytes(&state, "cxl_read"), 512);
    assert_eq!(bytes(&state, "rdma_read"), 512);
    for t in state["transfers"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|t| t["kind"] == "spill")
    {
        close(t["start"].as_f64().unwrap(), 1.0);
        assert!(t["completion"].as_f64().unwrap() < 2.0);
    }
    let ready = state["events"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["event"] == "kv_ready" && e["subrequest_index"] == 2)
        .unwrap();
    assert!(ready["time"].as_f64().unwrap() > 2.0);
    let request = results.iter().find(|r| r.request_id == 0).unwrap();
    let ResultSubRequest::Llm { token_latency, .. } = &request.sub_requests[2] else {
        panic!()
    };
    close(
        token_latency[0],
        1.0 + ready["kv_preparation_wait"].as_f64().unwrap(),
    );
    assert_eq!(
        take_compute_calls()
            .iter()
            .filter(|c| c.layer.name() == "attention")
            .map(|c| c.batch[0].q_len)
            .collect::<Vec<_>>(),
        vec![17, 1]
    );
    assert!(snapshots
        .iter()
        .any(|s| s["state"]["cxl_pool"]["blocks"]
            .as_array()
            .is_some_and(|blocks| blocks
                .iter()
                .any(|b| b["pending"]["tier"] == "pool" && b["location"]["tier"] == "host"))));
    assert_eq!(state["unfinished_requests"], 0);
    assert_eq!(state["unfinished_transfers"], 0);
}
#[test]
fn placement_preserves_payload_and_changes_boundary_direction() {
    for (devices, write_remote, read_remote) in [(0..8, 256, 256), (0..2, 0, 512), (4..6, 512, 0)] {
        let (_, state, _) = run(
            config(devices.map(|i| format!("D{i}")).collect(), true),
            workload(1.0),
        );
        let transfers = state["transfers"].as_array().unwrap();
        let remote_write: u64 = transfers
            .iter()
            .filter(|t| {
                t["kind"] == "spill" && t["dst"].as_str().unwrap()[1..].parse::<u32>().unwrap() >= 4
            })
            .map(|t| t["bytes"].as_u64().unwrap())
            .sum();
        let remote_read: u64 = transfers
            .iter()
            .filter(|t| {
                t["kind"] == "cxl_read"
                    && t["src"].as_str().unwrap()[1..].parse::<u32>().unwrap() < 4
            })
            .map(|t| t["bytes"].as_u64().unwrap())
            .sum();
        assert_eq!((remote_write, remote_read), (write_remote, read_remote));
        assert!(state["blocks"]
            .as_array()
            .unwrap()
            .iter()
            .all(|b| b["location"]["tier"] == "hbm"
                && b["location"]["worker"] == 1
                && b["ready"] == true));
    }
}
#[test]
fn early_resume_waits_for_published_pool_copy() {
    let (_, state, _) = run(config(vec!["D0".into(), "D1".into()], true), workload(0.0));
    let transfers = state["transfers"].as_array().unwrap();
    let last_write = transfers
        .iter()
        .filter(|t| t["kind"] == "spill")
        .map(|t| t["completion"].as_f64().unwrap())
        .fold(0.0, f64::max);
    assert!(transfers
        .iter()
        .filter(|t| t["kind"] == "cxl_read")
        .all(|t| t["start"].as_f64().unwrap() >= last_write));
    assert_eq!(bytes(&state, "cxl_read"), 512);
}
#[test]
fn no_pressure_spills_nothing_and_repeated_partial_hbm_hits_do_not_read_again() {
    let mut c = config(vec!["D0".into()], false);
    c["calls"][1]["blocks"] = json!(keys(8));
    c["calls"]
        .as_array_mut()
        .unwrap()
        .push(json!({"request_id":0,"subrequest_index":3,"worker":1,"blocks":keys(16)}));
    c["calls"]
        .as_array_mut()
        .unwrap()
        .push(json!({"request_id":0,"subrequest_index":4,"worker":1,"blocks":keys(16)}));
    let mut requests = workload(0.0);
    if let simthesizer_core::types::SubRequest::Llm { next, .. } = &mut requests[0].sub_requests[2]
    {
        *next = vec![3];
    }
    for next in [vec![4], vec![]] {
        requests[0].sub_requests.push(
            serde_json::from_value(
                json!({"kind":"llm","input_tokens":17,"output_tokens":1,"next":next}),
            )
            .unwrap(),
        );
    }
    let (_, state, _) = run(c, requests);
    assert_eq!(bytes(&state, "spill"), 0);
    assert_eq!(bytes(&state, "cxl_read"), 0);
    assert_eq!(bytes(&state, "rdma_read"), 1024);
    let ready = state["events"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["event"] == "kv_ready" && e["subrequest_index"] == 4)
        .unwrap();
    close(ready["kv_preparation_wait"].as_f64().unwrap(), 0.0);
}

#[test]
fn independent_ready_llm_runs_on_destination_while_reload_waits() {
    let mut c = config(vec!["D0".into(), "D1".into()], true);
    c["calls"]
        .as_array_mut()
        .unwrap()
        .push(json!({"request_id":2,"subrequest_index":0,"worker":1}));
    let mut requests = workload(1.0);
    requests.push(serde_json::from_value(json!({"request_id":2,"arrival_time":2.0,"initial":[0],"sub_requests":[{"kind":"llm","input_tokens":1,"output_tokens":1,"next":[]}]})).unwrap());
    let (_, state, snapshots) = run(c, requests);
    let completions = state["events"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|e| e["event"] == "session_complete")
        .collect::<Vec<_>>();
    let independent = completions.iter().find(|e| e["request_id"] == 2).unwrap()["time"]
        .as_f64()
        .unwrap();
    close(independent, 3.0);
    assert!(
        completions.iter().find(|e| e["request_id"] == 0).unwrap()["time"]
            .as_f64()
            .unwrap()
            > independent
    );
    assert!(snapshots.iter().any(|s| s["time"].as_f64().unwrap() == 2.0
        && s["state"]["instances"][1]["scheduler"]["running_requests"] == 1
        && s["state"]["cxl_pool"]["unfinished_transfers"]
            .as_u64()
            .unwrap()
            > 0));
}
#[test]
fn background_reads_contend_only_when_physical_ingress_overlaps() {
    let mut waits = vec![];
    for target in [None, Some(("dram0", "gpu1")), Some(("dram1", "gpu0"))] {
        let mut c = config(vec!["D0".into(), "D1".into()], true);
        if let Some((src, dst)) = target {
            c["background"] = json!((0..16)
                .map(
                    |_| json!({"request_id":0,"subrequest_index":2,"src":src,"dst":dst,"bytes":64})
                )
                .collect::<Vec<_>>());
        }
        let (_, state, _) = run(c, workload(1.0));
        let ready = state["events"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["event"] == "kv_ready" && e["subrequest_index"] == 2)
            .unwrap();
        waits.push(ready["kv_preparation_wait"].as_f64().unwrap());
    }
    close(waits[0], 1024.0 / 1e6);
    close(waits[1], 2048.0 / 1e6);
    close(waits[2], waits[0]);
}
#[test]
fn another_request_copies_resident_blocks_without_invalidating_active_owner() {
    let mut c = config(vec!["D0".into()], false);
    c["calls"]
        .as_array_mut()
        .unwrap()
        .push(json!({"request_id":2,"subrequest_index":0,"worker":0,"blocks":keys(16)}));
    let mut requests = workload(0.0);
    requests.push(serde_json::from_value(json!({"request_id":2,"arrival_time":1.5,"initial":[0],"sub_requests":[{"kind":"llm","input_tokens":17,"output_tokens":1,"next":[]}]})).unwrap());
    let (_, state, _) = run(c, requests);
    assert_eq!(bytes(&state, "rdma_read"), 1024);
    assert_eq!(bytes(&state, "hbm_read"), 1024);
    let events = state["events"].as_array().unwrap();
    let first_complete = events
        .iter()
        .find(|e| e["event"] == "session_complete" && e["request_id"] == 0)
        .unwrap()["time"]
        .as_f64()
        .unwrap();
    assert!(state["transfers"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|t| t["kind"] == "hbm_read")
        .all(|t| t["start"].as_f64().unwrap() < first_complete));
    assert!(state["blocks"]
        .as_array()
        .unwrap()
        .iter()
        .all(|b| b["location"]["worker"] == 0
            && b["ready"] == true
            && b["pins"].as_array().unwrap().is_empty()));
}

#[test]
fn parallel_tools_complete_at_their_own_times_and_join_before_call_interval() {
    let c = config(vec!["D0".into()], false);
    let request: Request = serde_json::from_value(
        json!({"request_id":9,"arrival_time":0.0,"initial":[0,1],"sub_requests":[
            {"kind":"tool_call","input_tokens":0,"output_tokens":0,"duration":1.0,"next":[2]},
            {"kind":"tool_call","input_tokens":0,"output_tokens":0,"duration":3.0,"next":[2]},
            {"kind":"llm","input_tokens":1,"output_tokens":1,"interval":2.0,"next":[]}
        ]}),
    )
    .unwrap();
    let (results, state, _) = run(c, vec![request]);
    let complete = state["events"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| e["event"] == "session_complete")
        .unwrap();
    close(complete["time"].as_f64().unwrap(), 6.0);
    let ResultSubRequest::Llm {
        token_latency,
        interval,
        ..
    } = &results[0].sub_requests[2]
    else {
        panic!()
    };
    assert_eq!(*interval, Some(2.0));
    assert_eq!(token_latency, &[1.0]);
}

#[test]
fn pressure_reserves_only_freed_space_and_never_overlaps_live_host_blocks() {
    let (_, _, snapshots) = run(config(vec!["D0".into()], true), workload(1.0));
    let mut admitted = false;
    for snapshot in snapshots {
        let pool = &snapshot["state"]["cxl_pool"];
        let host = pool["hosts"]
            .as_array()
            .unwrap()
            .iter()
            .find(|h| h["node"] == "node0")
            .unwrap();
        let reserved = host["pressure_reserved_bytes"].as_u64().unwrap();
        if reserved > 0 {
            admitted = true;
            assert_eq!(reserved, 512);
            assert!(host["block_bytes"].as_u64().unwrap() + reserved <= 1024);
            assert!(pool["transfers"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|t| t["kind"] == "spill")
                .all(|t| t["completion"].is_number()));
        }
    }
    assert!(admitted);
}

#[test]
fn reordered_or_suffix_keys_cannot_claim_restored_prefix_progress() {
    for invalid in [
        vec!["fixture:B1", "fixture:B0"],
        vec!["fixture:B8", "fixture:B9"],
    ] {
        let mut c = config(vec!["D0".into()], false);
        c["calls"][1]["blocks"] = json!(invalid);
        let mut system: Box<dyn System> = serde_json::from_value(c).unwrap();
        assert!(system
            .init()
            .unwrap_err()
            .to_string()
            .contains("incompatible prefix order"));
    }
}

fn tp_config(tp: usize) -> Value {
    let mut c = config((0..8).map(|i| format!("D{i}")).collect(), true);
    c.as_object_mut().unwrap().remove("block_bytes");
    for worker in c["instances"].as_array_mut().unwrap() {
        worker["model_runner"]["tp"] = json!(tp);
        worker["model_runner"]["link_bw"] = json!(1e9);
    }
    // Two KV heads: TP2 shards, TP4 replicates one head on each rank.
    for node in ["node0", "node1"] {
        c["host_capacity_bytes"][node] = json!(16 * 32 * tp);
    }
    c["pressures"][0]["bytes"] = json!(8 * 32 * tp);
    let bindings = c["topology"]["workers"].as_array().unwrap().clone();
    for binding in bindings {
        let worker = binding["worker"].as_u64().unwrap();
        for rank in 1..tp {
            let gpu = format!("gpu{worker}r{rank}");
            c["topology"]["endpoints"]
                .as_array_mut()
                .unwrap()
                .push(json!({"id":gpu,"kind":"gpu","node":format!("node{worker}")}));
            let id = format!("ingress{worker}-{gpu}");
            c["topology"]["links"].as_array_mut().unwrap().push(json!({
                "id":id,"a":format!("ingress{worker}"),"b":gpu,
                "a_port":format!("{id}:a"),"b_port":format!("{id}:b"),
                "bandwidth_ab":if worker==1 && rank==tp-1 {1000.0} else {1e6},"latency":0.0
            }));
            let mut b = binding.clone();
            b["rank"] = json!(rank);
            b["gpu"] = json!(gpu);
            c["topology"]["workers"].as_array_mut().unwrap().push(b);
        }
    }
    c
}

#[test]
fn reload_concurrency_limits_each_rank_not_the_whole_call() {
    // Each TP rank loads its own shard; one outstanding read per rank still
    // lets both ranks start together and never runs two reads on one rank.
    let mut config = tp_config(2);
    config["max_outstanding_reads"] = json!(1);
    let (_, state, _) = run(config, workload(1.0));
    let reads = state["transfers"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|t| t["dst_owner"][0] == 1)
        .collect::<Vec<_>>();
    let rank = |r: u64| {
        let mut spans = reads
            .iter()
            .filter(|t| t["dst_owner"][1] == r)
            .map(|t| {
                (
                    t["start"].as_f64().unwrap(),
                    t["completion"].as_f64().unwrap(),
                )
            })
            .collect::<Vec<_>>();
        spans.sort_by(|a, b| a.0.total_cmp(&b.0));
        spans
    };
    let (r0, r1) = (rank(0), rank(1));
    assert_eq!((r0.len(), r1.len()), (16, 16));
    close(r0[0].0, r1[0].0);
    for spans in [&r0, &r1] {
        assert!(spans.windows(2).all(|w| w[1].0 >= w[0].1 - 1e-12));
    }
}

#[test]
fn tp_reload_delivers_every_physical_shard_and_gates_on_slowest_rank() {
    // TP2 preserves 1024 B total. TP4 with only two KV heads requires 2048 B.
    for tp in [2, 4] {
        let (_, state, snapshots) = run(tp_config(tp), workload(1.0));
        assert_eq!(bytes(&state, "spill"), (8 * 32 * tp) as u64);
        assert_eq!(
            bytes(&state, "cxl_read") + bytes(&state, "rdma_read"),
            (16 * 32 * tp) as u64
        );
        let reads = state["transfers"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|t| t["dst_owner"][0] == 1)
            .collect::<Vec<_>>();
        assert_eq!(reads.len(), 16 * tp);
        let mut ends = vec![0.0f64; tp];
        for rank in 0..tp {
            let rank_reads = reads
                .iter()
                .filter(|t| t["dst_owner"][1] == rank)
                .collect::<Vec<_>>();
            assert_eq!(
                rank_reads
                    .iter()
                    .map(|t| t["bytes"].as_u64().unwrap())
                    .sum::<u64>(),
                512
            );
            ends[rank] = rank_reads
                .iter()
                .map(|t| t["completion"].as_f64().unwrap())
                .fold(0.0, f64::max);
        }
        assert!(ends[tp - 1] > ends[0]);
        let ready = state["events"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["event"] == "kv_ready" && e["subrequest_index"] == 2)
            .unwrap();
        close(ready["time"].as_f64().unwrap(), ends[tp - 1]);
        assert_eq!(ready["ranks"], json!((0..tp).collect::<Vec<_>>()));
        assert_eq!(ready["restored_tokens"], 16);
        for rank in 0..tp {
            let hbm = state["hbm_usage"]
                .as_array()
                .unwrap()
                .iter()
                .find(|h| h["worker"] == 1 && h["rank"] == rank)
                .unwrap();
            assert_eq!(hbm["occupied_or_reserved_bytes"], 512);
        }
        assert!(snapshots.iter().any(|s| s["state"]["cxl_pool"]["blocks"]
            .as_array()
            .unwrap()
            .iter()
            .any(|b| b["location"]["tier"] == "pool" && b["pending"]["rank"] == tp - 1)));
    }
}

#[test]
fn rank_native_pool_rejects_incompatible_physical_geometry_and_missing_rank_routes() {
    let mut missing = tp_config(2);
    missing["topology"]["workers"].as_array_mut().unwrap().pop();
    let mut system: Box<dyn System> = serde_json::from_value(missing).unwrap();
    assert!(system.init().is_err());
    let mut c = tp_config(2);
    // Explicit legacy size cannot replace each rank's physical payload.
    c["block_bytes"] = json!(1);
    assert!(common::run_system(c, &workload(1.0))
        .unwrap_err()
        .to_string()
        .contains("parallel block_bytes"));
    let mut c = tp_config(2);
    c["instances"][1]["model"]["num_key_value_heads"] = json!(4);
    assert!(common::run_system(c, &workload(1.0))
        .unwrap_err()
        .to_string()
        .contains("incompatible physical KV geometry"));
}

#[test]
fn legacy_proxy_geometry_and_partial_scheduler_blocks() {
    let mut c = config(vec!["D0".into()], false);
    c["instances"][1]["model"]["num_key_value_heads"] = json!(4);
    assert!(common::run_system(c, &workload(1.0))
        .unwrap_err()
        .to_string()
        .contains("incompatible physical KV geometry"));

    let mut c = config(vec!["D0".into()], false);
    c["instances"][1]["scheduler"]["block_size"] = json!(2);
    // Pool blocks and restored prefixes may be smaller than scheduler blocks.
    assert!(common::run_system(c.clone(), &workload(1.0)).is_ok());
    c["calls"][1]["blocks"] = json!(keys(15));
    assert!(common::run_system(c, &workload(1.0)).is_ok());
}

#[test]
fn ep_rank_copy_preserves_live_source_on_the_same_worker() {
    let mut c = tp_config(2);
    c["pressures"] = json!([]);
    for worker in c["instances"].as_array_mut().unwrap() {
        let mut runner = worker["model_runner"].clone();
        runner["kind"] = json!("moe_ep");
        runner.as_object_mut().unwrap().remove("tp");
        runner["ep"] = json!(2);
        let scheduler = worker["scheduler"].clone();
        *worker = json!({"kind":"moe_ep_multi_instance", "model":worker["model"],
            "model_runner":runner, "schedulers":[scheduler.clone(), scheduler]});
    }
    c["calls"]
        .as_array_mut()
        .unwrap()
        .push(json!({"request_id":2,"subrequest_index":0,"worker":1,"blocks":keys(16)}));
    let mut requests = workload(1.0);
    requests.push(
        serde_json::from_value(json!({"request_id":2,"arrival_time":2.5,"initial":[0],
        "sub_requests":[{"kind":"llm","input_tokens":17,"output_tokens":1,"next":[]}]}))
        .unwrap(),
    );
    let (results, state, snapshots) = run(c, requests);
    assert_eq!(results.len(), 3);
    assert_eq!(bytes(&state, "rdma_read"), 1024);
    assert_eq!(bytes(&state, "hbm_read"), 1024);
    let events = state["events"].as_array().unwrap();
    let first_complete = events
        .iter()
        .find(|e| e["event"] == "session_complete" && e["request_id"] == 0)
        .unwrap()["time"]
        .as_f64()
        .unwrap();
    assert!(first_complete > 2.5);
    let moved = state["transfers"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|t| t["kind"] == "hbm_read")
        .collect::<Vec<_>>();
    assert!(moved
        .iter()
        .all(|t| t["src_owner"] == json!([1, 0]) && t["dst_owner"] == json!([1, 1])));
    assert!(moved
        .iter()
        .all(|t| t["start"].as_f64().unwrap() < first_complete));
    let ready = events
        .iter()
        .find(|e| e["event"] == "kv_ready" && e["request_id"] == 2)
        .unwrap();
    assert_eq!(ready["ranks"], json!([1]));
    close(
        ready["time"].as_f64().unwrap(),
        moved
            .iter()
            .map(|t| t["completion"].as_f64().unwrap())
            .fold(0.0, f64::max),
    );
    assert!(snapshots.iter().any(|s| s["time"] == 2.5
        && s["state"]["cxl_pool"]["blocks"]
            .as_array()
            .unwrap()
            .iter()
            .all(|b| b["location"]["worker"] == 1
                && b["location"]["rank"] == 0
                && !b["pins"].as_array().unwrap().is_empty()
                && (b["pending"]["rank"] == 1
                    || b["replicas"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .any(|p| p["rank"] == 1)))));
    assert!(state["blocks"]
        .as_array()
        .unwrap()
        .iter()
        .all(|b| b["location"]["worker"] == 1
            && b["location"]["rank"] == 1
            && b["pins"].as_array().unwrap().is_empty()));
}

fn native_config(block_tokens: u32, host_bytes: u64) -> Value {
    let mut c = config(vec!["D0".into()], false);
    c["calls"] = json!([]);
    c["block_tokens"] = json!(block_tokens);
    c["block_bytes"] = json!(u64::from(block_tokens) * 64);
    c["host_capacity_bytes"]["node0"] = json!(host_bytes);
    c
}

#[test]
fn native_identity_reuses_content_across_sessions_but_not_token_counts() {
    take_compute_calls();
    let prefix = json!({"namespace":"dataset:revision:week:source:target:shared","block_tokens":1,"block_ids":[10,11,12,13]});
    let requests = (0..3).map(|id| {
        let mut sub = json!({"kind":"llm","input_tokens":5,"output_tokens":1,"session_id":if id == 1 {"another"} else {"same"},"next":[]});
        if id < 2 { sub["kv_prefix"] = prefix.clone(); }
        serde_json::from_value(json!({"request_id":id,"arrival_time":(id * 3) as f64,"initial":[0],"sub_requests":[sub]})).unwrap()
    }).collect();
    let (_, state, _) = run(native_config(1, 4096), requests);
    assert_eq!(
        take_compute_calls()
            .iter()
            .filter(|c| c.layer.name() == "attention")
            .map(|c| c.batch[0].q_len)
            .collect::<Vec<_>>(),
        vec![5, 1, 5]
    );
    assert_eq!(bytes(&state, "store"), 11 * 64);
    assert!(state["blocks"]
        .as_array()
        .unwrap()
        .iter()
        .all(|block| block["pins"].as_array().unwrap().is_empty()
            && block["sessions"].as_array().unwrap().is_empty()
            && block["replicas"].as_array().unwrap().len() == 2));
}

#[test]
fn full_block_stores_overlap_compute_and_reserve_both_copies() {
    let mut c = native_config(2, 4096);
    c["instances"][0]["scheduler"]["max_num_batched_tokens"] = json!(2);
    let request = serde_json::from_value(json!({"request_id":0,"arrival_time":0.0,"initial":[0],"sub_requests":[{"kind":"llm","input_tokens":5,"output_tokens":3,"next":[]}]})).unwrap();
    let (_, state, snapshots) = run(c, vec![request]);
    let completed = state["events"]
        .as_array()
        .unwrap()
        .iter()
        .find(|event| event["event"] == "session_complete")
        .unwrap()["time"]
        .as_f64()
        .unwrap();
    assert_eq!(bytes(&state, "store"), 3 * 128);
    assert!(state["transfers"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|t| t["kind"] == "store")
        .any(|t| t["start"].as_f64().unwrap() < completed));
    assert!(snapshots
        .iter()
        .any(|snapshot| snapshot["state"]["cxl_pool"]["blocks"]
            .as_array()
            .unwrap()
            .iter()
            .any(|block| block["location"]["tier"] == "hbm"
                && block["pending"]["tier"] == "host"
                && block["ready"] == true
                && block["pending_ready"] == false)));
    assert!(state["blocks"]
        .as_array()
        .unwrap()
        .iter()
        .all(|block| block["replicas"].as_array().unwrap().len() == 2));
}

#[test]
fn host_and_cxl_pressure_reclaim_only_copies_and_respect_capacities() {
    let mut c = native_config(2, 128);
    c["instances"][0]["scheduler"]["max_num_batched_tokens"] = json!(2);
    for endpoint in c["topology"]["endpoints"].as_array_mut().unwrap() {
        if endpoint["id"] == "D0" {
            endpoint["capacity_bytes"] = json!(128);
        }
    }
    let request = serde_json::from_value(json!({"request_id":0,"arrival_time":0.0,"initial":[0],"sub_requests":[{"kind":"llm","input_tokens":6,"output_tokens":1,"next":[]}]})).unwrap();
    let (_, state, snapshots) = run(c, vec![request]);
    assert_eq!(bytes(&state, "store"), 384);
    assert_eq!(bytes(&state, "spill"), 256);
    assert_eq!(state["blocks"].as_array().unwrap().len(), 3);
    assert!(state["events"]
        .as_array()
        .unwrap()
        .iter()
        .any(|event| event["event"] == "cache_reclaimed" && event["location"]["tier"] == "pool"));
    for snapshot in snapshots {
        let pool = &snapshot["state"]["cxl_pool"];
        assert!(pool["hosts"][0]["block_bytes"].as_u64().unwrap() <= 128);
        assert!(
            pool["pool_usage"][0]["occupied_or_reserved_bytes"]
                .as_u64()
                .unwrap()
                <= 128
        );
    }
}

fn hbm_capacity(c: &mut Value, tokens: u64) {
    let mut model: Box<dyn simthesizer_core::Model> =
        serde_json::from_value(c["instances"][0]["model"].clone()).unwrap();
    model
        .init(simthesizer_core::types::ParallelStrategy {
            tp: 1,
            pp: 1,
            ep: 1,
        })
        .unwrap();
    c["instances"][0]["scheduler"]["device_mem"] =
        json!(model.weight_bytes_per_device() + tokens * model.kv_bytes_per_token_per_device());
}

#[test]
fn hbm_pressure_preserves_backing_and_reloads_without_duplicate_store() {
    let mut c = native_config(1, 4096);
    hbm_capacity(&mut c, 5);
    let prefix = json!({"namespace":"shared:model:layout","block_tokens":1,"block_ids":[1,2,3,4]});
    let requests = (0..3).map(|id| {
        let mut sub = json!({"kind":"llm","input_tokens":5,"output_tokens":1,"next":[]});
        if id != 1 { sub["kv_prefix"] = prefix.clone(); }
        serde_json::from_value(json!({"request_id":id,"arrival_time":(id * 3) as f64,"initial":[0],"sub_requests":[sub]})).unwrap()
    }).collect();
    let (_, state, _) = run(c, requests);
    assert_eq!(bytes(&state, "rdma_read"), 4 * 64);
    assert_eq!(bytes(&state, "store"), 11 * 64);
    assert!(state["events"]
        .as_array()
        .unwrap()
        .iter()
        .any(|event| event["event"] == "cache_reclaimed" && event["location"]["tier"] == "hbm"));
    assert!(state["blocks"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|block| block["key"].as_str().unwrap().starts_with("shared:"))
        .all(|block| block["replicas"]
            .as_array()
            .unwrap()
            .iter()
            .any(|copy| copy["tier"] == "host")));
}

#[test]
fn external_preemption_releases_pins_and_resumes_surviving_prefix() {
    let mut c = native_config(1, 4096);
    c["instances"][0]["scheduler"]["max_batch"] = json!(2);
    hbm_capacity(&mut c, 6);
    let requests = (0..2).map(|id| serde_json::from_value(json!({"request_id":id,"arrival_time":0.0,"initial":[0],"sub_requests":[{"kind":"llm","input_tokens":2,"output_tokens":3,"next":[]}]})).unwrap()).collect();
    let (results, state, _) = run(c, requests);
    assert_eq!(results.len(), 2);
    assert!(state["events"]
        .as_array()
        .unwrap()
        .iter()
        .any(|event| event["event"] == "request_preempted"));
    assert!(state["events"]
        .as_array()
        .unwrap()
        .iter()
        .any(|event| event["event"] == "request_resumed"
            && event["restored_tokens"].as_u64().unwrap() > 0));
    assert!(state["blocks"]
        .as_array()
        .unwrap()
        .iter()
        .all(|block| block["pins"].as_array().unwrap().is_empty()));
}

#[test]
fn missing_suffix_recomputes_after_the_continuous_available_prefix() {
    take_compute_calls();
    let mut c = config(vec!["D0".into()], false);
    c["calls"][0]["seed_host_blocks"] = json!(keys(1));
    c["calls"][1]["blocks"] = json!(keys(3));
    let mut requests = workload(0.0);
    requests[0].sub_requests[2] =
        serde_json::from_value(json!({"kind":"llm","input_tokens":4,"output_tokens":1,"next":[]}))
            .unwrap();
    let (_, state, _) = run(c, requests);
    let ready = state["events"]
        .as_array()
        .unwrap()
        .iter()
        .find(|event| event["event"] == "kv_ready" && event["subrequest_index"] == 2)
        .unwrap();
    assert_eq!(ready["restored_tokens"], 1);
    assert_eq!(
        take_compute_calls()
            .iter()
            .filter(|call| call.layer.name() == "attention")
            .map(|call| call.batch[0].q_len)
            .collect::<Vec<_>>(),
        vec![17, 3]
    );
}

#[test]
fn pending_store_protects_hbm_until_the_copy_is_published() {
    let mut c = native_config(1, 4096);
    hbm_capacity(&mut c, 4);
    for link in c["topology"]["links"].as_array_mut().unwrap() {
        if link["a"] == "gpu0" && link["b"] == "nic0" {
            link["bandwidth_ab"] = json!(64.0);
        }
    }
    let requests = (0..2).map(|id| serde_json::from_value(json!({"request_id":id,"arrival_time":id as f64 * 1.1,"initial":[0],"sub_requests":[{"kind":"llm","input_tokens":4,"output_tokens":1,"next":[]}]})).unwrap()).collect();
    let (_, state, snapshots) = run(c, requests);
    let first_store_end = state["transfers"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|transfer| {
            transfer["kind"] == "store"
                && transfer["block"]
                    .as_str()
                    .unwrap()
                    .starts_with("private:0:")
        })
        .map(|transfer| transfer["completion"].as_f64().unwrap())
        .fold(0.0, f64::max);
    assert!(first_store_end > 1.1);
    assert!(state["events"]
        .as_array()
        .unwrap()
        .iter()
        .any(|event| event["event"] == "hbm_admission_wait"));
    assert!(snapshots
        .iter()
        .any(|snapshot| snapshot["time"].as_f64().unwrap() >= 1.1
            && snapshot["time"].as_f64().unwrap() < first_store_end
            && snapshot["state"]["cxl_pool"]["blocks"]
                .as_array()
                .unwrap()
                .iter()
                .all(|block| block["location"]["tier"] == "hbm")));
    assert_eq!(bytes(&state, "store"), 8 * 64);
}

fn recurrent_config(pattern: &str) -> Value {
    let mut c = native_config(2, 4096);
    c.as_object_mut().unwrap().remove("block_bytes");
    let model = json!({"kind":"nemotron_h","hidden_size":8,"intermediate_size":12,
        "num_hidden_layers":4,"hybrid_override_pattern":pattern,
        "num_attention_heads":2,"head_dim":4,"num_key_value_heads":1,"vocab_size":16,
        "mamba_num_heads":2,"mamba_head_dim":4,"n_groups":1,"ssm_state_size":3,"conv_kernel":3,
        "moe_intermediate_size":6,"moe_shared_expert_intermediate_size":4,
        "n_routed_experts":3,"n_shared_experts":1,"num_experts_per_tok":2});
    for worker in c["instances"].as_array_mut().unwrap() {
        worker["model"] = model.clone();
        worker["scheduler"]["block_size"] = json!(2);
        worker["model_runner"]["compute_sim"]["costs"]["mamba"] = json!(0.1);
    }
    c
}

fn recurrent_request(id: u64, outputs: u32, prefix: bool) -> Request {
    let mut sub = json!({"kind":"llm","input_tokens":5,"output_tokens":outputs,"next":[]});
    if prefix {
        sub["kv_prefix"] =
            json!({"namespace":"hybrid:source:target:shared","block_tokens":2,"block_ids":[1,2]});
    }
    serde_json::from_value(
        json!({"request_id":id,"arrival_time":id as f64 * 10.0,"initial":[0],"sub_requests":[sub]}),
    )
    .unwrap()
}

#[test]
fn hybrid_prefix_requires_sparse_recurrent_checkpoint_and_releases_source_pin() {
    take_compute_calls();
    let (results, state, snapshots) = run(
        recurrent_config("M*-E"),
        vec![recurrent_request(0, 3, true), recurrent_request(1, 1, true)],
    );
    let checkpoints = state["blocks"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|block| block["kind"] == "recurrent_state")
        .collect::<Vec<_>>();
    assert_eq!(checkpoints.len(), 1);
    assert_eq!(checkpoints[0]["bytes"], 152);
    assert_eq!(checkpoints[0]["checkpoint_tokens"], 4);
    let release = state["events"]
        .as_array()
        .unwrap()
        .iter()
        .find(|event| event["event"] == "recurrent_source_released" && event["request_id"] == 0)
        .unwrap();
    let ResultSubRequest::Llm { token_latency, .. } = &results
        .iter()
        .find(|result| result.request_id == 0)
        .unwrap()
        .sub_requests[0]
    else {
        panic!()
    };
    close(release["time"].as_f64().unwrap(), token_latency[0]);
    assert_eq!(bytes(&state, "store"), 3 * 32 + 152);
    let ready = state["events"]
        .as_array()
        .unwrap()
        .iter()
        .find(|event| event["event"] == "kv_ready" && event["request_id"] == 1)
        .unwrap();
    assert_eq!(ready["restored_tokens"], 4);
    assert!(snapshots.iter().any(|snapshot| {
        let pool = &snapshot["state"]["cxl_pool"];
        pool["unfinished_requests"].as_u64().unwrap() > 0
            && pool["blocks"].as_array().unwrap().iter().any(|block| {
                block["kind"] == "recurrent_state"
                    && block["pending"].is_null()
                    && block["pins"].as_array().unwrap().is_empty()
            })
    }));
    assert_eq!(
        take_compute_calls()
            .iter()
            .filter(|call| call.layer.name() == "attention")
            .map(|call| call.batch[0].q_len)
            .collect::<Vec<_>>(),
        vec![4, 1, 1, 1, 1]
    );
}

#[test]
fn hybrid_attention_only_fixture_is_a_cache_miss() {
    let mut c = recurrent_config("M*-E");
    c["calls"] = json!([{"request_id":0,"subrequest_index":0,"worker":0,"seed_host_blocks":keys(2)}, {"request_id":1,"subrequest_index":0,"worker":1,"blocks":keys(2)}]);
    let (_, state, _) = run(
        c,
        vec![
            recurrent_request(0, 1, false),
            recurrent_request(1, 1, false),
        ],
    );
    let ready = state["events"]
        .as_array()
        .unwrap()
        .iter()
        .find(|event| event["event"] == "kv_ready" && event["request_id"] == 1)
        .unwrap();
    assert_eq!(ready["restored_tokens"], 0);
    assert_eq!(bytes(&state, "rdma_read"), 0);
}

#[test]
fn hybrid_reload_transfers_state_and_attention_and_retains_backing() {
    let mut c = recurrent_config("M*-E");
    let mut model: Box<dyn simthesizer_core::Model> =
        serde_json::from_value(c["instances"][0]["model"].clone()).unwrap();
    model
        .init(simthesizer_core::types::ParallelStrategy {
            tp: 1,
            pp: 1,
            ep: 1,
        })
        .unwrap();
    c["instances"][0]["scheduler"]["device_mem"] = json!(model.weight_bytes_per_device() + 400);
    let (_, state, _) = run(
        c,
        vec![
            recurrent_request(0, 1, true),
            recurrent_request(1, 1, false),
            recurrent_request(2, 1, true),
        ],
    );
    assert_eq!(bytes(&state, "rdma_read"), 2 * 32 + 152);
    let reads = state["transfers"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|transfer| {
            transfer["kind"] == "rdma_read"
                && transfer["block"]
                    .as_str()
                    .unwrap()
                    .ends_with(":recurrent-state")
        })
        .collect::<Vec<_>>();
    assert_eq!(reads.len(), 1);
    assert_eq!(reads[0]["bytes"], 152);
    assert!(state["blocks"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|block| block["kind"] == "recurrent_state")
        .all(|block| block["replicas"]
            .as_array()
            .unwrap()
            .iter()
            .any(|copy| copy["tier"] == "host")));
}

#[test]
fn pure_recurrent_prefix_caches_only_state_bytes() {
    let (_, state, _) = run(
        recurrent_config("MMMM"),
        vec![recurrent_request(0, 1, true), recurrent_request(1, 1, true)],
    );
    assert_eq!(state["blocks"].as_array().unwrap().len(), 1);
    assert_eq!(state["blocks"][0]["kind"], "recurrent_state");
    assert_eq!(bytes(&state, "store"), 608);
    let ready = state["events"]
        .as_array()
        .unwrap()
        .iter()
        .find(|event| event["event"] == "kv_ready" && event["request_id"] == 1)
        .unwrap();
    assert_eq!(ready["restored_tokens"], 4);
}

#[test]
fn nested_session_membership_closes_before_its_sibling_finishes() {
    let request = serde_json::from_value(
        json!({"request_id":0,"arrival_time":0.0,"initial":[0,1],"sub_requests":[
            {"kind":"llm","input_tokens":2,"output_tokens":1,"session_id":"nested-a","next":[]},
            {"kind":"llm","input_tokens":2,"output_tokens":3,"session_id":"nested-b","next":[]}
        ]}),
    )
    .unwrap();
    let (_, state, snapshots) = run(native_config(1, 4096), vec![request]);
    assert!(snapshots.iter().any(|snapshot| {
        let pool = &snapshot["state"]["cxl_pool"];
        pool["unfinished_requests"] == 1
            && pool["events"].as_array().unwrap().iter().any(|event| {
                event["event"] == "session_references_released" && event["session_id"] == "nested-a"
            })
            && pool["blocks"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|block| block["key"].as_str().unwrap().starts_with("private:0:0:"))
                .all(|block| block["sessions"].as_array().unwrap().is_empty())
    }));
    assert!(state["blocks"]
        .as_array()
        .unwrap()
        .iter()
        .all(|block| block["sessions"].as_array().unwrap().is_empty()));
}

#[test]
fn hybrid_fragmented_host_storage_and_small_cxl_device_make_progress() {
    let mut c = recurrent_config("M*-E");
    c["host_capacity_bytes"]["node0"] = json!(400);
    c["placement_devices"] = json!(["D0", "D1"]);
    for device in c["topology"]["endpoints"].as_array_mut().unwrap() {
        if device["id"] == "D0" {
            device["capacity_bytes"] = json!(64);
        }
        if device["id"] == "D1" {
            device["capacity_bytes"] = json!(1024);
        }
    }
    let (_, state, snapshots) = run(
        c,
        (0..3).map(|id| recurrent_request(id, 1, false)).collect(),
    );
    assert_eq!(state["unfinished_transfers"], 0);
    assert!(snapshots
        .iter()
        .any(|snapshot| snapshot["state"]["cxl_pool"]["blocks"]
            .as_array()
            .unwrap()
            .iter()
            .any(
                |block| block["replicas"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|copy| copy["extents"].as_array().unwrap().len() > 1)
            )));
    assert!(state["blocks"]
        .as_array()
        .unwrap()
        .iter()
        .any(|block| block["kind"] == "recurrent_state"
            && block["replicas"]
                .as_array()
                .unwrap()
                .iter()
                .any(|copy| copy["device"] == "D1")));
    for snapshot in snapshots {
        let blocks = snapshot["state"]["cxl_pool"]["blocks"].as_array().unwrap();
        let mut extents = Vec::new();
        for block in blocks {
            for copy in block["replicas"]
                .as_array()
                .unwrap()
                .iter()
                .chain(block["pending"].as_object().map(|_| &block["pending"]))
            {
                if copy["tier"] == "host" && copy["node"] == "node0" {
                    for extent in copy["extents"].as_array().unwrap() {
                        let start = extent[0].as_u64().unwrap();
                        let end = start + extent[1].as_u64().unwrap();
                        assert!(end <= 400);
                        extents.push((start, end));
                    }
                }
            }
        }
        extents.sort_unstable();
        assert!(extents.windows(2).all(|pair| pair[0].1 <= pair[1].0));
    }
}

#[test]
fn session_membership_includes_trailing_scoped_tools() {
    let request = serde_json::from_value(json!({"request_id":0,"arrival_time":0.0,"initial":[0],"sub_requests":[
        {"kind":"llm","input_tokens":2,"output_tokens":1,"session_id":"source","next":[1]},
        {"kind":"tool_call","input_tokens":0,"output_tokens":0,"duration":5.0,"session_id":"source","next":[]}
    ]})).unwrap();
    let (_, state, snapshots) = run(native_config(1, 4096), vec![request]);
    assert!(snapshots.iter().any(|snapshot| {
        let pool = &snapshot["state"]["cxl_pool"];
        snapshot["time"].as_f64().unwrap() > 1.0
            && pool["unfinished_requests"] == 1
            && pool["blocks"].as_array().unwrap().iter().any(|block| {
                !block["sessions"].as_array().unwrap().is_empty()
                    && block["pins"].as_array().unwrap().is_empty()
            })
    }));
    let end = state["events"]
        .as_array()
        .unwrap()
        .iter()
        .find(|event| {
            event["event"] == "session_references_released" && event["session_id"] == "source"
        })
        .unwrap();
    close(end["time"].as_f64().unwrap(), 6.0);
    assert!(state["blocks"]
        .as_array()
        .unwrap()
        .iter()
        .all(|block| block["sessions"].as_array().unwrap().is_empty()));
}

#[test]
fn default_observations_use_aggregate_cache_and_transfer_counters() {
    let mut config = native_config(1, 4096);
    config["background"] =
        json!([{"request_id":0,"subrequest_index":0,"src":"dram0","dst":"gpu0","bytes":64}]);
    let mut system: Box<dyn System> = serde_json::from_value(config.clone()).unwrap();
    system.init().unwrap();
    let mut graph = Graph::new();
    let request: Request = serde_json::from_value(json!({"request_id":0,"arrival_time":0.0,"initial":[0],"sub_requests":[{"kind":"llm","input_tokens":2,"output_tokens":1,"next":[]}]})).unwrap();
    system
        .add_request_arrival(request.clone(), &mut graph, None, None)
        .unwrap();
    let mut state = Value::Null;
    let results = Engine::new(graph, system)
        .run_with_observer(|_, system| {
            state = system.observe()["cxl_pool"].clone();
            Ok(())
        })
        .unwrap();
    assert_eq!(state["block_count"], 2);
    assert_eq!(state["transfer_totals"]["store"], json!([2, 128]));
    assert_eq!(state["transfer_totals"]["background"], json!([1, 64]));
    assert_eq!(state["unfinished_transfers"], 0);
    assert!(
        state["blocks"].is_null()
            && state["transfers"].is_null()
            && state["events"].is_null()
            && state["topology"].is_null()
    );
    let (detailed_results, detailed_state, _) = run(config, vec![request]);
    assert_eq!(results, detailed_results);
    for (key, value) in state.as_object().unwrap() {
        if !matches!(
            key.as_str(),
            "detailed_observations" | "blocks" | "transfers" | "events" | "topology"
        ) {
            assert_eq!(value, &detailed_state[key], "{key}");
        }
    }
}

#[test]
fn hybrid_preemption_drains_state_and_transfer_reservations() {
    let mut c = recurrent_config("M*-E");
    let mut model: Box<dyn simthesizer_core::Model> =
        serde_json::from_value(c["instances"][0]["model"].clone()).unwrap();
    model
        .init(simthesizer_core::types::ParallelStrategy {
            tp: 1,
            pp: 1,
            ep: 1,
        })
        .unwrap();
    c["instances"][0]["scheduler"]["device_mem"] = json!(model.weight_bytes_per_device() + 640);
    c["instances"][0]["scheduler"]["max_batch"] = json!(2);
    let requests = (0..2)
        .map(|id| {
            let mut request = recurrent_request(id, 4, false);
            request.arrival_time = 0.0;
            request
        })
        .collect();
    let (results, state, snapshots) = run(c, requests);
    assert_eq!(results.len(), 2);
    assert!(state["events"]
        .as_array()
        .unwrap()
        .iter()
        .any(|event| event["event"] == "request_preempted"));
    let first_completion = state["events"]
        .as_array()
        .unwrap()
        .iter()
        .find(|event| event["event"] == "session_complete" && event["request_id"] == 0)
        .unwrap()["time"]
        .as_f64()
        .unwrap();
    assert!(state["events"]
        .as_array()
        .unwrap()
        .iter()
        .any(|event| event["event"] == "request_resumed"
            && event["request_id"] == 1
            && event["time"].as_f64().unwrap() < first_completion));
    assert_eq!(state["unfinished_transfers"], 0);
    assert!(state["blocks"]
        .as_array()
        .unwrap()
        .iter()
        .all(|block| block["pins"].as_array().unwrap().is_empty() && block["pending"].is_null()));
    assert!(snapshots
        .iter()
        .all(|snapshot| snapshot["state"]["cxl_pool"]["hbm_usage"][0]
            ["occupied_or_reserved_bytes"]
            .as_u64()
            .unwrap()
            <= 640));
}
