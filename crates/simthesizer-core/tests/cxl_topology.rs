mod common;

use std::any::Any;

use common::close;
use simthesizer_core::network_sims::flow::FlowNetworkSimulator;
use simthesizer_core::network_sims::topology::{
    Endpoint, EndpointKind, Link, Routing, Topology, WorkerBinding,
};
use simthesizer_core::{types::PollResult, Module, NetworkSimulator};

const MIB: u64 = 1_048_576;
const U: f64 = 1e9;

fn endpoint(id: String, kind: EndpointKind) -> Endpoint {
    Endpoint {
        id,
        kind,
        node: None,
        lane_budget: None,
        capacity_bytes: None,
        read_bandwidth: None,
        write_bandwidth: None,
    }
}

fn physical(topology: &mut Topology, a: &str, b: &str, width: u32) {
    let id = format!("L{}", topology.links.len());
    topology.links.push(Link {
        id: id.clone(),
        a: a.into(),
        b: b.into(),
        a_port: id.clone(),
        b_port: id,
        width: Some(width),
        bandwidth_ab: width as f64 * U,
        bandwidth_ba: Some(width as f64 * U),
        latency: 0.0,
    });
}

fn family(leaves: usize, hosts: usize, devices: usize, spines: usize, parallel: usize) -> Topology {
    let mut topology = Topology {
        endpoints: Vec::new(),
        links: Vec::new(),
        workers: Vec::new(),
        required_routes: Vec::new(),
        routing: Routing::EqualShortest,
    };
    for spine in 0..spines {
        topology
            .endpoints
            .push(endpoint(format!("S{spine}"), EndpointKind::Switch));
    }
    for leaf in 0..leaves {
        let switch = format!("L{leaf}");
        topology
            .endpoints
            .push(endpoint(switch.clone(), EndpointKind::Switch));
        for host in 0..hosts {
            let id = format!("A{leaf}_{host}");
            let mut adapter = endpoint(id.clone(), EndpointKind::Adapter);
            adapter.node = Some(format!("N{leaf}_{host}"));
            topology.endpoints.push(adapter);
            physical(&mut topology, &id, &switch, 16);
        }
        for device in 0..devices {
            let id = format!("D{leaf}_{device}");
            let mut memory = endpoint(id.clone(), EndpointKind::Memory);
            memory.capacity_bytes = Some(64_000_000_000);
            memory.read_bandwidth = Some(1e12);
            memory.write_bandwidth = Some(1e12);
            topology.endpoints.push(memory);
            physical(&mut topology, &id, &switch, 8);
        }
        for spine in 0..spines {
            for _ in 0..parallel {
                physical(&mut topology, &switch, &format!("S{spine}"), 16);
            }
        }
    }
    topology
}

fn finish(net: &mut FlowNetworkSimulator, transfers: &[(u64, u64, u64)]) -> Vec<f64> {
    let mut states: Vec<Option<Box<dyn Any>>> = (0..transfers.len()).map(|_| None).collect();
    let mut pending = (0..transfers.len())
        .map(|index| (0.0_f64, index))
        .collect::<Vec<_>>();
    let mut completed = vec![f64::NAN; transfers.len()];
    for _ in 0..100_000 {
        pending.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
        let Some((now, index)) = pending.pop() else {
            return completed;
        };
        let (src, dst, bytes) = transfers[index];
        match net
            .simulate(index as u64, src, dst, bytes, now, &mut states[index])
            .unwrap()
        {
            PollResult::Deferred => pending.push((now, index)),
            PollResult::Complete { latency } => completed[index] = now + latency,
            PollResult::Pending { latency } => {
                assert!(latency > 0.0 && now + latency > now);
                pending.push((now + latency, index));
            }
        }
        for (node, time) in net.take_wakeups() {
            let entry = pending
                .iter_mut()
                .find(|entry| entry.1 == node as usize)
                .unwrap();
            assert!(time >= now && time < entry.0);
            entry.0 = time;
        }
    }
    panic!("transfers did not drain")
}

#[test]
fn configured_families_deliver_predicted_local_and_remote_capacity() {
    for (
        name,
        leaves,
        hosts,
        devices,
        spines,
        parallel,
        local_lanes,
        remote_lanes,
        leaf_lanes,
        spine_lanes,
    ) in [
        ("single", 1, 8, 16, 0, 0, 128, 128, 256, 0),
        ("balanced", 8, 4, 8, 4, 2, 64, 64, 256, 256),
        ("fewer", 8, 4, 8, 1, 2, 64, 32, 160, 256),
        ("dense", 8, 6, 12, 4, 1, 96, 64, 256, 128),
    ] {
        for remote in [false, true] {
            let topology = family(leaves, hosts, devices, spines, parallel);
            let mut net = topology.build().unwrap();
            let inventory = net.observe();
            for leaf in 0..leaves {
                assert_eq!(inventory["lanes"][format!("L{leaf}")], leaf_lanes);
            }
            for spine in 0..spines {
                assert_eq!(inventory["lanes"][format!("S{spine}")], spine_lanes);
            }
            let mut transfers = Vec::new();
            for leaf in 0..leaves {
                let source_leaf = if remote {
                    (leaf + leaves / 2) % leaves
                } else {
                    leaf
                };
                for host in 0..hosts {
                    for half in 0..2 {
                        transfers.push((
                            net.endpoint(&format!("D{source_leaf}_{}", host * 2 + half))
                                .unwrap(),
                            net.endpoint(&format!("A{leaf}_{host}")).unwrap(),
                            MIB,
                        ));
                    }
                }
            }
            let expected = devices as f64 * MIB as f64
                / (if remote { remote_lanes } else { local_lanes }) as f64
                / U;
            let completed = finish(&mut net.flow, &transfers);
            for &time in &completed {
                close(time, expected);
            }
            let snapshot = net.observe();
            let links = snapshot["flow"]["links"].as_array().unwrap();
            let mut used_uplinks = 0;
            for link in &topology.links {
                if !link.b.starts_with('S') {
                    continue;
                }
                for direction in ["ab", "ba"] {
                    let stats = links
                        .iter()
                        .find(|stats| stats["name"] == format!("{}:{direction}", link.id))
                        .unwrap();
                    let bytes = stats["bytes"].as_f64().unwrap();
                    if remote && leaves > 1 {
                        used_uplinks += 1;
                        let expected_bytes =
                            devices as f64 * MIB as f64 / (spines * parallel) as f64;
                        close(bytes, expected_bytes);
                        close(stats["busy_seconds"].as_f64().unwrap(), expected);
                        assert!(stats["peak_flows"].as_u64().unwrap() > 0);
                    } else {
                        close(bytes, 0.0);
                    }
                }
            }
            assert_eq!(snapshot["flow"]["active_subflows"], 0);
            println!("{name} remote={remote} expected_s={expected:.12} observed_s={:.12} delivered_GBs={:.3} used_directed_uplinks={used_uplinks}", completed[0], devices as f64 * MIB as f64 / completed[0] / 1e9);
        }
    }
}

#[test]
fn physical_validation_rejects_bad_inventory_and_missing_routes() {
    let original = family(1, 8, 16, 0, 0);
    let mut bad = original.clone();
    bad.links[1].b_port = bad.links[0].b_port.clone();
    assert!(bad
        .build()
        .err()
        .unwrap()
        .to_string()
        .contains("physical port"));
    let mut bad = original.clone();
    bad.endpoints
        .iter_mut()
        .find(|e| e.kind == EndpointKind::Switch)
        .unwrap()
        .lane_budget = Some(255);
    assert!(bad
        .build()
        .err()
        .unwrap()
        .to_string()
        .contains("over budget"));
    let mut bad = original.clone();
    bad.links[0].b = "missing".into();
    assert!(bad
        .build()
        .err()
        .unwrap()
        .to_string()
        .contains("invalid endpoint"));
    let mut bad = original.clone();
    bad.links[0].width = Some(8);
    assert!(bad.build().err().unwrap().to_string().contains("x16"));
    let mut bad = original.clone();
    bad.endpoints
        .push(endpoint("isolated".into(), EndpointKind::Gpu));
    bad.required_routes
        .push(serde_json::from_value(serde_json::json!({"src":"D0_0", "dst":"isolated"})).unwrap());
    assert!(bad
        .build()
        .err()
        .unwrap()
        .to_string()
        .contains("required route"));
    let mut bad = original;
    bad.endpoints
        .iter_mut()
        .find(|e| e.kind == EndpointKind::Memory)
        .unwrap()
        .read_bandwidth = Some(0.0);
    assert!(bad.build().is_err());
}

#[test]
fn hot_device_and_hot_adapter_limit_payload() {
    for hot_device in [false, true] {
        let mut net = family(1, 8, 16, 0, 0).build().unwrap();
        let transfers = (0..8)
            .map(|index| {
                let device = if hot_device { 0 } else { index };
                let host = if hot_device { index } else { 0 };
                (
                    net.endpoint(&format!("D0_{device}")).unwrap(),
                    net.endpoint(&format!("A0_{host}")).unwrap(),
                    MIB,
                )
            })
            .collect::<Vec<_>>();
        let expected = 8.0 * MIB as f64 / (if hot_device { 8.0 } else { 16.0 }) / U;
        for time in finish(&mut net.flow, &transfers) {
            close(time, expected);
        }
    }
}

#[test]
fn concentrated_remote_demand_saturates_one_uplink_before_endpoints() {
    let topology = family(2, 2, 4, 1, 1);
    let mut net = topology.build().unwrap();
    let transfers = (0..4)
        .map(|device| {
            (
                net.endpoint(&format!("D0_{device}")).unwrap(),
                net.endpoint(&format!("A1_{}", device / 2)).unwrap(),
                MIB,
            )
        })
        .collect::<Vec<_>>();
    let expected = 4.0 * MIB as f64 / 16e9;
    for time in finish(&mut net.flow, &transfers) {
        close(time, expected);
    }
    let snapshot = net.observe();
    let links = snapshot["flow"]["links"].as_array().unwrap();
    for physical in &topology.links {
        if physical.a != "L0" || physical.b != "S0" {
            continue;
        }
        let uplink = links
            .iter()
            .find(|link| link["name"] == format!("{}:ab", physical.id))
            .unwrap();
        close(uplink["bytes"].as_f64().unwrap(), 4.0 * MIB as f64);
        close(
            uplink["bytes"].as_f64().unwrap() / uplink["busy_seconds"].as_f64().unwrap() / 1e9,
            16.0,
        );
    }
    // Each distinct x8 source supplies 4 GB/s: the x16 remote cut is limiting.
    for device in 0..4 {
        let service = links
            .iter()
            .find(|link| link["name"] == format!("D0_{device}:read"))
            .unwrap();
        close(service["bytes"].as_f64().unwrap() / expected / 1e9, 4.0);
    }
    println!(
        "hot_uplink expected_s={expected:.12} observed_s={:.12} delivered_GBs=16",
        snapshot["flow"]["time"].as_f64().unwrap()
    );
}

#[test]
fn memory_service_rate_is_separate_from_physical_link_rate() {
    let mut topology = family(1, 1, 1, 0, 0);
    let memory = topology
        .endpoints
        .iter_mut()
        .find(|endpoint| endpoint.kind == EndpointKind::Memory)
        .unwrap();
    memory.read_bandwidth = Some(2e9);
    memory.write_bandwidth = Some(4e9);
    for write in [false, true] {
        let mut net = topology.build().unwrap();
        let device = net.endpoint("D0_0").unwrap();
        let host = net.endpoint("A0_0").unwrap();
        let transfer = if write {
            (host, device, MIB)
        } else {
            (device, host, MIB)
        };
        close(
            finish(&mut net.flow, &[transfer])[0],
            MIB as f64 / if write { 4e9 } else { 2e9 },
        );
    }
}

#[test]
fn configured_gpu_ingress_shares_service_only_when_links_overlap() {
    for shared in [false, true] {
        let mut net = FlowNetworkSimulator::default();
        for _ in 0..6 {
            net.add_device();
        }
        for source in 0..2 {
            net.add_link(source, source + 2, 64e9, 0.0).unwrap();
            net.add_link(source + 2, if shared { 4 } else { source + 4 }, 64e9, 0.0)
                .unwrap();
        }
        if shared {
            net.add_link(4, 5, 64e9, 0.0).unwrap();
        }
        let transfers = [(0, if shared { 5 } else { 4 }, 8 * MIB), (1, 5, 8 * MIB)];
        let expected = if shared { 0.000262144 } else { 0.000131072 };
        for time in finish(&mut net, &transfers) {
            close(time, expected);
        }
        println!(
            "shared_ingress={shared} expected_s={expected:.12} observed_s={:.12}",
            net.observe()["time"].as_f64().unwrap()
        );
    }
}

fn bound_workers(shared_adapter: bool) -> Topology {
    let mut topology = family(1, 2, 2, 0, 0);
    for link in &mut topology.links {
        link.bandwidth_ab = 64e9;
        link.bandwidth_ba = Some(64e9);
    }
    topology
        .endpoints
        .push(endpoint("fabric".into(), EndpointKind::Fabric));
    for worker in 0..2 {
        let node = if shared_adapter { 0 } else { worker };
        let adapter = format!("A0_{node}");
        let dram = format!("dram{worker}");
        let nic = format!("nic{worker}");
        let gpu = format!("gpu{worker}");
        let ingress = format!("ingress{worker}");
        for (id, kind) in [
            (&dram, EndpointKind::Dram),
            (&nic, EndpointKind::Nic),
            (&gpu, EndpointKind::Gpu),
            (&ingress, EndpointKind::Junction),
        ] {
            topology.endpoints.push(endpoint(id.clone(), kind));
        }
        for (a, b, bidirectional, rate) in [
            (&dram, &adapter, false, 1e12),
            (&adapter, &ingress, false, 1e12),
            (&dram, &nic, false, 1e12),
            (&nic, &"fabric".to_owned(), true, 1e12),
            (&nic, &ingress, false, 1e12),
            (&dram, &ingress, false, 1e12),
            (&ingress, &gpu, false, 64e9),
        ] {
            let id = format!("{a}-{b}");
            topology.links.push(Link {
                id: id.clone(),
                a: a.clone(),
                b: b.clone(),
                a_port: id.clone(),
                b_port: id,
                width: None,
                bandwidth_ab: rate,
                bandwidth_ba: bidirectional.then_some(rate),
                latency: 0.0,
            });
        }
        topology.workers.push(WorkerBinding {
            worker,
            rank: 0,
            node: format!("N0_{node}"),
            adapters: vec![adapter],
            dram,
            nic,
            gpu,
        });
    }
    topology
}

#[test]
fn worker_bindings_share_one_adapter_and_keep_cxl_and_rdma_routes_separate() {
    for shared in [false, true] {
        let topology = bound_workers(shared);
        let mut net = topology.build().unwrap();
        let transfers = (0..2)
            .map(|worker| {
                (
                    net.endpoint(&format!("D0_{worker}")).unwrap(),
                    net.endpoint(&format!("gpu{worker}")).unwrap(),
                    8 * MIB,
                )
            })
            .collect::<Vec<_>>();
        let expected = if shared { 0.000262144 } else { 0.000131072 };
        for time in finish(&mut net.flow, &transfers) {
            close(time, expected);
        }
        println!(
            "shared_adapter={shared} expected_s={expected:.12} observed_s={:.12}",
            net.observe()["flow"]["time"].as_f64().unwrap()
        );
        let snapshot = net.observe();
        let links = snapshot["flow"]["links"].as_array().unwrap();
        for route in snapshot["flow"]["routes"].as_array().unwrap() {
            for path in route["paths"].as_array().unwrap() {
                for id in path["links"].as_array().unwrap() {
                    let name = links[id.as_u64().unwrap() as usize]["name"]
                        .as_str()
                        .unwrap();
                    assert!(!name.contains("nic") && !name.contains("dram"));
                }
            }
        }
        if !shared {
            let routes = net.route_info("dram0", "gpu1").unwrap();
            for route in routes {
                let names = route
                    .links
                    .iter()
                    .map(|&index| links[index]["name"].as_str().unwrap())
                    .collect::<Vec<_>>();
                assert!(names.contains(&"dram0-nic0:ab"));
                assert!(names.contains(&"nic0-fabric:ab"));
                assert!(names.contains(&"nic1-fabric:ba"));
                assert!(!names.iter().any(|name| name.starts_with('L')));
            }
        }
    }
    let mut invalid = bound_workers(false);
    invalid.workers[1].node = "N0_0".into();
    assert!(invalid
        .build()
        .err()
        .unwrap()
        .to_string()
        .contains("binding mismatch"));
    let mut multiple = bound_workers(false);
    multiple
        .endpoints
        .iter_mut()
        .find(|endpoint| endpoint.id == "A0_1")
        .unwrap()
        .node = Some("N0_0".into());
    multiple.workers[1].node = "N0_0".into();
    assert!(multiple.build().is_ok());
}

fn logical(topology: &mut Topology, a: &str, b: &str, bandwidth: f64) {
    let id = format!("{a}-{b}");
    topology.links.push(Link {
        id: id.clone(),
        a: a.into(),
        b: b.into(),
        a_port: id.clone(),
        b_port: id,
        width: None,
        bandwidth_ab: bandwidth,
        bandwidth_ba: None,
        latency: 0.0,
    });
}

fn multiple_adapters(common_downstream: bool) -> Topology {
    let mut topology = bound_workers(false);
    for endpoint in &mut topology.endpoints {
        if endpoint.kind == EndpointKind::Adapter {
            endpoint.node = Some("N0_0".into());
        }
    }
    for worker in &mut topology.workers {
        worker.node = "N0_0".into();
    }
    for (a, b) in [
        ("dram0", "A0_1"),
        ("dram1", "A0_0"),
        ("A0_0", "ingress1"),
        ("A0_1", "ingress0"),
    ] {
        logical(&mut topology, a, b, 1e12);
    }
    if common_downstream {
        topology
            .links
            .retain(|link| !link.id.starts_with("ingress"));
        for id in ["common", "downstream"] {
            topology
                .endpoints
                .push(endpoint(id.into(), EndpointKind::Junction));
        }
        for ingress in ["ingress0", "ingress1"] {
            logical(&mut topology, ingress, "common", 1e12);
        }
        logical(&mut topology, "common", "downstream", 32e9);
        for gpu in ["gpu0", "gpu1"] {
            logical(&mut topology, "downstream", gpu, 1e12);
        }
    }
    topology
}

#[test]
fn shared_and_dedicated_adapter_handles_preserve_physical_contention() {
    // Two 8 MiB reads: 16 MiB / 64 GB/s shared, 8 MiB / 64 GB/s dedicated,
    // or 16 MiB / 32 GB/s on the common downstream link.
    for (shared, common, expected) in [
        (true, false, 0.000262144),
        (false, false, 0.000131072),
        (false, true, 0.000524288),
    ] {
        let mut topology = multiple_adapters(common);
        if shared {
            topology.workers[1].adapters = vec!["A0_0".into()];
        }
        let mut net = topology.build().unwrap();
        assert_eq!(net.observe()["lanes"]["L0"], 48);
        let transfers = (0..2)
            .map(|worker| {
                let (src, dst) = net
                    .transfer_route(
                        &format!("D0_{worker}"),
                        &format!("gpu{worker}"),
                        None,
                        Some((worker, 0)),
                        0,
                    )
                    .unwrap();
                (src, dst, 8 * MIB)
            })
            .collect::<Vec<_>>();
        for time in finish(&mut net.flow, &transfers) {
            close(time, expected);
        }
        let observed = net.observe();
        close(
            observed["adapters"]["A0_0"]["incoming_bytes"]
                .as_f64()
                .unwrap(),
            (if shared { 16 } else { 8 }) as f64 * MIB as f64,
        );
        close(
            observed["adapters"]["A0_1"]["incoming_bytes"]
                .as_f64()
                .unwrap(),
            (if shared { 0 } else { 8 }) as f64 * MIB as f64,
        );
        assert_eq!(observed["flow"]["active_subflows"], 0);
        println!("shared={shared} common_downstream={common} expected_s={expected:.12} observed_s={:.12}", observed["flow"]["time"].as_f64().unwrap());
    }
}

#[test]
fn overlapping_physical_endpoints_keep_worker_routes_independent() {
    let mut topology = multiple_adapters(false);
    topology.workers[1].dram = "dram0".into();
    topology.workers[1].gpu = "gpu0".into();
    topology
        .links
        .iter_mut()
        .find(|link| link.id == "ingress0-gpu0")
        .unwrap()
        .bandwidth_ab = 1e12;
    let mut net = topology.build().unwrap();
    let routes = (0..2)
        .map(|worker| {
            net.transfer_route(&format!("D0_{worker}"), "gpu0", None, Some((worker, 0)), 0)
                .unwrap()
        })
        .collect::<Vec<_>>();
    for time in finish(
        &mut net.flow,
        &routes
            .iter()
            .map(|&(src, dst)| (src, dst, 8 * MIB))
            .collect::<Vec<_>>(),
    ) {
        close(time, 0.000131072);
    }
    for adapter in ["A0_0", "A0_1"] {
        close(
            net.observe()["adapters"][adapter]["incoming_bytes"]
                .as_f64()
                .unwrap(),
            8.0 * MIB as f64,
        );
    }
    let first = net
        .transfer_route("dram0", "D0_0", Some((0, 0)), None, 0)
        .unwrap();
    let second = net
        .transfer_route("dram0", "D0_0", Some((1, 0)), None, 0)
        .unwrap();
    assert_ne!(first, second);
    assert_eq!(
        first,
        net.transfer_route("dram0", "D0_0", Some((0, 0)), None, 0)
            .unwrap()
    );
    let observation = net.observe();
    assert_eq!(
        observation["transfer_routes"][2]["selected_adapter"],
        "A0_0"
    );
    assert_eq!(
        observation["transfer_routes"][3]["selected_adapter"],
        "A0_1"
    );
    let mut net = topology.build().unwrap();
    let writes = (0..2)
        .map(|worker| {
            let (src, dst) = net
                .transfer_route("dram0", "D0_0", Some((worker, 0)), None, 0)
                .unwrap();
            (src, dst, 8 * MIB)
        })
        .collect::<Vec<_>>();
    for time in finish(&mut net.flow, &writes) {
        close(time, 0.000262144);
    }
    for adapter in ["A0_0", "A0_1"] {
        close(
            net.observe()["adapters"][adapter]["outgoing_bytes"]
                .as_f64()
                .unwrap(),
            8.0 * MIB as f64,
        );
    }
    let (src, dst) = net
        .transfer_route("gpu0", "gpu0", Some((0, 0)), Some((0, 0)), 0)
        .unwrap();
    assert_eq!(src, dst);
}

#[test]
fn multi_adapter_block_and_rank_selection_is_deterministic() {
    let mut topology = multiple_adapters(false);
    topology.workers[0].adapters = vec!["A0_0".into(), "A0_1".into()];
    topology.workers[1].worker = 0;
    topology.workers[1].rank = 1;
    topology.workers[1].adapters = topology.workers[0].adapters.clone();
    topology
        .links
        .iter_mut()
        .find(|link| link.id == "ingress0-gpu0")
        .unwrap()
        .bandwidth_ab = 1e12;
    let mut net = topology.build().unwrap();
    let transfers = (0..2)
        .map(|block| {
            let (src, dst) = net
                .transfer_route(&format!("D0_{block}"), "gpu0", None, Some((0, 0)), block)
                .unwrap();
            (src, dst, 8 * MIB)
        })
        .collect::<Vec<_>>();
    for time in finish(&mut net.flow, &transfers) {
        close(time, 0.000131072);
    }
    net.transfer_route("D0_0", "gpu1", None, Some((0, 1)), 0)
        .unwrap();
    let observation = net.observe();
    assert_eq!(
        observation["transfer_routes"][0]["selected_adapter"],
        "A0_0"
    );
    assert_eq!(
        observation["transfer_routes"][1]["selected_adapter"],
        "A0_1"
    );
    assert_eq!(
        observation["transfer_routes"][2]["selected_adapter"],
        "A0_1"
    );
    assert_eq!(net.worker_rank(0, 0).unwrap().rank, 0);
    assert_eq!(net.worker_rank(0, 1).unwrap().rank, 1);
    assert!(net.worker_rank(0, 2).is_err());
}

#[test]
fn adapter_bindings_accept_legacy_scalar_and_reject_invalid_selections() {
    let scalar: WorkerBinding = serde_json::from_value(serde_json::json!({"worker":0,"node":"N0_0","adapter":"A0_0","dram":"dram0","nic":"nic0","gpu":"gpu0"})).unwrap();
    assert_eq!(scalar.rank, 0);
    assert_eq!(scalar.adapters, ["A0_0"]);
    for adapters in [
        vec![],
        vec!["A0_0".into(), "A0_0".into()],
        vec!["missing".into()],
    ] {
        let mut topology = multiple_adapters(false);
        topology.workers[0].adapters = adapters;
        assert!(topology.build().is_err());
    }
    let mut topology = multiple_adapters(false);
    topology.workers[1].rank = 0;
    topology.workers[1].worker = 0;
    assert!(topology
        .build()
        .err()
        .unwrap()
        .to_string()
        .contains("duplicate worker"));
    let mut topology = multiple_adapters(false);
    topology
        .required_routes
        .push(simthesizer_core::network_sims::topology::RouteRequirement {
            src: "D0_0".into(),
            dst: "gpu1".into(),
            via: vec!["A0_0".into()],
        });
    let mut net = topology.build().unwrap();
    assert!(net
        .transfer_route("D0_0", "gpu1", None, Some((1, 0)), 0)
        .unwrap_err()
        .to_string()
        .contains("bypasses selected adapter"));
}

/// Sixteen switches in a ring; worker 0 and device D on S0, worker 1 on S8.
/// Through worker 0's adapter, C2C and the NICs, D is fewer hops from worker
/// 1's adapter than around the ring.
fn ring_with_a_shorter_host_relay() -> Topology {
    let mut topology = Topology {
        endpoints: vec![endpoint("fabric".into(), EndpointKind::Fabric)],
        links: Vec::new(),
        workers: Vec::new(),
        required_routes: Vec::new(),
        routing: Routing::EqualShortest,
    };
    for s in 0..16 {
        let id = format!("S{s}");
        topology.endpoints.push(endpoint(id, EndpointKind::Switch));
    }
    for s in 0..16 {
        physical(
            &mut topology,
            &format!("S{s}"),
            &format!("S{}", (s + 1) % 16),
            16,
        );
    }
    let mut memory = endpoint("D".into(), EndpointKind::Memory);
    memory.capacity_bytes = Some(1 << 30);
    memory.read_bandwidth = Some(U);
    memory.write_bandwidth = Some(U);
    topology.endpoints.push(memory);
    physical(&mut topology, "D", "S0", 8);
    for (worker, switch) in [(0, 0), (1, 8)] {
        let [adapter, dram, c2c, nic, gpu] =
            ["adapter", "dram", "c2c", "nic", "gpu"].map(|kind| format!("{kind}{worker}"));
        for (id, kind) in [
            (&adapter, EndpointKind::Adapter),
            (&dram, EndpointKind::Dram),
            (&c2c, EndpointKind::Junction),
            (&nic, EndpointKind::Nic),
            (&gpu, EndpointKind::Gpu),
        ] {
            let mut node = endpoint(id.clone(), kind);
            node.node = Some(format!("node{worker}"));
            topology.endpoints.push(node);
        }
        physical(&mut topology, &adapter, &format!("S{switch}"), 16);
        logical(&mut topology, &dram, &adapter, U);
        logical(&mut topology, &adapter, &c2c, U);
        for (a, b) in [(&dram, &c2c), (&dram, &nic), (&nic, &c2c), (&c2c, &gpu)] {
            logical(&mut topology, a, b, U);
            logical(&mut topology, b, a, U);
        }
        logical(&mut topology, &nic, "fabric", U);
        logical(&mut topology, "fabric", &nic, U);
        topology.workers.push(WorkerBinding {
            worker,
            rank: 0,
            node: format!("node{worker}"),
            adapters: vec![adapter],
            dram,
            nic,
            gpu,
        });
    }
    topology
}

#[test]
fn cxl_transfers_stay_on_the_switch_fabric_when_a_host_relay_is_shorter() {
    let mut net = ring_with_a_shorter_host_relay().build().unwrap();
    let routes = [
        (
            net.transfer_route("D", "gpu1", None, Some((1, 0)), 0)
                .unwrap(),
            ["D:read", "adapter1-c2c1:ab", "c2c1-gpu1:ab"].as_slice(),
        ),
        (
            net.transfer_route("dram1", "D", Some((1, 0)), None, 0)
                .unwrap(),
            ["dram1-adapter1:ab", "D:write"].as_slice(),
        ),
    ];
    let links = net.observe()["flow"]["links"].clone();
    for ((src, dst), host) in routes {
        let paths = net.flow.route_info(src, dst).unwrap();
        // Both directions around the ring tie.
        assert_eq!(paths.len(), 2);
        for path in paths {
            let (physical, other): (Vec<_>, Vec<_>) = path
                .links
                .iter()
                .map(|&index| links[index]["name"].as_str().unwrap())
                .partition(|name| name.starts_with('L'));
            // D's port, eight ring hops and worker 1's adapter port.
            assert_eq!(physical.len(), 10);
            assert_eq!(other, host);
        }
    }
    // Without a CXL path, the host relay does not stand in for one.
    let mut cut = ring_with_a_shorter_host_relay();
    cut.links.retain(|link| link.id != "L3" && link.id != "L11");
    let error = cut.build().err().unwrap();
    assert!(format!("{error:#}").contains("no directed CXL fabric route"));
}
