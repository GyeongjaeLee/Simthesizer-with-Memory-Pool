mod common;

use std::any::Any;

use anyhow::{bail, Result};
use common::close;
use simthesizer_core::network_sims::flow::FlowNetworkSimulator;
use simthesizer_core::types::PollResult;
use simthesizer_core::NetworkSimulator;

fn network(devices: usize, links: &[(u64, u64, f64, f64)]) -> FlowNetworkSimulator {
    let mut net = FlowNetworkSimulator::default();
    for _ in 0..devices {
        net.add_device();
    }
    for &(src, dst, bandwidth, latency) in links {
        net.add_link(src, dst, bandwidth, latency).unwrap();
    }
    net
}

fn finish(net: &mut FlowNetworkSimulator, transfers: &[(u64, u64, u64, f64)]) -> Result<Vec<f64>> {
    let mut states: Vec<Option<Box<dyn Any>>> = (0..transfers.len()).map(|_| None).collect();
    let mut pending: Vec<_> = transfers
        .iter()
        .enumerate()
        .map(|(i, t)| (t.3, i))
        .collect();
    let mut done = vec![f64::NAN; transfers.len()];
    for _ in 0..10000 {
        pending.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| b.1.cmp(&a.1)));
        let Some((now, index)) = pending.pop() else {
            return Ok(done);
        };
        let (src, dst, bytes, _) = transfers[index];
        match net.simulate(index as u64, src, dst, bytes, now, &mut states[index])? {
            PollResult::Deferred => pending.push((now, index)),
            PollResult::Complete { latency } => done[index] = now + latency,
            PollResult::Pending { latency } => {
                assert!(latency.is_finite() && now + latency > now);
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
    bail!("network failed to drain")
}

#[test]
fn transfer_charges_bytes_over_bandwidth_then_propagation() {
    let mut net = network(2, &[(0, 1, 10.0, 0.5)]);
    close(finish(&mut net, &[(0, 1, 100, 2.0)]).unwrap()[0], 12.5);
}

#[test]
fn equal_flows_share_bandwidth() {
    let mut net = network(2, &[(0, 1, 10.0, 0.0)]);
    let done = finish(&mut net, &[(0, 1, 100, 0.0), (0, 1, 100, 0.0)]).unwrap();
    for time in done {
        close(time, 20.0);
    }
}

#[test]
fn a_short_flow_releases_bandwidth_without_delaying_the_long_flows_completion() {
    for bytes in [[10, 100], [100, 10]] {
        let mut net = network(2, &[(0, 1, 10.0, 0.0)]);
        let done = finish(&mut net, &[(0, 1, bytes[0], 0.0), (0, 1, bytes[1], 0.0)]).unwrap();
        for (bytes, time) in bytes.into_iter().zip(done) {
            close(time, if bytes == 10 { 2.0 } else { 11.0 });
        }
    }
}

#[test]
fn capacity_a_bottlenecked_flow_cannot_use_goes_to_the_others() {
    // 0 -> 2 is capped at 10 before the shared 2 -> 3 link (64). Max-min gives
    // 10 and 54, so both payloads finish together at 1 s; an equal split would
    // hold the second flow at 32 until the first ends (1.34375 s).
    let mut net = network(
        4,
        &[(0, 2, 10.0, 0.0), (1, 2, 1000.0, 0.0), (2, 3, 64.0, 0.0)],
    );
    let done = finish(&mut net, &[(0, 3, 10, 0.0), (1, 3, 54, 0.0)]).unwrap();
    close(done[0], 1.0);
    close(done[1], 1.0);
}

#[test]
fn a_later_flow_that_speeds_up_another_brings_its_completion_forward() {
    // A uses 0 -> 1, B uses 0 -> 1 -> 2, C uses 1 -> 2; all get 5, so A is due
    // at 20 s. D joins 1 -> 2 at 11 s, cutting B to 10/3, so A rises to 20/3
    // until D leaves at 12.5 s and finishes its last 35 bytes at 19.5 s.
    let mut net = network(3, &[(0, 1, 10.0, 0.0), (1, 2, 10.0, 0.0)]);
    let transfers = [
        (0, 1, 100, 0.0),
        (0, 2, 1000, 0.0),
        (1, 2, 1000, 0.0),
        (1, 2, 5, 11.0),
    ];
    let done = finish(&mut net, &transfers).unwrap();
    close(done[3], 12.5);
    close(done[0], 19.5);
}

#[test]
fn staggered_arrivals_change_rates_only_after_arrival() {
    let mut net = network(2, &[(0, 1, 10.0, 0.0)]);
    let done = finish(&mut net, &[(0, 1, 10, 0.0), (0, 1, 20, 0.5)]).unwrap();
    close(done[0], 1.5);
    close(done[1], 3.0);
}

#[test]
fn propagation_does_not_hold_link_bandwidth() {
    let mut net = network(2, &[(0, 1, 10.0, 2.0)]);
    let done = finish(&mut net, &[(0, 1, 10, 0.0), (0, 1, 10, 1.0)]).unwrap();
    close(done[0], 3.0);
    close(done[1], 4.0);
}

#[test]
fn disjoint_and_opposite_links_do_not_contend() {
    let mut net = network(
        4,
        &[(0, 1, 10.0, 0.0), (1, 0, 10.0, 0.0), (2, 3, 10.0, 0.0)],
    );
    for time in finish(
        &mut net,
        &[(0, 1, 10, 0.0), (1, 0, 10, 0.0), (2, 3, 10, 0.0)],
    )
    .unwrap()
    {
        close(time, 1.0);
    }
}

#[test]
fn widest_route_ties_use_end_to_end_latency() {
    let mut net = network(
        4,
        &[
            (0, 1, 10.0, 10.0),
            (0, 2, 8.0, 1.0),
            (2, 1, 8.0, 0.0),
            (1, 3, 5.0, 1.0),
        ],
    );
    // Both complete routes bottleneck at 5 B/s; 0->2->1->3 has 2 s latency.
    close(finish(&mut net, &[(0, 3, 40, 0.0)]).unwrap()[0], 10.0);
}

#[test]
fn a_topology_change_invalidates_cached_routes() {
    let mut net = network(2, &[(0, 1, 10.0, 1.0)]);
    close(finish(&mut net, &[(0, 1, 10, 0.0)]).unwrap()[0], 2.0);
    net.add_link(0, 1, 20.0, 0.0).unwrap();
    close(finish(&mut net, &[(0, 1, 10, 2.0)]).unwrap()[0], 2.5);
}

#[test]
fn zero_byte_and_local_transfers_need_no_link() {
    let mut net = network(2, &[]);
    assert_eq!(
        finish(&mut net, &[(0, 1, 0, 3.0), (0, 0, 100, 3.0)]).unwrap(),
        vec![3.0, 3.0]
    );
}

#[test]
fn unreachable_destination_and_invalid_links_fail() {
    let mut net = network(2, &[]);
    assert!(finish(&mut net, &[(0, 1, 1, 0.0)]).is_err());
    for bandwidth in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(net.add_link(0, 1, bandwidth, 0.0).is_err());
    }
    for latency in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(net.add_link(0, 1, 1.0, latency).is_err());
    }
}

#[test]
fn network_time_must_be_finite_and_monotone() {
    for now in [-1.0, f64::NAN, f64::INFINITY] {
        let mut net = network(2, &[(0, 1, 1.0, 0.0)]);
        assert!(net.simulate(0, 0, 1, 1, now, &mut None).is_err());
    }
    let mut net = network(2, &[]);
    net.simulate(0, 0, 0, 0, 2.0, &mut None).unwrap();
    assert!(net.simulate(0, 0, 0, 0, 1.0, &mut None).is_err());
}
