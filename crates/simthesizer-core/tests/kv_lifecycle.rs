mod common;

use serde_json::json;
use simthesizer_core::types::{KvPrefix, ParallelStrategy, ReadySubRequest, Request, RequestState};
use simthesizer_core::Scheduler;

fn scheduler(memory: u64, budget: u32, max_batch: usize, caching: bool) -> Box<dyn Scheduler> {
    configured(memory, budget, max_batch, caching, 0, true)
}

fn hybrid(memory: u64, budget: u32) -> Box<dyn Scheduler> {
    let mut scheduler = scheduler(memory, budget, 1, true);
    scheduler
        .init(
            0,
            8,
            1,
            ParallelStrategy {
                tp: 1,
                pp: 1,
                ep: 1,
            },
        )
        .unwrap();
    scheduler
}

/// Batch 3, budget 16, with `fixed` per-request state bytes.
fn stateful(memory: u64, fixed: u64, lookahead: u32, caching: bool) -> Box<dyn Scheduler> {
    let mut scheduler = configured(memory, 16, 3, caching, lookahead, true);
    scheduler
        .init(
            0,
            fixed,
            1,
            ParallelStrategy {
                tp: 1,
                pp: 1,
                ep: 1,
            },
        )
        .unwrap();
    scheduler
}

fn configured(
    memory: u64,
    budget: u32,
    max_batch: usize,
    caching: bool,
    lookahead: u32,
    reserve_full_input: bool,
) -> Box<dyn Scheduler> {
    let mut scheduler: Box<dyn Scheduler> = serde_json::from_value(json!({
        "kind": "chunked_prefill", "block_size": 4,
        "max_batch": max_batch, "max_num_batched_tokens": budget,
        "device_mem": memory, "device_mem_utilization": 1.0,
        "enable_prefix_caching": caching, "num_lookahead_tokens": lookahead,
        "scheduler_reserve_full_isl": reserve_full_input,
    }))
    .unwrap();
    scheduler
        .init(
            0,
            0,
            1,
            ParallelStrategy {
                tp: 1,
                pp: 1,
                ep: 1,
            },
        )
        .unwrap();
    scheduler
}

fn request(id: u64, input: u32, output: u32) -> ReadySubRequest {
    ReadySubRequest {
        request_id: id,
        subrequest_index: 0,
        completion_node_id: id,
        arrival_time: 0.0,
        input_tokens: input,
        output_tokens: output,
        known_tokens: input,
        kv_tokens: 0,
        input_token_ids: None,
        output_token_ids: None,
        kv_prefix: None,
    }
}

fn proxy(id: u64, namespace: &str, blocks: &[u64], output: u32) -> ReadySubRequest {
    let mut request = request(id, blocks.len() as u32 * 4, output);
    request.kv_prefix = Some(KvPrefix {
        namespace: namespace.into(),
        block_tokens: 4,
        block_ids: blocks.to_vec(),
    });
    request
}

fn shape(batch: &[RequestState]) -> Vec<(u32, u32)> {
    batch.iter().map(|item| (item.q_len, item.kv_len)).collect()
}

fn complete_one(scheduler: &mut dyn Scheduler, request: ReadySubRequest, now: f64) {
    let expected = request.request_id;
    scheduler.enqueue_sub_request(request);
    assert_eq!(scheduler.schedule().unwrap().len(), 1);
    let completed = scheduler.done_iteration(now);
    assert_eq!(completed.len(), 1);
    assert_eq!(completed[0].request_id, expected);
}

#[test]
fn compatible_proxy_identity_reuses_across_requests_after_release() {
    let mut scheduler = scheduler(64, 64, 1, true);
    complete_one(
        scheduler.as_mut(),
        proxy(0, "model/layout/scope", &[1, 2, 3], 1),
        1.0,
    );
    assert_eq!(scheduler.observe()["prefix_cache_bytes"], 12);
    assert_eq!(scheduler.observe()["running_requests"], 0);
    let mut later = proxy(99, "model/layout/scope", &[1, 2, 3], 1);
    later.arrival_time = 1000.0;
    scheduler.enqueue_sub_request(later);
    // The final block is recomputed for logits, even after a complete cache hit.
    assert_eq!(shape(&scheduler.schedule().unwrap()), [(4, 8)]);
    assert_eq!(scheduler.done_iteration(1001.0)[0].token_latency, [1.0]);
}

#[test]
fn proxy_namespace_and_continuous_parent_identity_control_hits() {
    for (namespace, blocks, expected) in [
        ("other-layout", [1, 2, 3], (12, 0)),
        ("shared-layout", [9, 2, 3], (12, 0)),
        ("shared-layout", [1, 9, 3], (8, 4)),
    ] {
        let mut scheduler = scheduler(64, 64, 1, true);
        complete_one(
            scheduler.as_mut(),
            proxy(0, "shared-layout", &[1, 2, 3], 1),
            1.0,
        );
        scheduler.enqueue_sub_request(proxy(1, namespace, &blocks, 1));
        assert_eq!(shape(&scheduler.schedule().unwrap()), [expected]);
    }
}

#[test]
fn original_sixteen_token_proxy_blocks_reuse_at_compatible_storage_boundaries() {
    for physical_block_tokens in [16, 32] {
        let mut scheduler: Box<dyn Scheduler> = serde_json::from_value(json!({
            "kind": "chunked_prefill", "block_size": physical_block_tokens,
            "max_batch": 1, "max_num_batched_tokens": 128,
            "device_mem": 512, "device_mem_utilization": 1.0,
            "enable_prefix_caching": true,
        }))
        .unwrap();
        scheduler
            .init(
                0,
                0,
                1,
                ParallelStrategy {
                    tp: 1,
                    pp: 1,
                    ep: 1,
                },
            )
            .unwrap();
        let mut first = request(0, 48, 1);
        first.kv_prefix = Some(KvPrefix {
            namespace: "revision/week/source/target/layout/scope".into(),
            block_tokens: 16,
            block_ids: vec![1, 2, 3],
        });
        complete_one(scheduler.as_mut(), first.clone(), 1.0);
        first.request_id = 1;
        first.completion_node_id = 1;
        scheduler.enqueue_sub_request(first);
        assert_eq!(shape(&scheduler.schedule().unwrap()), [(16, 32)]);
    }
}

#[test]
fn repeated_request_membership_without_identity_is_a_cache_miss() {
    let mut scheduler = scheduler(64, 64, 1, true);
    complete_one(scheduler.as_mut(), request(0, 12, 1), 1.0);
    let mut next_call = request(0, 12, 1);
    next_call.subrequest_index = 1;
    next_call.completion_node_id = 1;
    scheduler.enqueue_sub_request(next_call);
    assert_eq!(shape(&scheduler.schedule().unwrap()), [(12, 0)]);
    assert_eq!(scheduler.observe()["prefix_cache_bytes"], 12);
}

#[test]
fn generated_proxy_decode_cannot_establish_next_input_identity() {
    let mut scheduler = scheduler(64, 64, 1, true);
    scheduler.enqueue_sub_request(proxy(0, "shared", &[1, 2], 9));
    for now in 1..=9 {
        assert_eq!(scheduler.schedule().unwrap().len(), 1);
        let completed = scheduler.done_iteration(f64::from(now));
        assert_eq!(completed.len(), usize::from(now == 9));
    }
    assert_eq!(scheduler.observe()["prefix_cache_bytes"], 16);
    scheduler.enqueue_sub_request(proxy(1, "shared", &[1, 2, 3, 4], 1));
    assert_eq!(shape(&scheduler.schedule().unwrap()), [(8, 8)]);
}

#[test]
fn partial_input_identity_does_not_extend_through_unknown_tokens() {
    let mut scheduler = scheduler(64, 64, 1, true);
    let mut first = proxy(0, "shared", &[1], 1);
    first.input_tokens = 7;
    first.known_tokens = 7;
    complete_one(scheduler.as_mut(), first, 1.0);
    scheduler.enqueue_sub_request(proxy(1, "shared", &[1, 2, 3], 1));
    assert_eq!(shape(&scheduler.schedule().unwrap()), [(8, 4)]);
}

#[test]
fn cache_pressure_reclaims_only_the_tail_blocks_needed() {
    let mut scheduler = scheduler(12, 64, 1, true);
    complete_one(scheduler.as_mut(), proxy(0, "shared", &[1, 2, 3], 1), 1.0);
    // One opaque block replaces one cached tail and remains cached on release.
    complete_one(scheduler.as_mut(), request(1, 4, 1), 2.0);
    assert_eq!(scheduler.observe()["prefix_cache_bytes"], 12);
    scheduler.enqueue_sub_request(proxy(2, "shared", &[1, 2, 3], 1));
    assert_eq!(shape(&scheduler.schedule().unwrap()), [(4, 8)]);
}

#[test]
fn full_known_input_admission_does_not_physically_reserve_the_prompt() {
    let mut scheduler = scheduler(16, 1, 1, false);
    scheduler.enqueue_sub_request(request(0, 16, 1000));
    assert_eq!(shape(&scheduler.schedule().unwrap()), [(1, 0)]);
    assert_eq!(scheduler.observe()["hbm_allocated_bytes"], 4);
    assert!(scheduler.done_iteration(1.0).is_empty());
    assert_eq!(scheduler.observe()["hbm_allocated_bytes"], 4);
}

#[test]
fn full_input_admission_gate_does_not_evict_cache_for_future_prompt_chunks() {
    let mut scheduler = scheduler(16, 1, 1, true);
    scheduler.enqueue_sub_request(proxy(0, "shared", &[1, 2, 3], 1));
    for now in 1..=12 {
        scheduler.schedule().unwrap();
        scheduler.done_iteration(f64::from(now));
    }
    assert_eq!(scheduler.observe()["prefix_cache_bytes"], 12);
    scheduler.enqueue_sub_request(request(1, 16, 1));
    assert_eq!(shape(&scheduler.schedule().unwrap()), [(1, 0)]);
    // The full input fits after hypothetical reclamation; only four bytes are
    // needed now, so all twelve cached bytes remain physically resident.
    assert_eq!(scheduler.observe()["prefix_cache_bytes"], 12);
    assert_eq!(scheduler.observe()["hbm_allocated_bytes"], 16);
}

#[test]
fn impossible_full_input_gate_leaves_reclaimable_cache_intact() {
    let mut scheduler = scheduler(12, 64, 1, true);
    complete_one(scheduler.as_mut(), proxy(0, "shared", &[1, 2, 3], 1), 1.0);
    scheduler.enqueue_sub_request(request(1, 16, 1));
    assert!(scheduler.schedule().is_err());
    assert_eq!(scheduler.observe()["prefix_cache_bytes"], 12);
}

#[test]
fn future_output_length_does_not_change_initial_admission() {
    let mut scheduler = scheduler(8, 8, 2, false);
    scheduler.enqueue_sub_request(request(0, 3, 1000));
    scheduler.enqueue_sub_request(request(1, 3, 1000));
    assert_eq!(shape(&scheduler.schedule().unwrap()), [(3, 0), (3, 0)]);
    assert_eq!(scheduler.observe()["hbm_allocated_bytes"], 8);
}

#[test]
fn full_input_gate_blocks_the_fcfs_head_without_bypassing_it() {
    let mut scheduler = scheduler(8, 4, 3, false);
    scheduler.enqueue_sub_request(request(0, 3, 3));
    scheduler.enqueue_sub_request(request(1, 8, 1));
    scheduler.enqueue_sub_request(request(2, 1, 1));
    assert_eq!(shape(&scheduler.schedule().unwrap()), [(3, 0)]);
    assert!(scheduler.done_iteration(1.0).is_empty());
    assert_eq!(shape(&scheduler.schedule().unwrap()), [(1, 3)]);
    assert_eq!(scheduler.observe()["waiting_requests"], 2);
}

#[test]
fn disabling_full_input_gate_changes_chunk_admission_only() {
    let mut scheduler = configured(8, 4, 2, false, 0, false);
    scheduler.enqueue_sub_request(request(0, 3, 3));
    scheduler.enqueue_sub_request(request(1, 8, 1));
    assert_eq!(shape(&scheduler.schedule().unwrap()), [(3, 0), (1, 0)]);
    assert_eq!(scheduler.observe()["hbm_allocated_bytes"], 8);
}

#[test]
fn lookahead_uses_block_rounded_capacity_without_publishing_future_kv() {
    let mut scheduler = configured(8, 8, 2, true, 2, true);
    scheduler.enqueue_sub_request(proxy(0, "shared", &[1], 2));
    scheduler.enqueue_sub_request(request(1, 3, 1));
    assert_eq!(shape(&scheduler.schedule().unwrap()), [(4, 0)]);
    assert_eq!(scheduler.observe()["hbm_allocated_bytes"], 8);
    assert!(scheduler.done_iteration(1.0).is_empty());
    assert_eq!(scheduler.observe()["prefix_cache_bytes"], 4);
}

#[test]
fn lookahead_applies_to_current_work_not_the_full_input_admission_gate() {
    let mut scheduler = configured(16, 1, 1, false, 2, true);
    scheduler.enqueue_sub_request(request(0, 16, 1));
    assert_eq!(shape(&scheduler.schedule().unwrap()), [(1, 0)]);
    assert_eq!(scheduler.observe()["hbm_allocated_bytes"], 4);
}

#[test]
fn memory_growth_preempts_the_newest_running_request_and_resumes_it() {
    let mut scheduler = scheduler(16, 12, 3, false);
    for id in 0..3 {
        scheduler.enqueue_sub_request(request(id, 4, 3));
    }
    assert_eq!(
        shape(&scheduler.schedule().unwrap()),
        [(4, 0), (4, 0), (4, 0)]
    );
    assert!(scheduler.done_iteration(1.0).is_empty());
    assert_eq!(shape(&scheduler.schedule().unwrap()), [(1, 4), (1, 4)]);
    assert_eq!(scheduler.observe()["preemptions_total"], 1);
    assert!(scheduler.done_iteration(2.0).is_empty());
    assert_eq!(shape(&scheduler.schedule().unwrap()), [(1, 5), (1, 5)]);
    let completed = scheduler.done_iteration(3.0);
    assert_eq!(
        completed
            .iter()
            .map(|item| item.request_id)
            .collect::<Vec<_>>(),
        [0, 1]
    );
    // Request 2 resumes its five known tokens, including its already emitted token.
    assert_eq!(shape(&scheduler.schedule().unwrap()), [(5, 0)]);
    assert!(scheduler.done_iteration(4.0).is_empty());
    assert_eq!(shape(&scheduler.schedule().unwrap()), [(1, 5)]);
    assert_eq!(scheduler.done_iteration(5.0)[0].request_id, 2);
    assert!(scheduler.schedule().unwrap().is_empty());
}

#[test]
fn hybrid_replay_requires_a_retained_recurrent_checkpoint() {
    let mut scheduler = hybrid(64, 64);
    scheduler.enqueue_sub_request(proxy(0, "hybrid", &[1, 2], 1));
    assert_eq!(shape(&scheduler.schedule().unwrap()), [(4, 0)]);
    assert_eq!(scheduler.observe()["hbm_allocated_bytes"], 12);
    assert!(scheduler.done_iteration(1.0).is_empty());
    // Four attention bytes and the eight-byte replay checkpoint own the state.
    assert_eq!(scheduler.observe()["prefix_cache_bytes"], 12);
    assert_eq!(scheduler.observe()["hbm_allocated_bytes"], 12);
    assert_eq!(shape(&scheduler.schedule().unwrap()), [(4, 4)]);
    assert_eq!(scheduler.observe()["hbm_allocated_bytes"], 24);
    assert_eq!(scheduler.done_iteration(2.0).len(), 1);
    assert_eq!(scheduler.observe()["prefix_cache_bytes"], 16);
    assert_eq!(scheduler.observe()["hbm_allocated_bytes"], 16);

    // Both attention blocks survived, but only the boundary at four has state.
    scheduler.enqueue_sub_request(proxy(1, "hybrid", &[1, 2, 3], 1));
    assert_eq!(shape(&scheduler.schedule().unwrap()), [(4, 4)]);
    assert!(scheduler.done_iteration(3.0).is_empty());
    assert_eq!(shape(&scheduler.schedule().unwrap()), [(4, 8)]);
    assert_eq!(scheduler.done_iteration(4.0).len(), 1);
}

#[test]
fn hybrid_live_state_copies_exist_only_while_crossing_a_state_block() {
    let mut scheduler = hybrid(64, 64);
    scheduler.enqueue_sub_request(proxy(0, "hybrid", &[1, 2], 5));
    for now in 1..=2 {
        scheduler.schedule().unwrap();
        assert!(scheduler.done_iteration(f64::from(now)).is_empty());
    }
    assert_eq!(shape(&scheduler.schedule().unwrap()), [(1, 8)]);
    // Checkpoint 8 + attention 8 + new attention 4 + old/new live state 16.
    assert_eq!(scheduler.observe()["hbm_allocated_bytes"], 36);
    assert!(scheduler.done_iteration(3.0).is_empty());
    assert_eq!(scheduler.observe()["hbm_allocated_bytes"], 28);
    assert_eq!(shape(&scheduler.schedule().unwrap()), [(1, 9)]);
    assert_eq!(scheduler.observe()["hbm_allocated_bytes"], 28);
}

#[test]
fn hybrid_without_prefix_caching_uses_one_live_state() {
    let mut scheduler = scheduler(64, 64, 1, false);
    scheduler
        .init(
            0,
            8,
            1,
            ParallelStrategy {
                tp: 1,
                pp: 1,
                ep: 1,
            },
        )
        .unwrap();
    scheduler.enqueue_sub_request(request(0, 4, 2));
    assert_eq!(shape(&scheduler.schedule().unwrap()), [(4, 0)]);
    assert_eq!(scheduler.observe()["hbm_allocated_bytes"], 12);
    assert!(scheduler.done_iteration(1.0).is_empty());
    assert_eq!(shape(&scheduler.schedule().unwrap()), [(1, 4)]);
    assert_eq!(scheduler.observe()["hbm_allocated_bytes"], 16);
}

#[test]
fn hybrid_lookahead_reserves_attention_without_inventing_future_state_copies() {
    let mut scheduler = configured(16, 1, 1, true, 4, true);
    scheduler
        .init(
            0,
            8,
            1,
            ParallelStrategy {
                tp: 1,
                pp: 1,
                ep: 1,
            },
        )
        .unwrap();
    scheduler.enqueue_sub_request(proxy(0, "hybrid", &[1, 2], 1));
    assert_eq!(shape(&scheduler.schedule().unwrap()), [(1, 0)]);
    assert_eq!(scheduler.observe()["hbm_allocated_bytes"], 16);
    assert!(scheduler.done_iteration(1.0).is_empty());
    assert_eq!(shape(&scheduler.schedule().unwrap()), [(1, 1)]);
    assert_eq!(scheduler.observe()["hbm_allocated_bytes"], 16);
}

#[test]
fn hybrid_checkpoint_is_reclaimable_after_copy_source_consumption() {
    let mut scheduler = hybrid(32, 64);
    scheduler.enqueue_sub_request(proxy(0, "hybrid", &[1, 2], 5));
    for now in 1..=2 {
        scheduler.schedule().unwrap();
        assert!(scheduler.done_iteration(f64::from(now)).is_empty());
    }
    assert_eq!(scheduler.observe()["prefix_cache_bytes"], 16);
    // Attention stays execution-pinned, but the consumed checkpoint can yield
    // eight bytes for the next old/new live-state pair without preemption.
    assert_eq!(shape(&scheduler.schedule().unwrap()), [(1, 8)]);
    assert_eq!(scheduler.observe()["prefix_cache_bytes"], 8);
    assert_eq!(scheduler.observe()["hbm_allocated_bytes"], 28);
    assert_eq!(scheduler.observe()["preemptions_total"], 0);
}

#[test]
fn external_capacity_wait_preserves_the_fcfs_admission_barrier() {
    let mut scheduler = scheduler(64, 64, 2, false);
    scheduler.configure_external_kv().unwrap();
    for id in 0..3 {
        scheduler.enqueue_sub_request(request(id, 4, 1));
    }
    scheduler.set_external_admission_blocked(0, true);
    assert!(scheduler.schedule().unwrap().is_empty());
    assert_eq!(scheduler.observe()["hbm_admission_waiting_requests"], 3);
    assert_eq!(scheduler.observe()["scheduler_waiting_requests"], 0);
    scheduler.set_external_admission_blocked(0, false);
    assert_eq!(scheduler.schedule().unwrap().len(), 2);
    let completed = scheduler.done_iteration(1.0);
    assert_eq!(
        completed.iter().map(|r| r.request_id).collect::<Vec<_>>(),
        [0, 1]
    );
}

#[test]
fn external_capacity_wait_allows_already_running_work() {
    let mut scheduler = scheduler(64, 64, 2, false);
    scheduler.configure_external_kv().unwrap();
    scheduler.enqueue_sub_request(request(0, 4, 3));
    scheduler.schedule().unwrap();
    scheduler.done_iteration(1.0);
    scheduler.enqueue_sub_request(request(1, 4, 1));
    scheduler.enqueue_sub_request(request(2, 4, 1));
    scheduler.set_external_admission_blocked(1, true);
    assert_eq!(shape(&scheduler.schedule().unwrap()), [(1, 4)]);
    assert_eq!(scheduler.observe()["hbm_admission_waiting_requests"], 2);
    assert!(scheduler.done_iteration(2.0).is_empty());
    scheduler.set_external_admission_blocked(1, false);
    assert_eq!(shape(&scheduler.schedule().unwrap()), [(1, 5), (4, 0)]);
    let completed = scheduler.done_iteration(3.0);
    assert_eq!(
        completed.iter().map(|r| r.request_id).collect::<Vec<_>>(),
        [0, 1]
    );
}

#[test]
fn external_kv_preparation_can_be_skipped_without_bypassing_capacity() {
    let mut scheduler = scheduler(64, 64, 2, false);
    scheduler.configure_external_kv().unwrap();
    scheduler.enqueue_sub_request(request(0, 4, 1));
    scheduler.enqueue_sub_request(request(1, 4, 1));
    scheduler.set_external_blocked(0, true);
    assert_eq!(scheduler.schedule().unwrap().len(), 1);
    assert_eq!(scheduler.done_iteration(1.0)[0].request_id, 1);
    scheduler.set_external_blocked(0, false);
    assert_eq!(scheduler.schedule().unwrap().len(), 1);
    assert_eq!(scheduler.done_iteration(2.0)[0].request_id, 0);
}

#[test]
fn failed_admission_does_not_refresh_the_prefix_free_queue() {
    let mut scheduler = scheduler(20, 64, 2, true);
    complete_one(scheduler.as_mut(), proxy(0, "shared", &[1], 1), 1.0);
    complete_one(scheduler.as_mut(), proxy(1, "other", &[2], 1), 2.0);
    scheduler.enqueue_sub_request(request(2, 4, 13));
    scheduler.schedule().unwrap();
    scheduler.done_iteration(3.0);
    scheduler.enqueue_sub_request(proxy(3, "shared", &[1, 11, 12, 13, 14], 1));
    for step in 1..=12 {
        assert_eq!(scheduler.schedule().unwrap().len(), 1);
        let completed = scheduler.done_iteration(f64::from(step + 3));
        assert_eq!(completed.len(), usize::from(step == 12));
    }
    // Every failed admission probed block 1. It remained oldest, so running
    // growth reclaimed it instead of refreshing it ahead of the other prefix.
    assert_eq!(shape(&scheduler.schedule().unwrap()), [(20, 0)]);
}

#[test]
fn recurrent_prefix_reuses_only_with_a_counted_state_checkpoint() {
    let mut scheduler = stateful(32, 8, 0, true);
    for id in 0..2 {
        let mut request = request(id, 8, 1);
        request.kv_prefix = Some(KvPrefix {
            namespace: "hybrid-model/layout/shared".into(),
            block_tokens: 4,
            block_ids: vec![1, 2],
        });
        scheduler.enqueue_sub_request(request);
        if id == 0 {
            let batch = scheduler.schedule().unwrap();
            assert_eq!((batch[0].q_len, batch[0].kv_len), (4, 0));
            assert!(scheduler.done_iteration(1.0).is_empty());
            // One attention block and one recurrent snapshot: 4 + 8 bytes.
            assert_eq!(scheduler.observe()["prefix_cache_bytes"], 12);
        }
        let batch = scheduler.schedule().unwrap();
        assert_eq!((batch[0].q_len, batch[0].kv_len), (4, 4));
        assert!(scheduler.observe()["hbm_allocated_bytes"].as_u64().unwrap() <= 32);
        assert_eq!(scheduler.done_iteration(2.0 + id as f64).len(), 1);
        // Eight attention tokens plus the four-token-boundary snapshot.
        assert_eq!(scheduler.observe()["prefix_cache_bytes"], 16);
    }
}

#[test]
fn recurrent_hit_stops_at_state_even_when_more_attention_kv_matches() {
    let mut scheduler = stateful(64, 8, 0, true);
    for (id, input, blocks) in [(0, 8, vec![1, 2]), (1, 12, vec![1, 2, 3])] {
        let mut request = request(id, input, 1);
        request.kv_prefix = Some(KvPrefix {
            namespace: "hybrid-model/layout/shared".into(),
            block_tokens: 4,
            block_ids: blocks,
        });
        scheduler.enqueue_sub_request(request);
        let batch = scheduler.schedule().unwrap();
        assert_eq!(
            (batch[0].q_len, batch[0].kv_len),
            (4, if id == 0 { 0 } else { 4 })
        );
        assert!(scheduler.done_iteration(1.0 + id as f64 * 2.0).is_empty());
        scheduler.schedule().unwrap();
        assert_eq!(scheduler.done_iteration(2.0 + id as f64 * 2.0).len(), 1);
    }
}

#[test]
fn external_occupancy_cannot_consume_reserved_lookahead() {
    let mut scheduler = stateful(16, 0, 4, false);
    scheduler.configure_external_kv().unwrap();
    scheduler.enqueue_sub_request(request(0, 4, 2));
    scheduler.schedule().unwrap();
    assert_eq!(scheduler.observe()["hbm_allocated_bytes"], 8);
    // Eight private bytes include the still-uncomputed lookahead reservation.
    assert!(scheduler.set_external_kv_bytes(12).is_err());
    assert!(scheduler.observe()["hbm_allocated_bytes"].as_u64().unwrap() <= 16);
}

#[test]
fn lookahead_arithmetic_does_not_overflow_computed_progress() {
    let mut scheduler = stateful(1 << 40, 0, u32::MAX, false);
    scheduler.enqueue_sub_request(request(0, 1, 3));
    scheduler.schedule().unwrap();
    scheduler.done_iteration(1.0);
    let next = scheduler.schedule().unwrap();
    assert_eq!((next[0].q_len, next[0].kv_len), (1, 1));
    // Allocation remains nonzero after adding lookahead to computed progress.
    assert!(scheduler.observe()["hbm_allocated_bytes"].as_u64().unwrap() >= 4);
}

#[test]
fn pending_external_restore_prevents_compute_until_ready() {
    let mut scheduler = stateful(16, 0, 0, false);
    scheduler.configure_external_kv().unwrap();
    scheduler.enqueue_sub_request(request(0, 4, 2));
    scheduler.set_external_blocked(0, true);
    assert!(scheduler.schedule().unwrap().is_empty());
    scheduler.set_external_blocked(0, false);
    assert_eq!(scheduler.schedule().unwrap()[0].q_len, 4);
}

#[test]
fn incomplete_token_traces_cannot_bridge_unknown_prompt_content() {
    for (input_ids, output_ids) in [
        (vec![1, 2, 3, 4], vec![5, 6, 7, 8]),
        ((1..=8).collect(), vec![9, 10]),
    ] {
        let request: Request = serde_json::from_value(json!({
            "request_id": 0, "arrival_time": 0.0, "initial": [0],
            "sub_requests": [{"kind": "llm", "input_tokens": 8,
                "output_tokens": 4, "next": [], "input_token_ids": input_ids,
                "output_token_ids": output_ids}],
        }))
        .unwrap();
        assert!(request.validate().is_err());
    }
}

#[test]
fn external_progress_events_follow_completed_block_boundaries() {
    let mut scheduler = stateful(64, 0, 0, false);
    scheduler.configure_external_kv().unwrap();
    scheduler.enqueue_sub_request(request(0, 4, 9));
    let mut progress = Vec::new();
    for now in 1..=9 {
        scheduler.schedule().unwrap();
        scheduler.done_iteration(f64::from(now));
        progress.extend(scheduler.take_kv_progress());
    }
    assert_eq!(progress, [(0, 4), (0, 8), (0, 12)]);
}

#[test]
fn external_reclaim_retries_aggregate_batch_before_preempting() {
    let mut scheduler = stateful(24, 0, 0, false);
    scheduler.configure_external_kv().unwrap();
    scheduler.set_external_kv_bytes(16).unwrap();
    scheduler.set_external_kv_reclaimable_bytes(16);
    for id in 0..3 {
        scheduler.enqueue_sub_request(request(id, 4, 2));
    }
    assert!(scheduler.schedule().unwrap().is_empty());
    assert!(scheduler.external_reclaim_pending());
    assert_eq!(scheduler.external_kv_requirement().unwrap().1, 12);
    assert!(scheduler.take_external_preemptions().is_empty());
    assert!(scheduler.take_kv_progress().is_empty());
    assert_eq!(scheduler.observe()["generation_tokens_total"], 0);
    scheduler.cancel_iteration();
    scheduler.set_external_kv_bytes(12).unwrap();
    scheduler.set_external_kv_reclaimable_bytes(12);
    assert_eq!(scheduler.schedule().unwrap().len(), 3);
    assert!(!scheduler.external_reclaim_pending());
    assert_eq!(scheduler.observe()["hbm_allocated_bytes"], 24);
}

#[test]
fn external_wait_phases_are_disjoint_and_do_not_run_blocked_calls() {
    let mut scheduler = stateful(32, 0, 0, false);
    scheduler.configure_external_kv().unwrap();
    scheduler.enqueue_sub_request(request(0, 4, 2));
    scheduler.enqueue_sub_request(request(1, 4, 2));
    scheduler.set_external_admission_blocked(0, true);
    assert_eq!(scheduler.observe()["hbm_admission_waiting_requests"], 1);
    assert_eq!(scheduler.observe()["scheduler_waiting_requests"], 1);
    scheduler.set_external_blocked(0, true);
    assert_eq!(scheduler.observe()["hbm_admission_waiting_requests"], 0);
    assert_eq!(scheduler.observe()["kv_preparation_waiting_requests"], 1);
    assert_eq!(scheduler.observe()["scheduler_waiting_requests"], 1);
    scheduler.set_external_blocked(0, false);
    assert!(scheduler.schedule().unwrap().is_empty());
    scheduler.set_external_admission_blocked(0, false);
    assert_eq!(scheduler.schedule().unwrap().len(), 2);
    scheduler.done_iteration(1.0);
    assert_eq!(scheduler.take_kv_progress(), [(0, 4), (1, 4)]);
}

#[test]
fn trailing_tools_preserve_session_membership_in_native_results() {
    let request: Request = serde_json::from_value(json!({
        "request_id": 0, "arrival_time": 0.0, "initial": [0],
        "sub_requests": [
            {"kind": "llm", "input_tokens": 4, "output_tokens": 1,
             "session_id": "source-session", "next": [1]},
            {"kind": "tool_call", "input_tokens": 0, "output_tokens": 0,
             "duration": 3.0, "session_id": "source-session", "next": []}
        ]
    }))
    .unwrap();
    let single = common::single();
    let mut runner = single["model_runner"].clone();
    runner["kind"] = json!("moe_ep");
    runner.as_object_mut().unwrap().remove("tp");
    runner["ep"] = json!(2);
    let ep = json!({"kind": "moe_ep_multi_instance", "model": common::model(),
        "schedulers": [single["scheduler"], single["scheduler"]], "model_runner": runner});
    for config in [single, ep] {
        let (results, end) = common::run_system(config, std::slice::from_ref(&request)).unwrap();
        common::close(end, 4.0);
        let result = serde_json::to_value(&results[0]).unwrap();
        assert_eq!(result["sub_requests"][0]["session_id"], "source-session");
        assert_eq!(result["sub_requests"][1]["session_id"], "source-session");
    }
    let mut invalid = request;
    if let simthesizer_core::types::SubRequest::ToolCall { session_id, .. } =
        &mut invalid.sub_requests[1]
    {
        *session_id = Some(String::new());
    }
    assert!(invalid.validate().is_err());
}
