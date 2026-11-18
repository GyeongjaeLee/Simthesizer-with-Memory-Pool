mod common;

use common::{close, run_system};
use serde_json::{json, Value};
use simthesizer_core::types::{ParallelStrategy, Request, RequestResult};
use simthesizer_core::{Engine, Graph, Model, System};

/// Decode (requests, Σ kv) and balanced prefill (requests, Σ q, Σ kv) in ms,
/// so prefill attention work Σ q(q/2 + kv) is 8, 40, 32 and 96.
fn table(tp: u32, ep: u32) -> std::path::PathBuf {
    static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let path = std::env::temp_dir().join(format!(
        "forward-pass-{}-{}.json",
        std::process::id(),
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    ));
    let table = json!({
        "format_version": 1, "latency_unit": "ms",
        "deployment": {"tp": tp, "ep": ep, "system": "test"},
        "source": {"url": "test"},
        "decode": [[1, 0, 1.0], [1, 100, 2.0], [4, 0, 2.0], [4, 400, 4.0]],
        "prefill": [[1, 4, 0, 10.0], [1, 4, 8, 14.0], [1, 8, 0, 20.0], [1, 8, 8, 28.0]],
    });
    std::fs::write(&path, table.to_string()).unwrap();
    path
}

fn minimax() -> Value {
    json!({"kind":"minimax_m2", "hidden_size":3072, "intermediate_size":1536,
        "num_attention_heads":48, "num_hidden_layers":62, "num_key_value_heads":8,
        "num_local_experts":256, "num_experts_per_tok":8, "vocab_size":200064,
        "head_dim":128, "rotary_dim":64, "kv_dtype":"fp8"})
}

fn system(table_ep: u32) -> Value {
    json!({"kind":"single_instance", "model":minimax(),
        "scheduler":{"kind":"chunked_prefill", "block_size":16, "max_batch":2,
            "max_num_batched_tokens":8, "device_mem":200_000_000_000u64,
            "device_mem_utilization":1.0, "enable_prefix_caching":false},
        "model_runner":{"kind":"forward_pass", "tp":4, "table":table(4, table_ep)}})
}

fn llm(id: u64, arrival: f64, input: u32, output: u32) -> Request {
    serde_json::from_value(
        json!({"request_id":id, "arrival_time":arrival, "initial":[0],
        "sub_requests":[{"kind":"llm", "input_tokens":input, "output_tokens":output, "next":[]}]}),
    )
    .unwrap()
}

fn latencies(results: &[RequestResult], id: u64) -> Vec<f64> {
    let row = serde_json::to_value(results.iter().find(|r| r.request_id == id).unwrap()).unwrap();
    serde_json::from_value(row["sub_requests"][0]["token_latency"].clone()).unwrap()
}

#[test]
fn grid_points_and_interpolation_price_whole_iterations() {
    let (results, _) = run_system(system(1), &[llm(0, 0.0, 4, 3)]).unwrap();
    let l = latencies(&results, 0);
    // Prefill T=4, A=8 is a grid point; decodes interpolate Σ kv = 4 and 5.
    close(l[0], 10e-3);
    close(l[1], 1.04e-3);
    close(l[2], 1.05e-3);
}

#[test]
fn mixed_batches_add_decode_tokens_and_their_kv_reads_to_prefill_work() {
    let (results, _) = run_system(system(1), &[llm(0, 0.0, 4, 2), llm(1, 0.005, 4, 1)]).unwrap();
    // T = 4 + 1, A = 4·2 + 1·(0.5 + 4) = 12.5.
    // T=4 row: 10 + 4·(12.5 − 8)/32; T=8 row clamps to 20; T=5 interpolates.
    let at4 = 10.0 + 4.0 * 4.5 / 32.0;
    let mixed = (at4 + (20.0 - at4) / 4.0) * 1e-3;
    close(latencies(&results, 0)[1], mixed);
    close(latencies(&results, 1)[0], 10e-3 - 0.005 + mixed);
}

#[test]
fn costs_above_the_measured_range_extrapolate_without_decreasing() {
    let mut system: Box<dyn System> = serde_json::from_value(system(1)).unwrap();
    system.init().unwrap();
    let mut graph = Graph::new();
    system
        .add_request_arrival(llm(0, 0.0, 8, 150), &mut graph, None, None)
        .unwrap();
    let mut lookups = Value::Null;
    let results = Engine::new(graph, system)
        .run_with_observer(|_, system| {
            lookups = system.observe()["model_runner"]["lookups"].clone();
            Ok(())
        })
        .unwrap();
    let l = latencies(&results, 0);
    // Decode beyond Σ kv = 100 keeps the last slope: 1 + kv / 100 ms.
    close(l[149], (1.0 + 156.0 / 100.0) * 1e-3);
    assert!(l.windows(2).skip(1).all(|w| w[1] >= w[0]));
    assert_eq!(lookups["decode"], 149);
    assert_eq!(lookups["extrapolated"], 149 - 93);
}

#[test]
fn table_parallelism_must_match_the_runner() {
    let mut system: Box<dyn System> = serde_json::from_value(system(4)).unwrap();
    let error = system.init().unwrap_err().to_string();
    assert!(error.contains("TP4/EP4"), "{error}");
}

#[test]
fn minimax_memory_matches_checkpoint_tensors() {
    let build = |tp| {
        let mut model: Box<dyn Model> = serde_json::from_value(minimax()).unwrap();
        model.init(ParallelStrategy { tp, pp: 1, ep: 1 }).unwrap();
        model
    };
    // HF safetensors: 227,410,968,576 FP8 and 1,230,021,632 BF16 elements;
    // FP32 router/bias 48,774,656 plus 13,880,064 FP32 128x128 block scales.
    let checkpoint = 227_410_968_576 + 2 * 1_230_021_632 + 4 * (48_774_656 + 13_880_064);
    assert_eq!(build(1).weight_bytes_per_device(), checkpoint);
    // 2 local KV heads x 128 x K/V x 62 layers, FP8.
    assert_eq!(build(4).kv_bytes_per_token_per_device(), 31_744);
}
