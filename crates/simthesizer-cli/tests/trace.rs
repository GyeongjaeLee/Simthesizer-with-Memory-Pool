use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use serde_json::{json, Value};

fn config() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/scenarios/1xrtx-pro-6000--llama3_1_8b--realistic/simthesizer.toml")
}

fn run(workload: &Path, result: &Path, trace: Option<&Path>) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_simthesizer"));
    command
        .arg("--config")
        .arg(config())
        .arg("--workload")
        .arg(workload)
        .arg("--output-jsonl")
        .arg(result);
    if let Some(trace) = trace {
        command.arg("--output-engine-trace").arg(trace);
    }
    command.output().unwrap()
}

#[test]
fn trace_on_off_results_match_and_existing_outputs_are_preserved() {
    let root = tempfile::tempdir().unwrap();
    let workload = root.path().join("workload.jsonl");
    let request = json!({"request_id":0,"arrival_time":1.0,"initial":[0],
        "sub_requests":[{"kind":"llm","input_tokens":8,"output_tokens":2,"next":[]}]});
    std::fs::write(&workload, format!("{request}\n")).unwrap();
    let off = root.path().join("off.jsonl");
    let on = root.path().join("on.jsonl");
    let trace = root.path().join("trace.jsonl");
    let result = run(&workload, &off, None);
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    let result = run(&workload, &on, Some(&trace));
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(std::fs::read(&off).unwrap(), std::fs::read(&on).unwrap());
    let bytes = std::fs::read(&trace).unwrap();
    let frames: Vec<Value> = String::from_utf8(bytes.clone())
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(frames[0]["time_s"], 0.0);
    assert_eq!(
        frames
            .last()
            .unwrap()
            .pointer("/state/scheduler/generation_tokens_total"),
        Some(&json!(2))
    );
    assert!(!run(&workload, &on, Some(&trace)).status.success());
    assert_eq!(std::fs::read(&trace).unwrap(), bytes);
}

#[test]
fn failed_simulation_does_not_publish_trace_and_alias_paths_are_rejected() {
    let root = tempfile::tempdir().unwrap();
    let workload = root.path().join("workload.jsonl");
    let request = json!({"request_id":0,"arrival_time":0.0,"initial":[0],
        "sub_requests":[{"kind":"llm","input_tokens":8,"output_tokens":1,"next":[]}]});
    std::fs::write(&workload, format!("{request}\n{request}\n")).unwrap();
    let trace = root.path().join("trace.jsonl");
    let result = root.path().join("results.jsonl");
    assert!(!run(&workload, &result, Some(&trace)).status.success());
    assert!(!trace.exists());
    std::fs::write(&workload, format!("{request}\n")).unwrap();
    let alias = root.path().join("sub/../trace.jsonl");
    assert!(!run(&workload, &alias, Some(&trace)).status.success());
    assert!(!trace.exists());
}

#[test]
fn malformed_request_semantics_fail_without_publishing_outputs() {
    for request in [
        json!({"request_id":0,"arrival_time":-1.0,"initial":[0],
            "sub_requests":[{"kind":"llm","input_tokens":8,"output_tokens":1,"next":[]}]}),
        json!({"request_id":0,"arrival_time":0.0,"initial":[],
            "sub_requests":[{"kind":"tool_call","input_tokens":0,"output_tokens":0,"next":[0]}]}),
        json!({"request_id":0,"arrival_time":0.0,"initial":[0],
            "sub_requests":[{"kind":"tool_call","input_tokens":0,"output_tokens":0,"duration":-1.0,"next":[]}]}),
    ] {
        let root = tempfile::tempdir().unwrap();
        let workload = root.path().join("workload.jsonl");
        let result = root.path().join("results.jsonl");
        let trace = root.path().join("trace.jsonl");
        std::fs::write(&workload, format!("{request}\n")).unwrap();
        assert!(!run(&workload, &result, Some(&trace)).status.success());
        assert!(!result.exists());
        assert!(!trace.exists());
    }
}

#[test]
fn trace_times_keep_the_first_frame_at_or_after_each_time() {
    let root = tempfile::tempdir().unwrap();
    let workload = root.path().join("workload.jsonl");
    let request = json!({"request_id":0,"arrival_time":1.0,"initial":[0],
        "sub_requests":[{"kind":"llm","input_tokens":8,"output_tokens":4,"next":[]}]});
    std::fs::write(&workload, format!("{request}\n")).unwrap();
    let frames = |path: &Path| -> Vec<Value> {
        std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    };
    let full = root.path().join("full.jsonl");
    assert!(run(&workload, &root.path().join("a.jsonl"), Some(&full))
        .status
        .success());
    let all = frames(&full);
    let at = all[2]["time_s"].as_f64().unwrap();
    let sparse = root.path().join("sparse.jsonl");
    let output = run_with(
        &workload,
        &root.path().join("b.jsonl"),
        &sparse,
        &["0.5", &at.to_string()],
    );
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let first_after = |t: f64| {
        all.iter()
            .find(|f| f["time_s"].as_f64().unwrap() >= t)
            .unwrap()
    };
    assert_eq!(
        frames(&sparse),
        vec![first_after(0.5).clone(), first_after(at).clone()]
    );
}

fn run_with(workload: &Path, result: &Path, trace: &Path, times: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_simthesizer"))
        .arg("--config")
        .arg(config())
        .arg("--workload")
        .arg(workload)
        .arg("--output-jsonl")
        .arg(result)
        .arg("--output-engine-trace")
        .arg(trace)
        .arg("--trace-times")
        .args(times)
        .output()
        .unwrap()
}

#[test]
fn saturation_stops_an_overloaded_run_but_not_a_light_one() {
    let root = tempfile::tempdir().unwrap();
    let run = |name: &str, count: u64, gap: f64| {
        let workload = root.path().join(format!("{name}.jsonl"));
        let rows: String = (0..count)
            .map(|id| {
                format!(
                    "{}\n",
                    json!({"request_id":id,"arrival_time":id as f64 * gap,"initial":[0],
                        "sub_requests":[{"kind":"llm","input_tokens":2000,"output_tokens":64,"next":[]}]})
                )
            })
            .collect();
        std::fs::write(&workload, rows).unwrap();
        let result = root.path().join(format!("{name}-results.jsonl"));
        let output = Command::new(env!("CARGO_BIN_EXE_simthesizer"))
            .arg("--config")
            .arg(config())
            .arg("--workload")
            .arg(&workload)
            .arg("--output-jsonl")
            .arg(&result)
            .args(["--saturation-wait", "0.25"])
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let status: Value = serde_json::from_slice(&output.stdout).unwrap();
        (status, result.exists())
    };
    let (status, written) = run("overloaded", 1000, 0.005);
    assert_eq!(status["status"], "saturated");
    assert!(status["mean_wait_s"].as_f64().unwrap() > 0.25);
    assert!(!written);
    let (status, written) = run("light", 10, 1.0);
    assert_eq!(status, json!({"status": "complete"}));
    assert!(written);
}
