use std::io::BufRead;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::Parser;
use serde::Deserialize;
use simthesizer_core::types::Request;
use simthesizer_core::{Engine, Graph, System};

mod saturation;
mod write_jsonl;
mod write_trace;

use crate::saturation::{Saturated, Saturation};
use crate::write_jsonl::write_results_jsonl;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    config: PathBuf,
    #[arg(long)]
    workload: PathBuf,
    #[arg(long)]
    output_jsonl: PathBuf,
    /// Component state at the engine clock, as JSONL.
    #[arg(long)]
    output_engine_trace: Option<PathBuf>,
    /// Record only the first state at or after each of these times (s).
    #[arg(long, num_args = 1.., requires = "output_engine_trace")]
    trace_times: Vec<f64>,
    /// Stop once the mean first-token wait (TTFT minus prefill) of calls
    /// completed in a window exceeds this many seconds; then no results are
    /// written and the status line reports saturation.
    #[arg(long)]
    saturation_wait: Option<f64>,
    /// Window length in simulated seconds.
    #[arg(long, default_value_t = 600.0)]
    saturation_period: f64,
    /// Consecutive windows over the wait that saturate the run.
    #[arg(long, default_value_t = 2)]
    saturation_windows: u32,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Config {
    system: Box<dyn System>,
}

fn main() -> Result<()> {
    let mut args = Args::parse();
    args.output_jsonl = output_path(&args.output_jsonl)?;
    args.output_engine_trace = args
        .output_engine_trace
        .as_deref()
        .map(output_path)
        .transpose()?;
    if let Some(path) = &args.output_engine_trace {
        anyhow::ensure!(
            !path.exists(),
            "trace output already exists: {}",
            path.display()
        );
        anyhow::ensure!(
            path != &args.output_jsonl,
            "trace and result paths must differ"
        );
    }
    let config_path = args.config;
    let workload_path = args.workload;
    let mut config = load_config(&config_path)?;
    let system_base_path = config_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    config.system.resolve_path(&system_base_path)?;
    config.system.init()?;
    let mut graph = Graph::new();
    load_workload(config.system.as_mut(), &mut graph, &workload_path)?;

    let mut saturation = args
        .saturation_wait
        .map(|wait| Saturation::new(args.saturation_period, wait, args.saturation_windows))
        .transpose()?;
    anyhow::ensure!(
        saturation.is_none() || config.system.first_token_waits().is_some(),
        "this system does not report first-token waits for saturation detection"
    );
    let mut times = args.trace_times.clone();
    anyhow::ensure!(
        times.iter().all(|t| t.is_finite() && *t >= 0.0),
        "trace times must be finite and nonnegative"
    );
    times.sort_by(f64::total_cmp);
    let mut trace = match &args.output_engine_trace {
        Some(path) => {
            let parent = path
                .parent()
                .filter(|p| !p.as_os_str().is_empty())
                .unwrap_or(Path::new("."));
            std::fs::create_dir_all(parent)?;
            let file = tempfile::NamedTempFile::new_in(parent)?;
            let writer = std::io::BufWriter::new(file.as_file().try_clone()?);
            Some((file, write_trace::TraceWriter::new(writer)))
        }
        None => None,
    };
    let mut next = 0;
    let outcome = Engine::new(graph, config.system).run_with_observer(|time, system| {
        if let Some((_, writer)) = &mut trace {
            if times.is_empty() {
                writer.observe(time, system.observe())?;
            } else if next < times.len() && time >= times[next] {
                while next < times.len() && time >= times[next] {
                    next += 1;
                }
                writer.observe(time, system.observe())?;
            }
        }
        if let Some(guard) = &mut saturation {
            if let Some(mean_wait) = guard.check(time, system.first_token_waits().unwrap()) {
                return Err(Saturated { time, mean_wait }.into());
            }
        }
        Ok(())
    });
    let status = match outcome {
        Ok(results) => {
            write_results_jsonl(&args.output_jsonl, &results)?;
            serde_json::json!({"status": "complete"})
        }
        Err(error) => {
            let saturated = error.downcast::<Saturated>()?;
            serde_json::json!({"status": "saturated", "time_s": saturated.time,
                "mean_wait_s": saturated.mean_wait})
        }
    };
    if let (Some((file, writer)), Some(path)) = (trace, &args.output_engine_trace) {
        writer.finish()?;
        file.persist_noclobber(path)?;
    }
    println!("{status}");
    Ok(())
}

fn output_path(path: &Path) -> Result<PathBuf> {
    let name = path.file_name().context("output path needs a filename")?;
    let parent = path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    std::fs::create_dir_all(parent)?;
    Ok(parent.canonicalize()?.join(name))
}

fn load_config(config_path: &Path) -> Result<Config> {
    let raw = std::fs::read_to_string(config_path)
        .with_context(|| format!("failed to read {}", config_path.display()))?;
    toml::from_str(&raw).with_context(|| format!("failed to parse {}", config_path.display()))
}

fn load_workload(system: &mut dyn System, graph: &mut Graph, workload_path: &Path) -> Result<()> {
    let file = std::fs::File::open(workload_path)
        .with_context(|| format!("failed to open {}", workload_path.display()))?;
    let reader = std::io::BufReader::new(file);

    for (line_index, line) in reader.lines().enumerate() {
        let line = line.with_context(|| {
            format!(
                "failed to read line {} from {}",
                line_index + 1,
                workload_path.display()
            )
        })?;
        if line.trim().is_empty() {
            continue;
        }

        let request: Request = serde_json::from_str(&line).with_context(|| {
            format!(
                "failed to parse workload row {} from {}",
                line_index + 1,
                workload_path.display()
            )
        })?;
        request.validate().with_context(|| {
            format!(
                "invalid workload row {} from {}",
                line_index + 1,
                workload_path.display()
            )
        })?;
        system.add_request_arrival(request, graph, None, None)?;
    }
    Ok(())
}
