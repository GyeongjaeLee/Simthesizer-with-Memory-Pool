//! Whole-iteration costs from measured forward-pass tables.

use std::any::Any;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use anyhow::{anyhow, ensure, Context, Result};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::engine::Graph;
use crate::model::Model;
use crate::model_runner::ModelRunner;
use crate::module::{LogicalHandler, Module};
use crate::types::{NodeID, ParallelStrategy, RequestState};

/// Prices one iteration of a TP group from a forward-pass table.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ForwardPassRunner {
    tp: u32,
    table: PathBuf,
    /// Hypothetical speedup over the measured platform, for projections to
    /// unmeasured hardware; 1 keeps the measurement.
    #[serde(default = "unit")]
    compute_speedup: f64,
    #[serde(skip)]
    loaded: Option<Table>,
    #[serde(skip)]
    lookups: Lookups,
}

crate::register_model_runner!(ForwardPassRunner, "forward_pass");

fn unit() -> f64 {
    1.0
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TableFile {
    format_version: u32,
    latency_unit: String,
    deployment: Deployment,
    source: Value,
    /// `[requests, total cached KV tokens, latency]`, one new token per request.
    decode: Vec<[f64; 3]>,
    /// `[requests, total new tokens, total cached KV tokens, latency]`, balanced requests.
    prefill: Vec<[f64; 4]>,
}

#[derive(Deserialize)]
struct Deployment {
    tp: u32,
    ep: u32,
    #[serde(flatten)]
    _identity: BTreeMap<String, Value>,
}

#[derive(Debug, Default, Serialize)]
struct Lookups {
    decode: u64,
    prefill: u64,
    /// Prefill lookups that include one-token requests.
    mixed: u64,
    /// Lookups beyond a measured axis range.
    extrapolated: u64,
}

impl Module for ForwardPassRunner {
    fn resolve_path(&mut self, base: &Path) -> Result<()> {
        if self.table.is_relative() {
            self.table = base.join(&self.table);
        }
        Ok(())
    }

    fn observe(&self) -> Value {
        serde_json::json!({
            "tp": self.tp,
            "table": self.table,
            "compute_speedup": self.compute_speedup,
            "lookups": self.lookups,
        })
    }
}

impl LogicalHandler for ForwardPassRunner {
    fn handle(&mut self, payload: Box<dyn Any>, _: f64, _: NodeID, _: &mut Graph) -> Result<f64> {
        payload
            .downcast::<f64>()
            .map(|latency| *latency)
            .map_err(|_| anyhow!("forward_pass received an unexpected event"))
    }
}

impl ModelRunner for ForwardPassRunner {
    fn parallel_strategy(&self) -> ParallelStrategy {
        ParallelStrategy {
            tp: self.tp,
            pp: 1,
            ep: 1,
        }
    }

    fn init(&mut self, _: &dyn Model) -> Result<()> {
        ensure!(self.tp > 0, "forward_pass requires a positive TP size");
        ensure!(
            self.compute_speedup.is_finite() && self.compute_speedup > 0.0,
            "compute_speedup must be positive"
        );
        let file: TableFile = serde_json::from_slice(
            &std::fs::read(&self.table)
                .with_context(|| format!("opening {}", self.table.display()))?,
        )
        .with_context(|| format!("parsing {}", self.table.display()))?;
        ensure!(
            file.format_version == 1,
            "unsupported forward-pass table version"
        );
        ensure!(
            file.deployment.tp == self.tp && file.deployment.ep == 1,
            "forward-pass table measures TP{}/EP{}, runner is TP{}/EP1",
            file.deployment.tp,
            file.deployment.ep,
            self.tp
        );
        ensure!(!file.source.is_null(), "forward-pass table lacks a source");
        self.loaded = Some(Table::new(&file)?);
        self.lookups = Lookups::default();
        Ok(())
    }

    fn add_iteration(
        &mut self,
        _: &dyn Model,
        batch: &[RequestState],
        current: NodeID,
        graph: &mut Graph,
    ) -> Result<NodeID> {
        let table = self
            .loaded
            .as_ref()
            .ok_or_else(|| anyhow!("forward_pass runner is not initialized"))?;
        let (latency, inside) = table.iteration(batch, &mut self.lookups)?;
        self.lookups.extrapolated += u64::from(!inside);
        let owner = self as *mut Self as *mut dyn LogicalHandler;
        let node = graph.add_logical_node(owner, latency / self.compute_speedup);
        graph.add_edge(current, node)?;
        Ok(node)
    }
}

/// Linear between samples; above the last sample, the last slope continues
/// unless it would decrease cost.
fn interpolate(x: f64, (x0, y0): (f64, f64), (x1, y1): (f64, f64)) -> f64 {
    if x <= x1 {
        y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    } else {
        y1 + (y1 - y0).max(0.0) * (x - x1) / (x1 - x0)
    }
}

/// Piecewise-linear samples, clamped below. Above the last sample, cost grows
/// at the least-squares slope of the samples in the upper half of the range
/// (at least the last two), never negative: neighboring samples from different batch compositions differ by
/// noise over tiny coordinate gaps, so their slope would dominate.
struct Curve {
    points: Vec<(f64, f64)>,
    tail: f64,
}

impl Curve {
    fn new(mut points: Vec<(f64, f64)>) -> Result<Self> {
        ensure!(!points.is_empty(), "empty forward-pass curve");
        points.sort_by(|a, b| a.0.total_cmp(&b.0));
        // Equal coordinates from different measurements average.
        let mut merged: Vec<(f64, f64, f64)> = Vec::new();
        for (x, y) in points {
            ensure!(
                x.is_finite() && y.is_finite() && y > 0.0,
                "invalid forward-pass sample"
            );
            match merged.last_mut() {
                Some(last) if last.0 == x => {
                    last.1 += y;
                    last.2 += 1.0;
                }
                _ => merged.push((x, y, 1.0)),
            }
        }
        let points: Vec<(f64, f64)> = merged.into_iter().map(|(x, y, n)| (x, y / n)).collect();
        let last = points[points.len() - 1].0;
        let start = points
            .partition_point(|p| p.0 < last / 2.0)
            .min(points.len().saturating_sub(2));
        let upper = &points[start..];
        let n = upper.len() as f64;
        let (mx, my) = upper
            .iter()
            .fold((0.0, 0.0), |(x, y), p| (x + p.0 / n, y + p.1 / n));
        let spread: f64 = upper.iter().map(|p| (p.0 - mx).powi(2)).sum();
        let tail = if spread > 0.0 {
            upper.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum::<f64>() / spread
        } else {
            0.0
        };
        Ok(Self {
            points,
            tail: tail.max(0.0),
        })
    }

    fn at(&self, x: f64) -> (f64, bool) {
        let points = &self.points;
        let (first, last) = (points[0], points[points.len() - 1]);
        if x <= first.0 || points.len() == 1 {
            return (first.1, x >= first.0 && x <= last.0);
        }
        if x > last.0 {
            return (last.1 + self.tail * (x - last.0), false);
        }
        let i = points.partition_point(|p| p.0 < x);
        (interpolate(x, points[i - 1], points[i]), true)
    }
}

/// Curves along an inner axis, interpolated across an outer axis.
struct Surface(Vec<(f64, Curve)>);

impl Surface {
    fn new(rows: impl Iterator<Item = (f64, f64, f64)>) -> Result<Self> {
        let mut grouped: BTreeMap<u64, Vec<(f64, f64)>> = BTreeMap::new();
        for (outer, inner, latency) in rows {
            ensure!(
                outer >= 1.0 && outer.fract() == 0.0,
                "forward-pass outer coordinate must be a positive integer"
            );
            grouped
                .entry(outer as u64)
                .or_default()
                .push((inner, latency));
        }
        ensure!(!grouped.is_empty(), "empty forward-pass table phase");
        Ok(Self(
            grouped
                .into_iter()
                .map(|(outer, points)| Ok((outer as f64, Curve::new(points)?)))
                .collect::<Result<_>>()?,
        ))
    }

    fn at(&self, outer: f64, inner: f64) -> (f64, bool) {
        let rows = &self.0;
        if outer <= rows[0].0 || rows.len() == 1 {
            let (y, inside) = rows[0].1.at(inner);
            return (
                y,
                inside && outer >= rows[0].0 && outer <= rows[rows.len() - 1].0,
            );
        }
        let i = rows.partition_point(|r| r.0 < outer).min(rows.len() - 1);
        let ((a, lo), (b, hi)) = (&rows[i - 1], &rows[i]);
        let ((y0, in0), (y1, in1)) = (lo.at(inner), hi.at(inner));
        (
            interpolate(outer, (*a, y0), (*b, y1)),
            in0 && in1 && outer <= *b,
        )
    }
}

/// Decode: (requests, Σ kv). Otherwise: (Σ q, Σ q·(q/2 + kv)), whose
/// attention coordinate matches token-wise layers and causal attention work
/// across request counts, mixed decode tokens and unequal lengths.
struct Table {
    decode: Surface,
    prefill: Surface,
    seconds_per_unit: f64,
}

impl Table {
    fn new(file: &TableFile) -> Result<Self> {
        let seconds_per_unit = match file.latency_unit.as_str() {
            "ms" => 1e-3,
            "us" => 1e-6,
            "s" => 1.0,
            unit => return Err(anyhow!("unsupported latency unit {unit}")),
        };
        let decode = Surface::new(file.decode.iter().map(|&[b, k, y]| (b, k, y)))?;
        let prefill = Surface::new(
            file.prefill
                .iter()
                .map(|&[b, t, k, y]| (t, t * (t / (2.0 * b) + k / b), y)),
        )?;
        Ok(Self {
            decode,
            prefill,
            seconds_per_unit,
        })
    }

    fn iteration(&self, batch: &[RequestState], lookups: &mut Lookups) -> Result<(f64, bool)> {
        ensure!(!batch.is_empty(), "forward-pass iteration has no requests");
        let (latency, inside) = if batch.iter().all(|r| r.q_len == 1) {
            lookups.decode += 1;
            let kv = batch.iter().map(|r| f64::from(r.kv_len)).sum();
            self.decode.at(batch.len() as f64, kv)
        } else {
            lookups.prefill += 1;
            lookups.mixed += u64::from(batch.iter().any(|r| r.q_len == 1));
            let tokens = batch.iter().map(|r| f64::from(r.q_len)).sum();
            let attention = batch
                .iter()
                .map(|r| {
                    let q = f64::from(r.q_len);
                    q * (q / 2.0 + f64::from(r.kv_len))
                })
                .sum();
            self.prefill.at(tokens, attention)
        };
        Ok((latency * self.seconds_per_unit, inside))
    }
}
