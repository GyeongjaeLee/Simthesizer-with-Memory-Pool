# H200 MiniMax-M2.7 TP4 forward-pass validation

[`examples/data/h200/minimax_m2_7-tp4.json`](../../examples/data/h200/minimax_m2_7-tp4.json)
normalizes the [AISim FPM dataset](https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset)
self-benchmark table for MiniMax-M2.7 (GQA 48/8 heads, 256 experts top-8,
block-FP8 weights, FP8 KV) on four H200 GPUs with vLLM 0.25.1 pure TP4.
The same deployment's live engine stream records each iteration's scheduled
batch and wall time under injected burst, decode and mixed windows.

`iterations.py` prices every observed batch with an independent
re-implementation of the `forward_pass` lookup. Zero-time heartbeats and the
first busy iteration after an empty step are excluded: their wall time includes
waiting. Rust equals this lookup on deterministic batches.

```sh
python3 experiments/fpm-validation/acquire.py
src=output/fpm-validation/source/data/MiniMaxAI--MiniMax-M2.7/h200-sxm/vllm/0.25.1/pure-tp4/fpm/fpm_forward_perf
uv run --with pyarrow scripts/import_fpm.py --parquet $src.parquet --metadata $src.metadata.json \
  --source-url https://huggingface.co/datasets/nvidia/aisimulate-fpm-dataset/resolve/dbfcdd9eb62ea733efbc3f0cbd78277f2960061a/data/MiniMaxAI--MiniMax-M2.7/h200-sxm/vllm/0.25.1/pure-tp4/fpm/fpm_forward_perf.parquet \
  --output output/fpm-validation/minimax_m2_7-tp4.json   # equals the bundled table
uv run --with pyarrow --with pandas experiments/fpm-validation/iterations.py --out output/fpm-validation/l1
```

| Iteration class | n | MAPE | Median APE | Bias |
|---|---:|---:|---:|---:|
| Decode | 140,904 | 4.6% | 3.0% | −4.4% |
| Prefill | 4,228 | 2.5% | 1.4% | −2.1% |
| Mixed | 3,409 | 11.9% | 10.4% | −9.2% |

Mixed rows are absent from the table, so mixed batches rely on the folded-token
rule and run about 9% fast. The live stream peaks at 1,024 running decodes and
2.35 M decode KV tokens, consistent with the scenario's derived 2.42 M-token
HBM capacity. B200 public data were examined but offer only disaggregated
MiniMax-M2.7 serving; its decode workers diverge with load.
