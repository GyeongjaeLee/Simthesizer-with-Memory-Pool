# FreeInference load inputs

Python 3.14 supplies the zstd reader. Sources and generated workloads stay under
ignored `output/`; unit tests require no downloads.

```sh
python3 scripts/freeinference.py fetch --all-weeks --out output/freeinference/source
python3 scripts/freeinference.py index \
  --source output/freeinference/source --out output/freeinference/index
python3 scripts/freeinference.py generate \
  --index output/freeinference/index --out output/freeinference/poisson-2 \
  --arrivals poisson --sessions-per-second 2 --seed 42 --limit 1000 --invalid exclude
```

`index` and `generate` require fresh output directories. Sources stay compressed.
Manifests record source hashes, selected sessions, call mappings, exclusions and
arrival parameters. Use `--help` for selection controls.

Rate means top-level sessions/s. `original` preserves relative source arrivals;
`regular` uses `i/R`; `poisson` samples exponential gaps. Selection keeps whole
sessions without clipping or duplication. Only root arrivals change.

The default `provider-count` mode supplies compute demand without prefix identity.
`--token-mode proxy-prefix --target-model MODEL --kv-layout LAYOUT
--sharing-scope SCOPE` opts into source proxy blocks, which are not model token IDs. The native scenario selects operation
costs; gateway latency does not become simulated service. See
[workload semantics](../../architectures/workload.md) for mapping and limitations.

Harvard MaDSys, [FreeInference Agentic Trace](https://huggingface.co/datasets/harvardMadsys/freeinference_agentic_trace/tree/bb26cf72c9c20f2b05f765d88548156a74d3c253),
revision `bb26cf72c9c20f2b05f765d88548156a74d3c253`,
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Conversion, selection and arrivals are derived artifacts; compressed sources are unchanged.
