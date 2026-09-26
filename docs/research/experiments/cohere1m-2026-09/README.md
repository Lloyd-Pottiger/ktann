# Cohere1M experiment snapshot

This directory preserves research evidence, not a production feature. See the
[baseline/cache report](../../ann-baseline-2026-09-26.md) and
[rerank report](../../ann-rerank-2026-09-27.md) for conclusions and limitations.
No source, default configuration or CI workflow is changed by this documentation PR.

- `results.json`: construction/topology, two baseline curves, stage attribution,
  four cache-capacity runs and the rerank comparison. Values are extracted from
  successful reports; diagnostic QPS is deliberately not exported as performance evidence.
- `provenance.json`: source/executable/report/log hashes and run eligibility, with
  local paths and host process inventories omitted. Full reports, logs, frozen
  executables, original per-phase patches and the database remain in the author's
  ignored `.benchmark-data/results/ann-baseline-2026-09-26/` directory. These hashes
  identify local evidence; they do not make unavailable raw files downloadable.
- The three protocol files record each experiment's controls and acceptance rules.
- `experiment.patch`: final isolated harness and collector snapshot, including the
  previously untracked `src/quality_trace.rs`. It applies to revision
  `3a62957c458a87b5b708af58ea1975c679e54ac2`. Earlier baseline/cache runs used their
  recorded intermediate binaries; this final patch replays the methodology, not
  their byte-identical builds. Its component provenance is retained separately.

## Replay in a disposable worktree

Requirements: Rust 1.85 or newer with the repository's build dependencies, enough
space for Cohere1M and a persistent RocksDB fixture, and the dataset cache described
in [the benchmark README](../../../../benchmarks/README.md). Run builds and
benchmarks sequentially on an otherwise idle host. The diagnostics use a single
process-global collector and must run with the harness's serial diagnostic mode.
Never use diagnostic timings as search performance measurements.

From a checkout containing this document:

```sh
bundle="$PWD/docs/research/experiments/cohere1m-2026-09"
replay="$PWD/../ktann-ann-replay"
git worktree add --detach "$replay" 3a62957c458a87b5b708af58ea1975c679e54ac2
cd "$replay"
git apply --check "$bundle/experiment.patch"
git apply "$bundle/experiment.patch"

cargo fmt --all -- --check
cargo test -p ktann-benchmarks --lib
cargo clippy -p ktann-benchmarks --all-targets --features quality-diagnostics -- -D warnings

artifacts="$replay/.benchmark-data/results/replay"
mkdir -p "$artifacts"
export CARGO_TARGET_DIR="$replay/target"
cargo build --release -p ktann-benchmarks --bin ktann-bench
cp "$CARGO_TARGET_DIR/release/ktann-bench" "$artifacts/timing-bin"
cargo build --release -p ktann-benchmarks --bin ktann-bench --features quality-diagnostics
cp "$CARGO_TARGET_DIR/release/ktann-bench" "$artifacts/diagnostic-bin"
```

Set `KTANN_BENCH_DATASET_CACHE` to the existing VectorDBBench dataset directory.
Use a fresh shell or clear prior `KTANN_DIAG_*` experiment settings before setup:

```sh
: "${KTANN_BENCH_DATASET_CACHE:?set the dataset cache path}"
unset KTANN_DIAG_REUSE KTANN_DIAG_TRACE KTANN_DIAG_RERANK
unset KTANN_DIAG_CONCURRENCY KTANN_DIAG_WARMUP KTANN_DIAG_OPERATIONS
export KTANN_DIAG_DATABASE="$artifacts/cohere1m-db"
export KTANN_DIAG_RECEIPT="$artifacts/cohere1m-receipt.json"
export KTANN_DIAG_CACHE_MIB=512
export KTANN_DIAG_BEAMS=64,96,128,192,256,384

run_timing() {
  "$artifacts/timing-bin" run --backend rocksdb --profile large \
    --scenario quality-cohere-1m --worker-threads 8 --output "$artifacts/$1.json" \
    > "$artifacts/$1.log" 2>&1
}

# The database path must not exist. Keep the receipt with its database.
run_timing build
export KTANN_DIAG_REUSE=1
run_timing baseline-1
export KTANN_DIAG_BEAMS=96,192,128,256,384,64
run_timing baseline-2
```

The build's search timings are ineligible: only subsequent reuse runs verify the
logical database digest both before and after search. The digest must remain
unchanged across reuse runs. A newly built tree can differ from the published
fixture because import and maintenance run concurrently; compare all settings
against your own fixed tree and do not expect its hash or recall to equal ours.

For the capacity screen, set beams to `64,384`, leave rerank unset, and call
`run_timing` with unique names at cache budgets `512,1024,1024,512` in that order.
For the rerank screen, set beam `384` and cache `1024`, then set
`KTANN_DIAG_RERANK` to `150,125,100,100,125,150` in that order, with unique report
names. Do not overwrite an earlier run. All comparisons use the same frozen
normal executable and fixed database.

For stage attribution, use the diagnostic executable, with `KTANN_DIAG_TRACE=1`,
`KTANN_DIAG_REUSE=1` and the desired beam/cache/rerank settings. It selects serial
execution, no warmup and 1,000 distinct queries. Run each rerank budget separately;
compare `ATTRIBUTION` lines in their deterministic query order. Each line reports
truth-hit counts at visited entries, local interval, local cap, global interval,
global cap and exact output, and verifies collector-disabled/enabled parity.

```sh
KTANN_DIAG_TRACE=1 "$artifacts/diagnostic-bin" run \
  --backend rocksdb --profile large --scenario quality-cohere-1m \
  --worker-threads 8 --output "$artifacts/attribution.json" \
  > "$artifacts/attribution.log" 2>&1
```

The final patch includes cache capacity and rerank knobs and the single-beam
frontier validation fix. They exist solely for controlled experiments. They are
not stable configuration APIs or recommended production instrumentation.
