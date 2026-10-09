//! Streaming input capture and real core Bulk Build for the benchmark bridge.
use ktann::api::{BulkWorkerOptions, Error, ErrorKind, Index, IndexConfig, Record, Result};
use ktann::bulk::{ForestOptions, PreparedInput, PreparedInputWriter};
use ktann::construction::ConstructionOptions;
use ktann::runtime::Runtime;
use ktann::storage::backend::Backend;
use serde_json::json;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

const DISK_LIMIT: u64 = 64 * 1024 * 1024 * 1024;

/// Caller-owned input is encoded and hashed on receipt; EOF only seals it.
pub(super) struct InputCapture {
    root: PathBuf,
    config: IndexConfig,
    writer: PreparedInputWriter,
}
fn io(error: std::io::Error) -> Error {
    Error::with_source(ErrorKind::Other, error)
}
impl InputCapture {
    pub(super) fn new(root: PathBuf, config: IndexConfig) -> Result<Self> {
        if !root.is_absolute() {
            return Err(Error::new(ErrorKind::InvalidArgument));
        }
        fs::create_dir(&root).map_err(io)?;
        fs::create_dir(root.join("workspace")).map_err(io)?;
        let worker = BulkWorkerOptions::new(root.join("workspace"));
        let writer = PreparedInputWriter::new(
            &root.join("source"),
            &root.join("preparation"),
            config.clone(),
            DISK_LIMIT,
            ForestOptions {
                tree: construction_options(&config),
                sort_memory_bytes: worker.sort_memory_bytes,
                sort_scratch_bytes: worker.sort_scratch_bytes,
            },
        )?;
        Ok(Self {
            root,
            config,
            writer,
        })
    }
    pub(super) fn append(mut self, records: Vec<Record>) -> Result<Self> {
        self.writer = self.writer.append(records.into_iter().map(Ok))?;
        Ok(self)
    }
    fn finish(self) -> Result<PreparedInput> {
        self.writer.seal()
    }
}

pub(super) async fn build<B: Backend>(
    runtime: &Runtime<B>,
    staging: InputCapture,
) -> Result<(Index<B>, serde_json::Value)> {
    let workspace = staging.root.join("workspace");
    let config = staging.config.clone();
    let options = construction_options(&config);
    let start = Instant::now();
    let prepared = tokio::task::spawn_blocking(move || staging.finish())
        .await
        .map_err(|e| Error::with_source(ErrorKind::Other, e))??;
    let snapshot_seconds = start.elapsed().as_secs_f64();
    let source = prepared.source().clone();
    let job = runtime
        .start_bulk_build("vdbbench", &source, options)
        .await?;
    let worker = BulkWorkerOptions::new(workspace);
    let start = Instant::now();
    let stages = job
        .run_worker_with_prepared_input(worker.clone(), prepared, Default::default())
        .await?;
    let prepare_load_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let index = job.publish().await?;
    let publish_seconds = start.elapsed().as_secs_f64();
    Ok((
        index,
        json!({
            "sort_written_bytes": stages.forest_report.map(|r| r.sort_written_bytes),
            "tree_written_bytes": stages.forest_report.map(|r| r.tree_written_bytes),
            "peak_sort_scratch_bytes": stages.forest_report.map(|r| r.peak_sort_scratch_bytes),
            "peak_tree_scratch_bytes": stages.forest_report.map(|r| r.peak_tree_scratch_bytes),
            "forest_seconds": stages.forest.as_secs_f64(),
            "serving_seconds": stages.serving.as_secs_f64(),
            "load_seconds": stages.load.as_secs_f64(),
            "snapshot_seconds": snapshot_seconds, "prepare_load_seconds": prepare_load_seconds,
            "validate_publish_cleanup_seconds": publish_seconds,
            "source_sha256": source.manifest().sha256(), "source_bytes": source.manifest().bytes(),
            "rotation_seed": job.index_manifest().rotation_seed(),
        "sample_items": options.sample_items, "construction_memory_bytes": options.memory_bytes,
            "construction_scratch_bytes": options.scratch_bytes,
            "sort_memory_bytes": worker.sort_memory_bytes, "sort_scratch_bytes": worker.sort_scratch_bytes,
            "serving_memory_bytes": worker.serving_memory_bytes, "serving_scratch_bytes": worker.serving_scratch_bytes,
            "max_artifact_bytes": worker.max_artifact_bytes,
            "load_max_mutations": worker.load.max_mutations, "load_max_bytes": worker.load.max_bytes
        }),
    ))
}

fn construction_options(config: &IndexConfig) -> ConstructionOptions {
    ConstructionOptions {
        min_partition_entries: config.min_partition_entries(),
        max_partition_entries: config.max_partition_entries(),
        sample_items: 256,
        memory_bytes: 256 * 1024 * 1024,
        scratch_bytes: DISK_LIMIT,
    }
}
