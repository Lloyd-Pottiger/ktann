//! Streaming input capture and real core Bulk Build for the benchmark bridge.
use ktann::api::{BulkWorkerOptions, Error, ErrorKind, Index, IndexConfig, Record, Result};
use ktann::bulk::{InputSnapshot, InputSnapshotWriter};
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
    writer: InputSnapshotWriter,
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
        let writer = InputSnapshotWriter::new(&root.join("source"), config.clone(), DISK_LIMIT)?;
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
    fn finish(self) -> Result<InputSnapshot> {
        self.writer.seal()
    }
}

pub(super) async fn build<B: Backend>(
    runtime: &Runtime<B>,
    staging: InputCapture,
) -> Result<(Index<B>, serde_json::Value)> {
    let workspace = staging.root.join("workspace");
    let config = staging.config.clone();
    let options = ConstructionOptions {
        min_partition_entries: config.min_partition_entries(),
        max_partition_entries: config.max_partition_entries(),
        sample_items: 256,
        memory_bytes: 256 * 1024 * 1024,
        scratch_bytes: DISK_LIMIT,
    };
    let start = Instant::now();
    let source = tokio::task::spawn_blocking(move || staging.finish())
        .await
        .map_err(|e| Error::with_source(ErrorKind::Other, e))??;
    let snapshot_seconds = start.elapsed().as_secs_f64();
    let job = runtime
        .start_bulk_build("vdbbench", &source, options)
        .await?;
    let worker = BulkWorkerOptions::new(workspace);
    let start = Instant::now();
    job.run_worker(worker.clone()).await?;
    let prepare_load_seconds = start.elapsed().as_secs_f64();
    let start = Instant::now();
    let index = job.publish().await?;
    let publish_seconds = start.elapsed().as_secs_f64();
    Ok((
        index,
        json!({
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
