//! Streaming input capture and real core Bulk Build for the benchmark bridge.
use ktann::api::{
    BulkWorkerOptions, Error, ErrorKind, Index, IndexConfig, OperationOptions, Record, Result,
};
use ktann::bulk::{ForestOptions, PreparedInputWriter};
use ktann::construction::ConstructionOptions;
use ktann::runtime::Runtime;
use ktann::storage::backend::Backend;
use serde_json::json;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

const DISK_LIMIT: u64 = 64 * 1024 * 1024 * 1024;

/// Bridge-owned input is encoded and hashed on receipt; EOF only seals it.
pub(super) struct InputCapture {
    construction: ConstructionOptions,
    refinement_rounds: Option<u8>,
    worker: BulkWorkerOptions,
    writer: PreparedInputWriter,
}

fn io(error: std::io::Error) -> Error {
    Error::with_source(ErrorKind::Other, error)
}

impl InputCapture {
    pub(super) fn new(
        root: PathBuf,
        config: IndexConfig,
        experiment: Option<crate::routing::Experiment>,
        rounds: u8,
    ) -> Result<Self> {
        if !root.is_absolute() {
            return Err(Error::new(ErrorKind::InvalidArgument));
        }
        fs::create_dir(&root).map_err(io)?;
        fs::create_dir(root.join("workspace")).map_err(io)?;
        let mut worker = BulkWorkerOptions::new(root.join("workspace"));
        let mut construction = ConstructionOptions {
            sample_items: 1024,
            memory_bytes: 256 * 1024 * 1024,
            scratch_bytes: DISK_LIMIT,
        };
        let refinement_rounds = experiment.as_ref().map(|_| rounds);
        if let Some(experiment) = experiment {
            construction = experiment.construction();
            worker = experiment.worker(root.join("workspace"));
        }
        let writer = PreparedInputWriter::new(
            &root.join("source"),
            &root.join("preparation"),
            config,
            worker.max_artifact_bytes,
            ForestOptions {
                refinement_rounds,
                refinement_memory_bytes: worker.refinement_memory_bytes,
                sample_items: construction.sample_items,
                tree_memory_bytes: construction.memory_bytes,
                tree_scratch_bytes: construction.scratch_bytes,
                sort_memory_bytes: worker.sort_memory_bytes,
                sort_scratch_bytes: worker.sort_scratch_bytes,
            },
        )?;
        Ok(Self {
            construction,
            refinement_rounds,
            worker,
            writer,
        })
    }

    pub(super) fn append(mut self, records: Vec<Record>) -> Result<Self> {
        self.writer = self.writer.append(records.into_iter().map(Ok))?;
        Ok(self)
    }
}

pub(super) async fn build<B: Backend>(
    runtime: &Runtime<B>,
    staging: InputCapture,
    control: OperationOptions,
) -> Result<(Index<B>, serde_json::Value)> {
    let InputCapture {
        worker,
        refinement_rounds,
        construction: options,
        writer,
    } = staging;
    let settings = crate::report::bulk_build_settings(&options, &worker);
    let start = Instant::now();
    let prepared = tokio::task::spawn_blocking(move || writer.seal())
        .await
        .map_err(|e| Error::with_source(ErrorKind::Other, e))??;
    let snapshot_seconds = start.elapsed().as_secs_f64();
    let source = prepared.source().clone();
    let job = runtime
        .start_bulk_build(
            "vdbbench",
            &source,
            ktann::api::BulkBuildOptions {
                construction: options,
                refinement_rounds,
            },
        )
        .await?;
    let start = Instant::now();
    let (index, stages) = job
        .complete_with_report(worker, Some(prepared), control)
        .await?;
    let publish_seconds = stages.publish.as_secs_f64();
    let prepare_load_seconds = start.elapsed().as_secs_f64() - publish_seconds;
    let ready_resources = crate::resource::ResourceSnapshot::capture()
        .map_err(|error| io(std::io::Error::other(error)))?;
    let mut report = json!({
        "sort_written_bytes": stages.forest_report.map(|r| r.sort_written_bytes),
        "tree_written_bytes": stages.forest_report.map(|r| r.tree_written_bytes),
        "peak_sort_scratch_bytes": stages.forest_report.map(|r| r.peak_sort_scratch_bytes),
        "peak_tree_scratch_bytes": stages.forest_report.map(|r| r.peak_tree_scratch_bytes),
        "ready_peak_rss_bytes": ready_resources.peak_rss_bytes(),
        "serving_detail": stages.serving_report.map(|r| json!({
            "source_sort_seconds": r.source_sort.as_secs_f64(),
            "topology_sort_seconds": r.topology_sort.as_secs_f64(),
            "join_encode_seconds": r.join_encode.as_secs_f64(),
            "output_merge_seconds": r.output_merge.as_secs_f64(),
            "synopsis_seconds": r.synopsis.as_secs_f64(),
            "final_emit_seconds": r.final_emit.as_secs_f64(),
            "scratch_written_bytes": r.scratch_written_bytes,
            "peak_scratch_bytes": r.peak_scratch_bytes,
        })),
        "forest_seconds": stages.forest.as_secs_f64(),
        "serving_seconds": stages.serving.as_secs_f64(),
        "load_seconds": stages.load.as_secs_f64(),
        "snapshot_seconds": snapshot_seconds, "prepare_load_seconds": prepare_load_seconds,
        "validate_publish_cleanup_seconds": publish_seconds,
        "source_sha256": source.manifest().sha256(), "source_bytes": source.manifest().bytes(),
        "rotation_seed": stages.rotation_seed,
    });
    report
        .as_object_mut()
        .expect("bulk report object")
        .extend(settings);
    Ok((index, report))
}
