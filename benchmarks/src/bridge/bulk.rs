//! Bridge-owned input and retryable completion through the core Bulk Build job.
use ktann::api::{
    BulkBuildJob, BulkBuildOptions, BulkBuildReport, BulkWorkerOptions, Error, ErrorKind, Index,
    IndexConfig, OperationOptions, Record, Result,
};
use ktann::bulk::{ForestOptions, PreparedInput, PreparedInputWriter};
use ktann::construction::ConstructionOptions;
use ktann::runtime::Runtime;
use ktann::storage::backend::Backend;
use serde_json::{Value, json};
use std::fs;
use std::path::PathBuf;
use std::time::{Duration, Instant};

const DISK_LIMIT: u64 = 64 * 1024 * 1024 * 1024;

/// Captures one finite input and retains the core job across Optimize retries.
pub(super) struct Build<B: Backend> {
    options: BulkBuildOptions,
    worker: BulkWorkerOptions,
    writer: Option<PreparedInputWriter>,
    // Keep preparation until reservation succeeds; completion may consume it.
    prepared: Option<PreparedInput>,
    job: Option<BulkBuildJob<B>>,
    /// Effective controls and completed phase observations; available even on failure.
    pub(super) report: Value,
}

fn io(error: std::io::Error) -> Error {
    Error::with_source(ErrorKind::Other, error)
}

impl<B: Backend> Build<B> {
    pub(super) fn new(
        root: PathBuf,
        config: IndexConfig,
        experiment: Option<crate::routing::Experiment>,
        rounds: u8,
        refinement_memory_bytes: usize,
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
        if let Some(experiment) = experiment {
            construction = experiment.construction();
            worker = experiment.worker(root.join("workspace"));
        }
        // Reset owns this budget for every routing mode.
        worker.refinement_memory_bytes = refinement_memory_bytes;
        let refinement_rounds = Some(rounds);
        let writer = PreparedInputWriter::new(
            &root.join("source"),
            &root.join("preparation"),
            config,
            worker.max_artifact_bytes,
            ForestOptions {
                refinement_rounds,
                refinement_memory_bytes,
                sample_items: construction.sample_items,
                tree_memory_bytes: construction.memory_bytes,
                tree_scratch_bytes: construction.scratch_bytes,
                sort_memory_bytes: worker.sort_memory_bytes,
                sort_scratch_bytes: worker.sort_scratch_bytes,
            },
        )?;
        let mut report = Value::Object(crate::report::bulk_build_settings(&construction, &worker));
        report["refinement_rounds"] = json!(rounds);
        Ok(Self {
            options: BulkBuildOptions {
                construction,
                refinement_rounds,
            },
            worker,
            writer: Some(writer),
            prepared: None,
            job: None,
            report,
        })
    }

    /// Consuming the writer closes capture on failure; unacknowledged input is not replayed.
    pub(super) async fn append(&mut self, records: Vec<Record>) -> Result<()> {
        let writer = self
            .writer
            .take()
            .ok_or_else(|| Error::new(ErrorKind::InvalidArgument))?;
        self.writer = Some(
            tokio::task::spawn_blocking(move || writer.append(records.into_iter().map(Ok)))
                .await
                .map_err(|e| Error::with_source(ErrorKind::Other, e))??,
        );
        Ok(())
    }

    /// Core owns durable progress: retain preparation on reservation errors and the job on completion errors.
    pub(super) async fn complete(
        &mut self,
        runtime: &Runtime<B>,
        control: OperationOptions,
    ) -> Result<Index<B>> {
        if let Some(writer) = self.writer.take() {
            let start = Instant::now();
            let prepared = tokio::task::spawn_blocking(move || writer.seal())
                .await
                .map_err(|e| Error::with_source(ErrorKind::Other, e))??;
            self.report["snapshot_seconds"] = json!(start.elapsed().as_secs_f64());
            self.report["source_sha256"] = json!(prepared.source().manifest().sha256());
            self.report["source_bytes"] = json!(prepared.source().manifest().bytes());
            self.prepared = Some(prepared);
        }
        if self.job.is_none() {
            let prepared = self
                .prepared
                .as_ref()
                .ok_or_else(|| Error::new(ErrorKind::InvalidArgument))?;
            self.job = Some(
                runtime
                    .start_bulk_build("vdbbench", prepared.source(), self.options)
                    .await?,
            );
        }
        let start = Instant::now();
        let (index, stages) = self
            .job
            .as_ref()
            .expect("reserved build")
            .complete_with_report(self.worker.clone(), self.prepared.take(), control)
            .await?;
        self.report
            .as_object_mut()
            .expect("build report object")
            .extend(completion_report(stages, start.elapsed()));
        Ok(index)
    }
}

/// Core stage timings describe only the successful completion call, including reused stages.
fn completion_report(stages: BulkBuildReport, elapsed: Duration) -> serde_json::Map<String, Value> {
    let publish_seconds = stages.publish.as_secs_f64();
    let report = json!({
        "sort_written_bytes": stages.forest_report.map(|r| r.sort_written_bytes),
        "tree_written_bytes": stages.forest_report.map(|r| r.tree_written_bytes),
        "peak_sort_scratch_bytes": stages.forest_report.map(|r| r.peak_sort_scratch_bytes),
        "peak_tree_scratch_bytes": stages.forest_report.map(|r| r.peak_tree_scratch_bytes),
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
        "prepare_load_seconds": elapsed.as_secs_f64() - publish_seconds,
        "validate_publish_cleanup_seconds": publish_seconds,
        "rotation_seed": stages.rotation_seed,
    });
    report
        .as_object()
        .expect("completion report object")
        .clone()
}

#[cfg(all(test, feature = "rocksdb"))]
mod tests {
    use super::*;
    use bytes::Bytes;
    use ktann::api::{Metric, RuntimeConfig};
    use ktann_rocksdb::{BackendNamespace, RocksDbBackend};
    use std::sync::Arc;
    use tokio_util::sync::CancellationToken;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_completion_resumes_the_same_job() {
        let directory = tempfile::tempdir().unwrap();
        let mut options = rocksdb::Options::default();
        options.create_if_missing(true);
        let database = Arc::new(
            rocksdb::OptimisticTransactionDB::open(&options, directory.path().join("db")).unwrap(),
        );
        let backend = RocksDbBackend::new(database, BackendNamespace::new("resume").unwrap());
        let runtime = Runtime::new(
            backend,
            RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
        )
        .unwrap();
        let mut build = Build::new(
            directory.path().join("build"),
            IndexConfig::new(2, Metric::L2).unwrap(),
            None,
            2,
            32 << 20,
        )
        .unwrap();
        build
            .append(vec![
                Record::new(Bytes::from_static(b"one"), vec![1., 2.], vec![]).unwrap(),
            ])
            .await
            .unwrap();
        let cancelled = CancellationToken::new();
        cancelled.cancel();
        assert!(
            build
                .complete(
                    &runtime,
                    OperationOptions::default().with_cancellation(cancelled)
                )
                .await
                .is_err()
        );
        let identity = runtime
            .open_bulk_build("vdbbench")
            .await
            .unwrap()
            .logical_index_id();
        assert_eq!(
            runtime.open_index("vdbbench").await.unwrap_err().kind(),
            ErrorKind::IndexBuilding
        );
        let index = build
            .complete(&runtime, OperationOptions::default())
            .await
            .unwrap();
        assert_eq!(index.logical_index_id(), identity);
        runtime.shutdown().await.unwrap();
    }
}
