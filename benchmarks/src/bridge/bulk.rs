//! Bridge-owned input and retryable completion through the core Bulk Build job.
use ktann::api::{
    BulkBuildJob, BulkBuildOptions, BulkBuildReport, BulkWorkerOptions, Error, ErrorKind, Index,
    IndexConfig, OperationOptions, Record, Result,
};
use ktann::bulk::{InputSnapshot, InputSnapshotWriter};
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
    root: PathBuf,
    options: BulkBuildOptions,
    worker: BulkWorkerOptions,
    writer: Option<InputSnapshotWriter>,
    source: Option<InputSnapshot>,
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
            // Cover both wide routing nodes and large resident builds when requested.
            memory_bytes: refinement_memory_bytes.max(1024 * 1024 * 1024),
            scratch_bytes: DISK_LIMIT,
        };
        if let Some(experiment) = experiment {
            construction = experiment.construction();
            worker = experiment.worker(root.join("workspace"));
        }
        // Reset owns the worker memory ceiling across both routing modes.
        worker.memory_bytes = refinement_memory_bytes;
        let refinement_rounds = Some(rounds);
        let writer = InputSnapshotWriter::new(&root.join("source"), config, worker.scratch_bytes)?;
        let mut report = Value::Object(crate::report::bulk_build_settings(&construction, &worker));
        report["refinement_rounds"] = json!(rounds);
        Ok(Self {
            root,
            options: BulkBuildOptions {
                construction,
                refinement_rounds,
            },
            worker,
            writer: Some(writer),
            source: None,
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

    /// Retains the sealed source across reservation errors and the job across completion errors.
    pub(super) async fn complete(
        &mut self,
        runtime: &Runtime<B>,
        control: OperationOptions,
    ) -> Result<Index<B>> {
        if let Some(writer) = self.writer.take() {
            let start = Instant::now();
            let source = tokio::task::spawn_blocking(move || writer.seal())
                .await
                .map_err(|e| Error::with_source(ErrorKind::Other, e))??;
            self.report["snapshot_seconds"] = json!(start.elapsed().as_secs_f64());
            self.report["source_sha256"] = json!(source.manifest().sha256());
            self.report["source_bytes"] = json!(source.manifest().bytes());
            self.source = Some(source);
        }
        if self.job.is_none() {
            let source = self
                .source
                .as_ref()
                .ok_or_else(|| Error::new(ErrorKind::InvalidArgument))?;
            self.job = Some(
                runtime
                    .start_bulk_build("vdbbench", source, self.options)
                    .await?,
            );
        }
        let start = Instant::now();
        let (index, stages) = self
            .job
            .as_ref()
            .expect("reserved build")
            .complete_with_report(self.worker.clone(), control)
            .await?;
        self.report
            .as_object_mut()
            .expect("build report object")
            .extend(completion_report(stages, start.elapsed()));
        // Reclaim staged source segments off the Tokio executor once the Active index is published.
        self.source = None;
        let source_dir = self.root.join("source");
        let _ = tokio::task::spawn_blocking(move || fs::remove_dir_all(source_dir)).await;
        Ok(index)
    }
}

/// Core stage timings describe only the successful completion call, including reused stages.
fn completion_report(stages: BulkBuildReport, elapsed: Duration) -> serde_json::Map<String, Value> {
    let publish_seconds = stages.publish.as_secs_f64();
    let forest_seconds =
        (stages.plan + stages.ingest + stages.construct + stages.refine).as_secs_f64();
    let report = json!({
        "in_memory_mode": stages.in_memory_mode,
        "records": stages.records,
        "trees": stages.trees,
        "partitions": stages.partitions,
        "entries": stages.entries,
        "scratch_written_bytes": stages.scratch_written_bytes,
        "peak_scratch_bytes": stages.peak_scratch_bytes,
        "plan_seconds": stages.plan.as_secs_f64(),
        "ingest_seconds": stages.ingest.as_secs_f64(),
        "construct_seconds": stages.construct.as_secs_f64(),
        "refine_seconds": stages.refine.as_secs_f64(),
        "emit_seconds": stages.emit.as_secs_f64(),
        "forest_seconds": forest_seconds,
        "serving_seconds": 0.0,
        "load_seconds": stages.emit.as_secs_f64(),
        "prepare_load_seconds": (elapsed.as_secs_f64() - publish_seconds).max(0.0),
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
        let record = index
            .get(
                Bytes::from_static(b"one"),
                ktann::api::GetOptions::default(),
            )
            .await
            .unwrap()
            .expect("published record");
        assert_eq!(record.vector(), &[1., 2.]);
        runtime.shutdown().await.unwrap();
    }
}
