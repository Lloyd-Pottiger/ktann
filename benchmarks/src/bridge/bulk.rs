//! Finite input staging and real core Bulk Build for the benchmark bridge.
use bytes::Bytes;
use ktann::api::{BulkWorkerOptions, Error, ErrorKind, Index, IndexConfig, Record, Result, Value};
use ktann::bulk::InputSnapshot;
use ktann::construction::ConstructionOptions;
use ktann::runtime::Runtime;
use ktann::storage::backend::Backend;
use serde_json::json;
use std::fs::{self, File};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::PathBuf;
use std::time::Instant;

const DISK_LIMIT: u64 = 64 * 1024 * 1024 * 1024;

/// Caller-owned files are retained with the report; only core-owned attempts are reclaimed.
pub(super) struct Staging {
    root: PathBuf,
    config: IndexConfig,
    writer: BufWriter<File>,
    records: u64,
}
fn io(error: std::io::Error) -> Error {
    Error::with_source(ErrorKind::Other, error)
}
impl Staging {
    pub(super) fn new(root: PathBuf, config: IndexConfig) -> Result<Self> {
        if !root.is_absolute() {
            return Err(Error::new(ErrorKind::InvalidArgument));
        }
        fs::create_dir(&root).map_err(io)?;
        fs::create_dir(root.join("workspace")).map_err(io)?;
        let writer = BufWriter::new(File::create(root.join("input.bin")).map_err(io)?);
        Ok(Self {
            root,
            config,
            writer,
            records: 0,
        })
    }
    pub(super) fn append(mut self, records: Vec<Record>) -> Result<Self> {
        let bytes_per_record = 8 + self.config.dimension() as u64 * 4;
        let count = self
            .records
            .checked_add(records.len() as u64)
            .ok_or_else(|| Error::new(ErrorKind::InvalidArgument))?;
        if count > DISK_LIMIT / bytes_per_record {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
        for record in records {
            self.writer.write_all(record.id()).map_err(io)?;
            for value in record.vector() {
                self.writer.write_all(&value.to_le_bytes()).map_err(io)?;
            }
        }
        self.records = count;
        Ok(self)
    }
    fn finish(mut self) -> Result<InputSnapshot> {
        self.writer.flush().map_err(io)?;
        self.writer.get_ref().sync_all().map_err(io)?;
        drop(self.writer);
        let rows = Rows {
            reader: BufReader::new(File::open(self.root.join("input.bin")).map_err(io)?),
            dimension: self.config.dimension(),
            remaining: self.records,
            finished: false,
        };
        let source =
            InputSnapshot::create(&self.root.join("source"), self.config, DISK_LIMIT, rows)?;
        fs::remove_file(self.root.join("input.bin")).map_err(io)?;
        Ok(source)
    }
}
struct Rows {
    reader: BufReader<File>,
    dimension: usize,
    remaining: u64,
    finished: bool,
}
impl Iterator for Rows {
    type Item = Result<Record>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        if self.remaining == 0 {
            self.finished = true;
            return match self.reader.read(&mut [0]) {
                Ok(0) => None,
                Ok(_) => Some(Err(Error::new(ErrorKind::Corruption))),
                Err(e) => Some(Err(io(e))),
            };
        }
        let row = (|| {
            let mut id = [0; 8];
            self.reader.read_exact(&mut id).map_err(io)?;
            let mut vector = Vec::with_capacity(self.dimension);
            for _ in 0..self.dimension {
                let mut value = [0; 4];
                self.reader.read_exact(&mut value).map_err(io)?;
                vector.push(f32::from_le_bytes(value));
            }
            Record::new(
                Bytes::copy_from_slice(&id),
                vector,
                vec![Value::I64(i64::from_be_bytes(id))],
            )
        })();
        self.remaining -= 1;
        if row.is_err() {
            self.finished = true;
        }
        Some(row)
    }
}

pub(super) async fn build<B: Backend>(
    runtime: &Runtime<B>,
    staging: Staging,
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
