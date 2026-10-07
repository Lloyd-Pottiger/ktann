//! End-to-end SIFT build, paged publication, and held-out recall on RocksDB.
//! Usage: ktann-bulk-build DATASET_DIRECTORY NEW_OUTPUT_DIRECTORY [RECORD_LIMIT]
use bytes::Bytes;
use ktann::api::{
    BulkWorkerOptions, Error, ErrorKind, GetOptions, IndexConfig, Metric, Record, RuntimeConfig,
    SearchRequest,
};
use ktann::bulk::InputSnapshot;
use ktann::construction::ConstructionOptions;
use ktann::runtime::Runtime;
use ktann_rocksdb::{BackendNamespace, RocksDbBackend};
use rocksdb::{OptimisticTransactionDB, Options};
use std::{
    fs::{self, File},
    io::{BufReader, Read},
    path::Path,
    sync::Arc,
    time::Instant,
};

const DIMENSION: usize = 128;
fn vector(reader: &mut impl Read) -> std::io::Result<Vec<f32>> {
    let mut bytes = [0; 516];
    reader.read_exact(&mut bytes)?;
    if u32::from_le_bytes(bytes[..4].try_into().expect("dimension")) != DIMENSION as u32 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "expected SIFT dimension 128",
        ));
    }
    Ok(bytes[4..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect())
}
struct Source {
    reader: BufReader<File>,
    next: u64,
    total: u64,
}
impl Iterator for Source {
    type Item = ktann::api::Result<Record>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.next == self.total {
            return None;
        }
        let id = self.next;
        self.next += 1;
        Some(
            vector(&mut self.reader)
                .map_err(|e| Error::with_source(ErrorKind::Other, e))
                .and_then(|v| Record::new(Bytes::copy_from_slice(&id.to_be_bytes()), v, vec![])),
        )
    }
}
#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if !(2..=3).contains(&args.len()) {
        return Err("usage: ktann-bulk-build DATASET NEW_OUTPUT [RECORD_LIMIT]".into());
    }
    let dataset = Path::new(&args[0]);
    let output = Path::new(&args[1]);
    let records: u64 = args.get(2).map_or(Ok(1_000_000), |v| v.parse())?;
    if records == 0 || records > 1_000_000 {
        return Err("record limit must be 1..=1000000".into());
    }
    fs::create_dir(output)?;
    fs::create_dir(output.join("workspace"))?;
    let started = Instant::now();
    let source = InputSnapshot::create(
        &output.join("source"),
        IndexConfig::new(DIMENSION, Metric::L2)?.with_partition_entries(64, 256)?,
        64 * 1024 * 1024 * 1024,
        Source {
            reader: BufReader::new(File::open(dataset.join("sift_base.fvecs"))?),
            next: 0,
            total: records,
        },
    )?;
    let snapshot_seconds = started.elapsed().as_secs_f64();
    let mut db_options = Options::default();
    db_options.create_if_missing(true);
    db_options.set_max_background_jobs(2);
    let database = Arc::new(OptimisticTransactionDB::open(
        &db_options,
        output.join("database"),
    )?);
    let runtime = Runtime::new(
        RocksDbBackend::new(database, BackendNamespace::new("bulk-probe")?),
        RuntimeConfig::default().with_maintenance(0, 1)?,
    )?;
    let job = runtime
        .start_bulk_build(
            "sift",
            &source,
            ConstructionOptions {
                min_partition_entries: 64,
                max_partition_entries: 256,
                sample_items: 256,
                memory_bytes: 256 * 1024 * 1024,
                scratch_bytes: 64 * 1024 * 1024 * 1024,
            },
        )
        .await?;
    let worker = BulkWorkerOptions::new(output.join("workspace"));
    let build_started = Instant::now();
    job.run_worker(worker).await?;
    let build_and_load_seconds = build_started.elapsed().as_secs_f64();
    eprintln!(
        "loaded {records} records in {build_and_load_seconds:.3}s; validating and publishing"
    );
    let publication = Instant::now();
    let index = job.publish().await?;
    let validation_publication_cleanup_seconds = publication.elapsed().as_secs_f64();
    let build_total_seconds = started.elapsed().as_secs_f64();
    // Exercise ordinary serving against the original source, outside build time.
    let mut originals = BufReader::new(File::open(dataset.join("sift_base.fvecs"))?);
    for id in 0..records.min(16) {
        let expected = vector(&mut originals)?;
        let actual = index
            .get(
                Bytes::copy_from_slice(&id.to_be_bytes()),
                GetOptions::default(),
            )
            .await?
            .ok_or("published point read is missing")?;
        if actual.vector() != expected {
            return Err("published vector differs from source".into());
        }
    }
    let mut query_file = BufReader::new(File::open(dataset.join("sift_query.fvecs"))?);
    let mut truth_file = BufReader::new(File::open(dataset.join("sift_groundtruth.ivecs"))?);
    let queries = if records == 1_000_000 { 100 } else { 10 };
    let mut recall = 0.0;
    let mut latencies = Vec::new();
    for _ in 0..queries {
        let query = vector(&mut query_file)?;
        let mut size = [0; 4];
        truth_file.read_exact(&mut size)?;
        let count = u32::from_le_bytes(size) as usize;
        if count != 100 {
            return Err("unexpected SIFT truth width".into());
        }
        let mut truth = vec![0; count * 4];
        truth_file.read_exact(&mut truth)?;
        let search = Instant::now();
        let result = index.search(SearchRequest::new(query, 10)?).await?;
        latencies.push(search.elapsed().as_secs_f64());
        if result.hits.len() != 10 {
            return Err("published search did not return ten hits".into());
        }
        if records == 1_000_000 {
            let matches = truth
                .as_chunks::<4>()
                .0
                .iter()
                .take(10)
                .filter(|id| {
                    let id = u64::from(u32::from_le_bytes(**id)).to_be_bytes();
                    result.hits.iter().any(|hit| hit.id().as_ref() == id)
                })
                .count();
            recall += matches as f64 / 10.0;
        }
    }
    runtime.shutdown().await?;
    latencies.sort_by(f64::total_cmp);
    let report = serde_json::json!({"backend":"rocksdb","records":records,"dimension":DIMENSION,"source_manifest_sha256":source.manifest().sha256(),"snapshot_seconds":snapshot_seconds,"build_and_load_seconds":build_and_load_seconds,"validation_publication_cleanup_seconds":validation_publication_cleanup_seconds,"total_build_seconds":build_total_seconds,"held_out_queries":queries,"recall_at_10":(records==1_000_000).then_some(recall/queries as f64),"search_p50_seconds":latencies[queries/2],"search_p95_seconds":latencies[(queries*95/100).min(queries-1)],"scope":"source snapshot, durable worker, load, exact sealed backend validation, atomic publication, cleanup, and ordinary serving; no throughput comparison claimed"});
    let json = serde_json::to_string_pretty(&report)?;
    fs::write(output.join("report.json"), &json)?;
    println!("{json}");
    Ok(())
}
