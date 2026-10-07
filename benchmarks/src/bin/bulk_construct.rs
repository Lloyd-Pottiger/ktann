//! Streaming SIFT construction probe for the Bulk Build algorithm gate.
//!
//! Usage: ktann-bulk-construct INPUT.fvecs OUTPUT_DIRECTORY [RECORD_LIMIT] [FOREST_TREES] [--serving]
//! This measures pure construction, not load/validation/publication or an SLA.

use std::fs::{self, File};
use std::io::{BufReader, Read};
use std::path::Path;
use std::time::Instant;

use bytes::Bytes;
use ktann::api::{
    DataType, Error, ErrorKind, FieldId, FieldSchema, IndexConfig, Metric, Record, Value,
};
use ktann::bulk::{ForestArtifact, ForestOptions, InputSnapshot, TreeArtifact};
use ktann::construction::{CONSTRUCTION_VERSION, ConstructionOptions};
use sha2::{Digest, Sha256};

const DIMENSION: usize = 128;
const SEED: [u8; 32] = [7; 32];

struct Source {
    file: BufReader<File>,
    remaining: u64,
    ordinal: u64,
    trees: Option<u64>,
}

impl Iterator for Source {
    type Item = ktann::api::Result<Record>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        self.remaining -= 1;
        let mut bytes = [0; 4 * (DIMENSION + 1)];
        if self.file.read_exact(&mut bytes).is_err()
            || u32::from_le_bytes(bytes[..4].try_into().expect("four bytes")) != DIMENSION as u32
        {
            self.remaining = 0;
            return Some(Err(Error::new(ErrorKind::InvalidArgument)));
        }
        let id = Bytes::copy_from_slice(&self.ordinal.to_be_bytes());
        self.ordinal += 1;
        let vector: Vec<f32> = bytes[4..]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        Some(Record::new(
            id,
            vector,
            self.trees.filter(|n| *n > 0).map_or_else(Vec::new, |n| {
                vec![Value::I64(((self.ordinal - 1) % n) as i64)]
            }),
        ))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if !(2..=5).contains(&args.len()) {
        return Err(
            "usage: ktann-bulk-construct INPUT.fvecs OUTPUT_DIRECTORY [RECORD_LIMIT] [FOREST_TREES] [--serving]".into(),
        );
    }
    let encode_serving = args.get(4).is_some();
    if encode_serving && args[4].to_str() != Some("--serving") {
        return Err("expected --serving".into());
    }
    let forest_trees = args
        .get(3)
        .map(|n| {
            n.to_str()
                .ok_or("invalid forest tree count")?
                .parse::<u64>()
                .map_err(|_| "invalid forest tree count")
        })
        .transpose()?;
    if forest_trees.is_some_and(|n| n > i64::MAX as u64) {
        return Err("forest tree count is too large".into());
    }
    let source_path = Path::new(&args[0]);
    let output = Path::new(&args[1]);
    let started = Instant::now();
    let file = File::open(source_path)?;
    let source_bytes = file.metadata()?.len();
    let row_bytes = 4 * (DIMENSION as u64 + 1);
    if source_bytes % row_bytes != 0 {
        return Err("invalid SIFT fvecs length".into());
    }
    let records = if let Some(limit) = args.get(2) {
        limit
            .to_str()
            .ok_or("invalid record limit")?
            .parse::<u64>()?
            .min(source_bytes / row_bytes)
    } else {
        source_bytes / row_bytes
    };
    let mut hasher = Sha256::new();
    let mut reader = BufReader::new(file);
    let mut buffer = [0; 64 * 1024];
    loop {
        let bytes = reader.read(&mut buffer)?;
        if bytes == 0 {
            break;
        }
        hasher.update(&buffer[..bytes]);
    }
    let source_sha256 = format!("{:x}", hasher.finalize());
    let preparation_seconds = started.elapsed().as_secs_f64();
    fs::create_dir(output)?;
    let options = ConstructionOptions {
        min_partition_entries: 64,
        max_partition_entries: 256,
        sample_items: 256,
        memory_bytes: 256 * 1024 * 1024,
        scratch_bytes: 32 * 1024 * 1024 * 1024,
    };
    let mut config = IndexConfig::new(DIMENSION, Metric::L2)?.with_partition_entries(64, 256)?;
    if forest_trees.is_some_and(|n| n > 0) {
        config = config
            .with_fields(vec![FieldSchema::new("bucket", DataType::I64)?])?
            .with_tree_key_fields(vec![FieldId(0)])?;
    }
    let snapshot_started = Instant::now();
    let input = InputSnapshot::create(
        &output.join("input"),
        config.clone(),
        32 * 1024 * 1024 * 1024,
        Source {
            file: BufReader::new(File::open(source_path)?),
            remaining: records,
            ordinal: 0,
            trees: forest_trees,
        },
    )?;
    let snapshot_seconds = snapshot_started.elapsed().as_secs_f64();
    let construction_started = Instant::now();
    if let Some(requested_trees) = forest_trees {
        let forest_options = ForestOptions {
            tree: options,
            sort_memory_bytes: 64 * 1024 * 1024,
            sort_scratch_bytes: 32 * 1024 * 1024 * 1024,
        };
        let (plan, report) = ForestArtifact::build(
            &output.join("plan"),
            &input,
            SEED,
            forest_options,
            32 * 1024 * 1024 * 1024,
        )?;
        let construction_seconds = construction_started.elapsed().as_secs_f64();
        let verification_started = Instant::now();
        plan.verify()?;
        let verification_seconds = verification_started.elapsed().as_secs_f64();
        let mut result = serde_json::json!({
            "scope": "durable input, global duplicate detection and multi-tree plan; excludes backend loading, exact serving validation and publication",
            "source_sha256": source_sha256, "source_bytes": source_bytes,
            "records": report.records, "dimension": DIMENSION, "metric": "l2",
            "algorithm_version": CONSTRUCTION_VERSION, "rotation_seed": SEED,
            "requested_trees": requested_trees, "trees": report.trees, "partitions": report.partitions,
            "sample_items": options.sample_items, "min_partition_entries": options.min_partition_entries, "max_partition_entries": options.max_partition_entries,
            "tree_memory_budget_bytes": options.memory_bytes, "tree_scratch_budget_bytes": options.scratch_bytes,
            "sort_memory_budget_bytes": forest_options.sort_memory_bytes, "sort_scratch_budget_bytes": forest_options.sort_scratch_bytes,
            "peak_sort_scratch_bytes": report.peak_sort_scratch_bytes, "sort_written_bytes": report.sort_written_bytes,
            "peak_tree_scratch_bytes": report.peak_tree_scratch_bytes, "tree_written_bytes": report.tree_written_bytes,
            "preparation_seconds": preparation_seconds, "snapshot_seconds": snapshot_seconds,
            "construction_seconds": construction_seconds, "artifact_verification_seconds": verification_seconds,
            "total_seconds": started.elapsed().as_secs_f64(), "input_snapshot_bytes": input.manifest().bytes(),
            "plan_bytes": plan.manifest().bytes(), "plan_sha256": plan.manifest().sha256(),
        });
        if encode_serving {
            use ktann::api::LogicalIndexId;
            use ktann::bulk::{ServingArtifact, ServingOptions};
            use ktann::storage::backend::HardLimits;
            use ktann::storage::values::{BloomParameters, IndexLifecycle, IndexManifest};
            let bloom = config
                .fields()
                .iter()
                .map(|field| BloomParameters::derive(field.synopsis()))
                .collect::<ktann::api::Result<Vec<_>>>()?;
            let index = IndexManifest::new(
                IndexLifecycle::Building,
                LogicalIndexId::new(1)?,
                config.clone(),
                SEED,
                bloom,
            )?;
            let serving_options = ServingOptions {
                memory_bytes: 128 * 1024 * 1024,
                scratch_bytes: 64 * 1024 * 1024 * 1024,
                hard_limits: HardLimits {
                    max_key_bytes: 10_000,
                    max_value_bytes: 100_000,
                },
            };
            let serving_started = Instant::now();
            let (serving, serving_report) = ServingArtifact::build(
                &output.join("serving"),
                &input,
                &plan,
                &index,
                serving_options,
                64 * 1024 * 1024 * 1024,
            )?;
            let serving_seconds = serving_started.elapsed().as_secs_f64();
            let verify_started = Instant::now();
            serving.verify()?;
            result["serving"] = serde_json::json!({ "records": serving_report.records, "partitions": serving_report.partitions, "keys": serving_report.keys, "bytes": serving.manifest().bytes(), "sha256": serving.manifest().sha256(), "encoding_seconds": serving_seconds, "verification_seconds": verify_started.elapsed().as_secs_f64(), "peak_scratch_bytes": serving_report.peak_scratch_bytes, "scratch_written_bytes": serving_report.scratch_written_bytes, "memory_budget_bytes": serving_options.memory_bytes, "scratch_budget_bytes": serving_options.scratch_bytes, "max_key_bytes": serving_options.hard_limits.max_key_bytes, "max_value_bytes": serving_options.hard_limits.max_value_bytes });
            result["scope"] = serde_json::json!(
                "durable source, forest, exact joins and serving KV encoding; excludes backend loading, sealed backend validation, publication and recall"
            );
            result["total_seconds"] = serde_json::json!(started.elapsed().as_secs_f64());
        }
        let json = serde_json::to_vec_pretty(&result)?;
        fs::write(output.join("report.json"), &json)?;
        println!("{}", String::from_utf8(json)?);
        return Ok(());
    }
    let (plan, report) = TreeArtifact::build(
        &output.join("plan"),
        &input,
        SEED,
        options,
        32 * 1024 * 1024 * 1024,
    )?;
    let construction_seconds = construction_started.elapsed().as_secs_f64();
    let verification_started = Instant::now();
    plan.verify()?;
    let verification_seconds = verification_started.elapsed().as_secs_f64();
    let result = serde_json::json!({
        "scope": "durable input and single-tree plan; excludes backend loading, exact serving validation and publication",
        "source_sha256": source_sha256, "source_bytes": source_bytes,
        "records": report.records, "dimension": DIMENSION, "metric": "l2",
        "algorithm_version": CONSTRUCTION_VERSION, "rotation_seed": SEED,
        "sample_items": options.sample_items, "min_partition_entries": options.min_partition_entries,
        "max_partition_entries": options.max_partition_entries,
        "memory_budget_bytes": options.memory_bytes, "scratch_budget_bytes": options.scratch_bytes,
        "preparation_seconds": preparation_seconds, "construction_seconds": construction_seconds,
        "snapshot_seconds": snapshot_seconds, "artifact_verification_seconds": verification_seconds,
        "total_seconds": started.elapsed().as_secs_f64(), "partitions": report.partitions,
        "partition_high_water": report.partition_high_water,
        "peak_scratch_bytes": report.peak_scratch_bytes,
        "scratch_written_bytes": report.scratch_written_bytes,
        "input_snapshot_bytes": input.manifest().bytes(),
        "plan_bytes": plan.manifest().bytes(),
        "plan_sha256": plan.manifest().sha256(),
    });
    let json = serde_json::to_vec_pretty(&result)?;
    fs::write(output.join("report.json"), &json)?;
    println!("{}", String::from_utf8(json)?);
    Ok(())
}
