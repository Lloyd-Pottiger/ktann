//! External datasets for the optimized large ANN quality profile.

use std::collections::HashSet;
use std::env;
use std::fs::File;
use std::io::{BufReader, Read};
use std::mem::size_of;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow_array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Int32Array, Int64Array, LargeListArray,
    ListArray, UInt32Array, UInt64Array,
};
use bytes::Bytes;
use md5::{Digest as _, Md5};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Deserialize;
use sha2::Sha256;

use super::{BenchmarkDataset, checksum, validate_dimension};
use crate::report::{DatasetFileMetadata, DatasetMetadata, DatasetSourceMetadata};

const DEFAULT_CACHE_DIR: &str = "/tmp/vectordb_bench/dataset";

type IdVectors = (Vec<Bytes>, Vec<Arc<[f32]>>);

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq)]
enum DatasetFormat {
    #[serde(rename = "vectordbbench_parquet")]
    VectorDbBenchParquet,
    #[serde(rename = "texmex_fvecs")]
    TexMexFvecs,
}

#[derive(Clone, Debug, Deserialize)]
struct Manifest {
    id: String,
    source_revision: String,
    format: DatasetFormat,
    metric: String,
    dimension: usize,
    base_vectors: usize,
    source_query_vectors: usize,
    benchmark_query_vectors: usize,
    ground_truth_neighbors: usize,
    files: Vec<ManifestFile>,
}

#[derive(Clone, Debug, Deserialize)]
struct ManifestFile {
    role: String,
    path: String,
    url: String,
    bytes: u64,
    checksum: FileChecksum,
}

#[derive(Clone, Debug, Deserialize)]
struct FileChecksum {
    algorithm: String,
    value: String,
    part_bytes: Option<usize>,
}

/// Loads and validates one manifest-defined large-profile dataset from the shared cache.
///
/// # Errors
///
/// Returns an error when the manifest is invalid, a required file is absent or
/// fails its checksum, or its decoded vectors and supplied ground truth do not
/// match the declared shape.
pub fn load_large(name: &str) -> Result<BenchmarkDataset, String> {
    let manifest = parse_manifest(name)?;
    validate_manifest(&manifest)?;
    let cache = cache_dir();
    validate_files(&cache, &manifest)?;
    let (ids, base, queries, ground_truth) = match manifest.format {
        DatasetFormat::VectorDbBenchParquet => load_parquet(&cache, &manifest)?,
        DatasetFormat::TexMexFvecs => load_texmex(&cache, &manifest)?,
    };
    validate_loaded(&manifest, &ids, &base, &queries, &ground_truth)?;
    let checksum_xxh3_128 = checksum(&ids, &base, &queries);
    let source = DatasetSourceMetadata {
        manifest_id: manifest.id.clone(),
        source_revision: manifest.source_revision.clone(),
        files: manifest
            .files
            .iter()
            .map(|file| DatasetFileMetadata {
                role: file.role.clone(),
                path: file.path.clone(),
                bytes: file.bytes,
                checksum_algorithm: file.checksum.algorithm.clone(),
                checksum: file.checksum.value.clone(),
            })
            .collect(),
    };
    Ok(BenchmarkDataset {
        ids,
        base,
        queries,
        ground_truth: Some(ground_truth),
        metadata: DatasetMetadata {
            name: manifest.id,
            base_vectors: manifest.base_vectors,
            query_vectors: manifest.benchmark_query_vectors,
            dimension: manifest.dimension,
            checksum_xxh3_128,
            source: Some(source),
        },
    })
}

fn parse_manifest(name: &str) -> Result<Manifest, String> {
    let bytes = match name {
        "cohere-1m" => include_str!("../../datasets/cohere-1m.json"),
        "sift-1m" => include_str!("../../datasets/sift-1m.json"),
        _ => return Err(format!("unknown large dataset `{name}`")),
    };
    serde_json::from_str(bytes).map_err(|error| format!("decode {name} dataset manifest: {error}"))
}

fn validate_manifest(manifest: &Manifest) -> Result<(), String> {
    if manifest.dimension == 0
        || manifest.base_vectors < 500_000
        || manifest.benchmark_query_vectors == 0
        || manifest.benchmark_query_vectors > manifest.source_query_vectors
        || manifest.ground_truth_neighbors == 0
    {
        return Err(format!(
            "dataset manifest {} has invalid shape",
            manifest.id
        ));
    }
    if !matches!(manifest.metric.as_str(), "l2" | "cosine") {
        return Err(format!(
            "dataset manifest {} has invalid metric",
            manifest.id
        ));
    }
    for role in ["base", "queries", "ground_truth"] {
        if manifest
            .files
            .iter()
            .filter(|file| file.role == role)
            .count()
            != 1
        {
            return Err(format!(
                "dataset manifest {} must declare one {role} file",
                manifest.id
            ));
        }
    }
    for file in &manifest.files {
        if file.bytes == 0
            || file.path.is_empty()
            || file.url.is_empty()
            || Path::new(&file.path).is_absolute()
            || Path::new(&file.path)
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            return Err(format!(
                "dataset manifest {} has an invalid file entry",
                manifest.id
            ));
        }
    }
    Ok(())
}

fn cache_dir() -> PathBuf {
    env::var_os("KTANN_BENCH_DATASET_CACHE")
        .map_or_else(|| PathBuf::from(DEFAULT_CACHE_DIR), PathBuf::from)
}

fn file_for<'a>(manifest: &'a Manifest, role: &str) -> Result<&'a ManifestFile, String> {
    manifest
        .files
        .iter()
        .find(|file| file.role == role)
        .ok_or_else(|| format!("dataset manifest {} is missing {role}", manifest.id))
}

fn validate_files(cache: &Path, manifest: &Manifest) -> Result<(), String> {
    for file in &manifest.files {
        let path = cache.join(&file.path);
        let metadata = path.metadata().map_err(|error| {
            format!(
                "dataset file {} is unavailable: {error}; see benchmarks/README.md",
                path.display()
            )
        })?;
        if metadata.len() != file.bytes {
            return Err(format!(
                "dataset file {} has {} bytes; expected {}",
                path.display(),
                metadata.len(),
                file.bytes
            ));
        }
        let actual = match file.checksum.algorithm.as_str() {
            "sha256" => sha256(&path)?,
            "s3_etag_md5" => s3_etag(&path, file.checksum.part_bytes)?,
            algorithm => return Err(format!("unsupported dataset checksum `{algorithm}`")),
        };
        if actual != file.checksum.value {
            return Err(format!(
                "dataset file {} failed {} checksum: got {actual}, expected {}",
                path.display(),
                file.checksum.algorithm,
                file.checksum.value
            ));
        }
    }
    Ok(())
}

fn sha256(path: &Path) -> Result<String, String> {
    let mut reader = BufReader::new(
        File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?,
    );
    let mut hasher = Sha256::new();
    let mut buffer = vec![0_u8; 1 << 20];
    loop {
        let count = reader
            .read(&mut buffer)
            .map_err(|error| format!("checksum {}: {error}", path.display()))?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("{:x}", hasher.finalize()))
}

fn s3_etag(path: &Path, part_bytes: Option<usize>) -> Result<String, String> {
    let part_bytes = part_bytes.ok_or_else(|| "s3 ETag checksum needs part_bytes".to_owned())?;
    if part_bytes == 0 {
        return Err("s3 ETag part_bytes must be positive".to_owned());
    }
    let mut reader = BufReader::new(
        File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?,
    );
    let mut buffer = vec![0_u8; part_bytes];
    let mut parts = Vec::new();
    loop {
        let mut read = 0;
        while read < buffer.len() {
            let count = reader
                .read(&mut buffer[read..])
                .map_err(|error| format!("checksum {}: {error}", path.display()))?;
            if count == 0 {
                break;
            }
            read += count;
        }
        if read == 0 {
            break;
        }
        parts.push(Md5::digest(&buffer[..read]));
        if read < buffer.len() {
            break;
        }
    }
    if parts.len() == 1 {
        return Ok(format!("{:x}", parts[0]));
    }
    let mut combined = Md5::new();
    for part in &parts {
        combined.update(part);
    }
    Ok(format!("{:x}-{}", combined.finalize(), parts.len()))
}

type LoadedParts = (
    Vec<Bytes>,
    Vec<Arc<[f32]>>,
    Vec<Arc<[f32]>>,
    Vec<Vec<Bytes>>,
);

fn load_parquet(cache: &Path, manifest: &Manifest) -> Result<LoadedParts, String> {
    let base_path = cache.join(&file_for(manifest, "base")?.path);
    let query_path = cache.join(&file_for(manifest, "queries")?.path);
    let truth_path = cache.join(&file_for(manifest, "ground_truth")?.path);
    let (ids, base) = read_parquet_vectors(&base_path, None, manifest.dimension)?;
    let (_, queries) = read_parquet_vectors(
        &query_path,
        Some(manifest.benchmark_query_vectors),
        manifest.dimension,
    )?;
    let truth = read_parquet_truth(
        &truth_path,
        manifest.benchmark_query_vectors,
        manifest.ground_truth_neighbors,
    )?;
    Ok((ids, base, queries, truth))
}

fn read_parquet_vectors(
    path: &Path,
    limit: Option<usize>,
    dimension: usize,
) -> Result<IdVectors, String> {
    let file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|error| format!("read {}: {error}", path.display()))?
        .with_batch_size(1_024)
        .build()
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    let mut ids = Vec::new();
    let mut vectors = Vec::new();
    for batch in reader {
        let batch = batch.map_err(|error| format!("read {}: {error}", path.display()))?;
        let id = batch
            .column_by_name("id")
            .ok_or_else(|| format!("{} has no id column", path.display()))?;
        let emb = batch
            .column_by_name("emb")
            .ok_or_else(|| format!("{} has no emb column", path.display()))?;
        for row in 0..batch.num_rows() {
            if limit.is_some_and(|limit| vectors.len() >= limit) {
                return Ok((ids, vectors));
            }
            ids.push(encoded_id(integer_at(id.as_ref(), row, path)?));
            vectors.push(vector_at(emb.as_ref(), row, dimension, path)?);
        }
    }
    Ok((ids, vectors))
}

fn read_parquet_truth(
    path: &Path,
    queries: usize,
    neighbors: usize,
) -> Result<Vec<Vec<Bytes>>, String> {
    let file = File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)
        .map_err(|error| format!("read {}: {error}", path.display()))?
        .with_batch_size(1_024)
        .build()
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    let mut truth = Vec::with_capacity(queries);
    for batch in reader {
        let batch = batch.map_err(|error| format!("read {}: {error}", path.display()))?;
        let values = batch
            .column_by_name("neighbors_id")
            .ok_or_else(|| format!("{} has no neighbors_id column", path.display()))?;
        for row in 0..batch.num_rows() {
            if truth.len() == queries {
                return Ok(truth);
            }
            truth.push(integer_list_at(values.as_ref(), row, neighbors, path)?);
        }
    }
    Ok(truth)
}

fn integer_at(array: &dyn Array, row: usize, path: &Path) -> Result<u64, String> {
    if array.is_null(row) {
        return Err(format!("{} contains a NULL id", path.display()));
    }
    if let Some(values) = array.as_any().downcast_ref::<Int64Array>() {
        return u64::try_from(values.value(row))
            .map_err(|_| format!("{} contains a negative id", path.display()));
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt64Array>() {
        return Ok(values.value(row));
    }
    if let Some(values) = array.as_any().downcast_ref::<Int32Array>() {
        return u64::try_from(values.value(row))
            .map_err(|_| format!("{} contains a negative id", path.display()));
    }
    if let Some(values) = array.as_any().downcast_ref::<UInt32Array>() {
        return Ok(u64::from(values.value(row)));
    }
    Err(format!("{} has a non-integer id column", path.display()))
}

fn encoded_id(id: u64) -> Bytes {
    Bytes::copy_from_slice(&id.to_be_bytes())
}

fn vector_at(
    array: &dyn Array,
    row: usize,
    dimension: usize,
    path: &Path,
) -> Result<Arc<[f32]>, String> {
    float_values(
        list_value(array, row, path, "emb")?.as_ref(),
        dimension,
        path,
    )
}

/// Reads a row from any Arrow list representation used by the source files.
fn list_value(
    array: &dyn Array,
    row: usize,
    path: &Path,
    column: &str,
) -> Result<ArrayRef, String> {
    if let Some(list) = array.as_any().downcast_ref::<ListArray>() {
        return Ok(list.value(row));
    }
    if let Some(list) = array.as_any().downcast_ref::<FixedSizeListArray>() {
        return Ok(list.value(row));
    }
    if let Some(list) = array.as_any().downcast_ref::<LargeListArray>() {
        return Ok(list.value(row));
    }
    Err(format!("{} has a non-list {column} column", path.display()))
}

fn float_values(array: &dyn Array, dimension: usize, path: &Path) -> Result<Arc<[f32]>, String> {
    let values = array
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| format!("{} has non-f32 vectors", path.display()))?;
    if values.len() != dimension || values.null_count() != 0 {
        return Err(format!("{} has an invalid vector shape", path.display()));
    }
    Ok(Arc::from(values.values().as_ref()))
}

fn integer_list_at(
    array: &dyn Array,
    row: usize,
    neighbors: usize,
    path: &Path,
) -> Result<Vec<Bytes>, String> {
    let values = list_value(array, row, path, "ground truth")?;
    if values.len() < neighbors {
        return Err(format!("{} has too few exact neighbors", path.display()));
    }
    (0..neighbors)
        .map(|index| integer_at(values.as_ref(), index, path).map(encoded_id))
        .collect()
}

fn load_texmex(cache: &Path, manifest: &Manifest) -> Result<LoadedParts, String> {
    let base = read_fvecs(
        &cache.join(&file_for(manifest, "base")?.path),
        manifest.dimension,
        manifest.base_vectors,
    )?;
    let queries = read_fvecs(
        &cache.join(&file_for(manifest, "queries")?.path),
        manifest.dimension,
        manifest.benchmark_query_vectors,
    )?;
    let ground_truth = read_ivecs(
        &cache.join(&file_for(manifest, "ground_truth")?.path),
        manifest.benchmark_query_vectors,
        manifest.ground_truth_neighbors,
    )?;
    let ids = (0..base.len()).map(|id| encoded_id(id as u64)).collect();
    Ok((ids, base, queries, ground_truth))
}

fn read_fvecs(path: &Path, dimension: usize, limit: usize) -> Result<Vec<Arc<[f32]>>, String> {
    let mut reader = BufReader::new(
        File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?,
    );
    let mut vectors = Vec::with_capacity(limit);
    let mut bytes = vec![0_u8; dimension * size_of::<f32>()];
    for _ in 0..limit {
        let stored_dimension = read_i32(&mut reader, path)?;
        if usize::try_from(stored_dimension).ok() != Some(dimension) {
            return Err(format!("{} has an invalid fvecs dimension", path.display()));
        }
        reader
            .read_exact(&mut bytes)
            .map_err(|error| format!("read {}: {error}", path.display()))?;
        vectors.push(
            bytes
                .chunks_exact(4)
                .map(|bytes| f32::from_le_bytes(bytes.try_into().expect("four-byte chunk")))
                .collect(),
        );
    }
    Ok(vectors)
}

fn read_ivecs(path: &Path, queries: usize, neighbors: usize) -> Result<Vec<Vec<Bytes>>, String> {
    let mut reader = BufReader::new(
        File::open(path).map_err(|error| format!("open {}: {error}", path.display()))?,
    );
    let mut truth = Vec::with_capacity(queries);
    for _ in 0..queries {
        let count = read_i32(&mut reader, path)?;
        if usize::try_from(count).ok() != Some(neighbors) {
            return Err(format!("{} has an invalid ivecs width", path.display()));
        }
        let mut row = Vec::with_capacity(neighbors);
        for _ in 0..neighbors {
            let id = read_i32(&mut reader, path)?;
            let id = u64::try_from(id)
                .map_err(|_| format!("{} contains a negative neighbor id", path.display()))?;
            row.push(encoded_id(id));
        }
        truth.push(row);
    }
    Ok(truth)
}

fn read_i32(reader: &mut impl Read, path: &Path) -> Result<i32, String> {
    let mut bytes = [0_u8; 4];
    reader
        .read_exact(&mut bytes)
        .map_err(|error| format!("read {}: {error}", path.display()))?;
    Ok(i32::from_le_bytes(bytes))
}

fn validate_loaded(
    manifest: &Manifest,
    ids: &[Bytes],
    base: &[Arc<[f32]>],
    queries: &[Arc<[f32]>],
    truth: &[Vec<Bytes>],
) -> Result<(), String> {
    if ids.len() != manifest.base_vectors
        || base.len() != manifest.base_vectors
        || queries.len() != manifest.benchmark_query_vectors
        || truth.len() != manifest.benchmark_query_vectors
        || truth
            .iter()
            .any(|neighbors| neighbors.len() != manifest.ground_truth_neighbors)
    {
        return Err(format!(
            "dataset {} does not match its manifest",
            manifest.id
        ));
    }
    validate_dimension(base, manifest.dimension)?;
    validate_dimension(queries, manifest.dimension)?;
    let unique: HashSet<&Bytes> = ids.iter().collect();
    if unique.len() != ids.len() {
        return Err(format!(
            "dataset {} contains duplicate base IDs",
            manifest.id
        ));
    }
    if truth
        .iter()
        .flatten()
        .any(|neighbor| !unique.contains(neighbor))
    {
        return Err(format!(
            "dataset {} ground truth references an unknown base ID",
            manifest.id
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::{DatasetFormat, parse_manifest, s3_etag, sha256, validate_manifest};

    #[test]
    fn embedded_large_manifests_define_required_quality_inputs() {
        let cohere = parse_manifest("cohere-1m").expect("Cohere manifest");
        let sift = parse_manifest("sift-1m").expect("SIFT manifest");
        validate_manifest(&cohere).expect("valid Cohere manifest");
        validate_manifest(&sift).expect("valid SIFT manifest");
        assert_eq!(cohere.format, DatasetFormat::VectorDbBenchParquet);
        assert_eq!(cohere.metric, "cosine");
        assert_eq!(sift.format, DatasetFormat::TexMexFvecs);
        assert_eq!(sift.metric, "l2");
        assert!(cohere.base_vectors >= 1_000_000);
        assert!(sift.base_vectors >= 500_000);
        assert!(cohere.benchmark_query_vectors >= 1_000);
        assert!(sift.benchmark_query_vectors >= 1_000);
    }

    #[test]
    fn checksum_implementations_match_published_formats() {
        let directory = tempfile::tempdir().expect("temporary directory");
        let single = directory.path().join("single");
        fs::write(&single, b"abc").expect("write single part");
        assert_eq!(
            s3_etag(&single, Some(8)).expect("single-part ETag"),
            "900150983cd24fb0d6963f7d28e17f72"
        );
        assert_eq!(
            sha256(&single).expect("SHA-256"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );

        let multipart = directory.path().join("multipart");
        fs::write(&multipart, b"abcdefgh").expect("write multipart object");
        assert_eq!(
            s3_etag(&multipart, Some(4)).expect("multipart ETag"),
            "cb93ad6c9c920e2602b79a11ded63ddb-2"
        );
    }
}
