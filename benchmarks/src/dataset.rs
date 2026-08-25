//! Fixed public and replayable synthetic benchmark datasets.

use std::path::Path;
use std::sync::Arc;

use bytes::Bytes;
use xxhash_rust::xxh3::Xxh3;

use crate::report::DatasetMetadata;

#[path = "../../tests/support/dataset.rs"]
#[expect(
    dead_code,
    reason = "the shared corpus generator also exposes helpers used only by integration tests"
)]
mod corpus_dataset;
#[path = "../../tests/support/fixtures.rs"]
mod fixtures;

/// IDs, indexed vectors, and held-out queries before metadata is attached.
type DatasetParts = (Vec<Bytes>, Vec<Arc<[f32]>>, Vec<Arc<[f32]>>);

/// One benchmark dataset with held-out queries.
#[derive(Clone, Debug)]
pub struct BenchmarkDataset {
    /// Stable Record IDs aligned with `base`.
    pub ids: Vec<Bytes>,
    /// Indexed vectors.
    pub base: Vec<Arc<[f32]>>,
    /// Held-out query vectors.
    pub queries: Vec<Arc<[f32]>>,
    /// Reproducible identity and dimensions.
    pub metadata: DatasetMetadata,
}

/// Loads a checked-in public dataset or generates a synthetic distribution.
///
/// # Errors
///
/// Returns an error for unknown datasets, zero/overflowing sizes, requests
/// beyond a public fixture, or vectors whose dimension differs from `dimension`.
pub fn load(
    name: &str,
    base_count: usize,
    query_count: usize,
    dimension: usize,
    seed: u64,
) -> Result<BenchmarkDataset, String> {
    if base_count == 0 || query_count == 0 || dimension == 0 {
        return Err("dataset counts and dimension must be positive".to_owned());
    }
    let (ids, base, queries) = match name {
        "siftsmall" => load_public(
            "siftsmall_base.fvecs",
            "siftsmall_query.fvecs",
            base_count,
            query_count,
            dimension,
        )?,
        "fashion-mnist" => load_public(
            "fashion_mnist_base.idx3-ubyte",
            "fashion_mnist_query.idx3-ubyte",
            base_count,
            query_count,
            dimension,
        )?,
        "clustered" | "skewed" | "duplicates" => {
            load_synthetic(name, base_count, query_count, dimension, seed)?
        }
        _ => return Err(format!("unknown dataset `{name}`")),
    };
    let checksum_xxh3_128 = checksum(&ids, &base, &queries);
    let metadata = DatasetMetadata {
        name: name.to_owned(),
        base_vectors: base.len(),
        query_vectors: queries.len(),
        dimension,
        checksum_xxh3_128,
    };
    Ok(BenchmarkDataset {
        ids,
        base,
        queries,
        metadata,
    })
}

/// Resolves the checked-in integration fixtures independently of process cwd.
fn fixture_dir() -> std::path::PathBuf {
    // The benchmark crate is a workspace sibling of the canonical fixtures;
    // resolving from CARGO_MANIFEST_DIR keeps commands independent of cwd.
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../tests/datadriven/data")
}

/// Loads bounded prefixes from one public base/query fixture pair.
fn load_public(
    base_name: &str,
    query_name: &str,
    base_count: usize,
    query_count: usize,
    dimension: usize,
) -> Result<DatasetParts, String> {
    // Public inputs are checked-in immutable byte streams. Prefix truncation
    // defines the smoke profile without producing a second fixture identity.
    let directory = fixture_dir();
    let mut base = fixtures::read_vectors(&directory, base_name);
    let mut queries = fixtures::read_vectors(&directory, query_name);
    if base_count > base.len() || query_count > queries.len() {
        return Err(format!(
            "requested {base_count}/{query_count} vectors beyond fixture sizes {}/{}",
            base.len(),
            queries.len()
        ));
    }
    base.truncate(base_count);
    queries.truncate(query_count);
    validate_dimension(&base, dimension)?;
    validate_dimension(&queries, dimension)?;
    let ids = ordinal_ids(base.len());
    Ok((ids, base, queries))
}

/// Generates one replayable distribution using the shared integration corpus.
fn load_synthetic(
    name: &str,
    base_count: usize,
    query_count: usize,
    dimension: usize,
    seed: u64,
) -> Result<DatasetParts, String> {
    // Generate base and queries in one deterministic stream, then keep the
    // tail held out so no query is also an indexed record.
    let total = base_count
        .checked_add(query_count)
        .ok_or_else(|| "synthetic dataset size overflow".to_owned())?;
    let spec = match name {
        "clustered" => format!("clustered:{total}:8:2"),
        "skewed" => format!("skewed:{total}"),
        "duplicates" => format!("dups:{total}:32"),
        _ => return Err(format!("unknown synthetic dataset `{name}`")),
    };
    let mut generated = corpus_dataset::generate(&spec, dimension, seed);
    let queries = generated.vectors.split_off(base_count);
    generated.ids.truncate(base_count);
    Ok((generated.ids, generated.vectors, queries))
}

/// Creates stable Record IDs for public fixtures that carry vectors only.
fn ordinal_ids(count: usize) -> Vec<Bytes> {
    (0..count)
        .map(|ordinal| Bytes::from(format!("r{ordinal:06}")))
        .collect()
}

/// Enforces the Logical Index's single fixed vector dimension at input load.
fn validate_dimension(vectors: &[Arc<[f32]>], expected: usize) -> Result<(), String> {
    if vectors.iter().any(|vector| vector.len() != expected) {
        return Err(format!("fixture dimension does not equal {expected}"));
    }
    Ok(())
}

/// Hashes the complete logical dataset into a cross-process stable identity.
fn checksum(ids: &[Bytes], base: &[Arc<[f32]>], queries: &[Arc<[f32]>]) -> String {
    // Hash lengths and IEEE-754 bits explicitly; native byte layout and path
    // metadata therefore cannot change a dataset's identity.
    let mut hasher = Xxh3::new();
    hasher.update(&(ids.len() as u64).to_le_bytes());
    for (id, vector) in ids.iter().zip(base) {
        hasher.update(&(id.len() as u64).to_le_bytes());
        hasher.update(id);
        update_vector(&mut hasher, vector);
    }
    hasher.update(&(queries.len() as u64).to_le_bytes());
    for vector in queries {
        update_vector(&mut hasher, vector);
    }
    format!("{:032x}", hasher.digest128())
}

/// Adds a length-delimited vector using canonical IEEE-754 component bits.
fn update_vector(hasher: &mut Xxh3, vector: &[f32]) {
    hasher.update(&(vector.len() as u64).to_le_bytes());
    for component in vector {
        hasher.update(&component.to_bits().to_le_bytes());
    }
}

/// Replayable xorshift64 generator required by the shared corpus generator.
struct Rng(u64);

impl Rng {
    /// Returns the next pseudo-random word.
    fn next(&mut self) -> u64 {
        let mut value = self.0;
        value ^= value << 13;
        value ^= value >> 7;
        value ^= value << 17;
        self.0 = value;
        value
    }

    /// Returns a deterministic word modulo the generator's positive bound.
    fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

#[cfg(test)]
mod tests {
    use super::load;

    #[test]
    fn synthetic_dataset_is_replayable() {
        let first = load("clustered", 32, 4, 8, 7).expect("dataset");
        let second = load("clustered", 32, 4, 8, 7).expect("dataset");
        assert_eq!(
            first.metadata.checksum_xxh3_128,
            second.metadata.checksum_xxh3_128
        );
        assert_eq!(first.ids, second.ids);
    }
}
