//! Deterministic vector datasets for the data-driven corpus.
//!
//! Synthetic datasets are a pure function of their specification string and
//! the index dimension, generated through the repository's replayable
//! xorshift64 generator, so a corpus run is reproducible across processes and
//! machines. All synthetic arithmetic below is integer or IEEE-754 f64 with a
//! single rounding to f32 at the end, which is platform-independent.
//!
//! Real datasets come from the checked-in fixtures under
//! `tests/datadriven/data/` (see its README for provenance); they are fixed
//! byte streams, so they are replayable by construction and ignore `seed`.
//!
//! Specification grammar (`kind:count[:parameters]`, dimension comes from the
//! index under test):
//!
//! - `uniform:N` — components uniform in `[-1, 1)`.
//! - `clustered:N:C[:sep]` — `C` Gaussian clusters whose centroids are uniform
//!   in `[-2, 2)` scaled by `sep` (default 1); points scatter around their
//!   centroid with a fixed small deviation. The cluster structure is what
//!   gives a K-means tree meaningful routing decisions.
//! - `skewed:N` — a line of eight hotspots with geometrically decreasing
//!   weight, producing heavily unbalanced partitions under splits.
//! - `dups:N:M` — `M` base vectors; each record is an exact copy of a base or
//!   carries tiny jitter, exercising ties and duplicate-heavy regions.
//! - `file:NAME` — a fixture from `tests/datadriven/data/`; `.fvecs` and
//!   `.idx3-ubyte` (MNIST IDX) formats are supported. The fixture dimension
//!   must equal the index dimension.

use std::sync::Arc;

use bytes::Bytes;

use super::Rng;

/// One generated dataset: stable Record IDs and their vectors in index order.
pub struct Dataset {
    /// `ids[i]` is the Record ID of `vectors[i]`.
    pub ids: Vec<Bytes>,
    /// The generated vectors, each of the index dimension.
    pub vectors: Vec<Arc<[f32]>>,
}

impl Dataset {
    /// Returns the number of generated records.
    #[must_use]
    pub fn len(&self) -> usize {
        self.ids.len()
    }
}

/// Generates the dataset named by `spec` at `dimension` components per vector.
///
/// Bad specifications are corpus-authoring errors and panic with the spec.
#[must_use]
pub fn generate(spec: &str, dimension: usize, seed: u64) -> Dataset {
    let mut parts = spec.split(':');
    let kind = parts.next().unwrap_or("");
    if kind == "file" {
        let vectors = read_fixture(parts.next().unwrap_or(""), dimension, spec);
        let ids = (0..vectors.len())
            .map(|ordinal| Bytes::from(format!("r{ordinal:06}")))
            .collect();
        return Dataset { ids, vectors };
    }
    let count: usize = parse(parts.next(), spec);
    let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1);

    let vectors: Vec<Arc<[f32]>> = match kind {
        "uniform" => (0..count)
            .map(|_| {
                (0..dimension)
                    .map(|_| next_f32(&mut rng, -1.0, 1.0))
                    .collect()
            })
            .collect(),
        "clustered" => {
            let clusters: usize = parse(parts.next(), spec);
            assert!(
                clusters > 0,
                "clustered spec `{spec}` needs at least one cluster"
            );
            let separation: f64 = parts
                .next()
                .map(|part| part.parse().expect("clustered separation must be numeric"))
                .unwrap_or(1.0);
            let centroids: Vec<Vec<f64>> = (0..clusters)
                .map(|_| {
                    (0..dimension)
                        .map(|_| next_f64(&mut rng) * 4.0 * separation - 2.0 * separation)
                        .collect()
                })
                .collect();
            (0..count)
                .map(|_| {
                    let centroid = &centroids[rng.below(clusters as u64) as usize];
                    (0..dimension)
                        .map(|component| (centroid[component] + gaussian(&mut rng) * 0.15) as f32)
                        .collect()
                })
                .collect()
        }
        "skewed" => {
            // Eight hotspots along the diagonal; hotspot i takes roughly
            // 2^-i of the population, so partitions fill very unevenly.
            let hotspots: Vec<f64> = (0..8).map(|i| (i as f64) - 3.5).collect();
            (0..count)
                .map(|_| {
                    let mut pick = 0_usize;
                    while pick + 1 < hotspots.len() && rng.next() % 2 == 0 {
                        pick += 1;
                    }
                    let base = hotspots[pick];
                    (0..dimension)
                        .map(|_| (base + gaussian(&mut rng) * 0.1) as f32)
                        .collect()
                })
                .collect()
        }
        "dups" => {
            let distinct: usize = parse(parts.next(), spec);
            assert!(
                distinct > 0,
                "dups spec `{spec}` needs at least one base vector"
            );
            let bases: Vec<Vec<f32>> = (0..distinct)
                .map(|_| {
                    (0..dimension)
                        .map(|_| next_f32(&mut rng, -1.0, 1.0))
                        .collect()
                })
                .collect();
            (0..count)
                .map(|_| {
                    let base = &bases[rng.below(distinct as u64) as usize];
                    if rng.next() % 2 == 0 {
                        base.clone().into()
                    } else {
                        base.iter()
                            .map(|component| component + (gaussian(&mut rng) * 1e-3) as f32)
                            .collect()
                    }
                })
                .collect()
        }
        other => panic!("unknown dataset kind in spec `{spec}`: `{other}`"),
    };

    Dataset {
        ids: (0..count)
            .map(|ordinal| Bytes::from(format!("r{ordinal:06}")))
            .collect(),
        vectors,
    }
}

/// The next uniform f64 in `[0, 1)`, from the top 53 bits of the generator.
fn next_f64(rng: &mut Rng) -> f64 {
    (rng.next() >> 11) as f64 / (1_u64 << 53) as f64
}

/// The next uniform f32 in `[low, high)`.
fn next_f32(rng: &mut Rng, low: f64, high: f64) -> f32 {
    (low + next_f64(rng) * (high - low)) as f32
}

/// One standard normal sample via the Box-Muller transform.
fn gaussian(rng: &mut Rng) -> f64 {
    let u1 = 1.0 - next_f64(rng); // (0, 1]: the log must stay finite
    let u2 = next_f64(rng);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Parses one unsigned spec field, panicking with the spec on failure.
fn parse<T: std::str::FromStr>(part: Option<&str>, spec: &str) -> T {
    part.and_then(|part| part.parse().ok())
        .unwrap_or_else(|| panic!("bad dataset spec `{spec}`"))
}

/// The directory holding the checked-in real-dataset fixtures.
const FIXTURE_DIR: &str = "tests/datadriven/data";

/// Reads one fixture file by plain file name (no path separators).
fn fixture_bytes(name: &str, spec: &str) -> (std::path::PathBuf, Vec<u8>) {
    assert!(
        !name.is_empty() && !name.contains(['/', '\\']) && !name.contains(".."),
        "bad fixture name in spec `{spec}`"
    );
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join(FIXTURE_DIR)
        .join(name);
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|error| panic!("read fixture {}: {error}", path.display()));
    (path, bytes)
}

/// Loads the vector fixture named by a `file:` spec and checks that its
/// dimension equals the index dimension.
fn read_fixture(name: &str, dimension: usize, spec: &str) -> Vec<Arc<[f32]>> {
    let (path, bytes) = fixture_bytes(name, spec);
    let vectors = if name.ends_with(".fvecs") {
        parse_fvecs(&bytes, spec)
    } else if name.ends_with(".idx3-ubyte") {
        parse_idx3_ubyte(&bytes, spec)
    } else {
        panic!(
            "unknown fixture format in spec `{spec}`: {}",
            path.display()
        )
    };
    let found = vectors.first().map_or(0, |vector| vector.len());
    assert!(
        found == dimension,
        "fixture dimension {found} != index dimension {dimension} in spec `{spec}`"
    );
    vectors
}

/// Parses the `.fvecs` layout: per vector one little-endian i32 dimension
/// prefix followed by that many little-endian f32 components.
fn parse_fvecs(bytes: &[u8], spec: &str) -> Vec<Arc<[f32]>> {
    assert!(bytes.len() >= 4, "bad fvecs fixture in spec `{spec}`");
    let dimension = i32::from_le_bytes(bytes[..4].try_into().expect("prefix")) as usize;
    let record = 4 + dimension * 4;
    assert!(
        dimension > 0 && bytes.len() % record == 0,
        "truncated fvecs fixture in spec `{spec}`"
    );
    (0..bytes.len() / record)
        .map(|index| {
            let payload = &bytes[index * record..(index + 1) * record];
            assert!(
                i32::from_le_bytes(payload[..4].try_into().expect("prefix")) as usize == dimension,
                "inconsistent fvecs dimension in spec `{spec}`"
            );
            payload[4..]
                .chunks_exact(4)
                .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("component")))
                .collect()
        })
        .collect()
}

/// Parses the MNIST IDX ubyte layout: a big-endian header (magic 0x803,
/// count, rows, columns) followed by `count` row-major images, converted to
/// raw f32 intensities in `[0, 255]`.
fn parse_idx3_ubyte(bytes: &[u8], spec: &str) -> Vec<Arc<[f32]>> {
    let header = |index: usize| -> i32 {
        bytes
            .get(index * 4..index * 4 + 4)
            .and_then(|chunk| chunk.try_into().ok())
            .map(i32::from_be_bytes)
            .unwrap_or_else(|| panic!("bad idx3-ubyte fixture in spec `{spec}`"))
    };
    assert!(header(0) == 0x803, "bad idx3-ubyte magic in spec `{spec}`");
    let count = header(1) as usize;
    let dimension = (header(2) * header(3)) as usize;
    assert!(
        bytes.len() == 16 + count * dimension,
        "truncated idx3-ubyte fixture in spec `{spec}`"
    );
    (0..count)
        .map(|index| {
            bytes[16 + index * dimension..16 + (index + 1) * dimension]
                .iter()
                .map(|byte| f32::from(*byte))
                .collect()
        })
        .collect()
}

/// Reads an `.ivecs` fixture (the `.fvecs` layout with i32 components); used
/// to check the brute-force oracle against published ground truth.
#[must_use]
pub fn read_ivecs_fixture(name: &str) -> Vec<Vec<i32>> {
    let (_, bytes) = fixture_bytes(name, "ivecs fixture");
    assert!(bytes.len() >= 4, "bad ivecs fixture `{name}`");
    let width = i32::from_le_bytes(bytes[..4].try_into().expect("prefix")) as usize;
    let record = 4 + width * 4;
    assert!(
        width > 0 && bytes.len() % record == 0,
        "truncated ivecs fixture `{name}`"
    );
    (0..bytes.len() / record)
        .map(|index| {
            let payload = &bytes[index * record + 4..(index + 1) * record];
            payload
                .chunks_exact(4)
                .map(|chunk| i32::from_le_bytes(chunk.try_into().expect("component")))
                .collect()
        })
        .collect()
}
