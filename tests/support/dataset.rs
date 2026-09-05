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
//! - `file:NAME[:N]` — a fixture from `tests/datadriven/data/`; `.fvecs` and
//!   `.idx3-ubyte` (MNIST IDX) formats are supported. With `:N`, only the
//!   first N vectors are taken. The fixture dimension must equal the index
//!   dimension.
//!
//! Specifications with missing, extra, or non-numeric fields panic.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use super::fixtures;
use bytes::Bytes;

/// A replayable xorshift64 generator for model and fault histories; the
/// printed seed reproduces a failure.
pub struct Rng(pub u64);

impl Rng {
    /// Returns the next pseudo-random word.
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Returns the next word modulo `bound`.
    pub fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

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
        let mut vectors = read_fixture(parts.next().unwrap_or(""), dimension, spec);
        if let Some(part) = parts.next() {
            let take: usize = part
                .parse()
                .unwrap_or_else(|_| panic!("bad dataset spec `{spec}`"));
            assert!(
                take > 0 && take <= vectors.len(),
                "bad take count in dataset spec `{spec}`"
            );
            vectors.truncate(take);
        }
        assert!(parts.next().is_none(), "bad dataset spec `{spec}`");
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
    assert!(parts.next().is_none(), "bad dataset spec `{spec}`");

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

/// The directory holding the checked-in real-dataset fixtures, relative to
/// the crate manifest.
const FIXTURE_DIR: &str = "tests/datadriven/data";

fn fixture_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(FIXTURE_DIR)
}

/// Loads the vector fixture named by a `file:` spec and checks that its
/// dimension equals the index dimension.
fn read_fixture(name: &str, dimension: usize, spec: &str) -> Vec<Arc<[f32]>> {
    let vectors = fixtures::read_vectors(&fixture_dir(), name);
    let found = vectors.first().map_or(0, |vector| vector.len());
    assert!(
        found == dimension,
        "fixture dimension {found} != index dimension {dimension} in spec `{spec}`"
    );
    vectors
}

/// Reads an `.ivecs` fixture (the `.fvecs` layout with i32 components); used
/// to check the brute-force oracle against published ground truth.
#[must_use]
pub fn read_ivecs_fixture(name: &str) -> Vec<Vec<i32>> {
    fixtures::read_ivecs(&fixture_dir(), name)
}
