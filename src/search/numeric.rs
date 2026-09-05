//! Format-v1 Search vector validation, distance, and rotation semantics.
//!
//! Format v1 deliberately uses scalar IEEE-754 operations and a fixed seeded
//! rotation. These steps are persistent protocol: changing their order,
//! precision, constants, or random-word consumption would change stored
//! RaBitQ codes even if the resulting vectors remained mathematically close.

use std::cmp::Ordering;
use std::mem::size_of;

use crate::api::{Error, ErrorKind, MAX_DIMENSION, Metric, Result};

/// Format-v1 Givens rotations make exactly three independent pairing passes.
const ROTATION_ROUNDS: usize = 3;
/// The format-fixed nearest f32 representation of `1 / sqrt(2)`.
const ROTATION_COEFFICIENT: f32 = f32::from_bits(0x3f35_04f3);
/// One ChaCha block contains sixteen little-endian 32-bit output words.
const CHACHA_WORDS_PER_BLOCK: usize = 16;
/// The standard ChaCha "expand 32-byte k" state prefix.
const CHACHA_CONSTANTS: [u32; 4] = [0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574];

/// An exact distance and its deterministic ranking key.
///
/// L2 orders candidates by squared distance to avoid an unnecessary square
/// root in comparisons, but exposes Euclidean distance to the caller. The
/// other metrics use the same value for both fields.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ExactDistance {
    ranking: f64,
    distance: f64,
}

impl ExactDistance {
    /// Returns the value used to order exact candidates.
    pub(crate) const fn ranking(self) -> f64 {
        self.ranking
    }

    /// Returns the caller-visible exact distance.
    pub(crate) const fn distance(self) -> f64 {
        self.distance
    }
}

/// The format-v1 scalar vector kernel for one Logical Index.
#[derive(Clone, Debug)]
pub(crate) struct VectorKernel {
    dimension: usize,
    metric: Metric,
    rotation: Rotation,
}

impl VectorKernel {
    /// Builds the kernel and its deterministic rotation schedule.
    pub(crate) fn new(dimension: usize, metric: Metric, rotation_seed: [u8; 32]) -> Result<Self> {
        validate_dimension(dimension)?;
        Ok(Self {
            dimension,
            metric,
            rotation: Rotation::new(dimension, rotation_seed)?,
        })
    }

    /// Returns the configured vector dimension.
    pub(crate) const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns whether this kernel uses cosine routing semantics.
    pub(crate) const fn is_cosine(&self) -> bool {
        matches!(self.metric, Metric::Cosine)
    }

    /// Normalizes a finite routing-space centroid for cosine routing.
    ///
    /// Leaf records are normalized by [`Self::preprocess`], but a split
    /// centroid is their arithmetic mean and therefore needs this second
    /// normalization step. Zero centroids remain zero so degenerate internal
    /// partitions retain a deterministic neutral route.
    pub(crate) fn normalize_centroid(&self, centroid: &[f32]) -> Result<Box<[f32]>> {
        validate_vector(centroid, self.dimension, VectorSource::Persistent)?;
        if !self.is_cosine() {
            return Ok(centroid.to_vec().into_boxed_slice());
        }
        let norm = vector_norm(centroid, VectorSource::Persistent)?;
        if norm == 0.0 {
            return Ok(centroid.to_vec().into_boxed_slice());
        }
        let mut normalized = allocate_vec(self.dimension)?;
        push_normalized(&mut normalized, centroid, norm, VectorSource::Persistent)?;
        Ok(normalized.into_boxed_slice())
    }

    /// Validates, metric-preprocesses, and rotates a caller vector.
    ///
    /// Cosine is normalized with scalar f64 accumulation before conversion
    /// back to f32. L2 and inner product retain their original scale. Rotation
    /// then uses only format-fixed f32 steps because these bytes feed the
    /// persistent RaBitQ codec.
    pub(crate) fn preprocess(&self, vector: &[f32]) -> Result<Box<[f32]>> {
        validate_vector(vector, self.dimension, VectorSource::Caller)?;

        let mut processed = allocate_vec(self.dimension)?;
        match self.metric {
            Metric::Cosine => {
                let norm = vector_norm(vector, VectorSource::Caller)?;
                if norm == 0.0 {
                    return Err(vector_error(VectorSource::Caller));
                }
                push_normalized(&mut processed, vector, norm, VectorSource::Caller)?;
            }
            Metric::L2 | Metric::InnerProduct => processed.extend_from_slice(vector),
        }

        self.rotation
            .apply_in_place(&mut processed, VectorSource::Caller)?;
        Ok(processed.into_boxed_slice())
    }

    /// Computes the deterministic routing distance from a preprocessed routing
    /// vector to a persistent routing centroid; smaller is nearer.
    ///
    /// L2 accumulates squared distance in f64, which ranks identically to the
    /// Euclidean exact distance. Inner product ranks by the negated f64 dot
    /// product, exactly its distance definition. Cosine ranks by the negated
    /// dot product for cosine: both the preprocessed records and the
    /// persisted split centroids are unit-norm. A zero centroid has no
    /// direction and contributes a neutral distance of zero, preserving a
    /// total deterministic ordering for degenerate split states.
    ///
    /// Every input component is validated finite and the bounded dimension
    /// caps the f64 accumulations far below overflow, so a finite result is
    /// guaranteed. A malformed persistent centroid is `Corruption`.
    pub(crate) fn routing_distance(&self, routing: &[f32], centroid: &[f32]) -> Result<f64> {
        validate_vector(routing, self.dimension, VectorSource::Caller)?;
        validate_vector(centroid, self.dimension, VectorSource::Persistent)?;

        match self.metric {
            Metric::L2 => Ok(squared_l2(routing, centroid)),
            Metric::Cosine | Metric::InnerProduct => Ok(-dot_product(routing, centroid)),
        }
    }

    /// Computes the exact scalar-f64 distance to one committed Vector Record.
    ///
    /// The query is caller-owned, whereas the record is decoded persistent
    /// state. That distinction determines whether malformed numeric input is
    /// `InvalidArgument` or `Corruption`.
    pub(crate) fn exact_distance(&self, query: &[f32], record: &[f32]) -> Result<ExactDistance> {
        validate_vector(query, self.dimension, VectorSource::Caller)?;
        validate_vector(record, self.dimension, VectorSource::Persistent)?;

        match self.metric {
            Metric::L2 => {
                let squared = squared_l2(query, record);
                finite_exact_distance(squared, squared.sqrt())
            }
            Metric::InnerProduct => {
                let distance = -dot_product(query, record);
                finite_exact_distance(distance, distance)
            }
            Metric::Cosine => {
                let query_norm = vector_norm(query, VectorSource::Caller)?;
                if query_norm == 0.0 {
                    return Err(vector_error(VectorSource::Caller));
                }
                let record_norm = vector_norm(record, VectorSource::Persistent)?;
                if record_norm == 0.0 {
                    return Err(vector_error(VectorSource::Persistent));
                }

                let distance = 1.0 - dot_product(query, record) / (query_norm * record_norm);
                finite_exact_distance(distance, distance)
            }
        }
    }
}

/// Identifies which contract owns a vector so failures are classified safely.
#[derive(Clone, Copy)]
enum VectorSource {
    Caller,
    Persistent,
}

fn validate_dimension(dimension: usize) -> Result<()> {
    if !(1..=MAX_DIMENSION).contains(&dimension) {
        return Err(Error::invalid_argument());
    }

    // Prove the simultaneous preprocessing vector and Fisher-Yates workspace
    // fit usize before any allocation. The format cap keeps the real bound
    // small, while checked arithmetic preserves the invariant on all targets.
    dimension
        .checked_mul(size_of::<f32>())
        .and_then(|bytes| bytes.checked_add(dimension.checked_mul(size_of::<usize>())?))
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
    Ok(())
}

fn validate_vector(vector: &[f32], dimension: usize, source: VectorSource) -> Result<()> {
    if vector.len() != dimension || vector.iter().any(|component| !component.is_finite()) {
        return Err(vector_error(source));
    }
    Ok(())
}

fn vector_error(source: VectorSource) -> Error {
    match source {
        VectorSource::Caller => Error::invalid_argument(),
        VectorSource::Persistent => Error::new(ErrorKind::Corruption),
    }
}

/// Allocates exact capacity for one bounded workspace, classifying allocation
/// failure as LimitExceeded.
fn allocate_vec<T>(capacity: usize) -> Result<Vec<T>> {
    let mut vector = Vec::new();
    vector
        .try_reserve_exact(capacity)
        .map_err(|error| Error::with_source(ErrorKind::LimitExceeded, error))?;
    Ok(vector)
}

/// Pushes each component divided by `norm`, converting back to f32 per
/// component. The scalar f64 division order is persistent format; a
/// non-finite result fails closed by `source`.
fn push_normalized(
    output: &mut Vec<f32>,
    vector: &[f32],
    norm: f64,
    source: VectorSource,
) -> Result<()> {
    for &component in vector {
        let normalized = (f64::from(component) / norm) as f32;
        if !normalized.is_finite() {
            return Err(vector_error(source));
        }
        output.push(normalized);
    }
    Ok(())
}

fn dot_product(left: &[f32], right: &[f32]) -> f64 {
    let mut dot = 0.0;
    for index in 0..left.len() {
        dot += f64::from(left[index]) * f64::from(right[index]);
    }
    dot
}

fn squared_l2(left: &[f32], right: &[f32]) -> f64 {
    let mut squared = 0.0;
    for index in 0..left.len() {
        let difference = f64::from(left[index]) - f64::from(right[index]);
        squared += difference * difference;
    }
    squared
}

fn vector_norm(vector: &[f32], source: VectorSource) -> Result<f64> {
    // The order is part of the scalar-f64 protocol. Do not replace this with a
    // parallel reduction or f32/SIMD kernel without accounting for rounding.
    let mut squared = 0.0;
    for &component in vector {
        let component = f64::from(component);
        squared += component * component;
    }
    let norm = squared.sqrt();
    if !norm.is_finite() {
        return Err(vector_error(source));
    }
    Ok(norm)
}

fn finite_exact_distance(ranking: f64, distance: f64) -> Result<ExactDistance> {
    if !ranking.is_finite() || !distance.is_finite() {
        return Err(Error::invalid_argument());
    }
    Ok(ExactDistance { ranking, distance })
}

/// Orders two finite ranking values so `-0.0` and `0.0` tie and fall through
/// to the deterministic Record ID tie-breaker.
pub(crate) fn compare_finite(left: f64, right: f64) -> Ordering {
    if left < right {
        Ordering::Less
    } else if left > right {
        Ordering::Greater
    } else {
        Ordering::Equal
    }
}

/// A reusable schedule for the format-v1 seeded orthogonal transformation.
///
/// Each round stores disjoint pairs from one Fisher-Yates permutation. An odd
/// final index is absent from the pair list and therefore unchanged in that
/// round. Keeping the schedule avoids regenerating it per vector.
#[derive(Clone, Debug)]
struct Rotation {
    dimension: usize,
    rounds: [Box<[[usize; 2]]>; ROTATION_ROUNDS],
}

impl Rotation {
    fn new(dimension: usize, seed: [u8; 32]) -> Result<Self> {
        let pairs_per_round = dimension / 2;
        pairs_per_round
            .checked_mul(ROTATION_ROUNDS)
            .and_then(|pairs| pairs.checked_mul(size_of::<[usize; 2]>()))
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;

        let mut random = ChaCha8::new(seed);
        let first = generate_pairs(dimension, &mut random)?;
        let second = generate_pairs(dimension, &mut random)?;
        let third = generate_pairs(dimension, &mut random)?;
        Ok(Self {
            dimension,
            rounds: [first, second, third],
        })
    }

    fn apply_in_place(&self, vector: &mut [f32], source: VectorSource) -> Result<()> {
        if vector.len() != self.dimension {
            return Err(vector_error(source));
        }

        for round in &self.rounds {
            for &[left, right] in round.iter() {
                let x = vector[left];
                let y = vector[right];
                // Keep addition/subtraction and multiplication as distinct f32
                // steps. Their IEEE-754 rounding order is persistent format.
                let sum = x + y;
                let difference = x - y;
                let rotated_left = sum * ROTATION_COEFFICIENT;
                let rotated_right = difference * ROTATION_COEFFICIENT;
                if !rotated_left.is_finite() || !rotated_right.is_finite() {
                    return Err(vector_error(source));
                }
                vector[left] = rotated_left;
                vector[right] = rotated_right;
            }
        }
        Ok(())
    }
}

fn generate_pairs(dimension: usize, random: &mut ChaCha8) -> Result<Box<[[usize; 2]]>> {
    let mut permutation = allocate_vec(dimension)?;
    permutation.extend(0..dimension);

    // Descending Fisher-Yates consumes one unbiased bounded draw per suffix.
    // A fresh identity permutation is generated for every rotation round,
    // while the ChaCha stream itself continues across rounds.
    for bound in (2..=dimension).rev() {
        let selected = random.bounded(bound)?;
        permutation.swap(bound - 1, selected);
    }

    let pair_count = dimension / 2;
    let mut pairs = allocate_vec(pair_count)?;
    for pair in permutation.chunks_exact(2) {
        pairs.push([pair[0], pair[1]]);
    }
    Ok(pairs.into_boxed_slice())
}

/// Minimal format-v1 ChaCha8 word stream.
///
/// ChaCha8 is the eight-round variant of the ChaCha stream cipher. KTANN uses
/// it only as a deterministic pseudorandom generator: the persisted 32-byte
/// seed is the key, the 64-bit block counter starts at zero, and the 64-bit
/// stream identifier is zero. Implementing the small word stream here makes
/// persistent output independent of an external RNG crate's API or defaults.
#[derive(Clone)]
struct ChaCha8 {
    key: [u32; 8],
    block_counter: u64,
    words: [u32; CHACHA_WORDS_PER_BLOCK],
    next_word: usize,
}

impl ChaCha8 {
    fn new(seed: [u8; 32]) -> Self {
        // The format interprets each consecutive key word as little-endian.
        let mut key = [0; 8];
        for (word, bytes) in key.iter_mut().zip(seed.chunks_exact(4)) {
            *word = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        }
        Self {
            key,
            block_counter: 0,
            words: [0; CHACHA_WORDS_PER_BLOCK],
            next_word: CHACHA_WORDS_PER_BLOCK,
        }
    }

    fn bounded(&mut self, bound: usize) -> Result<usize> {
        let bound = u32::try_from(bound).map_err(|_| Error::new(ErrorKind::LimitExceeded))?;
        // Rejection sampling avoids the modulo bias from 2^32 not being evenly
        // divisible by every Fisher-Yates bound.
        let zone = (1_u64 << u32::BITS) / u64::from(bound) * u64::from(bound);
        loop {
            let value = u64::from(self.next_u32()?);
            if value < zone {
                return usize::try_from(value % u64::from(bound))
                    .map_err(|_| Error::new(ErrorKind::LimitExceeded));
            }
        }
    }

    fn next_u32(&mut self) -> Result<u32> {
        if self.next_word == CHACHA_WORDS_PER_BLOCK {
            self.refill()?;
        }
        let word = self.words[self.next_word];
        self.next_word += 1;
        Ok(word)
    }

    fn refill(&mut self) -> Result<()> {
        // State words 12..=13 are the little-endian 64-bit block counter;
        // 14..=15 are the fixed-zero 64-bit stream identifier.
        let mut state = [0; CHACHA_WORDS_PER_BLOCK];
        state[..4].copy_from_slice(&CHACHA_CONSTANTS);
        state[4..12].copy_from_slice(&self.key);
        state[12] = self.block_counter as u32;
        state[13] = (self.block_counter >> u32::BITS) as u32;

        let mut working = state;
        // Four double rounds are eight ChaCha rounds. Each double round applies
        // the four column quarter-rounds followed by four diagonal ones.
        for _ in 0..4 {
            quarter_round(&mut working, 0, 4, 8, 12);
            quarter_round(&mut working, 1, 5, 9, 13);
            quarter_round(&mut working, 2, 6, 10, 14);
            quarter_round(&mut working, 3, 7, 11, 15);
            quarter_round(&mut working, 0, 5, 10, 15);
            quarter_round(&mut working, 1, 6, 11, 12);
            quarter_round(&mut working, 2, 7, 8, 13);
            quarter_round(&mut working, 3, 4, 9, 14);
        }
        for index in 0..CHACHA_WORDS_PER_BLOCK {
            self.words[index] = working[index].wrapping_add(state[index]);
        }
        self.next_word = 0;
        self.block_counter = self
            .block_counter
            .checked_add(1)
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        Ok(())
    }
}

fn quarter_round(
    state: &mut [u32; CHACHA_WORDS_PER_BLOCK],
    a: usize,
    b: usize,
    c: usize,
    d: usize,
) {
    // ChaCha's ARX primitive: wrapping addition, XOR, then fixed rotation.
    state[a] = state[a].wrapping_add(state[b]);
    state[d] ^= state[a];
    state[d] = state[d].rotate_left(16);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] ^= state[c];
    state[b] = state[b].rotate_left(12);
    state[a] = state[a].wrapping_add(state[b]);
    state[d] ^= state[a];
    state[d] = state[d].rotate_left(8);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] ^= state[c];
    state[b] = state[b].rotate_left(7);
}

#[cfg(test)]
mod tests {
    use super::*;

    const SEED: [u8; 32] = [
        0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
        0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c, 0x1d,
        0x1e, 0x1f,
    ];

    fn assert_kind<T>(result: Result<T>, expected: ErrorKind) {
        assert_eq!(result.err().map(|error| error.kind()), Some(expected));
    }

    #[test]
    fn kernel_rejects_dimensions_outside_the_format_bounds() {
        assert_kind(
            VectorKernel::new(0, Metric::L2, SEED),
            ErrorKind::InvalidArgument,
        );
        assert_kind(
            VectorKernel::new(MAX_DIMENSION + 1, Metric::L2, SEED),
            ErrorKind::InvalidArgument,
        );
    }

    #[test]
    fn preprocessing_rejects_bad_caller_vectors() {
        let kernel = VectorKernel::new(2, Metric::L2, SEED).unwrap();
        assert_kind(kernel.preprocess(&[1.0]), ErrorKind::InvalidArgument);
        assert_kind(
            kernel.preprocess(&[1.0, f32::INFINITY]),
            ErrorKind::InvalidArgument,
        );
        assert_kind(
            kernel.preprocess(&[f32::MAX, f32::MAX]),
            ErrorKind::InvalidArgument,
        );
    }

    #[test]
    fn cosine_preprocessing_normalizes_before_rotation() {
        let kernel = VectorKernel::new(3, Metric::Cosine, SEED).unwrap();
        let processed = kernel.preprocess(&[3.0, 4.0, 0.0]).unwrap();
        let norm = processed
            .iter()
            .map(|&component| f64::from(component).powi(2))
            .sum::<f64>()
            .sqrt();
        assert!((norm - 1.0).abs() < 2.0e-7, "norm was {norm}");
        assert_kind(kernel.preprocess(&[0.0; 3]), ErrorKind::InvalidArgument);
    }

    #[test]
    fn exact_l2_uses_squared_ranking_and_euclidean_output() {
        let kernel = VectorKernel::new(3, Metric::L2, SEED).unwrap();
        let exact = kernel
            .exact_distance(&[1.0, 2.0, 3.0], &[4.0, 6.0, 3.0])
            .unwrap();
        assert_eq!(exact.ranking(), 25.0);
        assert_eq!(exact.distance(), 5.0);
    }

    #[test]
    fn exact_inner_product_is_negated() {
        let kernel = VectorKernel::new(3, Metric::InnerProduct, SEED).unwrap();
        let exact = kernel
            .exact_distance(&[1.0, -2.0, 3.0], &[4.0, 5.0, -6.0])
            .unwrap();
        assert_eq!(exact.ranking(), 24.0);
        assert_eq!(exact.distance(), 24.0);
    }

    #[test]
    fn exact_cosine_recomputes_f64_norms_without_clamping() {
        let kernel = VectorKernel::new(2, Metric::Cosine, SEED).unwrap();
        let same = kernel.exact_distance(&[3.0, 4.0], &[6.0, 8.0]).unwrap();
        assert_eq!(same.distance(), 0.0);

        let opposite = kernel.exact_distance(&[3.0, 4.0], &[-6.0, -8.0]).unwrap();
        assert_eq!(opposite.distance(), 2.0);
    }

    #[test]
    fn exact_distance_distinguishes_caller_errors_from_corruption() {
        let kernel = VectorKernel::new(2, Metric::Cosine, SEED).unwrap();
        assert_kind(
            kernel.exact_distance(&[0.0, 0.0], &[1.0, 0.0]),
            ErrorKind::InvalidArgument,
        );
        assert_kind(
            kernel.exact_distance(&[1.0, 0.0], &[0.0, 0.0]),
            ErrorKind::Corruption,
        );
        assert_kind(
            kernel.exact_distance(&[1.0, 0.0], &[f32::NAN, 0.0]),
            ErrorKind::Corruption,
        );
    }

    #[test]
    fn exact_distances_match_independent_numeric_oracles() {
        let mut state = 0x8f3a_2b19_u32;
        for dimension in 1..=32 {
            let mut query = Vec::with_capacity(dimension);
            let mut record = Vec::with_capacity(dimension);
            for _ in 0..dimension {
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                query.push((state as i32 % 20_001) as f32 / 100.0);
                state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                record.push((state as i32 % 20_001) as f32 / 100.0);
            }

            for metric in [Metric::L2, Metric::InnerProduct, Metric::Cosine] {
                let kernel = VectorKernel::new(dimension, metric, SEED).unwrap();
                let exact = kernel.exact_distance(&query, &record).unwrap();
                let query64: Vec<f64> = query.iter().map(|&value| f64::from(value)).collect();
                let record64: Vec<f64> = record.iter().map(|&value| f64::from(value)).collect();
                let oracle = match metric {
                    Metric::L2 => query64
                        .iter()
                        .zip(&record64)
                        .map(|(left, right)| (left - right).powi(2))
                        .sum::<f64>()
                        .sqrt(),
                    Metric::InnerProduct => -query64
                        .iter()
                        .zip(&record64)
                        .map(|(left, right)| left * right)
                        .sum::<f64>(),
                    Metric::Cosine => {
                        let dot = query64
                            .iter()
                            .zip(&record64)
                            .map(|(left, right)| left * right)
                            .sum::<f64>();
                        let query_norm = query64
                            .iter()
                            .map(|value| value.powi(2))
                            .sum::<f64>()
                            .sqrt();
                        let record_norm = record64
                            .iter()
                            .map(|value| value.powi(2))
                            .sum::<f64>()
                            .sqrt();
                        1.0 - dot / (query_norm * record_norm)
                    }
                };
                assert_eq!(exact.distance(), oracle);
            }
        }
    }

    #[test]
    fn rotation_has_a_cross_process_golden_vector() {
        let kernel = VectorKernel::new(7, Metric::L2, SEED).unwrap();
        let rotated = kernel
            .preprocess(&[1.0, -2.0, 3.5, 0.25, -4.0, 8.0, -0.5])
            .unwrap();
        let expected = [
            f32::from_bits(0x404b_a591),
            f32::from_bits(0x4078_e6ce),
            f32::from_bits(0xc0cb_a590),
            f32::from_bits(0xc043_a591),
            f32::from_bits(0x3fff_876d),
            f32::from_bits(0x4080_7367),
            f32::from_bits(0xbfc5_04f4),
        ];
        assert_eq!(&*rotated, &expected);
    }

    #[test]
    fn rotation_approximately_preserves_norm_for_bounded_vectors() {
        let kernel = VectorKernel::new(64, Metric::L2, SEED).unwrap();
        let vector: Vec<f32> = (0..64).map(|index| index as f32 - 31.5).collect();
        let rotated = kernel.preprocess(&vector).unwrap();
        let original_norm = vector
            .iter()
            .map(|&value| f64::from(value).powi(2))
            .sum::<f64>();
        let rotated_norm = rotated
            .iter()
            .map(|&value| f64::from(value).powi(2))
            .sum::<f64>();
        let relative_error = (original_norm - rotated_norm).abs() / original_norm;
        assert!(
            relative_error < 5.0e-7,
            "relative error was {relative_error}"
        );
    }

    #[test]
    fn odd_dimensions_leave_one_component_unpaired_each_round() {
        let rotation = Rotation::new(5, SEED).unwrap();
        assert!(rotation.rounds.iter().all(|round| round.len() == 2));
        for round in &rotation.rounds {
            let mut indexes: Vec<usize> = round.iter().flatten().copied().collect();
            indexes.sort_unstable();
            indexes.dedup();
            assert_eq!(indexes.len(), 4);
        }
    }

    #[test]
    fn rejection_sampling_has_a_rotation_golden() {
        // This seed makes round 3 reject word 4_294_961_629 at bound 13_451.
        // Hashing the full maximum-dimension result keeps the golden compact
        // while detecting a failure to discard that out-of-zone word.
        let mut seed = [0; 32];
        seed[..8].copy_from_slice(&6_u64.to_le_bytes());
        let kernel = VectorKernel::new(MAX_DIMENSION, Metric::L2, seed).unwrap();
        let vector: Vec<f32> = (0..MAX_DIMENSION)
            .map(|index| (index % 257) as f32 - 128.0)
            .collect();
        let rotated = kernel.preprocess(&vector).unwrap();

        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for component in rotated {
            for byte in component.to_bits().to_le_bytes() {
                hash ^= u64::from(byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        assert_eq!(hash, 0x2398_08d9_e8ca_5477);
    }

    #[test]
    fn routing_distance_uses_squared_l2_ranking() {
        let kernel = VectorKernel::new(3, Metric::L2, SEED).unwrap();
        let distance = kernel
            .routing_distance(&[1.0, 2.0, 3.0], &[4.0, 6.0, 3.0])
            .unwrap();
        assert_eq!(distance, 25.0);
    }

    #[test]
    fn routing_distance_negates_dot_product_for_inner_product() {
        let kernel = VectorKernel::new(3, Metric::InnerProduct, SEED).unwrap();
        let distance = kernel
            .routing_distance(&[1.0, -2.0, 3.0], &[4.0, 5.0, -6.0])
            .unwrap();
        assert_eq!(distance, 24.0);
    }

    #[test]
    fn cosine_centroid_normalization_is_scale_invariant() {
        let kernel = VectorKernel::new(2, Metric::Cosine, SEED).unwrap();
        let normalized = kernel.normalize_centroid(&[6.0, 8.0]).unwrap();
        assert_eq!(&*normalized, &[0.6, 0.8]);

        let opposite = kernel.normalize_centroid(&[-6.0, -8.0]).unwrap();
        assert_eq!(&*opposite, &[-0.6, -0.8]);
    }

    #[test]
    fn cosine_routing_accepts_a_zero_centroid() {
        let kernel = VectorKernel::new(2, Metric::Cosine, SEED).unwrap();
        let distance = kernel.routing_distance(&[0.6, 0.8], &[0.0, 0.0]).unwrap();
        assert_eq!(distance, 0.0);
    }

    #[test]
    fn routing_distance_distinguishes_caller_errors_from_corruption() {
        let kernel = VectorKernel::new(2, Metric::L2, SEED).unwrap();
        assert_kind(
            kernel.routing_distance(&[1.0], &[1.0, 0.0]),
            ErrorKind::InvalidArgument,
        );
        assert_kind(
            kernel.routing_distance(&[1.0, f32::NAN], &[1.0, 0.0]),
            ErrorKind::InvalidArgument,
        );
        assert_kind(
            kernel.routing_distance(&[1.0, 0.0], &[1.0]),
            ErrorKind::Corruption,
        );
        assert_kind(
            kernel.routing_distance(&[1.0, 0.0], &[f32::INFINITY, 0.0]),
            ErrorKind::Corruption,
        );
    }
}
