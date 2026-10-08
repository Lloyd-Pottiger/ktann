//! Bounded, synchronous construction of one tree in an exclusive scratch directory.
//!
//! This is the pure construction stage, not an index publication API. Callers
//! stream one Tree Key's original vectors and consume final partition plans.
//! It does no backend IO and must run on a blocking worker, not an async executor.

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use bytes::Bytes;
use xxhash_rust::xxh3::xxh3_128_with_seed;

use crate::api::{Error, ErrorKind, Metric, PartitionKey, Result, validate_id};
use crate::maintenance::training::train_sample;
use crate::search::numeric::VectorKernel;

const BUFFER_BYTES: usize = 8 * 1024;
const MAX_ID_BYTES: usize = crate::api::MAX_RECORD_ID_BYTES;
const SAMPLE_SEED: u64 = 0x6b74_616e_6e62_756c;

/// Version of the grouping, sampling, accumulation and partition-allocation protocol.
pub const CONSTRUCTION_VERSION: u32 = 1;

/// Bounds for the single-tree construction stage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ConstructionOptions {
    /// Serving split threshold; non-root groups are within min/max occupancy.
    pub max_partition_entries: u32,
    /// Minimum non-root group size; twice this must not exceed the maximum.
    pub min_partition_entries: u32,
    /// Maximum deterministic training sample size.
    pub sample_items: usize,
    /// Conservative bound for algorithm-owned working buffers, excluding caller IO.
    pub memory_bytes: usize,
    /// Maximum simultaneous scratch-file bytes, including merge inputs and output.
    pub scratch_bytes: u64,
}

/// One original vector, identified by its canonical Record ID.
pub struct ConstructionRecord {
    /// Record ID, unique across the containing Logical Index.
    pub id: Bytes,
    /// Original vector; metric preprocessing and rotation happen once here.
    pub vector: Box<[f32]>,
}

/// One final bounded partition. Leaf entries are Record IDs; internal entries
/// are big-endian Partition Keys. Centroids are in serving routing space.
pub struct PartitionPlan {
    /// Tree-local identity; the root is always key 1.
    pub key: PartitionKey,
    /// Leaves have level 1; every child is exactly one level below its parent.
    pub level: u32,
    /// Full-f32 mean of all assigned entries; the root does not persist this value.
    pub centroid: Box<[f32]>,
    /// Final assignments, in canonical identity order.
    pub entries: Vec<Bytes>,
}

/// Counts and resource evidence from a completed pure construction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ConstructionReport {
    /// Number of input records.
    pub records: u64,
    /// Number of emitted serving partitions.
    pub partitions: u64,
    /// Largest allocated partition identity, or zero for empty input.
    pub partition_high_water: u64,
    /// Peak simultaneous scratch bytes.
    pub peak_scratch_bytes: u64,
    /// All scratch writes, including repeated sorting and grouping passes.
    pub scratch_written_bytes: u64,
}

/// Constructs a bottom-up tree with bounded binary external merges and samples.
///
/// `directory` must not exist. The caller owns its reclamation, including after
/// failure. Final plans are emitted synchronously in child-before-parent order;
/// consumers must not publish them until this function and exact validation
/// complete. Duplicate Record IDs fail before any partition is emitted.
/// Cross-Tree-Key duplicate detection belongs to the input preparation stage.
///
/// The rotation seed must be the target Logical Index's persisted seed. Options
/// and this algorithm's version must be sealed in the enclosing build descriptor.
pub fn construct_tree(
    directory: &Path,
    dimension: usize,
    metric: Metric,
    rotation_seed: [u8; 32],
    options: ConstructionOptions,
    records: impl IntoIterator<Item = Result<ConstructionRecord>>,
    mut emit: impl FnMut(PartitionPlan) -> Result<()>,
) -> Result<ConstructionReport> {
    let kernel = VectorKernel::new(dimension, metric, rotation_seed)?;
    let sort_rows = validate_options(dimension, options)?;
    fs::create_dir(directory).map_err(io_error)?;
    let mut work = Workspace {
        directory: directory.to_owned(),
        next_file: 0,
        bytes: 0,
        report: ConstructionReport::default(),
        options,
        kernel,
        sort_rows,
        // A rank key is 20 bytes. Projection uses at most half the bytes of
        // even the shortest vector row, bounding retained-input scratch use.
        project_splits: 12 + 1 + dimension * 4 >= 2 * 20,
    };
    let mut input = work.writer()?;
    for record in records {
        let record = record?;
        validate_id(&record.id)?;
        let vector = work.kernel.preprocess(&record.vector)?;
        work.append(
            &mut input,
            &Row {
                order: 0.0,
                id: record.id,
                vector,
            },
        )?;
    }
    let input = work.finish(input)?;
    work.report.records = input.count;
    let input = work.sort(input, |_| Ok(()))?;
    let mut previous = None;
    let mut reader = work.reader(&input)?;
    while let Some(row) = reader.next()? {
        if previous.as_ref() == Some(&row.id) {
            return Err(Error::new(ErrorKind::RecordAlreadyExists));
        }
        previous = Some(row.id);
    }
    drop(reader);
    if input.count == 0 {
        work.remove(input)?;
        return Ok(work.report);
    }
    let mut level = 1;
    let mut input = input;
    loop {
        let root = input.count <= u64::from(options.max_partition_entries);
        let mut parents = work.writer()?;
        work.group(input, level, root, &mut parents, &mut emit)?;
        let parents = work.finish(parents)?;
        if root {
            work.remove(parents)?;
            return Ok(work.report);
        }
        input = parents;
        level = level.checked_add(1).ok_or_else(limit)?;
    }
}

fn io_error(_: std::io::Error) -> Error {
    Error::new(ErrorKind::Backend)
}
fn limit() -> Error {
    Error::new(ErrorKind::LimitExceeded)
}
fn corrupt() -> Error {
    Error::new(ErrorKind::Corruption)
}

/// An immutable sorted-run descriptor; each file belongs to this invocation.
struct Run {
    path: PathBuf,
    count: u64,
    bytes: u64,
    vector_bytes: usize,
}
struct Output {
    run: Run,
    writer: BufWriter<File>,
}
struct Input {
    reader: BufReader<File>,
    remaining: u64,
    encoded_vector: Vec<u8>,
}
struct Row {
    order: f64,
    id: Bytes,
    vector: Box<[f32]>,
}

impl Row {
    fn compare(&self, other: &Self) -> Ordering {
        self.order
            .total_cmp(&other.order)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl Input {
    fn next(&mut self) -> Result<Option<Row>> {
        if self.remaining == 0 {
            let mut tail = [0];
            if self.reader.read(&mut tail).map_err(io_error)? != 0 {
                return Err(corrupt());
            }
            return Ok(None);
        }
        let mut order = [0; 8];
        let mut len = [0; 4];
        self.reader.read_exact(&mut order).map_err(|_| corrupt())?;
        self.reader.read_exact(&mut len).map_err(|_| corrupt())?;
        let len = u32::from_be_bytes(len) as usize;
        let order = f64::from_bits(u64::from_be_bytes(order));
        if len == 0 || len > MAX_ID_BYTES || !order.is_finite() {
            return Err(corrupt());
        }
        let mut id = vec![0; len];
        self.reader.read_exact(&mut id).map_err(|_| corrupt())?;
        // Read once per vector. Per-component reads multiply across every
        // external merge pass and make throughput depend on library inlining.
        self.reader
            .read_exact(&mut self.encoded_vector)
            .map_err(|_| corrupt())?;
        let vector: Vec<f32> = self
            .encoded_vector
            .as_chunks::<4>()
            .0
            .iter()
            .map(|bytes| f32::from_be_bytes(*bytes))
            .collect();
        if vector
            .iter()
            .fold(false, |invalid, value| invalid | !value.is_finite())
        {
            return Err(corrupt());
        }
        self.remaining -= 1;
        Ok(Some(Row {
            order,
            id: Bytes::from(id),
            vector: vector.into_boxed_slice(),
        }))
    }
}

struct Workspace {
    directory: PathBuf,
    next_file: u64,
    bytes: u64,
    report: ConstructionReport,
    options: ConstructionOptions,
    kernel: VectorKernel,
    sort_rows: usize,
    project_splits: bool,
}

impl Workspace {
    fn writer(&mut self) -> Result<Output> {
        self.writer_with_vector_bytes(self.kernel.dimension() * 4)
    }

    // Split keys carry no vector payload; their merge runs share the same quota.
    fn writer_with_vector_bytes(&mut self, vector_bytes: usize) -> Result<Output> {
        let path = self.directory.join(format!("{:016x}.run", self.next_file));
        self.next_file = self.next_file.checked_add(1).ok_or_else(limit)?;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(io_error)?;
        Ok(Output {
            run: Run {
                path,
                count: 0,
                bytes: 0,
                vector_bytes,
            },
            writer: BufWriter::with_capacity(BUFFER_BYTES, file),
        })
    }

    fn append(&mut self, output: &mut Output, row: &Row) -> Result<()> {
        let bytes = 12 + row.id.len() as u64 + 4 * row.vector.len() as u64;
        let total = self.bytes.checked_add(bytes).ok_or_else(limit)?;
        if total > self.options.scratch_bytes {
            return Err(limit());
        }
        output
            .writer
            .write_all(&row.order.to_bits().to_be_bytes())
            .map_err(io_error)?;
        output
            .writer
            .write_all(&(row.id.len() as u32).to_be_bytes())
            .map_err(io_error)?;
        output.writer.write_all(&row.id).map_err(io_error)?;
        for component in &row.vector {
            output
                .writer
                .write_all(&component.to_bits().to_be_bytes())
                .map_err(io_error)?;
        }
        output.run.count = output.run.count.checked_add(1).ok_or_else(limit)?;
        output.run.bytes = output.run.bytes.checked_add(bytes).ok_or_else(limit)?;
        self.bytes = total;
        self.report.peak_scratch_bytes = self.report.peak_scratch_bytes.max(total);
        self.report.scratch_written_bytes = self
            .report
            .scratch_written_bytes
            .checked_add(bytes)
            .ok_or_else(limit)?;
        Ok(())
    }

    fn finish(&self, mut output: Output) -> Result<Run> {
        output.writer.flush().map_err(io_error)?;
        // These intermediate runs are recomputable, not accepted durable artifacts.
        drop(output.writer);
        Ok(output.run)
    }

    fn reader(&self, run: &Run) -> Result<Input> {
        let file = File::open(&run.path).map_err(io_error)?;
        if file.metadata().map_err(io_error)?.len() != run.bytes {
            return Err(corrupt());
        }
        Ok(Input {
            reader: BufReader::with_capacity(BUFFER_BYTES, file),
            remaining: run.count,
            encoded_vector: vec![0; run.vector_bytes],
        })
    }

    fn remove(&mut self, run: Run) -> Result<()> {
        fs::remove_file(run.path).map_err(io_error)?;
        self.bytes = self.bytes.checked_sub(run.bytes).ok_or_else(corrupt)?;
        Ok(())
    }

    /// Binary carry merging bounds run inventory to at most 64 descriptors and
    /// opens only two inputs plus one output regardless of input cardinality.
    fn sort(&mut self, input: Run, prepare: impl FnMut(&mut Row) -> Result<()>) -> Result<Run> {
        let vector_bytes = input.vector_bytes;
        let runs = self.sort_runs(&input, vector_bytes, prepare)?;
        self.remove(input)?;
        self.merge_runs(runs, vector_bytes)
    }

    // Project directly into bounded sort buffers, without materializing a second
    // unsorted input. The caller controls when the original run can be removed.
    fn sort_runs(
        &mut self,
        input: &Run,
        vector_bytes: usize,
        mut prepare: impl FnMut(&mut Row) -> Result<()>,
    ) -> Result<Vec<Option<Run>>> {
        let mut reader = self.reader(input)?;
        let mut levels: Vec<Option<Run>> = Vec::new();
        loop {
            // A quota is a ceiling, not an instruction to reserve it all for a
            // small/empty input. Allocation failure remains a resource error.
            let capacity = self
                .sort_rows
                .min(usize::try_from(reader.remaining).unwrap_or(usize::MAX));
            let mut rows = Vec::new();
            rows.try_reserve_exact(capacity).map_err(|_| limit())?;
            while rows.len() < self.sort_rows {
                let Some(mut row) = reader.next()? else { break };
                prepare(&mut row)?;
                rows.push(row);
            }
            if rows.is_empty() {
                break;
            }
            rows.sort_unstable_by(Row::compare);
            let mut output = self.writer_with_vector_bytes(vector_bytes)?;
            for row in rows {
                self.append(&mut output, &row)?;
            }
            let mut run = self.finish(output)?;
            let mut level = 0;
            loop {
                if level == levels.len() {
                    levels.push(None);
                }
                match levels[level].take() {
                    Some(previous) => {
                        run = self.merge(previous, run)?;
                        level += 1;
                    }
                    None => {
                        levels[level] = Some(run);
                        break;
                    }
                }
            }
        }
        Ok(levels)
    }

    fn merge_runs(&mut self, levels: Vec<Option<Run>>, vector_bytes: usize) -> Result<Run> {
        let mut result = None;
        for run in levels.into_iter().flatten() {
            result = Some(match result {
                None => run,
                Some(previous) => self.merge(previous, run)?,
            });
        }
        match result {
            Some(run) => Ok(run),
            None => {
                let empty = self.writer_with_vector_bytes(vector_bytes)?;
                self.finish(empty)
            }
        }
    }

    fn merge(&mut self, left: Run, right: Run) -> Result<Run> {
        let mut l = self.reader(&left)?;
        let mut r = self.reader(&right)?;
        let mut a = l.next()?;
        let mut b = r.next()?;
        let mut output = self.writer_with_vector_bytes(left.vector_bytes)?;
        while a.is_some() || b.is_some() {
            let take_left = match (&a, &b) {
                (Some(a), Some(b)) => a.compare(b).is_le(),
                (Some(_), None) => true,
                _ => false,
            };
            if take_left {
                if let Some(row) = a.take() {
                    self.append(&mut output, &row)?;
                }
                a = l.next()?;
            } else {
                if let Some(row) = b.take() {
                    self.append(&mut output, &row)?;
                }
                b = r.next()?;
            }
        }
        drop((l, r));
        self.remove(left)?;
        self.remove(right)?;
        self.finish(output)
    }

    fn group(
        &mut self,
        input: Run,
        level: u32,
        root: bool,
        parents: &mut Output,
        emit: &mut impl FnMut(PartitionPlan) -> Result<()>,
    ) -> Result<()> {
        if input.count <= u64::from(self.options.max_partition_entries) {
            // Projected splits scatter stably from preflight ID order. Their
            // ordinals resolve ties exactly like IDs, and terminal means can
            // stream directly in canonical order. Tiny rows use the original
            // full-row sorter, whose children still need this final ID sort.
            let input = if root || self.project_splits {
                input
            } else {
                self.sort(input, |_| Ok(()))?
            };
            let key = if root {
                1
            } else {
                self.report
                    .partition_high_water
                    .max(1)
                    .checked_add(1)
                    .ok_or_else(limit)?
            };
            self.report.partition_high_water = self.report.partition_high_water.max(key);
            let key = PartitionKey::new(key)?;
            let mut reader = self.reader(&input)?;
            let mut entries = Vec::with_capacity(input.count as usize);
            let mut sums = vec![0.0_f64; self.kernel.dimension()];
            while let Some(row) = reader.next()? {
                let vector = if self.kernel.is_cosine() {
                    self.kernel.normalize_centroid(&row.vector)?
                } else {
                    row.vector
                };
                for (sum, value) in sums.iter_mut().zip(vector.iter()) {
                    *sum += f64::from(*value);
                }
                entries.push(row.id);
            }
            let mean: Box<[f32]> = sums
                .into_iter()
                .map(|sum| (sum / input.count as f64) as f32)
                .collect();
            let centroid = self.kernel.normalize_centroid(&mean)?;
            if !root {
                self.append(
                    parents,
                    &Row {
                        order: 0.0,
                        id: Bytes::copy_from_slice(&key.get().to_be_bytes()),
                        vector: centroid.clone(),
                    },
                )?;
            }
            emit(PartitionPlan {
                key,
                level,
                centroid,
                entries,
            })?;
            self.report.partitions = self.report.partitions.checked_add(1).ok_or_else(limit)?;
            drop(reader);
            self.remove(input)?;
            return Ok(());
        }
        let mut reader = self.reader(&input)?;
        let mut sample = BinaryHeap::new();
        while let Some(row) = reader.next()? {
            let item = Sample {
                hash: xxh3_128_with_seed(&row.id, SAMPLE_SEED),
                row,
            };
            if sample.len() < self.options.sample_items {
                sample.push(item);
            } else if sample.peek().is_some_and(|max| &item < max) {
                sample.pop();
                sample.push(item);
            }
        }
        drop(reader);
        let centroids = train_sample(
            &self.kernel,
            sample
                .into_iter()
                .map(|item| (item.row.id, item.row.vector))
                .collect(),
        )?;
        let half = input.count / 2;
        let kernel = self.kernel.clone();
        let distance_difference = |row: &Row| -> Result<f64> {
            let distance = kernel.routing_distance(&row.vector, centroids.left().components())?
                - kernel.routing_distance(&row.vector, centroids.right().components())?;
            if !distance.is_finite() {
                return Err(Error::invalid_argument());
            }
            Ok(distance)
        };
        let boundary = if self.project_splits {
            // ID order is invariant along stable scatters. Fixed-width ordinals
            // give exactly the same distance ties without copying arbitrary IDs.
            let mut ordinal = 0_u64;
            let runs = self.sort_runs(&input, 0, |row| {
                row.order = distance_difference(row)?;
                row.id = Bytes::copy_from_slice(&ordinal.to_be_bytes());
                row.vector = Box::default();
                ordinal += 1;
                Ok(())
            })?;
            let keys = self.merge_runs(runs, 0)?;
            let mut reader = self.reader(&keys)?;
            let mut boundary = None;
            for _ in 0..half {
                boundary = reader.next()?;
            }
            let boundary = boundary.ok_or_else(corrupt)?;
            let ordinal =
                u64::from_be_bytes(boundary.id.as_ref().try_into().map_err(|_| corrupt())?);
            drop(reader);
            self.remove(keys)?;
            Some((boundary.order, ordinal))
        } else {
            None
        };
        let input = if boundary.is_some() {
            input
        } else {
            self.sort(input, |row| {
                row.order = distance_difference(row)?;
                Ok(())
            })?
        };
        let mut reader = self.reader(&input)?;
        let mut left = self.writer()?;
        let mut right = self.writer()?;
        let mut ordinal = 0_u64;
        while let Some(mut row) = reader.next()? {
            let to_left = match boundary {
                Some((distance, cut)) => distance_difference(&row)?
                    .total_cmp(&distance)
                    .then_with(|| ordinal.cmp(&cut))
                    .is_le(),
                None => ordinal < half,
            };
            ordinal += 1;
            row.order = 0.0;
            self.append(if to_left { &mut left } else { &mut right }, &row)?;
        }
        drop(reader);
        self.remove(input)?;
        drop((kernel, centroids));
        let left = self.finish(left)?;
        let right = self.finish(right)?;
        self.group(left, level, false, parents, emit)?;
        self.group(right, level, false, parents, emit)
    }
}

struct Sample {
    hash: u128,
    row: Row,
}
impl PartialEq for Sample {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}
impl Eq for Sample {}
impl PartialOrd for Sample {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Sample {
    fn cmp(&self, other: &Self) -> Ordering {
        self.hash
            .cmp(&other.hash)
            .then_with(|| self.row.id.cmp(&other.row.id))
    }
}

/// Validates resource ceilings before a job reserves a name.
pub(crate) fn validate_options(dimension: usize, options: ConstructionOptions) -> Result<usize> {
    let row_bound = dimension
        .checked_mul(4)
        .and_then(|n| n.checked_add(MAX_ID_BYTES + 128))
        .ok_or_else(limit)?;
    let resident_rows = options
        .sample_items
        .max(options.max_partition_entries as usize)
        .checked_mul(8)
        .ok_or_else(limit)?;
    let reserved = row_bound
        .checked_mul(resident_rows)
        .and_then(|n| n.checked_add(16 * BUFFER_BYTES))
        // A sort's source reader can overlap both binary-merge readers. Each
        // owns one encoded vector buffer in addition to its decoded row.
        .and_then(|n| n.checked_add(3 * dimension * 4))
        .ok_or_else(limit)?;
    if options.min_partition_entries == 0
        || options
            .min_partition_entries
            .checked_mul(2)
            .is_none_or(|n| n > options.max_partition_entries)
        || options.sample_items < 2
        || options.memory_bytes <= reserved.checked_add(row_bound).ok_or_else(limit)?
        || options.scratch_bytes == 0
    {
        return Err(Error::invalid_argument());
    }
    Ok((options.memory_bytes - reserved) / row_bound)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::{BTreeMap, BTreeSet};
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

    static NEXT: AtomicU64 = AtomicU64::new(0);

    struct Directory(PathBuf);
    impl Directory {
        fn new() -> Self {
            Self(std::env::temp_dir().join(format!(
                    "ktann-construction-{}-{}-{}",
                    std::process::id(),
                    std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos(),
                    NEXT.fetch_add(1, AtomicOrdering::Relaxed)
                )))
        }
    }
    impl Drop for Directory {
        fn drop(&mut self) {
            if self.0.exists() {
                fs::remove_dir_all(&self.0).unwrap();
            }
        }
    }

    #[test]
    fn scratch_vector_reads_cross_buffers_and_reject_truncation_or_nonfinite_values() {
        let directory = Directory::new();
        fs::create_dir(&directory.0).unwrap();
        let path = directory.0.join("input.run");
        let dimension = BUFFER_BYTES / 4 + 1;
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&0.0_f64.to_be_bytes());
        bytes.extend_from_slice(&3_u32.to_be_bytes());
        bytes.extend_from_slice(b"row");
        for _ in 0..dimension {
            bytes.extend_from_slice(&(-0.0_f32).to_be_bytes());
        }
        for case in 0..3 {
            let mut content = bytes.clone();
            match case {
                1 => {
                    content.pop();
                }
                2 => {
                    let end = content.len();
                    content[end - 4..].copy_from_slice(&f32::INFINITY.to_be_bytes());
                }
                _ => {}
            }
            fs::write(&path, content).unwrap();
            let mut input = Input {
                reader: BufReader::with_capacity(BUFFER_BYTES, File::open(&path).unwrap()),
                remaining: 1,
                encoded_vector: vec![0; dimension * 4],
            };
            if case == 0 {
                let row = input.next().unwrap().unwrap();
                assert_eq!(row.id.as_ref(), b"row");
                assert_eq!(row.vector.len(), dimension);
                assert!(
                    row.vector
                        .iter()
                        .all(|value| value.to_bits() == (-0.0_f32).to_bits())
                );
                assert!(input.next().unwrap().is_none());
            } else {
                assert_eq!(input.next().err().unwrap().kind(), ErrorKind::Corruption);
            }
        }
    }

    fn options() -> ConstructionOptions {
        ConstructionOptions {
            min_partition_entries: 2,
            max_partition_entries: 4,
            sample_items: 4,
            memory_bytes: 200_000,
            scratch_bytes: 16 * 1024 * 1024,
        }
    }

    fn records(count: u64, equal: bool) -> Vec<Result<ConstructionRecord>> {
        (0..count)
            .map(|id| {
                Ok(ConstructionRecord {
                    id: Bytes::copy_from_slice(&id.to_be_bytes()),
                    vector: if equal {
                        vec![1.0, 0.0]
                    } else {
                        vec![id as f32 / 100.0, ((id * 7) % 23) as f32]
                    }
                    .into_boxed_slice(),
                })
            })
            .collect()
    }

    fn build(
        input: Vec<Result<ConstructionRecord>>,
        options: ConstructionOptions,
        metric: Metric,
    ) -> (ConstructionReport, BTreeMap<u64, PartitionPlan>) {
        let dir = Directory::new();
        let mut plans = BTreeMap::new();
        let report = construct_tree(&dir.0, 2, metric, [7; 32], options, input, |plan| {
            assert!(plans.insert(plan.key.get(), plan).is_none());
            Ok(())
        })
        .unwrap();
        assert_eq!(
            fs::read_dir(&dir.0).unwrap().count(),
            0,
            "all intermediate runs reclaimed"
        );
        assert!(report.peak_scratch_bytes <= options.scratch_bytes);
        (report, plans)
    }

    #[test]
    fn balanced_bottom_up_topology_has_exact_membership_for_all_metrics() {
        for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
            for equal in [false, true] {
                // Avoid a zero original vector in the cosine case.
                let mut input = records(101, equal);
                if !equal {
                    input[0].as_mut().unwrap().vector[0] = 0.1;
                }
                let (report, plans) = build(input, options(), metric);
                assert_eq!(report.records, 101);
                assert_eq!(report.partitions, plans.len() as u64);
                assert_eq!(report.partition_high_water, report.partitions);
                let mut members = BTreeSet::new();
                let mut incoming = BTreeSet::new();
                for (&key, plan) in &plans {
                    assert!(plan.entries.len() <= 4);
                    if key != 1 {
                        assert!(plan.entries.len() >= 2);
                    }
                    assert!(plan.entries.windows(2).all(|pair| pair[0] < pair[1]));
                    assert!(plan.centroid.iter().all(|value| value.is_finite()));
                    for id in &plan.entries {
                        if plan.level == 1 {
                            assert!(members.insert(id.clone()));
                        } else {
                            let child = u64::from_be_bytes(id.as_ref().try_into().unwrap());
                            assert!(incoming.insert(child));
                            assert_eq!(plans[&child].level + 1, plan.level);
                        }
                    }
                }
                assert_eq!(members.len(), 101);
                assert_eq!(incoming.len() + 1, plans.len());
                assert!(!incoming.contains(&1));
            }
        }
    }

    #[test]
    fn spill_bound_and_input_order_do_not_change_the_plan() {
        let (first, a) = build(records(257, false), options(), Metric::L2);
        let mut reversed = records(257, false);
        reversed.reverse();
        let mut large = options();
        large.memory_bytes *= 3;
        let (_, b) = build(reversed, large, Metric::L2);
        assert!(first.scratch_written_bytes > first.peak_scratch_bytes);
        assert_eq!(a.len(), b.len());
        for (key, a) in a {
            let b = &b[&key];
            assert_eq!(a.level, b.level);
            assert_eq!(a.entries, b.entries);
            assert_eq!(a.centroid, b.centroid);
        }
    }

    #[test]
    fn projected_splits_keep_variable_id_ties_canonical_across_spill_budgets() {
        for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
            let make = |reverse: bool, budget: usize| {
                let directory = Directory::new();
                let mut settings = options();
                settings.memory_bytes = budget;
                let mut input: Vec<_> = (0_u64..513)
                    .map(|id| {
                        let encoded = id.to_be_bytes();
                        let first = encoded.iter().position(|byte| *byte != 0).unwrap_or(7);
                        ConstructionRecord {
                            id: Bytes::copy_from_slice(&encoded[first..]),
                            vector: vec![1.0, (id % 3) as f32, -0.0, 0.0, 0.0, 0.0, 0.0]
                                .into_boxed_slice(),
                        }
                    })
                    .collect();
                if reverse {
                    input.reverse();
                }
                let mut plans = BTreeMap::new();
                construct_tree(
                    &directory.0,
                    7,
                    metric,
                    [7; 32],
                    settings,
                    input.into_iter().map(Ok),
                    |plan| {
                        assert!(plan.entries.windows(2).all(|ids| ids[0] < ids[1]));
                        plans.insert(plan.key.get(), (plan.level, plan.entries, plan.centroid));
                        Ok(())
                    },
                )
                .unwrap();
                assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 0);
                plans
            };
            let a = make(false, 200_000);
            let b = make(true, 600_000);
            assert_eq!(a, b);
            let ids: std::collections::BTreeSet<_> = a
                .values()
                .filter(|(level, _, _)| *level == 1)
                .flat_map(|(_, entries, _)| entries.iter().cloned())
                .collect();
            assert_eq!(ids.len(), 513);
        }
    }

    #[test]
    fn empty_single_root_duplicates_and_quotas() {
        let (empty, plans) = build(vec![], options(), Metric::L2);
        assert_eq!(empty.records, 0);
        assert!(plans.is_empty());
        let (_, plans) = build(records(1, true), options(), Metric::L2);
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[&1].level, 1);

        let dir = Directory::new();
        let input = records(8, true).into_iter().chain(records(1, true));
        let error = construct_tree(&dir.0, 2, Metric::L2, [7; 32], options(), input, |_| {
            panic!("duplicate input cannot emit a plan")
        })
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::RecordAlreadyExists);

        let dir = Directory::new();
        let mut bounded = options();
        bounded.scratch_bytes = 50;
        let error = construct_tree(
            &dir.0,
            2,
            Metric::L2,
            [7; 32],
            bounded,
            records(4, true),
            |_| panic!("no complete input"),
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::LimitExceeded);
        let bytes: u64 = fs::read_dir(&dir.0)
            .unwrap()
            .map(|e| e.unwrap().metadata().unwrap().len())
            .sum();
        assert!(bytes <= bounded.scratch_bytes);
    }

    #[test]
    fn centroids_use_every_assigned_member_in_canonical_order() {
        for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
            let kernel = VectorKernel::new(2, metric, [7; 32]).unwrap();
            let mut input = records(101, false);
            input[0].as_mut().unwrap().vector[0] = 0.1;
            let originals: BTreeMap<_, _> = input
                .iter()
                .map(|row| {
                    let row = row.as_ref().unwrap();
                    (row.id.clone(), row.vector.clone())
                })
                .collect();
            let mut bounded = options();
            bounded.sample_items = 2;
            let (_, plans) = build(input, bounded, metric);
            for plan in plans.values() {
                let members: Vec<_> = plan
                    .entries
                    .iter()
                    .map(|id| {
                        let vector = if plan.level == 1 {
                            kernel.preprocess(&originals[id]).unwrap()
                        } else {
                            let key = u64::from_be_bytes(id.as_ref().try_into().unwrap());
                            plans[&key].centroid.clone()
                        };
                        kernel.normalize_centroid(&vector).unwrap()
                    })
                    .collect();
                let mean: Vec<_> = (0..2)
                    .map(|axis| {
                        (members.iter().map(|v| f64::from(v[axis])).sum::<f64>()
                            / members.len() as f64) as f32
                    })
                    .collect();
                let expected = kernel.normalize_centroid(&mean).unwrap();
                assert_eq!(
                    plan.centroid, expected,
                    "full membership mean, not sample mean"
                );
            }
        }
    }

    #[test]
    fn failures_propagate_without_replacing_caller_files_or_claiming_success() {
        let dir = Directory::new();
        fs::create_dir(&dir.0).unwrap();
        fs::write(dir.0.join("sentinel"), b"caller data").unwrap();
        assert!(
            construct_tree(
                &dir.0,
                2,
                Metric::L2,
                [7; 32],
                options(),
                records(1, true),
                |_| { panic!("existing directory cannot emit") }
            )
            .is_err()
        );
        assert_eq!(fs::read(dir.0.join("sentinel")).unwrap(), b"caller data");

        let dir = Directory::new();
        let input = records(8, true)
            .into_iter()
            .chain([Err(Error::new(ErrorKind::Cancelled))]);
        assert_eq!(
            construct_tree(&dir.0, 2, Metric::L2, [7; 32], options(), input, |_| {
                panic!("incomplete source cannot emit")
            })
            .unwrap_err()
            .kind(),
            ErrorKind::Cancelled
        );

        let dir = Directory::new();
        let mut emitted = 0;
        let error = construct_tree(
            &dir.0,
            2,
            Metric::L2,
            [7; 32],
            options(),
            records(20, true),
            |_| {
                emitted += 1;
                if emitted == 2 {
                    Err(Error::new(ErrorKind::Backend))
                } else {
                    Ok(())
                }
            },
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Backend);
        assert_eq!(emitted, 2);
    }

    #[test]
    fn invalid_options_and_vectors_fail_before_emitting_plans() {
        let mut large_budget = options();
        large_budget.memory_bytes = usize::MAX;
        let (_, plans) = build(records(1, true), large_budget, Metric::L2);
        assert_eq!(plans.len(), 1);
        let dir = Directory::new();
        let mut invalid = options();
        invalid.memory_bytes = 1;
        assert_eq!(
            construct_tree(
                &dir.0,
                2,
                Metric::L2,
                [7; 32],
                invalid,
                records(1, true),
                |_| panic!("invalid limits")
            )
            .unwrap_err()
            .kind(),
            ErrorKind::InvalidArgument
        );
        assert!(!dir.0.exists());
        for vector in [vec![1.0], vec![f32::NAN, 0.0], vec![0.0, 0.0]] {
            let dir = Directory::new();
            let input = [Ok(ConstructionRecord {
                id: Bytes::from_static(b"id"),
                vector: vector.into_boxed_slice(),
            })];
            assert_eq!(
                construct_tree(
                    &dir.0,
                    2,
                    Metric::Cosine,
                    [7; 32],
                    options(),
                    input,
                    |_| panic!("invalid vector")
                )
                .unwrap_err()
                .kind(),
                ErrorKind::InvalidArgument
            );
        }
    }
}
