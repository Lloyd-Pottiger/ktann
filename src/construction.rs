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
use crate::search::numeric::VectorKernel;

const BUFFER_BYTES: usize = 8 * 1024;
// Conservative row-slot and allocation overhead shared by memory planning.
const ROW_OVERHEAD_BYTES: usize = 128;
const MAX_ID_BYTES: usize = crate::api::MAX_RECORD_ID_BYTES;
const SAMPLE_SEED: u64 = 0x6b74_616e_6e62_756c;

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
    /// Maximum simultaneous scratch-file bytes, including the input and all child outputs.
    pub scratch_bytes: u64,
}

impl ConstructionOptions {
    /// Return the resident row allowance after reserving non-resident working space.
    /// Caller-owned input buffers and emitted plans retained by the consumer are
    /// outside this budget; construction owns only the buffers needed to emit them.
    pub(crate) fn resident_rows(self, dimension: usize) -> Result<usize> {
        let row_bound = dimension
            .checked_mul(4)
            .and_then(|n| n.checked_add(MAX_ID_BYTES + ROW_OVERHEAD_BYTES))
            .ok_or(Error::new(ErrorKind::LimitExceeded))?;
        // Reserve row-sized space for the bounded training sample, up to eight
        // centers and their f64 accumulators, and final partition emission. Charge
        // maximum-length IDs and allocation overhead even when actual rows are small.
        let reserved_rows = self
            .sample_items
            .max(self.max_partition_entries as usize)
            .checked_mul(8)
            .ok_or(Error::new(ErrorKind::LimitExceeded))?;
        // IO buffers and the reader's encoded-vector buffer are reserved separately
        // from decoded resident rows. Only the remainder may hold a resident group;
        // resident recursion borrows slices of that one allocation.
        let reserved = row_bound
            .checked_mul(reserved_rows)
            .and_then(|n| n.checked_add(16 * BUFFER_BYTES))
            // One sequential input reader retains an encoded vector buffer.
            .and_then(|n| n.checked_add(dimension * 4))
            .ok_or(Error::new(ErrorKind::LimitExceeded))?;
        if self.min_partition_entries == 0
            || self
                .min_partition_entries
                .checked_mul(2)
                .is_none_or(|minimum| minimum > self.max_partition_entries)
            || self.sample_items < 2
            || self.memory_bytes == 0
            || self.scratch_bytes == 0
        {
            return Err(Error::invalid_argument());
        }
        if self.memory_bytes
            <= reserved
                .checked_add(row_bound)
                .ok_or(Error::new(ErrorKind::LimitExceeded))?
        {
            return Err(Error::invalid_argument());
        }
        Ok((self.memory_bytes - reserved) / row_bound)
    }
}

/// Version of deterministic construction persisted in Bulk Build descriptors.
pub const CONSTRUCTION_VERSION: u32 = 1;

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
    /// All scratch writes, including repeated grouping passes.
    pub scratch_written_bytes: u64,
}

/// Constructs a bottom-up tree with bounded memory and stable external grouping.
///
/// `directory` must not exist. The caller owns its reclamation, including after
/// failure. Final plans are emitted synchronously in child-before-parent order;
/// consumers must not publish them until this function and exact validation
/// complete. Input must be strictly ordered by Record ID. Global uniqueness and
/// canonical input ordering belong to the enclosing forest preparation stage.
///
/// The rotation seed must be the target Logical Index's persisted seed. Options
/// must be sealed in the enclosing build descriptor.
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
    let resident_rows = options.resident_rows(dimension)?;
    fs::create_dir(directory).map_err(|error| Error::with_source(ErrorKind::Backend, error))?;
    let mut work = Workspace {
        directory: directory.to_owned(),
        next_file: 0,
        bytes: 0,
        report: ConstructionReport::default(),
        options,
        kernel,
        resident_rows,
    };
    let mut input = work.writer()?;
    for record in records {
        let record = record?;
        validate_id(&record.id)?;
        let vector = work.kernel.preprocess(&record.vector)?;
        work.append(&mut input, &record.id, &vector)?;
    }
    let input = input.finish()?;
    work.report.records = input.count;
    if input.count == 0 {
        work.remove(input)?;
        return Ok(work.report);
    }
    let mut level = 1;
    let mut input = input;
    loop {
        if input.count <= u64::from(options.max_partition_entries) {
            // The complete level fits in the root. It has no parent run, and
            // its input is already canonical, so emit it directly from disk.
            let mut reader = work.reader(&input)?;
            work.emit_partition(
                std::iter::from_fn(|| reader.next().transpose()),
                input.count,
                level,
                None,
                &mut emit,
            )?;
            drop(reader);
            work.remove(input)?;
            return Ok(work.report);
        }
        let mut parents = work.writer()?;
        work.group(input, level, &mut parents, &mut emit)?;
        let parents = parents.finish()?;
        input = parents;
        level = level
            .checked_add(1)
            .ok_or(Error::new(ErrorKind::LimitExceeded))?;
    }
}

/// An immutable canonical-ID run; each file belongs to this invocation.
struct Run {
    path: PathBuf,
    count: u64,
    bytes: u64,
}

/// An unfinished scratch run; flushing returns the descriptor used by later passes.
struct Output {
    run: Run,
    writer: BufWriter<File>,
}

impl Output {
    fn finish(mut self) -> Result<Run> {
        self.writer
            .flush()
            .map_err(|error| Error::with_source(ErrorKind::Backend, error))?;
        // These intermediate runs are recomputable, not accepted durable artifacts.
        drop(self.writer);
        Ok(self.run)
    }
}

/// Streams a run with a reusable vector decoding buffer and an exact remaining-row count.
struct Input {
    reader: BufReader<File>,
    remaining: u64,
    encoded_vector: Vec<u8>,
}

/// A construction point with a temporary cluster label; labels and IDs define deterministic order.
#[derive(Default)]
struct Row {
    label: usize,
    id: Bytes,
    vector: Box<[f32]>,
}

impl Row {
    fn compare(&self, other: &Self) -> Ordering {
        self.label
            .cmp(&other.label)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl Input {
    /// Reads one scratch row, rejecting invalid ID lengths, non-finite vector components, and trailing bytes.
    fn next(&mut self) -> Result<Option<Row>> {
        if self.remaining == 0 {
            let mut tail = [0];
            if self
                .reader
                .read(&mut tail)
                .map_err(|error| Error::with_source(ErrorKind::Backend, error))?
                != 0
            {
                return Err(Error::new(ErrorKind::Corruption));
            }
            return Ok(None);
        }
        let mut len = [0; 4];
        self.reader
            .read_exact(&mut len)
            .map_err(|error| Error::with_source(ErrorKind::Corruption, error))?;
        let len = u32::from_be_bytes(len) as usize;
        if len == 0 || len > MAX_ID_BYTES {
            return Err(Error::new(ErrorKind::Corruption));
        }
        let mut id = vec![0; len];
        self.reader
            .read_exact(&mut id)
            .map_err(|error| Error::with_source(ErrorKind::Corruption, error))?;
        // Read each vector in one call, including vectors larger than the IO buffer.
        self.reader
            .read_exact(&mut self.encoded_vector)
            .map_err(|error| Error::with_source(ErrorKind::Corruption, error))?;
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
            return Err(Error::new(ErrorKind::Corruption));
        }
        self.remaining -= 1;
        Ok(Some(Row {
            label: 0,
            id: Bytes::from(id),
            vector: vector.into_boxed_slice(),
        }))
    }
}

/// Owns construction scratch files and accounts their overlap against the shared disk ceiling.
/// The resident-row limit determines when a group can switch to the in-memory path.
struct Workspace {
    directory: PathBuf,
    next_file: u64,
    bytes: u64,
    report: ConstructionReport,
    options: ConstructionOptions,
    kernel: VectorKernel,
    resident_rows: usize,
}

impl Workspace {
    fn writer(&mut self) -> Result<Output> {
        let path = self.directory.join(format!("{:016x}.run", self.next_file));
        self.next_file = self
            .next_file
            .checked_add(1)
            .ok_or(Error::new(ErrorKind::LimitExceeded))?;
        let file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .map_err(|error| Error::with_source(ErrorKind::Backend, error))?;
        Ok(Output {
            run: Run {
                path,
                count: 0,
                bytes: 0,
            },
            writer: BufWriter::with_capacity(BUFFER_BYTES, file),
        })
    }

    /// Charges the live scratch budget before appending an ID and vector to an unfinished run.
    fn append(&mut self, output: &mut Output, id: &[u8], vector: &[f32]) -> Result<()> {
        let bytes = 4 + id.len() as u64 + 4 * vector.len() as u64;
        let total = self
            .bytes
            .checked_add(bytes)
            .ok_or(Error::new(ErrorKind::LimitExceeded))?;
        if total > self.options.scratch_bytes {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
        output
            .writer
            .write_all(&(id.len() as u32).to_be_bytes())
            .map_err(|error| Error::with_source(ErrorKind::Backend, error))?;
        output
            .writer
            .write_all(id)
            .map_err(|error| Error::with_source(ErrorKind::Backend, error))?;
        for component in vector {
            output
                .writer
                .write_all(&component.to_bits().to_be_bytes())
                .map_err(|error| Error::with_source(ErrorKind::Backend, error))?;
        }
        output.run.count = output
            .run
            .count
            .checked_add(1)
            .ok_or(Error::new(ErrorKind::LimitExceeded))?;
        output.run.bytes = output
            .run
            .bytes
            .checked_add(bytes)
            .ok_or(Error::new(ErrorKind::LimitExceeded))?;
        self.bytes = total;
        self.report.peak_scratch_bytes = self.report.peak_scratch_bytes.max(total);
        self.report.scratch_written_bytes = self
            .report
            .scratch_written_bytes
            .checked_add(bytes)
            .ok_or(Error::new(ErrorKind::LimitExceeded))?;
        Ok(())
    }

    fn reader(&self, run: &Run) -> Result<Input> {
        let file =
            File::open(&run.path).map_err(|error| Error::with_source(ErrorKind::Backend, error))?;
        if file
            .metadata()
            .map_err(|error| Error::with_source(ErrorKind::Backend, error))?
            .len()
            != run.bytes
        {
            return Err(Error::new(ErrorKind::Corruption));
        }
        Ok(Input {
            reader: BufReader::with_capacity(BUFFER_BYTES, file),
            remaining: run.count,
            encoded_vector: vec![0; self.kernel.dimension() * 4],
        })
    }

    fn remove(&mut self, run: Run) -> Result<()> {
        fs::remove_file(run.path).map_err(|error| Error::with_source(ErrorKind::Backend, error))?;
        self.bytes = self
            .bytes
            .checked_sub(run.bytes)
            .ok_or(Error::new(ErrorKind::Corruption))?;
        Ok(())
    }

    /// Emit a final group in canonical ID order, using every member for its mean.
    /// A parent output identifies a non-root partition; only the final root
    /// has no parent and receives the reserved key 1.
    /// Both grouping paths must restore this order before calling: f64 addition
    /// is order-sensitive, so equal membership alone does not guarantee equal bits.
    fn emit_partition(
        &mut self,
        rows: impl Iterator<Item = Result<Row>>,
        count: u64,
        level: u32,
        parents: Option<&mut Output>,
        emit: &mut impl FnMut(PartitionPlan) -> Result<()>,
    ) -> Result<()> {
        let key = if parents.is_none() {
            1
        } else {
            self.report
                .partition_high_water
                .max(1)
                .checked_add(1)
                .ok_or(Error::new(ErrorKind::LimitExceeded))?
        };
        self.report.partition_high_water = self.report.partition_high_water.max(key);
        let key = PartitionKey::new(key)?;
        let mut entries = Vec::with_capacity(count as usize);
        let mut sums = vec![0.0_f64; self.kernel.dimension()];
        for row in rows {
            let row = row?;
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
            .map(|sum| (sum / count as f64) as f32)
            .collect();
        let centroid = self.kernel.normalize_centroid(&mean)?;
        if let Some(parents) = parents {
            self.append(parents, &key.get().to_be_bytes(), &centroid)?;
        }
        emit(PartitionPlan {
            key,
            level,
            centroid,
            entries,
        })?;
        self.report.partitions = self
            .report
            .partitions
            .checked_add(1)
            .ok_or(Error::new(ErrorKind::LimitExceeded))?;
        Ok(())
    }

    // Resident recursion retains a single buffer within the row allowance.
    // Recursive children borrow disjoint slices; only the bounded training sample
    // is copied, and it is released before descending into either child.
    fn resident_group(
        &mut self,
        rows: &mut [Row],
        level: u32,
        parents: &mut Output,
        emit: &mut impl FnMut(PartitionPlan) -> Result<()>,
    ) -> Result<()> {
        if rows.len() <= self.options.max_partition_entries as usize {
            rows.sort_unstable_by(|a, b| a.id.cmp(&b.id));
            let count = rows.len() as u64;
            return self.emit_partition(
                rows.iter_mut().map(|row| Ok(std::mem::take(row))),
                count,
                level,
                Some(parents),
                emit,
            );
        }
        // Keep the smallest (seeded ID hash, ID) pairs, matching the external
        // sampler. Unique IDs resolve hash collisions; the ordinal only locates
        // the resident row and cannot affect which records enter the sample.
        let mut sample = BinaryHeap::new();
        for (ordinal, row) in rows.iter().enumerate() {
            let item = (xxh3_128_with_seed(&row.id, SAMPLE_SEED), &row.id, ordinal);
            if sample.len() < self.options.sample_items {
                sample.push(item);
            } else if sample.peek().is_some_and(|max| &item < max) {
                sample.pop();
                sample.push(item);
            }
        }
        let entries: Vec<_> = sample
            .into_iter()
            .map(|(_, _, ordinal)| {
                let row = &rows[ordinal];
                (row.id.clone(), row.vector.clone())
            })
            .collect();
        let centers = self.multi_centers(entries, rows.len() as u64)?;
        let mut counts = vec![0; centers.len()];
        for row in rows.iter_mut() {
            row.label = self.nearest(&row.vector, &centers);
            counts[row.label] += 1;
        }
        let sizes = group_sizes(&counts, u64::from(self.options.min_partition_entries));
        drop(centers);
        // Cutting the total (center label, ID) order selects exact memberships
        // without sorting each slice. Recursive training reorders its sample,
        // and final emission reorders members, so slice order is immaterial.
        let mut rest = rows;
        for count in sizes {
            let count = count as usize;
            if count < rest.len() {
                rest.select_nth_unstable_by(count, Row::compare);
            }
            let (child, tail) = rest.split_at_mut(count);
            self.resident_group(child, level, parents, emit)?;
            rest = tail;
        }
        Ok(())
    }

    /// Constructs one group in memory when it fits; otherwise samples, partitions, and recurses.
    /// Both paths use deterministic sampling and ordering so spill decisions do not change topology.
    fn group(
        &mut self,
        input: Run,
        level: u32,
        parents: &mut Output,
        emit: &mut impl FnMut(PartitionPlan) -> Result<()>,
    ) -> Result<()> {
        if input.count <= self.resident_rows as u64 {
            let mut rows = Vec::with_capacity(input.count as usize);
            let mut reader = self.reader(&input)?;
            while let Some(row) = reader.next()? {
                rows.push(row);
            }
            drop(reader);
            self.remove(input)?;
            return self.resident_group(&mut rows, level, parents, emit);
        }
        if input.count <= u64::from(self.options.max_partition_entries) {
            // Stable scatters preserve canonical ID order for streamed means.
            let mut reader = self.reader(&input)?;
            self.emit_partition(
                std::iter::from_fn(|| reader.next().transpose()),
                input.count,
                level,
                Some(parents),
                emit,
            )?;
            drop(reader);
            self.remove(input)?;
            return Ok(());
        }
        let mut reader = self.reader(&input)?;
        // The bounded max-heap retains the same smallest (hash, ID) pairs as
        // resident sampling, regardless of the order in which rows are visited.
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
        let entries: Vec<_> = sample
            .into_iter()
            .map(|item| (item.row.id, item.row.vector))
            .collect();
        let centers = self.multi_centers(entries, input.count)?;
        let mut counts = vec![0; centers.len()];
        let mut reader = self.reader(&input)?;
        while let Some(row) = reader.next()? {
            counts[self.nearest(&row.vector, &centers)] += 1;
        }
        drop(reader);
        let sizes = group_sizes(&counts, u64::from(self.options.min_partition_entries));
        let mut end = 0;
        let ends: Vec<_> = sizes
            .iter()
            .map(|size| {
                end += size;
                end
            })
            .collect();
        let mut offset = 0;
        let mut ranks: Vec<_> = counts
            .iter()
            .map(|count| {
                let start = offset;
                offset += count;
                start
            })
            .collect();
        let mut outputs: Vec<_> = (0..centers.len())
            .map(|_| self.writer())
            .collect::<Result<_>>()?;
        let mut reader = self.reader(&input)?;
        // Canonical input order gives each row its rank in (center, ID) order
        // without sorting vectors. Clamped boundaries cut that same total order
        // as resident selection, even when a boundary crosses a natural cluster.
        // Each output is an ID-ordered subsequence of the input, preserving the
        // canonical run invariant for the next recursion or centroid reduction.
        while let Some(row) = reader.next()? {
            let label = self.nearest(&row.vector, &centers);
            let child = ends.partition_point(|&end| end <= ranks[label]);
            ranks[label] += 1;
            self.append(&mut outputs[child], &row.id, &row.vector)?;
        }
        drop(reader);
        self.remove(input)?;
        drop(centers);
        let runs: Vec<_> = outputs
            .into_iter()
            .map(Output::finish)
            .collect::<Result<_>>()?;
        for run in runs {
            self.group(run, level, parents, emit)?;
        }
        Ok(())
    }

    /// Stable center-index ties make assignments independent of input traversal.
    /// Preprocessing, scratch decoding, and center normalization establish the
    /// finite, fixed-dimension invariant before these repeated distance scans.
    fn nearest(&self, vector: &[f32], centers: &[Box<[f32]>]) -> usize {
        let mut best = 0;
        let mut distance = f64::INFINITY;
        for (chunk_index, chunk) in centers.chunks(4).enumerate() {
            let distances = match chunk {
                [a, b, c, d] => self
                    .kernel
                    .routing_distances_validated([a, b, c, d], vector),
                [a, b, c] => {
                    let d = self.kernel.routing_distances_validated([a, b, c], vector);
                    [d[0], d[1], d[2], f64::INFINITY]
                }
                [a, b] => {
                    let d = self.kernel.routing_distances_validated([a, b], vector);
                    [d[0], d[1], f64::INFINITY, f64::INFINITY]
                }
                [a] => [
                    self.kernel.routing_distances_validated([vector], a)[0],
                    f64::INFINITY,
                    f64::INFINITY,
                    f64::INFINITY,
                ],
                _ => unreachable!("nonempty chunks of at most four centers"),
            };
            for (lane, candidate) in distances.into_iter().take(chunk.len()).enumerate() {
                if candidate.total_cmp(&distance).is_lt() {
                    best = chunk_index * 4 + lane;
                    distance = candidate;
                }
            }
        }
        best
    }

    /// Train at most eight centers, in canonical sample order, with ten Lloyd rounds.
    /// The first center is farthest from the sample mean; subsequent centers are
    /// farthest from their nearest seed. Empty centers retain their previous
    /// position; full-group counts determine the capacity-constrained boundaries.
    fn multi_centers(
        &self,
        mut entries: Vec<(Bytes, Box<[f32]>)>,
        n: u64,
    ) -> Result<Vec<Box<[f32]>>> {
        // Bound in u64 before converting so all platforms select the same fanout.
        let desired = n
            .div_ceil(u64::from(self.options.max_partition_entries))
            .min(8) as usize;
        let k = entries.len().min(desired).max(2);
        // Heap iteration and resident slice order may differ. Canonical sample
        // order fixes both floating-point reduction order and seed tie-breaking:
        // the strict farthest-distance comparison keeps the first ID on ties.
        entries.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        if self.kernel.is_cosine() {
            for (_, vector) in &mut entries {
                *vector = self.kernel.normalize_centroid(vector)?;
            }
        }
        let dim = self.kernel.dimension();
        let mut mean = vec![0.0_f64; dim];
        for (_, vector) in &entries {
            for (sum, value) in mean.iter_mut().zip(vector.iter()) {
                *sum += f64::from(*value);
            }
        }
        let mean: Vec<_> = mean
            .into_iter()
            .map(|v| (v / entries.len() as f64) as f32)
            .collect();
        let mut seeds: Vec<Box<[f32]>> = Vec::new();
        let mut distances = vec![f64::INFINITY; entries.len()];
        for seed in 0..k {
            let mut farthest = 0;
            let mut max = f64::NEG_INFINITY;
            for (i, (_, vector)) in entries.iter().enumerate() {
                let center = if seed == 0 {
                    &mean[..]
                } else {
                    &seeds[seed - 1][..]
                };
                let distance = self.kernel.routing_distances_validated([vector], center)[0];
                distances[i] = if seed <= 1 {
                    distance
                } else {
                    distances[i].min(distance)
                };
                if distances[i].total_cmp(&max).is_gt() {
                    max = distances[i];
                    farthest = i;
                }
            }
            seeds.push(entries[farthest].1.clone());
        }
        // Fixed iteration cap bounds training work. Stable assignments already
        // produced the current centers in the preceding round, so no further
        // update is needed; empty centers retain their current position.
        let mut previous = vec![usize::MAX; entries.len()];
        let mut sums = vec![vec![0.0_f64; dim]; k];
        let mut counts = vec![0_u64; k];
        for _ in 0..10 {
            for sum in &mut sums {
                sum.fill(0.0);
            }
            counts.fill(0);
            let mut changed = false;
            for (i, (_, vector)) in entries.iter().enumerate() {
                let label = self.nearest(vector, &seeds);
                changed |= previous[i] != label;
                previous[i] = label;
                counts[label] += 1;
                for (sum, value) in sums[label].iter_mut().zip(vector.iter()) {
                    *sum += f64::from(*value);
                }
            }
            if !changed {
                break;
            }
            for (label, sum) in sums.iter().enumerate() {
                if counts[label] != 0 {
                    let mean: Vec<_> = sum
                        .iter()
                        .map(|v| (v / counts[label] as f64) as f32)
                        .collect();
                    seeds[label] = self.kernel.normalize_centroid(&mean)?;
                }
            }
        }
        Ok(seeds)
    }
}

/// Keep natural boundaries wherever feasible. Clamp each boundary so every
/// child meets minimum occupancy and shrinks by at least one quarter.
/// Bounds also reserve enough rows and capacity for all remaining children.
fn group_sizes(counts: &[u64], min: u64) -> Vec<u64> {
    let n: u64 = counts.iter().sum();
    // This is a recursion bound, not the final partition size: oversized
    // children are split again until they meet max_partition_entries. Even
    // identical vectors concentrated in one center must make strict progress.
    let max = n - n.div_ceil(4);
    let mut natural_end = 0;
    let mut previous = 0;
    let mut sizes = Vec::with_capacity(counts.len());
    for (index, count) in counts.iter().enumerate() {
        natural_end += count;
        let remaining = (counts.len() - index - 1) as u64;
        // The lower bound gives this child at least min rows and leaves no
        // more than remaining * max rows. The upper bound caps this child at
        // max while reserving remaining * min rows. Clamping preserves each
        // natural boundary that satisfies both constraints. Saturation handles
        // remaining capacity larger than n without underflow or multiplication
        // overflow; that constraint then contributes a lower bound of zero.
        let lower = (previous + min).max(n.saturating_sub(remaining.saturating_mul(max)));
        let upper = previous.saturating_add(max).min(n - remaining * min);
        let end = natural_end.clamp(lower, upper);
        sizes.push(end - previous);
        previous = end;
    }
    sizes
}

/// Orders sample candidates by hash and ID so a bounded max-heap retains the same global sample.
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
    fn bottom_up_topology_has_exact_membership_for_all_metrics() {
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
    fn grouping_keeps_variable_id_ties_canonical_across_spill_budgets() {
        for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
            let make = |budget: usize| {
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
                input.sort_unstable_by(|a, b| a.id.cmp(&b.id));
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
            let a = make(200_000);
            let b = make(600_000);
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
    fn minimum_memory_and_resident_groups_emit_identical_plan_bits() {
        for dimension in [2, 7] {
            let mut constrained = options();
            let (mut low, mut high) = (0, constrained.memory_bytes);
            while low < high {
                let middle = low + (high - low) / 2;
                constrained.memory_bytes = middle;
                if constrained.resident_rows(dimension).is_ok() {
                    high = middle;
                } else {
                    low = middle + 1;
                }
            }
            constrained.memory_bytes = low;
            assert_eq!(constrained.resident_rows(dimension).unwrap(), 1);
            for metric in [Metric::L2, Metric::Cosine, Metric::InnerProduct] {
                let make = |memory| {
                    let directory = Directory::new();
                    let mut settings = constrained;
                    settings.memory_bytes = memory;
                    let mut records: Vec<_> = (0_u64..257)
                        .map(|id| {
                            let encoded = id.to_be_bytes();
                            let first = encoded.iter().position(|byte| *byte != 0).unwrap_or(7);
                            let mut key = encoded[first..].to_vec();
                            if id % 2 == 0 {
                                key.resize(MAX_ID_BYTES, 0);
                            }
                            ConstructionRecord {
                                id: Bytes::from(key),
                                vector: (0..dimension)
                                    .map(|axis| if axis == 0 { 1.0 } else { (id % 3) as f32 })
                                    .collect(),
                            }
                        })
                        .collect();
                    records.sort_unstable_by(|a, b| a.id.cmp(&b.id));
                    let mut plans = BTreeMap::new();
                    let report = construct_tree(
                        &directory.0,
                        dimension,
                        metric,
                        [7; 32],
                        settings,
                        records.into_iter().map(Ok),
                        |plan| {
                            plans.insert(
                                plan.key.get(),
                                (
                                    plan.level,
                                    plan.centroid
                                        .iter()
                                        .map(|value| value.to_bits())
                                        .collect::<Vec<_>>(),
                                    plan.entries,
                                ),
                            );
                            Ok(())
                        },
                    )
                    .unwrap();
                    if memory == low {
                        assert!(report.scratch_written_bytes > report.peak_scratch_bytes);
                    }
                    assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 0);
                    plans
                };
                let expected = make(4 * 1024 * 1024);
                assert_eq!(make(low), expected);
                assert_eq!(make(160_000), expected);
            }
        }
    }

    #[test]
    fn empty_single_root_and_quotas() {
        let (empty, plans) = build(vec![], options(), Metric::L2);
        assert_eq!(empty.records, 0);
        assert!(plans.is_empty());
        let (_, plans) = build(records(1, true), options(), Metric::L2);
        assert_eq!(plans.len(), 1);
        assert_eq!(plans[&1].level, 1);

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
        let error = construct_tree(
            &dir.0,
            2,
            Metric::L2,
            [7; 32],
            options(),
            records(1, true),
            |_| panic!("existing directory cannot emit"),
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Backend);
        let source = std::error::Error::source(&error).unwrap();
        assert!(source.downcast_ref::<std::io::Error>().is_some());
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

    #[test]
    fn separated_clusters_keep_their_exact_natural_membership() {
        let expected: BTreeSet<Vec<Bytes>> = [(0_u64, 8), (8, 17), (17, 27)]
            .into_iter()
            .map(|(start, end)| {
                (start..end)
                    .map(|id| Bytes::copy_from_slice(&id.to_be_bytes()))
                    .collect()
            })
            .collect();
        let mut settings = ConstructionOptions {
            min_partition_entries: 3,
            max_partition_entries: 10,
            sample_items: 27,
            ..options()
        };
        let row_bound = 2 * 4 + MAX_ID_BYTES + ROW_OVERHEAD_BYTES;
        let minimum = row_bound * (27 * 8 + 2) + 16 * BUFFER_BYTES + 2 * 4;
        let mut previous = None;
        for memory_bytes in [minimum, 4 * 1024 * 1024] {
            settings.memory_bytes = memory_bytes;
            let input = (0_u64..27)
                .map(|id| {
                    Ok(ConstructionRecord {
                        id: Bytes::copy_from_slice(&id.to_be_bytes()),
                        vector: vec![
                            if id < 8 {
                                1.0
                            } else if id < 17 {
                                100.0
                            } else {
                                1000.0
                            },
                            1.0,
                        ]
                        .into_boxed_slice(),
                    })
                })
                .collect();
            let (_, plans) = build(input, settings, Metric::L2);
            let leaves = plans
                .values()
                .filter(|plan| plan.level == 1)
                .map(|plan| plan.entries.clone())
                .collect::<BTreeSet<_>>();
            assert_eq!(leaves, expected);
            let bits = plans
                .into_iter()
                .map(|(key, plan)| {
                    (
                        key,
                        plan.level,
                        plan.centroid
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>(),
                        plan.entries,
                    )
                })
                .collect::<Vec<_>>();
            if let Some(previous) = &previous {
                assert_eq!(&bits, previous);
            }
            previous = Some(bits);
        }
    }

    #[test]
    fn natural_groups_with_tiny_samples_keep_minimum_budget_plans_identical() {
        let dimension = crate::api::MAX_DIMENSION;
        let row_bound = dimension * 4 + MAX_ID_BYTES + ROW_OVERHEAD_BYTES;
        let reserved = row_bound * 16 + 16 * BUFFER_BYTES + dimension * 4;
        let build = |row_capacity| {
            let directory = Directory::new();
            let options = ConstructionOptions {
                min_partition_entries: 1,
                max_partition_entries: 2,
                sample_items: 2,
                memory_bytes: reserved + row_capacity * row_bound,
                scratch_bytes: 64 * 1024 * 1024,
            };
            assert_eq!(options.resident_rows(dimension).unwrap(), row_capacity);
            let records = (0_u64..33).map(|id| {
                Ok(ConstructionRecord {
                    id: Bytes::copy_from_slice(&id.to_be_bytes()),
                    vector: (0..dimension)
                        .map(|axis| (id % 9) as f32 + (axis % 3) as f32 * 0.01 + 1.0)
                        .collect(),
                })
            });
            let mut plans = BTreeMap::new();
            let mut members = BTreeSet::new();
            construct_tree(
                &directory.0,
                dimension,
                Metric::Cosine,
                [7; 32],
                options,
                records,
                |plan| {
                    assert!(!plan.entries.is_empty() && plan.entries.len() <= 2);
                    if plan.level == 1 {
                        for id in &plan.entries {
                            assert!(members.insert(id.clone()));
                        }
                    }
                    plans.insert(
                        plan.key.get(),
                        (
                            plan.level,
                            plan.centroid
                                .iter()
                                .map(|v| v.to_bits())
                                .collect::<Vec<_>>(),
                            plan.entries,
                        ),
                    );
                    Ok(())
                },
            )
            .unwrap();
            assert_eq!(members.len(), 33);
            assert_eq!(fs::read_dir(&directory.0).unwrap().count(), 0);
            plans
        };
        assert_eq!(build(2), build(64));
    }

    #[test]
    fn boundaries_preserve_legal_counts_and_bound_degenerate_groups() {
        for min in [1_u64, 3, 64] {
            for k in 2..=8 {
                let n = k * min * 2 + 1;
                let max = n - n.div_ceil(4);
                let mut cases = vec![vec![n / k; k as usize]];
                cases[0][0] += n % k;
                for center in 0..k as usize {
                    let mut counts = vec![0; k as usize];
                    counts[center] = n;
                    cases.push(counts);
                }
                for first in 0..=n {
                    let mut counts = vec![0; k as usize];
                    counts[0] = first;
                    counts[k as usize - 1] = n - first;
                    cases.push(counts);
                }
                for counts in cases {
                    let sizes = group_sizes(&counts, min);
                    assert_eq!(sizes.iter().sum::<u64>(), n);
                    assert!(sizes.iter().all(|&size| size >= min && size <= max));
                    if counts.iter().all(|&count| count >= min && count <= max) {
                        assert_eq!(sizes, counts);
                    }
                }
            }
        }
        let n = u64::MAX;
        let sizes = group_sizes(&[n, 0], 1);
        assert_eq!(sizes, [n - n.div_ceil(4), n.div_ceil(4)]);
    }
}
