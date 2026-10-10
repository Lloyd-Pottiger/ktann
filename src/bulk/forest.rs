//! Complete finite-source grouping and multi-tree topology construction.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use bytes::{Buf, Bytes};
use sha2::{Digest, Sha256};

use crate::api::{DataType, Error, ErrorKind, IndexConfig, Metric, Record, Result};
use crate::construction::{
    CONSTRUCTION_VERSION, ConstructionOptions, ConstructionRecord, PartitionPlan, construct_tree,
};
use crate::storage::keys::{MAX_TREE_KEY_BYTES, TreeKey};

use super::files::{ArtifactManifest, Reader, Writer, corrupt, io_error};
use super::input::PreparedInput;
use super::sort::{Row, Sorter, Space};
use super::{InputSnapshot, plan};

/// Independent ceilings for global input sorting and per-tree construction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ForestOptions {
    /// Existing per-tree construction parameters and resource ceilings.
    pub tree: ConstructionOptions,
    /// Memory ceiling for global duplicate detection and Tree Key grouping.
    pub sort_memory_bytes: usize,
    /// Live scratch ceiling for global sorting, including merge inputs/outputs.
    pub sort_scratch_bytes: u64,
}

/// Completed forest counts and algorithm-owned scratch IO evidence.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ForestReport {
    /// Number of nonempty Tree Keys; empty input creates no synthetic tree.
    pub trees: u64,
    /// Number of unique source Records represented by leaves.
    pub records: u64,
    /// Total partitions across all trees.
    pub partitions: u64,
    /// Peak simultaneous global-sort scratch bytes.
    pub peak_sort_scratch_bytes: u64,
    /// All global-sort writes, including merge passes.
    pub sort_written_bytes: u64,
    /// Peak simultaneous construction scratch bytes for any one tree.
    pub peak_tree_scratch_bytes: u64,
    /// Sum of construction scratch writes across all trees.
    pub tree_written_bytes: u64,
}

/// A partition in one Tree Key's child-before-parent plan.
pub struct ForestPartition {
    /// Canonical Tree Key derived from the input Record's declared fields.
    pub tree_key: TreeKey,
    /// Topology and full-f32 centroid; Partition Keys are local to this tree.
    pub partition: PartitionPlan,
}

/// Immutable topology for every Tree Key of one fully verified input snapshot.
///
/// Global duplicate detection completes before any partition is emitted. This
/// artifact is not serving data: exact source/assignment joins, value encoding,
/// backend loading, and publication remain separate stages. Source Records and
/// payloads remain in the original input snapshot.
#[derive(Clone)]
pub struct ForestArtifact {
    directory: PathBuf,
    manifest: ArtifactManifest,
    config: IndexConfig,
    source: ArtifactManifest,
    seed: [u8; 32],
    pub(super) construction: ConstructionOptions,
}

// Volatile work derived from the same validated records as the sealed snapshot.
// Two sorters share a fixed memory ceiling and a single scratch quota.
pub(super) struct ForestPreparation {
    pub options: ForestOptions,
    space: Space,
    ids: Sorter,
    trees: Sorter,
    types: Vec<DataType>,
}

impl ForestPreparation {
    pub fn new(directory: &Path, config: &IndexConfig, options: ForestOptions) -> Result<Self> {
        validate(config, directory, options)?;
        let space = Space::new(directory, options.sort_scratch_bytes, maximum_row(config))?;
        let (ids, trees) = Sorter::pair(&space, options.sort_memory_bytes)?;
        Ok(Self {
            options,
            ids,
            trees,
            space,
            types: tree_types(config),
        })
    }

    pub fn append(&mut self, config: &IndexConfig, record: &Record) -> Result<()> {
        self.ids.push(
            &mut self.space,
            Row {
                key: record.id().to_vec(),
                value: Vec::new(),
            },
        )?;
        self.trees
            .push(&mut self.space, project(config, &self.types, record)?)
    }
}

impl ForestArtifact {
    /// Builds every nonempty tree into one sealed, canonically ordered artifact.
    ///
    /// Runs synchronously on a blocking caller. The new directory and any
    /// partial files remain caller owned on failure. Global sort scratch and
    /// per-tree scratch can overlap; their disk ceilings must be added. Sorting
    /// and construction run sequentially, with only a bounded source projection
    /// reader retained during construction. These memory ceilings exclude the
    /// source decoder and final frame encoder, each bounded by its codec.
    /// `maximum_bytes` independently bounds the durable output plus manifest.
    pub fn build(
        directory: &Path,
        input: &InputSnapshot,
        seed: [u8; 32],
        options: ForestOptions,
        maximum_bytes: u64,
    ) -> Result<(Self, ForestReport)> {
        validate(input.config(), directory, options)?;
        let config = input.config();
        let types = tree_types(config);
        let writer = Writer::new(directory, 2, binding(input, seed, options), maximum_bytes)?;
        let sort_directory = directory.join("sort");
        let mut space = Space::new(
            &sort_directory,
            options.sort_scratch_bytes,
            maximum_row(config),
        )?;
        // Sort only IDs for the global uniqueness check. Spool each tree
        // projection once, avoiding vector-sized duplicate-detection merges
        // without decoding the original payloads a second time.
        let (mut projected, mut projection_writer) = space.writer()?;
        let mut ids = Sorter::new(&space, options.sort_memory_bytes)?;
        for record in input.reader()? {
            let record = record?;
            ids.push(
                &mut space,
                Row {
                    key: record.id().to_vec(),
                    value: Vec::new(),
                },
            )?;
            space.append(
                &mut projected,
                &mut projection_writer,
                &project(config, &types, &record)?,
            )?;
        }
        projection_writer.flush().map_err(io_error)?;
        drop(projection_writer);
        // Reaching EOF above verifies the complete source identity. No output
        // frame is accepted before the global duplicate pass finishes, either.
        if !ids.unique_keys(&mut space)? {
            return Err(Error::new(ErrorKind::RecordAlreadyExists));
        }
        let mut source = space.reader(&projected)?;
        let mut trees = Sorter::new(&space, options.sort_memory_bytes)?;
        while let Some(row) = source.next()? {
            trees.push(&mut space, row)?;
        }
        drop(source);
        let tree_run = trees.finish(&mut space)?;
        space.remove(projected)?;
        Self::construct(directory, input, seed, options, writer, space, tree_run)
    }

    /// Consumes receipt-time sorting work without rescanning the source.
    /// The caller owns scratch directories on failure, just as with `build`.
    pub fn build_prepared(
        directory: &Path,
        prepared: PreparedInput,
        seed: [u8; 32],
        maximum_bytes: u64,
    ) -> Result<(Self, ForestReport)> {
        let PreparedInput {
            source,
            preparation,
        } = prepared;
        let options = preparation.options;
        validate(source.config(), directory, options)?;
        let writer = Writer::new(directory, 2, binding(&source, seed, options), maximum_bytes)?;
        let mut space = preparation.space;
        if !preparation.ids.unique_keys(&mut space)? {
            return Err(Error::new(ErrorKind::RecordAlreadyExists));
        }
        let run = preparation.trees.finish(&mut space)?;
        Self::construct(directory, &source, seed, options, writer, space, run)
    }

    fn construct(
        directory: &Path,
        input: &InputSnapshot,
        seed: [u8; 32],
        options: ForestOptions,
        mut writer: Writer,
        mut space: Space,
        tree_run: super::sort::Run,
    ) -> Result<(Self, ForestReport)> {
        let config = input.config();
        let types = tree_types(config);
        let mut source = space.reader(&tree_run)?;
        let mut next = source.next()?;
        let mut report = ForestReport::default();
        let scratch = directory.join("tree");
        while let Some(first) = &next {
            let tree = TreeKey::from_encoded(&types, Bytes::copy_from_slice(&first.key))?;
            let records = std::iter::from_fn(|| {
                if next.as_ref().is_none_or(|row| row.key != tree.as_bytes()) {
                    return None;
                }
                let row = next.take().expect("matching row");
                let result = projection(row.value, config.dimension());
                match source.next() {
                    Ok(row) => {
                        next = row;
                        Some(result)
                    }
                    Err(error) => Some(Err(error)),
                }
            });
            let built = construct_tree(
                &scratch,
                config.dimension(),
                config.metric(),
                seed,
                options.tree,
                records,
                |partition| {
                    let body = plan::encode(&partition);
                    let mut frame = Vec::with_capacity(4 + tree.as_bytes().len() + body.len());
                    frame.extend_from_slice(&(tree.as_bytes().len() as u32).to_be_bytes());
                    frame.extend_from_slice(tree.as_bytes());
                    frame.extend_from_slice(&body);
                    writer.append(&frame)
                },
            )?;
            fs::remove_dir(&scratch).map_err(io_error)?;
            report.trees = checked_add(report.trees, 1)?;
            report.records = checked_add(report.records, built.records)?;
            report.partitions = checked_add(report.partitions, built.partitions)?;
            report.peak_tree_scratch_bytes =
                report.peak_tree_scratch_bytes.max(built.peak_scratch_bytes);
            report.tree_written_bytes =
                checked_add(report.tree_written_bytes, built.scratch_written_bytes)?;
        }
        drop(source);
        space.remove(tree_run)?;
        space.reclaim_directory()?;
        if report.records != input.manifest().items() {
            return Err(corrupt());
        }
        report.peak_sort_scratch_bytes = space.peak;
        report.sort_written_bytes = space.written;
        let manifest = writer.seal()?;
        Ok((
            Self {
                directory: directory.to_owned(),
                manifest,
                config: config.clone(),
                source: input.manifest().clone(),
                seed,
                construction: options.tree,
            },
            report,
        ))
    }

    /// Reopens against the owner's descriptor and exact source/configuration.
    /// Full integrity is established only by exhausting the reader or `verify`.
    pub fn open(
        directory: &Path,
        expected: ArtifactManifest,
        input: &InputSnapshot,
        seed: [u8; 32],
        options: ForestOptions,
    ) -> Result<Self> {
        validate(input.config(), directory, options)?;
        if !expected.matches(2, binding(input, seed, options)) {
            return Err(Error::invalid_argument());
        }
        let artifact = Self {
            directory: directory.to_owned(),
            manifest: expected,
            config: input.config().clone(),
            source: input.manifest().clone(),
            seed,
            construction: options.tree,
        };
        artifact.reader()?;
        Ok(artifact)
    }

    pub(super) fn matches_source(
        &self,
        input: &InputSnapshot,
        manifest: &crate::storage::values::IndexManifest,
    ) -> bool {
        self.source == *input.manifest()
            && self.config == *manifest.config()
            && input.config() == manifest.config()
            && self.seed == *manifest.rotation_seed()
    }

    /// Sealed identity to persist before scheduling dependent work.
    #[must_use]
    pub const fn manifest(&self) -> &ArtifactManifest {
        &self.manifest
    }

    /// Reads trees in canonical Tree Key order, each ending with its root.
    pub fn reader(&self) -> Result<ForestReader> {
        let maximum_frame = 4
            + MAX_TREE_KEY_BYTES
            + 16
            + self.config.dimension() * 4
            + self.config.max_partition_entries() as usize * 258;
        Ok(ForestReader {
            reader: Reader::open(&self.directory, &self.manifest, maximum_frame)?,
            config: self.config.clone(),
            types: tree_types(&self.config),
            expected_records: self.source.items(),
            records: 0,
            tree: None,
            next_key: 2,
            level: 0,
            rooted: false,
            finished: false,
        })
    }

    /// Verifies framing, canonical tree order, local partition allocation, root
    /// closure, and total leaf assignments. Exact membership joins are separate.
    pub fn verify(&self) -> Result<()> {
        for partition in self.reader()? {
            partition?;
        }
        Ok(())
    }
}

/// Fused reader for a sealed forest; early termination verifies only a prefix.
pub struct ForestReader {
    reader: Reader,
    config: IndexConfig,
    types: Vec<DataType>,
    expected_records: u64,
    records: u64,
    tree: Option<TreeKey>,
    next_key: u64,
    level: u32,
    rooted: bool,
    finished: bool,
}

impl ForestReader {
    fn decode(&mut self, mut frame: Bytes) -> Result<ForestPartition> {
        if frame.len() < 4 {
            return Err(corrupt());
        }
        let length = frame.get_u32() as usize;
        if length > MAX_TREE_KEY_BYTES || length > frame.len() {
            return Err(corrupt());
        }
        let tree = TreeKey::from_encoded(&self.types, frame.split_to(length))?;
        let partition = plan::decode(
            frame,
            self.config.dimension(),
            self.config.max_partition_entries(),
        )?;
        if self.tree.as_ref() != Some(&tree) {
            if self
                .tree
                .as_ref()
                .is_some_and(|previous| previous >= &tree || !self.rooted)
            {
                return Err(corrupt());
            }
            self.tree = Some(tree.clone());
            self.next_key = 2;
            self.level = 0;
            self.rooted = false;
        }
        if self.rooted || partition.level < self.level {
            return Err(corrupt());
        }
        self.level = partition.level;
        if partition.key.get() == 1 {
            self.rooted = true;
        } else {
            if partition.key.get() != self.next_key
                || partition.entries.len() < self.config.min_partition_entries() as usize
            {
                return Err(corrupt());
            }
            self.next_key = self.next_key.checked_add(1).ok_or_else(corrupt)?;
        }
        if partition.level == 1 {
            self.records = self
                .records
                .checked_add(partition.entries.len() as u64)
                .ok_or_else(corrupt)?;
            if self.records > self.expected_records {
                return Err(corrupt());
            }
        }
        Ok(ForestPartition {
            tree_key: tree,
            partition,
        })
    }
}

impl Iterator for ForestReader {
    type Item = Result<ForestPartition>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        match self.reader.next() {
            Some(frame) => {
                let result = frame.and_then(|frame| self.decode(frame));
                self.finished = result.is_err();
                Some(result)
            }
            None => {
                self.finished = true;
                if self.records != self.expected_records || (self.tree.is_some() && !self.rooted) {
                    Some(Err(corrupt()))
                } else {
                    None
                }
            }
        }
    }
}
impl std::iter::FusedIterator for ForestReader {}

// Both receipt-time and snapshot-time preparation use identical projection bytes.
fn project(config: &IndexConfig, types: &[DataType], record: &Record) -> Result<Row> {
    let values = config
        .tree_key_fields()
        .iter()
        .map(|id| record.fields()[id.0 as usize].clone())
        .collect::<Vec<_>>();
    let key = TreeKey::encode(types, &values)?;
    let mut value = Vec::with_capacity(2 + record.id().len() + 4 * config.dimension());
    value.extend_from_slice(&(record.id().len() as u16).to_be_bytes());
    value.extend_from_slice(record.id());
    for component in record.vector() {
        value.extend_from_slice(&component.to_bits().to_be_bytes());
    }
    Ok(Row {
        key: key.as_bytes().to_vec(),
        value,
    })
}

fn tree_types(config: &IndexConfig) -> Vec<DataType> {
    config
        .tree_key_fields()
        .iter()
        .map(|id| config.fields()[id.0 as usize].data_type())
        .collect()
}
fn maximum_row(config: &IndexConfig) -> usize {
    8 + crate::api::MAX_RECORD_ID_BYTES + 4 + MAX_TREE_KEY_BYTES + 4 * config.dimension()
}
fn validate(config: &IndexConfig, directory: &Path, options: ForestOptions) -> Result<()> {
    crate::construction::validate_options(config.dimension(), options.tree)?;
    super::sort::validate_memory(options.sort_memory_bytes, maximum_row(config))?;
    if options.tree.min_partition_entries != config.min_partition_entries()
        || options.tree.max_partition_entries != config.max_partition_entries()
        || options.sort_scratch_bytes == 0
        || directory.as_os_str().len() > 4096
    {
        return Err(Error::invalid_argument());
    }
    Ok(())
}
fn projection(value: Vec<u8>, dimension: usize) -> Result<ConstructionRecord> {
    if value.len() < 2 {
        return Err(corrupt());
    }
    let length = u16::from_be_bytes(value[..2].try_into().expect("fixed length")) as usize;
    if length == 0
        || length > crate::api::MAX_RECORD_ID_BYTES
        || value.len() != 2 + length + dimension * 4
    {
        return Err(corrupt());
    }
    let vector = value[2 + length..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_bits(u32::from_be_bytes(*bytes)))
        .collect::<Box<[_]>>();
    Ok(ConstructionRecord {
        id: Bytes::copy_from_slice(&value[2..2 + length]),
        vector,
    })
}
fn checked_add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))
}
fn binding(input: &InputSnapshot, seed: [u8; 32], options: ForestOptions) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"KTANN forest plan v1");
    hash.update(CONSTRUCTION_VERSION.to_be_bytes());
    hash.update(input.manifest().encode());
    hash.update([match input.config().metric() {
        Metric::L2 => 0,
        Metric::Cosine => 1,
        Metric::InnerProduct => 2,
    }]);
    hash.update(seed);
    hash.update((input.config().tree_key_fields().len() as u16).to_be_bytes());
    for id in input.config().tree_key_fields() {
        hash.update(id.0.to_be_bytes());
    }
    hash.update(options.tree.min_partition_entries.to_be_bytes());
    hash.update(options.tree.max_partition_entries.to_be_bytes());
    for value in [
        options.tree.sample_items as u64,
        options.tree.memory_bytes as u64,
        options.tree.scratch_bytes,
        options.sort_memory_bytes as u64,
        options.sort_scratch_bytes,
    ] {
        hash.update(value.to_be_bytes());
    }
    hash.finalize().into()
}
