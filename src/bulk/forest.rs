//! Complete finite-source grouping and multi-tree topology construction.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

use bytes::{Buf, Bytes};
use sha2::{Digest, Sha256};

use crate::api::{DataType, Error, ErrorKind, IndexConfig, Metric, Record, Result};
use crate::construction::{ConstructionOptions, ConstructionRecord, PartitionPlan};
use crate::storage::keys::{
    MAX_TREE_KEY_BYTES, TreeKey, decode_escaped_terminated, push_escaped_terminated,
    scan_escaped_terminated,
};

use super::files::{ArtifactManifest, Reader, Writer, corrupt, io_error};
use super::sort::{Row, Sorter, Space};
use super::{InputSnapshot, plan};

/// Independent ceilings for global input sorting and per-tree construction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ForestOptions {
    /// Maximum training sample per construction group.
    pub sample_items: usize,
    /// Optional refinement after construction; zero refreshes centroids only.
    pub refinement_rounds: Option<u8>,
    /// Per-tree ceiling for refinement metadata, move planning and cached routing vectors.
    pub refinement_memory_bytes: usize,
    /// Memory ceiling for construction of one tree.
    pub tree_memory_bytes: usize,
    /// Live scratch ceiling for construction of one tree.
    pub tree_scratch_bytes: u64,
    /// Memory ceiling for global duplicate detection and Tree Key grouping.
    pub sort_memory_bytes: usize,
    /// Live scratch ceiling for global sorting, including merge inputs/outputs.
    pub sort_scratch_bytes: u64,
}

impl ForestOptions {
    pub(super) fn construction(self) -> ConstructionOptions {
        ConstructionOptions {
            sample_items: self.sample_items,
            memory_bytes: self.tree_memory_bytes,
            scratch_bytes: self.tree_scratch_bytes,
        }
    }
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
        Self::build_controlled(directory, input, seed, options, maximum_bytes, &|| Ok(()))
    }

    /// Preparation with runtime cancellation retained across refinement.
    pub(crate) fn build_controlled(
        directory: &Path,
        input: &InputSnapshot,
        seed: [u8; 32],
        options: ForestOptions,
        maximum_bytes: u64,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<(Self, ForestReport)> {
        check()?;
        let config = input.config();
        let types = tree_types(config);
        let builder = ForestBuilder::new(directory, input, seed, options, maximum_bytes)?;
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
        builder.projected(space, tree_run, check)
    }

    /// The default tree consumes the same ID-ordered full records as the serving join.
    pub(crate) fn build_ordered(
        directory: &Path,
        input: &InputSnapshot,
        seed: [u8; 32],
        options: ForestOptions,
        maximum_bytes: u64,
        ordered: &super::OrderedSource,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<(Self, ForestReport)> {
        let mut builder = ForestBuilder::new(directory, input, seed, options, maximum_bytes)?;
        if input.manifest().items() != 0 {
            builder.tree(
                &TreeKey::encode(&[], &[])?,
                ordered.construction_records(input)?,
                check,
            )?;
        }
        builder.seal()
    }

    /// Reopens using the trusted source identity and semantic parameters, without
    /// source files or construction budgets. Full integrity requires exhausting
    /// the reader.
    pub fn open(
        directory: &Path,
        expected: ArtifactManifest,
        source: &ArtifactManifest,
        config: IndexConfig,
        seed: [u8; 32],
        sample_items: usize,
        refinement_rounds: Option<u8>,
    ) -> Result<Self> {
        config.validate()?;
        if refinement_rounds
            .is_some_and(|rounds| usize::from(rounds) > crate::api::MAX_REFINEMENT_ROUNDS)
            || sample_items < 2
            || directory.as_os_str().len() > 4096
            || !source.matches(0, super::input::schema_binding(&config))
            || !expected.matches(
                2,
                binding(source, &config, seed, sample_items, refinement_rounds),
            )
        {
            return Err(Error::invalid_argument());
        }
        let artifact = Self {
            directory: directory.to_owned(),
            manifest: expected,
            config,
            source: source.clone(),
            seed,
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
            + 24
            + self.config.dimension() * 4
            + self
                .config
                .max_leaf_partition_entries()
                .max(self.config.max_internal_partition_entries()) as usize
                * (size_of::<u16>() + crate::api::MAX_RECORD_ID_BYTES);
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
    /// Checks each partition envelope and advances the cross-frame tree ordering state.
    /// A tree must end in its root before the next Tree Key can begin.
    fn decode(&mut self, mut frame: Bytes) -> Result<ForestPartition> {
        if frame.len() < 4 {
            return Err(corrupt());
        }
        let length = frame.get_u32() as usize;
        if length > MAX_TREE_KEY_BYTES || length > frame.len() {
            return Err(corrupt());
        }
        let tree = TreeKey::from_encoded(&self.types, frame.split_to(length))?;
        let partition = plan::decode(frame, &self.config)?;
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
                || partition.entries.len()
                    < self.config.min_partition_entries(partition.level) as usize
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

/// Writes canonical partition plans and accounts for each completed construction.
struct ForestBuilder<'a> {
    input: &'a InputSnapshot,
    scratch: PathBuf,
    writer: Writer,
    seed: [u8; 32],
    options: ForestOptions,
    report: ForestReport,
}

impl<'a> ForestBuilder<'a> {
    fn new(
        directory: &Path,
        input: &'a InputSnapshot,
        seed: [u8; 32],
        options: ForestOptions,
        maximum_bytes: u64,
    ) -> Result<Self> {
        validate(input.config(), directory, options)?;
        let writer = Writer::new(
            directory,
            2,
            binding(
                input.manifest(),
                input.config(),
                seed,
                options.sample_items,
                options.refinement_rounds,
            ),
            maximum_bytes,
        )?;
        Ok(Self {
            input,
            scratch: directory.join("tree"),
            writer,
            seed,
            options,
            report: ForestReport::default(),
        })
    }

    fn projected(
        mut self,
        mut space: Space,
        tree_run: super::sort::Run,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<(ForestArtifact, ForestReport)> {
        let config = self.input.config();
        let types = tree_types(config);
        let mut source = space.reader(&tree_run)?;
        let mut next = source.next()?;
        while let Some(first) = &next {
            check()?;
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
            self.tree(&tree, records, check)?;
        }
        drop(source);
        space.remove(tree_run)?;
        space.reclaim_directory()?;
        self.report.peak_sort_scratch_bytes = space.peak;
        self.report.sort_written_bytes = space.written;
        self.seal()
    }

    fn tree(
        &mut self,
        tree: &TreeKey,
        records: impl Iterator<Item = Result<ConstructionRecord>>,
        check: &dyn Fn() -> Result<()>,
    ) -> Result<()> {
        let emit = |partition: PartitionPlan| {
            let mut frame = Vec::new();
            frame.extend_from_slice(&(tree.as_bytes().len() as u32).to_be_bytes());
            frame.extend_from_slice(tree.as_bytes());
            plan::encode(&partition, &mut frame);
            self.writer.append(&frame)
        };
        let built = super::refine::build_tree(
            &self.scratch,
            self.input.config(),
            self.seed,
            self.options,
            records,
            emit,
            &check,
        )?;
        fs::remove_dir(&self.scratch).map_err(io_error)?;
        let report = &mut self.report;
        report.trees = checked_add(report.trees, 1)?;
        report.records = checked_add(report.records, built.records)?;
        report.partitions = checked_add(report.partitions, built.partitions)?;
        report.peak_tree_scratch_bytes =
            report.peak_tree_scratch_bytes.max(built.peak_scratch_bytes);
        report.tree_written_bytes =
            checked_add(report.tree_written_bytes, built.scratch_written_bytes)?;
        Ok(())
    }

    fn seal(self) -> Result<(ForestArtifact, ForestReport)> {
        if self.report.records != self.input.manifest().items() {
            return Err(corrupt());
        }
        let (directory, manifest) = self.writer.seal()?;
        Ok((
            ForestArtifact {
                directory,
                manifest,
                config: self.input.config().clone(),
                source: self.input.manifest().clone(),
                seed: self.seed,
            },
            self.report,
        ))
    }
}

fn project(config: &IndexConfig, types: &[DataType], record: &Record) -> Result<Row> {
    let values = config
        .tree_key_fields()
        .iter()
        .map(|id| record.fields()[id.0 as usize].clone())
        .collect::<Vec<_>>();
    let key = TreeKey::encode(types, &values)?;
    let mut value = Vec::with_capacity(2 + 2 * record.id().len() + 4 * config.dimension());
    // Tuple escaping preserves Record ID order within each Tree Key and
    // delimits the vector even when its first byte is 0xFF.
    push_escaped_terminated(&mut value, record.id());
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
    8 + 2 * crate::api::MAX_RECORD_ID_BYTES + 2 + MAX_TREE_KEY_BYTES + 4 * config.dimension()
}

fn validate(config: &IndexConfig, directory: &Path, options: ForestOptions) -> Result<()> {
    options.construction().resident_rows(config)?;
    super::sort::validate_memory(options.sort_memory_bytes, maximum_row(config))?;
    if options
        .refinement_rounds
        .is_some_and(|rounds| usize::from(rounds) > crate::api::MAX_REFINEMENT_ROUNDS)
        || options.refinement_memory_bytes == 0
        || options.sort_scratch_bytes == 0
        || directory.as_os_str().len() > 4096
    {
        return Err(Error::invalid_argument());
    }
    Ok(())
}

fn projection(value: Vec<u8>, dimension: usize) -> Result<ConstructionRecord> {
    let scan = scan_escaped_terminated(&value, crate::api::MAX_RECORD_ID_BYTES)?;
    let vector_offset = scan.consumed;
    if scan.decoded_len == 0 || value.len() != vector_offset + dimension * 4 {
        return Err(corrupt());
    }
    let vector = value[vector_offset..]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|bytes| f32::from_bits(u32::from_be_bytes(*bytes)))
        .collect::<Box<[_]>>();
    Ok(ConstructionRecord {
        id: Bytes::from(decode_escaped_terminated(&value, &scan)),
        vector,
    })
}

fn checked_add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))
}

pub(super) fn binding(
    source: &ArtifactManifest,
    config: &IndexConfig,
    seed: [u8; 32],
    sample_items: usize,
    refinement_rounds: Option<u8>,
) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"KTANN forest plan v1");
    hash.update(source.encode());
    hash.update([match config.metric() {
        Metric::L2 => 0,
        Metric::Cosine => 1,
        Metric::InnerProduct => 2,
    }]);
    hash.update(seed);
    hash.update((config.tree_key_fields().len() as u16).to_be_bytes());
    for id in config.tree_key_fields() {
        hash.update(id.0.to_be_bytes());
    }
    hash.update(config.min_leaf_partition_entries().to_be_bytes());
    hash.update(config.max_leaf_partition_entries().to_be_bytes());
    hash.update(config.min_internal_partition_entries().to_be_bytes());
    hash.update(config.max_internal_partition_entries().to_be_bytes());
    match config.local_graph() {
        None => hash.update([0]),
        Some(graph) => {
            hash.update([1]);
            hash.update(graph.m().to_be_bytes());
            hash.update(graph.ef_construction().to_be_bytes());
            hash.update(graph.seed());
        }
    }
    hash.update((sample_items as u64).to_be_bytes());
    hash.update([refinement_rounds.map_or(0, |rounds| rounds + 1)]);
    hash.update(b"artifact refinement v1");
    hash.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn projection_separates_escaped_id_from_vector_starting_with_ff() {
        let config = IndexConfig::new(1, Metric::L2).unwrap();
        let component = f32::from_bits(0xff00_0001);
        let record = Record::new(Bytes::from_static(b"a\0"), vec![component], vec![]).unwrap();
        let row = project(&config, &tree_types(&config), &record).unwrap();
        let decoded = projection(row.value, config.dimension()).unwrap();
        assert_eq!(decoded.id, *record.id());
        assert_eq!(decoded.vector[0].to_bits(), component.to_bits());
    }
}
