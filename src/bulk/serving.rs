//! Exact external joins from finite source and topology to serving key/value bytes.

use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use bytes::{Buf, Bytes};
use sha2::{Digest, Sha256};

use crate::api::{Error, ErrorKind, PartitionKey, Result};
use crate::search::numeric::VectorKernel;
use crate::search::rabitq::RaBitQ7;
use crate::storage::backend::HardLimits;
use crate::storage::keys::{self, LogicalKey, TreeKey};
use crate::storage::values::{
    ChildEntry, IndexLifecycle, IndexManifest, LeafEntry, MAX_VALUE_BYTES, OpaquePayload,
    PartitionCentroid, PartitionHeader, PartitionState, PartitionSynopsis, PartitionTransition,
    PersistentValue, RecordLocation, TreeManifest, ValueCodec, VectorRecord, source,
};

use super::files::{ArtifactManifest, Reader, Writer, corrupt, io_error};
use super::sort::{Row, Run, Sorter, Space};
use super::{ForestArtifact, InputSnapshot};

// The largest logical key has a full Tree Key, an escaped Record ID, and
// fixed identity/discriminator components. Leave room for scratch stream tags.
const MAX_KEY: usize = keys::MAX_TREE_KEY_BYTES + 2 * keys::MAX_RECORD_ID_BYTES + 64;
const MAX_ROW: usize = 8 + MAX_KEY + MAX_VALUE_BYTES + 1;

/// Resource ceilings and target adapter limits for serving-value preparation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ServingOptions {
    /// One external sort's allocated buffers, including bounded merge readers.
    /// Source/value decoding and one Synopsis are separately codec-bounded.
    pub memory_bytes: usize,
    /// All simultaneous scratch runs, including retained join inputs.
    pub scratch_bytes: u64,
    /// Adapter-declared logical key and value limits, checked before sealing.
    pub hard_limits: HardLimits,
}

/// Counts and scratch evidence from successfully encoded serving data.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ServingReport {
    /// Reading and sorting source records for the exact membership join.
    pub source_sort: Duration,
    /// Sorting forest assignments and parent references.
    pub topology_sort: Duration,
    /// Exact joins, value encoding, and output spill writes.
    pub join_encode: Duration,
    /// Merging sorted serving runs.
    pub output_merge: Duration,
    /// Reducing leaf projections into exact synopses.
    pub synopsis: Duration,
    /// Writing and sealing canonical serving data.
    pub final_emit: Duration,
    /// Exactly joined original records and leaf assignments.
    pub records: u64,
    /// Validated partitions, each with exactly one root path.
    pub partitions: u64,
    /// Encoded serving key/value pairs, excluding lifecycle bookkeeping.
    pub keys: u64,
    /// Peak simultaneous scratch bytes over all passes.
    pub peak_scratch_bytes: u64,
    /// Total scratch writes, including merge passes.
    pub scratch_written_bytes: u64,
}

/// One canonical logical key and value, before adapter namespace encoding.
/// This value conveys no authority to write Building data or publish an index.
pub struct ServingEntry {
    /// Canonical index-owned logical key bytes.
    pub key: Bytes,
    /// Canonical value bytes validated against the target Index Manifest.
    pub value: Bytes,
}

/// A sealed, sorted serving-data artifact for one immutable Logical Index.
///
/// Creation proves exact source/leaf membership and parent/child relationships
/// through external joins, then encodes all serving values through core codecs.
/// It excludes the Manifest, name directory, allocator and build bookkeeping.
/// Loading must still fence each transaction and validate the sealed backend
/// before publication; this artifact does not grant either authority.
#[derive(Clone)]
pub struct ServingArtifact {
    directory: PathBuf,
    artifact: ArtifactManifest,
    index: IndexManifest,
    limits: HardLimits,
}

impl ServingArtifact {
    /// Reopens an identity accepted by the fenced core preparation worker.
    pub(crate) fn accepted(
        directory: &Path,
        artifact: ArtifactManifest,
        index: &IndexManifest,
        limits: HardLimits,
    ) -> Result<Self> {
        let value = Self {
            directory: directory.to_owned(),
            artifact,
            index: index.clone(),
            limits,
        };
        value.reader()?;
        Ok(value)
    }

    /// Synchronously encodes a complete forest into a new caller-owned directory.
    ///
    /// Input and forest identities must match the target configuration and seed.
    /// Memory is independent of total records and trees. The scratch quota
    /// covers every join/sort run; `maximum_bytes` separately bounds final data
    /// and its manifest. Failed directories remain unsealed and caller owned.
    pub fn build(
        directory: &Path,
        input: &InputSnapshot,
        forest: &ForestArtifact,
        index: &IndexManifest,
        options: ServingOptions,
        maximum_bytes: u64,
    ) -> Result<(Self, ServingReport)> {
        validate(directory, input, forest, index, options)?;
        let codec = ValueCodec::for_index(index);
        let id = index.logical_index_id();
        let mut writer = Writer::new(
            directory,
            3,
            binding(input, forest, index, options)?,
            maximum_bytes,
        )?;
        let scratch = directory.join("scratch");
        let mut space = Space::new(&scratch, options.scratch_bytes, MAX_ROW)?;
        let mut report = ServingReport::default();
        let start = Instant::now();
        let mut records = Sorter::new(&space, options.memory_bytes)?;
        for record in input.reader()? {
            let mut record = record?;
            records.push(
                &mut space,
                Row {
                    key: record.id().to_vec(),
                    value: source::encode(index.config(), &mut record)?,
                },
            )?;
        }
        let record_run = records.finish(&mut space)?;
        report.source_sort = start.elapsed();
        let start = Instant::now();
        let mut topology = Sorter::new(&space, options.memory_bytes)?;
        for row in forest.reader()? {
            let row = row?;
            let plan = row.partition;
            let node_key = keys::header_key(id, &row.tree_key, plan.key);
            let mut meta = Vec::with_capacity(9 + 4 * index.config().dimension());
            meta.push(0);
            meta.extend_from_slice(&plan.level.to_be_bytes());
            meta.extend_from_slice(&(plan.entries.len() as u32).to_be_bytes());
            for component in &plan.centroid {
                meta.extend_from_slice(&component.to_bits().to_be_bytes());
            }
            topology.push(
                &mut space,
                Row {
                    key: tagged(1, &node_key),
                    value: meta,
                },
            )?;
            for entry in plan.entries {
                if plan.level == 1 {
                    topology.push(
                        &mut space,
                        Row {
                            key: tagged(0, &entry),
                            value: node_key.clone(),
                        },
                    )?;
                } else {
                    let child = PartitionKey::new(u64::from_be_bytes(
                        entry.as_ref().try_into().map_err(|_| corrupt())?,
                    ))
                    .map_err(|_| corrupt())?;
                    let mut parent = Vec::with_capacity(13);
                    parent.push(1);
                    parent.extend_from_slice(&plan.key.get().to_be_bytes());
                    parent.extend_from_slice(&plan.level.to_be_bytes());
                    topology.push(
                        &mut space,
                        Row {
                            key: tagged(1, &keys::header_key(id, &row.tree_key, child)),
                            value: parent,
                        },
                    )?;
                }
            }
        }
        let topology_run = topology.finish(&mut space)?;
        report.topology_sort = start.elapsed();
        let start = Instant::now();
        let mut records = space.reader(&record_run)?;
        let mut topology = space.reader(&topology_run)?;
        let mut next = topology.next()?;
        let mut output = Sorter::new(&space, options.memory_bytes)?;
        let kernel = VectorKernel::new(
            index.config().dimension(),
            index.config().metric(),
            *index.rotation_seed(),
        )?;
        let (types, count) = index.tree_key_types();
        let types = &types[..count];
        let mut previous = None;
        let mut previous_key = None;
        while let Some(record) = records.next()? {
            if previous.as_ref() == Some(&record.key) {
                return Err(Error::new(ErrorKind::RecordAlreadyExists));
            }
            let assignment = next.take().ok_or_else(corrupt)?;
            if assignment.key != tagged(0, &record.key) {
                return Err(corrupt());
            }
            previous = Some(record.key);
            let (tree, leaf) = node(&assignment.value, index)?;
            let record = source::decode(index.config(), Bytes::from(record.value))?;
            let values = index
                .config()
                .tree_key_fields()
                .iter()
                .map(|field| record.fields()[field.0 as usize].clone())
                .collect::<Vec<_>>();
            if TreeKey::encode(types, &values)? != tree {
                return Err(corrupt());
            }
            let record_id = record.id().clone();
            // Record IDs and their Record/Location/Payload suffixes already
            // follow the logical key order, before all tree/partition keys.
            let mut write_record = |key, value| {
                write_entry(
                    &mut writer,
                    &mut previous_key,
                    &mut report,
                    encoded(codec, options.hard_limits, key, value)?,
                )
            };
            write_record(
                LogicalKey::Record {
                    index: id,
                    id: record_id.clone(),
                },
                PersistentValue::VectorRecord(VectorRecord::new(
                    record_id.clone(),
                    Box::from(record.vector()),
                    Box::from(record.fields()),
                )),
            )?;
            write_record(
                LogicalKey::Location {
                    index: id,
                    id: record_id.clone(),
                },
                PersistentValue::RecordLocation(RecordLocation::new(tree.clone(), leaf)),
            )?;
            if let Some(payload) = record.payload() {
                write_record(
                    LogicalKey::Payload {
                        index: id,
                        id: record_id.clone(),
                    },
                    PersistentValue::OpaquePayload(OpaquePayload::new(payload.clone())?),
                )?;
            }
            let leaf_key = LogicalKey::LeafEntry {
                index: id,
                tree_key: tree,
                partition: leaf,
                id: record_id.clone(),
            };
            let entry = PersistentValue::LeafEntry(LeafEntry::new(
                record_id,
                Box::from(record.fields()),
                RaBitQ7::quantize(&kernel.preprocess(record.vector())?)?,
            ));
            let key = keys::encode_key(&leaf_key)?;
            let value = codec.encode_for_key(&leaf_key, &entry)?;
            check_limits(&key, &value, options.hard_limits)?;
            output.push(&mut space, Row { key, value })?;
            report.records = add(report.records, 1)?;
            next = topology.next()?;
        }
        drop((records, previous));
        if next.as_ref().is_some_and(|row| row.key.first() != Some(&1))
            || report.records != input.manifest().items()
        {
            return Err(corrupt());
        }
        let mut tree_state: Option<(TreeKey, PartitionKey)> = None;
        while let Some(row) = next.take() {
            if row.key.first() != Some(&1)
                || row.value.len() != 9 + 4 * index.config().dimension()
                || row.value[0] != 0
            {
                return Err(corrupt());
            }
            let (tree, partition) = node(&row.key[1..], index)?;
            let level = u32::from_be_bytes(row.value[1..5].try_into().expect("level"));
            let count = u32::from_be_bytes(row.value[5..9].try_into().expect("count"));
            let centroid: Box<[f32]> = row.value[9..]
                .as_chunks::<4>()
                .0
                .iter()
                .map(|bytes| f32::from_bits(u32::from_be_bytes(*bytes)))
                .collect();
            next = topology.next()?;
            if partition.get() == 1 {
                if next.as_ref().is_some_and(|entry| entry.key == row.key) {
                    return Err(corrupt());
                }
            } else {
                let parent = next.take().ok_or_else(corrupt)?;
                if parent.key != row.key || parent.value.len() != 13 || parent.value[0] != 1 {
                    return Err(corrupt());
                }
                let parent_key = PartitionKey::new(u64::from_be_bytes(
                    parent.value[1..9].try_into().expect("parent"),
                ))
                .map_err(|_| corrupt())?;
                let parent_level =
                    u32::from_be_bytes(parent.value[9..].try_into().expect("parent level"));
                if level.checked_add(1) != Some(parent_level) {
                    return Err(corrupt());
                }
                emit(
                    &mut space,
                    &mut output,
                    codec,
                    options.hard_limits,
                    LogicalKey::ChildEntry {
                        index: id,
                        tree_key: tree.clone(),
                        partition: parent_key,
                        child: partition,
                    },
                    PersistentValue::ChildEntry(ChildEntry::new(partition, centroid.clone())),
                )?;
                emit(
                    &mut space,
                    &mut output,
                    codec,
                    options.hard_limits,
                    LogicalKey::Centroid {
                        index: id,
                        tree_key: tree.clone(),
                        partition,
                    },
                    PersistentValue::PartitionCentroid(PartitionCentroid::new(centroid)),
                )?;
                next = topology.next()?;
                if next.as_ref().is_some_and(|entry| entry.key == row.key) {
                    return Err(corrupt());
                }
            }
            if tree_state
                .as_ref()
                .is_some_and(|(previous, _)| previous != &tree)
            {
                let (previous, high_water) = tree_state.take().expect("previous tree");
                emit_tree(
                    &mut space,
                    &mut output,
                    codec,
                    options.hard_limits,
                    index,
                    previous,
                    high_water,
                )?;
            }
            tree_state = Some((tree.clone(), partition));
            emit(
                &mut space,
                &mut output,
                codec,
                options.hard_limits,
                LogicalKey::Header {
                    index: id,
                    tree_key: tree.clone(),
                    partition,
                },
                PersistentValue::PartitionHeader(PartitionHeader::new(
                    level,
                    count,
                    1,
                    PartitionState::Ready,
                )?),
            )?;
            emit(
                &mut space,
                &mut output,
                codec,
                options.hard_limits,
                LogicalKey::State {
                    index: id,
                    tree_key: tree,
                    partition,
                },
                PersistentValue::PartitionState(PartitionTransition::Ready {
                    started_at_unix_millis: 0,
                }),
            )?;
            report.partitions = add(report.partitions, 1)?;
        }
        if let Some((tree, high_water)) = tree_state {
            emit_tree(
                &mut space,
                &mut output,
                codec,
                options.hard_limits,
                index,
                tree,
                high_water,
            )?;
        }
        drop(topology);
        report.join_encode = start.elapsed();
        let start = Instant::now();
        let output_run = output.finish(&mut space)?;
        space.remove(record_run)?;
        space.remove(topology_run)?;
        report.output_merge = start.elapsed();
        let start = Instant::now();
        // Leaf groups and Synopsis keys share the same Tree Key/Partition Key
        // ordering, so their reduced output needs no further sort.
        let mut output = space.reader(&output_run)?;
        let (mut synopsis_run, mut synopsis_writer) = space.writer()?;
        let mut synopsis: Option<(LogicalKey, PartitionSynopsis)> = None;
        while let Some(row) = output.next()? {
            let key = keys::decode_key(types, &Bytes::from(row.key))?;
            let LogicalKey::LeafEntry {
                tree_key,
                partition,
                ..
            } = &key
            else {
                continue;
            };
            let synopsis_key = LogicalKey::Synopsis {
                index: id,
                tree_key: tree_key.clone(),
                partition: *partition,
            };
            if synopsis
                .as_ref()
                .is_some_and(|(key, _)| key != &synopsis_key)
            {
                flush_synopsis(
                    &mut space,
                    &mut synopsis_run,
                    &mut synopsis_writer,
                    codec,
                    options.hard_limits,
                    synopsis.take().expect("previous leaf"),
                )?;
            }
            let value = codec.decode(&key, Bytes::from(row.value))?;
            let PersistentValue::LeafEntry(entry) = value else {
                return Err(corrupt());
            };
            let (_, accumulator) =
                synopsis.get_or_insert_with(|| (synopsis_key, PartitionSynopsis::empty(index)));
            accumulator.expand(index, entry.fields())?;
        }
        if let Some(synopsis) = synopsis {
            flush_synopsis(
                &mut space,
                &mut synopsis_run,
                &mut synopsis_writer,
                codec,
                options.hard_limits,
                synopsis,
            )?;
        }
        synopsis_writer.flush().map_err(io_error)?;
        drop((output, synopsis_writer));
        report.synopsis = start.elapsed();
        let start = Instant::now();
        let mut output = space.reader(&output_run)?;
        let mut next = output.next()?;
        let mut synopses = space.reader(&synopsis_run)?;
        let mut next_synopsis = synopses.next()?;
        while next.is_some() || next_synopsis.is_some() {
            let take_synopsis = match (&next, &next_synopsis) {
                (Some(row), Some(synopsis)) => synopsis.key < row.key,
                (None, Some(_)) => true,
                _ => false,
            };
            let row = if take_synopsis {
                let row = next_synopsis.take().expect("selected synopsis");
                next_synopsis = synopses.next()?;
                row
            } else {
                let row = next.take().expect("selected serving row");
                next = output.next()?;
                row
            };
            write_entry(&mut writer, &mut previous_key, &mut report, row)?;
        }
        drop((output, synopses));
        space.remove(output_run)?;
        space.remove(synopsis_run)?;
        fs::remove_dir(scratch).map_err(io_error)?;
        report.peak_scratch_bytes = space.peak;
        report.scratch_written_bytes = space.written;
        let artifact = writer.seal()?;
        report.final_emit = start.elapsed();
        Ok((
            Self {
                directory: directory.to_owned(),
                artifact,
                index: index.clone(),
                limits: options.hard_limits,
            },
            report,
        ))
    }

    /// Persisted identity of the complete serving file.
    #[must_use]
    pub const fn manifest(&self) -> &ArtifactManifest {
        &self.artifact
    }

    /// Reads strictly increasing, canonical serving pairs for the bound index.
    /// Successful exhaustion verifies the entire file; a prefix is not proof of
    /// complete output. Lifecycle/name/build keys are never accepted here.
    pub fn reader(&self) -> Result<ServingReader> {
        Ok(ServingReader {
            reader: Reader::open(
                &self.directory,
                &self.artifact,
                4 + MAX_KEY + MAX_VALUE_BYTES,
            )?,
            index: self.index.clone(),
            limits: self.limits,
            previous: None,
            finished: false,
        })
    }
}

/// Fused, validating iterator of canonical serving key/value bytes.
pub struct ServingReader {
    reader: Reader,
    index: IndexManifest,
    limits: HardLimits,
    previous: Option<Bytes>,
    finished: bool,
}
impl ServingReader {
    pub(crate) fn prefix_sha256(&self) -> [u8; 32] {
        self.reader.prefix_sha256()
    }

    fn decode(&mut self, mut bytes: Bytes) -> Result<ServingEntry> {
        if bytes.len() < 4 {
            return Err(corrupt());
        }
        let length = bytes.get_u32() as usize;
        if length == 0 || length > MAX_KEY || length >= bytes.len() {
            return Err(corrupt());
        }
        let key = bytes.split_to(length);
        check_limits(&key, &bytes, self.limits).map_err(|_| corrupt())?;
        if self
            .previous
            .as_ref()
            .is_some_and(|previous| previous >= &key)
        {
            return Err(corrupt());
        }
        let (types, count) = self.index.tree_key_types();
        let logical = keys::decode_key(&types[..count], &key)?;
        if logical.index() != Some(self.index.logical_index_id())
            || matches!(
                logical,
                LogicalKey::Manifest(_)
                    | LogicalKey::BuildDescriptor(_)
                    | LogicalKey::BuildProgress(_)
            )
        {
            return Err(corrupt());
        }
        ValueCodec::for_index(&self.index).decode(&logical, bytes.clone())?;
        self.previous = Some(key.clone());
        Ok(ServingEntry { key, value: bytes })
    }
}
impl Iterator for ServingReader {
    type Item = Result<ServingEntry>;
    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        let result = match self.reader.next() {
            Some(row) => row.and_then(|row| self.decode(row)),
            None => {
                self.finished = true;
                return None;
            }
        };
        self.finished = result.is_err();
        Some(result)
    }
}
impl std::iter::FusedIterator for ServingReader {}

fn tagged(tag: u8, key: &[u8]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(1 + key.len());
    bytes.push(tag);
    bytes.extend_from_slice(key);
    bytes
}
fn add(a: u64, b: u64) -> Result<u64> {
    a.checked_add(b)
        .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))
}
fn check_limits(key: &[u8], value: &[u8], limits: HardLimits) -> Result<()> {
    if key.len() > limits.max_key_bytes || value.len() > limits.max_value_bytes {
        return Err(Error::new(ErrorKind::LimitExceeded));
    }
    Ok(())
}
fn emit(
    space: &mut Space,
    output: &mut Sorter,
    codec: ValueCodec<'_>,
    limits: HardLimits,
    key: LogicalKey,
    value: PersistentValue,
) -> Result<()> {
    output.push(space, encoded(codec, limits, key, value)?)
}
fn encoded(
    codec: ValueCodec<'_>,
    limits: HardLimits,
    key: LogicalKey,
    value: PersistentValue,
) -> Result<Row> {
    let value = codec.encode_for_key(&key, &value)?;
    let key = keys::encode_key(&key)?;
    check_limits(&key, &value, limits)?;
    Ok(Row { key, value })
}
/// Enforces canonical order across the streamed Record Groups and sorted tree data.
fn write_entry(
    writer: &mut Writer,
    previous: &mut Option<Vec<u8>>,
    report: &mut ServingReport,
    row: Row,
) -> Result<()> {
    if previous.as_ref().is_some_and(|key| key >= &row.key) {
        return Err(corrupt());
    }
    let mut body = Vec::with_capacity(4 + row.key.len() + row.value.len());
    body.extend_from_slice(&(row.key.len() as u32).to_be_bytes());
    body.extend_from_slice(&row.key);
    body.extend_from_slice(&row.value);
    writer.append(&body)?;
    *previous = Some(row.key);
    report.keys = add(report.keys, 1)?;
    Ok(())
}

fn emit_tree(
    space: &mut Space,
    output: &mut Sorter,
    codec: ValueCodec<'_>,
    limits: HardLimits,
    index: &IndexManifest,
    tree: TreeKey,
    high_water: PartitionKey,
) -> Result<()> {
    emit(
        space,
        output,
        codec,
        limits,
        LogicalKey::TreeManifest {
            index: index.logical_index_id(),
            tree_key: tree,
        },
        PersistentValue::TreeManifest(TreeManifest::new(PartitionKey::new(1)?, high_water)?),
    )
}
fn flush_synopsis(
    space: &mut Space,
    run: &mut Run,
    writer: &mut BufWriter<File>,
    codec: ValueCodec<'_>,
    limits: HardLimits,
    (key, synopsis): (LogicalKey, PartitionSynopsis),
) -> Result<()> {
    let bytes = keys::encode_key(&key)?;
    let value = codec.encode_for_key(&key, &PersistentValue::PartitionSynopsis(synopsis))?;
    check_limits(&bytes, &value, limits)?;
    space.append(run, writer, &Row { key: bytes, value })
}
fn node(bytes: &[u8], index: &IndexManifest) -> Result<(TreeKey, PartitionKey)> {
    let (types, count) = index.tree_key_types();
    match keys::decode_key(&types[..count], &Bytes::copy_from_slice(bytes))? {
        LogicalKey::Header {
            index: owner,
            tree_key,
            partition,
        } if owner == index.logical_index_id() => Ok((tree_key, partition)),
        _ => Err(corrupt()),
    }
}
fn validate(
    directory: &Path,
    input: &InputSnapshot,
    forest: &ForestArtifact,
    index: &IndexManifest,
    options: ServingOptions,
) -> Result<()> {
    super::sort::validate_memory(options.memory_bytes, MAX_ROW)?;
    if !forest.matches_source(input, index)
        || directory.as_os_str().len() > 4096
        || options.scratch_bytes == 0
        || options.hard_limits.max_key_bytes == 0
        || options.hard_limits.max_value_bytes == 0
    {
        return Err(Error::invalid_argument());
    }
    Ok(())
}
fn binding(
    input: &InputSnapshot,
    forest: &ForestArtifact,
    index: &IndexManifest,
    options: ServingOptions,
) -> Result<[u8; 32]> {
    let mut hash = Sha256::new();
    hash.update(b"KTANN serving artifact v1");
    hash.update(input.manifest().encode());
    hash.update(forest.manifest().encode());
    hash.update(
        ValueCodec::bootstrap().encode(&PersistentValue::IndexManifest(
            index.with_lifecycle(IndexLifecycle::Building),
        ))?,
    );
    for value in [
        options.memory_bytes as u64,
        options.scratch_bytes,
        options.hard_limits.max_key_bytes as u64,
        options.hard_limits.max_value_bytes as u64,
    ] {
        hash.update(value.to_be_bytes());
    }
    Ok(hash.finalize().into())
}
