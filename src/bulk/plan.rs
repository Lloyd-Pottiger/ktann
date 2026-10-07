//! Durable child-before-parent partition plans for one Tree Key.

use std::fs;
use std::path::{Path, PathBuf};

use bytes::Bytes;
use sha2::{Digest, Sha256};

use crate::api::{Error, Metric, PartitionKey, Result};
use crate::construction::{
    CONSTRUCTION_VERSION, ConstructionOptions, ConstructionRecord, ConstructionReport,
    PartitionPlan, construct_tree,
};

use super::InputSnapshot;
use super::files::{ArtifactManifest, Reader, Writer, corrupt, io_error};

/// A sealed single-tree construction result, bound to its source and algorithm.
///
/// This is a topology artifact, not serving data. Complete records (including
/// fields and payloads) remain in the source for the later exact membership join.
#[derive(Clone)]
pub struct TreeArtifact {
    directory: PathBuf,
    manifest: ArtifactManifest,
    dimension: usize,
    maximum_entries: u32,
}

impl TreeArtifact {
    /// Constructs and seals a tree for a snapshot without Tree Key fields.
    ///
    /// The caller owns the new directory, including on failure. `maximum_bytes`
    /// bounds durable output; `options.scratch_bytes` separately bounds temporary
    /// sorting files. Source storage is not charged to either budget. The final
    /// manifest becomes visible only after source verification and construction
    /// have both completed successfully.
    pub fn build(
        directory: &Path,
        input: &InputSnapshot,
        rotation_seed: [u8; 32],
        options: ConstructionOptions,
        maximum_bytes: u64,
    ) -> Result<(Self, ConstructionReport)> {
        validate_options(input, options)?;
        let config = input.config();
        let records = input.reader()?.map(|record| {
            record.map(|record| ConstructionRecord {
                id: record.id().clone(),
                vector: record.vector().into(),
            })
        });
        let mut writer = Writer::new(
            directory,
            1,
            binding(input, rotation_seed, options),
            maximum_bytes,
        )?;
        let scratch = directory.join("scratch");
        let report = construct_tree(
            &scratch,
            config.dimension(),
            config.metric(),
            rotation_seed,
            options,
            records,
            |plan| writer.append(&encode(&plan)),
        )?;
        fs::remove_dir(scratch).map_err(io_error)?;
        let artifact = Self {
            directory: directory.to_path_buf(),
            manifest: writer.seal()?,
            dimension: config.dimension(),
            maximum_entries: options.max_partition_entries,
        };
        Ok((artifact, report))
    }

    /// Reopens a plan against a persisted descriptor and the exact build inputs.
    /// Successful reader exhaustion is required before trusting complete output.
    pub fn open(
        directory: &Path,
        expected: ArtifactManifest,
        input: &InputSnapshot,
        rotation_seed: [u8; 32],
        options: ConstructionOptions,
    ) -> Result<Self> {
        validate_options(input, options)?;
        if !expected.matches(1, binding(input, rotation_seed, options)) {
            return Err(Error::invalid_argument());
        }
        let artifact = Self {
            directory: directory.to_path_buf(),
            manifest: expected,
            dimension: input.config().dimension(),
            maximum_entries: options.max_partition_entries,
        };
        artifact.reader()?;
        Ok(artifact)
    }

    /// The sealed descriptor to persist before assigning dependent work.
    #[must_use]
    pub const fn manifest(&self) -> &ArtifactManifest {
        &self.manifest
    }

    /// Opens a bounded, checksumming stream of canonical partition plans.
    pub fn reader(&self) -> Result<PlanReader> {
        let maximum_frame = 16 + self.dimension * 4 + self.maximum_entries as usize * 258;
        Ok(PlanReader {
            reader: Reader::open(&self.directory, &self.manifest, maximum_frame)?,
            dimension: self.dimension,
            maximum_entries: self.maximum_entries,
            failed: false,
        })
    }

    /// Verifies file identity, canonical partition bodies, and complete framing.
    /// This does not replace the later exact source/assignment membership join.
    pub fn verify(&self) -> Result<()> {
        for plan in self.reader()? {
            plan?;
        }
        Ok(())
    }
}

/// A fused iterator of child-before-parent partition plans.
pub struct PlanReader {
    reader: Reader,
    dimension: usize,
    maximum_entries: u32,
    failed: bool,
}

impl Iterator for PlanReader {
    type Item = Result<PartitionPlan>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.failed {
            return None;
        }
        let result = self
            .reader
            .next()?
            .and_then(|bytes| decode(bytes, self.dimension, self.maximum_entries));
        self.failed = result.is_err();
        Some(result)
    }
}

impl std::iter::FusedIterator for PlanReader {}

fn validate_options(input: &InputSnapshot, options: ConstructionOptions) -> Result<()> {
    let config = input.config();
    if !config.tree_key_fields().is_empty()
        || options.min_partition_entries != config.min_partition_entries()
        || options.max_partition_entries != config.max_partition_entries()
    {
        return Err(Error::invalid_argument());
    }
    Ok(())
}

fn binding(input: &InputSnapshot, seed: [u8; 32], options: ConstructionOptions) -> [u8; 32] {
    let mut hash = Sha256::new();
    hash.update(b"KTANN single tree plan v1");
    hash.update(CONSTRUCTION_VERSION.to_be_bytes());
    hash.update(input.manifest().encode());
    hash.update([match input.config().metric() {
        Metric::L2 => 0,
        Metric::Cosine => 1,
        Metric::InnerProduct => 2,
    }]);
    hash.update(seed);
    hash.update(options.min_partition_entries.to_be_bytes());
    hash.update(options.max_partition_entries.to_be_bytes());
    hash.update((options.sample_items as u64).to_be_bytes());
    hash.update((options.memory_bytes as u64).to_be_bytes());
    hash.update(options.scratch_bytes.to_be_bytes());
    hash.finalize().into()
}

pub(super) fn encode(plan: &PartitionPlan) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&plan.key.get().to_be_bytes());
    bytes.extend_from_slice(&plan.level.to_be_bytes());
    bytes.extend_from_slice(&(plan.entries.len() as u32).to_be_bytes());
    for component in &plan.centroid {
        bytes.extend_from_slice(
            &if *component == 0.0 {
                0
            } else {
                component.to_bits()
            }
            .to_be_bytes(),
        );
    }
    for entry in &plan.entries {
        bytes.extend_from_slice(&(entry.len() as u16).to_be_bytes());
        bytes.extend_from_slice(entry);
    }
    bytes
}

pub(super) fn decode(
    mut bytes: Bytes,
    dimension: usize,
    maximum_entries: u32,
) -> Result<PartitionPlan> {
    use bytes::Buf;

    if bytes.len() < 16 + dimension * 4 {
        return Err(corrupt());
    }
    let key = PartitionKey::new(bytes.get_u64()).map_err(|_| corrupt())?;
    let level = bytes.get_u32();
    let count = bytes.get_u32();
    if level == 0 || count == 0 || count > maximum_entries {
        return Err(corrupt());
    }
    let mut centroid = Vec::with_capacity(dimension);
    for _ in 0..dimension {
        let bits = bytes.get_u32();
        let value = f32::from_bits(bits);
        if !value.is_finite() || bits == (-0.0_f32).to_bits() {
            return Err(corrupt());
        }
        centroid.push(value);
    }
    // Check a minimum body size before allocating the entry inventory.
    if count as usize > bytes.len() / 3 {
        return Err(corrupt());
    }
    let mut entries = Vec::with_capacity(count as usize);
    for _ in 0..count {
        if bytes.len() < 2 {
            return Err(corrupt());
        }
        let length = bytes.get_u16() as usize;
        if length == 0 || length > 256 || length > bytes.len() || (level > 1 && length != 8) {
            return Err(corrupt());
        }
        let entry = bytes.split_to(length);
        if entries.last().is_some_and(|previous| previous >= &entry)
            || (level > 1 && entry.as_ref() == [0; 8])
        {
            return Err(corrupt());
        }
        entries.push(entry);
    }
    if !bytes.is_empty() {
        return Err(corrupt());
    }
    Ok(PartitionPlan {
        key,
        level,
        centroid: centroid.into_boxed_slice(),
        entries,
    })
}
