//! Persistent namespace and Index Manifest values.

use crate::api::{
    DataType, Error, FieldId, FieldSchema, IndexConfig, LogicalIndexId, MAX_FIELDS,
    MAX_STRING_BYTES, Metric, Result, SynopsisConfig,
};

use super::wire::{Decoder, Encoder};
use super::{
    FORMAT_VERSION, MAX_SYNOPSIS_BYTES, ROTATION_SEED_BYTES, VALUE_CODEC_VERSION, corrupt,
    unsupported,
};

/// The lifecycle state persisted in an Index Manifest.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum IndexLifecycle {
    /// The Logical Index accepts ordinary operations.
    Active,
    /// The Logical Index is being deleted.
    Dropping,
}

/// Exact format parameters for one persisted Bloom synopsis.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct BloomParameters {
    bit_count: u32,
    hash_count: u8,
}

impl BloomParameters {
    /// Creates bounded, nonzero Bloom parameters.
    pub fn new(bit_count: u32, hash_count: u8) -> Result<Self> {
        let byte_count = usize::try_from(bit_count)
            .ok()
            .and_then(|bits| bits.checked_add(7))
            .map(|bits| bits / 8)
            .ok_or_else(Error::invalid_argument)?;
        if bit_count == 0 || hash_count == 0 || byte_count > MAX_SYNOPSIS_BYTES {
            return Err(Error::invalid_argument());
        }
        Ok(Self {
            bit_count,
            hash_count,
        })
    }

    /// Derives the canonical v1 Bloom shape for a synopsis configuration.
    pub fn derive(config: &SynopsisConfig) -> Result<Option<Self>> {
        config
            .bloom_shape()?
            .map(|(bit_count, hash_count)| Self::new(bit_count, hash_count))
            .transpose()
    }

    /// Returns the exact number of persisted bits.
    #[must_use]
    pub const fn bit_count(self) -> u32 {
        self.bit_count
    }

    /// Returns the exact number of hash probes.
    #[must_use]
    pub const fn hash_count(self) -> u8 {
        self.hash_count
    }

    pub(super) fn byte_count(self) -> usize {
        usize::try_from(self.bit_count)
            .expect("u32 fits usize on supported targets")
            .div_ceil(8)
    }
}

/// The authoritative persistent metadata for one Logical Index.
#[derive(Clone, Debug, PartialEq)]
pub struct IndexManifest {
    lifecycle: IndexLifecycle,
    logical_index_id: LogicalIndexId,
    config: IndexConfig,
    rotation_seed: [u8; ROTATION_SEED_BYTES],
    bloom_parameters: Box<[Option<BloomParameters>]>,
}

impl IndexManifest {
    /// Creates a supported version-1 Index Manifest.
    pub fn new(
        lifecycle: IndexLifecycle,
        logical_index_id: LogicalIndexId,
        config: IndexConfig,
        rotation_seed: [u8; ROTATION_SEED_BYTES],
        bloom_parameters: Vec<Option<BloomParameters>>,
    ) -> Result<Self> {
        let manifest = Self {
            lifecycle,
            logical_index_id,
            config,
            rotation_seed,
            bloom_parameters: bloom_parameters.into_boxed_slice(),
        };
        manifest.validate()?;
        Ok(manifest)
    }

    /// Returns the whole persistent format version.
    #[must_use]
    pub const fn format_version(&self) -> u16 {
        FORMAT_VERSION
    }

    /// Returns the logical value codec version.
    #[must_use]
    pub const fn value_codec_version(&self) -> u8 {
        VALUE_CODEC_VERSION
    }

    /// Returns the persistent lifecycle state.
    #[must_use]
    pub const fn lifecycle(&self) -> IndexLifecycle {
        self.lifecycle
    }

    /// Returns the owned Logical Index ID.
    #[must_use]
    pub const fn logical_index_id(&self) -> LogicalIndexId {
        self.logical_index_id
    }

    /// Returns the immutable Logical Index configuration.
    #[must_use]
    pub const fn config(&self) -> &IndexConfig {
        &self.config
    }

    /// Returns the persistent rotation seed.
    #[must_use]
    pub const fn rotation_seed(&self) -> &[u8; ROTATION_SEED_BYTES] {
        &self.rotation_seed
    }

    /// Returns exact Bloom parameters aligned with the field schema.
    #[must_use]
    pub fn bloom_parameters(&self) -> &[Option<BloomParameters>] {
        &self.bloom_parameters
    }

    fn validate(&self) -> Result<()> {
        self.config.validate()?;
        if self.bloom_parameters.len() != self.config.fields().len() {
            return Err(Error::invalid_argument());
        }
        let mut maximum_synopsis_size = 2_usize + 2;
        for (field, parameters) in self.config.fields().iter().zip(&self.bloom_parameters) {
            if BloomParameters::derive(field.synopsis())?.as_ref() != parameters.as_ref() {
                return Err(Error::invalid_argument());
            }
            let encoded_extrema = match field.data_type() {
                DataType::Bool => 2 * 2,
                DataType::I64 | DataType::F64 => 2 * 9,
                DataType::String => 2 * (1 + 4 + MAX_STRING_BYTES),
            };
            maximum_synopsis_size = maximum_synopsis_size
                .checked_add(1 + encoded_extrema)
                .and_then(|size| {
                    parameters.map_or(Some(size), |parameters| {
                        size.checked_add(4 + parameters.byte_count())
                    })
                })
                .ok_or_else(Error::invalid_argument)?;
        }
        if maximum_synopsis_size > MAX_SYNOPSIS_BYTES {
            return Err(Error::invalid_argument());
        }
        Ok(())
    }
}

/// A namespace allocator high-water mark; zero means no ID has been issued.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexIdAllocator {
    high_water: u64,
}

impl IndexIdAllocator {
    /// Creates an allocator value.
    #[must_use]
    pub const fn new(high_water: u64) -> Self {
        Self { high_water }
    }

    /// Returns the greatest Logical Index ID ever reserved.
    #[must_use]
    pub const fn high_water(self) -> u64 {
        self.high_water
    }
}

/// An Index Name directory mapping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexNameEntry {
    logical_index_id: LogicalIndexId,
}

impl IndexNameEntry {
    /// Creates a directory mapping.
    #[must_use]
    pub const fn new(logical_index_id: LogicalIndexId) -> Self {
        Self { logical_index_id }
    }

    /// Returns the mapped Logical Index ID.
    #[must_use]
    pub const fn logical_index_id(self) -> LogicalIndexId {
        self.logical_index_id
    }
}

pub(super) fn encode_index_id_allocator(encoder: &mut Encoder, value: IndexIdAllocator) {
    encoder.u64(value.high_water);
}

pub(super) fn decode_index_id_allocator(decoder: &mut Decoder) -> Result<IndexIdAllocator> {
    Ok(IndexIdAllocator::new(decoder.u64()?))
}

pub(super) fn encode_index_name_entry(encoder: &mut Encoder, value: IndexNameEntry) {
    encoder.u64(value.logical_index_id.get());
}

pub(super) fn decode_index_name_entry(decoder: &mut Decoder) -> Result<IndexNameEntry> {
    Ok(IndexNameEntry::new(decoder.logical_index_id()?))
}

pub(super) fn encode_index_manifest(encoder: &mut Encoder, manifest: &IndexManifest) -> Result<()> {
    encoder.u16(FORMAT_VERSION);
    encoder.u8(VALUE_CODEC_VERSION);
    encoder.u8(match manifest.lifecycle {
        IndexLifecycle::Active => 0,
        IndexLifecycle::Dropping => 1,
    });
    encoder.u64(manifest.logical_index_id.get());
    let config = manifest.config();
    encoder.u32(u32::try_from(config.dimension()).map_err(|_| Error::invalid_argument())?);
    encoder.u8(encode_metric(config.metric()));
    encoder.u16(u16::try_from(config.fields().len()).map_err(|_| Error::invalid_argument())?);
    for (field, bloom) in config.fields().iter().zip(manifest.bloom_parameters()) {
        encoder.sized_u8_bytes(field.name().as_bytes())?;
        encoder.u8(encode_data_type(field.data_type()));
        encoder.bool(field.is_nullable());
        match field.synopsis() {
            SynopsisConfig::MinMax => encoder.u8(0),
            SynopsisConfig::MinMaxBloom {
                expected_distinct,
                false_positive_rate,
            } => {
                let parameters = bloom.ok_or_else(Error::invalid_argument)?;
                encoder.u8(1);
                encoder.u32(expected_distinct.get());
                encoder.f64(*false_positive_rate)?;
                encoder.u32(parameters.bit_count);
                encoder.u8(parameters.hash_count);
            }
        }
    }
    encoder
        .u16(u16::try_from(config.tree_key_fields().len()).map_err(|_| Error::invalid_argument())?);
    for field in config.tree_key_fields() {
        encoder.u16(field.0);
    }
    encoder.u32(config.min_partition_entries());
    encoder.u32(config.max_partition_entries());
    encoder.bytes(&manifest.rotation_seed);
    Ok(())
}

pub(super) fn decode_index_manifest(decoder: &mut Decoder) -> Result<IndexManifest> {
    let format_version = decoder.u16()?;
    let declared_codec_version = decoder.u8()?;
    if format_version != FORMAT_VERSION || declared_codec_version != VALUE_CODEC_VERSION {
        return Err(unsupported());
    }
    let lifecycle = match decoder.u8()? {
        0 => IndexLifecycle::Active,
        1 => IndexLifecycle::Dropping,
        _ => return Err(corrupt()),
    };
    let logical_index_id = decoder.logical_index_id()?;
    let dimension = usize::try_from(decoder.u32()?).map_err(|_| corrupt())?;
    let metric = decode_metric(decoder.u8()?)?;
    let field_count = usize::from(decoder.u16()?);
    if field_count > MAX_FIELDS {
        return Err(corrupt());
    }
    let mut fields = Vec::with_capacity(field_count);
    let mut bloom_parameters = Vec::with_capacity(field_count);
    for _ in 0..field_count {
        let name_bytes = decoder.sized_u8_bytes()?;
        let name = std::str::from_utf8(&name_bytes).map_err(|_| corrupt())?;
        let data_type = decode_data_type(decoder.u8()?)?;
        let nullable = decoder.bool()?;
        let synopsis_tag = decoder.u8()?;
        let (synopsis, bloom) = match synopsis_tag {
            0 => (SynopsisConfig::MinMax, None),
            1 => {
                let expected_distinct =
                    std::num::NonZeroU32::new(decoder.u32()?).ok_or_else(corrupt)?;
                let false_positive_rate = decoder.canonical_f64()?;
                let bit_count = decoder.u32()?;
                let hash_count = decoder.u8()?;
                let bloom = BloomParameters::new(bit_count, hash_count).map_err(|_| corrupt())?;
                (
                    SynopsisConfig::MinMaxBloom {
                        expected_distinct,
                        false_positive_rate,
                    },
                    Some(bloom),
                )
            }
            _ => return Err(corrupt()),
        };
        let mut field = FieldSchema::new(name, data_type).map_err(|_| corrupt())?;
        if nullable {
            field = field.nullable();
        }
        field = field.with_synopsis(synopsis).map_err(|_| corrupt())?;
        fields.push(field);
        bloom_parameters.push(bloom);
    }
    let tree_key_count = usize::from(decoder.u16()?);
    if tree_key_count > field_count {
        return Err(corrupt());
    }
    let mut tree_key_fields = Vec::with_capacity(tree_key_count);
    for _ in 0..tree_key_count {
        tree_key_fields.push(FieldId(decoder.u16()?));
    }
    let minimum = decoder.u32()?;
    let maximum = decoder.u32()?;
    let rotation_seed = decoder.array::<ROTATION_SEED_BYTES>()?;
    let config = IndexConfig::new(dimension, metric)
        .and_then(|config| config.with_fields(fields))
        .and_then(|config| config.with_tree_key_fields(tree_key_fields))
        .and_then(|config| config.with_partition_entries(minimum, maximum))
        .and_then(|config| {
            config.validate()?;
            Ok(config)
        })
        .map_err(|_| corrupt())?;
    IndexManifest::new(
        lifecycle,
        logical_index_id,
        config,
        rotation_seed,
        bloom_parameters,
    )
    .map_err(|_| corrupt())
}

const fn encode_metric(metric: Metric) -> u8 {
    match metric {
        Metric::L2 => 0,
        Metric::Cosine => 1,
        Metric::InnerProduct => 2,
    }
}

fn decode_metric(tag: u8) -> Result<Metric> {
    match tag {
        0 => Ok(Metric::L2),
        1 => Ok(Metric::Cosine),
        2 => Ok(Metric::InnerProduct),
        _ => Err(corrupt()),
    }
}

const fn encode_data_type(data_type: DataType) -> u8 {
    match data_type {
        DataType::Bool => 0,
        DataType::I64 => 1,
        DataType::F64 => 2,
        DataType::String => 3,
    }
}

fn decode_data_type(tag: u8) -> Result<DataType> {
    match tag {
        0 => Ok(DataType::Bool),
        1 => Ok(DataType::I64),
        2 => Ok(DataType::F64),
        3 => Ok(DataType::String),
        _ => Err(corrupt()),
    }
}
