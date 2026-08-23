//! Persistent Partition Synopsis values.

use std::cmp::Ordering;
use std::fmt;

use bytes::Bytes;
use xxhash_rust::xxh3::Xxh3;

use crate::api::{DataType, Error, FieldSchema, MAX_STRING_BYTES, Result, Value, typed_order};
use crate::observe::metrics;

use super::MAX_SYNOPSIS_BYTES;
use super::corrupt;
use super::data::{decode_typed_value, encode_typed_value, visit_typed_value_bytes};
use super::manifest::{BloomParameters, IndexManifest};
use super::wire::{Decoder, Encoder};

/// Domain-separates v1 Bloom hashes from every other persistent hash.
const BLOOM_XXH3_SEED_V1: u64 = 0x4b54_414e_4e01_b100;

/// One field's conservative synopsis state.
#[derive(Clone, PartialEq)]
pub struct FieldSynopsis {
    has_null: bool,
    minimum: Option<Value>,
    maximum: Option<Value>,
    bloom: Option<Bytes>,
}

impl FieldSynopsis {
    /// Creates a field synopsis.
    #[must_use]
    fn new(
        has_null: bool,
        minimum: Option<Value>,
        maximum: Option<Value>,
        bloom: Option<Bytes>,
    ) -> Self {
        Self {
            has_null,
            minimum,
            maximum,
            bloom,
        }
    }

    /// Returns whether the synopsis has observed NULL.
    #[must_use]
    pub const fn has_null(&self) -> bool {
        self.has_null
    }

    /// Returns the minimum observed non-NULL value.
    #[must_use]
    pub const fn minimum(&self) -> Option<&Value> {
        self.minimum.as_ref()
    }

    /// Returns the maximum observed non-NULL value.
    #[must_use]
    pub const fn maximum(&self) -> Option<&Value> {
        self.maximum.as_ref()
    }

    /// Returns the optional fixed-size Bloom bytes.
    #[must_use]
    pub const fn bloom(&self) -> Option<&Bytes> {
        self.bloom.as_ref()
    }

    pub(crate) fn bloom_might_contain(
        &self,
        value: &Value,
        data_type: DataType,
        parameters: Option<BloomParameters>,
    ) -> bool {
        let (Some(parameters), Some(bloom)) = (parameters, self.bloom.as_ref()) else {
            return true;
        };
        let (first, step) = bloom_hash(value, data_type);
        bloom_probes(first, step, parameters).all(|bit| {
            let byte = bit / 8;
            bloom[byte] & (1_u8 << (bit % 8)) != 0
        })
    }
}

impl fmt::Debug for FieldSynopsis {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("FieldSynopsis([REDACTED])")
    }
}

/// Conservative synopses aligned with the complete Vector Record schema.
#[derive(Clone, PartialEq)]
pub struct PartitionSynopsis(Box<[FieldSynopsis]>);

impl PartitionSynopsis {
    /// Creates a Partition Synopsis envelope.
    #[must_use]
    fn new(fields: impl Into<Box<[FieldSynopsis]>>) -> Self {
        Self(fields.into())
    }

    /// Creates the canonical empty synopsis for an Index Manifest.
    #[must_use]
    pub fn empty(manifest: &IndexManifest) -> Self {
        let fields = manifest
            .bloom_parameters()
            .iter()
            .map(|parameters| FieldSynopsis {
                has_null: false,
                minimum: None,
                maximum: None,
                bloom: parameters.map(|parameters| Bytes::from(vec![0; parameters.byte_count()])),
            })
            .collect::<Vec<_>>();
        Self::new(fields)
    }

    /// Monotonically expands this synopsis with one exact Leaf projection.
    ///
    /// Delete and source-side movement deliberately do not call this method or
    /// otherwise shrink historical state. A split or merge target starts empty
    /// and calls this method once for each entry actually moved into it.
    ///
    /// # Errors
    ///
    /// Returns `InvalidArgument` if the projection or existing synopsis does
    /// not match the Index Manifest.
    pub fn expand(&mut self, manifest: &IndexManifest, projection: &[Value]) -> Result<()> {
        validate_synopsis_shape(self, manifest)?;
        validate_projection(projection, manifest.config().fields())?;

        for (((synopsis, value), field), parameters) in self
            .0
            .iter_mut()
            .zip(projection)
            .zip(manifest.config().fields())
            .zip(manifest.bloom_parameters())
        {
            expand_field(synopsis, value, field.data_type(), *parameters);
        }
        Ok(())
    }

    /// Returns field synopses in schema order.
    #[must_use]
    pub fn fields(&self) -> &[FieldSynopsis] {
        &self.0
    }

    pub(crate) fn has_shape_for(&self, manifest: &IndexManifest) -> bool {
        validate_synopsis_shape(self, manifest).is_ok()
    }
}

impl fmt::Debug for PartitionSynopsis {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("PartitionSynopsis([REDACTED])")
    }
}

pub(super) fn encode_partition_synopsis(
    encoder: &mut Encoder,
    manifest: &IndexManifest,
    synopsis: &PartitionSynopsis,
) -> Result<()> {
    validate_synopsis(synopsis, manifest)?;
    let schema = manifest.config().fields();
    encoder.u16(u16::try_from(schema.len()).map_err(|_| Error::invalid_argument())?);
    for ((field_synopsis, field), bloom_parameters) in synopsis
        .fields()
        .iter()
        .zip(schema)
        .zip(manifest.bloom_parameters())
    {
        let has_non_null = field_synopsis.minimum.is_some();
        encoder.u8(u8::from(field_synopsis.has_null) | (u8::from(has_non_null) << 1));
        if let (Some(minimum), Some(maximum)) = (&field_synopsis.minimum, &field_synopsis.maximum) {
            encode_typed_value(encoder, field.data_type(), field.is_nullable(), minimum)?;
            encode_typed_value(encoder, field.data_type(), field.is_nullable(), maximum)?;
        }
        if let Some(parameters) = bloom_parameters {
            let bloom = field_synopsis
                .bloom
                .as_ref()
                .ok_or_else(Error::invalid_argument)?;
            encoder.sized_bytes(bloom, parameters.byte_count())?;
        }
    }
    if encoder.len() > MAX_SYNOPSIS_BYTES {
        return Err(Error::invalid_argument());
    }
    Ok(())
}

pub(super) fn decode_partition_synopsis(
    decoder: &mut Decoder,
    manifest: &IndexManifest,
) -> Result<PartitionSynopsis> {
    let schema = manifest.config().fields();
    if usize::from(decoder.u16()?) != schema.len() {
        return Err(corrupt());
    }
    let mut fields = Vec::with_capacity(schema.len());
    for (field, bloom_parameters) in schema.iter().zip(manifest.bloom_parameters()) {
        let flags = decoder.u8()?;
        if flags & !0b11 != 0 {
            return Err(corrupt());
        }
        let has_null = flags & 1 != 0;
        let has_non_null = flags & 2 != 0;
        let (minimum, maximum) = if has_non_null {
            (
                Some(decode_typed_value(decoder, field.data_type(), false)?),
                Some(decode_typed_value(decoder, field.data_type(), false)?),
            )
        } else {
            (None, None)
        };
        let bloom = if let Some(parameters) = bloom_parameters {
            let bloom = decoder.sized_bytes(parameters.byte_count())?;
            Some(bloom)
        } else {
            None
        };
        fields.push(FieldSynopsis::new(has_null, minimum, maximum, bloom));
    }
    let synopsis = PartitionSynopsis::new(fields);
    validate_synopsis(&synopsis, manifest).map_err(|_| corrupt())?;
    Ok(synopsis)
}

fn validate_synopsis(synopsis: &PartitionSynopsis, manifest: &IndexManifest) -> Result<()> {
    validate_synopsis_shape(synopsis, manifest)?;
    for ((synopsis, field), parameters) in synopsis
        .fields()
        .iter()
        .zip(manifest.config().fields())
        .zip(manifest.bloom_parameters())
    {
        if let (Some(bytes), Some(parameters)) = (&synopsis.bloom, parameters) {
            if synopsis.minimum.is_none() && bytes.iter().any(|byte| *byte != 0) {
                return Err(Error::invalid_argument());
            }
            if synopsis.minimum.is_some() && bytes.iter().all(|byte| *byte == 0) {
                return Err(Error::invalid_argument());
            }
            if synopsis.minimum.as_ref().is_some_and(|minimum| {
                !synopsis.bloom_might_contain(minimum, field.data_type(), Some(*parameters))
            }) || synopsis.maximum.as_ref().is_some_and(|maximum| {
                !synopsis.bloom_might_contain(maximum, field.data_type(), Some(*parameters))
            }) {
                return Err(Error::invalid_argument());
            }
        }
    }
    Ok(())
}

fn validate_synopsis_shape(synopsis: &PartitionSynopsis, manifest: &IndexManifest) -> Result<()> {
    let fields = manifest.config().fields();
    if synopsis.fields().len() != fields.len() {
        return Err(Error::invalid_argument());
    }
    for ((synopsis, field), parameters) in synopsis
        .fields()
        .iter()
        .zip(fields)
        .zip(manifest.bloom_parameters())
    {
        validate_field_synopsis(synopsis, field, *parameters)?;
    }
    Ok(())
}

fn validate_projection(projection: &[Value], fields: &[FieldSchema]) -> Result<()> {
    if projection.len() != fields.len() {
        return Err(Error::invalid_argument());
    }
    for (value, field) in projection.iter().zip(fields) {
        match value {
            Value::Null if field.is_nullable() => {}
            Value::Null => return Err(Error::invalid_argument()),
            value => validate_non_null_value(value, field.data_type())?,
        }
    }
    Ok(())
}

/// Expands one already validated field without introducing a failure point.
fn expand_field(
    synopsis: &mut FieldSynopsis,
    value: &Value,
    data_type: DataType,
    bloom_parameters: Option<BloomParameters>,
) {
    if matches!(value, Value::Null) {
        synopsis.has_null = true;
        return;
    }

    let replace_minimum = synopsis
        .minimum
        .as_ref()
        .is_none_or(|minimum| value_order(value, minimum) == Ordering::Less);
    let replace_maximum = synopsis
        .maximum
        .as_ref()
        .is_none_or(|maximum| value_order(value, maximum) == Ordering::Greater);
    match (replace_minimum, replace_maximum) {
        (true, true) => {
            let canonical = canonical_value(value);
            synopsis.minimum = Some(canonical.clone());
            synopsis.maximum = Some(canonical);
        }
        (true, false) => synopsis.minimum = Some(canonical_value(value)),
        (false, true) => synopsis.maximum = Some(canonical_value(value)),
        (false, false) => {}
    }

    if let (Some(parameters), Some(bloom)) = (bloom_parameters, synopsis.bloom.take()) {
        let mut bloom = bloom
            .try_into_mut()
            .unwrap_or_else(|bytes| bytes.as_ref().into());
        let (first, step) = bloom_hash(value, data_type);
        for bit in bloom_probes(first, step, parameters) {
            let byte = bit / 8;
            bloom[byte] |= 1_u8 << (bit % 8);
        }
        // Saturation only weakens pruning; it never breaks the conservative
        // contract (ADR 0021), so a rising fill ratio is purely diagnostic.
        let set_bits: u64 = bloom.iter().map(|byte| u64::from(byte.count_ones())).sum();
        synopsis.bloom = Some(bloom.freeze());
        #[allow(clippy::cast_precision_loss)]
        metrics::bloom_fill_ratio(set_bits as f64 / f64::from(parameters.bit_count()));
    }
}

fn canonical_value(value: &Value) -> Value {
    match value {
        Value::F64(value) if *value == 0.0 => Value::F64(0.0),
        value => value.clone(),
    }
}

fn value_order(left: &Value, right: &Value) -> Ordering {
    typed_order(left, right).expect("validated synopsis values have the same scalar domain")
}

fn bloom_hash(value: &Value, data_type: DataType) -> (u64, u64) {
    let mut hasher = Xxh3::with_seed(BLOOM_XXH3_SEED_V1);
    visit_typed_value_bytes(data_type, false, value, |bytes| hasher.update(bytes))
        .expect("validated non-NULL synopsis value matches its scalar domain");
    let hash = hasher.digest128();
    (hash as u64, (hash >> 64) as u64)
}

fn bloom_probes(first: u64, step: u64, parameters: BloomParameters) -> impl Iterator<Item = usize> {
    (0..parameters.hash_count()).map(move |probe| {
        let bit = first.wrapping_add(u64::from(probe).wrapping_mul(step))
            % u64::from(parameters.bit_count());
        usize::try_from(bit).expect("Bloom bit index fits usize")
    })
}

fn validate_field_synopsis(
    synopsis: &FieldSynopsis,
    field: &FieldSchema,
    bloom_parameters: Option<BloomParameters>,
) -> Result<()> {
    if synopsis.has_null && !field.is_nullable() {
        return Err(Error::invalid_argument());
    }
    match (&synopsis.minimum, &synopsis.maximum) {
        (None, None) => {}
        (Some(minimum), Some(maximum)) => {
            validate_non_null_value(minimum, field.data_type())?;
            validate_non_null_value(maximum, field.data_type())?;
            if typed_order(minimum, maximum).ok_or_else(Error::invalid_argument)?
                == Ordering::Greater
            {
                return Err(Error::invalid_argument());
            }
        }
        _ => return Err(Error::invalid_argument()),
    }
    match (bloom_parameters, &synopsis.bloom) {
        (None, None) => {}
        (Some(parameters), Some(bytes)) if bytes.len() == parameters.byte_count() => {
            validate_bloom_padding(bytes, parameters).map_err(|_| Error::invalid_argument())?;
        }
        _ => return Err(Error::invalid_argument()),
    }
    Ok(())
}

fn validate_non_null_value(value: &Value, data_type: DataType) -> Result<()> {
    match (data_type, value) {
        (DataType::Bool, Value::Bool(_))
        | (DataType::I64, Value::I64(_))
        | (DataType::String, Value::String(_)) => {}
        (DataType::F64, Value::F64(value)) if value.is_finite() => {}
        _ => return Err(Error::invalid_argument()),
    }
    if let Value::String(value) = value {
        if value.len() > MAX_STRING_BYTES {
            return Err(Error::invalid_argument());
        }
    }
    Ok(())
}

fn validate_bloom_padding(bytes: &[u8], parameters: BloomParameters) -> Result<()> {
    if bytes.len() != parameters.byte_count() {
        return Err(corrupt());
    }
    let used_bits = parameters.bit_count() % 8;
    if used_bits != 0 {
        let mask = !((1_u8 << used_bits) - 1);
        if bytes.last().is_some_and(|byte| byte & mask != 0) {
            return Err(corrupt());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU32;

    use crate::api::{
        DataType, ErrorKind, FieldSchema, IndexConfig, LogicalIndexId, Metric, SynopsisConfig,
        Value,
    };

    use crate::storage::values::IndexLifecycle;

    use super::{BloomParameters, IndexManifest, PartitionSynopsis};

    #[test]
    fn expansion_is_atomic_and_canonical() {
        let fields = vec![
            FieldSchema::new("number", DataType::I64)
                .unwrap()
                .nullable(),
            FieldSchema::new("float", DataType::F64).unwrap(),
        ];
        let manifest = manifest(fields);
        let mut synopsis = PartitionSynopsis::empty(&manifest);
        synopsis
            .expand(&manifest, &[Value::Null, Value::F64(-0.0)])
            .unwrap();
        synopsis
            .expand(&manifest, &[Value::I64(5), Value::F64(2.0)])
            .unwrap();
        synopsis
            .expand(&manifest, &[Value::I64(-2), Value::F64(-1.0)])
            .unwrap();

        assert!(synopsis.fields()[0].has_null());
        assert_eq!(synopsis.fields()[0].minimum(), Some(&Value::I64(-2)));
        assert_eq!(synopsis.fields()[0].maximum(), Some(&Value::I64(5)));
        assert_eq!(synopsis.fields()[1].minimum(), Some(&Value::F64(-1.0)));
        assert_eq!(synopsis.fields()[1].maximum(), Some(&Value::F64(2.0)));

        let before = synopsis.clone();
        assert_eq!(
            synopsis
                .expand(&manifest, &[Value::I64(7), Value::F64(f64::NAN)])
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidArgument
        );
        assert_eq!(synopsis, before);
    }

    #[test]
    fn bloom_hash_has_stable_golden_values_for_every_scalar_family() {
        let cases = [
            (DataType::Bool, Value::Bool(true)),
            (DataType::I64, Value::I64(-7)),
            (DataType::F64, Value::F64(1.5)),
            (DataType::String, Value::String("ktann".to_owned())),
        ];
        let hashes = cases
            .into_iter()
            .map(|(data_type, value)| super::bloom_hash(&value, data_type))
            .collect::<Vec<_>>();

        assert_eq!(
            hashes,
            vec![
                (0x24f3_6f48_8673_1ec8, 0xb202_285b_0901_f22d),
                (0x20e4_eebe_2bf2_55d3, 0x53b7_79fe_8cda_6696),
                (0x20d8_9706_4446_f999, 0xb47c_3828_9492_441b),
                (0x61f2_bba0_0fba_cbec, 0xf8c5_68d7_1177_7557),
            ]
        );
    }

    #[test]
    fn bloom_padding_outside_the_derived_bit_count_is_corruption() {
        let config = SynopsisConfig::MinMaxBloom {
            expected_distinct: NonZeroU32::new(2).unwrap(),
            false_positive_rate: 0.03,
        };
        let parameters = BloomParameters::derive(&config)
            .unwrap()
            .expect("Bloom parameters");
        assert_ne!(parameters.bit_count() % 8, 0);
        let mut bytes = vec![0; parameters.byte_count()];
        *bytes.last_mut().unwrap() = 0x80;
        assert_eq!(
            super::validate_bloom_padding(&bytes, parameters)
                .unwrap_err()
                .kind(),
            ErrorKind::Corruption
        );
    }

    fn manifest(fields: Vec<FieldSchema>) -> IndexManifest {
        let bloom_parameters = fields
            .iter()
            .map(|field| BloomParameters::derive(field.synopsis()).unwrap())
            .collect();
        let config = IndexConfig::new(1, Metric::L2)
            .unwrap()
            .with_fields(fields)
            .unwrap();
        IndexManifest::new(
            IndexLifecycle::Active,
            LogicalIndexId::new(1).unwrap(),
            config,
            [0; 32],
            bloom_parameters,
        )
        .unwrap()
    }
}
