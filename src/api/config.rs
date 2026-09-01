//! Immutable Logical Index and process-local Runtime configuration.

use std::collections::HashSet;
use std::thread;
use std::time::Duration;

use super::schema::{MAX_ENCODED_SYNOPSIS_BYTES, MAX_FIELDS, MAX_STRING_BYTES};
use super::{DataType, Error, FieldId, FieldSchema, Metric, Result, SearchBudgets, SynopsisConfig};

pub(crate) const MAX_DIMENSION: usize = 16_384;
const MAX_BLOOM_FIELDS: usize = 4;
const MAX_TREE_KEY_BYTES: usize = 8 * 1_024;
const DEFAULT_MIN_PARTITION_ENTRIES: u32 = 16;
const DEFAULT_MAX_PARTITION_ENTRIES: u32 = 128;
const MAX_PARTITION_ENTRIES: u32 = 65_536;
const DEFAULT_FIXUP_QUEUE_CAPACITY: usize = 1_024;
const DEFAULT_FOREGROUND_OPERATION_LIMIT: usize = 1_024;
const MAX_FOREGROUND_OPERATION_LIMIT: usize = 65_536;
const DEFAULT_ATTEMPTS: u32 = 8;
const DEFAULT_PARTITION_CACHE_BYTES: u64 = 256 * 1_024 * 1_024;
const DEFAULT_TREE_KEY_SCAN_RANGES: u32 = 1_024;
const DEFAULT_WRITE_BEAM_SIZE: u32 = 1;
const MAX_WRITE_BEAM_SIZE: u32 = 16_384;

/// Immutable configuration persisted in an Index Manifest.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct IndexConfig {
    dimension: usize,
    metric: Metric,
    fields: Box<[FieldSchema]>,
    tree_key_fields: Box<[FieldId]>,
    min_partition_entries: u32,
    max_partition_entries: u32,
}

impl IndexConfig {
    /// Creates an immutable Logical Index configuration.
    pub fn new(dimension: usize, metric: Metric) -> Result<Self> {
        let config = Self {
            dimension,
            metric,
            fields: Box::new([]),
            tree_key_fields: Box::new([]),
            min_partition_entries: DEFAULT_MIN_PARTITION_ENTRIES,
            max_partition_entries: DEFAULT_MAX_PARTITION_ENTRIES,
        };
        config.validate()?;
        Ok(config)
    }

    /// Sets the ordered Vector Record schema.
    pub fn with_fields(mut self, fields: Vec<FieldSchema>) -> Result<Self> {
        self.fields = fields.into_boxed_slice();
        self.validate_schema()?;
        Ok(self)
    }

    /// Sets the ordered unique non-null Tree Key fields.
    pub fn with_tree_key_fields(mut self, fields: Vec<FieldId>) -> Result<Self> {
        self.tree_key_fields = fields.into_boxed_slice();
        let mut unique = HashSet::with_capacity(self.tree_key_fields.len());
        if self
            .tree_key_fields
            .iter()
            .any(|field| !unique.insert(*field))
        {
            return Err(Error::invalid_argument());
        }
        Ok(self)
    }

    /// Sets the minimum and maximum partition entry counts.
    pub fn with_partition_entries(mut self, minimum: u32, maximum: u32) -> Result<Self> {
        self.min_partition_entries = minimum;
        self.max_partition_entries = maximum;
        self.validate_partition_entries()?;
        Ok(self)
    }

    /// Validates every immutable and cross-field invariant.
    pub fn validate(&self) -> Result<()> {
        if !(1..=MAX_DIMENSION).contains(&self.dimension) {
            return Err(Error::invalid_argument());
        }
        self.validate_schema()?;
        self.validate_tree_key_fields()?;
        self.validate_partition_entries()
    }

    fn validate_schema(&self) -> Result<()> {
        if self.fields.len() > MAX_FIELDS {
            return Err(Error::invalid_argument());
        }
        let mut names = HashSet::with_capacity(self.fields.len());
        let mut bloom_fields = 0_usize;
        let mut encoded_synopsis_bytes = 0_usize;
        for field in &self.fields {
            if !names.insert(field.name()) {
                return Err(Error::invalid_argument());
            }
            field.synopsis().validate()?;
            if matches!(field.synopsis(), SynopsisConfig::MinMaxBloom { .. }) {
                bloom_fields += 1;
            }
            // One type tag, NULL flags, and two maximum encoded extrema are a
            // conservative bound until the canonical v1 value codec writes
            // the exact length. Bloom bytes use the requested probability
            // without weakening it.
            let encoded_value_bytes = match field.data_type() {
                DataType::String => MAX_STRING_BYTES + 5,
                DataType::Bool => 2,
                DataType::I64 | DataType::F64 => 9,
            };
            let extrema_bytes = encoded_value_bytes
                .checked_mul(2)
                .ok_or_else(Error::invalid_argument)?;
            let bloom_bytes = field.synopsis().bloom_bytes()?;
            let field_bytes = 2_usize
                .checked_add(extrema_bytes)
                .and_then(|bytes| bytes.checked_add(bloom_bytes))
                .ok_or_else(Error::invalid_argument)?;
            encoded_synopsis_bytes = encoded_synopsis_bytes
                .checked_add(field_bytes)
                .ok_or_else(Error::invalid_argument)?;
        }
        if bloom_fields > MAX_BLOOM_FIELDS || encoded_synopsis_bytes > MAX_ENCODED_SYNOPSIS_BYTES {
            return Err(Error::invalid_argument());
        }
        Ok(())
    }

    fn validate_tree_key_fields(&self) -> Result<()> {
        let mut tree_fields = HashSet::with_capacity(self.tree_key_fields.len());
        let mut worst_case_tree_key_bytes = 0_usize;
        for field_id in &self.tree_key_fields {
            if !tree_fields.insert(*field_id) {
                return Err(Error::invalid_argument());
            }
            let field = self
                .fields
                .get(usize::from(field_id.0))
                .ok_or_else(Error::invalid_argument)?;
            if field.is_nullable() {
                return Err(Error::invalid_argument());
            }
            // Every v1 scalar Tree Key value has at most the string value's
            // 1 KiB input plus one byte of tuple-encoding overhead per byte,
            // a type tag, and a terminator. This conservative schema-time bound
            // is independent of the later canonical codec implementation.
            let encoded_field_bytes = match field.data_type() {
                DataType::String => 2 * MAX_STRING_BYTES + 2,
                DataType::Bool => 3,
                DataType::I64 | DataType::F64 => 10,
            };
            worst_case_tree_key_bytes = worst_case_tree_key_bytes
                .checked_add(encoded_field_bytes)
                .ok_or_else(Error::invalid_argument)?;
        }
        if worst_case_tree_key_bytes > MAX_TREE_KEY_BYTES {
            return Err(Error::invalid_argument());
        }
        Ok(())
    }

    fn validate_partition_entries(&self) -> Result<()> {
        let twice_minimum = self
            .min_partition_entries
            .checked_mul(2)
            .ok_or_else(Error::invalid_argument)?;
        if self.min_partition_entries == 0
            || twice_minimum > self.max_partition_entries
            || self.max_partition_entries > MAX_PARTITION_ENTRIES
        {
            return Err(Error::invalid_argument());
        }
        Ok(())
    }

    /// Returns the exact vector dimension.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    /// Returns the exact distance metric.
    #[must_use]
    pub const fn metric(&self) -> Metric {
        self.metric
    }

    /// Returns the ordered Vector Record schema.
    #[must_use]
    pub fn fields(&self) -> &[FieldSchema] {
        &self.fields
    }

    /// Returns the ordered Tree Key field positions.
    #[must_use]
    pub fn tree_key_fields(&self) -> &[FieldId] {
        &self.tree_key_fields
    }

    /// Returns the configured minimum partition entry count.
    #[must_use]
    pub const fn min_partition_entries(&self) -> u32 {
        self.min_partition_entries
    }

    /// Returns the configured maximum partition entry count.
    #[must_use]
    pub const fn max_partition_entries(&self) -> u32 {
        self.max_partition_entries
    }
}

/// Process-local configuration owned by one Runtime.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct RuntimeConfig {
    foreground_operation_limit: usize,
    maintenance_workers: usize,
    fixup_queue_capacity: usize,
    fixup_attempts: u32,
    foreground_attempts: u32,
    partition_cache_bytes: u64,
    default_search_budgets: SearchBudgets,
    tree_key_scan_ranges: u32,
    write_beam_size: u32,
    import_max_in_flight_batches: usize,
    import_backlog_watermark: usize,
    stalled_timeout: Option<Duration>,
    retry_initial_backoff: Duration,
    retry_max_backoff: Duration,
}

impl Default for RuntimeConfig {
    fn default() -> Self {
        let available = thread::available_parallelism().map_or(1, usize::from);
        let workers = available.clamp(1, 8);
        Self {
            foreground_operation_limit: DEFAULT_FOREGROUND_OPERATION_LIMIT,
            maintenance_workers: workers,
            fixup_queue_capacity: DEFAULT_FIXUP_QUEUE_CAPACITY,
            fixup_attempts: DEFAULT_ATTEMPTS,
            foreground_attempts: DEFAULT_ATTEMPTS,
            partition_cache_bytes: DEFAULT_PARTITION_CACHE_BYTES,
            default_search_budgets: SearchBudgets::default(),
            tree_key_scan_ranges: DEFAULT_TREE_KEY_SCAN_RANGES,
            write_beam_size: DEFAULT_WRITE_BEAM_SIZE,
            import_max_in_flight_batches: available.clamp(1, 4),
            import_backlog_watermark: 2,
            stalled_timeout: None,
            retry_initial_backoff: Duration::from_millis(1),
            retry_max_backoff: Duration::from_millis(100),
        }
    }
}

impl RuntimeConfig {
    /// Sets equal bounds for running and waiting foreground operations.
    pub fn with_foreground_operation_limit(mut self, limit: usize) -> Result<Self> {
        if limit == 0 || limit > MAX_FOREGROUND_OPERATION_LIMIT {
            return Err(Error::invalid_argument());
        }
        self.foreground_operation_limit = limit;
        Ok(self)
    }

    /// Sets maintenance worker and pending/running Fixup capacity.
    ///
    /// Zero workers disables background Structure Maintenance: this Runtime's
    /// Fixup offers are dropped, and topology changes advance only through
    /// drives outside this Runtime. This stays correct — every committed
    /// intermediate state remains searchable — and exists for callers and
    /// tests that drive the state machines deterministically.
    pub fn with_maintenance(mut self, workers: usize, queue_capacity: usize) -> Result<Self> {
        if queue_capacity == 0 || queue_capacity < workers {
            return Err(Error::invalid_argument());
        }
        self.maintenance_workers = workers;
        self.fixup_queue_capacity = queue_capacity;
        Ok(self)
    }

    /// Sets whole-operation Fixup and Foreground Mutation attempt limits.
    pub fn with_attempts(mut self, fixup: u32, foreground: u32) -> Result<Self> {
        if fixup == 0 || foreground == 0 {
            return Err(Error::invalid_argument());
        }
        self.fixup_attempts = fixup;
        self.foreground_attempts = foreground;
        Ok(self)
    }

    /// Sets the decoded partition cache capacity in bytes; zero disables it.
    pub fn with_partition_cache_bytes(mut self, bytes: u64) -> Result<Self> {
        if usize::try_from(bytes).is_err() {
            return Err(Error::invalid_argument());
        }
        self.partition_cache_bytes = bytes;
        Ok(self)
    }

    /// Sets default Search Budgets within v1 hard caps.
    pub fn with_default_search_budgets(mut self, budgets: SearchBudgets) -> Result<Self> {
        budgets.validate_hard_caps()?;
        self.default_search_budgets = budgets;
        Ok(self)
    }

    /// Sets the bounded Tree Key range plan limit.
    pub fn with_tree_key_scan_ranges(mut self, ranges: u32) -> Result<Self> {
        if ranges == 0 {
            return Err(Error::invalid_argument());
        }
        self.tree_key_scan_ranges = ranges;
        Ok(self)
    }

    /// Sets the per-level beam used when routing inserted and upserted
    /// records. A record is still committed to exactly one leaf; a wider beam
    /// only keeps more candidate paths alive before that final choice.
    pub fn with_write_beam_size(mut self, beam: u32) -> Result<Self> {
        if beam == 0 || beam > MAX_WRITE_BEAM_SIZE {
            return Err(Error::invalid_argument());
        }
        self.write_beam_size = beam;
        Ok(self)
    }

    /// Sets the Import Session concurrency ceiling and backlog admission bound.
    ///
    /// A session starts with one active batch and learns useful concurrency up
    /// to `in_flight` from clean completions and retryable contention. A
    /// non-empty batch admits only once the process-local Fixup backlog
    /// (pending plus running) is below `backlog_watermark`; a zero watermark
    /// holds every non-empty batch. The gate is process-local backpressure and
    /// never a durable or cluster-wide barrier.
    pub fn with_import_limits(
        mut self,
        in_flight: usize,
        backlog_watermark: usize,
    ) -> Result<Self> {
        if in_flight == 0 {
            return Err(Error::invalid_argument());
        }
        self.import_max_in_flight_batches = in_flight;
        self.import_backlog_watermark = backlog_watermark;
        Ok(self)
    }

    /// Overrides the positive Structure Maintenance stalled timeout.
    pub fn with_stalled_timeout(mut self, timeout: Duration) -> Result<Self> {
        if timeout.is_zero() {
            return Err(Error::invalid_argument());
        }
        self.stalled_timeout = Some(timeout);
        Ok(self)
    }

    /// Sets the inclusive retry backoff range.
    pub fn with_retry_backoff(mut self, initial: Duration, maximum: Duration) -> Result<Self> {
        if initial.is_zero() || initial > maximum {
            return Err(Error::invalid_argument());
        }
        self.retry_initial_backoff = initial;
        self.retry_max_backoff = maximum;
        Ok(self)
    }

    /// Validates all process-local resource bounds.
    pub fn validate(&self) -> Result<()> {
        if self.foreground_operation_limit == 0
            || self.foreground_operation_limit > MAX_FOREGROUND_OPERATION_LIMIT
            || self.fixup_queue_capacity == 0
            || self.fixup_queue_capacity < self.maintenance_workers
            || self.fixup_attempts == 0
            || self.foreground_attempts == 0
            || usize::try_from(self.partition_cache_bytes).is_err()
            || self.tree_key_scan_ranges == 0
            || self.write_beam_size == 0
            || self.write_beam_size > MAX_WRITE_BEAM_SIZE
            || self.import_max_in_flight_batches == 0
            || self.import_backlog_watermark > self.fixup_queue_capacity
            || self.retry_initial_backoff.is_zero()
            || self.retry_initial_backoff > self.retry_max_backoff
        {
            return Err(Error::invalid_argument());
        }
        self.default_search_budgets.validate_hard_caps()
    }

    /// Returns each bound for running and waiting foreground operations.
    #[must_use]
    pub const fn foreground_operation_limit(&self) -> usize {
        self.foreground_operation_limit
    }

    /// Returns the maintenance worker count; zero disables background
    /// Structure Maintenance.
    #[must_use]
    pub const fn maintenance_workers(&self) -> usize {
        self.maintenance_workers
    }

    /// Returns the pending/running Fixup capacity.
    #[must_use]
    pub const fn fixup_queue_capacity(&self) -> usize {
        self.fixup_queue_capacity
    }

    /// Returns the whole-operation Fixup attempt limit.
    #[must_use]
    pub const fn fixup_attempts(&self) -> u32 {
        self.fixup_attempts
    }

    /// Returns the whole-operation Foreground Mutation attempt limit.
    #[must_use]
    pub const fn foreground_attempts(&self) -> u32 {
        self.foreground_attempts
    }

    /// Returns the decoded partition cache capacity in bytes.
    #[must_use]
    pub const fn partition_cache_bytes(&self) -> u64 {
        self.partition_cache_bytes
    }

    /// Returns the default Search Budgets.
    #[must_use]
    pub const fn default_search_budgets(&self) -> SearchBudgets {
        self.default_search_budgets
    }

    /// Returns the bounded Tree Key range plan limit.
    #[must_use]
    pub const fn tree_key_scan_ranges(&self) -> u32 {
        self.tree_key_scan_ranges
    }

    /// Returns the per-level beam used by foreground insert and upsert
    /// routing.
    #[must_use]
    pub const fn write_beam_size(&self) -> u32 {
        self.write_beam_size
    }

    /// Returns the Import Session adaptive-concurrency ceiling.
    #[must_use]
    pub const fn import_max_in_flight_batches(&self) -> usize {
        self.import_max_in_flight_batches
    }

    /// Returns the Import Session backlog watermark set by
    /// [`Self::with_import_limits`].
    #[must_use]
    pub const fn import_backlog_watermark(&self) -> usize {
        self.import_backlog_watermark
    }

    /// Resolves the stalled timeout for one Logical Index.
    ///
    /// Without an override, v1 uses checked
    /// `max(1 ms, 1 s * max_partition_entries / 128)`.
    pub fn stalled_timeout(&self, index: &IndexConfig) -> Result<Duration> {
        if let Some(timeout) = self.stalled_timeout {
            return Ok(timeout);
        }
        Duration::from_secs(1)
            .checked_mul(index.max_partition_entries())
            .map(|timeout| (timeout / 128).max(Duration::from_millis(1)))
            .ok_or_else(Error::invalid_argument)
    }

    /// Returns the first retry backoff interval.
    #[must_use]
    pub const fn retry_initial_backoff(&self) -> Duration {
        self.retry_initial_backoff
    }

    /// Returns the maximum retry backoff interval.
    #[must_use]
    pub const fn retry_max_backoff(&self) -> Duration {
        self.retry_max_backoff
    }
}
