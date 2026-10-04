//! Optional deterministic fault controls for the production memory adapter.
//!
//! Enable `test-support` only in test dependencies. Durable restart is a
//! simulation that shares an immutable snapshot, not disk persistence.

use crate::{BUDGET, HARD_LIMITS, MemoryBackend, State};
use bytes::Bytes;
use ktann::api::{Error, ErrorKind, Result};
use ktann::storage::backend::{AdmissionBudget, Capabilities, HardLimits, ScanLimits};
use ktann::storage::keys::KeyRange;
use std::collections::{BTreeMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

/// Whether committed data survives a simulated process restart.
///
/// A [`MemoryBackend::reopen`] of a [`Durable`](Durability::Durable)
/// backend carries the committed keyspace forward; an
/// [`Ephemeral`](Durability::Ephemeral) backend always restarts empty.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Durability {
    /// Committed data is lost on restart.
    Ephemeral,
    /// Committed data survives restart.
    Durable,
}

/// One step of the replayable commit fault plan.
///
/// The plan is consumed in order by [`ktann::storage::backend::WriteTxn::commit`]: each
/// commit pops the next step, or behaves as [`Normal`](CommitFault::Normal)
/// once the plan is exhausted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommitFault {
    /// Commit proceeds normally: conflict detection runs and the mutation is
    /// applied on success.
    Normal,
    /// A definite failure: nothing is applied and commit reports
    /// `ErrorKind::RetryableAbort`.
    Abort,
    /// If conflict validation succeeds, the mutation is applied but commit
    /// reports `ErrorKind::CommitOutcomeUnknown`. A conflicting transaction
    /// still aborts without applying its mutations.
    UnknownApplied,
    /// Nothing is applied but commit reports `ErrorKind::CommitOutcomeUnknown`.
    UnknownNotApplied,
}

/// The resolved outcome of one commit attempt, recorded in the history.
///
/// This is distinct from [`CommitFault`]: a `Normal` step can resolve to
/// [`Committed`](CommitOutcome::Committed) or [`Aborted`](CommitOutcome::Aborted)
/// depending on conflict detection, and a commit that would exceed a database
/// capacity limit resolves to [`LimitExceeded`](CommitOutcome::LimitExceeded).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CommitOutcome {
    /// Definite success; the mutation is applied.
    Committed,
    /// Definite failure; nothing is applied (`RetryableAbort`).
    Aborted,
    /// Unknown outcome; the mutation is applied.
    UnknownApplied,
    /// Unknown outcome; nothing is applied.
    UnknownNotApplied,
    /// A database capacity limit was exceeded; nothing is applied.
    LimitExceeded,
}

/// One redacted history entry describing a commit attempt.
///
/// Entries carry only counts and a deterministic content fingerprint, never raw
/// keys or values, so a failing run can be compared against a replay without
/// leaking caller data.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HistoryEntry {
    /// The committed version after this attempt (unchanged when nothing was
    /// applied).
    pub version: u64,
    /// The resolved commit outcome.
    pub outcome: CommitOutcome,
    /// The number of mutation operations charged in the transaction.
    pub mutations: usize,
    /// The total charged key-plus-value bytes.
    pub mutation_bytes: usize,
    /// The number of distinct keys in the transaction's mutation overlay.
    pub distinct_keys: usize,
    /// A deterministic fingerprint of the staged point and range mutations.
    pub fingerprint: u64,
}

/// Test-only limits and simulated capabilities for a [`MemoryBackend`].
///
/// The three public contract values (`hard_limits`, `admission_budget`, and
/// `capabilities`) mirror the [`Backend`](ktann::storage::backend::Backend)
/// accessors; the remaining fields are
/// explicit test-backend resource bounds so every unbounded dimension of the
/// model has a configurable ceiling and a boundary test.
#[derive(Clone, Copy, Debug)]
pub struct TestConfig {
    /// Stable hard key/value ceilings (see [`HardLimits`]).
    pub hard_limits: HardLimits,
    /// Conservative admission budget (see [`AdmissionBudget`]).
    pub admission_budget: AdmissionBudget,
    /// Declared capabilities (see [`Capabilities`]).
    pub capabilities: Capabilities,
    /// Whether committed data survives [`MemoryBackend::reopen`].
    pub durability: Durability,
    /// Maximum number of simultaneously open transactions (reads and writes).
    pub max_active_transactions: usize,
    /// Maximum number of retained committed versions kept for conflict
    /// detection. Older transactions become "too old" and their commit is
    /// rejected with `RetryableAbort` once their read version is evicted.
    pub max_retained_versions: usize,
    /// Maximum number of diagnostic history entries retained.
    pub max_history_entries: usize,
    /// Backend ceiling on the item count of one scan page.
    pub max_scan_page_items: usize,
    /// Backend ceiling on the byte total of one scan page.
    pub max_scan_page_bytes: usize,
    /// Maximum input length of `batch_get`, `batch_get_for_update`,
    /// `batch_scan`, and `batch_mutate`.
    pub max_batch_size: usize,
    /// Maximum number of distinct keys in one transaction's conflict set.
    pub max_read_set: usize,
    /// Maximum number of distinct point keys plus logical range clears in one
    /// transaction's mutation overlay.
    pub max_mutation_buffer: usize,
    /// Maximum number of distinct committed keys.
    pub max_db_keys: usize,
    /// Maximum total committed key-plus-value bytes.
    pub max_db_bytes: usize,
    /// Maximum number of fault-plan steps accepted by
    /// [`MemoryBackend::push_fault`] and
    /// [`MemoryBackend::set_fault_plan`].
    pub max_fault_plan: usize,
}

impl TestConfig {
    /// Constructs a configuration from the public contract values, applying
    /// generous defaults for every test-backend resource bound.
    #[must_use]
    pub fn new(
        hard_limits: HardLimits,
        admission_budget: AdmissionBudget,
        capabilities: Capabilities,
    ) -> Self {
        Self {
            hard_limits,
            admission_budget,
            capabilities,
            durability: Durability::Ephemeral,
            max_active_transactions: 1_024,
            max_retained_versions: 1_024,
            max_history_entries: 1_024,
            max_scan_page_items: 10_000,
            max_scan_page_bytes: 80 * 1_024,
            max_batch_size: 10_000,
            max_read_set: 10_000,
            max_mutation_buffer: 10_000,
            max_db_keys: 1_000_000,
            max_db_bytes: 1 << 30,
            max_fault_plan: 1_024,
        }
    }
}

impl Default for TestConfig {
    fn default() -> Self {
        Self::new(
            HardLimits {
                max_key_bytes: 1_024,
                max_value_bytes: 4_096,
            },
            AdmissionBudget {
                max_mutations: 1_000,
                max_mutation_bytes: 1 << 20,
                mutation_key_overhead_bytes: 0,
            },
            Capabilities {
                transactional_clear_range: false,
            },
        )
    }
}

/// Native operation call counts since the last reset.
///
/// Every counter increments once per backend-native call, regardless of how
/// many keys the call carries; the counts make read/write amplification
/// observable in deterministic tests.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct OperationCounts {
    /// Point `get` calls.
    pub get: usize,
    /// `batch_get` calls.
    pub batch_get: usize,
    /// Update-protected `get_for_update` calls.
    pub get_for_update: usize,
    /// `batch_get_for_update` calls.
    pub batch_get_for_update: usize,
    /// `put` calls.
    pub put: usize,
    /// Unique `insert` calls.
    pub insert: usize,
    /// `delete` calls.
    pub delete: usize,
    /// `batch_mutate` calls.
    pub batch_mutate: usize,
    /// `scan` calls.
    pub scan: usize,
    /// `batch_scan` calls.
    pub batch_scan: usize,
    /// Range legs submitted across all `batch_scan` calls.
    pub batch_scan_ranges: usize,
    /// `clear_range` calls.
    pub clear_range: usize,
}

impl TestConfig {
    pub(crate) fn production() -> Self {
        Self {
            max_active_transactions: usize::MAX,
            max_retained_versions: usize::MAX,
            max_scan_page_items: usize::MAX,
            max_scan_page_bytes: usize::MAX,
            max_batch_size: usize::MAX,
            max_read_set: usize::MAX,
            max_mutation_buffer: usize::MAX,
            max_db_keys: usize::MAX,
            max_db_bytes: usize::MAX,
            ..Self::new(
                HARD_LIMITS,
                BUDGET,
                Capabilities {
                    transactional_clear_range: false,
                },
            )
        }
    }
}

/// Shared controls; absent entirely from builds without `test-support`.
pub(crate) struct Control {
    pub(crate) config: TestConfig,
    counts: Mutex<OperationCounts>,
    active: AtomicUsize,
}

impl Default for Control {
    fn default() -> Self {
        Self::new(TestConfig::production())
    }
}

impl Control {
    pub(crate) fn new(config: TestConfig) -> Self {
        assert!(config.max_retained_versions > 0);
        assert!(config.max_history_entries > 0);
        assert!(config.max_scan_page_items > 0);
        assert!(config.max_scan_page_bytes > 0);
        Self {
            config,
            counts: Mutex::default(),
            active: AtomicUsize::new(0),
        }
    }

    pub(crate) fn count(&self, update: impl FnOnce(&mut OperationCounts)) {
        update(&mut self.counts.lock().expect("counts lock poisoned"));
    }

    pub(crate) fn admit(self: &Arc<Self>) -> Result<Lease> {
        self.active
            .try_update(Ordering::AcqRel, Ordering::Acquire, |n| {
                (n < self.config.max_active_transactions).then(|| n + 1)
            })
            .map_err(|_| limit_exceeded())?;
        Ok(Lease(Arc::clone(self)))
    }

    pub(crate) fn check_batch(&self, len: usize) -> Result<()> {
        if len > self.config.max_batch_size {
            return Err(limit_exceeded());
        }
        Ok(())
    }

    pub(crate) fn scan_limits(&self, limits: ScanLimits) -> ScanLimits {
        ScanLimits {
            item_limit: limits.item_limit.min(self.config.max_scan_page_items),
            byte_limit: limits.byte_limit.min(self.config.max_scan_page_bytes),
        }
    }
}

pub(crate) struct Lease(pub(crate) Arc<Control>);

impl Drop for Lease {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Diagnostic state guarded by the same lock as the committed data.
#[derive(Default)]
pub(crate) struct Diagnostics {
    pub(crate) faults: VecDeque<CommitFault>,
    entries: VecDeque<HistoryEntry>,
    truncated: bool,
    pub(crate) db_bytes: usize,
}

impl Diagnostics {
    pub(crate) fn record(&mut self, entry: HistoryEntry, capacity: usize) {
        if self.entries.len() == capacity {
            self.entries.pop_front();
            self.truncated = true;
        }
        self.entries.push_back(entry);
    }
}

impl MemoryBackend {
    /// Creates an isolated backend with explicit test controls and simulated limits.
    #[must_use]
    pub fn with_test_config(config: TestConfig) -> Self {
        let control = Arc::new(Control::new(config));
        let state = State {
            max_versions: Some(config.max_retained_versions),
            ..State::default()
        };
        Self {
            state: Arc::new(Mutex::new(state)),
            control,
        }
    }

    /// Native operation counts since the last reset.
    #[must_use]
    pub fn operation_counts(&self) -> OperationCounts {
        *self.control.counts.lock().expect("counts lock poisoned")
    }

    /// Resets operation counts without changing data or faults.
    pub fn reset_operation_counts(&self) {
        *self.control.counts.lock().expect("counts lock poisoned") = OperationCounts::default();
    }

    /// Appends a fault for the next commit attempt.
    pub fn push_fault(&self, fault: CommitFault) -> Result<()> {
        let mut state = self.lock()?;
        if state.diagnostics.faults.len() >= self.control.config.max_fault_plan {
            return Err(limit_exceeded());
        }
        state.diagnostics.faults.push_back(fault);
        Ok(())
    }

    /// Replaces the replayable commit fault plan.
    pub fn set_fault_plan(&self, plan: Vec<CommitFault>) -> Result<()> {
        if plan.len() > self.control.config.max_fault_plan {
            return Err(limit_exceeded());
        }
        self.lock()?.diagnostics.faults = plan.into();
        Ok(())
    }

    /// Number of live read and write transactions.
    #[must_use]
    pub fn active_transactions(&self) -> usize {
        self.control.active.load(Ordering::Acquire)
    }

    /// Number of committed keys.
    #[must_use]
    pub fn db_key_count(&self) -> usize {
        self.lock().expect("state lock poisoned").data.len()
    }

    /// Sum of committed key and value lengths.
    #[must_use]
    pub fn db_byte_count(&self) -> usize {
        self.lock()
            .expect("state lock poisoned")
            .diagnostics
            .db_bytes
    }

    /// Retained redacted commit outcomes in attempt order.
    #[must_use]
    pub fn history(&self) -> Vec<HistoryEntry> {
        self.lock()
            .expect("state lock poisoned")
            .diagnostics
            .entries
            .iter()
            .copied()
            .collect()
    }

    /// Whether any diagnostic history has been evicted.
    #[must_use]
    pub fn history_truncated(&self) -> bool {
        self.lock()
            .expect("state lock poisoned")
            .diagnostics
            .truncated
    }

    /// Simulates a restart with a fresh fault plan, history, and counters.
    /// Durable simulation carries the current immutable keyspace to the new instance.
    #[must_use]
    pub fn reopen(&self) -> Self {
        let reopened = Self::with_test_config(self.control.config);
        if self.control.config.durability == Durability::Durable {
            let source = self.lock().expect("state lock poisoned");
            let mut target = reopened.lock().expect("state lock poisoned");
            target.data = source.data.clone();
            target.diagnostics.db_bytes = source.diagnostics.db_bytes;
        }
        reopened
    }
}

pub(crate) fn limit_exceeded() -> Error {
    Error::new(ErrorKind::LimitExceeded)
}

pub(crate) fn merge_clear_range(clear_ranges: &[KeyRange], added: &KeyRange) -> Vec<KeyRange> {
    let mut merged = Vec::with_capacity(clear_ranges.len().saturating_add(1));
    let mut start = added.start().to_vec();
    let mut end = added.end().to_vec();
    let mut inserted = false;

    for range in clear_ranges {
        if range.end() < start.as_slice() {
            merged.push(range.clone());
        } else if end.as_slice() < range.start() {
            if !inserted {
                merged.push(KeyRange::new(start.clone(), end.clone()));
                inserted = true;
            }
            merged.push(range.clone());
        } else {
            if range.start() < start.as_slice() {
                start = range.start().to_vec();
            }
            if range.end() > end.as_slice() {
                end = range.end().to_vec();
            }
        }
    }

    if !inserted {
        merged.push(KeyRange::new(start, end));
    }
    merged
}

pub(crate) fn range_contains(range: &KeyRange, key: &[u8]) -> bool {
    range.start() <= key && key < range.end()
}

/// A deterministic 64-bit FNV-1a hasher for history fingerprints.
struct Fnv1a(u64);

impl Fnv1a {
    fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }

    fn write(&mut self, bytes: &[u8]) {
        for &byte in bytes {
            self.0 ^= u64::from(byte);
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }

    /// Writes a length-delimited field so adjacent fields cannot collide.
    fn write_field(&mut self, bytes: &[u8]) {
        let length = u64::try_from(bytes.len())
            .expect("supported Rust targets use no more than 64-bit usize");
        self.write(&length.to_le_bytes());
        self.write(bytes);
    }

    fn finish(self) -> u64 {
        self.0
    }
}

/// A deterministic content fingerprint of staged mutations, without raw keys
/// or values.
pub(crate) fn fingerprint(
    clear_ranges: &[KeyRange],
    pending: &BTreeMap<Bytes, Option<Bytes>>,
) -> u64 {
    let mut hasher = Fnv1a::new();
    hasher.write(b"ktann-deterministic-history-v1");
    for range in clear_ranges {
        hasher.write(&[3]);
        hasher.write_field(range.start());
        hasher.write_field(range.end());
    }
    for (key, value) in pending {
        match value {
            Some(value) => {
                hasher.write(&[1]);
                hasher.write_field(key);
                hasher.write_field(value);
            }
            None => {
                hasher.write(&[2]);
                hasher.write_field(key);
            }
        }
    }
    hasher.finish()
}
