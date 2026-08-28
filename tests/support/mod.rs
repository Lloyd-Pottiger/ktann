//! Shared test support for KTANN integration tests.
//!
//! This module owns the deterministic in-memory transactional KV backend used
//! to exercise the backend-neutral transaction contract. It lives only inside
//! `tests/`, per ADR 0002: it is a test double, not a public production
//! backend and not a persistent-format commitment.
//!
//! The backend models the full contract — versioned snapshots, read-your-writes,
//! update-protected point-read conflicts (including ABA), ordered half-open
//! scans, unique insertion, batch mutation, rollback, hard limits and admission
//! budgets, the range-clear capability, and explicit commit outcomes — while
//! additionally providing test-only seams for replayable fault injection,
//! bounded resource configuration, restart semantics, and a redacted diagnostic
//! history.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use ktann::api::{Error, ErrorKind, LogicalIndexId, Result, RuntimeConfig};
use ktann::storage::ReadLogicalTxn;
use ktann::storage::backend::{
    AdmissionBudget, Backend, Capabilities, CommitStart, HardLimits, InsertOutcome, Mutation,
    ReadOps, ReadTxn, ScanItem, ScanLimits, ScanPage, WriteTxn,
};
use ktann::storage::keys::{KeyRange, LogicalKey};
use ktann::storage::values::{IndexManifest, PersistentValue};

pub mod audit;
pub mod datadriven;
pub mod dataset;
pub mod fixtures;
pub mod load_index;
pub mod observe;
pub mod oracle;

/// A Runtime configuration without background maintenance workers.
///
/// Tests that drive the split/merge state machines one bounded transition at
/// a time — or assert exact intermediate topology under fixtures — use this
/// configuration so demand-driven Fixup scheduling cannot race their manual
/// drives. Scheduling itself is covered by `maintenance_scheduling.rs`.
pub fn manual_maintenance_config() -> RuntimeConfig {
    RuntimeConfig::default()
        .with_maintenance(0, 1)
        .and_then(|config| config.with_import_limits(1, 1))
        .expect("valid manual-maintenance config")
}

/// A committed keyspace snapshot mapping encoded keys to values.
type Keyspace = BTreeMap<Vec<u8>, Vec<u8>>;
/// A transaction's mutation overlay mapping keys to `Some(value)` or `None`
/// (delete).
type Overlay = BTreeMap<Vec<u8>, Option<Vec<u8>>>;

/// Whether committed data survives a simulated process restart.
///
/// A [`DeterministicBackend::reopen`] of a [`Durable`](Durability::Durable)
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
/// The plan is consumed in order by [`DeterministicWriteTxn::commit`]: each
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

/// A bounded, redacted ring of [`HistoryEntry`] values.
///
/// When the ring is full, the oldest entry is dropped and the truncation is
/// permanently exposed via [`History::truncated`]; a truncated history is never
/// presented as a complete replay.
#[derive(Debug)]
pub struct History {
    entries: VecDeque<HistoryEntry>,
    capacity: usize,
    truncated: bool,
}

impl History {
    /// Creates an empty history retaining at most `capacity` entries.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            entries: VecDeque::with_capacity(capacity),
            capacity,
            truncated: false,
        }
    }

    /// Pushes one entry, evicting the oldest entry once the ring is full.
    fn push(&mut self, entry: HistoryEntry) {
        if self.entries.len() >= self.capacity {
            self.entries.pop_front();
            self.truncated = true;
        }
        self.entries.push_back(entry);
    }

    /// Iterates over retained entries in commit order.
    pub fn entries(&self) -> impl Iterator<Item = &HistoryEntry> {
        self.entries.iter()
    }

    /// Returns `true` once any entry has been evicted from the ring.
    pub fn truncated(&self) -> bool {
        self.truncated
    }
}

/// Configuration for a [`DeterministicBackend`].
///
/// The three public contract values (`hard_limits`, `admission_budget`, and
/// `capabilities`) mirror the [`Backend`] accessors; the remaining fields are
/// explicit test-backend resource bounds so every unbounded dimension of the
/// model has a configurable ceiling and a boundary test.
#[derive(Clone, Copy, Debug)]
pub struct DeterministicConfig {
    /// Stable hard key/value ceilings (see [`HardLimits`]).
    pub hard_limits: HardLimits,
    /// Conservative admission budget (see [`AdmissionBudget`]).
    pub admission_budget: AdmissionBudget,
    /// Declared capabilities (see [`Capabilities`]).
    pub capabilities: Capabilities,
    /// Whether committed data survives [`DeterministicBackend::reopen`].
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
    /// Maximum input length of `batch_get`, `batch_get_for_update`, and
    /// `batch_mutate`.
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
    /// [`DeterministicBackend::push_fault`] and
    /// [`DeterministicBackend::set_fault_plan`].
    pub max_fault_plan: usize,
}

impl DeterministicConfig {
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

impl Default for DeterministicConfig {
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
    /// `clear_range` calls.
    pub clear_range: usize,
}

/// A deterministic in-memory transactional KV backend.
///
/// Committed state is an immutable `Arc<BTreeMap>` snapshot, replaced under one
/// mutex per commit so readers never observe a partial install. Point keys
/// record their last-write version for update-protected conflict detection,
/// including ABA changes. A bounded version log drives conflict retention, and
/// a bounded ring records a redacted diagnostic history.
pub struct DeterministicBackend {
    config: DeterministicConfig,
    state: Mutex<State>,
    counts: Mutex<OperationCounts>,
}

struct State {
    /// The last committed version (0 before the first commit).
    version: u64,
    /// The current head keyspace snapshot.
    committed: Arc<Keyspace>,
    /// Total committed key-plus-value bytes.
    db_bytes: usize,
    /// `key -> version` of the most recent retained write, for ABA detection.
    last_written: BTreeMap<Vec<u8>, u64>,
    /// Retained version log, oldest first, for conflict GC.
    versions: VecDeque<VersionRecord>,
    /// Redacted diagnostic history.
    history: History,
    /// Number of currently open transactions.
    active_txns: usize,
    /// Replayable commit fault plan, consumed in order.
    fault_plan: VecDeque<CommitFault>,
}

struct VersionRecord {
    version: u64,
    written_keys: Vec<Vec<u8>>,
    cleared_ranges: Vec<KeyRange>,
}

impl DeterministicBackend {
    /// Constructs a backend from `config`, validating the bounds that must be
    /// at least one for the model to remain coherent.
    #[must_use]
    pub fn new(config: DeterministicConfig) -> Self {
        assert!(
            config.max_retained_versions >= 1,
            "max_retained_versions must be >= 1"
        );
        assert!(
            config.max_scan_page_items >= 1,
            "max_scan_page_items must be >= 1"
        );
        assert!(
            config.max_scan_page_bytes >= 1,
            "max_scan_page_bytes must be >= 1"
        );
        assert!(
            config.max_history_entries >= 1,
            "max_history_entries must be >= 1"
        );
        let state = State {
            version: 0,
            committed: Arc::new(BTreeMap::new()),
            db_bytes: 0,
            last_written: BTreeMap::new(),
            versions: VecDeque::new(),
            history: History::with_capacity(config.max_history_entries),
            active_txns: 0,
            fault_plan: VecDeque::new(),
        };
        Self {
            config,
            state: Mutex::new(state),
            counts: Mutex::new(OperationCounts::default()),
        }
    }

    /// Counts one native operation call.
    fn count(&self, update: impl FnOnce(&mut OperationCounts)) {
        let mut counts = self.counts.lock().expect("counts lock poisoned");
        update(&mut counts);
    }

    /// Returns the native operation call counts since the last reset.
    #[must_use]
    pub fn operation_counts(&self) -> OperationCounts {
        *self.counts.lock().expect("counts lock poisoned")
    }

    /// Resets the native operation call counters.
    pub fn reset_operation_counts(&self) {
        *self.counts.lock().expect("counts lock poisoned") = OperationCounts::default();
    }

    /// Appends one fault step to the plan, bounded by `max_fault_plan`.
    pub fn push_fault(&self, fault: CommitFault) -> Result<()> {
        let mut state = self.state.lock().expect("state lock poisoned");
        if state.fault_plan.len() >= self.config.max_fault_plan {
            return Err(limit_exceeded());
        }
        state.fault_plan.push_back(fault);
        Ok(())
    }

    /// Replaces the fault plan, bounded by `max_fault_plan`.
    pub fn set_fault_plan(&self, plan: Vec<CommitFault>) -> Result<()> {
        if plan.len() > self.config.max_fault_plan {
            return Err(limit_exceeded());
        }
        let mut state = self.state.lock().expect("state lock poisoned");
        state.fault_plan = plan.into_iter().collect();
        Ok(())
    }

    /// The number of currently open transactions.
    #[must_use]
    pub fn active_transactions(&self) -> usize {
        self.state.lock().expect("state lock poisoned").active_txns
    }

    /// The number of distinct committed keys.
    #[must_use]
    pub fn db_key_count(&self) -> usize {
        self.state
            .lock()
            .expect("state lock poisoned")
            .committed
            .len()
    }

    /// The total committed key-plus-value bytes.
    #[must_use]
    pub fn db_byte_count(&self) -> usize {
        self.state.lock().expect("state lock poisoned").db_bytes
    }

    /// A snapshot of the retained diagnostic history in commit order.
    #[must_use]
    pub fn history(&self) -> Vec<HistoryEntry> {
        self.state
            .lock()
            .expect("state lock poisoned")
            .history
            .entries()
            .copied()
            .collect()
    }

    /// Returns `true` once the history ring has evicted any entry.
    #[must_use]
    pub fn history_truncated(&self) -> bool {
        self.state
            .lock()
            .expect("state lock poisoned")
            .history
            .truncated()
    }

    /// Simulates a process restart, returning a fresh backend with the same
    /// configuration.
    ///
    /// A [`Durable`](Durability::Durable) backend carries its committed
    /// keyspace forward; an [`Ephemeral`](Durability::Ephemeral) backend
    /// restarts empty. The fault plan and diagnostic history are process-local
    /// and are always reset.
    #[must_use]
    pub fn reopen(&self) -> DeterministicBackend {
        let committed = match self.config.durability {
            Durability::Durable => {
                let state = self.state.lock().expect("state lock poisoned");
                state.committed.clone()
            }
            Durability::Ephemeral => Arc::new(BTreeMap::new()),
        };
        let db_bytes =
            try_sum_bytes(&committed).expect("reopened committed byte accounting overflows usize");
        let state = State {
            version: 0,
            committed,
            db_bytes,
            last_written: BTreeMap::new(),
            versions: VecDeque::new(),
            history: History::with_capacity(self.config.max_history_entries),
            active_txns: 0,
            fault_plan: VecDeque::new(),
        };
        DeterministicBackend {
            config: self.config,
            state: Mutex::new(state),
            counts: Mutex::new(OperationCounts::default()),
        }
    }
}

impl Default for DeterministicBackend {
    fn default() -> Self {
        Self::new(DeterministicConfig::default())
    }
}

impl fmt::Debug for DeterministicBackend {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeterministicBackend")
            .field("durability", &self.config.durability)
            .field("hard_limits", &self.config.hard_limits)
            .field("admission_budget", &self.config.admission_budget)
            .field("capabilities", &self.config.capabilities)
            .finish_non_exhaustive()
    }
}

/// A consistent read transaction over one immutable snapshot.
///
/// The `PhantomData<Cell<()>>` field makes the type `Send` but not `Sync`,
/// proving the transaction interface requires only `Send`.
pub struct DeterministicReadTxn<'backend> {
    backend: &'backend DeterministicBackend,
    snapshot: Arc<Keyspace>,
    _not_sync: PhantomData<Cell<()>>,
}

impl fmt::Debug for DeterministicReadTxn<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeterministicReadTxn")
            .finish_non_exhaustive()
    }
}

impl Drop for DeterministicReadTxn<'_> {
    fn drop(&mut self) {
        self.backend
            .state
            .lock()
            .expect("state lock poisoned")
            .active_txns -= 1;
    }
}

impl DeterministicReadTxn<'_> {
    fn get_value(&self, key: &Bytes) -> Option<Bytes> {
        self.snapshot
            .get(&key[..])
            .map(|value| Bytes::copy_from_slice(value))
    }
}

/// An atomic write transaction with a snapshot and an uncommitted overlay.
pub struct DeterministicWriteTxn<'backend> {
    backend: &'backend DeterministicBackend,
    /// The committed version this transaction reads from.
    version: u64,
    snapshot: Arc<Keyspace>,
    /// Overlay of pending mutations; `None` marks a delete.
    pending: Overlay,
    /// Logical range clears, applied before `pending` so later point mutations
    /// win over earlier clears.
    clear_ranges: Vec<KeyRange>,
    /// Keys read via `get_for_update`, establishing point conflicts.
    read_set: BTreeSet<Vec<u8>>,
    mutation_count: usize,
    mutation_bytes: usize,
    _not_sync: PhantomData<Cell<()>>,
}

impl fmt::Debug for DeterministicWriteTxn<'_> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeterministicWriteTxn")
            .field("version", &self.version)
            .field("mutation_count", &self.mutation_count)
            .field("mutation_bytes", &self.mutation_bytes)
            .field("read_set_len", &self.read_set.len())
            .field("pending_len", &self.pending.len())
            .field("clear_ranges_len", &self.clear_ranges.len())
            .finish_non_exhaustive()
    }
}

impl Drop for DeterministicWriteTxn<'_> {
    fn drop(&mut self) {
        self.backend
            .state
            .lock()
            .expect("state lock poisoned")
            .active_txns -= 1;
    }
}

impl DeterministicWriteTxn<'_> {
    /// Reads through the pending overlay for read-your-writes.
    fn lookup(&self, key: &[u8]) -> Option<Bytes> {
        if let Some(entry) = self.pending.get(key) {
            return entry.as_ref().map(|value| Bytes::copy_from_slice(value));
        }
        if self
            .clear_ranges
            .iter()
            .any(|range| range_contains(range, key))
        {
            return None;
        }
        self.snapshot
            .get(key)
            .map(|value| Bytes::copy_from_slice(value))
    }

    /// Rejects a key or value that exceeds the backend's hard limits.
    fn check_key_value(&self, key: &[u8], value: Option<&[u8]>) -> Result<()> {
        check_key(&self.backend.config, key)?;
        if let Some(value) = value {
            if value.len() > self.backend.config.hard_limits.max_value_bytes {
                return Err(limit_exceeded());
            }
        }
        Ok(())
    }

    /// Charges mutation operations and bytes against the admission budget.
    fn charge(&mut self, count: usize, bytes: usize) -> Result<()> {
        let budget = self.backend.config.admission_budget;
        let next_count = self
            .mutation_count
            .checked_add(count)
            .ok_or_else(limit_exceeded)?;
        let next_bytes = self
            .mutation_bytes
            .checked_add(bytes)
            .ok_or_else(limit_exceeded)?;
        if next_count > budget.max_mutations || next_bytes > budget.max_mutation_bytes {
            return Err(limit_exceeded());
        }
        self.mutation_count = next_count;
        self.mutation_bytes = next_bytes;
        Ok(())
    }

    /// Establishes a point conflict on `key`, bounded by `max_read_set`.
    fn add_read(&mut self, key: &[u8]) -> Result<()> {
        if self.read_set.len() >= self.backend.config.max_read_set && !self.read_set.contains(key) {
            return Err(limit_exceeded());
        }
        self.read_set.insert(key.to_vec());
        Ok(())
    }

    /// Rejects pending-overlay growth past `max_mutation_buffer`.
    fn check_buffer_growth(&self, new_keys: &BTreeSet<Vec<u8>>) -> Result<()> {
        let added = new_keys
            .iter()
            .filter(|key| !self.pending.contains_key(*key))
            .count();
        let current = self
            .pending
            .len()
            .checked_add(self.clear_ranges.len())
            .ok_or_else(limit_exceeded)?;
        let next = current.checked_add(added).ok_or_else(limit_exceeded)?;
        if next > self.backend.config.max_mutation_buffer {
            return Err(limit_exceeded());
        }
        Ok(())
    }

    /// Normalizes one additional clear and enforces the mutation-buffer bound.
    fn prepare_clear_ranges(&self, range: &KeyRange) -> Result<Vec<KeyRange>> {
        let clear_ranges = merge_clear_range(&self.clear_ranges, range);
        let removed = self
            .pending
            .keys()
            .filter(|key| range_contains(range, key))
            .count();
        let retained_points = self
            .pending
            .len()
            .checked_sub(removed)
            .ok_or_else(limit_exceeded)?;
        let next = retained_points
            .checked_add(clear_ranges.len())
            .ok_or_else(limit_exceeded)?;
        if next > self.backend.config.max_mutation_buffer {
            return Err(limit_exceeded());
        }
        Ok(clear_ranges)
    }

    /// Performs the commit under the single state lock, returning the outcome.
    fn commit_impl(&self) -> Result<()> {
        let mut state = self.backend.state.lock().expect("state lock poisoned");
        let config = &self.backend.config;

        let fault = state.fault_plan.pop_front().unwrap_or(CommitFault::Normal);

        let too_old = !self.read_set.is_empty() && self.version < evicted_through(&state);
        let can_apply = !too_old && !self.conflicts(&state);
        let (outcome, should_apply) = match fault {
            CommitFault::Normal if can_apply => (CommitOutcome::Committed, true),
            CommitFault::Normal => (CommitOutcome::Aborted, false),
            CommitFault::Abort => (CommitOutcome::Aborted, false),
            CommitFault::UnknownApplied if can_apply => (CommitOutcome::UnknownApplied, true),
            CommitFault::UnknownApplied => (CommitOutcome::Aborted, false),
            CommitFault::UnknownNotApplied => (CommitOutcome::UnknownNotApplied, false),
        };

        let mut entry = HistoryEntry {
            version: state.version,
            outcome,
            mutations: self.mutation_count,
            mutation_bytes: self.mutation_bytes,
            distinct_keys: self.pending.len(),
            fingerprint: fingerprint(&self.clear_ranges, &self.pending),
        };

        let applied_version = if should_apply {
            let new_map = apply_staged(&state.committed, &self.clear_ranges, &self.pending);
            let new_db_bytes = try_sum_bytes(&new_map)?;
            if new_map.len() > config.max_db_keys || new_db_bytes > config.max_db_bytes {
                entry.outcome = CommitOutcome::LimitExceeded;
                state.history.push(entry);
                return Err(limit_exceeded());
            }
            let new_version = state.version.checked_add(1).ok_or_else(limit_exceeded)?;
            state.version = new_version;
            state.committed = Arc::new(new_map);
            state.db_bytes = new_db_bytes;
            let written_keys: Vec<Vec<u8>> = self.pending.keys().cloned().collect();
            for key in &written_keys {
                state.last_written.insert(key.clone(), new_version);
            }
            state.versions.push_back(VersionRecord {
                version: new_version,
                written_keys,
                cleared_ranges: self.clear_ranges.clone(),
            });
            gc_versions(&mut state, config.max_retained_versions);
            entry.version = new_version;
            Some(new_version)
        } else {
            None
        };

        state.history.push(entry);

        match (outcome, applied_version) {
            (CommitOutcome::Committed, Some(_)) => Ok(()),
            (CommitOutcome::UnknownApplied, Some(_)) => {
                Err(Error::new(ErrorKind::CommitOutcomeUnknown))
            }
            (CommitOutcome::Aborted, None) => Err(Error::new(ErrorKind::RetryableAbort)),
            (CommitOutcome::UnknownNotApplied, None) => {
                Err(Error::new(ErrorKind::CommitOutcomeUnknown))
            }
            _ => unreachable!("commit outcome and application state must agree"),
        }
    }

    /// Returns `true` when any update-protected read was written after this
    /// transaction's read version.
    fn conflicts(&self, state: &State) -> bool {
        self.read_set.iter().any(|key| {
            state
                .last_written
                .get(key)
                .is_some_and(|v| *v > self.version)
                || state.versions.iter().any(|record| {
                    record.version > self.version
                        && record
                            .cleared_ranges
                            .iter()
                            .any(|range| range_contains(range, key))
                })
        })
    }
}

impl Backend for DeterministicBackend {
    type ReadTxn<'backend> = DeterministicReadTxn<'backend>;
    type WriteTxn<'backend> = DeterministicWriteTxn<'backend>;

    fn hard_limits(&self) -> HardLimits {
        self.config.hard_limits
    }

    fn admission_budget(&self) -> AdmissionBudget {
        self.config.admission_budget
    }

    fn capabilities(&self) -> Capabilities {
        self.config.capabilities
    }

    async fn begin_read(&self) -> Result<DeterministicReadTxn<'_>> {
        let (snapshot, _) = self.begin_common()?;
        Ok(DeterministicReadTxn {
            backend: self,
            snapshot,
            _not_sync: PhantomData,
        })
    }

    async fn begin_write(&self) -> Result<DeterministicWriteTxn<'_>> {
        let (snapshot, version) = self.begin_common()?;
        Ok(DeterministicWriteTxn {
            backend: self,
            version,
            snapshot,
            pending: BTreeMap::new(),
            clear_ranges: Vec::new(),
            read_set: BTreeSet::new(),
            mutation_count: 0,
            mutation_bytes: 0,
            _not_sync: PhantomData,
        })
    }
}

impl DeterministicBackend {
    /// Admits one transaction, returning its snapshot and read version.
    fn begin_common(&self) -> Result<(Arc<Keyspace>, u64)> {
        let mut state = self.state.lock().expect("state lock poisoned");
        if state.active_txns >= self.config.max_active_transactions {
            return Err(limit_exceeded());
        }
        state.active_txns = state
            .active_txns
            .checked_add(1)
            .ok_or_else(limit_exceeded)?;
        Ok((state.committed.clone(), state.version))
    }
}

impl ReadOps for DeterministicReadTxn<'_> {
    async fn get(&mut self, key: Bytes) -> Result<Option<Bytes>> {
        self.backend.count(|counts| counts.get += 1);
        check_key(&self.backend.config, &key)?;
        Ok(self.get_value(&key))
    }

    async fn batch_get(&mut self, keys: Vec<Bytes>) -> Result<Vec<Option<Bytes>>> {
        self.backend.count(|counts| counts.batch_get += 1);
        if keys.len() > self.backend.config.max_batch_size {
            return Err(limit_exceeded());
        }
        check_keys(&self.backend.config, &keys)?;
        Ok(keys.into_iter().map(|key| self.get_value(&key)).collect())
    }

    async fn scan(&mut self, range: &KeyRange, limits: ScanLimits) -> Result<ScanPage> {
        self.backend.count(|counts| counts.scan += 1);
        let (item_limit, byte_limit) = resolve_scan(limits, &self.backend.config)?;
        if range.start() >= range.end() {
            return Ok(ScanPage::terminal(Vec::new()));
        }
        check_range(&self.backend.config, range)?;
        scan_map(
            &self.snapshot,
            range,
            item_limit,
            byte_limit,
            self.backend.config.hard_limits.max_key_bytes,
        )
    }
}

impl ReadTxn for DeterministicReadTxn<'_> {}

impl ReadOps for DeterministicWriteTxn<'_> {
    async fn get(&mut self, key: Bytes) -> Result<Option<Bytes>> {
        self.backend.count(|counts| counts.get += 1);
        check_key(&self.backend.config, &key)?;
        Ok(self.lookup(&key))
    }

    async fn batch_get(&mut self, keys: Vec<Bytes>) -> Result<Vec<Option<Bytes>>> {
        self.backend.count(|counts| counts.batch_get += 1);
        if keys.len() > self.backend.config.max_batch_size {
            return Err(limit_exceeded());
        }
        check_keys(&self.backend.config, &keys)?;
        Ok(keys.into_iter().map(|key| self.lookup(&key)).collect())
    }

    async fn scan(&mut self, range: &KeyRange, limits: ScanLimits) -> Result<ScanPage> {
        self.backend.count(|counts| counts.scan += 1);
        let (item_limit, byte_limit) = resolve_scan(limits, &self.backend.config)?;
        if range.start() >= range.end() {
            return Ok(ScanPage::terminal(Vec::new()));
        }
        check_range(&self.backend.config, range)?;
        let merged = apply_staged(&self.snapshot, &self.clear_ranges, &self.pending);
        scan_map(
            &merged,
            range,
            item_limit,
            byte_limit,
            self.backend.config.hard_limits.max_key_bytes,
        )
    }
}

impl WriteTxn for DeterministicWriteTxn<'_> {
    async fn get_for_update(&mut self, key: Bytes) -> Result<Option<Bytes>> {
        self.backend.count(|counts| counts.get_for_update += 1);
        check_key(&self.backend.config, &key)?;
        self.add_read(&key)?;
        Ok(self.lookup(&key))
    }

    async fn batch_get_for_update(&mut self, keys: Vec<Bytes>) -> Result<Vec<Option<Bytes>>> {
        self.backend
            .count(|counts| counts.batch_get_for_update += 1);
        if keys.len() > self.backend.config.max_batch_size {
            return Err(limit_exceeded());
        }
        check_keys(&self.backend.config, &keys)?;
        for key in &keys {
            self.add_read(key)?;
        }
        Ok(keys.into_iter().map(|key| self.lookup(&key)).collect())
    }

    async fn put(&mut self, key: Bytes, value: Bytes) -> Result<()> {
        self.backend.count(|counts| counts.put += 1);
        self.check_key_value(&key, Some(&value))?;
        let new_keys = once_set(key.to_vec());
        self.check_buffer_growth(&new_keys)?;
        self.charge(
            1,
            key.len()
                .checked_add(value.len())
                .ok_or_else(limit_exceeded)?,
        )?;
        self.pending.insert(key.to_vec(), Some(value.to_vec()));
        Ok(())
    }

    async fn insert(&mut self, key: Bytes, value: Bytes) -> Result<InsertOutcome> {
        self.backend.count(|counts| counts.insert += 1);
        self.check_key_value(&key, Some(&value))?;
        self.add_read(&key)?;
        if self.lookup(&key).is_some() {
            return Ok(InsertOutcome::AlreadyExists);
        }
        let new_keys = once_set(key.to_vec());
        self.check_buffer_growth(&new_keys)?;
        self.charge(
            1,
            key.len()
                .checked_add(value.len())
                .ok_or_else(limit_exceeded)?,
        )?;
        self.pending.insert(key.to_vec(), Some(value.to_vec()));
        Ok(InsertOutcome::Inserted)
    }

    async fn delete(&mut self, key: Bytes) -> Result<()> {
        self.backend.count(|counts| counts.delete += 1);
        self.check_key_value(&key, None)?;
        let new_keys = once_set(key.to_vec());
        self.check_buffer_growth(&new_keys)?;
        self.charge(1, key.len())?;
        self.pending.insert(key.to_vec(), None);
        Ok(())
    }

    async fn batch_mutate(&mut self, mutations: Vec<Mutation>) -> Result<()> {
        self.backend.count(|counts| counts.batch_mutate += 1);
        if mutations.len() > self.backend.config.max_batch_size {
            return Err(limit_exceeded());
        }
        // Validate the whole batch before mutating any transaction state so a
        // capacity failure leaves no partial overlay.
        let mut count_delta = 0usize;
        let mut bytes_delta = 0usize;
        let mut new_keys = BTreeSet::new();
        let mut ops: Vec<(Vec<u8>, Option<Vec<u8>>)> = Vec::with_capacity(mutations.len());
        for mutation in mutations {
            match mutation {
                Mutation::Put { key, value } => {
                    self.check_key_value(&key, Some(&value))?;
                    let bytes = key
                        .len()
                        .checked_add(value.len())
                        .ok_or_else(limit_exceeded)?;
                    count_delta = count_delta.checked_add(1).ok_or_else(limit_exceeded)?;
                    bytes_delta = bytes_delta.checked_add(bytes).ok_or_else(limit_exceeded)?;
                    new_keys.insert(key.to_vec());
                    ops.push((key.to_vec(), Some(value.to_vec())));
                }
                Mutation::Delete { key } => {
                    self.check_key_value(&key, None)?;
                    count_delta = count_delta.checked_add(1).ok_or_else(limit_exceeded)?;
                    bytes_delta = bytes_delta
                        .checked_add(key.len())
                        .ok_or_else(limit_exceeded)?;
                    new_keys.insert(key.to_vec());
                    ops.push((key.to_vec(), None));
                }
                _ => return Err(Error::new(ErrorKind::Unsupported)),
            }
        }
        self.check_buffer_growth(&new_keys)?;
        self.charge(count_delta, bytes_delta)?;
        for (key, value) in ops {
            self.pending.insert(key, value);
        }
        Ok(())
    }

    async fn clear_range(&mut self, range: &KeyRange) -> Result<()> {
        self.backend.count(|counts| counts.clear_range += 1);
        if !self.backend.config.capabilities.transactional_clear_range {
            return Err(Error::new(ErrorKind::Unsupported));
        }
        if range.start() >= range.end() {
            return Ok(());
        }
        check_range(&self.backend.config, range)?;
        let clear_ranges = self.prepare_clear_ranges(range)?;
        let bytes = range
            .start()
            .len()
            .checked_add(range.end().len())
            .ok_or_else(limit_exceeded)?;
        self.charge(1, bytes)?;
        self.pending.retain(|key, _| !range_contains(range, key));
        self.clear_ranges = clear_ranges;
        Ok(())
    }

    async fn commit_with(self, start: CommitStart) -> Result<()> {
        start.begin()?;
        self.commit_impl()
    }

    async fn rollback(self) {}
}

/// The highest committed version whose conflict metadata has been evicted.
///
/// A transaction whose read version is at or below this value can no longer be
/// validated, because writes in `(read_version, evicted_through]` have been
/// dropped from [`State::last_written`].
fn evicted_through(state: &State) -> u64 {
    state
        .versions
        .front()
        .map(|record| record.version - 1)
        .unwrap_or(0)
}

/// Sums the encoded key-plus-value bytes of a keyspace with checked arithmetic.
fn try_sum_bytes(map: &Keyspace) -> Result<usize> {
    map.iter().try_fold(0usize, |acc, (key, value)| {
        let item = key
            .len()
            .checked_add(value.len())
            .ok_or_else(limit_exceeded)?;
        acc.checked_add(item).ok_or_else(limit_exceeded)
    })
}

/// Materializes staged range clears and point mutations over a snapshot.
fn apply_staged(snapshot: &Keyspace, clear_ranges: &[KeyRange], pending: &Overlay) -> Keyspace {
    let mut map = snapshot.clone();
    // `clear_ranges` is a sorted, disjoint union and `BTreeMap::retain` visits
    // keys in ascending order, so one monotonic range cursor clears the map.
    let mut clear_index = 0usize;
    map.retain(|key, _| {
        while clear_index < clear_ranges.len() && clear_ranges[clear_index].end() <= key.as_slice()
        {
            clear_index += 1;
        }
        clear_index >= clear_ranges.len()
            || !range_contains(&clear_ranges[clear_index], key.as_slice())
    });
    for (key, value) in pending {
        match value {
            Some(value) => {
                map.insert(key.clone(), value.clone());
            }
            None => {
                map.remove(key);
            }
        }
    }
    map
}

/// Inserts one range into a sorted, disjoint range union.
fn merge_clear_range(clear_ranges: &[KeyRange], added: &KeyRange) -> Vec<KeyRange> {
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

fn range_contains(range: &KeyRange, key: &[u8]) -> bool {
    range.start() <= key && key < range.end()
}

/// Evicts retained versions beyond `max_retained_versions`, dropping the
/// conflict metadata of keys whose last write is being evicted.
fn gc_versions(state: &mut State, max_retained_versions: usize) {
    while state.versions.len() > max_retained_versions {
        if let Some(evicted) = state.versions.pop_front() {
            for key in &evicted.written_keys {
                if state.last_written.get(key) == Some(&evicted.version) {
                    state.last_written.remove(key);
                }
            }
        }
    }
}

/// Resolves scan limits, rejecting zero limits and applying backend ceilings.
fn resolve_scan(limits: ScanLimits, config: &DeterministicConfig) -> Result<(usize, usize)> {
    if limits.item_limit == 0 || limits.byte_limit == 0 {
        return Err(Error::new(ErrorKind::InvalidArgument));
    }
    Ok((
        limits.item_limit.min(config.max_scan_page_items),
        limits.byte_limit.min(config.max_scan_page_bytes),
    ))
}

/// Enforces the backend's hard logical-key ceiling.
fn check_key(config: &DeterministicConfig, key: &[u8]) -> Result<()> {
    if key.len() > config.hard_limits.max_key_bytes {
        return Err(limit_exceeded());
    }
    Ok(())
}

/// Validates every key before a batch read performs any transaction work.
fn check_keys(config: &DeterministicConfig, keys: &[Bytes]) -> Result<()> {
    for key in keys {
        check_key(config, key)?;
    }
    Ok(())
}

/// Enforces hard key limits on both bounds of a non-empty logical range.
fn check_range(config: &DeterministicConfig, range: &KeyRange) -> Result<()> {
    check_key(config, range.start())?;
    check_key(config, range.end())
}

/// Scans an ordered map, returning one bounded page whose continuation is the
/// byte-lexicographic successor of its last key.
fn scan_map(
    map: &Keyspace,
    range: &KeyRange,
    item_limit: usize,
    byte_limit: usize,
    max_key_bytes: usize,
) -> Result<ScanPage> {
    let start = range.start().to_vec();
    let end = range.end().to_vec();
    let mut items = Vec::new();
    let mut byte_total = 0usize;
    let mut more = false;
    for (key, value) in map.range(start..end) {
        let item_bytes = key
            .len()
            .checked_add(value.len())
            .ok_or_else(limit_exceeded)?;
        if items.is_empty() {
            // A single oversized first item is returned alone so any legal
            // value stays readable.
            items.push(ScanItem::new(
                Bytes::copy_from_slice(key),
                Bytes::copy_from_slice(value),
            ));
            byte_total = byte_total
                .checked_add(item_bytes)
                .ok_or_else(limit_exceeded)?;
            continue;
        }
        if items.len() >= item_limit
            || byte_total
                .checked_add(item_bytes)
                .ok_or_else(limit_exceeded)?
                > byte_limit
        {
            more = true;
            break;
        }
        items.push(ScanItem::new(
            Bytes::copy_from_slice(key),
            Bytes::copy_from_slice(value),
        ));
        byte_total = byte_total
            .checked_add(item_bytes)
            .ok_or_else(limit_exceeded)?;
    }
    if more {
        ScanPage::continued(items, max_key_bytes)
    } else {
        Ok(ScanPage::terminal(items))
    }
}

/// A one-element key set used to check overlay growth for single-key writes.
fn once_set(key: Vec<u8>) -> BTreeSet<Vec<u8>> {
    let mut set = BTreeSet::new();
    set.insert(key);
    set
}

fn limit_exceeded() -> Error {
    Error::new(ErrorKind::LimitExceeded)
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
fn fingerprint(clear_ranges: &[KeyRange], pending: &Overlay) -> u64 {
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

/// A replayable xorshift64 generator for model and fault histories; the
/// printed seed reproduces a failure.
pub struct Rng(pub u64);

impl Rng {
    /// Returns the next pseudo-random word.
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Returns the next word modulo `bound`.
    pub fn below(&mut self, bound: u64) -> u64 {
        self.next() % bound
    }
}

/// A shareable [`DeterministicBackend`] handle: the Runtime under test and the
/// test's own inspection transactions drive the same backend.
#[derive(Clone)]
pub struct SharedBackend {
    inner: Arc<DeterministicBackend>,
}

impl SharedBackend {
    /// Wraps one deterministic backend.
    pub fn new(inner: DeterministicBackend) -> Self {
        Self {
            inner: Arc::new(inner),
        }
    }

    /// The shared inner backend, for fault injection and inspection.
    pub fn inner(&self) -> &Arc<DeterministicBackend> {
        &self.inner
    }
}

impl Backend for SharedBackend {
    type ReadTxn<'backend> = DeterministicReadTxn<'backend>;
    type WriteTxn<'backend> = DeterministicWriteTxn<'backend>;

    fn hard_limits(&self) -> HardLimits {
        self.inner.hard_limits()
    }

    fn admission_budget(&self) -> AdmissionBudget {
        self.inner.admission_budget()
    }

    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }

    async fn shutdown(&self) {
        self.inner.shutdown().await;
    }

    async fn begin_read(&self) -> Result<Self::ReadTxn<'_>> {
        self.inner.begin_read().await
    }

    async fn begin_write(&self) -> Result<Self::WriteTxn<'_>> {
        self.inner.begin_write().await
    }
}

/// Reads the persisted Index Manifest of one Logical Index.
pub async fn read_manifest(backend: &SharedBackend, index: LogicalIndexId) -> IndexManifest {
    let raw = backend.begin_read().await.expect("begin read");
    let mut txn = ReadLogicalTxn::bootstrap(raw);
    match txn
        .get(LogicalKey::Manifest(index))
        .await
        .expect("read manifest")
    {
        Some(PersistentValue::IndexManifest(manifest)) => manifest,
        _ => panic!("manifest must exist"),
    }
}
