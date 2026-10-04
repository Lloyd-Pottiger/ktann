#![doc = include_str!("../README.md")]
#![forbid(unsafe_code)]

#[cfg(test)]
mod tests;

#[cfg(feature = "test-support")]
pub mod test_support;
#[cfg(feature = "test-support")]
use test_support::{
    CommitFault, CommitOutcome, Control, Diagnostics, HistoryEntry, Lease, range_contains,
};

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::ops::Bound::{Excluded, Included};
use std::sync::{Arc, Mutex, MutexGuard};

use bytes::Bytes;
use imbl::OrdMap;
use ktann::api::{Error, ErrorKind, Result};
use ktann::storage::backend::{
    AdmissionBudget, Backend, Capabilities, CommitStart, HardLimits, InsertOutcome, Mutation,
    ReadOps, ReadTxn, ScanItem, ScanLimits, ScanPage, WriteTxn,
};
use ktann::storage::keys::KeyRange;

// Fixed adapter ceilings keep individual operations and transactions bounded.
const HARD_LIMITS: HardLimits = HardLimits {
    max_key_bytes: 10_000,
    max_value_bytes: 100_000,
};
const BUDGET: AdmissionBudget = AdmissionBudget {
    max_mutations: 10_000,
    max_mutation_bytes: 1 << 20,
    mutation_key_overhead_bytes: 0,
};
const MAX_HISTORY_KEYS: usize = 100_000;
const MAX_HISTORY_BYTES: usize = 8 << 20;

/// An isolated, non-durable transactional keyspace.
///
/// Creation needs no files, namespace prefix, native libraries, or async setup.
/// Readers retain immutable snapshots without holding locks. Writers stage
/// changes privately and serialize only snapshot registration and commit.
/// Conflicting protected reads fail with [`ErrorKind::RetryableAbort`].
/// Cloning shares the keyspace; [`Self::new`] creates an independent keyspace.
#[derive(Clone, Default)]
pub struct MemoryBackend {
    state: Arc<Mutex<State>>,
    #[cfg(feature = "test-support")]
    control: Arc<Control>,
}

impl std::fmt::Debug for MemoryBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MemoryBackend([REDACTED])")
    }
}

/// Committed data and conflict metadata retained by live write snapshots.
#[derive(Default)]
struct State {
    data: OrdMap<Bytes, Bytes>,
    revision: u64,
    writers: BTreeMap<u64, usize>,
    // Deleted keys remain here until no writer can observe their old absence.
    modified: BTreeMap<Bytes, u64>,
    history: VecDeque<Revision>,
    history_keys: usize,
    history_bytes: usize,
    // Snapshots older than this revision lack complete conflict evidence.
    conflict_floor: u64,
    #[cfg(feature = "test-support")]
    max_versions: Option<usize>,
    #[cfg(feature = "test-support")]
    diagnostics: Diagnostics,
}

/// Conflict evidence for one committed revision, retained only while needed.
struct Revision {
    number: u64,
    keys: Vec<Bytes>,
    #[cfg(feature = "test-support")]
    cleared: Vec<KeyRange>,
}

impl State {
    /// Releases conflict evidence older than every live write snapshot.
    fn release_writer(&mut self, revision: u64) {
        let count = self.writers.get_mut(&revision).expect("registered writer");
        *count -= 1;
        if *count == 0 {
            self.writers.remove(&revision);
        }
        self.prune_history();
    }

    fn prune_history(&mut self) {
        let oldest = self
            .writers
            .first_key_value()
            .map_or(self.revision, |(&r, _)| r);
        loop {
            let expired = self.history.front().is_some_and(|r| r.number <= oldest);
            let full = self.history_keys > MAX_HISTORY_KEYS
                || self.history_bytes > MAX_HISTORY_BYTES
                || self.history.len() > MAX_HISTORY_KEYS;
            #[cfg(feature = "test-support")]
            let full = full
                || self
                    .max_versions
                    .is_some_and(|max| self.history.len() > max);
            if !expired && !full {
                break;
            }
            let revision = self.history.pop_front().expect("history front");
            self.conflict_floor = revision.number;
            self.history_keys -= revision.keys.len();
            for key in revision.keys {
                self.history_bytes -= key.len();
                if self.modified.get(&key) == Some(&revision.number) {
                    self.modified.remove(&key);
                }
            }
            #[cfg(feature = "test-support")]
            for range in revision.cleared {
                self.history_keys -= 2;
                self.history_bytes -= range.start().len() + range.end().len();
            }
        }
    }
}

impl MemoryBackend {
    /// Creates a new empty, isolated in-memory backend.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn lock(&self) -> Result<MutexGuard<'_, State>> {
        self.state
            .lock()
            .map_err(|_| Error::new(ErrorKind::Backend))
    }
}

impl Backend for MemoryBackend {
    type ReadTxn<'a> = MemoryReadTxn;
    type WriteTxn<'a> = MemoryWriteTxn<'a>;

    fn hard_limits(&self) -> HardLimits {
        #[cfg(feature = "test-support")]
        {
            self.control.config.hard_limits
        }
        #[cfg(not(feature = "test-support"))]
        {
            HARD_LIMITS
        }
    }

    fn admission_budget(&self) -> AdmissionBudget {
        #[cfg(feature = "test-support")]
        {
            self.control.config.admission_budget
        }
        #[cfg(not(feature = "test-support"))]
        {
            BUDGET
        }
    }

    fn capabilities(&self) -> Capabilities {
        #[cfg(feature = "test-support")]
        {
            self.control.config.capabilities
        }
        #[cfg(not(feature = "test-support"))]
        {
            Capabilities {
                transactional_clear_range: false,
            }
        }
    }

    async fn begin_read(&self) -> Result<MemoryReadTxn> {
        #[cfg(feature = "test-support")]
        let lease = self.control.admit()?;
        Ok(MemoryReadTxn {
            data: self.lock()?.data.clone(),
            #[cfg(feature = "test-support")]
            lease,
        })
    }

    async fn begin_write(&self) -> Result<MemoryWriteTxn<'_>> {
        #[cfg(feature = "test-support")]
        let lease = self.control.admit()?;
        let mut state = self.lock()?;
        let revision = state.revision;
        *state.writers.entry(revision).or_default() += 1;
        Ok(MemoryWriteTxn {
            backend: self,
            revision,
            view: MemoryReadTxn {
                data: state.data.clone(),
                #[cfg(feature = "test-support")]
                lease,
            },
            protected: BTreeSet::new(),
            writes: BTreeMap::new(),
            mutation_count: 0,
            mutation_bytes: 0,
            #[cfg(feature = "test-support")]
            clears: Vec::new(),
        })
    }
}

/// A consistent immutable read snapshot, released when dropped.
pub struct MemoryReadTxn {
    data: OrdMap<Bytes, Bytes>,
    #[cfg(feature = "test-support")]
    lease: Lease,
}

impl std::fmt::Debug for MemoryReadTxn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MemoryReadTxn([REDACTED])")
    }
}

impl MemoryReadTxn {
    fn hard_limits(&self) -> HardLimits {
        #[cfg(feature = "test-support")]
        {
            self.lease.0.config.hard_limits
        }
        #[cfg(not(feature = "test-support"))]
        {
            HARD_LIMITS
        }
    }

    fn validate_key(&self, key: &[u8]) -> Result<()> {
        if key.len() > self.hard_limits().max_key_bytes {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
        Ok(())
    }

    fn lookup(&self, key: &Bytes) -> Result<Option<Bytes>> {
        self.validate_key(key)?;
        Ok(self.data.get(key).cloned())
    }

    fn lookup_batch(&self, keys: &[Bytes]) -> Result<Vec<Option<Bytes>>> {
        #[cfg(feature = "test-support")]
        self.lease.0.check_batch(keys.len())?;
        keys.iter().map(|key| self.lookup(key)).collect()
    }

    fn scan_page(&self, range: &KeyRange, limits: ScanLimits) -> Result<ScanPage> {
        validate_scan(limits)?;
        if range.start() >= range.end() {
            return Ok(ScanPage::terminal(Vec::new()));
        }
        self.validate_key(range.start())?;
        self.validate_key(range.end())?;
        #[cfg(feature = "test-support")]
        let limits = self.lease.0.scan_limits(limits);
        let mut items = Vec::new();
        let mut bytes = 0_usize;
        for (key, value) in self
            .data
            .range::<_, [u8]>((Included(range.start()), Excluded(range.end())))
        {
            let item_bytes = key.len() + value.len();
            if !items.is_empty()
                && (items.len() == limits.item_limit
                    || item_bytes > limits.byte_limit.saturating_sub(bytes))
            {
                return ScanPage::continued(items, self.hard_limits().max_key_bytes);
            }
            bytes += item_bytes;
            items.push(ScanItem::new(key.clone(), value.clone()));
        }
        Ok(ScanPage::terminal(items))
    }
}

fn validate_scan(limits: ScanLimits) -> Result<()> {
    if limits.item_limit == 0 || limits.byte_limit == 0 {
        return Err(Error::new(ErrorKind::InvalidArgument));
    }
    Ok(())
}

impl ReadOps for MemoryReadTxn {
    async fn get(&mut self, key: Bytes) -> Result<Option<Bytes>> {
        #[cfg(feature = "test-support")]
        self.lease.0.count(|c| c.get += 1);
        self.lookup(&key)
    }
    async fn batch_get(&mut self, keys: Vec<Bytes>) -> Result<Vec<Option<Bytes>>> {
        #[cfg(feature = "test-support")]
        self.lease.0.count(|c| c.batch_get += 1);
        self.lookup_batch(&keys)
    }
    async fn scan(&mut self, range: &KeyRange, limits: ScanLimits) -> Result<ScanPage> {
        #[cfg(feature = "test-support")]
        self.lease.0.count(|c| c.scan += 1);
        self.scan_page(range, limits)
    }
    async fn batch_scan(
        &mut self,
        ranges: &[KeyRange],
        limits: ScanLimits,
    ) -> Result<Vec<ScanPage>> {
        #[cfg(feature = "test-support")]
        {
            self.lease.0.count(|c| {
                c.batch_scan += 1;
                c.batch_scan_ranges += ranges.len();
            });
            self.lease.0.check_batch(ranges.len())?;
        }
        validate_scan(limits)?;
        ranges
            .iter()
            .map(|range| self.scan_page(range, limits))
            .collect()
    }
}
impl ReadTxn for MemoryReadTxn {}

/// An optimistic write transaction with a private read-your-writes snapshot.
///
/// Unprotected reads and range scans do not establish conflicts. Dropping or
/// rolling back discards staged writes and releases retained conflict history.
pub struct MemoryWriteTxn<'a> {
    backend: &'a MemoryBackend,
    revision: u64,
    view: MemoryReadTxn,
    protected: BTreeSet<Bytes>,
    writes: BTreeMap<Bytes, Option<Bytes>>,
    mutation_count: usize,
    mutation_bytes: usize,
    #[cfg(feature = "test-support")]
    clears: Vec<KeyRange>,
}

impl std::fmt::Debug for MemoryWriteTxn<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("MemoryWriteTxn([REDACTED])")
    }
}

impl MemoryWriteTxn<'_> {
    /// Validates admission before staging, so a rejected mutation changes nothing.
    fn stage(&mut self, key: Bytes, value: Option<Bytes>) -> Result<()> {
        self.view.validate_key(&key)?;
        #[cfg(feature = "test-support")]
        self.check_buffer(usize::from(!self.writes.contains_key(&key)))?;
        let budget = self.backend.admission_budget();
        let value_bytes = value.as_ref().map_or(0, Bytes::len);
        if value_bytes > self.backend.hard_limits().max_value_bytes
            || self.mutation_count == budget.max_mutations
            || key.len() + value_bytes > budget.max_mutation_bytes - self.mutation_bytes
        {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
        self.mutation_count += 1;
        self.mutation_bytes += key.len() + value_bytes;
        self.apply_point(key, value);
        Ok(())
    }

    /// Keeps the private read view and final commit delta in step after admission.
    fn apply_point(&mut self, key: Bytes, value: Option<Bytes>) {
        if let Some(value) = &value {
            self.view.data.insert(key.clone(), value.clone());
        } else {
            self.view.data.remove(&key);
        }
        self.writes.insert(key, value);
    }
}

impl ReadOps for MemoryWriteTxn<'_> {
    async fn get(&mut self, key: Bytes) -> Result<Option<Bytes>> {
        self.view.get(key).await
    }

    async fn batch_get(&mut self, keys: Vec<Bytes>) -> Result<Vec<Option<Bytes>>> {
        self.view.batch_get(keys).await
    }

    async fn scan(&mut self, range: &KeyRange, limits: ScanLimits) -> Result<ScanPage> {
        self.view.scan(range, limits).await
    }

    async fn batch_scan(
        &mut self,
        ranges: &[KeyRange],
        limits: ScanLimits,
    ) -> Result<Vec<ScanPage>> {
        self.view.batch_scan(ranges, limits).await
    }
}

impl MemoryWriteTxn<'_> {
    #[cfg(feature = "test-support")]
    fn check_buffer(&self, added: usize) -> Result<()> {
        if self.writes.len() + self.clears.len() + added
            > self.backend.control.config.max_mutation_buffer
        {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
        Ok(())
    }

    fn protect(&mut self, keys: &[Bytes]) -> Result<()> {
        #[cfg(feature = "test-support")]
        {
            let remaining = self.backend.control.config.max_read_set - self.protected.len();
            if keys.len() > remaining
                && keys
                    .iter()
                    .filter(|key| !self.protected.contains(*key))
                    .collect::<BTreeSet<_>>()
                    .len()
                    > remaining
            {
                return Err(Error::new(ErrorKind::LimitExceeded));
            }
        }
        self.protected.extend(keys.iter().cloned());
        Ok(())
    }

    #[cfg(feature = "test-support")]
    fn record(&self, state: &mut State, outcome: CommitOutcome) {
        state.diagnostics.record(
            HistoryEntry {
                version: state.revision,
                outcome,
                mutations: self.mutation_count,
                mutation_bytes: self.mutation_bytes,
                distinct_keys: self.writes.len(),
                fingerprint: test_support::fingerprint(&self.clears, &self.writes),
            },
            self.backend.control.config.max_history_entries,
        );
    }
}

impl WriteTxn for MemoryWriteTxn<'_> {
    async fn get_for_update(&mut self, key: Bytes) -> Result<Option<Bytes>> {
        #[cfg(feature = "test-support")]
        self.backend.control.count(|c| c.get_for_update += 1);
        let value = self.view.lookup(&key)?;
        self.protect(std::slice::from_ref(&key))?;
        Ok(value)
    }

    async fn batch_get_for_update(&mut self, keys: Vec<Bytes>) -> Result<Vec<Option<Bytes>>> {
        #[cfg(feature = "test-support")]
        self.backend.control.count(|c| c.batch_get_for_update += 1);
        let values = self.view.lookup_batch(&keys)?;
        self.protect(&keys)?;
        Ok(values)
    }

    async fn put(&mut self, key: Bytes, value: Bytes) -> Result<()> {
        #[cfg(feature = "test-support")]
        self.backend.control.count(|c| c.put += 1);
        self.stage(key, Some(value))
    }

    async fn insert(&mut self, key: Bytes, value: Bytes) -> Result<InsertOutcome> {
        #[cfg(feature = "test-support")]
        self.backend.control.count(|c| c.insert += 1);
        if value.len() > self.backend.hard_limits().max_value_bytes {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
        let previous = self.view.lookup(&key)?;
        self.protect(std::slice::from_ref(&key))?;
        if previous.is_some() {
            return Ok(InsertOutcome::AlreadyExists);
        }
        self.stage(key, Some(value))?;
        Ok(InsertOutcome::Inserted)
    }

    async fn delete(&mut self, key: Bytes) -> Result<()> {
        #[cfg(feature = "test-support")]
        self.backend.control.count(|c| c.delete += 1);
        self.stage(key, None)
    }

    async fn batch_mutate(&mut self, mutations: Vec<Mutation>) -> Result<()> {
        #[cfg(feature = "test-support")]
        {
            self.backend.control.count(|c| c.batch_mutate += 1);
            self.backend.control.check_batch(mutations.len())?;
        }
        let budget = self.backend.admission_budget();
        let mut bytes = self.mutation_bytes;
        if mutations.len() > budget.max_mutations - self.mutation_count {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
        // Validate and count the entire batch before changing the private view.
        #[cfg(feature = "test-support")]
        let mut added = BTreeSet::new();
        for mutation in &mutations {
            let (key, value_bytes) = match mutation {
                Mutation::Put { key, value } => (key, value.len()),
                Mutation::Delete { key } => (key, 0),
                _ => return Err(Error::new(ErrorKind::Unsupported)),
            };
            self.view.validate_key(key)?;
            if value_bytes > self.backend.hard_limits().max_value_bytes
                || key.len() + value_bytes > budget.max_mutation_bytes - bytes
            {
                return Err(Error::new(ErrorKind::LimitExceeded));
            }
            bytes += key.len() + value_bytes;
            #[cfg(feature = "test-support")]
            if !self.writes.contains_key(key) {
                added.insert(key);
            }
        }
        #[cfg(feature = "test-support")]
        self.check_buffer(added.len())?;
        self.mutation_count += mutations.len();
        self.mutation_bytes = bytes;
        for mutation in mutations {
            match mutation {
                Mutation::Put { key, value } => self.apply_point(key, Some(value)),
                Mutation::Delete { key } => self.apply_point(key, None),
                _ => unreachable!("prevalidated mutation"),
            }
        }
        Ok(())
    }

    async fn clear_range(&mut self, _range: &KeyRange) -> Result<()> {
        #[cfg(feature = "test-support")]
        {
            self.backend.control.count(|c| c.clear_range += 1);
            if self.backend.capabilities().transactional_clear_range {
                return self.stage_clear(_range);
            }
        }
        Err(Error::new(ErrorKind::Unsupported))
    }

    async fn commit_with(self, start: CommitStart) -> Result<()> {
        let mut state = self.backend.lock()?;
        // No await follows this claim: fault resolution and publication are one
        // serialized native commit operation.
        start.begin()?;
        #[cfg(feature = "test-support")]
        let fault = state
            .diagnostics
            .faults
            .pop_front()
            .unwrap_or(CommitFault::Normal);
        #[cfg(feature = "test-support")]
        match fault {
            CommitFault::Abort => {
                self.record(&mut state, CommitOutcome::Aborted);
                return Err(Error::new(ErrorKind::RetryableAbort));
            }
            CommitFault::UnknownNotApplied => {
                self.record(&mut state, CommitOutcome::UnknownNotApplied);
                return Err(Error::new(ErrorKind::CommitOutcomeUnknown));
            }
            _ => {}
        }
        let conflict = !self.protected.is_empty()
            && (self.revision < state.conflict_floor
                || self
                    .protected
                    .iter()
                    .any(|key| state.modified.get(key).is_some_and(|r| *r > self.revision)));
        #[cfg(feature = "test-support")]
        let conflict = conflict
            || (!self.protected.is_empty()
                && state.history.iter().any(|r| {
                    r.number > self.revision
                        && r.cleared.iter().any(|range| {
                            self.protected.iter().any(|key| range_contains(range, key))
                        })
                }));
        if conflict {
            #[cfg(feature = "test-support")]
            self.record(&mut state, CommitOutcome::Aborted);
            return Err(Error::new(ErrorKind::RetryableAbort));
        }
        let revision = state
            .revision
            .checked_add(1)
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        // Apply to the latest root, preserving unrelated concurrent commits.
        let mut data = state.data.clone();
        #[cfg(feature = "test-support")]
        let mut db_bytes = state.diagnostics.db_bytes;
        #[cfg(feature = "test-support")]
        for range in &self.clears {
            let keys: Vec<_> = data
                .range::<_, [u8]>((Included(range.start()), Excluded(range.end())))
                .map(|(key, _)| key.clone())
                .collect();
            for key in keys {
                let value = data.remove(&key).expect("range member");
                db_bytes -= key.len() + value.len();
            }
        }
        for (key, value) in &self.writes {
            #[cfg(feature = "test-support")]
            if let Some(previous) = data.get(key) {
                db_bytes -= key.len() + previous.len();
            }
            if let Some(value) = value {
                data.insert(key.clone(), value.clone());
                #[cfg(feature = "test-support")]
                {
                    db_bytes += key.len() + value.len();
                }
            } else {
                data.remove(key);
            }
        }
        #[cfg(feature = "test-support")]
        {
            let config = &self.backend.control.config;
            if data.len() > config.max_db_keys || db_bytes > config.max_db_bytes {
                self.record(&mut state, CommitOutcome::LimitExceeded);
                return Err(Error::new(ErrorKind::LimitExceeded));
            }
            state.diagnostics.db_bytes = db_bytes;
        }
        for key in self.writes.keys() {
            state.modified.insert(key.clone(), revision);
        }
        state.history_keys += self.writes.len();
        state.history_bytes += self.writes.keys().map(Bytes::len).sum::<usize>();
        #[cfg(feature = "test-support")]
        for range in &self.clears {
            state.history_keys += 2;
            state.history_bytes += range.start().len() + range.end().len();
        }
        state.history.push_back(Revision {
            number: revision,
            keys: self.writes.keys().cloned().collect(),
            #[cfg(feature = "test-support")]
            cleared: self.clears.clone(),
        });
        state.data = data;
        state.revision = revision;
        state.prune_history();
        #[cfg(feature = "test-support")]
        {
            if fault == CommitFault::UnknownApplied {
                self.record(&mut state, CommitOutcome::UnknownApplied);
                return Err(Error::new(ErrorKind::CommitOutcomeUnknown));
            }
            self.record(&mut state, CommitOutcome::Committed);
        }
        Ok(())
    }

    async fn rollback(self) {}
}

#[cfg(feature = "test-support")]
impl MemoryWriteTxn<'_> {
    /// Simulates an adapter's native range-clear capability for core protocols.
    fn stage_clear(&mut self, range: &KeyRange) -> Result<()> {
        if range.start() >= range.end() {
            return Ok(());
        }
        self.view.validate_key(range.start())?;
        self.view.validate_key(range.end())?;
        let clears = test_support::merge_clear_range(&self.clears, range);
        let retained = self
            .writes
            .keys()
            .filter(|key| !range_contains(range, key))
            .count();
        let bytes = range.start().len() + range.end().len();
        let budget = self.backend.admission_budget();
        if retained + clears.len() > self.backend.control.config.max_mutation_buffer
            || self.mutation_count == budget.max_mutations
            || bytes > budget.max_mutation_bytes - self.mutation_bytes
        {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
        self.mutation_count += 1;
        self.mutation_bytes += bytes;
        self.writes.retain(|key, _| !range_contains(range, key));
        let keys: Vec<_> = self
            .view
            .data
            .range::<_, [u8]>((Included(range.start()), Excluded(range.end())))
            .map(|(key, _)| key.clone())
            .collect();
        for key in keys {
            self.view.data.remove(&key);
        }
        self.clears = clears;
        Ok(())
    }
}

impl Drop for MemoryWriteTxn<'_> {
    fn drop(&mut self) {
        if let Ok(mut state) = self.backend.state.lock() {
            state.release_writer(self.revision);
        }
    }
}
