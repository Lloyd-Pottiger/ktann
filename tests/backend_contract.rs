//! Backend-neutral transaction contract tests.
//!
//! These tests drive a deterministic in-memory mock backend exclusively through
//! the public [`ktann::storage::backend`] seam. The mock models the contract
//! (snapshot isolation, read-your-writes, update-protected conflicts, ordered
//! scans and batches, unique insertion, rollback, commit outcomes, hard limits
//! and admission budgets, and the range-clear capability) without depending on
//! any production adapter.

use std::cell::Cell;
use std::collections::{BTreeMap, BTreeSet};
use std::future::Future;
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use ktann::api::{Error, ErrorKind, Result};
use ktann::storage::backend::{
    AdmissionBudget, Backend, Capabilities, HardLimits, InsertOutcome, Mutation, ReadOps, ReadTxn,
    ScanItem, ScanLimits, ScanPage, WriteTxn,
};
use ktann::storage::keys::KeyRange;

/// Drives an already-borrowing future to completion on a current-thread
/// runtime. The mock's futures never park, so this never blocks.
fn block_on<F: Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("current-thread runtime builds")
        .block_on(future)
}

/// A deterministic in-memory KV backend implementing [`Backend`].
///
/// Committed data is an immutable `Arc<BTreeMap>` snapshot; every read and
/// write transaction captures its own clone at begin time. Writes apply to a
/// per-transaction overlay and install a fresh snapshot on commit, giving
/// snapshot isolation and read-your-writes without a versioned log.
struct MockBackend {
    hard_limits: HardLimits,
    admission_budget: AdmissionBudget,
    capabilities: Capabilities,
    state: Mutex<MockState>,
    fault: Mutex<CommitFault>,
}

#[derive(Default)]
struct MockState {
    committed: Arc<BTreeMap<Vec<u8>, Vec<u8>>>,
}

/// An injected commit outcome for fault-injection tests.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum CommitFault {
    /// Commit proceeds normally.
    None,
    /// The mutation is applied but the result is reported unknown.
    Unknown,
    /// The mutation is discarded and commit reports a retryable abort.
    Fail,
}

impl MockBackend {
    fn new(
        hard_limits: HardLimits,
        admission_budget: AdmissionBudget,
        capabilities: Capabilities,
    ) -> Self {
        Self {
            hard_limits,
            admission_budget,
            capabilities,
            state: Mutex::new(MockState {
                committed: Arc::new(BTreeMap::new()),
            }),
            fault: Mutex::new(CommitFault::None),
        }
    }

    fn set_fault(&self, fault: CommitFault) {
        *self.fault.lock().expect("fault lock poisoned") = fault;
    }
}

/// A read transaction: one immutable snapshot.
///
/// The `PhantomData` reference ties the transaction's lifetime and auto-traits
/// to its backend; the `Cell` marker makes the type `Send` but not `Sync`,
/// proving the interface does not require `Sync`.
struct MockReadTxn<'backend> {
    _backend: PhantomData<&'backend MockBackend>,
    snapshot: Arc<BTreeMap<Vec<u8>, Vec<u8>>>,
    _not_sync: PhantomData<Cell<()>>,
}

/// A write transaction: a snapshot plus an uncommitted overlay.
struct MockWriteTxn<'backend> {
    backend: &'backend MockBackend,
    snapshot: Arc<BTreeMap<Vec<u8>, Vec<u8>>>,
    /// Overlay of pending mutations; `None` marks a delete.
    pending: BTreeMap<Vec<u8>, Option<Vec<u8>>>,
    /// Keys read via `get_for_update`, establishing point conflicts.
    read_set: BTreeSet<Vec<u8>>,
    mutation_count: usize,
    mutation_bytes: usize,
    /// Makes the type `Send` but not `Sync` (same purpose as in the read txn).
    _not_sync: PhantomData<Cell<()>>,
}

impl MockReadTxn<'_> {
    fn lookup(&self, key: &[u8]) -> Option<Bytes> {
        self.snapshot
            .get(key)
            .map(|value| Bytes::from(value.clone()))
    }
}

impl MockWriteTxn<'_> {
    /// Reads through the pending overlay for read-your-writes.
    fn lookup(&self, key: &[u8]) -> Option<Bytes> {
        if let Some(entry) = self.pending.get(key) {
            return entry.as_ref().map(|value| Bytes::from(value.clone()));
        }
        self.snapshot
            .get(key)
            .map(|value| Bytes::from(value.clone()))
    }

    /// Charges one mutation against hard limits and the admission budget.
    fn account(&mut self, key: &[u8], value_bytes: usize) -> Result<()> {
        let hard = self.backend.hard_limits;
        if key.len() > hard.max_key_bytes || value_bytes > hard.max_value_bytes {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
        let budget = self.backend.admission_budget;
        if self.mutation_count >= budget.max_mutations {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
        let total = key
            .len()
            .checked_add(value_bytes)
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        let next_bytes = self
            .mutation_bytes
            .checked_add(total)
            .ok_or_else(|| Error::new(ErrorKind::LimitExceeded))?;
        if next_bytes > budget.max_mutation_bytes {
            return Err(Error::new(ErrorKind::LimitExceeded));
        }
        self.mutation_count += 1;
        self.mutation_bytes = next_bytes;
        Ok(())
    }

    /// Installs the pending overlay into the committed snapshot.
    fn apply(&self, state: &mut MockState) {
        let mut committed = (*state.committed).clone();
        for (key, value) in &self.pending {
            match value {
                Some(value) => {
                    committed.insert(key.clone(), value.clone());
                }
                None => {
                    committed.remove(key);
                }
            }
        }
        state.committed = Arc::new(committed);
    }
}

/// Scans the merged view of a snapshot and an optional pending overlay.
fn scan_merged(
    snapshot: &BTreeMap<Vec<u8>, Vec<u8>>,
    pending: Option<&BTreeMap<Vec<u8>, Option<Vec<u8>>>>,
    range: &KeyRange,
    limits: ScanLimits,
) -> Result<ScanPage> {
    if limits.item_limit == 0 || limits.byte_limit == 0 {
        return Err(Error::new(ErrorKind::InvalidArgument));
    }

    let merged: BTreeMap<Vec<u8>, Vec<u8>> = match pending {
        None => snapshot.clone(),
        Some(pending) => {
            let mut merged = snapshot.clone();
            for (key, value) in pending {
                match value {
                    Some(value) => {
                        merged.insert(key.clone(), value.clone());
                    }
                    None => {
                        merged.remove(key);
                    }
                }
            }
            merged
        }
    };

    let start = range.start();
    let end = range.end();
    let mut items = Vec::new();
    let mut byte_total = 0_usize;
    let mut next = None;
    for (key, value) in merged.range(start.to_vec()..end.to_vec()) {
        let item_bytes = key.len() + value.len();
        if items.is_empty() {
            // A single oversized first item is returned alone so a legal value
            // stays readable; every later item must fit within the byte limit.
            items.push(ScanItem::new(
                Bytes::copy_from_slice(key),
                Bytes::copy_from_slice(value),
            ));
            byte_total += item_bytes;
            continue;
        }
        if items.len() >= limits.item_limit || byte_total + item_bytes > limits.byte_limit {
            next = Some(Bytes::copy_from_slice(key));
            break;
        }
        items.push(ScanItem::new(
            Bytes::copy_from_slice(key),
            Bytes::copy_from_slice(value),
        ));
        byte_total += item_bytes;
    }
    Ok(ScanPage::new(items, next))
}

impl Backend for MockBackend {
    type ReadTxn<'backend> = MockReadTxn<'backend>;
    type WriteTxn<'backend> = MockWriteTxn<'backend>;

    fn hard_limits(&self) -> HardLimits {
        self.hard_limits
    }

    fn admission_budget(&self) -> AdmissionBudget {
        self.admission_budget
    }

    fn capabilities(&self) -> Capabilities {
        self.capabilities
    }

    async fn begin_read(&self) -> Result<MockReadTxn<'_>> {
        let committed = self
            .state
            .lock()
            .expect("state lock poisoned")
            .committed
            .clone();
        Ok(MockReadTxn {
            _backend: PhantomData,
            snapshot: committed,
            _not_sync: PhantomData,
        })
    }

    async fn begin_write(&self) -> Result<MockWriteTxn<'_>> {
        let committed = self
            .state
            .lock()
            .expect("state lock poisoned")
            .committed
            .clone();
        Ok(MockWriteTxn {
            backend: self,
            snapshot: committed,
            pending: BTreeMap::new(),
            read_set: BTreeSet::new(),
            mutation_count: 0,
            mutation_bytes: 0,
            _not_sync: PhantomData,
        })
    }
}

impl ReadOps for MockReadTxn<'_> {
    async fn get(&mut self, key: Bytes) -> Result<Option<Bytes>> {
        Ok(self.lookup(&key))
    }

    async fn batch_get(&mut self, keys: Vec<Bytes>) -> Result<Vec<Option<Bytes>>> {
        Ok(keys.into_iter().map(|key| self.lookup(&key)).collect())
    }

    async fn scan(&mut self, range: &KeyRange, limits: ScanLimits) -> Result<ScanPage> {
        scan_merged(&self.snapshot, None, range, limits)
    }
}

impl ReadOps for MockWriteTxn<'_> {
    async fn get(&mut self, key: Bytes) -> Result<Option<Bytes>> {
        Ok(self.lookup(&key))
    }

    async fn batch_get(&mut self, keys: Vec<Bytes>) -> Result<Vec<Option<Bytes>>> {
        Ok(keys.into_iter().map(|key| self.lookup(&key)).collect())
    }

    async fn scan(&mut self, range: &KeyRange, limits: ScanLimits) -> Result<ScanPage> {
        scan_merged(&self.snapshot, Some(&self.pending), range, limits)
    }
}

impl ReadTxn for MockReadTxn<'_> {}

impl WriteTxn for MockWriteTxn<'_> {
    async fn get_for_update(&mut self, key: Bytes) -> Result<Option<Bytes>> {
        self.read_set.insert(key.to_vec());
        Ok(self.lookup(&key))
    }

    async fn batch_get_for_update(&mut self, keys: Vec<Bytes>) -> Result<Vec<Option<Bytes>>> {
        for key in &keys {
            self.read_set.insert(key.to_vec());
        }
        Ok(keys.into_iter().map(|key| self.lookup(&key)).collect())
    }

    async fn put(&mut self, key: Bytes, value: Bytes) -> Result<()> {
        self.account(&key, value.len())?;
        self.pending.insert(key.to_vec(), Some(value.to_vec()));
        Ok(())
    }

    async fn insert(&mut self, key: Bytes, value: Bytes) -> Result<InsertOutcome> {
        self.account(&key, value.len())?;
        if self.lookup(&key).is_some() {
            return Ok(InsertOutcome::AlreadyExists);
        }
        self.pending.insert(key.to_vec(), Some(value.to_vec()));
        Ok(InsertOutcome::Inserted)
    }

    async fn delete(&mut self, key: Bytes) -> Result<()> {
        self.account(&key, 0)?;
        self.pending.insert(key.to_vec(), None);
        Ok(())
    }

    async fn batch_mutate(&mut self, mutations: Vec<Mutation>) -> Result<()> {
        for mutation in mutations {
            match mutation {
                Mutation::Put { key, value } => {
                    self.account(&key, value.len())?;
                    self.pending.insert(key.to_vec(), Some(value.to_vec()));
                }
                Mutation::Delete { key } => {
                    self.account(&key, 0)?;
                    self.pending.insert(key.to_vec(), None);
                }
                _ => return Err(Error::new(ErrorKind::Unsupported)),
            }
        }
        Ok(())
    }

    async fn clear_range(&mut self, range: &KeyRange) -> Result<()> {
        if !self.backend.capabilities.transactional_clear_range {
            return Err(Error::new(ErrorKind::Unsupported));
        }
        let start = range.start();
        let end = range.end();
        let mut keys = BTreeSet::new();
        for key in self.snapshot.keys() {
            if key.as_slice() >= start && key.as_slice() < end {
                keys.insert(key.clone());
            }
        }
        for key in self.pending.keys() {
            if key.as_slice() >= start && key.as_slice() < end {
                keys.insert(key.clone());
            }
        }
        for key in keys {
            self.pending.insert(key, None);
        }
        Ok(())
    }

    async fn commit(self) -> Result<()> {
        let fault = *self.backend.fault.lock().expect("fault lock poisoned");
        let mut state = self.backend.state.lock().expect("state lock poisoned");

        match fault {
            CommitFault::Fail => Err(Error::new(ErrorKind::RetryableAbort)),
            CommitFault::Unknown => {
                self.apply(&mut state);
                Err(Error::new(ErrorKind::CommitOutcomeUnknown))
            }
            CommitFault::None => {
                for key in &self.read_set {
                    if state.committed.get(key) != self.snapshot.get(key) {
                        return Err(Error::new(ErrorKind::RetryableAbort));
                    }
                }
                self.apply(&mut state);
                Ok(())
            }
        }
    }

    async fn rollback(self) {}
}

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

fn key(bytes: &'static [u8]) -> Bytes {
    Bytes::from_static(bytes)
}

fn range(start: &[u8], end: &[u8]) -> KeyRange {
    KeyRange::new(start.to_vec(), end.to_vec())
}

fn mock() -> MockBackend {
    MockBackend::new(
        HardLimits {
            max_key_bytes: 1_024,
            max_value_bytes: 4_096,
        },
        AdmissionBudget {
            max_mutations: 1_000,
            max_mutation_bytes: 1 << 20,
        },
        Capabilities {
            transactional_clear_range: false,
        },
    )
}

fn mock_with_clear() -> MockBackend {
    MockBackend::new(
        HardLimits {
            max_key_bytes: 1_024,
            max_value_bytes: 4_096,
        },
        AdmissionBudget {
            max_mutations: 1_000,
            max_mutation_bytes: 1 << 20,
        },
        Capabilities {
            transactional_clear_range: true,
        },
    )
}

fn seed(backend: &MockBackend, entries: &[(&'static [u8], &'static [u8])]) {
    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        for (entry_key, entry_value) in entries {
            txn.put(key(entry_key), key(entry_value))
                .await
                .expect("seed put succeeds");
        }
        txn.commit().await.expect("seed commit succeeds");
    });
}

fn scanned_keys(page: &ScanPage) -> Vec<&[u8]> {
    page.items()
        .iter()
        .map(|item| item.key().as_ref())
        .collect()
}

// ---------------------------------------------------------------------------
// Compile-time interface shape.
// ---------------------------------------------------------------------------

#[test]
fn transaction_types_are_send() {
    fn assert_send<T: Send>() {}
    // `MockReadTxn` and `MockWriteTxn` are `Send` for any backend lifetime.
    // Their `PhantomData<Cell<()>>` field makes them `!Sync`, and yet they still
    // satisfy `ReadTxn`/`WriteTxn`, proving the interface requires only `Send`.
    assert_send::<MockReadTxn<'static>>();
    assert_send::<MockWriteTxn<'static>>();
}

#[test]
fn backend_declares_limits_budget_and_capabilities() {
    let backend = mock();
    assert_eq!(backend.hard_limits().max_key_bytes, 1_024);
    assert_eq!(backend.hard_limits().max_value_bytes, 4_096);
    assert_eq!(backend.admission_budget().max_mutations, 1_000);
    assert_eq!(backend.admission_budget().max_mutation_bytes, 1 << 20);
    assert!(!backend.capabilities().transactional_clear_range);
    assert!(mock_with_clear().capabilities().transactional_clear_range);
}

// ---------------------------------------------------------------------------
// Snapshot and point/batch reads.
// ---------------------------------------------------------------------------

#[test]
fn snapshot_consistency_isolates_later_commits() {
    let backend = mock();
    seed(&backend, &[(b"a", b"1")]);

    block_on(async {
        let mut reader = backend.begin_read().await.expect("begin read");
        // A later commit must not become visible to this open snapshot.
        let mut writer = backend.begin_write().await.expect("begin write");
        writer.put(key(b"a"), key(b"2")).await.expect("put");
        writer.put(key(b"b"), key(b"3")).await.expect("put");
        writer.commit().await.expect("commit");

        assert_eq!(reader.get(key(b"a")).await.expect("get"), Some(key(b"1")));
        assert_eq!(reader.get(key(b"b")).await.expect("get"), None);
    });
}

#[test]
fn write_transaction_reads_its_own_writes() {
    let backend = mock();
    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        assert_eq!(txn.get(key(b"a")).await.expect("get"), None);
        txn.put(key(b"a"), key(b"v")).await.expect("put");
        assert_eq!(txn.get(key(b"a")).await.expect("get"), Some(key(b"v")));
        txn.delete(key(b"a")).await.expect("delete");
        assert_eq!(txn.get(key(b"a")).await.expect("get"), None);
        txn.rollback().await;
    });
}

#[test]
fn batch_get_preserves_order_duplicates_and_empty_input() {
    let backend = mock();
    seed(&backend, &[(b"a", b"1"), (b"b", b"2")]);

    block_on(async {
        let mut txn = backend.begin_read().await.expect("begin read");

        let empty = txn.batch_get(Vec::new()).await.expect("empty batch");
        assert!(empty.is_empty());

        let result = txn
            .batch_get(vec![key(b"b"), key(b"a"), key(b"b"), key(b"missing")])
            .await
            .expect("batch get");
        assert_eq!(
            result,
            vec![Some(key(b"2")), Some(key(b"1")), Some(key(b"2")), None]
        );
    });
}

// ---------------------------------------------------------------------------
// Scans.
// ---------------------------------------------------------------------------

#[test]
fn scan_returns_sorted_items_in_half_open_range() {
    let backend = mock();
    seed(
        &backend,
        &[(b"c", b"3"), (b"a", b"1"), (b"b", b"2"), (b"d", b"4")],
    );

    block_on(async {
        let mut txn = backend.begin_read().await.expect("begin read");
        let limits = ScanLimits {
            item_limit: 100,
            byte_limit: 1_024,
        };
        let page = txn.scan(&range(b"a", b"d"), limits).await.expect("scan");
        assert_eq!(scanned_keys(&page), vec![&b"a"[..], &b"b"[..], &b"c"[..]]);
        assert!(page.next_start().is_none());
    });
}

#[test]
fn scan_paginates_with_strictly_advancing_next_start() {
    let backend = mock();
    seed(&backend, &[(b"a", b"1"), (b"b", b"2"), (b"c", b"3")]);

    block_on(async {
        let mut txn = backend.begin_read().await.expect("begin read");
        let limits = ScanLimits {
            item_limit: 2,
            byte_limit: 1_024,
        };

        let first = txn
            .scan(&range(b"a", b"\xff"), limits)
            .await
            .expect("first");
        assert_eq!(scanned_keys(&first), vec![&b"a"[..], &b"b"[..]]);
        let next = first.next_start().expect("non-terminal has cursor").clone();
        assert_eq!(next.as_ref(), b"c");

        let second = txn
            .scan(&KeyRange::new(next.to_vec(), b"\xff".to_vec()), limits)
            .await
            .expect("second");
        assert_eq!(scanned_keys(&second), vec![&b"c"[..]]);
        assert!(second.next_start().is_none());
    });
}

#[test]
fn scan_accounts_bytes_and_allows_oversized_first_item() {
    let backend = mock();
    seed(&backend, &[(b"a", b"12345"), (b"b", b"67890")]);

    block_on(async {
        let mut txn = backend.begin_read().await.expect("begin read");

        // First item is 6 bytes; the second would make 12, past the limit.
        let page = txn
            .scan(
                &range(b"a", b"\xff"),
                ScanLimits {
                    item_limit: 100,
                    byte_limit: 6,
                },
            )
            .await
            .expect("scan");
        assert_eq!(page.items().len(), 1);
        assert_eq!(page.items()[0].key().as_ref(), b"a");
        assert_eq!(page.next_start().expect("more").as_ref(), b"b");

        // A first item larger than the byte limit is returned alone.
        let page = txn
            .scan(
                &range(b"a", b"\xff"),
                ScanLimits {
                    item_limit: 100,
                    byte_limit: 4,
                },
            )
            .await
            .expect("scan");
        assert_eq!(page.items().len(), 1);
        assert_eq!(page.items()[0].value().as_ref(), b"12345");
        assert_eq!(page.next_start().expect("more").as_ref(), b"b");
    });
}

#[test]
fn scan_rejects_zero_limits_before_work() {
    let backend = mock();
    seed(&backend, &[(b"a", b"1")]);

    block_on(async {
        let mut txn = backend.begin_read().await.expect("begin read");
        let zero_items = txn
            .scan(
                &range(b"a", b"\xff"),
                ScanLimits {
                    item_limit: 0,
                    byte_limit: 10,
                },
            )
            .await
            .expect_err("zero item limit");
        assert_eq!(zero_items.kind(), ErrorKind::InvalidArgument);

        let zero_bytes = txn
            .scan(
                &range(b"a", b"\xff"),
                ScanLimits {
                    item_limit: 10,
                    byte_limit: 0,
                },
            )
            .await
            .expect_err("zero byte limit");
        assert_eq!(zero_bytes.kind(), ErrorKind::InvalidArgument);
    });
}

// ---------------------------------------------------------------------------
// Mutations.
// ---------------------------------------------------------------------------

#[test]
fn batch_mutate_applies_in_input_order() {
    let backend = mock();
    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        txn.batch_mutate(vec![
            Mutation::Put {
                key: key(b"a"),
                value: key(b"1"),
            },
            Mutation::Put {
                key: key(b"a"),
                value: key(b"2"),
            },
            Mutation::Delete { key: key(b"a") },
            Mutation::Put {
                key: key(b"a"),
                value: key(b"3"),
            },
        ])
        .await
        .expect("batch mutate");
        assert_eq!(txn.get(key(b"a")).await.expect("get"), Some(key(b"3")));
        txn.commit().await.expect("commit");
    });

    block_on(async {
        let mut txn = backend.begin_read().await.expect("begin read");
        assert_eq!(txn.get(key(b"a")).await.expect("get"), Some(key(b"3")));
    });
}

#[test]
fn empty_batch_mutate_succeeds() {
    let backend = mock();
    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        txn.batch_mutate(Vec::new()).await.expect("empty batch");
        txn.commit().await.expect("commit");
    });
}

#[test]
fn unique_insert_distinguishes_inserted_from_existing() {
    let backend = mock();
    seed(&backend, &[(b"existing", b"old")]);

    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        assert_eq!(
            txn.insert(key(b"fresh"), key(b"v")).await.expect("insert"),
            InsertOutcome::Inserted
        );
        assert_eq!(
            txn.insert(key(b"existing"), key(b"v"))
                .await
                .expect("insert"),
            InsertOutcome::AlreadyExists
        );
        // The existing value is left unchanged.
        assert_eq!(
            txn.get(key(b"existing")).await.expect("get"),
            Some(key(b"old"))
        );
        // Re-inserting the just-inserted key within the same txn sees it.
        assert_eq!(
            txn.insert(key(b"fresh"), key(b"again"))
                .await
                .expect("insert"),
            InsertOutcome::AlreadyExists
        );
        txn.commit().await.expect("commit");
    });
}

#[test]
fn update_protected_read_conflict_aborts() {
    let backend = mock();
    block_on(async {
        let mut first = backend.begin_write().await.expect("begin write");
        let mut second = backend.begin_write().await.expect("begin write");

        first.get_for_update(key(b"k")).await.expect("read");
        second.get_for_update(key(b"k")).await.expect("read");

        first.put(key(b"k"), key(b"1")).await.expect("put");
        first.commit().await.expect("first commit wins");

        second.put(key(b"k"), key(b"2")).await.expect("put");
        let conflict = second.commit().await.expect_err("conflict");
        assert_eq!(conflict.kind(), ErrorKind::RetryableAbort);
    });

    block_on(async {
        let mut txn = backend.begin_read().await.expect("begin read");
        assert_eq!(txn.get(key(b"k")).await.expect("get"), Some(key(b"1")));
    });
}

// ---------------------------------------------------------------------------
// Commit outcomes and rollback.
// ---------------------------------------------------------------------------

#[test]
fn rollback_persists_nothing() {
    let backend = mock();
    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        txn.put(key(b"a"), key(b"v")).await.expect("put");
        txn.rollback().await;
    });

    block_on(async {
        let mut txn = backend.begin_read().await.expect("begin read");
        assert_eq!(txn.get(key(b"a")).await.expect("get"), None);
    });
}

#[test]
fn commit_success_is_visible_to_subsequent_snapshots() {
    let backend = mock();
    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        txn.put(key(b"a"), key(b"v")).await.expect("put");
        txn.commit().await.expect("commit succeeds");
    });

    block_on(async {
        let mut txn = backend.begin_read().await.expect("begin read");
        assert_eq!(txn.get(key(b"a")).await.expect("get"), Some(key(b"v")));
    });
}

#[test]
fn commit_failure_and_unknown_outcome_are_distinct() {
    let backend = mock();

    // Definite failure: an injected fault reports a retryable abort.
    block_on(async {
        backend.set_fault(CommitFault::Fail);
        let mut txn = backend.begin_write().await.expect("begin write");
        txn.put(key(b"a"), key(b"v")).await.expect("put");
        let error = txn.commit().await.expect_err("faulted commit fails");
        assert_eq!(error.kind(), ErrorKind::RetryableAbort);
    });

    // Unknown outcome: the mutation lands but the result is reported unknown.
    block_on(async {
        backend.set_fault(CommitFault::Unknown);
        let mut txn = backend.begin_write().await.expect("begin write");
        txn.put(key(b"b"), key(b"v")).await.expect("put");
        let error = txn.commit().await.expect_err("faulted commit is unknown");
        assert_eq!(error.kind(), ErrorKind::CommitOutcomeUnknown);
    });

    backend.set_fault(CommitFault::None);
}

// ---------------------------------------------------------------------------
// Hard limits and admission budgets.
// ---------------------------------------------------------------------------

#[test]
fn hard_limit_rejects_oversized_key_and_value() {
    let backend = MockBackend::new(
        HardLimits {
            max_key_bytes: 2,
            max_value_bytes: 2,
        },
        AdmissionBudget {
            max_mutations: 100,
            max_mutation_bytes: 1 << 20,
        },
        Capabilities {
            transactional_clear_range: false,
        },
    );

    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        let error = txn
            .put(key(b"abc"), key(b"v"))
            .await
            .expect_err("oversized key");
        assert_eq!(error.kind(), ErrorKind::LimitExceeded);

        let error = txn
            .put(key(b"a"), key(b"vvv"))
            .await
            .expect_err("oversized value");
        assert_eq!(error.kind(), ErrorKind::LimitExceeded);
    });
}

#[test]
fn admission_budget_rejects_excess_mutations_and_bytes() {
    let backend = MockBackend::new(
        HardLimits {
            max_key_bytes: 1_024,
            max_value_bytes: 1_024,
        },
        AdmissionBudget {
            max_mutations: 2,
            max_mutation_bytes: 4,
        },
        Capabilities {
            transactional_clear_range: false,
        },
    );

    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        txn.put(key(b"a"), key(b"1")).await.expect("first fits");
        txn.put(key(b"b"), key(b"2")).await.expect("second fits");
        let error = txn
            .put(key(b"c"), key(b"3"))
            .await
            .expect_err("exceeds mutation count");
        assert_eq!(error.kind(), ErrorKind::LimitExceeded);
    });

    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        // A 1-byte key plus 5-byte value overflows the four-byte budget.
        let error = txn
            .put(key(b"a"), key(b"12345"))
            .await
            .expect_err("exceeds mutation bytes");
        assert_eq!(error.kind(), ErrorKind::LimitExceeded);
    });
}

// ---------------------------------------------------------------------------
// Range clear capability.
// ---------------------------------------------------------------------------

#[test]
fn clear_range_is_declined_when_unsupported() {
    let backend = mock();
    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        let error = txn
            .clear_range(&range(b"a", b"z"))
            .await
            .expect_err("unsupported clear");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
    });
}

#[test]
fn clear_range_clears_transactionally_when_supported() {
    let backend = mock_with_clear();
    seed(&backend, &[(b"a", b"1"), (b"b", b"2"), (b"z", b"3")]);

    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        txn.clear_range(&range(b"a", b"c")).await.expect("clear");
        assert_eq!(txn.get(key(b"a")).await.expect("get"), None);
        assert_eq!(txn.get(key(b"b")).await.expect("get"), None);
        assert_eq!(txn.get(key(b"z")).await.expect("get"), Some(key(b"3")));
        txn.commit().await.expect("commit");
    });

    block_on(async {
        let mut txn = backend.begin_read().await.expect("begin read");
        assert_eq!(txn.get(key(b"a")).await.expect("get"), None);
        assert_eq!(txn.get(key(b"b")).await.expect("get"), None);
        assert_eq!(txn.get(key(b"z")).await.expect("get"), Some(key(b"3")));
    });
}
