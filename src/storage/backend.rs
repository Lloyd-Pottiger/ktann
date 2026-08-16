//! Backend-neutral transactional key-value contract.
//!
//! This module is the seam between KTANN's logical storage and any concrete
//! transactional KV backend. It owns no persistent bytes: keys and values are
//! opaque [`bytes::Bytes`], and every adapter prepends its own bounded
//! physical prefix. Concrete backends live in the `ktann-foundationdb` and
//! `ktann-rocksdb` crates; a deterministic in-memory mock exists only in tests.
//!
//! # Snapshot and transaction semantics
//!
//! A [`Backend`] opens a [`ReadTxn`] over one consistent snapshot or a
//! [`WriteTxn`] that additionally sees its own uncommitted writes
//! (read-your-writes) and commits them atomically. Conflicts are established
//! only through update-protected point reads ([`WriteTxn::get_for_update`] and
//! [`WriteTxn::batch_get_for_update`]); the contract makes no promise about
//! range conflict semantics.
//!
//! A [`WriteTxn`]'s `commit` reports one of three distinguishable outcomes:
//! definite success (`Ok`), definite failure (`ErrorKind::RetryableAbort`),
//! or an unknown outcome (`ErrorKind::CommitOutcomeUnknown`). `rollback`
//! abandons the transaction without persisting any write. Dropping a
//! transaction before commit is equivalent to rollback.
//!
//! # Static dispatch and borrowing
//!
//! [`Backend`] uses generic associated transaction types so that a transaction
//! handle can borrow its backend ([`ReadTxn<'backend>`](ReadTxn) and
//! [`WriteTxn<'backend>`](WriteTxn)), matching native drivers whose
//! transaction lifetime is tied to their database. Trait methods return stable
//! `impl Future + Send` futures directly — no `async_trait`, boxed futures,
//! trait objects, closed enum, or lifetime-erasing `unsafe`. Transaction types
//! are required to be `Send` but deliberately not required to be `Sync`; a
//! backend may hold thread-affine or cell-like state.
//!
//! Every operation takes `&mut self`, serializing access within one
//! transaction; deliberate parallelism is expressed only through the batch
//! primitives ([`ReadOps::batch_get`], [`WriteTxn::batch_get_for_update`], and
//! [`WriteTxn::batch_mutate`]).

use std::fmt;
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use bytes::Bytes;

use crate::api::{Error, ErrorKind, Result};
use crate::storage::keys::KeyRange;

/// Stable physical ceilings that a backend declares as storage-engine facts.
///
/// These are the backend's hard limits on encoded keys and values. They are
/// distinct from the conservative [`AdmissionBudget`]: exceeding a hard limit
/// is a native rejection, while the budget bounds KTANN's own work early.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HardLimits {
    /// The maximum encoded key length in bytes.
    pub max_key_bytes: usize,
    /// The maximum value length in bytes.
    pub max_value_bytes: usize,
}

/// A conservative adapter admission budget used to bound KTANN work early.
///
/// This is policy, not a storage-engine fact: staying below the budget does not
/// prove that a backend's native affected-data accounting will accept a
/// transaction. It bounds the mutations one transaction may attempt.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AdmissionBudget {
    /// The maximum number of mutations in one transaction.
    pub max_mutations: usize,
    /// The maximum total encoded key plus value bytes mutated in one
    /// transaction.
    pub max_mutation_bytes: usize,
}

/// Backend capabilities that adapters declare explicitly.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Capabilities {
    /// Whether [`WriteTxn::clear_range`] is supported.
    pub transactional_clear_range: bool,
}

/// The adapter-side right to cross an irreversible commit point.
///
/// A backend adapter must consume this handle with [`CommitStart::begin`]
/// after all cancellation-safe admission waits and immediately before starting
/// its native commit. This lets Runtime cancellation race atomically with the
/// real backend commit point instead of an earlier caller-side approximation.
#[derive(Debug)]
pub struct CommitStart {
    claim: Option<Arc<AtomicBool>>,
}

impl CommitStart {
    fn uncontrolled() -> Self {
        Self { claim: None }
    }

    /// Claims commit ownership immediately before native commit begins.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Cancelled`] when pre-commit cancellation won the
    /// race. After this succeeds, the adapter must start native commit without
    /// another cancellation-safe wait.
    pub fn begin(self) -> Result<()> {
        let Some(claim) = self.claim else {
            return Ok(());
        };
        claim
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| ())
            .map_err(|_| Error::new(ErrorKind::Cancelled))
    }
}

/// The Runtime-side right to cancel before native commit starts.
#[derive(Debug)]
pub(crate) struct CommitCancellation {
    claim: Arc<AtomicBool>,
}

impl CommitCancellation {
    pub(crate) fn pair() -> (Self, CommitStart) {
        let claim = Arc::new(AtomicBool::new(false));
        (
            Self {
                claim: Arc::clone(&claim),
            },
            CommitStart { claim: Some(claim) },
        )
    }

    pub(crate) fn cancel(&self) -> bool {
        self.claim
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }
}

/// Explicit bounds on one forward scan page.
///
/// Both limits must be non-zero; [`WriteTxn::scan`] and [`ReadTxn::scan`]
/// reject a zero limit before doing any work.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ScanLimits {
    /// The maximum number of items returned by one page.
    pub item_limit: usize,
    /// The maximum total page bytes (encoded key length plus value length).
    pub byte_limit: usize,
}

/// One ordered key/value item returned by a scan.
///
/// `Debug` is redacted because keys and values may carry sensitive bytes.
#[derive(Clone, Eq, PartialEq)]
pub struct ScanItem {
    key: Bytes,
    value: Bytes,
}

impl ScanItem {
    /// Constructs a scan item from owned key and value bytes.
    #[must_use]
    pub fn new(key: Bytes, value: Bytes) -> Self {
        Self { key, value }
    }

    /// The item's key.
    #[must_use]
    pub fn key(&self) -> &Bytes {
        &self.key
    }

    /// The item's value.
    #[must_use]
    pub fn value(&self) -> &Bytes {
        &self.value
    }
}

impl fmt::Debug for ScanItem {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ScanItem")
            .field("key", &"[REDACTED]")
            .field("value", &"[REDACTED]")
            .finish()
    }
}

/// One bounded, strictly ordered page of scan results.
///
/// A page is terminal when [`next_start`](ScanPage::next_start) is `None`; a
/// non-terminal page is always non-empty and its `next_start` strictly follows
/// the page's last key. `Debug` is redacted because keys and values may carry
/// sensitive bytes.
#[derive(Clone, Eq, PartialEq)]
pub struct ScanPage {
    items: Vec<ScanItem>,
    next_start: Option<Bytes>,
}

impl ScanPage {
    /// Constructs a scan page from its items and optional cursor.
    #[must_use]
    pub fn new(items: Vec<ScanItem>, next_start: Option<Bytes>) -> Self {
        Self { items, next_start }
    }

    /// The ordered page items.
    #[must_use]
    pub fn items(&self) -> &[ScanItem] {
        &self.items
    }

    /// The strict lower bound for the next page, or `None` if the range is
    /// exhausted.
    #[must_use]
    pub fn next_start(&self) -> Option<&Bytes> {
        self.next_start.as_ref()
    }
}

impl fmt::Debug for ScanPage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ScanPage")
            .field("items", &self.items.len())
            .field(
                "next_start",
                &self.next_start.as_ref().map(|_| "[REDACTED]"),
            )
            .finish()
    }
}

/// One point mutation in an ordered [`WriteTxn::batch_mutate`].
///
/// Unique insertion is a standalone primitive ([`WriteTxn::insert`]); batches
/// carry plain puts and deletes that take effect in input order. `Debug` is
/// redacted because keys and values may carry sensitive bytes.
#[derive(Clone, Eq, PartialEq)]
#[non_exhaustive]
pub enum Mutation {
    /// Writes or replaces `key` with `value`.
    Put {
        /// The key.
        key: Bytes,
        /// The value.
        value: Bytes,
    },
    /// Deletes `key`; deleting an absent key succeeds.
    Delete {
        /// The key.
        key: Bytes,
    },
}

impl fmt::Debug for Mutation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::Put { .. } => "Put([REDACTED])",
            Self::Delete { .. } => "Delete([REDACTED])",
        })
    }
}

/// The result of a unique [`WriteTxn::insert`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum InsertOutcome {
    /// The key was absent and has been inserted.
    Inserted,
    /// The key already exists and was left unchanged.
    AlreadyExists,
}

/// The shared snapshot read operations.
///
/// A type implementing [`ReadOps`] reads from one consistent snapshot. It must
/// be `Send`; it is not required to be `Sync`. All methods take `&mut self` so
/// callers serialize access within one transaction.
pub trait ReadOps: Send {
    /// Reads the value at `key`, or `None` if absent.
    fn get(&mut self, key: Bytes) -> impl Future<Output = Result<Option<Bytes>>> + Send;

    /// Reads the value for each key, preserving input order and duplicate keys.
    ///
    /// The result has one entry per input key: `Some(value)` when present and
    /// `None` when absent. An empty input succeeds with an empty result.
    fn batch_get(
        &mut self,
        keys: Vec<Bytes>,
    ) -> impl Future<Output = Result<Vec<Option<Bytes>>>> + Send;

    /// Scans the half-open `[start, end)` key range in ascending key order.
    ///
    /// The page returns at most `limits.item_limit` items and `limits.byte_limit`
    /// total bytes, where each item counts its encoded key length plus its value
    /// length. Both limits must be non-zero and are checked before any work.
    /// A single oversized first item may be returned alone even when it exceeds
    /// `byte_limit`, so any value within the backend's hard limits remains
    /// readable; no other item may push the page past `byte_limit`. A
    /// non-terminal page is non-empty and carries a `next_start` that strictly
    /// follows its last key.
    fn scan(
        &mut self,
        range: &KeyRange,
        limits: ScanLimits,
    ) -> impl Future<Output = Result<ScanPage>> + Send;
}

/// A read transaction opened by [`Backend::begin_read`].
///
/// This marker trait carries no mutation capability so read-only index paths
/// can never receive a write transaction. It reads one consistent snapshot.
pub trait ReadTxn: ReadOps {}

/// A write transaction opened by [`Backend::begin_write`].
///
/// A write transaction reads one consistent snapshot, sees its own uncommitted
/// writes, establishes conflicts through update-protected point reads, and
/// commits or rolls back atomically. It is `Send` but not required to be
/// `Sync`.
pub trait WriteTxn: ReadOps {
    /// Reads the value at `key` and establishes a conflict on that key.
    ///
    /// A later commit reports `ErrorKind::RetryableAbort` if another
    /// transaction committed a change to `key` after this read.
    fn get_for_update(&mut self, key: Bytes) -> impl Future<Output = Result<Option<Bytes>>> + Send;

    /// Reads the values for `keys` and establishes a conflict on each key,
    /// preserving input order and duplicate keys.
    fn batch_get_for_update(
        &mut self,
        keys: Vec<Bytes>,
    ) -> impl Future<Output = Result<Vec<Option<Bytes>>>> + Send;

    /// Writes or replaces `key` with `value`.
    fn put(&mut self, key: Bytes, value: Bytes) -> impl Future<Output = Result<()>> + Send;

    /// Inserts `value` at `key` only if `key` is absent.
    ///
    /// Returns [`InsertOutcome::Inserted`] or [`InsertOutcome::AlreadyExists`]
    /// without overwriting an existing value.
    fn insert(
        &mut self,
        key: Bytes,
        value: Bytes,
    ) -> impl Future<Output = Result<InsertOutcome>> + Send;

    /// Deletes `key`; deleting an absent key succeeds.
    fn delete(&mut self, key: Bytes) -> impl Future<Output = Result<()>> + Send;

    /// Applies point mutations in input order within this transaction.
    ///
    /// An empty batch succeeds. Mutations take effect in the order given, so a
    /// later mutation to the same key supersedes an earlier one.
    fn batch_mutate(&mut self, mutations: Vec<Mutation>)
    -> impl Future<Output = Result<()>> + Send;

    /// Clears every key in the half-open `[start, end)` range transactionally.
    ///
    /// Returns `ErrorKind::Unsupported` when the backend does not advertise
    /// transactional range clear. When supported, the clear composes with the
    /// transaction's other reads and mutations and takes effect on commit.
    fn clear_range(&mut self, range: &KeyRange) -> impl Future<Output = Result<()>> + Send;

    /// Commits this transaction's mutations atomically, consuming it.
    ///
    /// Returns `Ok(())` on definite success, `ErrorKind::RetryableAbort` on a
    /// definite failure that left nothing committed, or
    /// `ErrorKind::CommitOutcomeUnknown` when the result cannot be
    /// determined. A successful commit makes every mutation visible to any
    /// transaction begun after it returns.
    fn commit(self) -> impl Future<Output = Result<()>> + Send
    where
        Self: Sized,
    {
        self.commit_with(CommitStart::uncontrolled())
    }

    /// Commits after coordinating the adapter's actual native commit point.
    ///
    /// Adapters must call [`CommitStart::begin`] after any asynchronous
    /// resource admission and immediately before starting native commit.
    fn commit_with(self, start: CommitStart) -> impl Future<Output = Result<()>> + Send;

    /// Abandons this transaction without persisting any mutation, consuming it.
    ///
    /// Rollback is infallible and equivalent to dropping the transaction.
    fn rollback(self) -> impl Future<Output = ()> + Send;
}

/// A transactional KV backend.
///
/// `Backend` is `Send + Sync + 'static` and exposes generic associated
/// transaction types that borrow the backend, so transaction handles never
/// escape the backend they were opened against. Adapters clone an
/// `Arc<Backend>` and begin transactions on their own async stack.
pub trait Backend: Send + Sync + 'static {
    /// The read transaction type borrowing this backend.
    type ReadTxn<'backend>: ReadTxn + Send + 'backend
    where
        Self: 'backend;

    /// The write transaction type borrowing this backend.
    type WriteTxn<'backend>: WriteTxn + Send + 'backend
    where
        Self: 'backend;

    /// The backend's hard key and value limits.
    fn hard_limits(&self) -> HardLimits;

    /// The backend's conservative admission budget.
    fn admission_budget(&self) -> AdmissionBudget;

    /// The backend's declared capabilities.
    fn capabilities(&self) -> Capabilities;

    /// Waits for backend-native resources detached from transaction handles.
    ///
    /// Runtime calls this after foreground work drains and before dropping the
    /// backend. Adapters without detached cleanup use the no-op default.
    fn shutdown(&self) -> impl Future<Output = ()> + Send + '_ {
        async {}
    }

    /// Opens a read transaction over the current consistent snapshot.
    fn begin_read(&self) -> impl Future<Output = Result<Self::ReadTxn<'_>>> + Send + '_;

    /// Opens a write transaction over the current consistent snapshot.
    fn begin_write(&self) -> impl Future<Output = Result<Self::WriteTxn<'_>>> + Send + '_;
}
