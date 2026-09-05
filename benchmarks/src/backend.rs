//! Benchmark-only Backend instrumentation for logical IO and write amplification.
//!
//! The wrapper preserves the production Backend's transaction and error
//! semantics. It counts calls at the same boundary KTANN sees: reads are
//! charged only after the Backend returns data, while mutations are charged
//! when attempted so retry work remains visible. The runner snapshots these
//! monotonic counters immediately before and after the timed region.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bytes::Bytes;
use ktann::api::{ErrorKind, Result};
use ktann::storage::backend::{
    AdmissionBudget, Backend, Capabilities, CommitStart, HardLimits, InsertOutcome, Mutation,
    ReadOps, ReadTxn, ScanLimits, ScanPage, WriteTxn,
};
use ktann::storage::keys::KeyRange;

use crate::report::BackendIo;

/// A production Backend decorated with benchmark-only logical IO counters.
#[derive(Debug)]
pub struct MeasuredBackend<B> {
    /// Production adapter receiving every operation unchanged.
    inner: Arc<B>,
    /// Monotonic logical-IO observations shared with all child transactions.
    counters: Arc<Counters>,
}

impl<B> MeasuredBackend<B> {
    /// Wraps a Backend and returns a shareable counter handle.
    #[must_use]
    pub fn new(inner: B) -> (Self, BackendCounters) {
        let counters = Arc::new(Counters::default());
        (
            Self {
                inner: Arc::new(inner),
                counters: Arc::clone(&counters),
            },
            BackendCounters { counters },
        )
    }
}

impl<B> Clone for MeasuredBackend<B> {
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
            counters: Arc::clone(&self.counters),
        }
    }
}

/// Shareable snapshots of benchmark Backend counters.
#[derive(Clone, Debug)]
pub struct BackendCounters {
    /// Same counter set owned by the Backend wrapper and its transactions.
    counters: Arc<Counters>,
}

impl BackendCounters {
    /// Reads all counters without resetting them.
    #[must_use]
    pub fn snapshot(&self) -> BackendIo {
        self.counters.snapshot()
    }

    /// Returns work performed since `before`.
    #[must_use]
    pub fn since(&self, before: &BackendIo) -> BackendIo {
        subtract(self.snapshot(), before)
    }
}

#[derive(Debug, Default)]
/// Monotonic process-local counters shared by every wrapped transaction.
///
/// Relaxed atomics are sufficient because counters carry no synchronization
/// invariant: the runner joins all measured operations before its final
/// snapshot, and each field is an independent saturating observation.
struct Counters {
    /// Successfully opened read transactions.
    read_transactions: AtomicU64,
    /// Successfully opened write transactions, including whole-operation retries.
    write_transactions: AtomicU64,
    /// Keys requested through point and batch point reads.
    point_read_keys: AtomicU64,
    /// Bounded range-scan calls issued to the Backend.
    scans: AtomicU64,
    /// Present logical key/value pairs returned by reads.
    items_read: AtomicU64,
    /// Logical key plus value bytes returned by reads.
    bytes_read: AtomicU64,
    /// Attempted point mutations, including attempts later retried.
    mutation_operations: AtomicU64,
    /// Attempted logical mutation key/value bytes.
    mutation_bytes: AtomicU64,
    /// Attempted transactional range clears.
    range_clears: AtomicU64,
    /// Definitely committed write transactions.
    commits: AtomicU64,
    /// Commit attempts classified as definitely retryable.
    retryable_commits: AtomicU64,
    /// Commit attempts whose outcome is unknown.
    unknown_commits: AtomicU64,
    /// Commit attempts with another terminal error.
    failed_commits: AtomicU64,
}

impl Counters {
    /// Copies the independent atomic observations into the report schema.
    fn snapshot(&self) -> BackendIo {
        BackendIo {
            read_transactions: load(&self.read_transactions),
            write_transactions: load(&self.write_transactions),
            point_read_keys: load(&self.point_read_keys),
            scans: load(&self.scans),
            items_read: load(&self.items_read),
            bytes_read: load(&self.bytes_read),
            mutation_operations: load(&self.mutation_operations),
            mutation_bytes: load(&self.mutation_bytes),
            range_clears: load(&self.range_clears),
            commits: load(&self.commits),
            retryable_commits: load(&self.retryable_commits),
            unknown_commits: load(&self.unknown_commits),
            failed_commits: load(&self.failed_commits),
        }
    }

    /// Charges one present point-read result by its logical key/value bytes.
    fn read_result(&self, key_bytes: usize, value: Option<&Bytes>) {
        add(&self.items_read, usize::from(value.is_some()));
        add(
            &self.bytes_read,
            value.map_or(0, |value| key_bytes.saturating_add(value.len())),
        );
    }

    /// Aggregates one batch result before touching shared atomic counters.
    fn batch_read_result(&self, key_bytes: &[usize], values: &[Option<Bytes>]) {
        let (items, bytes) = key_bytes.iter().zip(values).fold(
            (0_usize, 0_usize),
            |(items, bytes), (key_bytes, value)| match value {
                Some(value) => (
                    items.saturating_add(1),
                    bytes.saturating_add(key_bytes.saturating_add(value.len())),
                ),
                None => (items, bytes),
            },
        );
        add(&self.items_read, items);
        add(&self.bytes_read, bytes);
    }

    /// Charges all logical items returned by one bounded scan page.
    fn scan_result(&self, page: &ScanPage) {
        add(&self.items_read, page.items().len());
        let bytes = page.items().iter().fold(0_usize, |sum, item| {
            sum.saturating_add(item.key().len())
                .saturating_add(item.value().len())
        });
        add(&self.bytes_read, bytes);
    }

    /// Classifies the Backend's caller-visible commit outcome exactly once.
    fn commit_result(&self, result: &Result<()>) {
        let counter = match result {
            Ok(()) => &self.commits,
            Err(error) if error.kind() == ErrorKind::RetryableAbort => &self.retryable_commits,
            Err(error) if error.kind() == ErrorKind::CommitOutcomeUnknown => &self.unknown_commits,
            Err(_) => &self.failed_commits,
        };
        add(counter, 1);
    }
}

/// Loads one independent observation without imposing synchronization on the hot path.
fn load(counter: &AtomicU64) -> u64 {
    counter.load(Ordering::Relaxed)
}

/// Adds one bounded workload charge with a single relaxed atomic operation.
fn add(counter: &AtomicU64, value: usize) {
    let value = u64::try_from(value).unwrap_or(u64::MAX);
    // Scenario and Backend hard limits bound every counter many orders of
    // magnitude below u64::MAX, so overflow is not a reachable benchmark state.
    counter.fetch_add(value, Ordering::Relaxed);
}

/// Computes the timed-region delta from two monotonic snapshots.
fn subtract(after: BackendIo, before: &BackendIo) -> BackendIo {
    after
        .checked_sub(before)
        .expect("monotonic Backend counters never decrease")
}

/// Read transaction forwarding calls while accounting returned logical data.
#[derive(Debug)]
pub struct MeasuredReadTxn<T> {
    /// Production read transaction receiving forwarded calls.
    inner: T,
    /// Shared observations updated only after successful read results.
    counters: Arc<Counters>,
}

/// Write transaction forwarding calls while accounting attempted logical work.
#[derive(Debug)]
pub struct MeasuredWriteTxn<T> {
    /// Production write transaction receiving forwarded calls.
    inner: T,
    /// Shared observations charged when logical mutations are attempted.
    counters: Arc<Counters>,
}

impl<T: ReadOps> ReadOps for MeasuredReadTxn<T> {
    fn get(&mut self, key: Bytes) -> impl Future<Output = Result<Option<Bytes>>> + Send {
        let key_bytes = key.len();
        add(&self.counters.point_read_keys, 1);
        async move {
            let result = self.inner.get(key).await?;
            self.counters.read_result(key_bytes, result.as_ref());
            Ok(result)
        }
    }

    fn batch_get(
        &mut self,
        keys: Vec<Bytes>,
    ) -> impl Future<Output = Result<Vec<Option<Bytes>>>> + Send {
        let key_bytes: Vec<usize> = keys.iter().map(Bytes::len).collect();
        add(&self.counters.point_read_keys, keys.len());
        async move {
            let result = self.inner.batch_get(keys).await?;
            self.counters.batch_read_result(&key_bytes, &result);
            Ok(result)
        }
    }

    fn scan(
        &mut self,
        range: &KeyRange,
        limits: ScanLimits,
    ) -> impl Future<Output = Result<ScanPage>> + Send {
        add(&self.counters.scans, 1);
        async move {
            let result = self.inner.scan(range, limits).await?;
            self.counters.scan_result(&result);
            Ok(result)
        }
    }

    fn batch_scan(
        &mut self,
        ranges: &[KeyRange],
        limits: ScanLimits,
    ) -> impl Future<Output = Result<Vec<ScanPage>>> + Send {
        // One batched call is one backend interaction regardless of range
        // count, so it charges one scan; returned items still charge fully.
        add(&self.counters.scans, 1);
        async move {
            let pages = self.inner.batch_scan(ranges, limits).await?;
            for page in &pages {
                self.counters.scan_result(page);
            }
            Ok(pages)
        }
    }
}

impl<T: ReadTxn> ReadTxn for MeasuredReadTxn<T> {}

impl<T: WriteTxn> ReadOps for MeasuredWriteTxn<T> {
    fn get(&mut self, key: Bytes) -> impl Future<Output = Result<Option<Bytes>>> + Send {
        let key_bytes = key.len();
        add(&self.counters.point_read_keys, 1);
        async move {
            let result = self.inner.get(key).await?;
            self.counters.read_result(key_bytes, result.as_ref());
            Ok(result)
        }
    }

    fn batch_get(
        &mut self,
        keys: Vec<Bytes>,
    ) -> impl Future<Output = Result<Vec<Option<Bytes>>>> + Send {
        let key_bytes: Vec<usize> = keys.iter().map(Bytes::len).collect();
        add(&self.counters.point_read_keys, keys.len());
        async move {
            let result = self.inner.batch_get(keys).await?;
            self.counters.batch_read_result(&key_bytes, &result);
            Ok(result)
        }
    }

    fn scan(
        &mut self,
        range: &KeyRange,
        limits: ScanLimits,
    ) -> impl Future<Output = Result<ScanPage>> + Send {
        add(&self.counters.scans, 1);
        async move {
            let result = self.inner.scan(range, limits).await?;
            self.counters.scan_result(&result);
            Ok(result)
        }
    }

    fn batch_scan(
        &mut self,
        ranges: &[KeyRange],
        limits: ScanLimits,
    ) -> impl Future<Output = Result<Vec<ScanPage>>> + Send {
        // One batched call is one backend interaction regardless of range
        // count, so it charges one scan; returned items still charge fully.
        add(&self.counters.scans, 1);
        async move {
            let pages = self.inner.batch_scan(ranges, limits).await?;
            for page in &pages {
                self.counters.scan_result(page);
            }
            Ok(pages)
        }
    }
}

impl<T: WriteTxn> WriteTxn for MeasuredWriteTxn<T> {
    fn get_for_update(&mut self, key: Bytes) -> impl Future<Output = Result<Option<Bytes>>> + Send {
        let key_bytes = key.len();
        add(&self.counters.point_read_keys, 1);
        async move {
            let result = self.inner.get_for_update(key).await?;
            self.counters.read_result(key_bytes, result.as_ref());
            Ok(result)
        }
    }

    fn batch_get_for_update(
        &mut self,
        keys: Vec<Bytes>,
    ) -> impl Future<Output = Result<Vec<Option<Bytes>>>> + Send {
        let key_bytes: Vec<usize> = keys.iter().map(Bytes::len).collect();
        add(&self.counters.point_read_keys, keys.len());
        async move {
            let result = self.inner.batch_get_for_update(keys).await?;
            self.counters.batch_read_result(&key_bytes, &result);
            Ok(result)
        }
    }

    fn put(&mut self, key: Bytes, value: Bytes) -> impl Future<Output = Result<()>> + Send {
        add(&self.counters.mutation_operations, 1);
        add(
            &self.counters.mutation_bytes,
            key.len().saturating_add(value.len()),
        );
        self.inner.put(key, value)
    }

    fn insert(
        &mut self,
        key: Bytes,
        value: Bytes,
    ) -> impl Future<Output = Result<InsertOutcome>> + Send {
        add(&self.counters.mutation_operations, 1);
        add(
            &self.counters.mutation_bytes,
            key.len().saturating_add(value.len()),
        );
        self.inner.insert(key, value)
    }

    fn delete(&mut self, key: Bytes) -> impl Future<Output = Result<()>> + Send {
        add(&self.counters.mutation_operations, 1);
        add(&self.counters.mutation_bytes, key.len());
        self.inner.delete(key)
    }

    fn batch_mutate(
        &mut self,
        mutations: Vec<Mutation>,
    ) -> impl Future<Output = Result<()>> + Send {
        add(&self.counters.mutation_operations, mutations.len());
        // Mutation fields remain available at this boundary, so logical bytes
        // include exactly the encoded keys and values handed to the Backend.
        let bytes = mutations.iter().fold(0_usize, |sum, mutation| {
            let charge = match mutation {
                Mutation::Put { key, value } => key.len().saturating_add(value.len()),
                Mutation::Delete { key } => key.len(),
                _ => 0,
            };
            sum.saturating_add(charge)
        });
        add(&self.counters.mutation_bytes, bytes);
        self.inner.batch_mutate(mutations)
    }

    fn clear_range(&mut self, range: &KeyRange) -> impl Future<Output = Result<()>> + Send {
        add(&self.counters.range_clears, 1);
        self.inner.clear_range(range)
    }

    async fn commit_with(self, start: CommitStart) -> Result<()> {
        // Count only the result of the native commit attempt. A rollback and a
        // step error never appear as commits, while a retryable commit remains
        // visible even if Runtime later replays the whole transaction.
        let Self { inner, counters } = self;
        let result = inner.commit_with(start).await;
        counters.commit_result(&result);
        result
    }

    fn rollback(self) -> impl Future<Output = ()> + Send {
        self.inner.rollback()
    }
}

impl<B: Backend> Backend for MeasuredBackend<B> {
    type ReadTxn<'backend>
        = MeasuredReadTxn<B::ReadTxn<'backend>>
    where
        Self: 'backend;
    type WriteTxn<'backend>
        = MeasuredWriteTxn<B::WriteTxn<'backend>>
    where
        Self: 'backend;

    fn hard_limits(&self) -> HardLimits {
        self.inner.hard_limits()
    }

    fn admission_budget(&self) -> AdmissionBudget {
        self.inner.admission_budget()
    }

    fn capabilities(&self) -> Capabilities {
        self.inner.capabilities()
    }

    fn shutdown(&self) -> impl Future<Output = ()> + Send + '_ {
        self.inner.shutdown()
    }

    async fn begin_read(&self) -> Result<Self::ReadTxn<'_>> {
        let inner = self.inner.begin_read().await?;
        add(&self.counters.read_transactions, 1);
        Ok(MeasuredReadTxn {
            inner,
            counters: Arc::clone(&self.counters),
        })
    }

    async fn begin_write(&self) -> Result<Self::WriteTxn<'_>> {
        let inner = self.inner.begin_write().await?;
        add(&self.counters.write_transactions, 1);
        Ok(MeasuredWriteTxn {
            inner,
            counters: Arc::clone(&self.counters),
        })
    }
}
