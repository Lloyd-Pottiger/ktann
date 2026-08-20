//! Process-local foreground admission and orderly shutdown.

use std::fmt;
use std::future::{Future, pending};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use tokio::runtime::{Handle, RuntimeFlavor};
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore, TryAcquireError};
use tokio::task::{AbortHandle, JoinHandle};
use tokio_util::sync::CancellationToken;

use crate::api::{
    Error, ErrorKind, Index, IndexConfig, IndexName, OperationOptions, Result, RuntimeConfig,
};
use crate::search::cache::PartitionCache;
use crate::storage::backend::{Backend, CommitCancellation, CommitStart};

pub(crate) mod import;
pub(crate) mod lifecycle;
pub(crate) mod reads;
pub(crate) mod search;

/// Owns one backend and its process-local foreground operation lifecycle.
///
/// Clone handles share admission capacity and shutdown state. Dropping the last
/// handle begins the same non-blocking cleanup as [`shutdown`](Self::shutdown);
/// callers that must know cleanup completed should call `shutdown` explicitly.
pub struct Runtime<B: Backend> {
    handle: Arc<RuntimeHandle<B>>,
}

impl<B: Backend> Runtime<B> {
    /// Creates a Runtime on the current Tokio multi-thread runtime.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::InvalidArgument`] when `config` is invalid or the
    /// caller is not inside an active Tokio multi-thread runtime.
    pub fn new(backend: B, config: RuntimeConfig) -> Result<Self> {
        config.validate()?;
        let executor = Handle::try_current()
            .map_err(|source| Error::with_source(ErrorKind::InvalidArgument, source))?;
        if executor.runtime_flavor() != RuntimeFlavor::MultiThread {
            return Err(Error::new(ErrorKind::InvalidArgument));
        }

        let foreground_limit = config.foreground_operation_limit();
        let partition_cache = Arc::new(PartitionCache::new(config.partition_cache_bytes()));
        Ok(Self {
            handle: Arc::new(RuntimeHandle {
                inner: Arc::new(RuntimeInner {
                    executor,
                    config,
                    partition_cache,
                    foreground: Arc::new(Semaphore::new(foreground_limit)),
                    foreground_waiting: Arc::new(Semaphore::new(foreground_limit)),
                    lifecycle: Mutex::new(Lifecycle {
                        phase: Phase::Accepting,
                        active: 0,
                        backend: Some(Arc::new(backend)),
                    }),
                    terminal: Notify::new(),
                }),
            }),
        })
    }

    /// Returns the validated process-local configuration.
    #[must_use]
    pub fn config(&self) -> &RuntimeConfig {
        &self.handle.inner.config
    }

    /// Creates one Logical Index and returns a cloneable Active handle.
    ///
    /// Create is idempotent for the same Index Name and identical immutable
    /// configuration: retrying after an unknown commit outcome recovers the
    /// current index from a fresh snapshot. A different configuration returns
    /// [`ErrorKind::IndexAlreadyExists`], and a Dropping same-name index
    /// returns [`ErrorKind::IndexDropping`].
    pub async fn create_index(&self, name: &str, config: IndexConfig) -> Result<Index<B>> {
        self.create_index_with_control(name, config, OperationOptions::default())
            .await
    }

    /// Creates one Logical Index with explicit operation control.
    pub async fn create_index_with_control(
        &self,
        name: &str,
        config: IndexConfig,
        options: OperationOptions,
    ) -> Result<Index<B>> {
        let name = IndexName::new(name)?;
        config.validate()?;
        let retry = lifecycle::RetryPolicy::from_config(self.config());
        let handle_name = name.clone();
        let manifest = self
            .run_foreground(options, move |mut context| async move {
                lifecycle::create_index(&mut context, name, config, retry).await
            })
            .await?;
        Index::new(Arc::clone(&self.handle.inner), handle_name, manifest)
    }

    /// Opens the current Active Logical Index for one Index Name.
    ///
    /// [`ErrorKind::IndexNotFound`] means no Index Name mapping exists.
    /// [`ErrorKind::IndexDropping`] reports a persisted Dropping Manifest, and
    /// an unsupported or malformed Manifest fails closed before a handle is
    /// returned.
    pub async fn open_index(&self, name: &str) -> Result<Index<B>> {
        self.open_index_with_control(name, OperationOptions::default())
            .await
    }

    /// Opens one Logical Index with explicit operation control.
    pub async fn open_index_with_control(
        &self,
        name: &str,
        options: OperationOptions,
    ) -> Result<Index<B>> {
        let name = IndexName::new(name)?;
        let handle_name = name.clone();
        let manifest = self
            .run_foreground(options, move |mut context| async move {
                lifecycle::open_index(&mut context, name).await
            })
            .await?;
        Index::new(Arc::clone(&self.handle.inner), handle_name, manifest)
    }

    /// Drops one Logical Index idempotently.
    ///
    /// Drop first persists the Dropping Manifest state, deletes only the
    /// index-owned range, and removes the Index Name mapping last. It is
    /// bounded and resumable on backends without transactional range clear,
    /// and safe to retry after a commit of unknown outcome.
    pub async fn drop_index(&self, name: &str) -> Result<()> {
        self.drop_index_with_control(name, OperationOptions::default())
            .await
    }

    /// Drops one Logical Index with explicit operation control.
    pub async fn drop_index_with_control(
        &self,
        name: &str,
        options: OperationOptions,
    ) -> Result<()> {
        let name = IndexName::new(name)?;
        let retry = lifecycle::RetryPolicy::from_config(self.config());
        self.run_foreground(options, move |mut context| async move {
            lifecycle::drop_index(&mut context, name, retry).await
        })
        .await
    }

    /// Stops admission and waits for all admitted foreground work to finish.
    ///
    /// Shutdown is idempotent. Operations admitted before shutdown retain their
    /// real results; only later admission returns [`ErrorKind::RuntimeClosed`].
    pub async fn shutdown(&self) -> Result<()> {
        self.handle.inner.begin_shutdown();
        loop {
            let terminal = self.handle.inner.terminal.notified();
            if self.handle.inner.phase() == Phase::Closed {
                return Ok(());
            }
            terminal.await;
        }
    }

    pub(crate) async fn run_foreground<T, F, Fut>(
        &self,
        options: OperationOptions,
        operation: F,
    ) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(OperationContext<B>) -> Fut + Send + 'static,
        Fut: Future<Output = Result<T>> + Send + 'static,
    {
        self.handle.inner.run_foreground(options, operation).await
    }
}

impl<B: Backend> Clone for Runtime<B> {
    fn clone(&self) -> Self {
        Self {
            handle: Arc::clone(&self.handle),
        }
    }
}

impl<B: Backend> fmt::Debug for Runtime<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let lifecycle = self.handle.inner.lock_lifecycle();
        formatter
            .debug_struct("Runtime")
            .field("phase", &lifecycle.phase)
            .field("active_foreground", &lifecycle.active)
            .field(
                "foreground_limit",
                &self.handle.inner.config.foreground_operation_limit(),
            )
            .finish()
    }
}

struct RuntimeHandle<B: Backend> {
    inner: Arc<RuntimeInner<B>>,
}

impl<B: Backend> Drop for RuntimeHandle<B> {
    fn drop(&mut self) {
        self.inner.begin_shutdown();
    }
}

pub(crate) struct OperationContext<B: Backend> {
    backend: Arc<B>,
    options: OperationOptions,
    commit_start: Option<CommitStart>,
}

impl<B: Backend> OperationContext<B> {
    pub(crate) fn backend(&self) -> Arc<B> {
        Arc::clone(&self.backend)
    }

    pub(crate) fn checkpoint(&self) -> Result<()> {
        check_control(&self.options)
    }

    /// Starts one commit attempt at the Runtime's cancellation boundary.
    ///
    /// The first attempt consumes the guarded boundary. A resumable lifecycle
    /// operation may perform several bounded transactions; after the first
    /// native commit has started, the caller-side cancellation race is already
    /// decided and later attempts commit without re-arming that one-shot
    /// boundary. Callers still call [`OperationContext::checkpoint`] before
    /// every attempt through [`OperationContext::commit`].
    pub(crate) async fn commit<T, F, Fut>(&mut self, commit: F) -> Result<T>
    where
        F: FnOnce(CommitStart) -> Fut,
        Fut: Future<Output = Result<T>>,
    {
        self.checkpoint()?;
        let start = self
            .commit_start
            .take()
            .unwrap_or_else(CommitStart::uncontrolled);
        commit(start).await
    }
}

/// Cancels owned work on caller drop only while commit has not started.
struct CancelBeforeCommit {
    task: Option<AbortHandle>,
    commit_cancellation: CommitCancellation,
}

impl CancelBeforeCommit {
    fn disarm(&mut self) {
        self.task = None;
    }
}

impl Drop for CancelBeforeCommit {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            if self.commit_cancellation.cancel() {
                task.abort();
            }
        }
    }
}

async fn join_task<T>(task: &mut JoinHandle<Result<T>>) -> Result<T> {
    task.await
        .map_err(|source| Error::with_source(ErrorKind::Backend, source))?
}

async fn cancel_task_before_commit<T>(
    task: &mut JoinHandle<Result<T>>,
    commit_cancellation: &CommitCancellation,
    error_kind: ErrorKind,
) -> Result<T> {
    if commit_cancellation.cancel() {
        task.abort();
        let _cancelled = task.await;
        Err(Error::new(error_kind))
    } else {
        join_task(task).await
    }
}

pub(crate) struct RuntimeInner<B: Backend> {
    executor: Handle,
    config: RuntimeConfig,
    partition_cache: Arc<PartitionCache>,
    foreground: Arc<Semaphore>,
    foreground_waiting: Arc<Semaphore>,
    lifecycle: Mutex<Lifecycle<B>>,
    terminal: Notify,
}

impl<B: Backend> RuntimeInner<B> {
    /// Admits one foreground operation and runs it under operation control.
    ///
    /// The Runtime handle and every [`Index`] handle share this path: it
    /// enforces the foreground admission semaphores, the cancellation/deadline
    /// checks, and the commit-boundary race before the operation's closure
    /// starts.
    pub(crate) async fn run_foreground<T, F, Fut>(
        self: &Arc<Self>,
        options: OperationOptions,
        operation: F,
    ) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(OperationContext<B>) -> Fut + Send + 'static,
        Fut: Future<Output = Result<T>> + Send + 'static,
    {
        let (admission, backend) = self.admit(&options).await?;
        let cancellation = options.cancellation().cloned();
        let deadline = options.deadline();
        let (commit_cancellation, commit_start) = CommitCancellation::pair();
        let context = OperationContext {
            backend,
            options,
            commit_start: Some(commit_start),
        };
        let mut task = self.executor.spawn(async move {
            let result = match context.checkpoint() {
                Ok(()) => operation(context).await,
                Err(error) => {
                    drop(context);
                    Err(error)
                }
            };
            drop(admission);
            result
        });

        let mut caller = CancelBeforeCommit {
            task: Some(task.abort_handle()),
            commit_cancellation,
        };
        let result = if cancellation.is_none() && deadline.is_none() {
            join_task(&mut task).await
        } else {
            let cancellation = wait_for_cancellation(cancellation.as_ref());
            let deadline = wait_for_deadline(deadline);
            tokio::select! {
                biased;
                () = cancellation => {
                    cancel_task_before_commit(
                        &mut task,
                        &caller.commit_cancellation,
                        ErrorKind::Cancelled,
                    ).await
                }
                () = deadline => {
                    cancel_task_before_commit(
                        &mut task,
                        &caller.commit_cancellation,
                        ErrorKind::DeadlineExceeded,
                    ).await
                }
                result = join_task(&mut task) => result,
            }
        };
        caller.disarm();
        result
    }

    async fn admit(self: &Arc<Self>, options: &OperationOptions) -> Result<(Admission<B>, Arc<B>)> {
        check_control(options)?;
        let acquired = match self.foreground.clone().try_acquire_owned() {
            Ok(permit) => Ok(permit),
            Err(TryAcquireError::Closed) => Err(Error::new(ErrorKind::RuntimeClosed)),
            Err(TryAcquireError::NoPermits) => {
                let waiting = match self.foreground_waiting.clone().try_acquire_owned() {
                    Ok(waiting) => waiting,
                    Err(TryAcquireError::NoPermits) => {
                        check_control(options)?;
                        return Err(Error::new(ErrorKind::LimitExceeded));
                    }
                    Err(TryAcquireError::Closed) => {
                        check_control(options)?;
                        return Err(Error::new(ErrorKind::RuntimeClosed));
                    }
                };
                let acquired = self.foreground.clone().acquire_owned();
                let cancellation = wait_for_cancellation(options.cancellation());
                let deadline = wait_for_deadline(options.deadline());
                let acquired = tokio::select! {
                    permit = acquired => permit
                        .map_err(|_closed| Error::new(ErrorKind::RuntimeClosed)),
                    () = cancellation => Err(Error::new(ErrorKind::Cancelled)),
                    () = deadline => Err(Error::new(ErrorKind::DeadlineExceeded)),
                };

                drop(waiting);
                acquired
            }
        };
        let permit = match acquired {
            Ok(permit) => permit,
            Err(error) => {
                check_control(options)?;
                if self.phase() != Phase::Accepting {
                    return Err(Error::new(ErrorKind::RuntimeClosed));
                }
                return Err(error);
            }
        };

        let backend = {
            let mut lifecycle = self.lock_lifecycle();
            check_control(options)?;
            if lifecycle.phase != Phase::Accepting {
                return Err(Error::new(ErrorKind::RuntimeClosed));
            }
            let Some(backend) = lifecycle.backend.as_ref().map(Arc::clone) else {
                return Err(Error::new(ErrorKind::RuntimeClosed));
            };
            lifecycle.active += 1;
            backend
        };

        Ok((
            Admission {
                inner: Arc::clone(self),
                _permit: permit,
            },
            backend,
        ))
    }

    fn begin_shutdown(self: &Arc<Self>) {
        let (released_backend, started_closing) = {
            let mut lifecycle = self.lock_lifecycle();
            let started_closing = lifecycle.phase == Phase::Accepting;
            if started_closing {
                lifecycle.phase = Phase::Closing;
            }
            (lifecycle.start_release_if_drained(), started_closing)
        };
        if started_closing {
            self.foreground.close();
        }
        if let Some(backend) = released_backend {
            self.release_backend(backend);
        }
    }

    fn finish_operation(self: &Arc<Self>) {
        let released_backend = {
            let mut lifecycle = self.lock_lifecycle();
            debug_assert!(lifecycle.active > 0);
            if lifecycle.active > 0 {
                lifecycle.active -= 1;
            }
            lifecycle.start_release_if_drained()
        };
        if let Some(backend) = released_backend {
            self.release_backend(backend);
        }
    }

    fn release_backend(self: &Arc<Self>, backend: Arc<B>) {
        let inner = Arc::clone(self);
        let executor = self.executor.clone();
        drop(self.executor.spawn(async move {
            backend.shutdown().await;
            let _ = executor.spawn_blocking(move || drop(backend)).await;
            inner.finish_release();
        }));
    }

    fn finish_release(&self) {
        {
            let mut lifecycle = self.lock_lifecycle();
            debug_assert_eq!(lifecycle.phase, Phase::Releasing);
            lifecycle.phase = Phase::Closed;
        }
        self.terminal.notify_waiters();
    }

    fn phase(&self) -> Phase {
        self.lock_lifecycle().phase
    }

    /// Returns whether the Runtime still admits new foreground and import work.
    pub(crate) fn is_accepting(&self) -> bool {
        self.phase() == Phase::Accepting
    }

    /// Returns the validated process-local configuration.
    pub(crate) fn config(&self) -> &RuntimeConfig {
        &self.config
    }

    /// Returns a shared handle to the process-wide snapshot-validated
    /// Partition Cache.
    pub(crate) fn partition_cache(&self) -> Arc<PartitionCache> {
        Arc::clone(&self.partition_cache)
    }

    fn lock_lifecycle(&self) -> MutexGuard<'_, Lifecycle<B>> {
        match self.lifecycle.lock() {
            Ok(lifecycle) => lifecycle,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

struct Lifecycle<B: Backend> {
    phase: Phase,
    active: usize,
    backend: Option<Arc<B>>,
}

impl<B: Backend> Lifecycle<B> {
    fn start_release_if_drained(&mut self) -> Option<Arc<B>> {
        if self.phase == Phase::Closing && self.active == 0 {
            self.phase = Phase::Releasing;
            let backend = self.backend.take();
            debug_assert!(backend.is_some());
            backend
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Phase {
    Accepting,
    Closing,
    Releasing,
    Closed,
}

struct Admission<B: Backend> {
    inner: Arc<RuntimeInner<B>>,
    _permit: OwnedSemaphorePermit,
}

impl<B: Backend> Drop for Admission<B> {
    fn drop(&mut self) {
        self.inner.finish_operation();
    }
}

fn check_control(options: &OperationOptions) -> Result<()> {
    if options
        .cancellation()
        .is_some_and(CancellationToken::is_cancelled)
    {
        return Err(Error::new(ErrorKind::Cancelled));
    }
    if options
        .deadline()
        .is_some_and(|deadline| deadline <= Instant::now())
    {
        return Err(Error::new(ErrorKind::DeadlineExceeded));
    }
    Ok(())
}

async fn wait_for_cancellation(cancellation: Option<&CancellationToken>) {
    match cancellation {
        Some(cancellation) => cancellation.cancelled().await,
        None => pending().await,
    }
}

async fn wait_for_deadline(deadline: Option<Instant>) {
    match deadline {
        Some(deadline) => tokio::time::sleep_until(deadline.into()).await,
        None => pending().await,
    }
}

#[cfg(test)]
mod tests {
    use std::future::{Ready, ready};
    use std::sync::Condvar;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    use bytes::Bytes;
    use tokio::sync::Notify;

    use super::*;
    use crate::storage::backend::{
        AdmissionBudget, Capabilities, HardLimits, InsertOutcome, Mutation, ReadOps, ReadTxn,
        ScanLimits, ScanPage, WriteTxn,
    };
    use crate::storage::keys::KeyRange;

    struct TestBackend {
        drops: Arc<AtomicUsize>,
        shutdowns: Arc<AtomicUsize>,
        drop_control: Option<Arc<DropControl>>,
        debug_sentinel: &'static str,
    }

    struct DropControl {
        started: AtomicBool,
        release: Mutex<bool>,
        released: Condvar,
    }

    impl fmt::Debug for TestBackend {
        fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
            formatter
                .debug_struct("TestBackend")
                .field("debug_sentinel", &self.debug_sentinel)
                .finish()
        }
    }

    impl Drop for TestBackend {
        fn drop(&mut self) {
            if let Some(control) = &self.drop_control {
                control.started.store(true, Ordering::SeqCst);
                let mut release = match control.release.lock() {
                    Ok(release) => release,
                    Err(poisoned) => poisoned.into_inner(),
                };
                while !*release {
                    release = match control.released.wait(release) {
                        Ok(release) => release,
                        Err(poisoned) => poisoned.into_inner(),
                    };
                }
            }
            self.drops.fetch_add(1, Ordering::SeqCst);
        }
    }

    struct TestTxn;

    fn backend_error<T>() -> Ready<Result<T>> {
        ready(Err(Error::new(ErrorKind::Backend)))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn commit_boundary_has_exactly_one_winner() {
        for _ in 0..100 {
            let (cancellation, start) = CommitCancellation::pair();
            let ready = Arc::new(tokio::sync::Barrier::new(3));
            let commit = tokio::spawn({
                let ready = Arc::clone(&ready);
                async move {
                    ready.wait().await;
                    start.begin().is_ok()
                }
            });
            let cancel = tokio::spawn({
                let ready = Arc::clone(&ready);
                async move {
                    ready.wait().await;
                    cancellation.cancel()
                }
            });

            ready.wait().await;
            let commit_won = commit.await.expect("commit claimant did not panic");
            let cancel_won = cancel.await.expect("cancel claimant did not panic");
            assert_ne!(commit_won, cancel_won);
        }
    }

    impl ReadOps for TestTxn {
        fn get(&mut self, _key: Bytes) -> impl Future<Output = Result<Option<Bytes>>> + Send {
            backend_error()
        }

        fn batch_get(
            &mut self,
            _keys: Vec<Bytes>,
        ) -> impl Future<Output = Result<Vec<Option<Bytes>>>> + Send {
            backend_error()
        }

        fn scan(
            &mut self,
            _range: &KeyRange,
            _limits: ScanLimits,
        ) -> impl Future<Output = Result<ScanPage>> + Send {
            backend_error()
        }
    }

    impl ReadTxn for TestTxn {}

    impl WriteTxn for TestTxn {
        fn get_for_update(
            &mut self,
            _key: Bytes,
        ) -> impl Future<Output = Result<Option<Bytes>>> + Send {
            backend_error()
        }

        fn batch_get_for_update(
            &mut self,
            _keys: Vec<Bytes>,
        ) -> impl Future<Output = Result<Vec<Option<Bytes>>>> + Send {
            backend_error()
        }

        fn put(&mut self, _key: Bytes, _value: Bytes) -> impl Future<Output = Result<()>> + Send {
            backend_error()
        }

        fn insert(
            &mut self,
            _key: Bytes,
            _value: Bytes,
        ) -> impl Future<Output = Result<InsertOutcome>> + Send {
            backend_error()
        }

        fn delete(&mut self, _key: Bytes) -> impl Future<Output = Result<()>> + Send {
            backend_error()
        }

        fn batch_mutate(
            &mut self,
            _mutations: Vec<Mutation>,
        ) -> impl Future<Output = Result<()>> + Send {
            backend_error()
        }

        fn clear_range(&mut self, _range: &KeyRange) -> impl Future<Output = Result<()>> + Send {
            backend_error()
        }

        async fn commit_with(self, start: CommitStart) -> Result<()> {
            start.begin()?;
            Err(Error::new(ErrorKind::Backend))
        }

        fn rollback(self) -> impl Future<Output = ()> + Send {
            ready(())
        }
    }

    impl Backend for TestBackend {
        type ReadTxn<'backend> = TestTxn;
        type WriteTxn<'backend> = TestTxn;

        fn hard_limits(&self) -> HardLimits {
            HardLimits {
                max_key_bytes: usize::MAX,
                max_value_bytes: usize::MAX,
            }
        }

        fn admission_budget(&self) -> AdmissionBudget {
            AdmissionBudget {
                max_mutations: usize::MAX,
                max_mutation_bytes: usize::MAX,
            }
        }

        fn capabilities(&self) -> Capabilities {
            Capabilities {
                transactional_clear_range: false,
            }
        }

        async fn shutdown(&self) {
            self.shutdowns.fetch_add(1, Ordering::SeqCst);
        }

        fn begin_read(&self) -> impl Future<Output = Result<Self::ReadTxn<'_>>> + Send + '_ {
            ready(Ok(TestTxn))
        }

        fn begin_write(&self) -> impl Future<Output = Result<Self::WriteTxn<'_>>> + Send + '_ {
            ready(Ok(TestTxn))
        }
    }

    fn test_runtime(limit: usize) -> (Runtime<TestBackend>, Arc<AtomicUsize>) {
        let drops = Arc::new(AtomicUsize::new(0));
        let shutdowns = Arc::new(AtomicUsize::new(0));
        let config = RuntimeConfig::default()
            .with_foreground_operation_limit(limit)
            .expect("test limit is positive");
        let runtime = Runtime::new(
            TestBackend {
                drops: Arc::clone(&drops),
                shutdowns,
                drop_control: None,
                debug_sentinel: "sensitive-backend-sentinel",
            },
            config,
        )
        .expect("test runs on a multi-thread runtime");
        (runtime, drops)
    }

    #[tokio::test(flavor = "current_thread")]
    async fn construction_rejects_current_thread_runtime() {
        let drops = Arc::new(AtomicUsize::new(0));
        let shutdowns = Arc::new(AtomicUsize::new(0));
        let error = Runtime::new(
            TestBackend {
                drops: Arc::clone(&drops),
                shutdowns,
                drop_control: None,
                debug_sentinel: "sensitive-backend-sentinel",
            },
            RuntimeConfig::default(),
        )
        .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::InvalidArgument);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn saturation_waits_without_starting_extra_work() {
        let (runtime, _drops) = test_runtime(1);
        let first_started = Arc::new(Notify::new());
        let release_first = Arc::new(Notify::new());
        let second_started = Arc::new(AtomicBool::new(false));

        let first = tokio::spawn({
            let runtime = runtime.clone();
            let first_started = Arc::clone(&first_started);
            let release_first = Arc::clone(&release_first);
            async move {
                runtime
                    .run_foreground(OperationOptions::default(), move |context| async move {
                        let _backend = context.backend();
                        first_started.notify_one();
                        release_first.notified().await;
                        context.checkpoint()?;
                        Ok(1_u8)
                    })
                    .await
            }
        });
        first_started.notified().await;

        let second = tokio::spawn({
            let runtime = runtime.clone();
            let second_started = Arc::clone(&second_started);
            async move {
                runtime
                    .run_foreground(OperationOptions::default(), move |_context| async move {
                        second_started.store(true, Ordering::SeqCst);
                        Ok(2_u8)
                    })
                    .await
            }
        });
        while runtime.handle.inner.foreground_waiting.available_permits() != 0 {
            tokio::task::yield_now().await;
        }
        assert!(!second_started.load(Ordering::SeqCst));

        let saturation_error = runtime
            .run_foreground(OperationOptions::default(), |_context| async { Ok(()) })
            .await
            .unwrap_err();
        assert_eq!(saturation_error.kind(), ErrorKind::LimitExceeded);

        release_first.notify_one();
        assert_eq!(
            first
                .await
                .expect("caller task did not panic")
                .expect("first operation succeeds"),
            1
        );
        assert_eq!(
            second
                .await
                .expect("caller task did not panic")
                .expect("second operation succeeds"),
            2
        );
        runtime.shutdown().await.expect("shutdown succeeds");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancellation_while_saturated_never_starts_operation() {
        let (runtime, _drops) = test_runtime(1);
        let first_started = Arc::new(Notify::new());
        let release_first = Arc::new(Notify::new());
        let second_started = Arc::new(AtomicBool::new(false));

        let first = tokio::spawn({
            let runtime = runtime.clone();
            let first_started = Arc::clone(&first_started);
            let release_first = Arc::clone(&release_first);
            async move {
                runtime
                    .run_foreground(OperationOptions::default(), move |_context| async move {
                        first_started.notify_one();
                        release_first.notified().await;
                        Ok(())
                    })
                    .await
            }
        });
        first_started.notified().await;

        let cancellation = CancellationToken::new();
        let second = tokio::spawn({
            let runtime = runtime.clone();
            let second_started = Arc::clone(&second_started);
            let options = OperationOptions::default().with_cancellation(cancellation.clone());
            async move {
                runtime
                    .run_foreground(options, move |_context| async move {
                        second_started.store(true, Ordering::SeqCst);
                        Ok(())
                    })
                    .await
            }
        });
        while runtime.handle.inner.foreground_waiting.available_permits() != 0 {
            tokio::task::yield_now().await;
        }
        cancellation.cancel();

        let error = second
            .await
            .expect("caller task did not panic")
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Cancelled);
        assert!(!second_started.load(Ordering::SeqCst));
        assert_eq!(
            runtime.handle.inner.foreground_waiting.available_permits(),
            1
        );
        release_first.notify_one();
        first
            .await
            .expect("caller task did not panic")
            .expect("first operation succeeds");
        runtime.shutdown().await.expect("shutdown succeeds");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn expired_deadline_and_cancelled_token_have_stable_priority() {
        let (runtime, _drops) = test_runtime(1);
        let deadline_error = runtime
            .run_foreground(
                OperationOptions::default().with_deadline(Instant::now()),
                |_context| async { Ok(()) },
            )
            .await
            .unwrap_err();
        assert_eq!(deadline_error.kind(), ErrorKind::DeadlineExceeded);

        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let options = OperationOptions::default()
            .with_deadline(Instant::now())
            .with_cancellation(cancellation);

        let error = runtime
            .run_foreground(options, |_context| async { Ok(()) })
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Cancelled);
        runtime.shutdown().await.expect("shutdown succeeds");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropped_caller_cancels_before_commit_boundary() {
        let (runtime, _drops) = test_runtime(1);
        let operation_started = Arc::new(Notify::new());
        let check_control = Arc::new(Notify::new());
        let commit_started = Arc::new(AtomicBool::new(false));

        let caller = tokio::spawn({
            let runtime = runtime.clone();
            let operation_started = Arc::clone(&operation_started);
            let check_control = Arc::clone(&check_control);
            let commit_started = Arc::clone(&commit_started);
            async move {
                runtime
                    .run_foreground(OperationOptions::default(), move |context| async move {
                        operation_started.notify_one();
                        check_control.notified().await;
                        context.checkpoint()?;
                        commit_started.store(true, Ordering::SeqCst);
                        Ok(())
                    })
                    .await
            }
        });
        operation_started.notified().await;
        caller.abort();
        caller.await.expect_err("caller task was aborted");
        check_control.notify_one();

        runtime
            .shutdown()
            .await
            .expect("shutdown drains cancellation");
        assert!(!commit_started.load(Ordering::SeqCst));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropped_caller_cancels_a_precommit_backend_wait() {
        let (runtime, _drops) = test_runtime(1);
        let waiting_for_backend = Arc::new(Notify::new());
        let backend_capacity = Arc::new(Notify::new());
        let backend_call_started = Arc::new(AtomicBool::new(false));

        let caller = tokio::spawn({
            let runtime = runtime.clone();
            let waiting_for_backend = Arc::clone(&waiting_for_backend);
            let backend_capacity = Arc::clone(&backend_capacity);
            let backend_call_started = Arc::clone(&backend_call_started);
            async move {
                runtime
                    .run_foreground(OperationOptions::default(), move |mut context| async move {
                        context
                            .commit(move |commit_start| async move {
                                waiting_for_backend.notify_one();
                                backend_capacity.notified().await;
                                commit_start.begin()?;
                                backend_call_started.store(true, Ordering::SeqCst);
                                Ok(())
                            })
                            .await
                    })
                    .await
            }
        });
        waiting_for_backend.notified().await;
        caller.abort();
        caller.await.expect_err("caller task was aborted");

        tokio::time::timeout(Duration::from_secs(1), runtime.shutdown())
            .await
            .expect("shutdown does not wait for abandoned precommit work")
            .expect("shutdown succeeds");
        assert!(!backend_call_started.load(Ordering::SeqCst));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancellation_before_commit_abandons_operation() {
        let (runtime, _drops) = test_runtime(1);
        let cancellation = CancellationToken::new();
        let started = Arc::new(Notify::new());
        let check_control = Arc::new(Notify::new());
        let options = OperationOptions::default().with_cancellation(cancellation.clone());

        let operation = tokio::spawn({
            let runtime = runtime.clone();
            let started = Arc::clone(&started);
            let check_control = Arc::clone(&check_control);
            async move {
                runtime
                    .run_foreground(options, move |context| async move {
                        started.notify_one();
                        check_control.notified().await;
                        context.checkpoint()?;
                        Ok(())
                    })
                    .await
            }
        });
        started.notified().await;
        cancellation.cancel();
        check_control.notify_one();

        let error = operation
            .await
            .expect("caller task did not panic")
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::Cancelled);
        runtime.shutdown().await.expect("shutdown succeeds");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancellation_interrupts_a_precommit_backend_wait() {
        let (runtime, _drops) = test_runtime(1);
        let cancellation = CancellationToken::new();
        let waiting_for_backend = Arc::new(Notify::new());
        let backend_capacity = Arc::new(Notify::new());
        let backend_call_started = Arc::new(AtomicBool::new(false));
        let options = OperationOptions::default().with_cancellation(cancellation.clone());

        let operation = tokio::spawn({
            let runtime = runtime.clone();
            let waiting_for_backend = Arc::clone(&waiting_for_backend);
            let backend_capacity = Arc::clone(&backend_capacity);
            let backend_call_started = Arc::clone(&backend_call_started);
            async move {
                runtime
                    .run_foreground(options, move |mut context| async move {
                        context
                            .commit(move |commit_start| async move {
                                waiting_for_backend.notify_one();
                                backend_capacity.notified().await;
                                commit_start.begin()?;
                                backend_call_started.store(true, Ordering::SeqCst);
                                Ok(())
                            })
                            .await
                    })
                    .await
            }
        });
        waiting_for_backend.notified().await;
        cancellation.cancel();

        let error = tokio::time::timeout(Duration::from_secs(1), operation)
            .await
            .expect("cancelled precommit wait completes")
            .expect("caller task did not panic")
            .expect_err("operation is cancelled");
        assert_eq!(error.kind(), ErrorKind::Cancelled);
        assert!(!backend_call_started.load(Ordering::SeqCst));
        runtime.shutdown().await.expect("shutdown succeeds");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn deadline_interrupts_a_precommit_backend_wait() {
        let (runtime, _drops) = test_runtime(1);
        let waiting_for_backend = Arc::new(Notify::new());
        let backend_capacity = Arc::new(Notify::new());
        let backend_call_started = Arc::new(AtomicBool::new(false));
        let options =
            OperationOptions::default().with_deadline(Instant::now() + Duration::from_millis(200));

        let operation = tokio::spawn({
            let runtime = runtime.clone();
            let waiting_for_backend = Arc::clone(&waiting_for_backend);
            let backend_capacity = Arc::clone(&backend_capacity);
            let backend_call_started = Arc::clone(&backend_call_started);
            async move {
                runtime
                    .run_foreground(options, move |mut context| async move {
                        context
                            .commit(move |commit_start| async move {
                                waiting_for_backend.notify_one();
                                backend_capacity.notified().await;
                                commit_start.begin()?;
                                backend_call_started.store(true, Ordering::SeqCst);
                                Ok(())
                            })
                            .await
                    })
                    .await
            }
        });
        tokio::time::timeout(Duration::from_secs(1), waiting_for_backend.notified())
            .await
            .expect("operation reaches the backend wait before its deadline");

        let error = tokio::time::timeout(Duration::from_secs(1), operation)
            .await
            .expect("expired precommit wait completes")
            .expect("caller task did not panic")
            .expect_err("operation deadline expires");
        assert_eq!(error.kind(), ErrorKind::DeadlineExceeded);
        assert!(!backend_call_started.load(Ordering::SeqCst));
        runtime.shutdown().await.expect("shutdown succeeds");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn commit_result_wins_over_later_cancellation() {
        let (runtime, _drops) = test_runtime(1);
        let cancellation = CancellationToken::new();
        let commit_started = Arc::new(Notify::new());
        let finish_commit = Arc::new(Notify::new());
        let options = OperationOptions::default().with_cancellation(cancellation.clone());

        let operation = tokio::spawn({
            let runtime = runtime.clone();
            let commit_started = Arc::clone(&commit_started);
            let finish_commit = Arc::clone(&finish_commit);
            async move {
                runtime
                    .run_foreground(options, move |mut context| async move {
                        context
                            .commit(move |commit_start| async move {
                                commit_start.begin()?;
                                commit_started.notify_one();
                                finish_commit.notified().await;
                                Err::<(), _>(Error::new(ErrorKind::CommitOutcomeUnknown))
                            })
                            .await
                    })
                    .await
            }
        });
        commit_started.notified().await;
        cancellation.cancel();
        finish_commit.notify_one();

        let error = operation
            .await
            .expect("caller task did not panic")
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::CommitOutcomeUnknown);
        runtime.shutdown().await.expect("shutdown succeeds");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropped_caller_does_not_cancel_started_commit_or_shutdown_drain() {
        let (runtime, drops) = test_runtime(1);
        let commit_started = Arc::new(Notify::new());
        let finish_commit = Arc::new(Notify::new());
        let commit_finished = Arc::new(AtomicBool::new(false));

        let caller = tokio::spawn({
            let runtime = runtime.clone();
            let commit_started = Arc::clone(&commit_started);
            let finish_commit = Arc::clone(&finish_commit);
            let commit_finished = Arc::clone(&commit_finished);
            async move {
                runtime
                    .run_foreground(OperationOptions::default(), move |mut context| async move {
                        context
                            .commit(move |commit_start| async move {
                                commit_start.begin()?;
                                commit_started.notify_one();
                                finish_commit.notified().await;
                                commit_finished.store(true, Ordering::SeqCst);
                                Ok(())
                            })
                            .await
                    })
                    .await
            }
        });
        commit_started.notified().await;
        caller.abort();
        caller.await.expect_err("caller task was aborted");

        let shutdown = tokio::spawn({
            let runtime = runtime.clone();
            async move { runtime.shutdown().await }
        });
        while runtime.handle.inner.phase() == Phase::Accepting {
            tokio::task::yield_now().await;
        }
        assert!(!shutdown.is_finished());
        assert!(!commit_finished.load(Ordering::SeqCst));

        finish_commit.notify_one();
        shutdown
            .await
            .expect("shutdown task did not panic")
            .expect("shutdown succeeds");
        assert!(commit_finished.load(Ordering::SeqCst));
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn dropping_final_public_handle_finishes_owned_commit_cleanup() {
        let (runtime, drops) = test_runtime(1);
        let commit_started = Arc::new(Notify::new());
        let finish_commit = Arc::new(Notify::new());
        let commit_finished = Arc::new(Notify::new());

        let caller = tokio::spawn({
            let runtime = runtime.clone();
            let commit_started = Arc::clone(&commit_started);
            let finish_commit = Arc::clone(&finish_commit);
            let commit_finished = Arc::clone(&commit_finished);
            async move {
                runtime
                    .run_foreground(OperationOptions::default(), move |mut context| async move {
                        context
                            .commit(move |commit_start| async move {
                                commit_start.begin()?;
                                commit_started.notify_one();
                                finish_commit.notified().await;
                                commit_finished.notify_one();
                                Ok(())
                            })
                            .await
                    })
                    .await
            }
        });
        commit_started.notified().await;
        drop(runtime);
        caller.abort();
        caller.await.expect_err("caller task was aborted");

        finish_commit.notify_one();
        tokio::time::timeout(Duration::from_secs(1), commit_finished.notified())
            .await
            .expect("owned commit completes after final handle drop");
        tokio::time::timeout(Duration::from_secs(1), async {
            while drops.load(Ordering::SeqCst) == 0 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("terminal cleanup releases the backend");
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shutdown_rejects_new_work_and_preserves_admitted_result() {
        let (runtime, drops) = test_runtime(1);
        let operation_started = Arc::new(Notify::new());
        let finish_operation = Arc::new(Notify::new());
        let queued_started = Arc::new(AtomicBool::new(false));

        let admitted = tokio::spawn({
            let runtime = runtime.clone();
            let operation_started = Arc::clone(&operation_started);
            let finish_operation = Arc::clone(&finish_operation);
            async move {
                runtime
                    .run_foreground(OperationOptions::default(), move |_context| async move {
                        operation_started.notify_one();
                        finish_operation.notified().await;
                        Ok(7_u8)
                    })
                    .await
            }
        });
        operation_started.notified().await;

        let queued = tokio::spawn({
            let runtime = runtime.clone();
            let queued_started = Arc::clone(&queued_started);
            async move {
                runtime
                    .run_foreground(OperationOptions::default(), move |_context| async move {
                        queued_started.store(true, Ordering::SeqCst);
                        Ok(())
                    })
                    .await
            }
        });
        while runtime.handle.inner.foreground_waiting.available_permits() != 0 {
            tokio::task::yield_now().await;
        }

        let shutdown = tokio::spawn({
            let runtime = runtime.clone();
            async move { runtime.shutdown().await }
        });
        while runtime.handle.inner.phase() == Phase::Accepting {
            tokio::task::yield_now().await;
        }

        let queued_error = queued
            .await
            .expect("queued caller task did not panic")
            .unwrap_err();
        assert_eq!(queued_error.kind(), ErrorKind::RuntimeClosed);
        assert!(!queued_started.load(Ordering::SeqCst));

        let error = runtime
            .run_foreground(OperationOptions::default(), |_context| async { Ok(()) })
            .await
            .unwrap_err();
        assert_eq!(error.kind(), ErrorKind::RuntimeClosed);
        assert!(!shutdown.is_finished());

        finish_operation.notify_one();
        assert_eq!(
            admitted
                .await
                .expect("caller task did not panic")
                .expect("admitted operation succeeds"),
            7
        );
        shutdown
            .await
            .expect("shutdown task did not panic")
            .expect("shutdown succeeds");
        runtime.shutdown().await.expect("shutdown is idempotent");
        assert_eq!(runtime.handle.inner.phase(), Phase::Closed);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn shutdown_reaches_terminal_state_after_blocking_backend_drop() {
        let drops = Arc::new(AtomicUsize::new(0));
        let shutdowns = Arc::new(AtomicUsize::new(0));
        let drop_control = Arc::new(DropControl {
            started: AtomicBool::new(false),
            release: Mutex::new(false),
            released: Condvar::new(),
        });
        let runtime = Runtime::new(
            TestBackend {
                drops: Arc::clone(&drops),
                shutdowns: Arc::clone(&shutdowns),
                drop_control: Some(Arc::clone(&drop_control)),
                debug_sentinel: "sensitive-backend-sentinel",
            },
            RuntimeConfig::default(),
        )
        .expect("test runs on a multi-thread runtime");

        let operation_started = Arc::new(Notify::new());
        let finish_operation = Arc::new(Notify::new());
        let operation = tokio::spawn({
            let runtime = runtime.clone();
            let operation_started = Arc::clone(&operation_started);
            let finish_operation = Arc::clone(&finish_operation);
            async move {
                runtime
                    .run_foreground(OperationOptions::default(), move |_context| async move {
                        operation_started.notify_one();
                        finish_operation.notified().await;
                        Ok(())
                    })
                    .await
            }
        });
        operation_started.notified().await;

        let shutdown = tokio::spawn({
            let runtime = runtime.clone();
            async move { runtime.shutdown().await }
        });
        while runtime.handle.inner.phase() == Phase::Accepting {
            tokio::task::yield_now().await;
        }
        finish_operation.notify_one();
        operation
            .await
            .expect("operation task did not panic")
            .expect("admitted operation succeeds");
        tokio::time::timeout(Duration::from_secs(1), async {
            while !drop_control.started.load(Ordering::SeqCst) {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("backend release starts");
        assert_eq!(shutdowns.load(Ordering::SeqCst), 1);
        assert_eq!(runtime.handle.inner.phase(), Phase::Releasing);
        assert!(!shutdown.is_finished());

        {
            let mut release = match drop_control.release.lock() {
                Ok(release) => release,
                Err(poisoned) => poisoned.into_inner(),
            };
            *release = true;
            drop_control.released.notify_one();
        }
        shutdown
            .await
            .expect("shutdown task did not panic")
            .expect("shutdown succeeds");
        assert_eq!(runtime.handle.inner.phase(), Phase::Closed);
        assert_eq!(drops.load(Ordering::SeqCst), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn debug_output_excludes_backend_diagnostics() {
        let (runtime, _drops) = test_runtime(1);
        let debug = format!("{runtime:?}");
        assert!(!debug.contains("sensitive-backend-sentinel"));
        runtime.shutdown().await.expect("shutdown succeeds");
    }
}
