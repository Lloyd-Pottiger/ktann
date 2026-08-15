use std::any::Any;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};

use ktann::api::{Error, ErrorKind, Result};
use tokio::runtime::{Handle, RuntimeFlavor};
use tokio::sync::{Semaphore, SemaphorePermit};

/// Admits native RocksDB resources without exposing permits to the adapter interface.
#[derive(Debug)]
pub(crate) struct BlockingAdmission {
    permits: Semaphore,
}

impl BlockingAdmission {
    pub(crate) fn new(limit: usize) -> Self {
        Self {
            permits: Semaphore::new(limit),
        }
    }

    /// Waits asynchronously before opening a native RocksDB resource.
    ///
    /// Admission and execution are deliberately separate so callers create a
    /// closure borrowing a non-`Sync` transaction only after this await.
    pub(crate) async fn admit(&self) -> Result<BlockingSection<'_>> {
        let runtime = Handle::try_current()
            .map_err(|source| Error::with_source(ErrorKind::InvalidArgument, source))?;
        if runtime.runtime_flavor() != RuntimeFlavor::MultiThread {
            return Err(Error::new(ErrorKind::InvalidArgument));
        }
        let permit = self
            .permits
            .acquire()
            .await
            .map_err(|source| Error::with_source(ErrorKind::Backend, source))?;
        Ok(BlockingSection { permit })
    }
}

/// One admitted synchronous call that has not entered its blocking section yet.
pub(crate) struct BlockingSection<'admission> {
    permit: SemaphorePermit<'admission>,
}

impl<'admission> BlockingSection<'admission> {
    /// Runs one native call synchronously and releases admission on return.
    pub(crate) fn run<T>(self, operation: impl FnOnce() -> Result<T>) -> Result<T> {
        run_supported(operation)
    }

    /// Opens a native handle whose eventual destructor reuses this admission.
    pub(crate) fn open<T: Send>(
        self,
        operation: impl FnOnce() -> T,
    ) -> Result<AdmittedHandle<'admission, T>> {
        let value = run_supported(|| Ok(operation()))?;
        Ok(AdmittedHandle {
            admitted: Some((value, self.permit)),
        })
    }
}

/// Owns one native handle whose destructor must use blocking admission.
pub(crate) struct AdmittedHandle<'admission, T: Send> {
    admitted: Option<(T, SemaphorePermit<'admission>)>,
}

impl<'admission, T: Send> AdmittedHandle<'admission, T> {
    /// Verifies that this task context permits `block_in_place`.
    pub(crate) fn ensure_supported(&self) -> Result<()> {
        run_supported(|| Ok(()))
    }

    /// Runs one native call while retaining this handle's reserved admission.
    pub(crate) fn run<R>(&self, operation: impl FnOnce(&T) -> Result<R>) -> Result<R> {
        let Some((value, _permit)) = self.admitted.as_ref() else {
            unreachable!("an admitted handle cannot be accessed after transfer");
        };
        run_supported(|| operation(value))
    }

    /// Transfers the native handle and its reserved terminal section.
    pub(crate) fn into_section(mut self) -> (T, BlockingSection<'admission>) {
        let Some((value, permit)) = self.admitted.take() else {
            unreachable!("an admitted handle can transfer ownership only once");
        };
        (value, BlockingSection { permit })
    }
}

impl<T: Send> Drop for AdmittedHandle<'_, T> {
    fn drop(&mut self) {
        if let Some((value, _permit)) = self.admitted.take() {
            let cleanup = || drop(value);
            match try_block_in_place(cleanup) {
                Ok(()) => {}
                Err(BlockInPlaceError::Unsupported(cleanup)) => {
                    std::thread::scope(|scope| {
                        if let Err(panic) = scope.spawn(cleanup).join() {
                            resume_unwind(panic);
                        }
                    });
                }
                Err(BlockInPlaceError::Panicked(panic)) => resume_unwind(panic),
            }
        }
    }
}

fn run_supported<T>(operation: impl FnOnce() -> Result<T>) -> Result<T> {
    match try_block_in_place(operation) {
        Ok(result) => result,
        Err(BlockInPlaceError::Unsupported(_)) => Err(Error::new(ErrorKind::InvalidArgument)),
        Err(BlockInPlaceError::Panicked(panic)) => resume_unwind(panic),
    }
}

enum BlockInPlaceError<F> {
    Unsupported(F),
    Panicked(Box<dyn Any + Send>),
}

/// Preserves an unstarted operation when Tokio rejects the current task context.
///
/// Tokio exposes no public LocalSet predicate. `block_in_place` panics before
/// invoking its closure there, so retaining the closure distinguishes that
/// context rejection from a panic produced by the native operation itself.
fn try_block_in_place<F, T>(operation: F) -> std::result::Result<T, BlockInPlaceError<F>>
where
    F: FnOnce() -> T,
{
    let mut operation = Some(operation);
    match catch_unwind(AssertUnwindSafe(|| {
        tokio::task::block_in_place(|| {
            let Some(operation) = operation.take() else {
                unreachable!("a blocking operation runs exactly once");
            };
            operation()
        })
    })) {
        Ok(result) => Ok(result),
        Err(panic) => match operation {
            Some(operation) => Err(BlockInPlaceError::Unsupported(operation)),
            None => Err(BlockInPlaceError::Panicked(panic)),
        },
    }
}

#[cfg(test)]
mod tests {
    use std::future::{Future, poll_fn};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Condvar, Mutex, MutexGuard};
    use std::task::Poll;
    use std::time::Duration;

    use tokio::sync::mpsc;

    use super::*;

    #[derive(Default)]
    struct Gate {
        open: Mutex<bool>,
        changed: Condvar,
    }

    impl Gate {
        fn wait(&self) {
            let mut open = self.lock();
            while !*open {
                open = match self.changed.wait(open) {
                    Ok(open) => open,
                    Err(poisoned) => poisoned.into_inner(),
                };
            }
        }

        fn open(&self) {
            *self.lock() = true;
            self.changed.notify_all();
        }

        fn lock(&self) -> MutexGuard<'_, bool> {
            match self.open.lock() {
                Ok(open) => open,
                Err(poisoned) => poisoned.into_inner(),
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn saturation_never_executes_more_than_the_configured_limit() {
        let admission = Arc::new(BlockingAdmission::new(2));
        let gate = Arc::new(Gate::default());
        let active = Arc::new(AtomicUsize::new(0));
        let maximum = Arc::new(AtomicUsize::new(0));
        let (started, mut starts) = mpsc::unbounded_channel();
        let mut tasks = Vec::new();

        for _ in 0..4 {
            let admission = Arc::clone(&admission);
            let gate = Arc::clone(&gate);
            let active = Arc::clone(&active);
            let maximum = Arc::clone(&maximum);
            let started = started.clone();
            tasks.push(tokio::spawn(async move {
                admission.admit().await?.run(|| {
                    let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                    maximum.fetch_max(now, Ordering::SeqCst);
                    started
                        .send(())
                        .map_err(|source| Error::with_source(ErrorKind::Backend, source))?;
                    gate.wait();
                    active.fetch_sub(1, Ordering::SeqCst);
                    Ok::<(), Error>(())
                })
            }));
        }
        drop(started);

        starts.recv().await.expect("first call starts");
        starts.recv().await.expect("second call starts");
        assert!(
            tokio::time::timeout(Duration::from_millis(50), starts.recv())
                .await
                .is_err(),
            "a third call entered while both permits were held",
        );
        assert_eq!(maximum.load(Ordering::SeqCst), 2);

        gate.open();
        for task in tasks {
            task.await
                .expect("blocking task did not panic")
                .expect("blocking call succeeds");
        }
        assert_eq!(maximum.load(Ordering::SeqCst), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelling_while_waiting_removes_the_waiter_without_leaking_capacity() {
        let admission = BlockingAdmission::new(1);
        let holder = admission.admit().await.expect("holder acquires permit");
        let mut waiter = Box::pin(admission.admit());
        poll_fn(|context| match waiter.as_mut().poll(context) {
            Poll::Pending => Poll::Ready(()),
            Poll::Ready(_) => panic!("waiter acquired a held permit"),
        })
        .await;
        drop(waiter);
        drop(holder);

        let value = admission
            .admit()
            .await
            .expect("capacity is reusable")
            .run(|| Ok::<_, Error>(7))
            .expect("successor call starts");
        assert_eq!(value, 7);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn native_handle_reserves_capacity_through_cleanup() {
        struct DropProbe(Arc<AtomicUsize>);

        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let admission = BlockingAdmission::new(1);
        let dropped = Arc::new(AtomicUsize::new(0));
        let handle = admission
            .admit()
            .await
            .expect("handle creation acquires admission")
            .open(|| DropProbe(Arc::clone(&dropped)))
            .expect("native handle opens");
        let mut waiter = Box::pin(admission.admit());
        poll_fn(|context| match waiter.as_mut().poll(context) {
            Poll::Pending => Poll::Ready(()),
            Poll::Ready(_) => panic!("a live native handle released its resource slot"),
        })
        .await;

        drop(handle);
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
        waiter
            .await
            .expect("cleanup releases capacity")
            .run(|| Ok::<(), Error>(()))
            .expect("successor call starts");
    }

    #[tokio::test(flavor = "current_thread")]
    async fn current_thread_runtime_is_rejected_before_admission() {
        let error = match BlockingAdmission::new(1).admit().await {
            Ok(_) => panic!("current-thread runtime is unsupported"),
            Err(error) => error,
        };
        assert_eq!(error.kind(), ErrorKind::InvalidArgument);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn local_set_calls_are_rejected_and_admitted_handles_still_drop() {
        struct DropProbe(Arc<AtomicUsize>);

        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }

        let admission = BlockingAdmission::new(1);
        let dropped = Arc::new(AtomicUsize::new(0));
        let handle = admission
            .admit()
            .await
            .expect("handle creation acquires admission")
            .open(|| DropProbe(Arc::clone(&dropped)))
            .expect("native handle opens");

        let error = tokio::task::LocalSet::new()
            .run_until(async {
                let error = handle.run(|_| Ok::<(), Error>(()));
                drop(handle);
                error.expect_err("LocalSet cannot enter block_in_place")
            })
            .await;
        assert_eq!(error.kind(), ErrorKind::InvalidArgument);
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
    }
}
