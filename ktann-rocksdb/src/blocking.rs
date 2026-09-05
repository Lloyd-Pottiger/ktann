use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use ktann::api::{Error, ErrorKind, Result};
use tokio::runtime::Handle;
use tokio::sync::{Notify, OwnedSemaphorePermit, Semaphore, mpsc, oneshot};

use crate::observe;

const COMMAND_CAPACITY: usize = 1;

/// Admits native transaction actors without exposing permits to the adapter interface.
#[derive(Debug)]
pub(crate) struct BlockingAdmission {
    state: Arc<AdmissionState>,
}

#[derive(Debug)]
struct AdmissionState {
    permits: Arc<Semaphore>,
    active: AtomicUsize,
    idle: Notify,
}

impl BlockingAdmission {
    pub(crate) fn new(limit: usize) -> Self {
        Self {
            state: Arc::new(AdmissionState {
                permits: Arc::new(Semaphore::new(limit)),
                active: AtomicUsize::new(0),
                idle: Notify::new(),
            }),
        }
    }

    /// Waits asynchronously, then starts one bounded native transaction actor.
    ///
    /// The actor owns its permit until its native state has been destroyed. Its
    /// one-command channel keeps cancelled callers from creating an unbounded
    /// queue while preserving serialized access to the native transaction.
    pub(crate) async fn start<C>(
        &self,
        actor: impl FnOnce(mpsc::Receiver<C>, oneshot::Sender<()>) + Send + 'static,
    ) -> Result<NativeWorker<C>>
    where
        C: Send + 'static,
    {
        Handle::try_current()
            .map_err(|source| Error::with_source(ErrorKind::InvalidArgument, source))?;
        let wait_started = Instant::now();
        let permit = Arc::clone(&self.state.permits)
            .acquire_owned()
            .await
            .map_err(|source| Error::with_source(ErrorKind::Backend, source))?;
        observe::blocking_wait(wait_started.elapsed());
        self.state.active.fetch_add(1, Ordering::AcqRel);
        let permit = ActivePermit {
            _permit: permit,
            state: Arc::clone(&self.state),
            admitted_at: Instant::now(),
        };
        let (commands, receiver) = mpsc::channel(COMMAND_CAPACITY);
        let (ready, opened) = oneshot::channel();

        std::thread::Builder::new()
            .name("ktann-rocksdb-actor".to_owned())
            .spawn(move || {
                let _permit = permit;
                actor(receiver, ready);
            })
            .map_err(|source| Error::with_source(ErrorKind::Backend, source))?;

        opened
            .await
            .map_err(|source| Error::with_source(ErrorKind::Backend, source))?;
        Ok(NativeWorker { commands })
    }

    /// Asynchronously waits until every admitted actor has finished cleanup.
    pub(crate) async fn wait_for_idle(&self) {
        while self.state.active.load(Ordering::Acquire) != 0 {
            self.state.idle.notified().await;
        }
    }
}

struct ActivePermit {
    _permit: OwnedSemaphorePermit,
    state: Arc<AdmissionState>,
    admitted_at: Instant,
}

impl Drop for ActivePermit {
    fn drop(&mut self) {
        observe::blocking_held(self.admitted_at.elapsed());
        if self.state.active.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.state.idle.notify_one();
        }
    }
}

/// A nonblocking handle to one admitted native transaction actor.
///
/// Dropping the handle only closes its bounded command channel. The actor then
/// destroys its native state on its existing native thread before
/// releasing admission.
pub(crate) struct NativeWorker<C> {
    commands: mpsc::Sender<C>,
}

impl<C> NativeWorker<C> {
    /// Sends one serialized command to the native actor.
    pub(crate) async fn send(&self, command: C) -> Result<()> {
        self.commands
            .send(command)
            .await
            .map_err(|_| Error::new(ErrorKind::Backend))
    }

    /// Sends one command and asynchronously awaits its response.
    pub(crate) async fn request<T>(
        &self,
        command: impl FnOnce(oneshot::Sender<Result<T>>) -> C,
    ) -> Result<T> {
        let (response, result) = oneshot::channel();
        self.send(command(response)).await?;
        result
            .await
            .map_err(|source| Error::with_source(ErrorKind::Backend, source))?
    }
}

#[cfg(test)]
mod tests {
    use std::future::{Future, pending, poll_fn};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Condvar, Mutex, MutexGuard, mpsc as std_mpsc};
    use std::task::Poll;
    use std::time::{Duration, Instant};

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
            self.open
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
        }
    }

    struct SlowDrop {
        started: Option<oneshot::Sender<()>>,
        gate: Arc<Gate>,
    }

    impl Drop for SlowDrop {
        fn drop(&mut self) {
            if let Some(started) = self.started.take() {
                let _ = started.send(());
            }
            self.gate.wait();
        }
    }

    async fn idle_worker(admission: &BlockingAdmission) -> Result<NativeWorker<()>> {
        admission
            .start(|mut commands, ready| {
                if ready.send(()).is_err() {
                    return;
                }
                while commands.blocking_recv().is_some() {}
            })
            .await
    }

    async fn value_worker(
        admission: &BlockingAdmission,
    ) -> Result<NativeWorker<oneshot::Sender<Result<usize>>>> {
        admission
            .start(
                |mut commands: mpsc::Receiver<oneshot::Sender<Result<usize>>>, ready| {
                    if ready.send(()).is_err() {
                        return;
                    }
                    while let Some(response) = commands.blocking_recv() {
                        let _ = response.send(Ok(7));
                    }
                },
            )
            .await
    }

    async fn slow_worker(
        admission: &BlockingAdmission,
        gate: Arc<Gate>,
    ) -> Result<(NativeWorker<()>, oneshot::Receiver<()>)> {
        let (cleanup_started, started) = oneshot::channel();
        let worker = admission
            .start(move |mut commands: mpsc::Receiver<()>, ready| {
                let _slow = SlowDrop {
                    started: Some(cleanup_started),
                    gate,
                };
                if ready.send(()).is_err() {
                    return;
                }
                while commands.blocking_recv().is_some() {}
            })
            .await?;
        Ok((worker, started))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn live_actor_uses_native_calls_while_next_admission_waits() {
        let admission = BlockingAdmission::new(1);
        let holder = value_worker(&admission).await.expect("holder starts");
        let mut waiter = Box::pin(idle_worker(&admission));
        poll_fn(|context| match waiter.as_mut().poll(context) {
            Poll::Pending => Poll::Ready(()),
            Poll::Ready(_) => panic!("a live actor released its resource slot"),
        })
        .await;
        assert_eq!(
            holder
                .request(|response| response)
                .await
                .expect("admitted actor remains operable"),
            7,
        );

        drop(holder);
        let successor = tokio::time::timeout(Duration::from_secs(1), waiter)
            .await
            .expect("cleanup releases capacity")
            .expect("successor starts");
        drop(successor);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelling_before_admission_removes_the_waiter() {
        let admission = BlockingAdmission::new(1);
        let holder = idle_worker(&admission).await.expect("holder starts");
        let mut waiter = Box::pin(idle_worker(&admission));
        poll_fn(|context| match waiter.as_mut().poll(context) {
            Poll::Pending => Poll::Ready(()),
            Poll::Ready(_) => panic!("waiter acquired a held permit"),
        })
        .await;
        drop(waiter);
        drop(holder);

        let successor = tokio::time::timeout(Duration::from_secs(1), idle_worker(&admission))
            .await
            .expect("cancelled admission does not leak capacity")
            .expect("successor starts");
        drop(successor);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn current_thread_runtime_can_use_native_actor() {
        let admission = BlockingAdmission::new(1);
        let worker = idle_worker(&admission)
            .await
            .expect("current-thread runtime starts blocking actor");
        drop(worker);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn local_set_drop_does_not_block_unrelated_tasks_during_slow_cleanup() {
        let admission = BlockingAdmission::new(1);
        let gate = Arc::new(Gate::default());
        let (worker, started) = slow_worker(&admission, Arc::clone(&gate))
            .await
            .expect("worker starts");

        let (scheduled, ran) = oneshot::channel();
        tokio::task::LocalSet::new()
            .run_until(async move {
                tokio::task::spawn_local(async move {
                    let _ = scheduled.send(());
                });
                drop(worker);
                started.await.expect("cleanup starts");
                tokio::time::timeout(Duration::from_secs(1), ran)
                    .await
                    .expect("unrelated LocalSet task remains schedulable")
                    .expect("unrelated LocalSet task runs");
            })
            .await;
        gate.open();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn orderly_shutdown_waits_for_native_cleanup_completion() {
        let admission = BlockingAdmission::new(1);
        let gate = Arc::new(Gate::default());
        let (worker, started) = slow_worker(&admission, Arc::clone(&gate))
            .await
            .expect("worker starts");
        drop(worker);
        started.await.expect("cleanup starts");

        let mut shutdown = Box::pin(admission.wait_for_idle());
        poll_fn(|context| match shutdown.as_mut().poll(context) {
            Poll::Pending => Poll::Ready(()),
            Poll::Ready(()) => panic!("shutdown completed before native cleanup"),
        })
        .await;
        gate.open();
        tokio::time::timeout(Duration::from_secs(1), shutdown)
            .await
            .expect("shutdown completes after native cleanup");
    }

    #[test]
    fn runtime_shutdown_does_not_wait_for_slow_native_cleanup() {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("runtime builds");
        let admission = BlockingAdmission::new(1);
        let gate = Arc::new(Gate::default());
        let (worker, mut started) = runtime
            .block_on(slow_worker(&admission, Arc::clone(&gate)))
            .expect("worker starts");
        runtime.spawn(async move {
            let _worker = worker;
            pending::<()>().await;
        });

        let (shutdown_finished, finished) = std_mpsc::channel();
        std::thread::spawn(move || {
            drop(runtime);
            let _ = shutdown_finished.send(());
        });
        let cleanup_deadline = Instant::now() + Duration::from_secs(1);
        loop {
            match started.try_recv() {
                Ok(()) => break,
                Err(oneshot::error::TryRecvError::Empty) if Instant::now() < cleanup_deadline => {
                    std::thread::yield_now();
                }
                Err(error) => panic!("runtime shutdown did not start native cleanup: {error}"),
            }
        }
        finished
            .recv_timeout(Duration::from_secs(1))
            .expect("runtime shutdown does not wait for slow cleanup");
        gate.open();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cleanup_panic_releases_admission() {
        let admission = BlockingAdmission::new(1);
        let panics = Arc::new(AtomicUsize::new(0));
        let panic_count = Arc::clone(&panics);
        let worker = admission
            .start(move |mut commands: mpsc::Receiver<()>, ready| {
                struct PanicOnDrop(Arc<AtomicUsize>);

                impl Drop for PanicOnDrop {
                    fn drop(&mut self) {
                        self.0.fetch_add(1, Ordering::SeqCst);
                        panic!("cleanup probe panic");
                    }
                }

                let _panic = PanicOnDrop(panic_count);
                if ready.send(()).is_err() {
                    return;
                }
                while commands.blocking_recv().is_some() {}
            })
            .await
            .expect("worker starts");
        drop(worker);

        let successor = tokio::time::timeout(Duration::from_secs(1), idle_worker(&admission))
            .await
            .expect("panic releases capacity")
            .expect("successor starts");
        assert_eq!(panics.load(Ordering::SeqCst), 1);
        drop(successor);
    }
}
