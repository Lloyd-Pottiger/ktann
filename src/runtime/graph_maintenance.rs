//! Runtime-owned admission for private topology graph preparation.

use std::sync::Arc;

use tokio::sync::Semaphore;

use super::{CancellationToken, check_control, compute::controlled_compute};
use crate::api::{Error, ErrorKind, OperationOptions, Result, RuntimeConfig};

/// Bounds concurrent native preparations and retains admission until CPU exit,
/// including when cancellation drops the awaiting maintenance future.
pub struct GraphMaintenance {
    working_bytes: usize,
    slots: Arc<Semaphore>,
    cancellation: CancellationToken,
}

impl GraphMaintenance {
    /// Creates one reusable admission owner for explicit maintenance. Automatic
    /// maintenance shares one owner per Runtime. With automatic workers disabled,
    /// explicit calls still have one preparation slot.
    pub fn new(config: &RuntimeConfig, cancellation: CancellationToken) -> Self {
        Self {
            working_bytes: config.routing_working_bytes(),
            slots: Arc::new(Semaphore::new(config.maintenance_workers().max(1))),
            cancellation,
        }
    }

    pub(crate) fn working_bytes(&self) -> usize {
        self.working_bytes
    }

    pub(crate) async fn prepare<T: Send + 'static>(
        &self,
        work: impl FnOnce(usize, &dyn Fn() -> Result<()>) -> Result<T> + Send + 'static,
    ) -> Result<T> {
        let permit = tokio::select! {
            biased;
            () = self.cancellation.cancelled() => return Err(Error::new(ErrorKind::Cancelled)),
            permit = Arc::clone(&self.slots).acquire_owned() => {
                permit.map_err(|_| Error::new(ErrorKind::Cancelled))?
            }
        };
        let working_bytes = self.working_bytes;
        let options = OperationOptions::default().with_cancellation(self.cancellation.clone());
        controlled_compute(options, permit, move |options| {
            work(working_bytes, &|| check_control(options))
        })
        .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_waiter_retains_native_admission_until_cpu_exit() {
        let config = RuntimeConfig::default().with_maintenance(0, 1).unwrap();
        let executor = Arc::new(GraphMaintenance::new(&config, CancellationToken::new()));
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let first = tokio::spawn({
            let executor = Arc::clone(&executor);
            async move {
                executor
                    .prepare(move |_, check| {
                        entered_tx.send(()).unwrap();
                        release_rx.recv().unwrap();
                        check()
                    })
                    .await
            }
        });
        entered_rx.await.unwrap();
        first.abort();
        assert!(first.await.unwrap_err().is_cancelled());
        assert_eq!(executor.slots.available_permits(), 0);
        release_tx.send(()).unwrap();
        // Acquiring the only permit proves the detached native task has exited.
        executor.prepare(|_, check| check()).await.unwrap();
        assert_eq!(executor.slots.available_permits(), 1);
    }
}
