//! Runtime-owned admission for private topology graph preparation.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

use tokio::sync::{Mutex as AsyncMutex, OwnedMutexGuard, Semaphore};

use super::{CancellationToken, check_control, compute::controlled_compute};
use crate::api::{
    Error, ErrorKind, LogicalIndexId, OperationOptions, PartitionKey, Result, RuntimeConfig,
};
use crate::storage::keys::TreeKey;

/// Bounds concurrent native preparations and retains admission until CPU exit,
/// including when cancellation drops the awaiting maintenance future.
pub struct GraphMaintenance {
    working_bytes: usize,
    slots: Arc<Semaphore>,
    cancellation: CancellationToken,
    bodies: Mutex<HashMap<BodyKey, Weak<AsyncMutex<()>>>>,
}

/// Process-local graph identity; backend source fencing remains authoritative.
#[derive(Clone, Hash, PartialEq, Eq)]
struct BodyKey(LogicalIndexId, TreeKey, PartitionKey);

/// Held through native preparation and atomic publication, including cancellation
/// of a waiter while its detached native computation is still running.
pub(crate) struct GraphBodyGuards {
    _guards: Vec<OwnedMutexGuard<()>>,
    pub(crate) waited: bool,
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
            bodies: Mutex::new(HashMap::new()),
        }
    }

    /// Coordinates only overlapping graph bodies. Sorted acquisition prevents
    /// multi-body drains from deadlocking; weak entries bound the registry to
    /// active participants plus the last acquisition's keys.
    pub(crate) async fn lock_bodies(
        &self,
        index: LogicalIndexId,
        tree: &TreeKey,
        mut partitions: Vec<PartitionKey>,
    ) -> Result<GraphBodyGuards> {
        if self.cancellation.is_cancelled() {
            return Err(Error::new(ErrorKind::Cancelled));
        }
        partitions.sort_unstable();
        partitions.dedup();
        let locks = {
            let mut bodies = self
                .bodies
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            bodies.retain(|_, lock| lock.strong_count() > 0);
            partitions
                .into_iter()
                .map(|partition| {
                    let key = BodyKey(index, tree.clone(), partition);
                    if let Some(lock) = bodies.get(&key).and_then(Weak::upgrade) {
                        lock
                    } else {
                        let lock = Arc::new(AsyncMutex::new(()));
                        bodies.insert(key, Arc::downgrade(&lock));
                        lock
                    }
                })
                .collect::<Vec<_>>()
        };
        let mut guards = GraphBodyGuards {
            _guards: Vec::with_capacity(locks.len()),
            waited: false,
        };
        for lock in locks {
            let guard = match Arc::clone(&lock).try_lock_owned() {
                Ok(guard) => guard,
                Err(_) => {
                    guards.waited = true;
                    tokio::select! {
                        biased;
                        () = self.cancellation.cancelled() => return Err(Error::new(ErrorKind::Cancelled)),
                        guard = lock.lock_owned() => guard,
                    }
                }
            };
            guards._guards.push(guard);
        }
        Ok(guards)
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
    use std::time::Duration;

    #[tokio::test]
    async fn body_coordination_serializes_overlap_but_not_independent_graphs() {
        let config = RuntimeConfig::default();
        let owner = GraphMaintenance::new(&config, CancellationToken::new());
        let index = LogicalIndexId::new(1).unwrap();
        let tree = TreeKey::encode(&[], &[]).unwrap();
        let one = PartitionKey::new(1).unwrap();
        let two = PartitionKey::new(2).unwrap();
        let first = owner
            .lock_bodies(index, &tree, vec![two, one])
            .await
            .unwrap();
        let _other = owner
            .lock_bodies(index, &tree, vec![PartitionKey::new(3).unwrap()])
            .await
            .unwrap();
        let _other_index = owner
            .lock_bodies(LogicalIndexId::new(2).unwrap(), &tree, vec![one])
            .await
            .unwrap();
        let waiting = owner.lock_bodies(index, &tree, vec![one, two, one]);
        tokio::pin!(waiting);
        assert!(
            tokio::time::timeout(Duration::from_millis(10), &mut waiting)
                .await
                .is_err()
        );
        drop(first);
        let _next = tokio::time::timeout(Duration::from_secs(1), waiting)
            .await
            .unwrap()
            .unwrap();
    }

    #[tokio::test]
    async fn cancelled_body_wait_releases_acquired_locks_and_registry_is_bounded() {
        let config = RuntimeConfig::default();
        let cancellation = CancellationToken::new();
        let owner = GraphMaintenance::new(&config, cancellation.clone());
        let index = LogicalIndexId::new(1).unwrap();
        let tree = TreeKey::encode(&[], &[]).unwrap();
        let one = PartitionKey::new(1).unwrap();
        let two = PartitionKey::new(2).unwrap();
        let held = owner.lock_bodies(index, &tree, vec![two]).await.unwrap();
        let waiting = owner.lock_bodies(index, &tree, vec![one, two]);
        tokio::pin!(waiting);
        assert!(
            tokio::time::timeout(Duration::from_millis(10), &mut waiting)
                .await
                .is_err()
        );
        cancellation.cancel();
        assert!(matches!(waiting.await, Err(error) if error.kind() == ErrorKind::Cancelled));
        drop(held);
        assert!(
            owner
                .bodies
                .lock()
                .unwrap()
                .values()
                .all(|lock| lock.strong_count() == 0)
        );
        let owner = GraphMaintenance::new(&config, CancellationToken::new());
        for ordinal in 3..100 {
            drop(
                owner
                    .lock_bodies(index, &tree, vec![PartitionKey::new(ordinal).unwrap()])
                    .await
                    .unwrap(),
            );
            assert_eq!(owner.bodies.lock().unwrap().len(), 1);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_waiter_retains_native_admission_until_cpu_exit() {
        let config = RuntimeConfig::default().with_maintenance(0, 1).unwrap();
        let executor = Arc::new(GraphMaintenance::new(&config, CancellationToken::new()));
        let index = LogicalIndexId::new(1).unwrap();
        let tree = TreeKey::encode(&[], &[]).unwrap();
        let partition = PartitionKey::new(1).unwrap();
        let body = executor
            .lock_bodies(index, &tree, vec![partition])
            .await
            .unwrap();
        let (entered_tx, entered_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let first = tokio::spawn({
            let executor = Arc::clone(&executor);
            async move {
                executor
                    .prepare(move |_, check| {
                        entered_tx.send(()).unwrap();
                        release_rx.recv().unwrap();
                        check()?;
                        Ok(body)
                    })
                    .await
            }
        });
        entered_rx.await.unwrap();
        first.abort();
        assert!(matches!(first.await, Err(error) if error.is_cancelled()));
        assert_eq!(executor.slots.available_permits(), 0);
        let waiting = executor.lock_bodies(index, &tree, vec![partition]);
        tokio::pin!(waiting);
        assert!(
            tokio::time::timeout(Duration::from_millis(10), &mut waiting)
                .await
                .is_err()
        );
        release_tx.send(()).unwrap();
        // Acquiring the only permit proves the detached native task has exited.
        executor.prepare(|_, check| check()).await.unwrap();
        assert_eq!(executor.slots.available_permits(), 1);
        let _guard = waiting.await.unwrap();
    }
}
