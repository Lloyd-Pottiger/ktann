//! Whole-attempt retries include backend admission and precommit reads.

use std::time::Duration;

use bytes::Bytes;
use ktann::api::{
    ErrorKind, GetOptions, IndexConfig, Metric, OperationOptions, PartitionKey, Record,
    RuntimeConfig,
};
use ktann::runtime::{RetryPolicy, Runtime};
use ktann::storage::keys::TreeKey;
use ktann_memory::MemoryBackend;
use ktann_memory::test_support::{CommitFault, TestConfig, WriteFaultPoint};
use tokio_util::sync::CancellationToken;

#[allow(dead_code)]
mod support;

fn backend() -> MemoryBackend {
    MemoryBackend::with_test_config(TestConfig {
        max_active_transactions: 1,
        ..TestConfig::default()
    })
}

fn arm(backend: &MemoryBackend, point: WriteFaultPoint, count: usize, kind: ErrorKind) {
    backend
        .set_write_fault_plan(point, vec![kind; count])
        .unwrap();
    backend.reset_operation_counts();
}

fn assert_attempts(backend: &MemoryBackend, expected: usize) {
    assert_eq!(backend.operation_counts().begin_write, expected);
    assert_eq!(
        backend.active_transactions(),
        0,
        "every attempt releases its snapshot"
    );
}

fn config() -> IndexConfig {
    IndexConfig::new(2, Metric::L2).unwrap()
}

fn runtime(backend: &MemoryBackend) -> Runtime<MemoryBackend> {
    Runtime::new(
        backend.clone(),
        RuntimeConfig::default()
            .with_maintenance(0, 1)
            .unwrap()
            .with_attempts(3, 3)
            .unwrap()
            .with_retry_backoff(Duration::from_nanos(1), Duration::from_nanos(1))
            .unwrap(),
    )
    .unwrap()
}

fn record() -> Record {
    Record::new(Bytes::from_static(b"r"), vec![1.0, 2.0], vec![]).unwrap()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn foreground_retries_open_and_manifest_aborts() {
    for stage in [WriteFaultPoint::BeginWrite, WriteFaultPoint::GetForUpdate] {
        let backend = backend();
        let runtime = runtime(&backend);
        let index = runtime.create_index("index", config()).await.unwrap();
        arm(&backend, stage, 1, ErrorKind::RetryableAbort);
        index.insert(record()).await.unwrap();
        assert_attempts(&backend, 2);
        assert!(
            index
                .get(Bytes::from_static(b"r"), GetOptions::default())
                .await
                .unwrap()
                .is_some()
        );
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn foreground_precommit_aborts_exhaust_and_other_errors_are_not_replayed() {
    for stage in [WriteFaultPoint::BeginWrite, WriteFaultPoint::GetForUpdate] {
        for (kind, expected, attempts) in [
            (ErrorKind::RetryableAbort, ErrorKind::ContentionExhausted, 3),
            (ErrorKind::Backend, ErrorKind::Backend, 1),
            (ErrorKind::Corruption, ErrorKind::Corruption, 1),
        ] {
            let backend = backend();
            let runtime = runtime(&backend);
            let index = runtime.create_index("index", config()).await.unwrap();
            arm(&backend, stage, 10, kind);
            assert_eq!(index.insert(record()).await.unwrap_err().kind(), expected);
            assert_attempts(&backend, attempts);
            assert!(
                index
                    .get(Bytes::from_static(b"r"), GetOptions::default())
                    .await
                    .unwrap()
                    .is_none()
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_commit_after_precommit_retry_is_not_replayed() {
    for fault in [CommitFault::UnknownApplied, CommitFault::UnknownNotApplied] {
        let backend = backend();
        let runtime = runtime(&backend);
        let index = runtime.create_index("index", config()).await.unwrap();
        arm(
            &backend,
            WriteFaultPoint::GetForUpdate,
            1,
            ErrorKind::RetryableAbort,
        );
        backend.push_fault(fault).unwrap();
        assert_eq!(
            index.insert(record()).await.unwrap_err().kind(),
            ErrorKind::CommitOutcomeUnknown
        );
        assert_attempts(&backend, 2);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancellation_and_deadline_interrupt_precommit_backoff() {
    for stage in [WriteFaultPoint::BeginWrite, WriteFaultPoint::GetForUpdate] {
        for deadline in [false, true] {
            let backend = backend();
            let runtime = Runtime::new(
                backend.clone(),
                RuntimeConfig::default()
                    .with_maintenance(0, 1)
                    .unwrap()
                    .with_retry_backoff(Duration::from_secs(60), Duration::from_secs(60))
                    .unwrap(),
            )
            .unwrap();
            let index = runtime.create_index("index", config()).await.unwrap();
            let cancel = CancellationToken::new();
            let options = if deadline {
                OperationOptions::default()
                    .with_deadline(std::time::Instant::now() + Duration::from_secs(1))
            } else {
                OperationOptions::default().with_cancellation(cancel.clone())
            };
            arm(&backend, stage, 10, ErrorKind::RetryableAbort);
            // Synchronize through the existing counter, without a custom hook or task.
            let interrupt = async {
                while backend.operation_counts().begin_write == 0 {
                    tokio::task::yield_now().await;
                }
                if !deadline {
                    cancel.cancel();
                }
            };
            let (result, ()) = tokio::time::timeout(Duration::from_secs(2), async {
                tokio::join!(index.insert_with_control(record(), options), interrupt)
            })
            .await
            .unwrap();
            assert_eq!(
                result.unwrap_err().kind(),
                if deadline {
                    ErrorKind::DeadlineExceeded
                } else {
                    ErrorKind::Cancelled
                }
            );
            // The caller may return before the owned task observes cancellation.
            runtime.shutdown().await.unwrap();
            assert!(backend.operation_counts().begin_write > 0);
            assert_eq!(backend.active_transactions(), 0);
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn maintenance_retries_manifest_and_open_with_its_own_budget() {
    for stage in [WriteFaultPoint::BeginWrite, WriteFaultPoint::GetForUpdate] {
        for aborts in [1, 10] {
            let backend = backend();
            let runtime = runtime(&backend);
            let index = runtime.create_index("index", config()).await.unwrap();
            let manifest = support::read_manifest(&backend, index.logical_index_id()).await;
            arm(&backend, stage, aborts, ErrorKind::RetryableAbort);
            let retry = RetryPolicy::for_fixup(
                &RuntimeConfig::default()
                    .with_attempts(2, 3)
                    .unwrap()
                    .with_retry_backoff(Duration::from_nanos(1), Duration::from_nanos(1))
                    .unwrap(),
            );
            let result = ktann::maintenance::split::begin_split(
                &backend,
                &manifest,
                &TreeKey::encode(&[], &[]).unwrap(),
                PartitionKey::new(1).unwrap(),
                0,
                &retry,
            )
            .await;
            if aborts == 1 {
                assert!(result.is_ok());
            } else {
                assert_eq!(result.unwrap_err().kind(), ErrorKind::ContentionExhausted);
            }
            assert_attempts(&backend, 2);
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lifecycle_retries_complete_precommit_attempts() {
    for stage in [
        WriteFaultPoint::BeginWrite,
        WriteFaultPoint::GetForUpdate,
        WriteFaultPoint::BatchMutate,
    ] {
        let backend = backend();
        let runtime = runtime(&backend);
        arm(&backend, stage, 1, ErrorKind::RetryableAbort);
        let index = runtime.create_index("index", config()).await.unwrap();
        assert_eq!(
            index.logical_index_id().get(),
            1,
            "failed create allocates no durable identity"
        );
        assert_attempts(&backend, 2);
        arm(&backend, stage, 1, ErrorKind::RetryableAbort);
        runtime.drop_index("index").await.unwrap();
        assert_eq!(
            runtime.open_index("index").await.unwrap_err().kind(),
            ErrorKind::IndexNotFound
        );
        assert_eq!(backend.active_transactions(), 0);
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn lifecycle_precommit_errors_are_bounded_and_classified() {
    for stage in [
        WriteFaultPoint::BeginWrite,
        WriteFaultPoint::GetForUpdate,
        WriteFaultPoint::BatchMutate,
    ] {
        for (kind, expected, attempts) in [
            (ErrorKind::RetryableAbort, ErrorKind::ContentionExhausted, 3),
            (ErrorKind::Backend, ErrorKind::Backend, 1),
        ] {
            let backend = backend();
            let runtime = runtime(&backend);
            arm(&backend, stage, 10, kind);
            assert_eq!(
                runtime
                    .create_index("index", config())
                    .await
                    .unwrap_err()
                    .kind(),
                expected
            );
            assert_attempts(&backend, attempts);
            arm(&backend, stage, 0, kind);
            runtime.create_index("index", config()).await.unwrap();
            arm(&backend, stage, 10, kind);
            assert_eq!(
                runtime.drop_index("index").await.unwrap_err().kind(),
                expected
            );
            assert_attempts(&backend, attempts);
        }
    }
}
