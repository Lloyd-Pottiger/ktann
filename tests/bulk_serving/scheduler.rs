//! Distributed discovery and recovery using independent Runtime instances.
use super::complete::worker_options;
use super::*;
use ktann::api::{BulkSchedulerOptions, LogicalIndexId};

fn settings() -> BulkSchedulerOptions {
    BulkSchedulerOptions {
        max_jobs: 2,
        poll_interval: Duration::from_millis(10),
        lease_duration: Duration::from_millis(180),
    }
}
async fn wait_active<B: Backend>(job: &BulkBuildJob<B>) {
    tokio::time::timeout(Duration::from_secs(15), async {
        loop {
            match job.status().await {
                Ok(BulkBuildStatus::Published) => break,
                Ok(BulkBuildStatus::Failed { kind }) => panic!("scheduled build failed: {kind:?}"),
                Ok(_) => {}
                // The single-permit case deliberately saturates admission;
                // observation follows the same bounded retry contract as users.
                Err(error) if error.kind() == ErrorKind::LimitExceeded => {}
                Err(error) => panic!("status failed: {error:?}"),
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
}
async fn wait_empty(memory: &MemoryBackend, id: LogicalIndexId) {
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let mut txn = ReadLogicalTxn::bootstrap(memory.begin_read().await.unwrap());
            if txn
                .get(LogicalKey::BuildSchedule(id))
                .await
                .unwrap()
                .is_none()
            {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn automatic_publish_across_runtimes_with_single_foreground_permit() {
    let dir = Directory::new();
    let memory = MemoryBackend::with_test_config(TestConfig::default());
    let config = RuntimeConfig::default()
        .with_maintenance(0, 1)
        .unwrap()
        .with_foreground_operation_limit(1)
        .unwrap();
    let first = Runtime::new(memory.clone(), config.clone()).unwrap();
    let second = Runtime::new(memory.clone(), config).unwrap();
    let (job, _) = fixture(&first, &memory, &dir, 73).await;
    let options = worker_options(&dir);
    job.schedule(options.clone()).await.unwrap();
    job.schedule(options).await.unwrap();
    let cancel = CancellationToken::new();
    let a = tokio::spawn({
        let runtime = first.clone();
        let cancel = cancel.clone();
        async move {
            runtime
                .run_bulk_scheduler(
                    settings(),
                    OperationOptions::default().with_cancellation(cancel),
                )
                .await
        }
    });
    let b = tokio::spawn({
        let runtime = second.clone();
        let cancel = cancel.clone();
        async move {
            runtime
                .run_bulk_scheduler(
                    settings(),
                    OperationOptions::default().with_cancellation(cancel),
                )
                .await
        }
    });
    wait_active(&job).await;
    wait_empty(&memory, job.logical_index_id()).await;
    let index = second.open_index("bulk").await.unwrap();
    assert!(
        index
            .verify(VerifyOptions::default())
            .await
            .unwrap()
            .issues
            .is_empty()
    );
    cancel.cancel();
    assert_eq!(a.await.unwrap().unwrap_err().kind(), ErrorKind::Cancelled);
    assert_eq!(b.await.unwrap().unwrap_err().kind(), ErrorKind::Cancelled);
    first.shutdown().await.unwrap();
    second.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dropped_scheduler_is_replaced_after_lease_expiry() {
    let dir = Directory::new();
    let memory = MemoryBackend::with_test_config(TestConfig::default());
    let gate = Arc::new(Gate::default());
    let first = Runtime::new(
        Gated {
            memory: memory.clone(),
            gate: gate.clone(),
        },
        RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
    )
    .unwrap();
    let second = Runtime::new(
        memory.clone(),
        RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
    )
    .unwrap();
    let (job, artifact) = fixture(&first, &memory, &dir, 19).await;
    let options = worker_options(&dir);
    job.schedule(options.clone()).await.unwrap();
    gate.arm(3);
    let old = tokio::spawn({
        let runtime = first.clone();
        async move {
            runtime
                .run_bulk_scheduler(settings(), OperationOptions::default())
                .await
        }
    });
    gate.wait().await;
    assert_eq!(
        job.load_serving(&artifact, load_options())
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::BulkBuildBusy
    );
    old.abort();
    assert!(old.await.unwrap_err().is_cancelled());
    gate.release.notify_one();
    let stop = CancellationToken::new();
    let replacement = tokio::spawn({
        let runtime = second.clone();
        let stop = stop.clone();
        async move {
            runtime
                .run_bulk_scheduler(
                    settings(),
                    OperationOptions::default().with_cancellation(stop),
                )
                .await
        }
    });
    let recovered = second.open_bulk_build("bulk").await.unwrap();
    wait_active(&recovered).await;
    wait_empty(&memory, recovered.logical_index_id()).await;
    stop.cancel();
    assert_eq!(
        replacement.await.unwrap().unwrap_err().kind(),
        ErrorKind::Cancelled
    );
    first.shutdown().await.unwrap();
    second.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn renewal_prevents_takeover_while_preparation_is_paused() {
    let settings = BulkSchedulerOptions {
        lease_duration: Duration::from_secs(2),
        ..settings()
    };
    let dir = Directory::new();
    let memory = MemoryBackend::with_test_config(TestConfig::default());
    let gate = Arc::new(Gate::default());
    let first = Runtime::new(
        Gated {
            memory: memory.clone(),
            gate: gate.clone(),
        },
        RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
    )
    .unwrap();
    let second = Runtime::new(
        memory.clone(),
        RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
    )
    .unwrap();
    let (job, _) = fixture(&first, &memory, &dir, 19).await;
    job.schedule(worker_options(&dir)).await.unwrap();
    gate.arm(3);
    let stop = CancellationToken::new();
    let a = tokio::spawn({
        let runtime = first.clone();
        let stop = stop.clone();
        async move {
            runtime
                .run_bulk_scheduler(
                    settings,
                    OperationOptions::default().with_cancellation(stop),
                )
                .await
        }
    });
    gate.wait().await;
    let before = read_queue(&memory, job.logical_index_id()).await;
    let b = tokio::spawn({
        let runtime = second.clone();
        let stop = stop.clone();
        async move {
            runtime
                .run_bulk_scheduler(
                    settings,
                    OperationOptions::default().with_cancellation(stop),
                )
                .await
        }
    });
    tokio::time::sleep(Duration::from_millis(4200)).await;
    let after = read_queue(&memory, job.logical_index_id()).await;
    assert_eq!(
        &before[before.len() - 40..before.len() - 8],
        &after[after.len() - 40..after.len() - 8],
        "owner must be renewed, not replaced"
    );
    assert!(after[after.len() - 8..] > before[before.len() - 8..]);
    gate.release.notify_one();
    wait_active(&job).await;
    wait_empty(&memory, job.logical_index_id()).await;
    stop.cancel();
    let _ = a.await.unwrap();
    let _ = b.await.unwrap();
    first.shutdown().await.unwrap();
    second.shutdown().await.unwrap();
}
async fn read_queue(memory: &MemoryBackend, id: LogicalIndexId) -> Vec<u8> {
    use ktann::storage::backend::ReadOps;
    memory
        .begin_read()
        .await
        .unwrap()
        .get(ktann::storage::keys::build_schedule_key(id).into())
        .await
        .unwrap()
        .unwrap()
        .to_vec()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_enqueue_and_claim_resolve_without_duplicate_jobs() {
    for fault in [CommitFault::UnknownApplied, CommitFault::UnknownNotApplied] {
        let dir = Directory::new();
        let memory = MemoryBackend::with_test_config(TestConfig::default());
        let runtime = Runtime::new(
            memory.clone(),
            RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
        )
        .unwrap();
        let (job, _) = fixture(&runtime, &memory, &dir, 19).await;
        memory.set_fault_plan(vec![fault]).unwrap();
        job.schedule(worker_options(&dir)).await.unwrap();
        memory.set_fault_plan(vec![fault]).unwrap();
        let stop = CancellationToken::new();
        let scheduler = tokio::spawn({
            let runtime = runtime.clone();
            let stop = stop.clone();
            async move {
                runtime
                    .run_bulk_scheduler(
                        settings(),
                        OperationOptions::default().with_cancellation(stop),
                    )
                    .await
            }
        });
        wait_active(&job).await;
        wait_empty(&memory, job.logical_index_id()).await;
        stop.cancel();
        let _ = scheduler.await.unwrap();
        runtime.shutdown().await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn terminal_failure_and_queued_abort_leave_no_scheduler_work() {
    for fail in [true, false] {
        let dir = Directory::new();
        let memory = MemoryBackend::with_test_config(TestConfig::default());
        let runtime = Runtime::new(
            memory.clone(),
            RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
        )
        .unwrap();
        let (job, _) = fixture(&runtime, &memory, &dir, 19).await;
        let mut options = worker_options(&dir);
        if fail {
            options.max_artifact_bytes = 100;
        }
        job.schedule(options).await.unwrap();
        if !fail {
            job.abort().await.unwrap();
        }
        let stop = CancellationToken::new();
        let scheduler = tokio::spawn({
            let runtime = runtime.clone();
            let stop = stop.clone();
            async move {
                runtime
                    .run_bulk_scheduler(
                        settings(),
                        OperationOptions::default().with_cancellation(stop),
                    )
                    .await
            }
        });
        wait_empty(&memory, job.logical_index_id()).await;
        if fail {
            assert_eq!(
                job.status().await.unwrap(),
                BulkBuildStatus::Failed {
                    kind: ErrorKind::LimitExceeded
                }
            );
            job.abort().await.unwrap();
        }
        assert!(
            !scheduler.is_finished(),
            "one failed job must not stop queue discovery"
        );
        stop.cancel();
        let _ = scheduler.await.unwrap();
        runtime.shutdown().await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn takeover_fences_a_scheduled_chunk_prepared_by_the_old_owner() {
    let dir = Directory::new();
    let memory = MemoryBackend::with_test_config(TestConfig::default());
    let gate = Arc::new(Gate::default());
    let first = Runtime::new(
        Gated {
            memory: memory.clone(),
            gate: gate.clone(),
        },
        RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
    )
    .unwrap();
    let second = Runtime::new(
        memory.clone(),
        RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
    )
    .unwrap();
    let (job, _) = fixture(&first, &memory, &dir, 19).await;
    job.schedule(worker_options(&dir)).await.unwrap();
    gate.arm(6);
    let stop = CancellationToken::new();
    let a = tokio::spawn({
        let runtime = first.clone();
        let stop = stop.clone();
        async move {
            runtime
                .run_bulk_scheduler(
                    BulkSchedulerOptions {
                        lease_duration: Duration::from_secs(30),
                        ..settings()
                    },
                    OperationOptions::default().with_cancellation(stop),
                )
                .await
        }
    });
    gate.wait().await;
    // Make this lease eligible without sleeping or allowing the old heartbeat
    // to race the fixture. The second Runtime performs the actual token takeover.
    let key: Bytes = ktann::storage::keys::build_schedule_key(job.logical_index_id()).into();
    let mut txn = memory.begin_write().await.unwrap();
    let mut bytes = txn
        .get_for_update(key.clone())
        .await
        .unwrap()
        .unwrap()
        .to_vec();
    let len = bytes.len();
    bytes[len - 8..].copy_from_slice(&1_u64.to_be_bytes());
    txn.put(key, bytes.into()).await.unwrap();
    txn.commit().await.unwrap();
    let b = tokio::spawn({
        let runtime = second.clone();
        let stop = stop.clone();
        async move {
            runtime
                .run_bulk_scheduler(
                    settings(),
                    OperationOptions::default().with_cancellation(stop),
                )
                .await
        }
    });
    let recovered = second.open_bulk_build("bulk").await.unwrap();
    wait_active(&recovered).await;
    gate.release.notify_one();
    wait_empty(&memory, job.logical_index_id()).await;
    let index = second.open_index("bulk").await.unwrap();
    assert!(
        index
            .verify(VerifyOptions::default())
            .await
            .unwrap()
            .issues
            .is_empty()
    );
    stop.cancel();
    let _ = a.await.unwrap();
    let _ = b.await.unwrap();
    first.shutdown().await.unwrap();
    second.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn separate_runtimes_progress_different_jobs_in_one_workspace() {
    let dir = Directory::new();
    let memory = MemoryBackend::with_test_config(TestConfig::default());
    let gate = Arc::new(Gate::default());
    let first = Runtime::new(
        Gated {
            memory: memory.clone(),
            gate: gate.clone(),
        },
        RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
    )
    .unwrap();
    let second = Runtime::new(
        memory.clone(),
        RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
    )
    .unwrap();
    let (one, _) = fixture(&first, &memory, &dir, 19).await;
    let input = InputSnapshot::create(
        &dir.0.join("source-two"),
        config(Metric::L2, true),
        1024 * 1024,
        (100..119).map(|id| Ok(record(id))),
    )
    .unwrap();
    let two = second
        .start_bulk_build("bulk-two", &input, options().tree)
        .await
        .unwrap();
    let worker = worker_options(&dir);
    one.schedule(worker.clone()).await.unwrap();
    two.schedule(worker).await.unwrap();
    let settings = BulkSchedulerOptions {
        max_jobs: 1,
        lease_duration: Duration::from_secs(30),
        ..settings()
    };
    let stop = CancellationToken::new();
    gate.arm(3);
    let a = tokio::spawn({
        let runtime = first.clone();
        let stop = stop.clone();
        async move {
            runtime
                .run_bulk_scheduler(
                    settings,
                    OperationOptions::default().with_cancellation(stop),
                )
                .await
        }
    });
    gate.wait().await;
    let b = tokio::spawn({
        let runtime = second.clone();
        let stop = stop.clone();
        async move {
            runtime
                .run_bulk_scheduler(
                    settings,
                    OperationOptions::default().with_cancellation(stop),
                )
                .await
        }
    });
    wait_active(&two).await;
    assert_ne!(one.status().await.unwrap(), BulkBuildStatus::Published);
    gate.release.notify_one();
    wait_active(&one).await;
    wait_empty(&memory, one.logical_index_id()).await;
    wait_empty(&memory, two.logical_index_id()).await;
    stop.cancel();
    let _ = a.await.unwrap();
    let _ = b.await.unwrap();
    first.shutdown().await.unwrap();
    second.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn enqueue_rejects_a_file_as_workspace_before_creating_queue_state() {
    let dir = Directory::new();
    let memory = MemoryBackend::with_test_config(TestConfig::default());
    let runtime = Runtime::new(
        memory.clone(),
        RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
    )
    .unwrap();
    let (job, _) = fixture(&runtime, &memory, &dir, 0).await;
    let options = ktann::api::BulkWorkerOptions::new(dir.0.join("source/data.bin"));
    assert_eq!(
        job.schedule(options).await.unwrap_err().kind(),
        ErrorKind::InvalidArgument
    );
    wait_empty(&memory, job.logical_index_id()).await;
    runtime.shutdown().await.unwrap();
}
