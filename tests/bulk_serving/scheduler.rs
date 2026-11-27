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
            let sched_empty = txn
                .get(LogicalKey::BuildSchedule(id))
                .await
                .unwrap()
                .is_none();
            let mut slots_empty = true;
            for slot in 0..4 {
                if txn
                    .get(LogicalKey::BuildScheduleSlot { index: id, slot })
                    .await
                    .unwrap()
                    .is_some()
                {
                    slots_empty = false;
                    break;
                }
            }
            if sched_empty && slots_empty {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn heartbeat_waiting_for_a_transaction_slot_allows_progress_and_cancellation() {
    for cancel_while_waiting in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let memory = MemoryBackend::with_test_config(TestConfig::default());
        let gate = Arc::new(Gate {
            write_slots: Some(Arc::new(tokio::sync::Semaphore::new(1))),
            ..Gate::default()
        });
        let runtime = Runtime::new(
            Gated {
                memory: memory.clone(),
                gate: gate.clone(),
            },
            RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
        )
        .unwrap();
        let job = reservation(&runtime, &dir, 5).await.0;
        job.schedule(worker_options(&dir)).await.unwrap();
        // Claim the schedule, then hold the workspace transaction across a heartbeat.
        gate.arm(2);
        let stop = CancellationToken::new();
        let scheduler = tokio::spawn({
            let runtime = runtime.clone();
            let stop = stop.clone();
            async move {
                runtime
                    .run_bulk_scheduler(
                        BulkSchedulerOptions {
                            max_jobs: 1,
                            ..settings()
                        },
                        OperationOptions::default().with_cancellation(stop),
                    )
                    .await
            }
        });
        gate.wait().await;
        tokio::time::timeout(Duration::from_secs(5), gate.waiting_for_slot.notified())
            .await
            .unwrap();
        if cancel_while_waiting {
            stop.cancel();
        }
        gate.release.notify_one();
        if !cancel_while_waiting {
            wait_active(&job).await;
            wait_empty(&memory, job.logical_index_id()).await;
        }
        stop.cancel();
        assert_eq!(
            tokio::time::timeout(Duration::from_secs(5), scheduler)
                .await
                .unwrap()
                .unwrap()
                .unwrap_err()
                .kind(),
            ErrorKind::Cancelled
        );
        runtime.shutdown().await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn scheduler_reports_missing_or_corrupt_build_descriptor() {
    for corrupt in [false, true] {
        let dir = tempfile::tempdir().unwrap();
        let memory = MemoryBackend::with_test_config(TestConfig::default());
        let runtime = Runtime::new(
            memory.clone(),
            RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
        )
        .unwrap();
        let job = reservation(&runtime, &dir, 0).await.0;
        job.schedule(worker_options(&dir)).await.unwrap();
        let mut txn = memory.begin_write().await.unwrap();
        let key = keys::build_descriptor_key(job.logical_index_id()).into();
        if corrupt {
            txn.put(key, Bytes::from_static(b"invalid descriptor"))
                .await
                .unwrap();
        } else {
            txn.delete(key).await.unwrap();
        }
        txn.commit().await.unwrap();
        assert_eq!(
            tokio::time::timeout(
                Duration::from_secs(2),
                runtime.run_bulk_scheduler(settings(), OperationOptions::default()),
            )
            .await
            .unwrap()
            .unwrap_err()
            .kind(),
            ErrorKind::Corruption
        );
        runtime.shutdown().await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn automatic_publish_across_runtimes_with_single_foreground_permit() {
    let dir = tempfile::tempdir().unwrap();
    let memory = MemoryBackend::with_test_config(TestConfig::default());
    let config = RuntimeConfig::default()
        .with_maintenance(0, 1)
        .unwrap()
        .with_foreground_operation_limit(1)
        .unwrap();
    let first = Runtime::new(memory.clone(), config.clone()).unwrap();
    let second = Runtime::new(memory.clone(), config).unwrap();
    let job = reservation(&first, &dir, 73).await.0;
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
    // Stop discovery before the final reads need the sole foreground permit.
    cancel.cancel();
    assert_eq!(a.await.unwrap().unwrap_err().kind(), ErrorKind::Cancelled);
    assert_eq!(b.await.unwrap().unwrap_err().kind(), ErrorKind::Cancelled);
    let index = second.open_index("bulk").await.unwrap();
    assert!(
        index
            .verify(VerifyOptions::default())
            .await
            .unwrap()
            .issues
            .is_empty()
    );
    first.shutdown().await.unwrap();
    second.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn dropped_scheduler_is_replaced_after_lease_expiry() {
    let dir = tempfile::tempdir().unwrap();
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
    let (job, _) = reservation(&first, &dir, 19).await;
    let mut options = worker_options(&dir);
    options.parallelism = 1;
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
        job.run_worker(options.clone()).await.unwrap_err().kind(),
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
    let dir = tempfile::tempdir().unwrap();
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
    let job = reservation(&first, &dir, 19).await.0;
    let mut paused_opts = worker_options(&dir);
    paused_opts.parallelism = 1;
    job.schedule(paused_opts).await.unwrap();
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
    let before = read_slot(&memory, job.logical_index_id()).await;
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
    let after = read_slot(&memory, job.logical_index_id()).await;
    assert_ne!(&before[9..41], &[0; 32]);
    assert_eq!(
        &before[9..41],
        &after[9..41],
        "owner must be renewed, not replaced"
    );
    assert!(after[41..49] > before[41..49]);
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
async fn read_slot(memory: &MemoryBackend, id: LogicalIndexId) -> Vec<u8> {
    use ktann::storage::backend::ReadOps;
    memory
        .begin_read()
        .await
        .unwrap()
        .get(ktann::storage::keys::build_schedule_slot_key(id, 0).into())
        .await
        .unwrap()
        .unwrap()
        .to_vec()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn unknown_enqueue_and_claim_resolve_without_duplicate_jobs() {
    for fault in [CommitFault::UnknownApplied, CommitFault::UnknownNotApplied] {
        let dir = tempfile::tempdir().unwrap();
        let memory = MemoryBackend::with_test_config(TestConfig::default());
        let runtime = Runtime::new(
            memory.clone(),
            RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
        )
        .unwrap();
        let job = reservation(&runtime, &dir, 19).await.0;
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

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_enqueue_exhaustion_preserves_uncertainty() {
    for fault in [CommitFault::UnknownApplied, CommitFault::UnknownNotApplied] {
        let dir = tempfile::tempdir().unwrap();
        let memory = MemoryBackend::with_test_config(TestConfig::default());
        let runtime = Runtime::new(
            memory.clone(),
            RuntimeConfig::default()
                .with_maintenance(0, 1)
                .unwrap()
                .with_attempts(1, 1)
                .unwrap(),
        )
        .unwrap();
        let job = reservation(&runtime, &dir, 0).await.0;
        let options = worker_options(&dir);
        memory.set_fault_plan(vec![fault]).unwrap();
        assert_eq!(
            job.schedule(options.clone()).await.unwrap_err().kind(),
            ErrorKind::CommitOutcomeUnknown
        );
        job.schedule(options).await.unwrap();
        runtime.shutdown().await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn discovery_passes_leased_entries_without_a_poll_per_entry() {
    let dir = tempfile::tempdir().unwrap();
    let memory = MemoryBackend::new();
    let runtime = Runtime::new(
        memory.clone(),
        RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
    )
    .unwrap();
    let mut leased = Vec::new();
    for n in 0..32 {
        leased.push(
            runtime
                .create_index(&format!("leased-{n}"), config(Metric::L2, true))
                .await
                .unwrap()
                .logical_index_id(),
        );
    }
    let (job, source) = reservation(&runtime, &dir, 0).await;
    let options = worker_options(&dir);
    job.schedule(options.clone()).await.unwrap();
    let second = runtime
        .start_bulk_build("second", &source, construction_options())
        .await
        .unwrap();
    second.schedule(options).await.unwrap();
    // A published job can retain its queue entry while its owner cleans up.
    // Keep those earlier entries leased throughout the discovery test.
    let entry = read_queue(&memory, job.logical_index_id()).await;
    let mut slot_bytes = vec![0_u8; 266];
    slot_bytes[0] = 0x10;
    slot_bytes[1..9].copy_from_slice(&1_u64.to_be_bytes());
    slot_bytes[9..41].fill(1);
    slot_bytes[41..49].copy_from_slice(&u64::MAX.to_be_bytes());
    let mut txn = memory.begin_write().await.unwrap();
    for id in leased {
        txn.put(
            keys::build_schedule_key(id).into(),
            Bytes::copy_from_slice(&entry),
        )
        .await
        .unwrap();
        txn.put(
            keys::build_schedule_slot_key(id, 0).into(),
            Bytes::copy_from_slice(&slot_bytes),
        )
        .await
        .unwrap();
    }
    txn.commit().await.unwrap();
    let stop = CancellationToken::new();
    let scheduler = tokio::spawn({
        let runtime = runtime.clone();
        let stop = stop.clone();
        async move {
            runtime
                .run_bulk_scheduler(
                    BulkSchedulerOptions::default(),
                    OperationOptions::default().with_cancellation(stop),
                )
                .await
        }
    });
    // One slot and a one-second poll: scanning one entry per poll would need
    // over 32 seconds, beyond wait_active's bound. Both runnable jobs must run.
    wait_active(&job).await;
    wait_active(&second).await;
    stop.cancel();
    assert_eq!(
        scheduler.await.unwrap().unwrap_err().kind(),
        ErrorKind::Cancelled
    );
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn terminal_failure_and_queued_abort_leave_no_scheduler_work() {
    for fail in [true, false] {
        let dir = tempfile::tempdir().unwrap();
        let memory = MemoryBackend::with_test_config(TestConfig::default());
        let runtime = Runtime::new(
            memory.clone(),
            RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
        )
        .unwrap();
        let job = reservation(&runtime, &dir, 19).await.0;
        let mut options = worker_options(&dir);
        if fail {
            options.memory_bytes = 8 * 1024;
            options.scratch_bytes = 100;
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
    let dir = tempfile::tempdir().unwrap();
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
    let job = reservation(&first, &dir, 19).await.0;
    let mut paused_opts = worker_options(&dir);
    paused_opts.parallelism = 1;
    job.schedule(paused_opts).await.unwrap();
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
    let key: Bytes =
        ktann::storage::keys::build_schedule_slot_key(job.logical_index_id(), 0).into();
    let mut txn = memory.begin_write().await.unwrap();
    let mut bytes = txn
        .get_for_update(key.clone())
        .await
        .unwrap()
        .unwrap()
        .to_vec();
    bytes[41..49].copy_from_slice(&1_u64.to_be_bytes());
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
    let dir = tempfile::tempdir().unwrap();
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
    let one = reservation(&first, &dir, 19).await.0;
    let input = InputSnapshot::create(
        &dir.path().join("source-two"),
        config(Metric::L2, true),
        1024 * 1024,
        (100..119).map(|id| Ok(record(id))),
    )
    .unwrap();
    let two = second
        .start_bulk_build("bulk-two", &input, construction_options())
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
    // Pause the workspace claim with the shared root lock already held.
    gate.arm(2);
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
    // Wait until the published job has encountered busy cleanup and released
    // its queue claim for retry, before allowing the other worker to finish.
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            let slot = read_slot(&memory, two.logical_index_id()).await;
            if slot[9..41] == [0; 32] {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    gate.release.notify_one();
    wait_active(&one).await;
    wait_empty(&memory, one.logical_index_id()).await;
    wait_empty(&memory, two.logical_index_id()).await;
    stop.cancel();
    assert_eq!(a.await.unwrap().unwrap_err().kind(), ErrorKind::Cancelled);
    assert_eq!(b.await.unwrap().unwrap_err().kind(), ErrorKind::Cancelled);
    let mut txn = ReadLogicalTxn::bootstrap(memory.begin_read().await.unwrap());
    for id in [one.logical_index_id(), two.logical_index_id()] {
        assert!(
            txn.get(LogicalKey::BuildWorkspace(id))
                .await
                .unwrap()
                .is_none()
        );
    }
    first.shutdown().await.unwrap();
    second.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn enqueue_rejects_a_file_as_workspace_before_creating_queue_state() {
    let dir = tempfile::tempdir().unwrap();
    let memory = MemoryBackend::with_test_config(TestConfig::default());
    let runtime = Runtime::new(
        memory.clone(),
        RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
    )
    .unwrap();
    let job = reservation(&runtime, &dir, 0).await.0;
    let options = ktann::api::BulkWorkerOptions::new(dir.path().join("source/manifest.bin"));
    assert_eq!(
        job.schedule(options).await.unwrap_err().kind(),
        ErrorKind::InvalidArgument
    );
    wait_empty(&memory, job.logical_index_id()).await;
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn collaborative_schedulers_share_one_queued_build() {
    let dir = tempfile::tempdir().unwrap();
    let memory = MemoryBackend::new();
    let first = Runtime::new(
        memory.clone(),
        RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
    )
    .unwrap();
    let second = Runtime::new(
        memory.clone(),
        RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
    )
    .unwrap();
    let job = reservation(&first, &dir, 120).await.0;
    let mut opts = worker_options(&dir);
    opts.distributed_workers = 2;
    opts.load = BulkLoadOptions::default();
    job.schedule(opts).await.unwrap();

    let stop = CancellationToken::new();
    let a = tokio::spawn({
        let rt = first.clone();
        let stop = stop.clone();
        async move {
            rt.run_bulk_scheduler(
                settings(),
                OperationOptions::default().with_cancellation(stop),
            )
            .await
        }
    });
    let b = tokio::spawn({
        let rt = second.clone();
        let stop = stop.clone();
        async move {
            rt.run_bulk_scheduler(
                settings(),
                OperationOptions::default().with_cancellation(stop),
            )
            .await
        }
    });
    wait_active(&job).await;
    wait_empty(&memory, job.logical_index_id()).await;
    stop.cancel();
    let _ = a.await.unwrap();
    let _ = b.await.unwrap();
    let lid = job.logical_index_id();
    let mut txn = ReadLogicalTxn::bootstrap(memory.begin_read().await.unwrap());
    assert!(
        txn.get(LogicalKey::BuildWorkspace(lid))
            .await
            .unwrap()
            .is_none()
    );
    match txn.get(LogicalKey::BuildProgress(lid)).await.unwrap() {
        Some(PersistentValue::BuildProgress(p)) => {
            assert!(p.total_entries() > 0);
            assert_eq!(p.validation_slices(), 1);
        }
        other => panic!("expected compact BuildProgress, got {other:?}"),
    }
    for slot in 0..2 {
        assert!(
            txn.get(LogicalKey::BuildScheduleSlot { index: lid, slot })
                .await
                .unwrap()
                .is_none()
        );
    }
    drop(txn);
    let index = second.open_index("bulk").await.unwrap();
    let verified = index.verify(VerifyOptions::default()).await.unwrap();
    assert!(verified.complete && verified.issues.is_empty());
    assert_eq!(verified.objects.vector_records, 120);
    first.shutdown().await.unwrap();
    second.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn abort_without_or_during_scheduler_retires_schedule_without_stopping_daemon() {
    let dir = tempfile::tempdir().unwrap();
    let memory = MemoryBackend::with_test_config(TestConfig::default());
    let gate = Arc::new(Gate::default());
    let rt = Runtime::new(
        Gated {
            memory: memory.clone(),
            gate: gate.clone(),
        },
        RuntimeConfig::default().with_maintenance(0, 2).unwrap(),
    )
    .unwrap();

    // 1. Scheduled job aborted before any scheduler runs retires `BuildSchedule` immediately.
    let (job_idle, _) = reservation(&rt, &dir, 10).await;
    let idle_id = job_idle.logical_index_id();
    job_idle.schedule(worker_options(&dir)).await.unwrap();
    job_idle.abort().await.unwrap();
    wait_empty(&memory, idle_id).await;

    // 2. Scheduled job aborted mid-flight while `run_bulk_scheduler` holds it does not kill the daemon.
    let dir2 = tempfile::tempdir().unwrap();
    let (job_active, _) = reservation(&rt, &dir2, 25).await;
    let active_id = job_active.logical_index_id();
    job_active.schedule(worker_options(&dir2)).await.unwrap();
    gate.arm(2);
    let stop = CancellationToken::new();
    let daemon = tokio::spawn({
        let rt = rt.clone();
        let stop = stop.clone();
        async move {
            rt.run_bulk_scheduler(
                settings(),
                OperationOptions::default().with_cancellation(stop),
            )
            .await
        }
    });
    gate.wait().await;
    let abort_fut = tokio::spawn({
        let job_active = job_active.clone();
        async move { job_active.abort().await }
    });
    tokio::time::sleep(Duration::from_millis(20)).await;
    gate.release.notify_waiters();
    abort_fut.await.unwrap().unwrap();
    wait_empty(&memory, active_id).await;
    assert!(
        !daemon.is_finished(),
        "mid-flight job.abort() must not terminate run_bulk_scheduler"
    );
    stop.cancel();
    assert_eq!(
        daemon.await.unwrap().unwrap_err().kind(),
        ErrorKind::Cancelled
    );
    rt.shutdown().await.unwrap();
}
