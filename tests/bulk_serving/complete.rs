//! Complete worker, multi-worker collaboration, publication, and reclamation tests.

use super::*;
use ktann::api::BulkWorkerOptions;

pub(super) fn worker_options(dir: &tempfile::TempDir) -> BulkWorkerOptions {
    let root = dir.path().join("work");
    if !root.exists() {
        fs::create_dir(&root).unwrap();
    }
    let mut worker = BulkWorkerOptions::new(root);
    worker.parallelism = 2;
    worker.memory_bytes = 8 * 1024 * 1024;
    worker.scratch_bytes = 64 * 1024 * 1024;
    worker.load = load_options();
    worker
}

fn artifact_directories(root: &std::path::Path) -> usize {
    fs::read_dir(root)
        .unwrap()
        .map(|x| x.unwrap())
        .filter(|x| x.file_type().unwrap().is_dir())
        .count()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn build_publish_reopen_cleanup_and_online_mutations_are_complete() {
    for count in [0, 73] {
        let dir = tempfile::tempdir().unwrap();
        let memory = MemoryBackend::new();
        let first = runtime(memory.clone());
        let (job, _) = reservation(&first, &dir, count).await;
        let opts = worker_options(&dir);
        job.run_worker(opts.clone()).await.unwrap();
        let id = job.logical_index_id();
        assert!(matches!(
            job.status().await.unwrap(),
            BulkBuildStatus::Loaded { .. }
        ));
        fs::rename(
            dir.path().join("source"),
            dir.path().join("relocated-source"),
        )
        .unwrap();
        first.shutdown().await.unwrap();
        let runtime = runtime(memory.clone());
        let job = runtime.open_bulk_build("bulk").await.unwrap();
        job.run_worker(opts.clone()).await.unwrap(); // sealed Loaded state survives source relocation
        let index = job.complete(opts.clone()).await.unwrap();
        assert_eq!(index.logical_index_id(), id);
        assert_eq!(
            index.config().local_graph(),
            Some(ktann::api::LocalGraphConfig::default())
        );
        let reopened = runtime.open_index("bulk").await.unwrap();
        let hits = reopened
            .search(ktann::api::SearchRequest::new(record(0).vector().to_vec(), 1).unwrap())
            .await
            .unwrap();
        assert_eq!(hits.hits.len(), usize::from(count != 0));
        assert_eq!(job.status().await.unwrap(), BulkBuildStatus::Published);
        assert_eq!(
            job.abort().await.unwrap_err().kind(),
            ErrorKind::InvalidArgument
        );
        assert_eq!(
            runtime.open_index("bulk").await.unwrap().logical_index_id(),
            id
        );

        assert_eq!(artifact_directories(&opts.workspace), 0);
        assert!(dir.path().join("relocated-source/manifest.bin").exists());
        let report = index.verify(VerifyOptions::default()).await.unwrap();
        assert!(
            report.complete && report.issues.is_empty(),
            "{:?}",
            report.issues
        );
        assert_eq!(report.objects.vector_records, count);
        for n in 0..count {
            let actual = index
                .get(record(n).id().clone(), GetOptions::default().with_payload())
                .await
                .unwrap()
                .unwrap();
            assert_eq!(actual.vector(), record(n).vector());
            assert_eq!(
                actual.payload(),
                &record(n)
                    .payload()
                    .cloned()
                    .map_or(PayloadProjection::Absent, PayloadProjection::Present)
            );
        }
        index
            .batch_mutate(vec![
                Mutation::Insert(record(999)),
                Mutation::Upsert(record(1000)),
            ])
            .await
            .unwrap();
        assert!(
            index
                .get(record(999).id().clone(), GetOptions::default())
                .await
                .unwrap()
                .is_some()
        );
        assert_eq!(job.publish().await.unwrap().logical_index_id(), id);
        job.cleanup().await.unwrap();
        runtime.drop_index("bulk").await.unwrap();
        let replacement = runtime
            .create_index("bulk", config(Metric::L2, true))
            .await
            .unwrap();
        assert_ne!(replacement.logical_index_id(), id);
        assert_eq!(
            job.publish().await.unwrap_err().kind(),
            ErrorKind::IndexNotFound
        );
        runtime.shutdown().await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn sealed_backend_damage_is_terminal_and_cannot_publish() {
    for damage in ["extra", "missing", "different"] {
        let dir = tempfile::tempdir().unwrap();
        let memory = MemoryBackend::new();
        let runtime = runtime(memory.clone());
        let (job, _) = reservation(&runtime, &dir, 19).await;
        let opts = worker_options(&dir);
        job.run_worker(opts.clone()).await.unwrap();
        let mut txn = memory.begin_write().await.unwrap();
        if damage != "missing" {
            txn.put(
                Bytes::from(
                    keys::record_key(
                        job.logical_index_id(),
                        record(if damage == "extra" { 999 } else { 0 }).id(),
                    )
                    .unwrap(),
                ),
                Bytes::from(
                    ValueCodec::for_index(job.index_manifest())
                        .encode(&PersistentValue::VectorRecord(
                            ktann::storage::values::VectorRecord::new(
                                record(if damage == "extra" { 999 } else { 0 }).id().clone(),
                                record(999).vector().to_vec(),
                                record(999).fields().to_vec(),
                            ),
                        ))
                        .unwrap(),
                ),
            )
            .await
            .unwrap();
        } else {
            txn.delete(Bytes::from(
                keys::record_key(job.logical_index_id(), record(0).id()).unwrap(),
            ))
            .await
            .unwrap();
        }
        txn.commit().await.unwrap();
        assert_eq!(
            job.publish().await.unwrap_err().kind(),
            ErrorKind::Corruption
        );
        assert_eq!(
            job.status().await.unwrap(),
            BulkBuildStatus::Failed {
                kind: ErrorKind::Corruption
            }
        );
        assert_eq!(
            runtime.open_index("bulk").await.unwrap_err().kind(),
            ErrorKind::IndexBuilding
        );
        assert_eq!(
            job.run_worker(opts.clone()).await.unwrap_err().kind(),
            ErrorKind::Corruption
        );
        job.abort().await.unwrap();
        assert_eq!(artifact_directories(&opts.workspace), 0);
        runtime.shutdown().await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cancelled_validation_reuses_slices_and_rejects_worker_takeover() {
    let dir = tempfile::tempdir().unwrap();
    let memory = MemoryBackend::with_test_config(TestConfig::default());
    let gate = Arc::new(Gate::default());
    let runtime = Runtime::new(
        Gated {
            memory: memory.clone(),
            gate: gate.clone(),
        },
        RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
    )
    .unwrap();
    let (job, _) = reservation(&runtime, &dir, 19).await;
    let opts = worker_options(&dir);
    job.run_worker(opts.clone()).await.unwrap();
    gate.arm(2);
    let worker = job.clone();
    let cancellation = CancellationToken::new();
    let control = OperationOptions::default().with_cancellation(cancellation.clone());
    let pending = tokio::spawn(async move { worker.publish_with_control(control).await });
    gate.wait().await;
    assert_eq!(
        job.run_worker(opts.clone()).await.unwrap_err().kind(),
        ErrorKind::BulkBuildSuperseded
    );
    cancellation.cancel();
    gate.release.notify_one();
    assert_eq!(
        pending.await.unwrap().unwrap_err().kind(),
        ErrorKind::Cancelled
    );
    assert!(matches!(
        job.status().await.unwrap(),
        BulkBuildStatus::Validating { .. }
    ));
    let index = runtime
        .open_bulk_build("bulk")
        .await
        .unwrap()
        .complete(opts)
        .await
        .unwrap();
    assert!(
        index
            .verify(VerifyOptions::default())
            .await
            .unwrap()
            .issues
            .is_empty()
    );
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn abort_during_worker_execution_fences_writes_and_reclaims_workspace() {
    let dir = tempfile::tempdir().unwrap();
    let memory = MemoryBackend::with_test_config(TestConfig::default());
    let gate = Arc::new(Gate::default());
    let runtime = Runtime::new(
        Gated {
            memory: memory.clone(),
            gate: gate.clone(),
        },
        RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
    )
    .unwrap();
    let (job, _) = reservation(&runtime, &dir, 19).await;
    let mut opts = worker_options(&dir);
    opts.parallelism = 1;
    gate.arm(2);
    let worker = job.clone();
    let args = opts.clone();
    let pending = tokio::spawn(async move { worker.run_worker(args).await });
    gate.wait().await;
    job.abort().await.unwrap();
    assert_eq!(job.status().await.unwrap(), BulkBuildStatus::Aborted);
    gate.release.notify_one();
    assert!(matches!(
        pending.await.unwrap().unwrap_err().kind(),
        ErrorKind::IndexNotFound | ErrorKind::IndexDropping
    ));
    let report = runtime.cleanup_bulk_builds(10, None).await.unwrap();
    assert!(report.reclaimed <= 1);
    assert_eq!(report.pending, 0);
    assert_eq!(artifact_directories(&opts.workspace), 0);
    assert!(dir.path().join("source/manifest.bin").exists());
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_outcomes_at_sealing_proof_and_publication_are_idempotent() {
    for fault in [
        CommitFault::Abort,
        CommitFault::UnknownApplied,
        CommitFault::UnknownNotApplied,
    ] {
        for point in [0, 1, 2] {
            let dir = tempfile::tempdir().unwrap();
            let memory = MemoryBackend::with_test_config(TestConfig::default());
            let runtime = runtime(memory.clone());
            let (job, _) = reservation(&runtime, &dir, 19).await;
            job.run_worker(worker_options(&dir)).await.unwrap();
            let mut plan = vec![CommitFault::Normal; point];
            plan.push(fault);
            memory.set_fault_plan(plan).unwrap();
            let index = job.publish().await.unwrap();
            assert!(
                index
                    .verify(VerifyOptions::default())
                    .await
                    .unwrap()
                    .issues
                    .is_empty()
            );
            assert_eq!(job.status().await.unwrap(), BulkBuildStatus::Published);
            runtime.shutdown().await.unwrap();
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn worker_claim_and_chunk_unknown_outcomes_recover() {
    for fault in [CommitFault::UnknownApplied, CommitFault::UnknownNotApplied] {
        for position in [0, 1, 2] {
            let dir = tempfile::tempdir().unwrap();
            let memory = MemoryBackend::with_test_config(TestConfig::default());
            let runtime = runtime(memory.clone());
            let (job, _) = reservation(&runtime, &dir, 19).await;
            let mut opts = worker_options(&dir);
            opts.parallelism = 1;
            let mut plan = vec![CommitFault::Normal; position];
            plan.push(fault);
            memory.set_fault_plan(plan).unwrap();
            let first = job.run_worker(opts.clone()).await;
            if position == 0 {
                assert_eq!(first.unwrap_err().kind(), ErrorKind::CommitOutcomeUnknown);
                job.run_worker(opts.clone()).await.unwrap();
            } else {
                first.unwrap();
            }
            let index = job.publish().await.unwrap();
            assert!(
                index
                    .verify(VerifyOptions::default())
                    .await
                    .unwrap()
                    .issues
                    .is_empty()
            );
            assert_eq!(artifact_directories(&opts.workspace), 0);
            runtime.shutdown().await.unwrap();
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn abort_wins_against_a_prepared_publication_transaction() {
    let dir = tempfile::tempdir().unwrap();
    let memory = MemoryBackend::with_test_config(TestConfig::default());
    let gate = Arc::new(Gate::default());
    let runtime = Runtime::new(
        Gated {
            memory: memory.clone(),
            gate: gate.clone(),
        },
        RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
    )
    .unwrap();
    let (job, _) = reservation(&runtime, &dir, 19).await;
    let opts = worker_options(&dir);
    job.run_worker(opts.clone()).await.unwrap();
    gate.arm(3); // begin_validation + 1 validation slice -> hold activate commit
    let publisher = job.clone();
    let pending = tokio::spawn(async move { publisher.publish().await });
    gate.wait().await;
    job.abort().await.unwrap();
    gate.release.notify_one();
    assert_eq!(
        pending.await.unwrap().unwrap_err().kind(),
        ErrorKind::IndexNotFound
    );
    assert_eq!(job.status().await.unwrap(), BulkBuildStatus::Aborted);
    let _ = runtime.cleanup_bulk_builds(10, None).await.unwrap();
    assert_eq!(artifact_directories(&opts.workspace), 0);
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn preparation_failure_is_durable_and_partial_files_are_owned_until_abort() {
    let dir = tempfile::tempdir().unwrap();
    let memory = MemoryBackend::new();
    let runtime = runtime(memory.clone());
    let (job, _) = reservation(&runtime, &dir, 19).await;
    let mut opts = worker_options(&dir);
    opts.memory_bytes = 8 * 1024; // trigger disk mode
    opts.scratch_bytes = 100; // fail scratch quota
    assert_eq!(
        job.run_worker(opts.clone()).await.unwrap_err().kind(),
        ErrorKind::LimitExceeded
    );
    assert_eq!(
        runtime
            .open_bulk_build("bulk")
            .await
            .unwrap()
            .status()
            .await
            .unwrap(),
        BulkBuildStatus::Failed {
            kind: ErrorKind::LimitExceeded
        }
    );
    assert_eq!(
        job.publish().await.unwrap_err().kind(),
        ErrorKind::LimitExceeded
    );
    assert_eq!(artifact_directories(&opts.workspace), 1);
    job.abort().await.unwrap();
    assert_eq!(artifact_directories(&opts.workspace), 0);
    assert!(dir.path().join("source/manifest.bin").exists());
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cleanup_outcome_unknown_and_pagination_are_recoverable_after_drop() {
    let dir = tempfile::tempdir().unwrap();
    let memory = MemoryBackend::with_test_config(TestConfig::default());
    let runtime = runtime(memory.clone());
    let (job, _) = reservation(&runtime, &dir, 19).await;
    let mut opts = worker_options(&dir);
    opts.memory_bytes = 8 * 1024; // trigger disk mode so owned scratch files exist before cleanup
    job.run_worker(opts.clone()).await.unwrap();
    runtime.drop_index("bulk").await.unwrap();
    assert_eq!(artifact_directories(&opts.workspace), 1);
    memory
        .set_fault_plan(vec![CommitFault::UnknownApplied])
        .unwrap();
    let page = runtime.cleanup_bulk_builds(1, None).await.unwrap();
    assert_eq!(page.reclaimed, 1);
    assert_eq!(page.pending, 0);
    assert_eq!(artifact_directories(&opts.workspace), 0);
    assert_eq!(
        runtime
            .cleanup_bulk_builds(1, page.next)
            .await
            .unwrap()
            .reclaimed,
        0
    );
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_publishers_resolve_the_same_active_identity() {
    let dir = tempfile::tempdir().unwrap();
    let memory = MemoryBackend::with_test_config(TestConfig::default());
    let gate = Arc::new(Gate::default());
    let runtime = Runtime::new(
        Gated {
            memory: memory.clone(),
            gate: gate.clone(),
        },
        RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
    )
    .unwrap();
    let (job, _) = reservation(&runtime, &dir, 19).await;
    job.run_worker(worker_options(&dir)).await.unwrap();
    gate.arm(2);
    let other = job.clone();
    let pending = tokio::spawn(async move { other.publish().await });
    gate.wait().await;
    let first = job.publish().await.unwrap();
    gate.release.notify_one();
    let second = pending.await.unwrap().unwrap();
    assert_eq!(first.logical_index_id(), second.logical_index_id());
    job.cleanup().await.unwrap();
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn single_worker_takeover_fences_stale_writer_lanes() {
    let dir = tempfile::tempdir().unwrap();
    let memory = MemoryBackend::with_test_config(TestConfig::default());
    let gate = Arc::new(Gate::default());
    let runtime = Runtime::new(
        Gated {
            memory: memory.clone(),
            gate: gate.clone(),
        },
        RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
    )
    .unwrap();
    let (job, _) = reservation(&runtime, &dir, 19).await;
    let mut opts = worker_options(&dir);
    opts.parallelism = 1;
    gate.arm(2);
    let old = job.clone();
    let old_options = opts.clone();
    let pending = tokio::spawn(async move { old.run_worker(old_options).await });
    gate.wait().await;
    job.run_worker(opts.clone()).await.unwrap();
    gate.release.notify_one();
    assert_eq!(
        pending.await.unwrap().unwrap_err().kind(),
        ErrorKind::BulkBuildSuperseded
    );
    let index = job.publish().await.unwrap();
    let verified = index.verify(VerifyOptions::default()).await.unwrap();
    assert!(verified.complete && verified.issues.is_empty());
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn premature_publish_does_not_fail_an_in_progress_load() {
    for commit in [1, 2, 3] {
        let dir = tempfile::tempdir().unwrap();
        let memory = MemoryBackend::with_test_config(TestConfig::default());
        let gate = Arc::new(Gate::default());
        let runtime = Runtime::new(
            Gated {
                memory: memory.clone(),
                gate: gate.clone(),
            },
            RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
        )
        .unwrap();
        let (job, _) = reservation(&runtime, &dir, 19).await;
        assert_eq!(
            job.publish().await.unwrap_err().kind(),
            ErrorKind::BulkBuildBusy
        );
        let mut opts = worker_options(&dir);
        opts.parallelism = 1;
        gate.arm(commit);
        let worker = job.clone();
        let pending = tokio::spawn(async move { worker.run_worker(opts).await });
        gate.wait().await;
        assert_eq!(
            job.publish().await.unwrap_err().kind(),
            ErrorKind::BulkBuildBusy
        );
        assert!(!matches!(
            job.status().await.unwrap(),
            BulkBuildStatus::Failed { .. }
        ));
        gate.release.notify_one();
        pending.await.unwrap().unwrap();
        let index = job.publish().await.unwrap();
        assert!(
            index
                .verify(VerifyOptions::default())
                .await
                .unwrap()
                .issues
                .is_empty()
        );
        runtime.shutdown().await.unwrap();
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn complete_resumes_after_concurrent_publication_and_reclaimed_paths() {
    let dir = tempfile::tempdir().unwrap();
    let memory = MemoryBackend::with_test_config(TestConfig::default());
    let gate = Arc::new(Gate::default());
    let runtime = Runtime::new(
        Gated {
            memory: memory.clone(),
            gate: gate.clone(),
        },
        RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
    )
    .unwrap();
    let (job, _) = reservation(&runtime, &dir, 73).await;
    let mut options = worker_options(&dir);
    options.parallelism = 1;
    gate.arm(1);
    let worker = job.clone();
    let first_options = options.clone();
    let pending = tokio::spawn(async move { worker.complete(first_options).await });
    gate.wait().await;
    let first = job.complete(options.clone()).await.unwrap();
    gate.release.notify_one();
    let second = pending.await.unwrap().unwrap();
    assert_eq!(first.logical_index_id(), second.logical_index_id());
    fs::remove_dir_all(dir.path().join("source")).unwrap();
    fs::remove_dir_all(&options.workspace).unwrap();
    let recovered = job.complete(options).await.unwrap();
    assert_eq!(recovered.logical_index_id(), first.logical_index_id());
    let report = recovered.verify(VerifyOptions::default()).await.unwrap();
    assert!(report.complete && report.issues.is_empty());
    assert_eq!(report.objects.vector_records, 73);
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn completed_validation_publishes_without_workspace_files() {
    let dir = tempfile::tempdir().unwrap();
    let memory = MemoryBackend::with_test_config(TestConfig::default());
    let gate = Arc::new(Gate::default());
    let runtime = Runtime::new(
        Gated {
            memory: memory.clone(),
            gate: gate.clone(),
        },
        RuntimeConfig::default().with_maintenance(0, 1).unwrap(),
    )
    .unwrap();
    let (job, _) = reservation(&runtime, &dir, 0).await;
    let options = worker_options(&dir);
    job.run_worker(options.clone()).await.unwrap();
    gate.arm(3); // registration + 1 validation slice precede activation
    let worker = job.clone();
    let cancellation = CancellationToken::new();
    let control = OperationOptions::default().with_cancellation(cancellation.clone());
    let pending = tokio::spawn(async move { worker.publish_with_control(control).await });
    gate.wait().await;
    memory.set_fault_plan(vec![CommitFault::Abort]).unwrap();
    cancellation.cancel();
    gate.release.notify_one();
    assert_eq!(
        pending.await.unwrap().unwrap_err().kind(),
        ErrorKind::Cancelled
    );
    assert!(matches!(
        job.status().await.unwrap(),
        BulkBuildStatus::Validating { .. }
    ));
    for entry in fs::read_dir(&options.workspace).unwrap() {
        let entry = entry.unwrap();
        if entry.file_type().unwrap().is_dir() {
            fs::remove_dir_all(entry.path()).unwrap();
        }
    }
    let index = job.complete(options).await.unwrap();
    assert!(
        index
            .verify(VerifyOptions::default())
            .await
            .unwrap()
            .issues
            .is_empty()
    );
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn collaborative_workers_execute_disjoint_units_and_reclaim_crashed_leases() {
    let dir = tempfile::tempdir().unwrap();
    let memory = MemoryBackend::new();
    let rt1 = runtime(memory.clone());
    let rt2 = runtime(memory.clone());
    let rt3 = runtime(memory.clone());

    let input = InputSnapshot::create(
        &dir.path().join("source"),
        config(Metric::Cosine, false),
        16 << 20,
        (0..250).map(|id| Ok(record(id))),
    )
    .unwrap();
    let job1 = rt1
        .start_bulk_build(
            "collab",
            &input,
            ktann::api::BulkBuildOptions {
                refinement_rounds: Some(1),
                ..construction_options()
            },
        )
        .await
        .unwrap();
    let job2 = rt2.open_bulk_build("collab").await.unwrap();
    let job3 = rt3.open_bulk_build("collab").await.unwrap();

    let mut opts = worker_options(&dir);
    opts.distributed_workers = 3;
    opts.parallelism = 2;
    opts.unit_lease = Duration::from_millis(80);
    opts.load = BulkLoadOptions::default();

    // Start worker 1, cancel it mid-flight to leave leased units, and let workers 2 & 3 finish!
    let cancel = CancellationToken::new();
    let c_tok = cancel.clone();
    let o1 = opts.clone();
    let w1 = tokio::spawn(async move {
        let ctrl = OperationOptions::default().with_cancellation(c_tok);
        let _ = job1.run_worker_with_control(o1, ctrl).await;
    });
    tokio::time::sleep(Duration::from_millis(15)).await;
    cancel.cancel();
    let _ = w1.await;

    let o2 = opts.clone();
    let o3 = opts.clone();
    let (r2, r3) = tokio::join!(job2.complete(o2), job3.complete(o3));
    let idx = r2.unwrap();
    assert_eq!(idx.logical_index_id(), r3.unwrap().logical_index_id());
    let verified = idx.verify(VerifyOptions::default()).await.unwrap();
    assert!(verified.complete && verified.issues.is_empty());
    assert_eq!(verified.objects.vector_records, 250);

    rt1.shutdown().await.unwrap();
    rt2.shutdown().await.unwrap();
    rt3.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unkeyed_build_reports_real_stage_metrics_and_reuses_published_index() {
    let dir = tempfile::tempdir().unwrap();
    let runtime = runtime(MemoryBackend::new());
    let input = InputSnapshot::create(
        &dir.path().join("source"),
        config(Metric::Cosine, false),
        4_000_000,
        (0..73).rev().map(|id| Ok(record(id))),
    )
    .unwrap();
    let job = runtime
        .start_bulk_build("bulk", &input, construction_options())
        .await
        .unwrap();
    let opts = worker_options(&dir);
    let (index, report) = job
        .complete_with_report(opts.clone(), Default::default())
        .await
        .unwrap();
    assert!(!report.ingest.is_zero() && !report.construct.is_zero());
    assert!(!report.emit.is_zero() && !report.publish.is_zero());
    assert!(report.in_memory_mode);
    assert_eq!(report.scratch_written_bytes, 0);
    assert_eq!(report.records, 73);
    assert_eq!(report.trees, 1);
    assert_eq!(report.rotation_seed, *job.index_manifest().rotation_seed());

    let (_, reused) = job
        .complete_with_report(opts, Default::default())
        .await
        .unwrap();
    assert!(reused.ingest.is_zero() && reused.construct.is_zero() && reused.emit.is_zero());

    let verified = index.verify(VerifyOptions::default()).await.unwrap();
    assert!(verified.complete && verified.issues.is_empty());
    assert_eq!(verified.objects.vector_records, 73);
    for id in 0..3 {
        let expected = record(id);
        let stored = index
            .get(expected.id().clone(), GetOptions::default().with_payload())
            .await
            .unwrap()
            .unwrap();
        assert_eq!(stored.vector(), expected.vector());
        assert_eq!(stored.fields(), expected.fields());
    }
    runtime.shutdown().await.unwrap();
}
