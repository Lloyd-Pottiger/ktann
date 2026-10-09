//! Complete worker/publication/reclamation recovery through public APIs.
use super::*;
use ktann::api::BulkWorkerOptions;

pub(super) fn worker_options(dir: &Directory) -> BulkWorkerOptions {
    let root = dir.0.join("work");
    fs::create_dir(&root).unwrap();
    let mut worker = BulkWorkerOptions::new(root);
    worker.sort_memory_bytes = options().sort_memory_bytes;
    worker.sort_scratch_bytes = options().sort_scratch_bytes;
    worker.serving_memory_bytes = 8 * 1024 * 1024;
    worker.serving_scratch_bytes = 64 * 1024 * 1024;
    worker.max_artifact_bytes = 4 * 1024 * 1024;
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
        let dir = Directory::new();
        let memory = MemoryBackend::new();
        let first = runtime(memory.clone());
        let (job, _) = fixture(&first, &memory, &dir, count).await;
        let opts = worker_options(&dir);
        job.run_worker(opts.clone()).await.unwrap();
        let id = job.logical_index_id();
        assert!(matches!(
            job.status().await.unwrap(),
            BulkBuildStatus::Loaded { .. }
        ));
        first.shutdown().await.unwrap();
        let runtime = runtime(memory.clone());
        let job = runtime.open_bulk_build("bulk").await.unwrap();
        job.run_worker(opts.clone()).await.unwrap(); // accepted immutable outputs survive restart
        let index = job.publish().await.unwrap();
        assert_eq!(index.logical_index_id(), id);
        assert_eq!(job.status().await.unwrap(), BulkBuildStatus::Published);
        assert_eq!(artifact_directories(&opts.workspace), 0);
        assert!(dir.0.join("source/data.bin").exists());
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
    for (count, extra) in [(19, false), (19, true), (0, true)] {
        let dir = Directory::new();
        let memory = MemoryBackend::new();
        let runtime = runtime(memory.clone());
        let (job, _) = fixture(&runtime, &memory, &dir, count).await;
        let opts = worker_options(&dir);
        job.run_worker(opts.clone()).await.unwrap();
        let mut txn = memory.begin_write().await.unwrap();
        if extra {
            txn.put(
                Bytes::from(keys::record_key(job.logical_index_id(), record(999).id()).unwrap()),
                Bytes::from(
                    ValueCodec::for_index(job.index_manifest())
                        .encode(&PersistentValue::VectorRecord(
                            ktann::storage::values::VectorRecord::new(
                                record(999).id().clone(),
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
async fn cancelled_validation_reuses_proofs_and_rejects_further_loads() {
    let dir = Directory::new();
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
    let (job, artifact) = fixture(&runtime, &memory, &dir, 19).await;
    let opts = worker_options(&dir);
    job.run_worker(opts).await.unwrap();
    gate.arm(2);
    let worker = job.clone();
    let cancellation = CancellationToken::new();
    let control = OperationOptions::default().with_cancellation(cancellation.clone());
    let pending = tokio::spawn(async move { worker.publish_with_control(control).await });
    gate.wait().await;
    assert_eq!(
        job.load_serving(&artifact, load_options())
            .await
            .unwrap_err()
            .kind(),
        ErrorKind::InvalidArgument
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
        .publish()
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
async fn abort_during_native_preparation_leaves_recoverable_namespace_cleanup() {
    let dir = Directory::new();
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
    let (job, _) = fixture(&runtime, &memory, &dir, 19).await;
    let opts = worker_options(&dir);
    gate.arm(2);
    let worker = job.clone();
    let args = opts.clone();
    let pending = tokio::spawn(async move { worker.run_worker(args).await });
    gate.wait().await;
    job.abort().await.unwrap();
    assert_eq!(job.status().await.unwrap(), BulkBuildStatus::Aborted);
    assert_eq!(artifact_directories(&opts.workspace), 1);
    gate.release.notify_one();
    assert_eq!(
        pending.await.unwrap().unwrap_err().kind(),
        ErrorKind::IndexNotFound
    );
    let report = runtime.cleanup_bulk_builds(10, None).await.unwrap();
    assert_eq!(report.reclaimed, 1);
    assert_eq!(report.pending, 0);
    assert_eq!(report.next, None);
    assert_eq!(artifact_directories(&opts.workspace), 0);
    assert!(dir.0.join("source/data.bin").exists());
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
            let dir = Directory::new();
            let memory = MemoryBackend::with_test_config(TestConfig::default());
            let runtime = runtime(memory.clone());
            let (job, artifact) = fixture(&runtime, &memory, &dir, 19).await;
            job.run_worker(worker_options(&dir)).await.unwrap();
            let pages = (artifact.manifest().items() + 3)
                .div_ceil(load_options().max_mutations as u64) as usize;
            let position = if point == 2 { pages + 1 } else { point };
            let mut plan = vec![CommitFault::Normal; position];
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
async fn worker_claim_and_artifact_acceptance_unknown_outcomes_recover() {
    for fault in [CommitFault::UnknownApplied, CommitFault::UnknownNotApplied] {
        for position in [0, 1, 2] {
            let dir = Directory::new();
            let memory = MemoryBackend::with_test_config(TestConfig::default());
            let runtime = runtime(memory.clone());
            let (job, _) = fixture(&runtime, &memory, &dir, 19).await;
            let opts = worker_options(&dir);
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
    let dir = Directory::new();
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
    let (job, artifact) = fixture(&runtime, &memory, &dir, 19).await;
    let opts = worker_options(&dir);
    job.run_worker(opts.clone()).await.unwrap();
    let pages =
        (artifact.manifest().items() + 3).div_ceil(load_options().max_mutations as u64) as usize;
    gate.arm(pages + 2);
    let publisher = job.clone();
    let pending = tokio::spawn(async move { publisher.publish().await });
    gate.wait().await;
    assert!(matches!(
        checkpoint(&memory, &job).await.phase(),
        BuildPhase::Validated
    ));
    assert_eq!(
        job.status().await.unwrap(),
        BulkBuildStatus::Validating {
            verified_entries: artifact.manifest().items(),
            total_entries: artifact.manifest().items(),
        }
    );
    job.abort().await.unwrap();
    gate.release.notify_one();
    assert_eq!(
        pending.await.unwrap().unwrap_err().kind(),
        ErrorKind::IndexNotFound
    );
    assert_eq!(job.status().await.unwrap(), BulkBuildStatus::Aborted);
    assert_eq!(
        runtime
            .cleanup_bulk_builds(10, None)
            .await
            .unwrap()
            .reclaimed,
        1
    );
    assert_eq!(artifact_directories(&opts.workspace), 0);
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn preparation_failure_is_durable_and_partial_files_are_owned_until_abort() {
    let dir = Directory::new();
    let memory = MemoryBackend::new();
    let runtime = runtime(memory.clone());
    let (job, _) = fixture(&runtime, &memory, &dir, 19).await;
    let mut opts = worker_options(&dir);
    opts.max_artifact_bytes = 100;
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
    assert!(dir.0.join("source/data.bin").exists());
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cleanup_outcome_unknown_and_pagination_are_recoverable_after_drop() {
    let dir = Directory::new();
    let memory = MemoryBackend::with_test_config(TestConfig::default());
    let runtime = runtime(memory.clone());
    let (job, _) = fixture(&runtime, &memory, &dir, 19).await;
    let opts = worker_options(&dir);
    job.run_worker(opts.clone()).await.unwrap();
    // The ordinary drop path removes index data but retains the cleanup ledger.
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
    let dir = Directory::new();
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
    let (job, _) = fixture(&runtime, &memory, &dir, 19).await;
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
async fn preparation_takeover_fences_the_old_artifact_acceptance() {
    let dir = Directory::new();
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
    let (job, _) = fixture(&runtime, &memory, &dir, 19).await;
    let opts = worker_options(&dir);
    gate.arm(2);
    let old = job.clone();
    let old_options = opts.clone();
    let pending = tokio::spawn(async move { old.run_worker(old_options).await });
    gate.wait().await;
    job.run_worker(opts).await.unwrap();
    gate.release.notify_one();
    assert_eq!(
        pending.await.unwrap().unwrap_err().kind(),
        ErrorKind::BulkBuildSuperseded
    );
    job.publish().await.unwrap();
    runtime.shutdown().await.unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn premature_publish_does_not_fail_an_in_progress_load() {
    for commit in [4, 5] {
        let dir = Directory::new();
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
        let (job, _) = fixture(&runtime, &memory, &dir, 19).await;
        let opts = worker_options(&dir);
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
async fn prepared_receipt_and_lost_preparation_both_publish_complete_indexes() {
    use ktann::bulk::PreparedInputWriter;
    for mode in 0..3 {
        let dir = Directory::new();
        let memory = MemoryBackend::new();
        let runtime = runtime(memory);
        let mut opts = worker_options(&dir);
        opts.sort_memory_bytes *= 2;
        let mut forest = options();
        forest.sort_memory_bytes = opts.sort_memory_bytes;
        forest.sort_scratch_bytes = opts.sort_scratch_bytes;
        let prepared = PreparedInputWriter::new(
            &dir.0.join("source"),
            &dir.0.join("receipt"),
            config(Metric::L2, true),
            4_000_000,
            forest,
        )
        .unwrap()
        .append((0..73).map(|id| Ok(record(id))))
        .unwrap()
        .seal()
        .unwrap();
        let source = prepared.source().clone();
        let job = runtime
            .start_bulk_build("bulk", &source, forest.tree)
            .await
            .unwrap();
        let report = if mode == 0 {
            job.run_worker_with_prepared_input(opts.clone(), prepared, Default::default())
                .await
                .unwrap()
        } else {
            if mode == 1 {
                drop(prepared);
            } else {
                let mut mismatched = opts.clone();
                mismatched.sort_memory_bytes *= 2;
                assert_eq!(
                    job.run_worker_with_prepared_input(mismatched, prepared, Default::default())
                        .await
                        .unwrap_err()
                        .kind(),
                    ErrorKind::InvalidArgument
                );
            }
            fs::remove_dir_all(dir.0.join("receipt")).unwrap();
            job.run_worker(opts.clone()).await.unwrap()
        };
        assert!(!report.forest.is_zero());
        assert!(!report.serving.is_zero());
        assert!(!report.load.is_zero());
        let index = job.publish().await.unwrap();
        let verified = index.verify(VerifyOptions::default()).await.unwrap();
        assert!(verified.complete && verified.issues.is_empty());
        assert_eq!(verified.objects.vector_records, 73);
        runtime.shutdown().await.unwrap();
    }
}
