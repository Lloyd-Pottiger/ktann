//! Backend-neutral transaction contract tests.
//!
//! These tests drive a deterministic in-memory mock backend exclusively through
//! the public [`ktann::storage::backend`] seam. The backend models the contract
//! (snapshot isolation, read-your-writes, update-protected conflicts including
//! ABA, ordered scans and batches, unique insertion, rollback, commit outcomes,
//! hard limits and admission budgets, and the range-clear capability) without
//! depending on any production adapter. Resource bounds, fault injection,
//! replay history, and restart semantics use the test-backend-specific seams in
//! [`support`].

use std::future::Future;

use bytes::Bytes;
use ktann::api::ErrorKind;
use ktann::storage::backend::{
    AdmissionBudget, Backend, Capabilities, HardLimits, InsertOutcome, Mutation, ReadOps,
    ScanLimits, ScanPage, WriteTxn,
};
use ktann::storage::keys::KeyRange;

#[path = "support/backend_contract.rs"]
mod shared_backend_contract;
mod storage_operations;
#[allow(dead_code)]
mod support;

use shared_backend_contract::{BackendHarness, Fault, FaultInjection, RestartMode};
use support::{
    CommitFault, CommitOutcome, DeterministicBackend, DeterministicConfig, DeterministicReadTxn,
    DeterministicWriteTxn, Durability, HistoryEntry,
};

/// Drives an already-borrowing future to completion on a current-thread
/// runtime. The backend's futures never park, so this never blocks.
fn block_on<F: Future>(future: F) -> F::Output {
    tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("current-thread runtime builds")
        .block_on(future)
}

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

fn key(bytes: &'static [u8]) -> Bytes {
    Bytes::from_static(bytes)
}

fn range(start: &[u8], end: &[u8]) -> KeyRange {
    KeyRange::new(start.to_vec(), end.to_vec())
}

fn mock() -> DeterministicBackend {
    DeterministicBackend::new(DeterministicConfig::default())
}

fn mock_with_clear() -> DeterministicBackend {
    let config = DeterministicConfig {
        capabilities: Capabilities {
            transactional_clear_range: true,
        },
        ..Default::default()
    };
    DeterministicBackend::new(config)
}

fn seed(backend: &DeterministicBackend, entries: &[(&'static [u8], &'static [u8])]) {
    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        for (entry_key, entry_value) in entries {
            txn.put(key(entry_key), key(entry_value))
                .await
                .expect("seed put succeeds");
        }
        txn.commit().await.expect("seed commit succeeds");
    });
}

fn scanned_keys(page: &ScanPage) -> Vec<&[u8]> {
    page.items()
        .iter()
        .map(|item| item.key().as_ref())
        .collect()
}

fn outcomes(history: &[HistoryEntry]) -> Vec<CommitOutcome> {
    history.iter().map(|entry| entry.outcome).collect()
}

/// Adapts the deterministic test backend to the shared [`BackendHarness`] seam.
///
/// Controlled faults map one-to-one onto the backend's fault plan, and restart
/// reuses [`DeterministicBackend::reopen`], so the shared durability case
/// observes the configured [`Durability`].
struct DeterministicHarness {
    backend: DeterministicBackend,
    restart: RestartMode,
}

impl DeterministicHarness {
    fn new(config: DeterministicConfig) -> Self {
        let restart = match config.durability {
            Durability::Durable => RestartMode::Durable,
            Durability::Ephemeral => RestartMode::Ephemeral,
        };
        Self {
            backend: DeterministicBackend::new(config),
            restart,
        }
    }
}

impl BackendHarness for DeterministicHarness {
    type Backend = DeterministicBackend;

    fn backend(&self) -> &DeterministicBackend {
        &self.backend
    }

    fn fault_injection(&self) -> FaultInjection {
        FaultInjection::Controlled
    }

    fn inject_fault(&self, fault: Fault) {
        let commit_fault = match fault {
            Fault::Abort => CommitFault::Abort,
            Fault::UnknownApplied => CommitFault::UnknownApplied,
            Fault::UnknownNotApplied => CommitFault::UnknownNotApplied,
        };
        self.backend
            .set_fault_plan(vec![commit_fault])
            .expect("single-step fault plan fits");
    }

    fn restart_mode(&self) -> RestartMode {
        self.restart
    }

    fn restart(&self) -> Self {
        Self {
            backend: self.backend.reopen(),
            restart: self.restart,
        }
    }
}

// ---------------------------------------------------------------------------
// Compile-time interface shape.
// ---------------------------------------------------------------------------

#[test]
fn transaction_types_are_send() {
    fn assert_send<T: Send>() {}
    // `DeterministicReadTxn` and `DeterministicWriteTxn` are `Send` for any
    // backend lifetime. Their `PhantomData<Cell<()>>` field makes them `!Sync`,
    // and yet they still satisfy `ReadTxn`/`WriteTxn`, proving the interface
    // requires only `Send`.
    assert_send::<DeterministicReadTxn<'static>>();
    assert_send::<DeterministicWriteTxn<'static>>();
}

#[test]
fn deterministic_backend_runs_the_shared_contract_suite_durable() {
    let harness = DeterministicHarness::new(DeterministicConfig {
        durability: Durability::Durable,
        capabilities: Capabilities {
            transactional_clear_range: true,
        },
        ..Default::default()
    });
    block_on(shared_backend_contract::run_suite(&harness));
}

#[test]
fn deterministic_backend_runs_the_shared_contract_suite_ephemeral() {
    let harness = DeterministicHarness::new(DeterministicConfig {
        durability: Durability::Ephemeral,
        capabilities: Capabilities {
            transactional_clear_range: true,
        },
        ..Default::default()
    });
    block_on(shared_backend_contract::run_suite(&harness));
}

#[test]
fn deterministic_backend_declines_range_clear_in_the_shared_suite() {
    let harness = DeterministicHarness::new(DeterministicConfig {
        durability: Durability::Durable,
        capabilities: Capabilities {
            transactional_clear_range: false,
        },
        ..Default::default()
    });
    block_on(shared_backend_contract::run_suite(&harness));
}

#[test]
fn backend_declares_limits_budget_and_capabilities() {
    let backend = mock();
    assert_eq!(backend.hard_limits().max_key_bytes, 1_024);
    assert_eq!(backend.hard_limits().max_value_bytes, 4_096);
    assert_eq!(backend.admission_budget().max_mutations, 1_000);
    assert_eq!(backend.admission_budget().max_mutation_bytes, 1 << 20);
    assert!(!backend.capabilities().transactional_clear_range);
    assert!(mock_with_clear().capabilities().transactional_clear_range);
}

// ---------------------------------------------------------------------------
// Snapshot and point/batch reads.
// ---------------------------------------------------------------------------

#[test]
fn snapshot_consistency_isolates_later_commits() {
    let backend = mock();
    seed(&backend, &[(b"a", b"1")]);

    block_on(async {
        let mut reader = backend.begin_read().await.expect("begin read");
        // A later commit must not become visible to this open snapshot.
        let mut writer = backend.begin_write().await.expect("begin write");
        writer.put(key(b"a"), key(b"2")).await.expect("put");
        writer.put(key(b"b"), key(b"3")).await.expect("put");
        writer.commit().await.expect("commit");

        assert_eq!(reader.get(key(b"a")).await.expect("get"), Some(key(b"1")));
        assert_eq!(reader.get(key(b"b")).await.expect("get"), None);
    });
}

#[test]
fn long_lived_read_txn_observes_original_version() {
    let backend = mock();
    seed(&backend, &[(b"a", b"1")]);

    block_on(async {
        let mut reader = backend.begin_read().await.expect("begin read");
        // Several subsequent commits must leave the open snapshot untouched.
        for (entry_key, entry_value) in [(b"b", b"2"), (b"c", b"3"), (b"a", b"4")] {
            let mut writer = backend.begin_write().await.expect("begin write");
            writer
                .put(key(entry_key), key(entry_value))
                .await
                .expect("put");
            writer.commit().await.expect("commit");
        }

        assert_eq!(reader.get(key(b"a")).await.expect("get"), Some(key(b"1")));
        assert_eq!(reader.get(key(b"b")).await.expect("get"), None);
        assert_eq!(reader.get(key(b"c")).await.expect("get"), None);
    });
}

#[test]
fn write_transaction_reads_its_own_writes() {
    let backend = mock();
    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        assert_eq!(txn.get(key(b"a")).await.expect("get"), None);
        txn.put(key(b"a"), key(b"v")).await.expect("put");
        assert_eq!(txn.get(key(b"a")).await.expect("get"), Some(key(b"v")));
        txn.delete(key(b"a")).await.expect("delete");
        assert_eq!(txn.get(key(b"a")).await.expect("get"), None);
        txn.rollback().await;
    });
}

#[test]
fn batch_get_preserves_order_duplicates_and_empty_input() {
    let backend = mock();
    seed(&backend, &[(b"a", b"1"), (b"b", b"2")]);

    block_on(async {
        let mut txn = backend.begin_read().await.expect("begin read");

        let empty = txn.batch_get(Vec::new()).await.expect("empty batch");
        assert!(empty.is_empty());

        let result = txn
            .batch_get(vec![key(b"b"), key(b"a"), key(b"b"), key(b"missing")])
            .await
            .expect("batch get");
        assert_eq!(
            result,
            vec![Some(key(b"2")), Some(key(b"1")), Some(key(b"2")), None]
        );
    });
}

// ---------------------------------------------------------------------------
// Scans.
// ---------------------------------------------------------------------------

#[test]
fn scan_returns_sorted_items_in_half_open_range() {
    let backend = mock();
    seed(
        &backend,
        &[(b"c", b"3"), (b"a", b"1"), (b"b", b"2"), (b"d", b"4")],
    );

    block_on(async {
        let mut txn = backend.begin_read().await.expect("begin read");
        let limits = ScanLimits {
            item_limit: 100,
            byte_limit: 1_024,
        };
        let page = txn.scan(&range(b"a", b"d"), limits).await.expect("scan");
        assert_eq!(scanned_keys(&page), vec![&b"a"[..], &b"b"[..], &b"c"[..]]);
        assert!(page.next_start().is_none());
    });
}

#[test]
fn scan_paginates_with_strictly_advancing_next_start() {
    let backend = mock();
    seed(&backend, &[(b"a", b"1"), (b"b", b"2"), (b"c", b"3")]);

    block_on(async {
        let mut txn = backend.begin_read().await.expect("begin read");
        let limits = ScanLimits {
            item_limit: 2,
            byte_limit: 1_024,
        };

        let first = txn
            .scan(&range(b"a", b"\xff"), limits)
            .await
            .expect("first");
        assert_eq!(scanned_keys(&first), vec![&b"a"[..], &b"b"[..]]);
        let next = first.next_start().expect("non-terminal has cursor").clone();
        // The continuation is the byte-successor of the last key, so the next
        // page resumes without skipping or duplicating `b`.
        assert_eq!(next.as_ref(), b"b\x00");

        let second = txn
            .scan(&KeyRange::new(next.to_vec(), b"\xff".to_vec()), limits)
            .await
            .expect("second");
        assert_eq!(scanned_keys(&second), vec![&b"c"[..]]);
        assert!(second.next_start().is_none());
    });
}

#[test]
fn scan_accounts_bytes_and_allows_oversized_first_item() {
    let backend = mock();
    seed(&backend, &[(b"a", b"12345"), (b"b", b"67890")]);

    block_on(async {
        let mut txn = backend.begin_read().await.expect("begin read");

        // First item is 6 bytes; the second would make 12, past the limit.
        let page = txn
            .scan(
                &range(b"a", b"\xff"),
                ScanLimits {
                    item_limit: 100,
                    byte_limit: 6,
                },
            )
            .await
            .expect("scan");
        assert_eq!(page.items().len(), 1);
        assert_eq!(page.items()[0].key().as_ref(), b"a");
        assert_eq!(page.next_start().expect("more").as_ref(), b"a\x00");

        // A first item larger than the byte limit is returned alone.
        let page = txn
            .scan(
                &range(b"a", b"\xff"),
                ScanLimits {
                    item_limit: 100,
                    byte_limit: 4,
                },
            )
            .await
            .expect("scan");
        assert_eq!(page.items().len(), 1);
        assert_eq!(page.items()[0].value().as_ref(), b"12345");
        assert_eq!(page.next_start().expect("more").as_ref(), b"a\x00");
    });
}

#[test]
fn scan_resumes_after_a_key_at_the_backend_length_ceiling() {
    let config = DeterministicConfig {
        hard_limits: HardLimits {
            max_key_bytes: 2,
            max_value_bytes: 4_096,
        },
        ..Default::default()
    };
    let backend = DeterministicBackend::new(config);
    seed(&backend, &[(b"\x01\x01", b"1"), (b"\x01\x02", b"2")]);

    block_on(async {
        let mut txn = backend.begin_read().await.expect("begin read");
        let limits = ScanLimits {
            item_limit: 1,
            byte_limit: 1_024,
        };
        let first = txn
            .scan(&range(b"\x01\x01", b"\x02"), limits)
            .await
            .expect("first");
        assert_eq!(scanned_keys(&first), vec![&b"\x01\x01"[..]]);
        // The continuation stays within the 2-byte ceiling instead of appending
        // a byte and overflowing the limit on resume.
        let next = first.next_start().expect("non-terminal at ceiling").clone();
        assert_eq!(next.as_ref(), b"\x01\x02");

        let second = txn
            .scan(&KeyRange::new(next.to_vec(), b"\x02".to_vec()), limits)
            .await
            .expect("second page at the ceiling");
        assert_eq!(scanned_keys(&second), vec![&b"\x01\x02"[..]]);
        assert!(second.next_start().is_none());
    });
}

#[test]
fn scan_rejects_zero_limits_before_work() {
    let backend = mock();
    seed(&backend, &[(b"a", b"1")]);

    block_on(async {
        let mut txn = backend.begin_read().await.expect("begin read");
        let zero_items = txn
            .scan(
                &range(b"a", b"\xff"),
                ScanLimits {
                    item_limit: 0,
                    byte_limit: 10,
                },
            )
            .await
            .expect_err("zero item limit");
        assert_eq!(zero_items.kind(), ErrorKind::InvalidArgument);

        let zero_bytes = txn
            .scan(
                &range(b"a", b"\xff"),
                ScanLimits {
                    item_limit: 10,
                    byte_limit: 0,
                },
            )
            .await
            .expect_err("zero byte limit");
        assert_eq!(zero_bytes.kind(), ErrorKind::InvalidArgument);
    });
}

#[test]
fn scan_empty_range_returns_empty_page() {
    let backend = mock();
    seed(&backend, &[(b"a", b"1")]);

    block_on(async {
        let mut txn = backend.begin_read().await.expect("begin read");
        let page = txn
            .scan(
                &range(b"b", b"a"),
                ScanLimits {
                    item_limit: 10,
                    byte_limit: 10,
                },
            )
            .await
            .expect("scan");
        assert!(page.items().is_empty());
        assert!(page.next_start().is_none());
    });
}

#[test]
fn scan_page_item_cap_is_enforced() {
    let config = DeterministicConfig {
        max_scan_page_items: 1,
        ..Default::default()
    };
    let backend = DeterministicBackend::new(config);
    seed(&backend, &[(b"a", b"1"), (b"b", b"2")]);

    block_on(async {
        let mut txn = backend.begin_read().await.expect("begin read");
        let page = txn
            .scan(
                &range(b"a", b"\xff"),
                ScanLimits {
                    item_limit: 100,
                    byte_limit: 1_024,
                },
            )
            .await
            .expect("scan");
        assert_eq!(page.items().len(), 1);
        assert_eq!(page.items()[0].key().as_ref(), b"a");
        assert!(page.next_start().is_some());
    });
}

#[test]
fn scan_page_byte_cap_is_enforced() {
    let config = DeterministicConfig {
        max_scan_page_bytes: 2,
        ..Default::default()
    };
    let backend = DeterministicBackend::new(config);
    seed(&backend, &[(b"a", b"1"), (b"b", b"2")]);

    block_on(async {
        let mut txn = backend.begin_read().await.expect("begin read");
        let page = txn
            .scan(
                &range(b"a", b"\xff"),
                ScanLimits {
                    item_limit: 100,
                    byte_limit: 1_024,
                },
            )
            .await
            .expect("scan");
        assert_eq!(page.items().len(), 1);
        assert_eq!(page.items()[0].key().as_ref(), b"a");
    });
}

// ---------------------------------------------------------------------------
// Mutations.
// ---------------------------------------------------------------------------

#[test]
fn batch_mutate_applies_in_input_order() {
    let backend = mock();
    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        txn.batch_mutate(vec![
            Mutation::Put {
                key: key(b"a"),
                value: key(b"1"),
            },
            Mutation::Put {
                key: key(b"a"),
                value: key(b"2"),
            },
            Mutation::Delete { key: key(b"a") },
            Mutation::Put {
                key: key(b"a"),
                value: key(b"3"),
            },
        ])
        .await
        .expect("batch mutate");
        assert_eq!(txn.get(key(b"a")).await.expect("get"), Some(key(b"3")));
        txn.commit().await.expect("commit");
    });

    block_on(async {
        let mut txn = backend.begin_read().await.expect("begin read");
        assert_eq!(txn.get(key(b"a")).await.expect("get"), Some(key(b"3")));
    });
}

#[test]
fn empty_batch_mutate_succeeds() {
    let backend = mock();
    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        txn.batch_mutate(Vec::new()).await.expect("empty batch");
        txn.commit().await.expect("commit");
    });
}

#[test]
fn batch_mutate_capacity_failure_leaves_no_partial_state() {
    let backend = DeterministicBackend::new(DeterministicConfig::new(
        HardLimits {
            max_key_bytes: 1_024,
            max_value_bytes: 1_024,
        },
        AdmissionBudget {
            max_mutations: 2,
            max_mutation_bytes: 1 << 20,
        },
        Capabilities {
            transactional_clear_range: false,
        },
    ));

    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        let error = txn
            .batch_mutate(vec![
                Mutation::Put {
                    key: key(b"a"),
                    value: key(b"1"),
                },
                Mutation::Put {
                    key: key(b"b"),
                    value: key(b"2"),
                },
                Mutation::Put {
                    key: key(b"c"),
                    value: key(b"3"),
                },
            ])
            .await
            .expect_err("exceeds mutation budget");
        assert_eq!(error.kind(), ErrorKind::LimitExceeded);

        // Nothing was applied, and the transaction is still usable.
        assert_eq!(txn.get(key(b"a")).await.expect("get"), None);
        assert_eq!(txn.get(key(b"b")).await.expect("get"), None);
        txn.batch_mutate(vec![Mutation::Put {
            key: key(b"a"),
            value: key(b"1"),
        }])
        .await
        .expect("single mutation fits");
        txn.commit().await.expect("commit");
    });
}

#[test]
fn write_txn_scan_sees_its_own_mutations() {
    let backend = mock();
    seed(&backend, &[(b"a", b"1"), (b"b", b"2"), (b"c", b"3")]);

    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        txn.put(key(b"x"), key(b"9")).await.expect("put");
        txn.delete(key(b"b")).await.expect("delete");
        let page = txn
            .scan(
                &range(b"a", b"\xff"),
                ScanLimits {
                    item_limit: 100,
                    byte_limit: 1_024,
                },
            )
            .await
            .expect("scan");
        assert_eq!(scanned_keys(&page), vec![&b"a"[..], &b"c"[..], &b"x"[..]]);
        txn.rollback().await;
    });
}

#[test]
fn unique_insert_distinguishes_inserted_from_existing() {
    let backend = mock();
    seed(&backend, &[(b"existing", b"old")]);

    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        assert_eq!(
            txn.insert(key(b"fresh"), key(b"v")).await.expect("insert"),
            InsertOutcome::Inserted
        );
        assert_eq!(
            txn.insert(key(b"existing"), key(b"v"))
                .await
                .expect("insert"),
            InsertOutcome::AlreadyExists
        );
        // The existing value is left unchanged.
        assert_eq!(
            txn.get(key(b"existing")).await.expect("get"),
            Some(key(b"old"))
        );
        // Re-inserting the just-inserted key within the same txn sees it.
        assert_eq!(
            txn.insert(key(b"fresh"), key(b"again"))
                .await
                .expect("insert"),
            InsertOutcome::AlreadyExists
        );
        txn.commit().await.expect("commit");
    });
}

#[test]
fn update_protected_read_conflict_aborts() {
    let backend = mock();
    block_on(async {
        let mut first = backend.begin_write().await.expect("begin write");
        let mut second = backend.begin_write().await.expect("begin write");

        first.get_for_update(key(b"k")).await.expect("read");
        second.get_for_update(key(b"k")).await.expect("read");

        first.put(key(b"k"), key(b"1")).await.expect("put");
        first.commit().await.expect("first commit wins");

        second.put(key(b"k"), key(b"2")).await.expect("put");
        let conflict = second.commit().await.expect_err("conflict");
        assert_eq!(conflict.kind(), ErrorKind::RetryableAbort);
    });

    block_on(async {
        let mut txn = backend.begin_read().await.expect("begin read");
        assert_eq!(txn.get(key(b"k")).await.expect("get"), Some(key(b"1")));
    });
}

#[test]
fn key_restored_to_original_value_still_conflicts() {
    let backend = mock();
    block_on(async {
        // Reads the key absent at version 0.
        let mut reader = backend.begin_write().await.expect("begin write");
        reader.get_for_update(key(b"k")).await.expect("read");

        // A write then a delete restore the original (absent) value.
        let mut first = backend.begin_write().await.expect("begin write");
        first.put(key(b"k"), key(b"1")).await.expect("put");
        first.commit().await.expect("commit");

        let mut second = backend.begin_write().await.expect("begin write");
        second.delete(key(b"k")).await.expect("delete");
        second.commit().await.expect("commit");

        // The value matches the reader's snapshot, but the version changed.
        reader.put(key(b"k"), key(b"2")).await.expect("put");
        let error = reader.commit().await.expect_err("ABA conflict");
        assert_eq!(error.kind(), ErrorKind::RetryableAbort);
    });
}

#[test]
fn concurrent_unique_insert_does_not_silently_overwrite() {
    let backend = mock();
    block_on(async {
        let mut first = backend.begin_write().await.expect("begin write");
        let mut second = backend.begin_write().await.expect("begin write");

        assert_eq!(
            first.insert(key(b"k"), key(b"1")).await.expect("insert"),
            InsertOutcome::Inserted
        );
        assert_eq!(
            second.insert(key(b"k"), key(b"2")).await.expect("insert"),
            InsertOutcome::Inserted
        );

        first.commit().await.expect("first commit wins");
        let error = second.commit().await.expect_err("second insert conflicts");
        assert_eq!(error.kind(), ErrorKind::RetryableAbort);
    });

    block_on(async {
        let mut txn = backend.begin_read().await.expect("begin read");
        assert_eq!(txn.get(key(b"k")).await.expect("get"), Some(key(b"1")));
    });
}

// ---------------------------------------------------------------------------
// Commit outcomes and rollback.
// ---------------------------------------------------------------------------

#[test]
fn rollback_persists_nothing() {
    let backend = mock();
    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        txn.put(key(b"a"), key(b"v")).await.expect("put");
        txn.rollback().await;
    });

    block_on(async {
        let mut txn = backend.begin_read().await.expect("begin read");
        assert_eq!(txn.get(key(b"a")).await.expect("get"), None);
    });
}

#[test]
fn dropping_a_transaction_persists_nothing() {
    let backend = mock();
    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        txn.put(key(b"a"), key(b"v")).await.expect("put");
        drop(txn);
    });

    block_on(async {
        let mut txn = backend.begin_read().await.expect("begin read");
        assert_eq!(txn.get(key(b"a")).await.expect("get"), None);
    });
}

#[test]
fn commit_success_is_visible_to_subsequent_snapshots() {
    let backend = mock();
    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        txn.put(key(b"a"), key(b"v")).await.expect("put");
        txn.commit().await.expect("commit succeeds");
    });

    block_on(async {
        let mut txn = backend.begin_read().await.expect("begin read");
        assert_eq!(txn.get(key(b"a")).await.expect("get"), Some(key(b"v")));
    });
}

#[test]
fn commit_installs_complete_new_version_never_partial() {
    let backend = mock();
    seed(&backend, &[(b"a", b"1"), (b"b", b"2")]);

    block_on(async {
        let mut reader = backend.begin_read().await.expect("begin read");
        let mut writer = backend.begin_write().await.expect("begin write");
        writer.put(key(b"a"), key(b"x")).await.expect("put");
        writer.put(key(b"b"), key(b"y")).await.expect("put");
        writer.commit().await.expect("commit");

        // The pre-commit reader sees the complete old version, never a mix.
        assert_eq!(reader.get(key(b"a")).await.expect("get"), Some(key(b"1")));
        assert_eq!(reader.get(key(b"b")).await.expect("get"), Some(key(b"2")));

        // A post-commit reader sees the complete new version.
        let mut after = backend.begin_read().await.expect("begin read");
        assert_eq!(after.get(key(b"a")).await.expect("get"), Some(key(b"x")));
        assert_eq!(after.get(key(b"b")).await.expect("get"), Some(key(b"y")));
    });
}

#[test]
fn commit_failure_and_unknown_outcome_are_distinct() {
    let backend = mock();

    // Definite failure: an injected fault reports a retryable abort.
    backend.push_fault(CommitFault::Abort).expect("fault");
    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        txn.put(key(b"a"), key(b"v")).await.expect("put");
        let error = txn.commit().await.expect_err("faulted commit fails");
        assert_eq!(error.kind(), ErrorKind::RetryableAbort);
    });

    // Unknown outcome: the mutation lands but the result is reported unknown.
    backend
        .push_fault(CommitFault::UnknownApplied)
        .expect("fault");
    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        txn.put(key(b"b"), key(b"v")).await.expect("put");
        let error = txn.commit().await.expect_err("faulted commit is unknown");
        assert_eq!(error.kind(), ErrorKind::CommitOutcomeUnknown);
    });
}

#[test]
fn unknown_applied_persists_all_mutations_atomically() {
    let backend = mock();
    backend
        .push_fault(CommitFault::UnknownApplied)
        .expect("fault");

    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        txn.put(key(b"a"), key(b"1")).await.expect("put");
        txn.put(key(b"b"), key(b"2")).await.expect("put");
        let error = txn.commit().await.expect_err("unknown outcome");
        assert_eq!(error.kind(), ErrorKind::CommitOutcomeUnknown);
    });

    block_on(async {
        let mut txn = backend.begin_read().await.expect("begin read");
        assert_eq!(txn.get(key(b"a")).await.expect("get"), Some(key(b"1")));
        assert_eq!(txn.get(key(b"b")).await.expect("get"), Some(key(b"2")));
    });
}

#[test]
fn unknown_not_applied_persists_nothing() {
    let backend = mock();
    backend
        .push_fault(CommitFault::UnknownNotApplied)
        .expect("fault");

    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        txn.put(key(b"a"), key(b"1")).await.expect("put");
        let error = txn.commit().await.expect_err("unknown outcome");
        assert_eq!(error.kind(), ErrorKind::CommitOutcomeUnknown);
    });

    assert_eq!(backend.db_key_count(), 0);
    block_on(async {
        let mut txn = backend.begin_read().await.expect("begin read");
        assert_eq!(txn.get(key(b"a")).await.expect("get"), None);
    });
}

// ---------------------------------------------------------------------------
// Hard limits and admission budgets.
// ---------------------------------------------------------------------------

#[test]
fn hard_limit_rejects_oversized_key_and_value() {
    let backend = DeterministicBackend::new(DeterministicConfig::new(
        HardLimits {
            max_key_bytes: 2,
            max_value_bytes: 2,
        },
        AdmissionBudget {
            max_mutations: 100,
            max_mutation_bytes: 1 << 20,
        },
        Capabilities {
            transactional_clear_range: false,
        },
    ));

    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        let error = txn
            .put(key(b"abc"), key(b"v"))
            .await
            .expect_err("oversized key");
        assert_eq!(error.kind(), ErrorKind::LimitExceeded);

        let error = txn
            .put(key(b"a"), key(b"vvv"))
            .await
            .expect_err("oversized value");
        assert_eq!(error.kind(), ErrorKind::LimitExceeded);
    });
}

#[test]
fn admission_budget_rejects_excess_mutations_and_bytes() {
    let backend = DeterministicBackend::new(DeterministicConfig::new(
        HardLimits {
            max_key_bytes: 1_024,
            max_value_bytes: 1_024,
        },
        AdmissionBudget {
            max_mutations: 2,
            max_mutation_bytes: 4,
        },
        Capabilities {
            transactional_clear_range: false,
        },
    ));

    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        txn.put(key(b"a"), key(b"1")).await.expect("first fits");
        txn.put(key(b"b"), key(b"2")).await.expect("second fits");
        let error = txn
            .put(key(b"c"), key(b"3"))
            .await
            .expect_err("exceeds mutation count");
        assert_eq!(error.kind(), ErrorKind::LimitExceeded);
    });

    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        // A 1-byte key plus 5-byte value overflows the four-byte budget.
        let error = txn
            .put(key(b"a"), key(b"12345"))
            .await
            .expect_err("exceeds mutation bytes");
        assert_eq!(error.kind(), ErrorKind::LimitExceeded);
    });
}

#[test]
fn active_transaction_limit_is_enforced() {
    let config = DeterministicConfig {
        max_active_transactions: 1,
        ..Default::default()
    };
    let backend = DeterministicBackend::new(config);

    block_on(async {
        let open = backend.begin_read().await.expect("first admits");
        assert_eq!(backend.active_transactions(), 1);
        let error = backend
            .begin_write()
            .await
            .expect_err("second exceeds active limit");
        assert_eq!(error.kind(), ErrorKind::LimitExceeded);

        drop(open);
        assert_eq!(backend.active_transactions(), 0);
        backend
            .begin_write()
            .await
            .expect("third admits after release");
    });
}

#[test]
fn read_set_limit_is_enforced() {
    let config = DeterministicConfig {
        max_read_set: 1,
        ..Default::default()
    };
    let backend = DeterministicBackend::new(config);

    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        txn.get_for_update(key(b"a")).await.expect("first read");
        let error = txn
            .get_for_update(key(b"b"))
            .await
            .expect_err("read set full");
        assert_eq!(error.kind(), ErrorKind::LimitExceeded);
    });
}

#[test]
fn mutation_buffer_limit_is_enforced() {
    let config = DeterministicConfig {
        max_mutation_buffer: 1,
        ..Default::default()
    };
    let backend = DeterministicBackend::new(config);

    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        txn.put(key(b"a"), key(b"1")).await.expect("first key");
        let error = txn
            .put(key(b"b"), key(b"2"))
            .await
            .expect_err("buffer full");
        assert_eq!(error.kind(), ErrorKind::LimitExceeded);
    });
}

#[test]
fn batch_size_limit_is_enforced() {
    let config = DeterministicConfig {
        max_batch_size: 2,
        ..Default::default()
    };
    let backend = DeterministicBackend::new(config);

    block_on(async {
        let mut txn = backend.begin_read().await.expect("begin read");
        let error = txn
            .batch_get(vec![key(b"a"), key(b"b"), key(b"c")])
            .await
            .expect_err("batch too large");
        assert_eq!(error.kind(), ErrorKind::LimitExceeded);
    });
}

#[test]
fn db_key_limit_is_enforced() {
    let config = DeterministicConfig {
        max_db_keys: 1,
        ..Default::default()
    };
    let backend = DeterministicBackend::new(config);
    seed(&backend, &[(b"a", b"1")]);

    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        txn.put(key(b"b"), key(b"2")).await.expect("put accepted");
        let error = txn.commit().await.expect_err("exceeds db key limit");
        assert_eq!(error.kind(), ErrorKind::LimitExceeded);
    });

    assert_eq!(backend.db_key_count(), 1);
}

#[test]
fn db_byte_limit_is_enforced() {
    let config = DeterministicConfig {
        max_db_bytes: 4,
        ..Default::default()
    };
    let backend = DeterministicBackend::new(config);
    seed(&backend, &[(b"a", b"1"), (b"b", b"2")]);

    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        txn.put(key(b"c"), key(b"3")).await.expect("put accepted");
        let error = txn.commit().await.expect_err("exceeds db byte limit");
        assert_eq!(error.kind(), ErrorKind::LimitExceeded);
    });

    assert_eq!(backend.db_byte_count(), 4);
}

#[test]
fn retained_version_eviction_rejects_too_old_commit() {
    let config = DeterministicConfig {
        max_retained_versions: 1,
        ..Default::default()
    };
    let backend = DeterministicBackend::new(config);

    block_on(async {
        let mut stale = backend.begin_write().await.expect("begin write");
        stale
            .get_for_update(key(b"k"))
            .await
            .expect("read at version 0");

        // Two commits evict the version the stale transaction read from.
        let mut first = backend.begin_write().await.expect("begin write");
        first.put(key(b"x"), key(b"1")).await.expect("put");
        first.commit().await.expect("commit");

        let mut second = backend.begin_write().await.expect("begin write");
        second.put(key(b"y"), key(b"2")).await.expect("put");
        second.commit().await.expect("commit");

        stale.put(key(b"k"), key(b"v")).await.expect("put");
        let error = stale.commit().await.expect_err("read version evicted");
        assert_eq!(error.kind(), ErrorKind::RetryableAbort);
    });
}

#[test]
fn fault_plan_limit_is_enforced() {
    let config = DeterministicConfig {
        max_fault_plan: 2,
        ..Default::default()
    };
    let backend = DeterministicBackend::new(config);

    backend
        .set_fault_plan(vec![CommitFault::Normal, CommitFault::Abort])
        .expect("plan fits");
    let error = backend
        .push_fault(CommitFault::Abort)
        .expect_err("plan full");
    assert_eq!(error.kind(), ErrorKind::LimitExceeded);
}

// ---------------------------------------------------------------------------
// Range clear capability.
// ---------------------------------------------------------------------------

#[test]
fn clear_range_is_declined_when_unsupported() {
    let backend = mock();
    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        let error = txn
            .clear_range(&range(b"a", b"z"))
            .await
            .expect_err("unsupported clear");
        assert_eq!(error.kind(), ErrorKind::Unsupported);
    });
}

#[test]
fn clear_range_clears_transactionally_when_supported() {
    let backend = mock_with_clear();
    seed(&backend, &[(b"a", b"1"), (b"b", b"2"), (b"z", b"3")]);

    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        txn.clear_range(&range(b"a", b"c")).await.expect("clear");
        assert_eq!(txn.get(key(b"a")).await.expect("get"), None);
        assert_eq!(txn.get(key(b"b")).await.expect("get"), None);
        assert_eq!(txn.get(key(b"z")).await.expect("get"), Some(key(b"3")));
        txn.commit().await.expect("commit");
    });

    block_on(async {
        let mut txn = backend.begin_read().await.expect("begin read");
        assert_eq!(txn.get(key(b"a")).await.expect("get"), None);
        assert_eq!(txn.get(key(b"b")).await.expect("get"), None);
        assert_eq!(txn.get(key(b"z")).await.expect("get"), Some(key(b"3")));
    });
}

#[test]
fn clear_range_composes_with_other_mutations_in_txn() {
    let backend = mock_with_clear();
    seed(&backend, &[(b"a", b"1"), (b"b", b"2")]);

    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        txn.clear_range(&range(b"a", b"c")).await.expect("clear");
        txn.put(key(b"b"), key(b"reinserted")).await.expect("put");
        // The put wins over the range clear for the same key.
        assert_eq!(
            txn.get(key(b"b")).await.expect("get"),
            Some(key(b"reinserted"))
        );
        txn.commit().await.expect("commit");
    });

    block_on(async {
        let mut txn = backend.begin_read().await.expect("begin read");
        assert_eq!(txn.get(key(b"a")).await.expect("get"), None);
        assert_eq!(
            txn.get(key(b"b")).await.expect("get"),
            Some(key(b"reinserted"))
        );
    });
}

// ---------------------------------------------------------------------------
// Durability and restart.
// ---------------------------------------------------------------------------

#[test]
fn durable_restart_preserves_committed_data() {
    let config = DeterministicConfig {
        durability: Durability::Durable,
        ..Default::default()
    };
    let backend = DeterministicBackend::new(config);
    seed(&backend, &[(b"a", b"1"), (b"b", b"2")]);

    let restarted = backend.reopen();
    assert_eq!(restarted.db_key_count(), 2);
    block_on(async {
        let mut txn = restarted.begin_read().await.expect("begin read");
        assert_eq!(txn.get(key(b"a")).await.expect("get"), Some(key(b"1")));
        assert_eq!(txn.get(key(b"b")).await.expect("get"), Some(key(b"2")));
    });
}

#[test]
fn ephemeral_restart_starts_empty() {
    let backend = DeterministicBackend::new(DeterministicConfig::default());
    seed(&backend, &[(b"a", b"1")]);

    let restarted = backend.reopen();
    assert_eq!(restarted.db_key_count(), 0);
    block_on(async {
        let mut txn = restarted.begin_read().await.expect("begin read");
        assert_eq!(txn.get(key(b"a")).await.expect("get"), None);
    });
}

// ---------------------------------------------------------------------------
// Deterministic history and replay.
// ---------------------------------------------------------------------------

#[test]
fn history_records_outcomes_without_raw_bytes() {
    let backend = mock();
    backend
        .set_fault_plan(vec![
            CommitFault::Normal,
            CommitFault::UnknownApplied,
            CommitFault::Abort,
            CommitFault::UnknownNotApplied,
        ])
        .expect("plan fits");

    block_on(async {
        for value in [b"1", b"2", b"3", b"4"] {
            let mut txn = backend.begin_write().await.expect("begin write");
            txn.put(key(b"secret-key"), key(value)).await.expect("put");
            let _ = txn.commit().await;
        }
    });

    let history = backend.history();
    assert_eq!(
        outcomes(&history),
        vec![
            CommitOutcome::Committed,
            CommitOutcome::UnknownApplied,
            CommitOutcome::Aborted,
            CommitOutcome::UnknownNotApplied,
        ]
    );
    for entry in &history {
        let debug = format!("{entry:?}");
        assert!(!debug.contains("secret-key"));
    }
}

#[test]
fn same_seed_and_plan_reproduce_identical_history() {
    let plan = vec![
        CommitFault::Normal,
        CommitFault::UnknownApplied,
        CommitFault::Abort,
        CommitFault::UnknownNotApplied,
        CommitFault::Normal,
    ];

    let backend_a = DeterministicBackend::new(DeterministicConfig::default());
    backend_a.set_fault_plan(plan.clone()).expect("plan fits");
    let backend_b = DeterministicBackend::new(DeterministicConfig::default());
    backend_b.set_fault_plan(plan).expect("plan fits");

    let replay = |backend: &DeterministicBackend| {
        block_on(async {
            for value in [b"1", b"2", b"3", b"4", b"5"] {
                let mut txn = backend.begin_write().await.expect("begin write");
                txn.put(key(b"k"), key(value)).await.expect("put");
                let _ = txn.commit().await;
            }
        });
        backend.history()
    };

    let history_a = replay(&backend_a);
    let history_b = replay(&backend_b);
    assert!(!history_a.is_empty());
    assert_eq!(history_a, history_b);
}

#[test]
fn history_ring_truncation_is_exposed() {
    let config = DeterministicConfig {
        max_history_entries: 2,
        ..Default::default()
    };
    let backend = DeterministicBackend::new(config);
    assert!(!backend.history_truncated());

    seed(&backend, &[(b"a", b"1")]);
    seed(&backend, &[(b"b", b"2")]);
    seed(&backend, &[(b"c", b"3")]);

    assert_eq!(backend.history().len(), 2);
    assert!(backend.history_truncated());
}

// ---------------------------------------------------------------------------
// Redaction.
// ---------------------------------------------------------------------------

#[test]
fn debug_and_history_redact_keys_and_values() {
    let backend = mock();
    seed(&backend, &[(b"secret-key", b"secret-value")]);

    let backend_debug = format!("{backend:?}");
    assert!(!backend_debug.contains("secret-key"));
    assert!(!backend_debug.contains("secret-value"));

    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        txn.put(key(b"another-key"), key(b"another-value"))
            .await
            .expect("put");
        let txn_debug = format!("{txn:?}");
        assert!(!txn_debug.contains("another-key"));
        assert!(!txn_debug.contains("another-value"));
        txn.commit().await.expect("commit");
    });

    for entry in &backend.history() {
        let debug = format!("{entry:?}");
        assert!(!debug.contains("another-key"));
        assert!(!debug.contains("another-value"));
        assert!(!debug.contains("secret-key"));
    }
}

#[test]
fn scan_rejects_zero_limits_for_empty_ranges() {
    let backend = mock();

    block_on(async {
        let empty_ranges = [range(b"a", b"a"), range(b"z", b"a")];
        for empty_range in &empty_ranges {
            let mut reader = backend.begin_read().await.expect("begin read");
            let read_error = reader
                .scan(
                    empty_range,
                    ScanLimits {
                        item_limit: 0,
                        byte_limit: 1,
                    },
                )
                .await
                .expect_err("read scan rejects zero limit");
            assert_eq!(read_error.kind(), ErrorKind::InvalidArgument);

            let mut writer = backend.begin_write().await.expect("begin write");
            let write_error = writer
                .scan(
                    empty_range,
                    ScanLimits {
                        item_limit: 1,
                        byte_limit: 0,
                    },
                )
                .await
                .expect_err("write scan rejects zero limit");
            assert_eq!(write_error.kind(), ErrorKind::InvalidArgument);
        }
    });
}

#[test]
fn hard_key_limit_applies_to_every_point_read() {
    let config = DeterministicConfig {
        hard_limits: HardLimits {
            max_key_bytes: 1,
            max_value_bytes: 4_096,
        },
        max_read_set: 1,
        ..Default::default()
    };
    let backend = DeterministicBackend::new(config);
    let oversized = Bytes::from_static(b"too long");

    block_on(async {
        let mut reader = backend.begin_read().await.expect("begin read");
        assert_eq!(
            reader
                .get(oversized.clone())
                .await
                .expect_err("get rejects oversized key")
                .kind(),
            ErrorKind::LimitExceeded,
        );
        assert_eq!(
            reader
                .batch_get(vec![oversized.clone()])
                .await
                .expect_err("batch get rejects oversized key")
                .kind(),
            ErrorKind::LimitExceeded,
        );

        let mut writer = backend.begin_write().await.expect("begin write");
        assert_eq!(
            writer
                .get(oversized.clone())
                .await
                .expect_err("write get rejects oversized key")
                .kind(),
            ErrorKind::LimitExceeded,
        );
        assert_eq!(
            writer
                .batch_get(vec![oversized.clone()])
                .await
                .expect_err("write batch get rejects oversized key")
                .kind(),
            ErrorKind::LimitExceeded,
        );
        assert_eq!(
            writer
                .get_for_update(oversized.clone())
                .await
                .expect_err("protected get rejects oversized key")
                .kind(),
            ErrorKind::LimitExceeded,
        );
        writer
            .get_for_update(key(b"a"))
            .await
            .expect("failed protected get does not consume read-set capacity");

        let mut batch_writer = backend.begin_write().await.expect("begin write");
        assert_eq!(
            batch_writer
                .batch_get_for_update(vec![oversized])
                .await
                .expect_err("protected batch rejects oversized key")
                .kind(),
            ErrorKind::LimitExceeded,
        );
        batch_writer
            .get_for_update(key(b"a"))
            .await
            .expect("failed protected batch does not consume read-set capacity");
    });
}

#[test]
fn unknown_applied_fault_does_not_bypass_point_conflicts() {
    let backend = mock();

    block_on(async {
        let mut first = backend.begin_write().await.expect("begin first");
        let mut second = backend.begin_write().await.expect("begin second");
        assert_eq!(
            first
                .insert(key(b"unique"), key(b"first"))
                .await
                .expect("first insert"),
            InsertOutcome::Inserted,
        );
        assert_eq!(
            second
                .insert(key(b"unique"), key(b"second"))
                .await
                .expect("second insert"),
            InsertOutcome::Inserted,
        );

        first.commit().await.expect("first commit");
        backend
            .push_fault(CommitFault::UnknownApplied)
            .expect("inject fault");
        assert_eq!(
            second
                .commit()
                .await
                .expect_err("conflicting commit aborts")
                .kind(),
            ErrorKind::RetryableAbort,
        );

        let mut reader = backend.begin_read().await.expect("begin read");
        assert_eq!(
            reader.get(key(b"unique")).await.expect("read winner"),
            Some(key(b"first")),
        );
    });
}

#[test]
fn clear_range_charges_one_mutation_and_its_boundaries() {
    let config = DeterministicConfig {
        admission_budget: AdmissionBudget {
            max_mutations: 1,
            max_mutation_bytes: 2,
        },
        capabilities: Capabilities {
            transactional_clear_range: true,
        },
        max_mutation_buffer: 1,
        ..Default::default()
    };
    let backend = DeterministicBackend::new(config);
    seed(&backend, &[(b"a", b"")]);
    seed(&backend, &[(b"b", b"")]);

    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin clear");
        txn.clear_range(&range(b"a", b"c"))
            .await
            .expect("range contents do not count against the mutation budget");
        txn.commit().await.expect("commit clear");

        let mut reader = backend.begin_read().await.expect("begin read");
        assert_eq!(reader.get(key(b"a")).await.expect("get a"), None);
        assert_eq!(reader.get(key(b"b")).await.expect("get b"), None);
    });
}

#[test]
fn history_fingerprint_delimits_keys_and_values() {
    let left = mock();
    let right = mock();

    block_on(async {
        let mut left_txn = left.begin_write().await.expect("begin left");
        left_txn.put(key(b"a"), key(b"bc")).await.expect("left put");
        left_txn.commit().await.expect("left commit");

        let mut right_txn = right.begin_write().await.expect("begin right");
        right_txn
            .put(key(b"ab"), key(b"c"))
            .await
            .expect("right put");
        right_txn.commit().await.expect("right commit");
    });

    assert_ne!(left.history(), right.history());
}

#[test]
fn committed_range_clear_conflicts_with_a_protected_point_read() {
    let backend = mock_with_clear();

    block_on(async {
        let mut stale = backend.begin_write().await.expect("begin stale writer");
        stale
            .get_for_update(key(b"middle"))
            .await
            .expect("protected read");

        let mut clear = backend.begin_write().await.expect("begin clear");
        clear
            .clear_range(&range(b"a", b"z"))
            .await
            .expect("stage clear");
        clear.commit().await.expect("commit clear");

        stale
            .put(key(b"middle"), key(b"stale"))
            .await
            .expect("stage stale write");
        assert_eq!(
            stale
                .commit()
                .await
                .expect_err("range clear conflicts")
                .kind(),
            ErrorKind::RetryableAbort,
        );
    });
}

#[test]
fn range_clear_union_normalizes_overlap_adjacency_and_order() {
    let config = DeterministicConfig {
        capabilities: Capabilities {
            transactional_clear_range: true,
        },
        max_mutation_buffer: 1,
        ..Default::default()
    };
    let backend = DeterministicBackend::new(config);
    for (entry_key, value) in [
        (b"b".as_slice(), b"1".as_slice()),
        (b"y".as_slice(), b"2".as_slice()),
        (b"z".as_slice(), b"3".as_slice()),
        (b"zz".as_slice(), b"4".as_slice()),
    ] {
        seed(&backend, &[(entry_key, value)]);
    }

    block_on(async {
        let mut txn = backend.begin_write().await.expect("begin write");
        txn.clear_range(&range(b"m", b"z"))
            .await
            .expect("first clear");
        txn.clear_range(&range(b"a", b"n"))
            .await
            .expect("out-of-order overlapping clear shares the range");
        txn.clear_range(&range(b"z", b"zz"))
            .await
            .expect("adjacent clear shares the range");
        txn.clear_range(&range(b"b", b"c"))
            .await
            .expect("contained clear shares the range");
        txn.commit().await.expect("commit normalized clears");

        let mut reader = backend.begin_read().await.expect("begin read");
        assert_eq!(
            reader
                .batch_get(vec![key(b"b"), key(b"y"), key(b"z"), key(b"zz")])
                .await
                .expect("read normalized clear result"),
            vec![None, None, None, Some(key(b"4"))],
        );
    });
}
