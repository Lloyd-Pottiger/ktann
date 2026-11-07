//! Shared backend contract suite and harness.
//!
//! This module owns the backend-neutral transaction contract cases and the
//! [`BackendHarness`] seam every adapter implements to run them. Cases drive a
//! backend through public storage interfaces, including typed logical transactions,
//! so they run unchanged against each production adapter, including
//! Memory with its optional test controls. Each case is deterministic and named;
//! failures report the case name together
//! with a safe error category or count, never a raw key, value, or backend
//! error source.
//!
//! Case selection and assertions live entirely in this suite. A harness may
//! only declare a capability unavailable ([`FaultInjection::Unavailable`] or
//! [`RestartMode::Unsupported`]); it can never supply a weakened assertion.

use std::fmt;

use bytes::Bytes;
use ktann::api::ErrorKind;
use ktann::storage::backend::{Backend, InsertOutcome, Mutation, ReadOps, ScanLimits, WriteTxn};
use ktann::storage::keys::KeyRange;

/// Whether a harness can inject controlled commit faults.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(
    dead_code,
    reason = "variants are used by different harnesses in different crates"
)]
pub enum FaultInjection {
    /// The harness can stage the next commit outcome via
    /// [`BackendHarness::inject_fault`].
    Controlled,
    /// The harness cannot stage a controlled outcome, so the suite skips the
    /// definite/unknown commit-outcome cases.
    Unavailable,
}

/// A controlled commit fault the suite may request during fault injection.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(
    dead_code,
    reason = "variants are used by different harnesses in different crates"
)]
pub enum Fault {
    /// Commit reports a definite abort and applies nothing.
    Abort,
    /// Commit applies the mutation but reports an unknown outcome.
    UnknownApplied,
    /// Commit applies nothing but reports an unknown outcome.
    UnknownNotApplied,
}

/// The durability semantics of a harness restart.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[allow(
    dead_code,
    reason = "variants are used by different harnesses in different crates"
)]
pub enum RestartMode {
    /// A restart preserves committed data.
    Durable,
    /// A restart starts empty.
    Ephemeral,
    /// The harness cannot simulate a restart, so the suite skips the
    /// durability-mapping case.
    Unsupported,
}

/// The adapter-side seam that runs the shared contract suite.
///
/// A harness owns setup and cleanup and provides one isolated backend keyspace.
/// It declares whether controlled commit faults and restarts are available.
/// The suite owns every case and assertion; a harness only reports capabilities
/// and maps the few operations (fault injection, restart) that only some
/// backends support.
pub trait BackendHarness: Send + Sync {
    /// The backend type under test.
    type Backend: Backend;

    /// Returns the backend the cases drive.
    fn backend(&self) -> &Self::Backend;

    /// Whether controlled commit faults can be injected.
    fn fault_injection(&self) -> FaultInjection;

    /// Stages `fault` as the next commit outcome.
    ///
    /// # Panics
    ///
    /// Only called when [`fault_injection`](BackendHarness::fault_injection)
    /// reports [`FaultInjection::Controlled`].
    fn inject_fault(&self, fault: Fault);

    /// The durability semantics of a restart, or `Unsupported` when the
    /// harness cannot simulate a restart.
    fn restart_mode(&self) -> RestartMode;

    /// Restarts the backend and returns a fresh harness observing post-restart
    /// committed state.
    ///
    /// Only called when [`restart_mode`](BackendHarness::restart_mode) reports
    /// [`RestartMode::Durable`] or [`RestartMode::Ephemeral`].
    fn restart(&self) -> Self
    where
        Self: Sized;
}

/// Runs every applicable contract case against `harness`.
///
/// Cases whose capability is unavailable ([`FaultInjection::Unavailable`] or
/// [`RestartMode::Unsupported`]) are skipped by the suite itself, not by the
/// adapter. Case order is stable so a replay against a fresh harness observes
/// the same sequence of operations.
pub async fn run_suite<H: BackendHarness>(harness: &H) {
    case_declared_limits_and_capabilities(harness, "declared_limits").await;
    case_snapshot_consistency(harness, "snapshot_consistency").await;
    case_read_your_writes(harness, "read_your_writes").await;
    case_ordered_writes(harness, "ordered_writes").await;
    case_scan_pagination(harness, "scan_pagination").await;
    case_unique_insert(harness, "unique_insert").await;
    case_update_protected_conflict(harness, "update_protected_conflict").await;
    case_aba_conflict(harness, "aba_conflict").await;
    case_rollback_and_drop(harness, "rollback_and_drop").await;
    case_range_clear_capability(harness, "range_clear_capability").await;
    case_commit_definite_abort(harness, "commit_definite_abort").await;
    case_commit_unknown_applied(harness, "commit_unknown_applied").await;
    case_commit_unknown_not_applied(harness, "commit_unknown_not_applied").await;
    case_durability_mapping(harness, "durability_mapping").await;
    case_scan_no_skip(harness, "scan_no_skip").await;
    case_scan_empty_range(harness, "scan_empty_range").await;
    case_batch_scan(harness, "batch_scan").await;
    case_graph_storage(harness, "graph_storage").await;
}

// ---------------------------------------------------------------------------
// Redacted assertion helpers.
//
// These helpers never interpolate raw keys or values into a panic message, so a
// failing case leaks only its name and a safe category or count.
// ---------------------------------------------------------------------------

/// Compares bytes without including them in failure diagnostics.
#[track_caller]
fn check_value(ctx: &str, what: impl fmt::Display, got: Option<&[u8]>, expected: Option<&[u8]>) {
    assert!(
        got == expected,
        "case `{ctx}`: {what}: expected byte length {:?}, got {:?}",
        expected.map(<[u8]>::len),
        got.map(<[u8]>::len),
    );
}

/// Preserves per-item diagnostics while keeping stored bytes out of failures.
#[track_caller]
fn check_batch(ctx: &str, what: &str, got: &[Option<Bytes>], expected: &[Option<&[u8]>]) {
    assert_eq!(
        got.len(),
        expected.len(),
        "case `{ctx}`: {what}: batch length"
    );
    for (index, (got, expected)) in got.iter().zip(expected).enumerate() {
        check_value(
            ctx,
            format_args!("{what}, item {index}"),
            got.as_deref(),
            *expected,
        );
    }
}

// ---------------------------------------------------------------------------
// Key and range builders.
// ---------------------------------------------------------------------------

/// Builds a case-scoped key so distinct cases cannot collide within one backend.
fn case_key(ctx: &str, suffix: &str) -> Bytes {
    Bytes::from(format!("contract/{ctx}/{suffix}"))
}

/// Builds the `[start, end)` range covering every key under `prefix`.
fn case_subrange(ctx: &str, prefix: &str) -> KeyRange {
    let start = case_key(ctx, prefix);
    KeyRange::new(start.to_vec(), prefix_end(&start))
}

/// Returns the exclusive upper bound of every key sharing `prefix`.
fn prefix_end(prefix: &[u8]) -> Vec<u8> {
    let mut end = prefix.to_vec();
    while let Some(last) = end.last_mut() {
        if *last == 0xff {
            end.pop();
        } else {
            *last += 1;
            return end;
        }
    }
    Vec::new()
}

/// Builds a static value fixture.
fn value(value: &'static [u8]) -> Bytes {
    Bytes::from_static(value)
}

// ---------------------------------------------------------------------------
// Cases.
// ---------------------------------------------------------------------------

/// Every backend declares positive hard limits and admission budgets.
async fn case_declared_limits_and_capabilities<H: BackendHarness>(harness: &H, ctx: &str) {
    let backend = harness.backend();
    let hard_limits = backend.hard_limits();
    let budget = backend.admission_budget();
    assert!(
        hard_limits.max_key_bytes > 0,
        "case `{ctx}`: hard key limit is positive"
    );
    assert!(
        hard_limits.max_value_bytes > 0,
        "case `{ctx}`: hard value limit is positive"
    );
    assert!(
        budget.max_mutations > 0,
        "case `{ctx}`: mutation count budget is positive"
    );
    assert!(
        budget.max_mutation_bytes > 0,
        "case `{ctx}`: mutation byte budget is positive"
    );
    // The range-clear capability flag is verified by the range-clear case.
    let _ = backend.capabilities();
}

/// An open read snapshot stays isolated from later commits.
async fn case_snapshot_consistency<H: BackendHarness>(harness: &H, ctx: &str) {
    let backend = harness.backend();
    let key = case_key(ctx, "snapshot");
    {
        let mut txn = backend.begin_write().await.expect("begin seed");
        txn.put(key.clone(), value(b"old")).await.expect("put seed");
        txn.commit().await.expect("commit seed");
    }

    let mut old = backend.begin_read().await.expect("begin old snapshot");
    {
        let mut txn = backend.begin_write().await.expect("begin update");
        txn.put(key, value(b"new")).await.expect("put update");
        txn.put(case_key(ctx, "later"), value(b"visible"))
            .await
            .expect("put later");
        txn.commit().await.expect("commit update");
    }

    let old_value = old.get(case_key(ctx, "snapshot")).await.expect("read old");
    check_value(
        ctx,
        "old snapshot keeps old value",
        old_value.as_deref(),
        Some(b"old"),
    );
    let later = old.get(case_key(ctx, "later")).await.expect("read later");
    check_value(
        ctx,
        "old snapshot does not see later key",
        later.as_deref(),
        None,
    );
}

/// A write transaction reads its own uncommitted writes.
async fn case_read_your_writes<H: BackendHarness>(harness: &H, ctx: &str) {
    let backend = harness.backend();
    let key = case_key(ctx, "k");
    let mut txn = backend.begin_write().await.expect("begin write");
    let before = txn.get(key.clone()).await.expect("get");
    check_value(ctx, "absent before put", before.as_deref(), None);

    txn.put(key.clone(), value(b"v")).await.expect("put");
    let after_put = txn.get(key.clone()).await.expect("get");
    check_value(ctx, "own put is visible", after_put.as_deref(), Some(b"v"));

    txn.delete(key.clone()).await.expect("delete");
    let after_delete = txn.get(key).await.expect("get");
    check_value(ctx, "own delete is visible", after_delete.as_deref(), None);

    txn.rollback().await;
}

/// A batch mutation applies in input order and reads back through the overlay.
async fn case_ordered_writes<H: BackendHarness>(harness: &H, ctx: &str) {
    let backend = harness.backend();
    let mut txn = backend.begin_write().await.expect("begin write");
    txn.batch_mutate(vec![
        Mutation::Put {
            key: case_key(ctx, "a"),
            value: value(b"first"),
        },
        Mutation::Put {
            key: case_key(ctx, "a"),
            value: value(b"second"),
        },
        Mutation::Put {
            key: case_key(ctx, "b"),
            value: value(b"deleted"),
        },
        Mutation::Delete {
            key: case_key(ctx, "b"),
        },
    ])
    .await
    .expect("batch mutate");

    let got = txn
        .batch_get(vec![
            case_key(ctx, "a"),
            case_key(ctx, "b"),
            case_key(ctx, "a"),
        ])
        .await
        .expect("batch get");
    check_batch(
        ctx,
        "ordered batch reads own writes",
        &got,
        &[Some(&b"second"[..]), None, Some(&b"second"[..])],
    );

    let page = txn
        .scan(
            &case_subrange(ctx, ""),
            ScanLimits {
                item_limit: 10,
                byte_limit: 1_024,
            },
        )
        .await
        .expect("write scan");
    assert_eq!(
        page.items().len(),
        1,
        "case `{ctx}`: ordered batch scan count"
    );
    check_value(
        ctx,
        "ordered batch scan value",
        Some(page.items()[0].value().as_ref()),
        Some(b"second"),
    );

    txn.commit().await.expect("commit");
}

/// A scan returns ordered pages with a strictly advancing cursor and rejects
/// zero limits.
async fn case_scan_pagination<H: BackendHarness>(harness: &H, ctx: &str) {
    let backend = harness.backend();
    {
        let mut txn = backend.begin_write().await.expect("begin seed");
        txn.put(case_key(ctx, "scan/a"), value(b"1"))
            .await
            .expect("put a");
        txn.put(case_key(ctx, "scan/b"), value(b"2"))
            .await
            .expect("put b");
        txn.put(case_key(ctx, "scan/c"), value(b"12345"))
            .await
            .expect("put c");
        txn.commit().await.expect("commit seed");
    }

    let mut txn = backend.begin_read().await.expect("begin read");
    let range = case_subrange(ctx, "scan/");
    let first = txn
        .scan(
            &range,
            ScanLimits {
                item_limit: 2,
                byte_limit: 1_024,
            },
        )
        .await
        .expect("first page");
    assert_eq!(first.items().len(), 2, "case `{ctx}`: first page count");
    assert!(
        first.items()[0].key().as_ref() == case_key(ctx, "scan/a").as_ref(),
        "case `{ctx}`: first page first key"
    );
    assert!(
        first.items()[1].key().as_ref() == case_key(ctx, "scan/b").as_ref(),
        "case `{ctx}`: first page second key"
    );

    let next = first.next_start().expect("non-terminal cursor").clone();
    let mut expected_next = case_key(ctx, "scan/b").to_vec();
    expected_next.push(0x00);
    assert!(
        next.as_ref() == expected_next.as_slice(),
        "case `{ctx}`: cursor is the byte-successor of the last key"
    );

    let second = txn
        .scan(
            &KeyRange::new(next.to_vec(), case_key(ctx, "scan0").to_vec()),
            ScanLimits {
                item_limit: 2,
                byte_limit: 1,
            },
        )
        .await
        .expect("second page");
    assert_eq!(second.items().len(), 1, "case `{ctx}`: second page count");
    check_value(
        ctx,
        "second page value",
        Some(second.items()[0].value().as_ref()),
        Some(b"12345"),
    );
    assert!(
        second.next_start().is_none(),
        "case `{ctx}`: second page is terminal"
    );

    let error = txn
        .scan(
            &range,
            ScanLimits {
                item_limit: 0,
                byte_limit: 1,
            },
        )
        .await
        .expect_err("zero item limit");
    assert_eq!(
        error.kind(),
        ErrorKind::InvalidArgument,
        "case `{ctx}`: zero scan limit is invalid"
    );
}

/// Drains an entire range one page at a time, returning the ordered keys and
/// the number of pages consumed.
async fn drain_scan<H: BackendHarness>(
    harness: &H,
    range: &KeyRange,
    limits: ScanLimits,
) -> (Vec<Vec<u8>>, usize) {
    let backend = harness.backend();
    let mut keys = Vec::new();
    let mut pages = 0usize;
    let mut start = range.start().to_vec();
    loop {
        let mut txn = backend.begin_read().await.expect("begin read");
        let page = txn
            .scan(&KeyRange::new(start.clone(), range.end().to_vec()), limits)
            .await
            .expect("scan page");
        for item in page.items() {
            keys.push(item.key().to_vec());
        }
        pages += 1;
        match page.next_start() {
            Some(next) => start = next.to_vec(),
            None => break,
        }
    }
    (keys, pages)
}

/// A scan paginates gap-free and duplicate-free across item and byte bounds,
/// an oversized first item, and exact-boundary exhaustion.
async fn case_scan_no_skip<H: BackendHarness>(harness: &H, ctx: &str) {
    let backend = harness.backend();
    let suffixes = ["a", "b", "c", "d", "e"];
    let expected: Vec<Vec<u8>> = suffixes
        .iter()
        .map(|suffix| case_key(ctx, &format!("round/{suffix}")).to_vec())
        .collect();
    {
        let mut txn = backend.begin_write().await.expect("begin seed");
        for (index, suffix) in suffixes.iter().enumerate() {
            let item_value = if index == 2 {
                Bytes::from(vec![0x7f; 64])
            } else {
                value(b"v")
            };
            txn.put(case_key(ctx, &format!("round/{suffix}")), item_value)
                .await
                .expect("put");
        }
        txn.commit().await.expect("commit seed");
    }

    let range = case_subrange(ctx, "round/");

    // Item boundary: a generous byte limit lets the item limit drive paging.
    let (item_keys, item_pages) = drain_scan(
        harness,
        &range,
        ScanLimits {
            item_limit: 2,
            byte_limit: 1_024,
        },
    )
    .await;
    assert!(
        item_keys == expected,
        "case `{ctx}`: item-bounded paging has no gap or duplicate"
    );
    assert_eq!(item_pages, 3, "case `{ctx}`: item-bounded page count");

    // Byte boundary: a tight byte limit splits pages mid-value while the
    // oversized middle value is still returned alone, and the final page
    // exhausts the range exactly.
    let (byte_keys, byte_pages) = drain_scan(
        harness,
        &range,
        ScanLimits {
            item_limit: 1_024,
            byte_limit: 48,
        },
    )
    .await;
    assert!(
        byte_keys == expected,
        "case `{ctx}`: byte-bounded paging has no gap or duplicate"
    );
    assert_eq!(
        byte_pages,
        suffixes.len(),
        "case `{ctx}`: byte-bounded page count"
    );
}

/// A scan of an inverted or keyless range yields an empty terminal page, never
/// a continuation.
async fn case_scan_empty_range<H: BackendHarness>(harness: &H, ctx: &str) {
    let backend = harness.backend();
    {
        let mut txn = backend.begin_write().await.expect("begin seed");
        txn.put(case_key(ctx, "empty/inside"), value(b"1"))
            .await
            .expect("put");
        txn.commit().await.expect("commit seed");
    }

    let mut txn = backend.begin_read().await.expect("begin read");

    // An inverted `[end, start)` range is empty even when keys exist elsewhere.
    let inverted = txn
        .scan(
            &KeyRange::new(
                case_key(ctx, "empty/b").to_vec(),
                case_key(ctx, "empty/a").to_vec(),
            ),
            ScanLimits {
                item_limit: 10,
                byte_limit: 1_024,
            },
        )
        .await
        .expect("inverted scan");
    assert_eq!(
        inverted.items().len(),
        0,
        "case `{ctx}`: inverted range is empty"
    );
    assert!(
        inverted.is_terminal(),
        "case `{ctx}`: inverted range is terminal"
    );

    // A well-formed half-open range that happens to contain no keys is empty.
    let gap = txn
        .scan(
            &case_subrange(ctx, "empty/gap/"),
            ScanLimits {
                item_limit: 10,
                byte_limit: 1_024,
            },
        )
        .await
        .expect("empty gap scan");
    assert_eq!(gap.items().len(), 0, "case `{ctx}`: keyless range is empty");
    assert!(gap.is_terminal(), "case `{ctx}`: keyless range is terminal");
}

/// A batched scan returns one independently paginated page per input range,
/// preserving input order, duplicates, and empty or inverted ranges.
async fn case_batch_scan<H: BackendHarness>(harness: &H, ctx: &str) {
    let backend = harness.backend();
    {
        let mut txn = backend.begin_write().await.expect("begin seed");
        txn.put(case_key(ctx, "a/1"), value(b"1"))
            .await
            .expect("put a1");
        txn.put(case_key(ctx, "a/2"), value(b"22"))
            .await
            .expect("put a2");
        txn.put(case_key(ctx, "a/3"), value(b"333"))
            .await
            .expect("put a3");
        txn.put(case_key(ctx, "b/1"), value(b"x"))
            .await
            .expect("put b1");
        txn.put(case_key(ctx, "b/2"), value(b"yy"))
            .await
            .expect("put b2");
        txn.put(case_key(ctx, "big/k"), Bytes::from(vec![0x7f; 64]))
            .await
            .expect("put big");
        txn.commit().await.expect("commit seed");
    }

    let range_a = case_subrange(ctx, "a/");
    let range_b = case_subrange(ctx, "b/");
    let limits = ScanLimits {
        item_limit: 2,
        byte_limit: 1_024,
    };

    let mut txn = backend.begin_read().await.expect("begin read");

    // An empty input succeeds with an empty result.
    let none = txn.batch_scan(&[], limits).await.expect("empty batch scan");
    assert_eq!(none.len(), 0, "case `{ctx}`: empty batch page count");

    let inverted = KeyRange::new(case_key(ctx, "b/2").to_vec(), case_key(ctx, "b/1").to_vec());
    let pages = txn
        .batch_scan(
            &[range_a.clone(), range_b.clone(), inverted, range_a.clone()],
            limits,
        )
        .await
        .expect("batch scan");
    assert_eq!(pages.len(), 4, "case `{ctx}`: one page per input range");

    // Range A paginates independently at the item limit.
    assert_eq!(
        pages[0].items().len(),
        2,
        "case `{ctx}`: range a first page count"
    );
    assert!(
        pages[0].items()[0].key().as_ref() == case_key(ctx, "a/1").as_ref()
            && pages[0].items()[1].key().as_ref() == case_key(ctx, "a/2").as_ref(),
        "case `{ctx}`: range a page order"
    );
    let mut expected_next = case_key(ctx, "a/2").to_vec();
    expected_next.push(0x00);
    assert!(
        pages[0]
            .next_start()
            .is_some_and(|next| next.as_ref() == expected_next.as_slice()),
        "case `{ctx}`: range a cursor is the byte-successor of its last key"
    );

    // Range B exhausts in one terminal page.
    assert_eq!(
        pages[1].items().len(),
        2,
        "case `{ctx}`: range b page count"
    );
    assert!(
        pages[1].is_terminal(),
        "case `{ctx}`: range b page is terminal"
    );

    // The inverted range is a terminal empty page at its own position.
    assert_eq!(
        pages[2].items().len(),
        0,
        "case `{ctx}`: inverted range page count"
    );
    assert!(
        pages[2].is_terminal(),
        "case `{ctx}`: inverted range page is terminal"
    );

    // A repeated range is read independently and returns the same page.
    assert!(
        pages[3] == pages[0],
        "case `{ctx}`: duplicate range repeats"
    );

    // Resuming range A in a later batch skips no key and repeats none.
    let resume = KeyRange::new(expected_next, range_a.end().to_vec());
    let pages = txn
        .batch_scan(&[resume], limits)
        .await
        .expect("resumed batch scan");
    assert_eq!(
        pages[0].items().len(),
        1,
        "case `{ctx}`: resumed page count"
    );
    check_value(
        ctx,
        "resumed value",
        Some(pages[0].items()[0].value().as_ref()),
        Some(b"333"),
    );
    assert!(
        pages[0].is_terminal(),
        "case `{ctx}`: resumed page is terminal"
    );

    // A single oversized first item is returned alone even past the byte
    // limit.
    let pages = txn
        .batch_scan(
            &[case_subrange(ctx, "big/")],
            ScanLimits {
                item_limit: 8,
                byte_limit: 48,
            },
        )
        .await
        .expect("oversized batch scan");
    assert_eq!(
        pages[0].items().len(),
        1,
        "case `{ctx}`: oversized first item alone"
    );
    assert!(
        pages[0].is_terminal(),
        "case `{ctx}`: oversized page is terminal"
    );

    // A zero limit is invalid before any work.
    let error = txn
        .batch_scan(
            std::slice::from_ref(&range_a),
            ScanLimits {
                item_limit: 0,
                byte_limit: 1,
            },
        )
        .await
        .expect_err("zero item limit");
    assert_eq!(
        error.kind(),
        ErrorKind::InvalidArgument,
        "case `{ctx}`: zero batch scan limit is invalid"
    );
    drop(txn);

    // A write transaction's batched scan reads its own staged writes.
    let mut txn = backend.begin_write().await.expect("begin write");
    txn.put(case_key(ctx, "c/1"), value(b"staged"))
        .await
        .expect("put staged");
    let pages = txn
        .batch_scan(&[case_subrange(ctx, "c/"), range_b], limits)
        .await
        .expect("write batch scan");
    assert_eq!(
        pages[0].items().len(),
        1,
        "case `{ctx}`: staged range count"
    );
    check_value(
        ctx,
        "staged value is visible",
        Some(pages[0].items()[0].value().as_ref()),
        Some(b"staged"),
    );
    assert_eq!(
        pages[1].items().len(),
        2,
        "case `{ctx}`: committed range count"
    );
    txn.rollback().await;
}

/// Unique insertion distinguishes inserted from existing and conflicts on
/// concurrent insertion.
async fn case_unique_insert<H: BackendHarness>(harness: &H, ctx: &str) {
    let backend = harness.backend();
    {
        let mut txn = backend.begin_write().await.expect("begin seed");
        txn.put(case_key(ctx, "existing"), value(b"old"))
            .await
            .expect("put seed");
        txn.commit().await.expect("commit seed");
    }

    let mut txn = backend.begin_write().await.expect("begin write");
    assert_eq!(
        txn.insert(case_key(ctx, "fresh"), value(b"v"))
            .await
            .expect("insert"),
        InsertOutcome::Inserted,
        "case `{ctx}`: fresh insert"
    );
    assert_eq!(
        txn.insert(case_key(ctx, "existing"), value(b"v"))
            .await
            .expect("insert"),
        InsertOutcome::AlreadyExists,
        "case `{ctx}`: existing insert"
    );
    let existing = txn.get(case_key(ctx, "existing")).await.expect("get");
    check_value(
        ctx,
        "existing value unchanged",
        existing.as_deref(),
        Some(b"old"),
    );
    assert_eq!(
        txn.insert(case_key(ctx, "fresh"), value(b"again"))
            .await
            .expect("insert"),
        InsertOutcome::AlreadyExists,
        "case `{ctx}`: re-insert own write"
    );
    txn.commit().await.expect("commit");
}

/// An update-protected read conflicts with a concurrent write.
async fn case_update_protected_conflict<H: BackendHarness>(harness: &H, ctx: &str) {
    let backend = harness.backend();
    let key = case_key(ctx, "k");
    let mut first = backend.begin_write().await.expect("begin first");
    let mut second = backend.begin_write().await.expect("begin second");

    first.get_for_update(key.clone()).await.expect("first read");
    second
        .get_for_update(key.clone())
        .await
        .expect("second read");

    first
        .put(key.clone(), value(b"1"))
        .await
        .expect("first put");
    first.commit().await.expect("first commit wins");

    second
        .put(key.clone(), value(b"2"))
        .await
        .expect("second put");
    let error = second.commit().await.expect_err("second conflicts");
    assert_eq!(
        error.kind(),
        ErrorKind::RetryableAbort,
        "case `{ctx}`: conflict maps to RetryableAbort"
    );

    let mut read = backend.begin_read().await.expect("read winner");
    let winner = read.get(key).await.expect("get");
    check_value(ctx, "winner value persists", winner.as_deref(), Some(b"1"));
}

/// Restoring a key to its original value still conflicts with a stale writer.
async fn case_aba_conflict<H: BackendHarness>(harness: &H, ctx: &str) {
    let backend = harness.backend();
    let key = case_key(ctx, "k");
    let mut reader = backend.begin_write().await.expect("begin reader");
    reader
        .get_for_update(key.clone())
        .await
        .expect("read absent");

    let mut first = backend.begin_write().await.expect("begin first");
    first.put(key.clone(), value(b"1")).await.expect("put");
    first.commit().await.expect("commit");

    let mut second = backend.begin_write().await.expect("begin second");
    second.delete(key.clone()).await.expect("delete");
    second.commit().await.expect("commit");

    reader
        .put(key, value(b"2"))
        .await
        .expect("stage stale write");
    let error = reader.commit().await.expect_err("ABA conflict");
    assert_eq!(
        error.kind(),
        ErrorKind::RetryableAbort,
        "case `{ctx}`: ABA conflict maps to RetryableAbort"
    );
}

/// Rollback and dropping a transaction both persist nothing.
async fn case_rollback_and_drop<H: BackendHarness>(harness: &H, ctx: &str) {
    let backend = harness.backend();
    {
        let mut txn = backend.begin_write().await.expect("begin rollback");
        txn.put(case_key(ctx, "rollback"), value(b"hidden"))
            .await
            .expect("put");
        txn.rollback().await;
    }
    {
        let mut txn = backend.begin_write().await.expect("begin drop");
        txn.put(case_key(ctx, "drop"), value(b"hidden"))
            .await
            .expect("put");
        drop(txn);
    }

    let mut read = backend.begin_read().await.expect("read after abandon");
    let got = read
        .batch_get(vec![case_key(ctx, "rollback"), case_key(ctx, "drop")])
        .await
        .expect("batch get");
    check_batch(
        ctx,
        "rollback and drop persist nothing",
        &got,
        &[None, None],
    );
}

/// The range-clear capability either clears transactionally or is declined.
async fn case_range_clear_capability<H: BackendHarness>(harness: &H, ctx: &str) {
    let backend = harness.backend();
    let clear = case_subrange(ctx, "clear/");
    if backend.capabilities().transactional_clear_range {
        {
            let mut txn = backend.begin_write().await.expect("begin seed");
            txn.put(case_key(ctx, "clear/a"), value(b"1"))
                .await
                .expect("put a");
            txn.put(case_key(ctx, "clear/b"), value(b"2"))
                .await
                .expect("put b");
            txn.put(case_key(ctx, "outside"), value(b"3"))
                .await
                .expect("put outside");
            txn.commit().await.expect("commit seed");
        }

        {
            let mut txn = backend.begin_write().await.expect("begin clear");
            txn.clear_range(&clear).await.expect("stage clear");

            // A concurrent commit to a cleared key is made invisible by the
            // later range clear.
            let mut concurrent = backend.begin_write().await.expect("begin concurrent");
            concurrent
                .put(case_key(ctx, "clear/concurrent"), value(b"removed"))
                .await
                .expect("put concurrent");
            concurrent.commit().await.expect("commit concurrent");

            txn.commit().await.expect("commit clear");
        }

        let mut read = backend.begin_read().await.expect("read cleared");
        let a = read.get(case_key(ctx, "clear/a")).await.expect("get a");
        check_value(ctx, "cleared key a is absent", a.as_deref(), None);
        let b = read.get(case_key(ctx, "clear/b")).await.expect("get b");
        check_value(ctx, "cleared key b is absent", b.as_deref(), None);
        let concurrent = read
            .get(case_key(ctx, "clear/concurrent"))
            .await
            .expect("get concurrent");
        check_value(
            ctx,
            "concurrent key is cleared",
            concurrent.as_deref(),
            None,
        );
        let outside = read
            .get(case_key(ctx, "outside"))
            .await
            .expect("get outside");
        check_value(
            ctx,
            "outside key is retained",
            outside.as_deref(),
            Some(b"3"),
        );
    } else {
        let mut txn = backend
            .begin_write()
            .await
            .expect("begin unsupported clear");
        let error = txn
            .clear_range(&clear)
            .await
            .expect_err("range clear is declined");
        assert_eq!(
            error.kind(),
            ErrorKind::Unsupported,
            "case `{ctx}`: range clear is declined"
        );
    }
}

/// A controlled definite abort reports `RetryableAbort` and persists nothing.
async fn case_commit_definite_abort<H: BackendHarness>(harness: &H, ctx: &str) {
    if harness.fault_injection() != FaultInjection::Controlled {
        return; // Skipped: this harness cannot stage a controlled abort.
    }
    harness.inject_fault(Fault::Abort);

    let backend = harness.backend();
    let mut txn = backend.begin_write().await.expect("begin write");
    txn.put(case_key(ctx, "abort"), value(b"v"))
        .await
        .expect("put");
    let error = txn.commit().await.expect_err("faulted commit aborts");
    assert_eq!(
        error.kind(),
        ErrorKind::RetryableAbort,
        "case `{ctx}`: definite abort maps to RetryableAbort"
    );

    let mut read = backend.begin_read().await.expect("read after abort");
    let got = read.get(case_key(ctx, "abort")).await.expect("get");
    check_value(ctx, "definite abort persists nothing", got.as_deref(), None);
}

/// A controlled unknown-applied outcome reports `CommitOutcomeUnknown` and
/// persists every mutation atomically.
async fn case_commit_unknown_applied<H: BackendHarness>(harness: &H, ctx: &str) {
    if harness.fault_injection() != FaultInjection::Controlled {
        return; // Skipped: this harness cannot stage an unknown outcome.
    }
    harness.inject_fault(Fault::UnknownApplied);

    let backend = harness.backend();
    let mut txn = backend.begin_write().await.expect("begin write");
    txn.put(case_key(ctx, "a"), value(b"1"))
        .await
        .expect("put a");
    txn.put(case_key(ctx, "b"), value(b"2"))
        .await
        .expect("put b");
    let error = txn.commit().await.expect_err("unknown applied");
    assert_eq!(
        error.kind(),
        ErrorKind::CommitOutcomeUnknown,
        "case `{ctx}`: unknown applied maps to CommitOutcomeUnknown"
    );

    let mut read = backend.begin_read().await.expect("read after unknown");
    let a = read.get(case_key(ctx, "a")).await.expect("get a");
    check_value(ctx, "unknown applied persists a", a.as_deref(), Some(b"1"));
    let b = read.get(case_key(ctx, "b")).await.expect("get b");
    check_value(ctx, "unknown applied persists b", b.as_deref(), Some(b"2"));
}

/// A controlled unknown-not-applied outcome reports `CommitOutcomeUnknown` and
/// persists nothing.
async fn case_commit_unknown_not_applied<H: BackendHarness>(harness: &H, ctx: &str) {
    if harness.fault_injection() != FaultInjection::Controlled {
        return; // Skipped: this harness cannot stage an unknown outcome.
    }
    harness.inject_fault(Fault::UnknownNotApplied);

    let backend = harness.backend();
    let mut txn = backend.begin_write().await.expect("begin write");
    txn.put(case_key(ctx, "a"), value(b"1"))
        .await
        .expect("put a");
    let error = txn.commit().await.expect_err("unknown not applied");
    assert_eq!(
        error.kind(),
        ErrorKind::CommitOutcomeUnknown,
        "case `{ctx}`: unknown not applied maps to CommitOutcomeUnknown"
    );

    let mut read = backend.begin_read().await.expect("read after unknown");
    let got = read.get(case_key(ctx, "a")).await.expect("get a");
    check_value(
        ctx,
        "unknown not applied persists nothing",
        got.as_deref(),
        None,
    );
}

/// A restart preserves or drops committed data per the harness's restart mode.
async fn case_durability_mapping<H: BackendHarness>(harness: &H, ctx: &str) {
    let mode = harness.restart_mode();
    if mode == RestartMode::Unsupported {
        return; // Skipped: this harness cannot simulate a restart.
    }

    {
        let backend = harness.backend();
        let mut txn = backend.begin_write().await.expect("begin write");
        txn.put(case_key(ctx, "durable"), value(b"kept"))
            .await
            .expect("put");
        txn.commit().await.expect("commit");
    }

    let restarted = harness.restart();
    let backend = restarted.backend();
    let mut read = backend
        .begin_read()
        .await
        .expect("begin read after restart");
    let got = read.get(case_key(ctx, "durable")).await.expect("get");
    match mode {
        RestartMode::Durable => {
            check_value(
                ctx,
                "durable data survives restart",
                got.as_deref(),
                Some(b"kept"),
            );
        }
        RestartMode::Ephemeral => {
            check_value(
                ctx,
                "ephemeral data is lost on restart",
                got.as_deref(),
                None,
            );
        }
        RestartMode::Unsupported => {}
    }
}

/// Full-size graph values and the final chunk survive typed writes and pagination.
/// Topology restoration and source binding are checked by the owning core module.
async fn case_graph_storage<H: BackendHarness>(harness: &H, ctx: &str) {
    use ktann::api::{IndexConfig, LogicalIndexId, Metric, PartitionKey};
    use ktann::storage::keys::{LogicalKey, TreeKey};
    use ktann::storage::values::{
        GRAPH_CHUNK_BYTES, GraphChunk, IndexLifecycle, IndexManifest, PersistentValue, ValueCodec,
    };
    use ktann::storage::{LogicalRange, ReadLogicalTxn, WriteLogicalTxn};

    let backend = harness.backend();
    let index = LogicalIndexId::new(0x19504).unwrap();
    let partition = PartitionKey::new(1).unwrap();
    let tree = TreeKey::encode(&[], &[]).unwrap();
    let manifest = IndexManifest::new(
        IndexLifecycle::Active,
        index,
        IndexConfig::new(1, Metric::L2)
            .unwrap()
            .with_internal_partition_entries(128, 16384)
            .unwrap(),
        [0; 32],
        vec![],
    )
    .unwrap();
    let key = LogicalKey::GraphDescriptor {
        index,
        tree_key: tree.clone(),
        partition,
    };
    let codec = ValueCodec::for_index(&manifest);
    // Canonical descriptor fixture: a sufficiently large node bound permits the body.
    let mut bytes = vec![0x12, if cfg!(target_endian = "little") { 1 } else { 2 }];
    bytes.extend_from_slice(&9000_u32.to_be_bytes());
    bytes.extend_from_slice(&1_u64.to_be_bytes());
    bytes.extend_from_slice(&((GRAPH_CHUNK_BYTES + 1) as u64).to_be_bytes());
    bytes.extend_from_slice(&[0; 64]);
    let descriptor = codec.decode(&key, bytes.into()).unwrap();
    let mut write = WriteLogicalTxn::for_index(
        backend.begin_write().await.unwrap(),
        &manifest,
        backend.hard_limits(),
        backend.admission_budget(),
    );
    write.put(key.clone(), descriptor.clone()).await.unwrap();
    let mut expected = Vec::new();
    for (chunk, length) in [(0, GRAPH_CHUNK_BYTES), (1, 1)] {
        let key = LogicalKey::GraphChunk {
            index,
            tree_key: tree.clone(),
            partition,
            chunk,
        };
        let value = PersistentValue::GraphChunk(
            GraphChunk::new(Bytes::from(vec![chunk as u8; length])).unwrap(),
        );
        write.put(key.clone(), value.clone()).await.unwrap();
        expected.push((key, value));
    }
    write.commit().await.unwrap();
    let mut read = ReadLogicalTxn::for_index(backend.begin_read().await.unwrap(), &manifest);
    let stored = read.get(key).await.unwrap();
    assert!(
        stored == Some(descriptor),
        "case `{ctx}`: graph descriptor round-trips"
    );
    let range = LogicalRange::graph_chunks(&manifest, &tree, partition).unwrap();
    let mut cursor = None;
    let mut found = Vec::new();
    loop {
        let page = read
            .scan(
                &range,
                cursor.as_ref(),
                ScanLimits {
                    item_limit: 1,
                    byte_limit: 128 * 1024,
                },
            )
            .await
            .unwrap();
        for item in page.items() {
            found.push((item.key().clone(), item.value().clone()));
        }
        cursor = page.into_next_cursor();
        if cursor.is_none() {
            break;
        }
    }
    assert!(
        found == expected,
        "case `{ctx}`: complete ordered graph chunks round-trip"
    );
}
