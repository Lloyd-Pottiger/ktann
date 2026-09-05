//! Shared backend contract suite and harness.
//!
//! This module owns the backend-neutral transaction contract cases and the
//! [`BackendHarness`] seam every adapter implements to run them. Cases drive a
//! backend exclusively through the public [`ktann::storage::backend`]
//! interface, so they run unchanged against the deterministic test backend and
//! each production adapter. Each case is a small, named async function with a
//! stable replay seed; on failure it reports the case name and seed together
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

/// The identity of one case invocation, used to label failures for replay.
///
/// The seed is fixed per case and does not drive any randomness: the suite is
/// fully deterministic, so re-running the same case against a fresh harness
/// reproduces the failure exactly.
struct CaseContext {
    name: &'static str,
    seed: u64,
}

impl CaseContext {
    const fn new(name: &'static str, seed: u64) -> Self {
        Self { name, seed }
    }
}

const SEED_DECLARED: u64 = 1;
const SEED_SNAPSHOT: u64 = 2;
const SEED_READ_YOUR_WRITES: u64 = 3;
const SEED_ORDERED_WRITES: u64 = 4;
const SEED_PAGINATION: u64 = 5;
const SEED_UNIQUE_INSERT: u64 = 6;
const SEED_CONFLICT: u64 = 7;
const SEED_ABA: u64 = 8;
const SEED_ROLLBACK: u64 = 9;
const SEED_RANGE_CLEAR: u64 = 10;
const SEED_ABORT: u64 = 11;
const SEED_UNKNOWN_APPLIED: u64 = 12;
const SEED_UNKNOWN_NOT_APPLIED: u64 = 13;
const SEED_DURABILITY: u64 = 14;
const SEED_SCAN_NO_SKIP: u64 = 15;
const SEED_SCAN_EMPTY: u64 = 16;
const SEED_BATCH_SCAN: u64 = 17;

/// Runs every applicable contract case against `harness`.
///
/// Cases whose capability is unavailable ([`FaultInjection::Unavailable`] or
/// [`RestartMode::Unsupported`]) are skipped by the suite itself, not by the
/// adapter. Case order is stable so a replay against a fresh harness observes
/// the same sequence of operations.
pub async fn run_suite<H: BackendHarness>(harness: &H) {
    let ctx = CaseContext::new("declared_limits", SEED_DECLARED);
    case_declared_limits_and_capabilities(harness, &ctx).await;

    let ctx = CaseContext::new("snapshot_consistency", SEED_SNAPSHOT);
    case_snapshot_consistency(harness, &ctx).await;

    let ctx = CaseContext::new("read_your_writes", SEED_READ_YOUR_WRITES);
    case_read_your_writes(harness, &ctx).await;

    let ctx = CaseContext::new("ordered_writes", SEED_ORDERED_WRITES);
    case_ordered_writes(harness, &ctx).await;

    let ctx = CaseContext::new("scan_pagination", SEED_PAGINATION);
    case_scan_pagination(harness, &ctx).await;

    let ctx = CaseContext::new("unique_insert", SEED_UNIQUE_INSERT);
    case_unique_insert(harness, &ctx).await;

    let ctx = CaseContext::new("update_protected_conflict", SEED_CONFLICT);
    case_update_protected_conflict(harness, &ctx).await;

    let ctx = CaseContext::new("aba_conflict", SEED_ABA);
    case_aba_conflict(harness, &ctx).await;

    let ctx = CaseContext::new("rollback_and_drop", SEED_ROLLBACK);
    case_rollback_and_drop(harness, &ctx).await;

    let ctx = CaseContext::new("range_clear_capability", SEED_RANGE_CLEAR);
    case_range_clear_capability(harness, &ctx).await;

    let ctx = CaseContext::new("commit_definite_abort", SEED_ABORT);
    case_commit_definite_abort(harness, &ctx).await;

    let ctx = CaseContext::new("commit_unknown_applied", SEED_UNKNOWN_APPLIED);
    case_commit_unknown_applied(harness, &ctx).await;

    let ctx = CaseContext::new("commit_unknown_not_applied", SEED_UNKNOWN_NOT_APPLIED);
    case_commit_unknown_not_applied(harness, &ctx).await;

    let ctx = CaseContext::new("durability_mapping", SEED_DURABILITY);
    case_durability_mapping(harness, &ctx).await;

    let ctx = CaseContext::new("scan_no_skip", SEED_SCAN_NO_SKIP);
    case_scan_no_skip(harness, &ctx).await;

    let ctx = CaseContext::new("scan_empty_range", SEED_SCAN_EMPTY);
    case_scan_empty_range(harness, &ctx).await;

    let ctx = CaseContext::new("batch_scan", SEED_BATCH_SCAN);
    case_batch_scan(harness, &ctx).await;
}

// ---------------------------------------------------------------------------
// Redacted assertion helpers.
//
// These helpers never interpolate raw keys or values into a panic message, so a
// failing case leaks only its name, seed, and a safe category or count.
// ---------------------------------------------------------------------------

/// Fails with the case name, replay seed, and a safe detail.
#[track_caller]
fn fail(ctx: &CaseContext, what: &str, detail: impl fmt::Display) -> ! {
    panic!(
        "case `{}` (seed {:#x}): {what}: {detail}",
        ctx.name, ctx.seed
    );
}

/// Asserts a stable, redacted error category.
#[track_caller]
fn check_kind(ctx: &CaseContext, what: &str, got: ErrorKind, expected: ErrorKind) {
    if got != expected {
        fail(
            ctx,
            what,
            format_args!("expected {expected:?}, got {got:?}"),
        );
    }
}

/// Asserts a count.
#[track_caller]
fn check_count(ctx: &CaseContext, what: &str, got: usize, expected: usize) {
    if got != expected {
        fail(ctx, what, format_args!("expected {expected}, got {got}"));
    }
}

/// Asserts a unique-insert outcome.
#[track_caller]
fn check_insert(ctx: &CaseContext, what: &str, got: InsertOutcome, expected: InsertOutcome) {
    if got != expected {
        fail(
            ctx,
            what,
            format_args!("expected {expected:?}, got {got:?}"),
        );
    }
}

/// Asserts a boolean condition without printing the operands.
#[track_caller]
fn check_true(ctx: &CaseContext, what: &str, condition: bool) {
    if !condition {
        fail(ctx, what, "condition is false");
    }
}

/// Asserts that `got` is present and equals `expected` without printing bytes.
#[track_caller]
fn check_present(ctx: &CaseContext, what: &str, got: Option<&[u8]>, expected: &[u8]) {
    match got {
        Some(value) if value == expected => {}
        Some(value) => fail(
            ctx,
            what,
            format_args!(
                "value mismatch (got {} bytes, expected {} bytes)",
                value.len(),
                expected.len(),
            ),
        ),
        None => fail(
            ctx,
            what,
            format_args!("expected present ({} bytes), got absent", expected.len()),
        ),
    }
}

/// Asserts that `got` is absent without printing bytes.
#[track_caller]
fn check_absent(ctx: &CaseContext, what: &str, got: Option<&[u8]>) {
    if got.is_some() {
        fail(ctx, what, "expected absent, got present");
    }
}

/// Asserts a batch read against expected presence and values without printing
/// bytes.
#[track_caller]
fn check_batch(ctx: &CaseContext, what: &str, got: &[Option<Bytes>], expected: &[Option<&[u8]>]) {
    if got.len() != expected.len() {
        fail(
            ctx,
            what,
            format_args!("length mismatch ({} vs {})", got.len(), expected.len()),
        );
    }
    for (index, (got, expected)) in got.iter().zip(expected).enumerate() {
        match (got.as_deref(), *expected) {
            (Some(value), Some(expected)) if value == expected => {}
            (Some(value), Some(expected)) => fail(
                ctx,
                what,
                format_args!(
                    "item {index}: value mismatch (got {} bytes, expected {} bytes)",
                    value.len(),
                    expected.len(),
                ),
            ),
            (Some(_), None) => fail(ctx, what, format_args!("item {index}: expected absent")),
            (None, Some(expected)) => fail(
                ctx,
                what,
                format_args!("item {index}: expected present ({} bytes)", expected.len()),
            ),
            (None, None) => {}
        }
    }
}

// ---------------------------------------------------------------------------
// Key and range builders.
// ---------------------------------------------------------------------------

/// Builds a case-scoped key so distinct cases cannot collide within one backend.
fn case_key(ctx: &CaseContext, suffix: &str) -> Bytes {
    let mut bytes = Vec::with_capacity(b"contract/".len() + ctx.name.len() + 1 + suffix.len());
    bytes.extend_from_slice(b"contract/");
    bytes.extend_from_slice(ctx.name.as_bytes());
    bytes.push(b'/');
    bytes.extend_from_slice(suffix.as_bytes());
    Bytes::from(bytes)
}

/// Builds the `[start, end)` range covering every key under `prefix`.
fn case_subrange(ctx: &CaseContext, prefix: &str) -> KeyRange {
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
async fn case_declared_limits_and_capabilities<H: BackendHarness>(harness: &H, ctx: &CaseContext) {
    let backend = harness.backend();
    let hard_limits = backend.hard_limits();
    let budget = backend.admission_budget();
    check_true(
        ctx,
        "hard key limit is positive",
        hard_limits.max_key_bytes > 0,
    );
    check_true(
        ctx,
        "hard value limit is positive",
        hard_limits.max_value_bytes > 0,
    );
    check_true(
        ctx,
        "mutation count budget is positive",
        budget.max_mutations > 0,
    );
    check_true(
        ctx,
        "mutation byte budget is positive",
        budget.max_mutation_bytes > 0,
    );
    // The range-clear capability flag is verified by the range-clear case.
    let _ = backend.capabilities();
}

/// An open read snapshot stays isolated from later commits.
async fn case_snapshot_consistency<H: BackendHarness>(harness: &H, ctx: &CaseContext) {
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
    check_present(
        ctx,
        "old snapshot keeps old value",
        old_value.as_deref(),
        b"old",
    );
    let later = old.get(case_key(ctx, "later")).await.expect("read later");
    check_absent(ctx, "old snapshot does not see later key", later.as_deref());
}

/// A write transaction reads its own uncommitted writes.
async fn case_read_your_writes<H: BackendHarness>(harness: &H, ctx: &CaseContext) {
    let backend = harness.backend();
    let key = case_key(ctx, "k");
    let mut txn = backend.begin_write().await.expect("begin write");
    let before = txn.get(key.clone()).await.expect("get");
    check_absent(ctx, "absent before put", before.as_deref());

    txn.put(key.clone(), value(b"v")).await.expect("put");
    let after_put = txn.get(key.clone()).await.expect("get");
    check_present(ctx, "own put is visible", after_put.as_deref(), b"v");

    txn.delete(key.clone()).await.expect("delete");
    let after_delete = txn.get(key).await.expect("get");
    check_absent(ctx, "own delete is visible", after_delete.as_deref());

    txn.rollback().await;
}

/// A batch mutation applies in input order and reads back through the overlay.
async fn case_ordered_writes<H: BackendHarness>(harness: &H, ctx: &CaseContext) {
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
    check_count(ctx, "ordered batch scan count", page.items().len(), 1);
    check_present(
        ctx,
        "ordered batch scan value",
        Some(page.items()[0].value().as_ref()),
        b"second",
    );

    txn.commit().await.expect("commit");
}

/// A scan returns ordered pages with a strictly advancing cursor and rejects
/// zero limits.
async fn case_scan_pagination<H: BackendHarness>(harness: &H, ctx: &CaseContext) {
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
    check_count(ctx, "first page count", first.items().len(), 2);
    check_true(
        ctx,
        "first page first key",
        first.items()[0].key().as_ref() == case_key(ctx, "scan/a").as_ref(),
    );
    check_true(
        ctx,
        "first page second key",
        first.items()[1].key().as_ref() == case_key(ctx, "scan/b").as_ref(),
    );

    let next = first.next_start().expect("non-terminal cursor").clone();
    let mut expected_next = case_key(ctx, "scan/b").to_vec();
    expected_next.push(0x00);
    check_true(
        ctx,
        "cursor is the byte-successor of the last key",
        next.as_ref() == expected_next.as_slice(),
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
    check_count(ctx, "second page count", second.items().len(), 1);
    check_present(
        ctx,
        "second page value",
        Some(second.items()[0].value().as_ref()),
        b"12345",
    );
    check_true(
        ctx,
        "second page is terminal",
        second.next_start().is_none(),
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
    check_kind(
        ctx,
        "zero scan limit is invalid",
        error.kind(),
        ErrorKind::InvalidArgument,
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
async fn case_scan_no_skip<H: BackendHarness>(harness: &H, ctx: &CaseContext) {
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
    check_true(
        ctx,
        "item-bounded paging has no gap or duplicate",
        item_keys == expected,
    );
    check_count(ctx, "item-bounded page count", item_pages, 3);

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
    check_true(
        ctx,
        "byte-bounded paging has no gap or duplicate",
        byte_keys == expected,
    );
    check_count(ctx, "byte-bounded page count", byte_pages, suffixes.len());
}

/// A scan of an inverted or keyless range yields an empty terminal page, never
/// a continuation.
async fn case_scan_empty_range<H: BackendHarness>(harness: &H, ctx: &CaseContext) {
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
    check_count(ctx, "inverted range is empty", inverted.items().len(), 0);
    check_true(ctx, "inverted range is terminal", inverted.is_terminal());

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
    check_count(ctx, "keyless range is empty", gap.items().len(), 0);
    check_true(ctx, "keyless range is terminal", gap.is_terminal());
}

/// A batched scan returns one independently paginated page per input range,
/// preserving input order, duplicates, and empty or inverted ranges.
async fn case_batch_scan<H: BackendHarness>(harness: &H, ctx: &CaseContext) {
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
    check_count(ctx, "empty batch page count", none.len(), 0);

    let inverted = KeyRange::new(case_key(ctx, "b/2").to_vec(), case_key(ctx, "b/1").to_vec());
    let pages = txn
        .batch_scan(
            &[range_a.clone(), range_b.clone(), inverted, range_a.clone()],
            limits,
        )
        .await
        .expect("batch scan");
    check_count(ctx, "one page per input range", pages.len(), 4);

    // Range A paginates independently at the item limit.
    check_count(ctx, "range a first page count", pages[0].items().len(), 2);
    check_true(
        ctx,
        "range a page order",
        pages[0].items()[0].key().as_ref() == case_key(ctx, "a/1").as_ref()
            && pages[0].items()[1].key().as_ref() == case_key(ctx, "a/2").as_ref(),
    );
    let mut expected_next = case_key(ctx, "a/2").to_vec();
    expected_next.push(0x00);
    check_true(
        ctx,
        "range a cursor is the byte-successor of its last key",
        pages[0]
            .next_start()
            .is_some_and(|next| next.as_ref() == expected_next.as_slice()),
    );

    // Range B exhausts in one terminal page.
    check_count(ctx, "range b page count", pages[1].items().len(), 2);
    check_true(ctx, "range b page is terminal", pages[1].is_terminal());

    // The inverted range is a terminal empty page at its own position.
    check_count(ctx, "inverted range page count", pages[2].items().len(), 0);
    check_true(
        ctx,
        "inverted range page is terminal",
        pages[2].is_terminal(),
    );

    // A repeated range is read independently and returns the same page.
    check_true(ctx, "duplicate range repeats", pages[3] == pages[0]);

    // Resuming range A in a later batch skips no key and repeats none.
    let resume = KeyRange::new(expected_next, range_a.end().to_vec());
    let pages = txn
        .batch_scan(&[resume], limits)
        .await
        .expect("resumed batch scan");
    check_count(ctx, "resumed page count", pages[0].items().len(), 1);
    check_present(
        ctx,
        "resumed value",
        Some(pages[0].items()[0].value().as_ref()),
        b"333",
    );
    check_true(ctx, "resumed page is terminal", pages[0].is_terminal());

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
    check_count(ctx, "oversized first item alone", pages[0].items().len(), 1);
    check_true(ctx, "oversized page is terminal", pages[0].is_terminal());

    // A zero limit is invalid before any work.
    let error = txn
        .batch_scan(
            &[range_a.clone()],
            ScanLimits {
                item_limit: 0,
                byte_limit: 1,
            },
        )
        .await
        .expect_err("zero item limit");
    check_kind(
        ctx,
        "zero batch scan limit is invalid",
        error.kind(),
        ErrorKind::InvalidArgument,
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
    check_count(ctx, "staged range count", pages[0].items().len(), 1);
    check_present(
        ctx,
        "staged value is visible",
        Some(pages[0].items()[0].value().as_ref()),
        b"staged",
    );
    check_count(ctx, "committed range count", pages[1].items().len(), 2);
    txn.rollback().await;
}

/// Unique insertion distinguishes inserted from existing and conflicts on
/// concurrent insertion.
async fn case_unique_insert<H: BackendHarness>(harness: &H, ctx: &CaseContext) {
    let backend = harness.backend();
    {
        let mut txn = backend.begin_write().await.expect("begin seed");
        txn.put(case_key(ctx, "existing"), value(b"old"))
            .await
            .expect("put seed");
        txn.commit().await.expect("commit seed");
    }

    let mut txn = backend.begin_write().await.expect("begin write");
    check_insert(
        ctx,
        "fresh insert",
        txn.insert(case_key(ctx, "fresh"), value(b"v"))
            .await
            .expect("insert"),
        InsertOutcome::Inserted,
    );
    check_insert(
        ctx,
        "existing insert",
        txn.insert(case_key(ctx, "existing"), value(b"v"))
            .await
            .expect("insert"),
        InsertOutcome::AlreadyExists,
    );
    let existing = txn.get(case_key(ctx, "existing")).await.expect("get");
    check_present(ctx, "existing value unchanged", existing.as_deref(), b"old");
    check_insert(
        ctx,
        "re-insert own write",
        txn.insert(case_key(ctx, "fresh"), value(b"again"))
            .await
            .expect("insert"),
        InsertOutcome::AlreadyExists,
    );
    txn.commit().await.expect("commit");
}

/// An update-protected read conflicts with a concurrent write.
async fn case_update_protected_conflict<H: BackendHarness>(harness: &H, ctx: &CaseContext) {
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
    check_kind(
        ctx,
        "conflict maps to RetryableAbort",
        error.kind(),
        ErrorKind::RetryableAbort,
    );

    let mut read = backend.begin_read().await.expect("read winner");
    let winner = read.get(key).await.expect("get");
    check_present(ctx, "winner value persists", winner.as_deref(), b"1");
}

/// Restoring a key to its original value still conflicts with a stale writer.
async fn case_aba_conflict<H: BackendHarness>(harness: &H, ctx: &CaseContext) {
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
    check_kind(
        ctx,
        "ABA conflict maps to RetryableAbort",
        error.kind(),
        ErrorKind::RetryableAbort,
    );
}

/// Rollback and dropping a transaction both persist nothing.
async fn case_rollback_and_drop<H: BackendHarness>(harness: &H, ctx: &CaseContext) {
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
async fn case_range_clear_capability<H: BackendHarness>(harness: &H, ctx: &CaseContext) {
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
        check_absent(ctx, "cleared key a is absent", a.as_deref());
        let b = read.get(case_key(ctx, "clear/b")).await.expect("get b");
        check_absent(ctx, "cleared key b is absent", b.as_deref());
        let concurrent = read
            .get(case_key(ctx, "clear/concurrent"))
            .await
            .expect("get concurrent");
        check_absent(ctx, "concurrent key is cleared", concurrent.as_deref());
        let outside = read
            .get(case_key(ctx, "outside"))
            .await
            .expect("get outside");
        check_present(ctx, "outside key is retained", outside.as_deref(), b"3");
    } else {
        let mut txn = backend
            .begin_write()
            .await
            .expect("begin unsupported clear");
        let error = txn
            .clear_range(&clear)
            .await
            .expect_err("range clear is declined");
        check_kind(
            ctx,
            "range clear is declined",
            error.kind(),
            ErrorKind::Unsupported,
        );
    }
}

/// A controlled definite abort reports `RetryableAbort` and persists nothing.
async fn case_commit_definite_abort<H: BackendHarness>(harness: &H, ctx: &CaseContext) {
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
    check_kind(
        ctx,
        "definite abort maps to RetryableAbort",
        error.kind(),
        ErrorKind::RetryableAbort,
    );

    let mut read = backend.begin_read().await.expect("read after abort");
    let got = read.get(case_key(ctx, "abort")).await.expect("get");
    check_absent(ctx, "definite abort persists nothing", got.as_deref());
}

/// A controlled unknown-applied outcome reports `CommitOutcomeUnknown` and
/// persists every mutation atomically.
async fn case_commit_unknown_applied<H: BackendHarness>(harness: &H, ctx: &CaseContext) {
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
    check_kind(
        ctx,
        "unknown applied maps to CommitOutcomeUnknown",
        error.kind(),
        ErrorKind::CommitOutcomeUnknown,
    );

    let mut read = backend.begin_read().await.expect("read after unknown");
    let a = read.get(case_key(ctx, "a")).await.expect("get a");
    check_present(ctx, "unknown applied persists a", a.as_deref(), b"1");
    let b = read.get(case_key(ctx, "b")).await.expect("get b");
    check_present(ctx, "unknown applied persists b", b.as_deref(), b"2");
}

/// A controlled unknown-not-applied outcome reports `CommitOutcomeUnknown` and
/// persists nothing.
async fn case_commit_unknown_not_applied<H: BackendHarness>(harness: &H, ctx: &CaseContext) {
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
    check_kind(
        ctx,
        "unknown not applied maps to CommitOutcomeUnknown",
        error.kind(),
        ErrorKind::CommitOutcomeUnknown,
    );

    let mut read = backend.begin_read().await.expect("read after unknown");
    let got = read.get(case_key(ctx, "a")).await.expect("get a");
    check_absent(ctx, "unknown not applied persists nothing", got.as_deref());
}

/// A restart preserves or drops committed data per the harness's restart mode.
async fn case_durability_mapping<H: BackendHarness>(harness: &H, ctx: &CaseContext) {
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
            check_present(
                ctx,
                "durable data survives restart",
                got.as_deref(),
                b"kept",
            );
        }
        RestartMode::Ephemeral => {
            check_absent(ctx, "ephemeral data is lost on restart", got.as_deref());
        }
        RestartMode::Unsupported => {}
    }
}
