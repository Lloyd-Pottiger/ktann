# Require one transactional KV interface

KTANN supports FoundationDB and RocksDB through adapters that provide consistent
snapshots, atomic multi-key transactions, ordered range scans, update-protected
reads, unique insertion, transaction limits, and explicit commit-outcome
classification. This is a Rust behavioral interface, not a cross-backend
storage protocol. Each adapter owns its physical keyspace and transaction
mapping. Adapters do not implement index algorithms; the typed index-storage
module owns logical key/value types, versioned value codecs, algorithmic
validation, and atomic record/partition operations. Reusing those codecs is an
implementation choice and does not make persisted indexes transferable.

The first stable release ships FoundationDB and RocksDB adapters. A later TiKV
adapter would implement the same Rust operations without adding a migration or
interchange contract with an existing adapter. Native dependencies are isolated in separate
`ktann-foundationdb` and `ktann-rocksdb` workspace crates around `ktann`. A
deterministic in-memory implementation exists only inside tests for behavioral
contract, conflict, unknown-commit, and crash-point injection; it is neither a
public production backend nor a persistent-format commitment.

The Rust interface uses static backend polymorphism: `Runtime<B: Backend>` and `Index<B: Backend>`, with transaction handles represented by generic associated types rather than trait objects or a closed adapter enum. `ReadTxn<'backend>` and `WriteTxn<'backend>` are Send but not required to be Sync and borrow their Backend, accurately matching libraries such as `rust-rocksdb`, whose transaction lifetime is tied to its TransactionDB. Runtime tasks clone the owning `Arc<B>` and only then begin transactions within their own async stack; a transaction may cross awaits but cannot escape that backend borrow. This exposes the backend type and transaction lifetime in internal signatures in exchange for an extensible, safe seam with no per-operation virtual dispatch, boxed futures, self-referential handles, or lifetime-erasing unsafe code.

RocksDB uses OptimisticTransactionDB with explicit snapshots for every read,
point-key conflict fencing, WAL enabled, and synchronous commits. Synchronous
FFI calls run on permit-bounded dedicated native transaction actors;
no native object, iterator, or borrowed value crosses into an async task and no
unsafe lifetime erasure is introduced. Transactional DeleteRange is
unavailable, so RocksDB uses paged point deletes. FoundationDB uses one native
transactional range clear for an entire dropped index because its affected-data
accounting charges the range boundaries rather than the data inside the cleared
range.

KTANN explicitly targets Tokio. The RocksDB adapter supports both current-thread
and multi-thread runtimes because async tasks communicate with dedicated actors
through capacity-one channels instead of calling `block_in_place`. One permit
is acquired before a snapshot or write transaction actor starts and is released
only after that actor destroys its native state. All ordinary calls, commit,
rollback, cancellation cleanup, and destruction reuse the same actor and never
reacquire admission. Consequently, retaining the configured maximum number of
transactions delays only another transaction open; it cannot prevent an
existing transaction from making progress or releasing its permit. The actor
model creates one dedicated OS thread per live transaction, bounded by public
configuration, in exchange for safe lifetime ownership and nonblocking Drop on
LocalSet, cancellation, panic, and runtime-shutdown paths. Dedicated threads
avoid a hidden dependency on Tokio's configurable blocking-pool capacity; the
public limit is the only actor-count authority. Each actor's bounded queue
rejects unbounded work accumulation. Runtime calls the backend shutdown hook
after foreground drain and waits for actor cleanup before releasing the
adapter. Direct adapter users consume it with the same asynchronous barrier
before orderly database reopen or teardown.

Adapters distinguish backend hard limits from adapter admission budgets. Stable
native key, value, and optional affected-data ceilings are facts; positive
mutation-byte, mutation-count, and scan-page budgets are conservative policy.
The typed IndexTxn exactly counts its own encoded mutations and abandons work
that exceeds an admission budget, but does not claim this predicts every native
charge such as FoundationDB conflict ranges. Backend-native size rejection
maps to TransactionTooLarge and remains final authority. Caller batches remain
atomic and are never split automatically, while structural maintenance chooses
bounded chunks before opening each write transaction. FoundationDB defaults to
a 1 MiB mutation-byte budget, 10,000 mutations, and an 80 KiB scan page rather
than treating its 10 MB affected-data ceiling as a safe working target.

Adapters also expose hard maximum encoded key and value sizes. Create uses only
the selected adapter's limits. A smaller limit restricts configurations on that
backend rather than preventing adapter initialization. Create validates the
worst-case key and every value kind implied by IndexConfig, while each actual
encoding is checked again before entering a transaction. Validation covers
every persistent key/value kind and uses that adapter's real codec encoded-
length calculation with maximum Tree Key, Record ID, dimension, field values,
and Bloom configuration rather than a parallel hand-maintained estimate.
Create success guarantees that every schema-valid single Record fits this
adapter's limits; aggregate batch limits remain TransactionTooLarge. An
oversized logical key or value is InvalidArgument with its key/value kind in
structured context.

Ordered `scan` accepts a complete physical half-open byte range plus item and byte limits and returns ordered key/value items with an optional `next_start` physical key. When present, it is the strict lower bound after the page's last key and is passed back with the original end bound; the adapter generates it and core never computes an arbitrary byte-key successor. This deliberately avoids a separate opaque transaction-bound continuation concept. A nonterminal page must contain at least one item and `next_start` must strictly exceed its last key; violation is a Backend error rather than a loop every caller must defend. Page byte accounting is exact encoded key plus value size. One item larger than the requested page byte limit may be returned alone, subject to backend hard limits, so pagination cannot make a valid value unreadable. Typed key codecs, not adapters, construct exact prefix ends. Core consumers request the next page only after confirming remaining search, transaction, or deletion capacity and perform no budget-external prefetch.

WriteTxn may optionally expose an atomic logical range clear. A supported clear composes with the transaction's other reads and mutations, makes prior keys invisible to new snapshots after commit, follows native affected-data accounting and adapter budgets, and makes no promise about physical space reclamation. Unsupported adapters explicitly decline the capability and typed storage falls back to bounded in-transaction scan plus point deletes. FoundationDB clears the entire Logical Index data range and deletes its Dropping Manifest in one transaction; only range boundaries, not cleared contents, count toward affected data. RocksDB v1 does not advertise transactional range clear because OptimisticTransactionDB does not support it; DB-level DeleteRange is not used as a shortcut.

Trait async methods return `impl Future + Send` directly, since static dispatch removes any object-safety need for `async_trait` boxing or explicit future GATs. Transaction operations uniformly take `&mut self`, serializing requests within one transaction; deliberate parallelism is exposed only through batch primitives with interface-defined semantics. A small `ReadOps` trait owns snapshot get/batch-get/ordered-scan, while independent ReadTxn and `WriteTxn: ReadOps` traits ensure read-only index paths never receive mutation capabilities.

Dropping a mutation future before commit begins abandons its transaction. Once commit begins, cancellation cannot establish whether it committed; the operation is not cancellation-safe and is observationally equivalent to CommitOutcomeUnknown. Replaying a complete idempotent upsert or delete can recover, while insert-if-absent requires the caller to read the authoritative record before deciding whether to retry. A RocksDB actor claims commit ownership immediately before `CommitStart::begin`; cancellation before that claim abandons the actor, while an admitted synchronous commit runs to completion even if its caller then drops the future. Cancelling reads stops scheduling new work and drops the snapshot transaction; already-running RocksDB actor calls may finish with discarded results. An explicitly cancelled or expired Search returns Cancelled and never partial hits.

Runtime admission therefore installs an owned foreground in-flight guard before commit can begin. The completion path holds that guard until the backend commit finishes even if the caller drops its future. Shutdown stops new admission and waits for these guards; it never replaces an admitted operation's real result with RuntimeClosed.

Public operation control carries an optional monotonic Instant deadline and a cloneable `tokio_util::sync::CancellationToken`. Before commit, an expired deadline returns the distinct stable DeadlineExceeded kind and explicit token cancellation returns Cancelled; internal work stops scheduling new reads, routing, or backoff. Once KTANN itself starts commit, it deliberately ignores subsequent deadline/token changes and awaits the definite backend result, so its own timeout policy does not manufacture an unknown outcome. This does not make a future safe for external task abortion: dropping it during commit retains the unknown-outcome rule above.
