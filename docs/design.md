# KTANN Vector Index Design

Status: Implementation-ready

This document specifies the first stable KTANN design and is self-contained for
implementation work. Architectural rationale is recorded separately under
[`docs/adr`](adr/); the domain vocabulary is defined in
[`CONTEXT.md`](../CONTEXT.md).

## 1. Purpose and scope

KTANN is an asynchronous Rust library that stores source vectors and a searchable approximate-nearest-neighbor index atomically in one transactional KV backend. Its index is a Tree-Key-sharded forest of incrementally maintained binary K-means trees. Leaves use partition synopses for safe predicate pruning, exact typed fields for final filtering, absolute RaBitQ7 codes for approximate ranking, and original vectors for exact reranking.

The first stable release supports FoundationDB and RocksDB. TiKV may be added
later as an independent adapter. Redis is out of scope because the design
depends on consistent snapshots, multi-key transactions, ordered scans,
update-protected reads, and explicit commit outcome semantics.

### Goals

- Atomically commit a source Vector Record and every required index membership change.
- Support insert, full-replacement upsert, exact delete, and atomic batches while the index remains online.
- Keep every committed split and merge intermediate state searchable and recoverable without a durable work queue or maintenance leader.
- Safely pre-filter leaf partitions using min-max, NULL flags, and optional Bloom filters, while guaranteeing that every returned hit satisfies the predicate.
- Reuse one index algorithm, logical value codec, and typed-storage
  implementation while each backend adapter owns its physical keyspace and
  transaction mapping. This reuse creates no cross-backend data contract.
- Bound search work, maintenance transactions, caches, queues, and retry loops.
- Prefer a small number of explicit invariants and states over speculative configurability and defensive hot-path cross-checks.

### Non-goals

- Exact global top-k or a guarantee to return k results.
- Multiple randomized recall trees. A forest is sharding by Tree Key; each record belongs to one tree.
- Caller-defined source-value encoding or indexing host business rows by callback.
- Redis support; opening, copying, migrating, or comparing persistent index
  data across backend types; or accepting the same IndexConfig on every backend.
- Online schema, metric, dimension, quantizer, Tree Key, or persistent-format migration.
- Bulk-build generations, staging indexes, durable maintenance jobs, leases, repair-on-read, or automatic index repair.
- A production in-memory backend or a fixed recall/latency SLA without benchmark evidence.

## 2. Core invariants

These invariants are authoritative. Normal hot paths rely on them rather than repeatedly re-proving them.

1. Each Vector Record has exactly one Record Location and exactly one Leaf Entry in every committed state.
2. Record, Location, Leaf Entry, affected Header/Synopsis values, and payload replacement commit atomically for a foreground mutation.
3. Every ordinary non-root partition has exactly one incoming Child Entry. During a root split, an unexposed target instead has exactly one exclusive incoming reference from the root State slot.
4. Structural movement atomically inserts the target entry, deletes the source entry, updates exact counts and cache epochs, and updates Record Location for leaf movement.
5. Partition Header count is exact. Zero count is sufficient to complete structural removal; the implementation does not rescan to re-prove emptiness.
6. Every committed topology state is searchable. Cold intermediate states may persist until a future relevant access restarts maintenance.
7. Partition Synopsis is conservative. If it returns NoMatch, no Leaf Entry in the partition can make the predicate TRUE. If it returns AllMatch, every entry does.
8. Search uses one consistent backend snapshot for Manifest validation, Tree Key enumeration, traversal, filtering, Vector Record reads, and exact reranking.
9. Persistent identifiers are never reused. Gaps are valid.
10. Any naturally encountered invariant mismatch is Corruption; it is not skipped, repaired, or normalized into an apparently successful result.

## 3. Workspace and module boundaries

The intended workspace is:

```text
ktann/                  public API, algorithm, typed storage
ktann-foundationdb/     FoundationDB adapter and physical key encoding
ktann-rocksdb/          RocksDB OptimisticTransactionDB adapter
```

The core crate owns:

- public Vector Record, schema, predicate, search, lifecycle, and error APIs;
- logical key construction and versioned value codecs;
- `IndexTxn`, which exposes domain storage operations and maintains mutation-size accounting;
- routing, search, RaBitQ7, exact rerank, cache, split/merge, and maintenance scheduling.

Backend crates own only transaction mechanics, physical prefix encoding, limits, error classification, and optional capabilities. They must not implement or specialize index algorithms.

The workspace uses Rust Edition 2024 with MSRV 1.85. CI compiles and tests on both MSRV and current stable; production code must not require nightly features.

## 4. Public Rust API

Signatures below define the required public shape and ownership. Ordinary
implementation spelling may be refined without changing documented behavior,
exhaustiveness, validation, or documented behavior.

```rust
pub struct Runtime<B: Backend> { /* cheap Arc clone */ }
pub struct Index<B: Backend> { /* cheap Arc clone */ }

impl<B: Backend> Runtime<B> {
    pub fn new(backend: B, config: RuntimeConfig) -> Result<Self>;
    pub async fn create_index(&self, name: &str, config: IndexConfig) -> Result<Index<B>>;
    pub async fn open_index(&self, name: &str) -> Result<Index<B>>;
    pub async fn drop_index(&self, name: &str) -> Result<()>;
    pub async fn shutdown(&self) -> Result<()>;
}
```

`Runtime::new` requires an active Tokio multi-thread runtime, validates configuration, and immediately starts maintenance workers. Runtime and Index are `Send + Sync` and cheap to clone. Workers retain `Weak` references. Shutdown is idempotent, stops new maintenance admission, cancels queued work, waits for already-started short maintenance transactions and foreground commits, but does not wait for topology convergence. Every surviving handle returns `RuntimeClosed` afterward.

### 4.1 Records and mutations

```rust
pub struct Record {
    pub id: Bytes,
    pub vector: Arc<[f32]>,
    pub fields: Box<[Value]>,
    pub payload: Option<Bytes>,
}

pub enum Mutation {
    Insert(Record),
    Upsert(Record),
    Delete(Bytes),
}

pub enum MutationOutcome {
    Inserted,
    Upserted { replaced: bool },
    Deleted { existed: bool },
}

pub enum UpsertResult { Created, Replaced }
```

Index exposes `insert`, `upsert`, `delete`, `batch_mutate`, `get`, and `batch_get`, plus `_with_options` forms that accept OperationOptions. Single insert returns `Result<()>`, upsert returns `Result<UpsertResult>`, and delete returns `Result<bool>`. Insert is insert-if-absent and returns `RecordAlreadyExists` if the ID exists. Upsert is last-write-wins full replacement. `payload: None` deletes an old payload; `Some(empty)` stores an existing empty payload. Delete is idempotent and returns whether a record existed. The flat MutationOutcome variants are used only by mixed batch mutation and correspond exactly to each input Mutation.

An empty batch succeeds with an empty result. `batch_get` preserves input order and duplicate IDs. `batch_mutate` rejects duplicate IDs and commits in one transaction; it never splits. It returns an input-length ordered outcome vector only after all validation and commit succeed. Any item failure returns one operation Error containing the input position and Record ID, never partial outcomes.

Record ID is opaque `1..=256` raw bytes. Payload is `0..=64 KiB`. A record has exactly the configured dimension and exactly one Value per schema field. Original and derived vectors must be finite; cosine rejects zero vectors.

Point reads return a distinct non-exhaustive StoredRecord rather than the write input type:

```rust
#[non_exhaustive]
pub struct StoredRecord {
    pub id: Bytes,
    pub vector: Arc<[f32]>,
    pub fields: Box<[Value]>,
    pub payload: PayloadProjection,
}

pub enum PayloadProjection {
    NotLoaded,
    Absent,
    Present(Bytes),
}
```

PayloadProjection is deliberately closed so callers can exhaustively distinguish a skipped read, no stored payload, and stored bytes including empty. StoredRecord implements Clone and redacted/safe Debug but not PartialEq/Eq, avoiding a public floating-point record-equivalence contract. `get(id, options)` returns `Option<StoredRecord>`. `batch_get(ids, options)` is same-order and same-length with `Option` per input; duplicate IDs produce repeated cheap clones while storage may load once. Record Location is never exposed.

### 4.2 Schema, values, and predicates

```rust
pub struct FieldId(pub u16);

pub enum Metric { L2, Cosine, InnerProduct }

pub enum DataType { Bool, I64, F64, String }

#[non_exhaustive]
pub enum SynopsisConfig {
    MinMax,
    MinMaxBloom { expected_distinct: NonZeroU32, false_positive_rate: f64 },
}

pub enum CompareOp { Eq, NotEq, Lt, LessOrEqual, Gt, GreaterOrEqual }

pub struct FieldSchema {
    pub name: String,
    pub data_type: DataType,
    pub nullable: bool,
    pub synopsis: SynopsisConfig,
}

#[non_exhaustive]
pub enum Value {
    Null,
    Bool(bool),
    I64(i64),
    F64(f64),
    String(String),
}

pub enum Predicate {
    And(Vec<Predicate>),
    Or(Vec<Predicate>),
    Not(Box<Predicate>),
    Compare { field: FieldId, op: CompareOp, value: Value },
    In { field: FieldId, values: Vec<Value> },
    IsNull(FieldId),
    IsNotNull(FieldId),
}
```

Field names are nonempty UTF-8, at most 255 bytes, unique by original case-sensitive bytes, and are not normalized. Strings are UTF-8, at most 1 KiB, and compare by raw UTF-8 bytes. F64 is finite and canonicalizes `-0.0` to `+0.0`. The only NULL representation is `Value::Null`.

Predicate limits are 1,024 AST nodes, depth 64, and 1,024 values in one IN. `And([])` is TRUE, `Or([])` and `In([])` are FALSE. Compare-to-NULL and NULL inside IN are invalid; callers use IsNull/IsNotNull. Range is expressed by two comparisons joined with And.

One schema-owned validator is used by create configuration, record mutation, predicate compilation, and persistent decoding. A validated predicate compiles within one search into an internal SQL three-valued evaluator and Tree Key range plan; it is never persisted.

### 4.3 Search

```rust
pub struct SearchRequest {
    pub vector: Arc<[f32]>,
    pub k: usize,
    pub predicate: Option<Predicate>,
    pub options: SearchOptions,
}

pub struct SearchHit {
    pub record_id: Bytes,
    pub distance: f64,
}

#[non_exhaustive]
pub struct SearchOptions {
    /* private optional NonZeroUsize overrides with consuming with_* setters */
}

#[non_exhaustive]
pub struct SearchBudgetUsage {
    pub tree_keys: usize,
    pub partitions: usize,
    pub leaf_entries: usize,
    pub rerank_candidates: usize,
}

#[non_exhaustive]
pub struct SearchBudgetExhaustion {
    pub tree_keys: bool,
    pub partitions: bool,
    pub leaf_entries: bool,
    pub rerank_candidates: bool,
}

#[non_exhaustive]
pub struct SearchOutcome {
    pub hits: Vec<SearchHit>,
    pub usage: SearchBudgetUsage,
    pub exhausted: SearchBudgetExhaustion,
    pub rabitq_overlap_truncated: bool,
}
```

`Index::search(SearchRequest)` and `search_with_control(request, OperationOptions)` return at most k hits. SearchHit deliberately contains no fields or payload. A later get uses a new snapshot; KTANN has no cross-call ReadSession.

Search options contain optional nonzero budget overrides. None uses the Runtime
default. Values above a fixed hard cap are InvalidArgument, not silently
clamped. `k` is in `1..=65,536`; effective rerank budget must be at least k.
When no rerank budget is supplied, checked arithmetic computes
`min(max(4 * k, 100), 65,536)`. The 4,096 value is the default ceiling only
for runtime-configured overrides, not a second validity cap on the k-derived
default.

### 4.4 Operation control and errors

```rust
pub struct OperationOptions {
    pub deadline: Option<std::time::Instant>,
    pub cancellation: Option<tokio_util::sync::CancellationToken>,
}

#[non_exhaustive]
pub enum ErrorKind {
    InvalidArgument,
    IndexAlreadyExists,
    IndexNotFound,
    IndexDropping,
    RecordAlreadyExists,
    UnsupportedFormat,
    TransactionTooLarge,
    ContentionExhausted,
    CommitOutcomeUnknown,
    IdExhausted,
    DeadlineExceeded,
    Cancelled,
    RuntimeClosed,
    Backend,
    Corruption,
}
```

`GetOptions` has one boolean `include_payload`, false by default, with a
consuming setter. `SearchOptions`, `VerifyOptions`, `RuntimeConfig`, and
`IndexConfig` keep fields private and expose getters plus consuming builders so
construction cannot bypass validation. Result/report structs have public fields
and are non-exhaustive. Closed semantic result enums such as UpsertResult,
MutationOutcome, and PayloadProjection remain exhaustive; extensible protocol,
configuration, and error enums are non-exhaustive.

The public `Error` implements `std::error::Error`, exposes `kind() -> ErrorKind`
and redacted logical context, and retains backend/internal sources without
exposing vendor error types in signatures. ErrorKind is stable and
non-exhaustive; Display never contains raw names, keys, fields, vectors, or
payloads.

ErrorKind is stable for matching. Error may carry backend, operation, Logical Index ID, Tree Key hash, Partition Key, Record ID, and batch input position. Default Display never renders raw Tree Keys, field values, vectors, or payloads. Backend sources are chained without promising their concrete type.

A deadline or cancellation prevents new reads, routing, backoff, or commit. Once KTANN starts commit it awaits the definite result instead of manufacturing an unknown outcome. External future drop during commit is still not cancellation-safe and may leave the caller with an unknown result.

## 5. Configuration

Configuration is split by ownership:

- `IndexConfig` is caller-supplied, persisted, and immutable.
- `RuntimeConfig` is process-local.
- `SearchOptions` overrides one request's search defaults.
- adapter configuration owns backend-specific resources such as RocksDB blocking concurrency.

Public configuration structs have private fields, are `#[non_exhaustive]`, and do not derive or feature-gate serde; applications that need file formats own their DTO and conversion. `IndexConfig::new(dimension, metric, fields)` accepts mandatory values and provides consuming `with_tree_key(...)` and `with_partition_entries(min, max)` setters. `FieldSchema::new(name, data_type)` defaults to non-null MinMax and provides consuming `nullable()` and `with_synopsis(...)`. Constructors perform cheap local validation; create performs schema-wide codec/backend validation. RuntimeConfig implements Default and consuming `with_*` setters, with cross-field validation in Runtime::new.

IndexConfig contains only dimension, metric, ordered field schema, ordered Tree Key FieldIds, per-field SynopsisConfig, and min/max partition entries. RaBitQ7, binary fanout, Lloyd rounds, rotation algorithm, persistent codecs, and safety hard caps are fixed v1 protocol.

Validation includes:

| Item | Contract |
| --- | --- |
| Dimension | `1..=16,384` |
| Fields | at most 16 |
| Bloom fields | at most 4 |
| Tree Key | ordered unique non-nullable FieldIds; may be empty for one tree |
| Encoded Tree Key | create verifies the schema's worst case is at most 8 KiB |
| Partition minimum | at least 1; default 16 |
| Partition maximum | at least 2; default 128 |
| Threshold relation | checked `2 * min <= max` |
| Synopsis value | at most 64 KiB encoded |

The dimension ceiling is an algorithm/resource bound. Create also uses only the
current adapter's key/value admission limits. It invokes that adapter's real
codec encoded-length logic for every key/value kind at maximum schema-valid
Record ID, Tree Key, dimension, fields, and Bloom shape. Create success
guarantees that every individually schema-valid record can be encoded by that
adapter; a caller batch can still exceed aggregate transaction limits.

Persisted Manifest stores canonical caller config separately from derived config such as Logical Index ID, generated rotation seed, concrete Bloom bit/hash counts, and codec versions. Idempotent create compares only canonical caller config.

Runtime defaults and hard caps are:

| Setting | Default | Runtime hard cap / validation |
| --- | ---: | ---: |
| Maintenance workers | `min(available_parallelism, 8)`, min 1 | must be >= 1 |
| Pending/running fixups | 1,024 | total pending + running; must be >= workers |
| Fixup retries | 8 | must be >= 1 |
| Foreground mutation attempts | 8 | must be >= 1 |
| Partition cache | 256 MiB | 0 disables |
| Scanned Tree Keys | 4,096 | 65,536 |
| Visited partitions | 1,024 | 16,384 |
| Visited Leaf Entries | 65,536 | 1,048,576 |
| Exact rerank candidates | `min(max(4*k,100),65,536)`; configured default <= 4,096 | 65,536 |
| TreeKey scan ranges | at most 1,024 | wider safe fallback |
| Import in-flight batches | `min(available_parallelism,4)`, min 1 | configured positive |
| Import backlog watermark | half Runtime queue | within queue capacity |

Stalled timeout defaults to checked `max(1ms, 1s * max_partition_entries / 128)` and must be positive and representable.

Partition cache capacity is configured as u64 bytes; zero alone disables it, and Runtime::new rejects values that cannot convert to platform usize. Retry backoff starts at 1 ms, doubles to a 100 ms cap, and uses full jitter uniformly from zero through the current cap. Foreground mutations and fixups use the same algorithm with separate attempt counters; deadline/cancellation may stop before the next attempt.

## 6. Transactional KV contract

`Runtime<B>` and `Index<B>` use static backend polymorphism. A single Backend trait owns generic associated transaction types borrowing the backend, matching `rust-rocksdb::Transaction<'db, DB>` without unsafe lifetime erasure:

```rust
pub trait Backend: Send + Sync + 'static {
    type ReadTxn<'a>: ReadTxn + Send where Self: 'a;
    type WriteTxn<'a>: WriteTxn + Send where Self: 'a;

    fn begin_read(&self) -> impl Future<Output = Result<Self::ReadTxn<'_>>> + Send;
    fn begin_write(&self) -> impl Future<Output = Result<Self::WriteTxn<'_>>> + Send;
    fn constraints(&self) -> BackendConstraints;
}
```

Stable Rust supports this GAT plus RPITIT shape without `async_trait` or boxed futures. ReadOps contains only `get`, same-order same-length `batch_get`, and bounded `scan`. WriteTxn extends ReadOps with `get_for_update`, `batch_get_for_update`, `put`, unique `insert`, `delete`, `batch_mutate`, optional range clear, and consuming `commit(self)` / `rollback(self)`. Dropping a transaction never commits; explicit rollback releases backend resources promptly. There is no reverse scan, exists, or multi-range primitive in the common interface.

The semantic contract is:

- one consistent read snapshot;
- atomic read/write transaction;
- snapshot get, batch-get, and ordered forward scan;
- update-protected get and batch-get for conflict establishment;
- put, unique insert, delete, and batch mutation;
- commit and rollback;
- optional transactional logical range clear;
- explicit backend hard limits, adapter admission budgets, and capabilities;
- errors classified as RetryableAbort, CommitOutcomeUnknown, or Permanent.

Transaction operations use `&mut self`; intentional concurrency is represented by batch primitives. ReadTxn and WriteTxn are distinct small traits sharing ReadOps. There is no general CAS. State transitions use update-protected read, typed validation, and put. Unique insert is reserved for internal first creation.

`BackendConstraints` separates facts about the storage engine from conservative
adapter policy:

```rust
pub struct BackendConstraints {
    pub hard: BackendHardLimits,
    pub budgets: BackendBudgets,
    pub capabilities: BackendCapabilities,
}

pub struct BackendHardLimits {
    pub max_key_bytes: Option<NonZeroUsize>,
    pub max_value_bytes: Option<NonZeroUsize>,
    pub max_transaction_affected_bytes: Option<NonZeroUsize>,
}

pub struct BackendBudgets {
    pub max_encoded_key_bytes: NonZeroUsize,
    pub max_encoded_value_bytes: NonZeroUsize,
    pub max_transaction_mutation_bytes: NonZeroUsize,
    pub max_transaction_mutations: NonZeroUsize,
    pub scan_page_bytes: NonZeroUsize,
}

pub struct BackendCapabilities {
    pub transactional_range_clear: bool,
}
```

A hard-limit field is present only for a stable backend-native limit with the
declared accounting semantics. Absence means no reportable hard limit, not that
resources are unlimited. Budgets are positive adapter-selected admission and
pagination policy, not backend facts. KTANN imposes no shared storage-size
profile; create validates encoded keys and values against only the selected
adapter's admission budgets.
An adapter configuration may lower budgets. It cannot set a budget above a
corresponding known hard limit, but a budget with no corresponding hard-limit
field remains valid. RocksDB therefore reports no fictional native key/value
ceiling while still enforcing finite configured admission budgets.

An adapter may transparently retry only a pre-commit, outcome-definite,
idempotent transport read; it never retries commit or replays a transaction.
KTANN retries the complete logical mutation in a fresh transaction after
RetryableAbort. It never retries CommitOutcomeUnknown. The FoundationDB adapter
uses an explicit transaction and one commit rather than `Database::run` or an
`on_error` loop. It classifies maybe-committed errors, including a commit-time
cluster-version change, timeout, or cancellation with uncertain outcome, as
CommitOutcomeUnknown before considering retryability; only definitely
not-committed retryable errors become RetryableAbort.

### 6.1 Ordered scans

`scan` accepts a complete physical half-open `[start, end)` range plus item and byte limits. `ScanPage` contains ordered key/value items and `next_start: Option<Bytes>`. When present, `next_start` is the strict lower bound immediately after the page's last returned key; the next call scans `[next_start, original_end)`. It is an ordinary physical key, not an opaque transaction-bound token. The adapter generates it, and core never computes an arbitrary byte-key successor. A nonterminal page must contain an item and `next_start` must strictly exceed its last key; violation is Backend error. Exact page bytes are encoded key plus value bytes. One item above the requested page-byte limit may occupy a page alone, subject to backend hard limits. Core typed codecs construct prefix ends; adapters do not interpret logical prefixes. Consumers do not prefetch beyond confirmed search, transaction, or delete capacity. A new transaction may reuse a saved logical lower bound where the caller's protocol permits it, although current structural/delete recovery normally restarts at its prefix beginning.

When a native range byte target is soft, the adapter locally truncates the
returned batch by exact encoded key-plus-value bytes. If native results remain,
it returns `next_start` after the last delivered item even if the native response
itself says there is no further batch. Adapter-side pagination therefore neither
violates the common limit nor loses rows.

### 6.2 Transaction sizing

IndexTxn counts its encoded mutation key/value bytes and mutation count as it
builds a transaction. Exceeding an adapter admission budget abandons the
transaction early with TransactionTooLarge; this is exact only for KTANN's
budget accounting, not a proof that a transaction below budget satisfies every
backend-native accounting rule. A backend-native aggregate rejection maps to
the same kind and remains final authority. Single-key or single-value hard-limit
failure is InvalidArgument with key/value-kind context. User batches never
auto-split. Maintenance selects bounded chunks before opening a write
transaction.

FoundationDB reports 10,000-byte key, 100,000-byte value, and 10,000,000-byte
transaction affected-data hard limits. Its default adapter budgets are 1 MiB
of encoded mutations, 10,000 mutations, and an 80 KiB scan page. The affected-
data limit additionally counts mutation range boundaries and read/write
conflict ranges, so IndexTxn neither predicts it exactly nor uses the 10 MB
ceiling as its normal batching target. A FoundationDB range-read byte target is
soft and may return one larger item; the common scan contract still guarantees
progress.

FoundationDB transactions normally lose the ability to perform remote reads
about five seconds after their first read. KTANN keeps write transactions short
and never tries to renew one logical snapshot across transactions. Search obeys
caller deadline/cancellation and reports a native snapshot-expiry failure as a
backend error rather than imposing a hidden four-second deadline. Large verify
runs may therefore need an offline copy or backend-native snapshot facility.

### 6.3 Range clear

Optional `clear_range` must compose atomically with the transaction's reads and
writes, make prior keys invisible to new snapshots after commit, follow the
backend's affected-data accounting and adapter budgets, and make no physical
reclamation promise. A declared
`transactional_range_clear` capability is checked before the method is called;
an unsupported response after advertising the capability is a Backend contract
violation. Capability absence is an internal branch that always falls back to
paged point deletion, not a public operation error.

FoundationDB advertises the capability and clears the complete
`Data[LogicalIndexId]` range in one transaction that update-protects the
Dropping Manifest and then removes it. FoundationDB accounts the range boundary
keys, not all cleared data, toward the transaction affected-data limit, so data
size does not require chunking. CommitOutcomeUnknown recovers by rereading the
Manifest: absence means drop completed; the same Dropping Manifest means cleanup
is safely retried. RocksDB v1 does not advertise the capability because
OptimisticTransactionDB has no transactional DeleteRange; core uses paged point
deletion rather than DB-level DeleteRange.

### 6.4 RocksDB execution

The adapter uses OptimisticTransactionDB. ReadTxn owns a database Snapshot.
WriteTxn enables a transaction snapshot, and every get, batch-get, and scan
explicitly binds ReadOptions to that snapshot; merely enabling the transaction
option does not make default gets snapshot reads. Reads still observe the
transaction's own pending writes. `get_for_update` establishes point conflicts;
commit Busy/TryAgain maps to RetryableAbort. Snapshot scans do not establish a
predicate/range conflict in RocksDB, so every state transition is fenced by an
update-protected State, epoch, count, or the individual keys it changes.

The Rust binding exposes no native multi-get-for-update, so the adapter may
loop over point `get_for_update` calls inside one blocking section without
promising one storage IO. Iterators and borrowed values never cross that
section. KTANN requires a Tokio multi-thread runtime. Before each synchronous
RocksDB call the adapter asynchronously acquires one permit from a bounded
semaphore, invokes the complete call with `tokio::task::block_in_place`, and
releases the permit immediately. The semaphore bounds admission; this is not a
separate blocking actor/pool. There is no unsafe owned transaction wrapper or
per-transaction actor.

RocksDB writes always keep WAL enabled and use `sync=true`. A successful commit
therefore means the atomic Vector Record and index mutations are durable across
power loss, not merely visible from a new snapshot. A synchronous commit cannot
be interrupted; externally dropping its future still has unknown outcome as
specified above.

## 7. Persistent identity and lifecycle

Within one Backend Namespace, IndexName is unique, nonempty UTF-8, and at most 255 bytes. The caller supplies this stable name. KTANN allocates a nonzero u64 Logical Index ID from a persistent monotonic namespace high-water mark one at a time. IDs never repeat.

Manifest is keyed by IndexName and contains Logical Index ID, status, format, canonical caller config, and derived config. Status is Active or Dropping.

### Create

Create update-protects the name, validates configuration and codecs against the current backend, allocates an ID, and atomically inserts an Active Manifest. Same name plus identical canonical caller config is idempotent and returns the existing Index. Different config returns IndexAlreadyExists. A commit-unknown result triggers one new-snapshot lookup by IndexName: identical Active Manifest recovers success; different config returns IndexAlreadyExists; absence returns CommitOutcomeUnknown and does not allocate another ID.

### Open

Open accepts only IndexName and reconstructs immutable config from Manifest. It rejects unsupported format/backend combinations. Dropping returns IndexDropping. Every Index operation validates that the snapshot-visible name Manifest is Active and still names the handle's Logical Index ID. Reads batch this with initial reads where possible; mutations update-protect it. Recreating the same name gets a new ID, so an old handle fails ID validation.

### Drop

Drop is synchronous, idempotent, and resumable:

```text
Active -> Dropping -> clear Data[IndexId] -> delete Manifest
```

An absent Manifest is success. Same-name create returns IndexDropping while cleanup retains the recovery entry. Each fallback delete transaction update-protects Dropping with the same ID, scans and point-deletes one bounded prefix page, and restarts from the prefix beginning after commit unknown. An empty page permits Manifest deletion. No persistent cursor is necessary because deleted leading keys immediately expose the first remaining key.

## 8. Logical keyspace and persistent values

The following layout is the algorithm's ownership model, not a shared physical
key format. Each adapter chooses its physical encoding. Within
one adapter, all data for one Logical Index ID occupies one clearable range;
Manifest and namespace ID allocator sit outside it.

```text
Manifest[IndexName]
LogicalIndexIdHighWater

Data[IndexId]/
  Records/
    [RecordId]/Record
    [RecordId]/Location
    [RecordId]/Payload

  TreeDirectory/
    [TreeKey] -> TreeManifest

  Trees/
    [TreeKey]/
      Partition[PartitionKey]/
        State
        Centroid
        Header
        Synopsis
        ChildEntry[ChildPartitionKey]
        LeafEntry[RecordId]
```

LogicalIndexId and PartitionKey are fixed-width big-endian nonzero u64; zero is invalid. Root PartitionKey is 1. TreeKey and RecordId use one self-terminating memcomparable component encoding. TreeKey is not assigned a separate TreeId and therefore repeats in partition/entry keys; this is an accepted storage/comparison cost in exchange for one identity and no directory lookup on point routing.

TreeKey is a versioned memcomparable typed tuple whose order matches the typed comparator and supports field-prefix half-open ranges. Its complete encoded component is at most 8 KiB and must be appended to physical keys without an unbounded second escaping pass.

Each value has a type tag and codec version. The Manifest declares the overall format and supported codec combination. Unsupported Manifest format returns UnsupportedFormat. Unknown tags, variants, State discriminants, or illegal combinations inside a supported format are Corruption. Values use backend checksums and do not add application CRC.

### 8.1 Vector Record and membership

Vector Record stores original f32 vector and exact typed fields. Payload is a separate optional value. Record Location stores canonical TreeKey and Leaf PartitionKey. Leaf Entry stores RecordId-keyed RaBitQ7 data plus exact typed fields. Location and membership are updated atomically; structural moves do not rewrite Vector Record or Payload.

Point get batch-reads Record and optional Payload; internal paths separately load Location when required. Public GetOptions contains only `include_payload` and determines whether PayloadProjection is NotLoaded versus Absent/Present.

### 8.2 Tree Manifest and partition allocation

Tree Manifest is both directory entry and per-tree PartitionKey high-water. The first record for a TreeKey atomically unique-inserts TreeManifest and its empty root. Empty trees persist until index drop.

Each Runtime caches `(IndexId, TreeKey) -> [next, end)` and reserves 1,024 IDs by update-protecting TreeManifest. Unused suffixes, unknown-outcome reservations, and issued-but-unused IDs are permanently abandoned. Near u64 exhaustion the final shorter suffix is allowed; exhaustion returns IdExhausted.

### 8.3 Partition values

Partition storage is fine-grained:

- State: topology variant, associated keys, codec version, state-start wall time;
- Centroid: immutable full-f32 routing centroid for non-root partitions;
- Header: level, exact count, cache epoch;
- Synopsis: strong leaf predicate summary;
- one KV per ChildEntry or LeafEntry.

Leaf level is 0. Every internal ChildEntry descends exactly one level. Non-root level and centroid are immutable. Root promotion alone increments root level. Header count is authoritative and transactionally exact.

Removing a partition uses a transactional prefix clear if supported. Fallback scan/point-delete remains inside one write transaction that validates the source's terminal maintenance state. Only an empty page permits final topology/metadata deletion.

## 9. Vector and distance semantics

Index metric is immutable: L2, cosine, or inner product. Dimension is immutable.

Write/query routing preprocessing is:

1. validate dimension and finite f32 values;
2. for cosine, compute norm in f64, reject zero, normalize, and convert each finite result to f32; L2/IP keep original values;
3. apply the index-persisted seeded Givens rotation;
4. encode/search RaBitQ7.

Internal centroids are full f32 routing vectors. Exact rerank reads the unrotated original Vector Record and accumulates in f64. L2 traversal ranks squared distance but public SearchHit returns Euclidean distance. Cosine exact rerank recomputes norm and dot in f64. Inner-product distance is negative dot. No metric output is clamped to hide numerical issues; invalid/non-finite derived values fail.

### 9.1 Absolute RaBitQ7

KTANN uses a partition-independent absolute-vector codec. Each vector uses one
sign bit and six unsigned magnitude bits per dimension, with a fixed header:

```text
scale: f32
code_norm_squared: u32
reconstruction_error_upper: f32
sign_bits: ceil(d / 8)
magnitude_bits: ceil(6 * d / 8)
```

The 12-byte header is little-endian. Sign and magnitude streams visit dimensions
in order and place their first bit at the least-significant bit of the first
byte. Unused high padding bits must be zero. The initial symmetric f32 scale is
`max(abs(x_i)) / 63`; unsigned magnitudes use nearest rounding with halfway
cases away from zero and clamp to `0..=63`. A zero magnitude always uses the
positive sign. After codes are fixed, f64 sums compute the nonnegative
least-squares scale `sum(x_i*m_i) / sum(m_i^2)`, which is stored as f32.
`code_norm_squared` stores the exact integer `sum(m_i^2)`; the dimension limit
keeps it within u32. Reconstruction and error are then computed against the
stored scale. The error is encoded as the least finite f32 not below the f64
result. A zero vector has a zero header and zero streams. Manifest dimension
determines exact payload length. Wrong length, nonzero padding, inconsistent
code norm, or non-finite/negative numeric metadata is Corruption.

Because code is absolute rather than centroid-relative, split/merge copies
unchanged bytes; only insert or vector-changing upsert re-encodes. A lossless
f32 quantizer is a test oracle, not a production option.

For query `q`, reconstructed vector `x_hat`, and reconstruction-error upper
bound `E`, rough distance and bound are:

| Metric | `d_hat` | `B` |
|---|---:|---:|
| Inner product | `-q dot x_hat` | `||q|| E` |
| Cosine | `1 - q dot x_hat` | `||q|| E` |
| Squared L2 | `max(0, ||q||^2 + ||x_hat||^2 - 2 q dot x_hat)` | `2 sqrt(d_hat) E + E^2` |

The interval is `[d_hat - B, d_hat + B]`; only its squared-L2 lower endpoint
clamps to zero. Cosine is not clamped. Non-finite decoded or derived data is
Corruption.

The scalar f64 implementation is the test oracle. Production may select
optimized f32 scalar or SIMD dot-product kernels by CPU capability. Those
kernels are tested for numerical error, interval coverage, and recall, but not
bitwise identity. Floating-point accumulation can change overlap candidates at
an interval or budget boundary, so approximate search results are not promised
to be identical across CPU classes. Exact rerank still accumulates in f64 and
orders the selected candidates by exact distance then RecordId.

## 10. Sharded forest and Tree Key routing

Each record belongs to one tree selected by the ordered immutable Tree Key fields. An empty field list means one tree. Tree Key changes during upsert atomically relocate membership between trees. There is no maximum tree count.

A predicate that fully constrains every Tree Key field through equality, finite IN, and OR routes directly. Otherwise search enumerates TreeDirectory under `max_tree_keys_scanned`:

1. compile the longest equality/IN prefix into at most 1,024 ordered memcomparable ranges;
2. if range expansion would exceed that bound, safely widen ranges rather than reject or omit keys;
3. sort and merge ranges; simple NOT may use complement ranges;
4. scan and decode TreeKeys, exactly evaluate remaining TreeKey constraints;
5. finish enumeration before traversal, then advance eligible trees fairly.

If enumeration budget blocks remaining keys, search still processes the deterministic encoded-order prefix and reports Tree Key exhaustion; this has an explicit ordering bias. There is no budget-external directory read-ahead.

## 11. Predicate filtering and Synopsis

Filter semantics are SQL WHERE three-valued logic. Exact comparison order is `false < true`, numeric i64/f64 order, and raw UTF-8 byte order; NULL is outside min/max. Every returned hit made its predicate TRUE against Leaf Entry exact fields. Search does not reread Vector Record merely to repeat predicate evaluation.

### 11.1 Persistent synopsis

Each configured field is MinMax or MinMaxBloom. Tree Key fields need no synopsis but may opt in. The schema-ordered independent codec stores kind, NULL flags, typed canonical extrema, and optional Bloom parameters/bits. At most four fields use Bloom. Create derives fixed Bloom bit/hash counts from expected distinct values (default 128) and false-positive rate (default 0.01); capacity is not a hard entry limit.

Canonical typed values use codec-versioned, domain-separated XXH3-128. Its halves drive double hashing `h1 + i*h2`. IN is NoMatch only if every candidate is excluded. Saturation only reduces pruning and is observable in metrics.

Synopsis evaluation returns an internal bitmask over possible `{TRUE, FALSE, UNKNOWN}` results. Atomic comparisons derive conservative masks. AND, OR, and NOT combine them through SQL truth tables, so NOT swaps TRUE/FALSE and retains UNKNOWN. The public classification is:

- NoMatch: TRUE absent;
- AllMatch: TRUE is the only possibility;
- MayMatch: otherwise.

AllMatch skips per-entry exact predicate execution but entries still consume visited-entry budget. Empty leaf has an empty truth mask and yields NoMatch.

The mask describes every value represented by the monotonic historical
Synopsis, a superset of current entries. NULL-present and non-NULL-present flags
only change from false to true; min/max only expand. Bloom evidence can exclude
a value and thereby help prove NoMatch, but a Bloom hit never proves presence or
TRUE and cannot by itself contribute to AllMatch. An atomic mask is `{TRUE}`
only when NULL flags and exact min/max relations prove every represented value
makes that atom TRUE. Therefore stale values left by delete/update may weaken
AllMatch to MayMatch but cannot create a false AllMatch. Property tests include
arbitrary deletion and replacement histories before comparing every current
entry against both NoMatch and AllMatch conclusions.

Foreground put/replace only expands Synopsis; delete never shrinks it. A new split target starts canonical-empty. Every actual entry relocation atomically expands only its target. Merge uses the same rule. No whole-source union or completion scan occurs.

## 12. Search algorithm

One consistent read transaction covers Manifest check, TreeDirectory enumeration, partition Header/State/Synopsis, bodies, Vector Record reads, and exact rerank. Manifest validation and initial root/Header reads are batchable into the first RPC.

Traversal is a level-scaled beam deterministic for one codec and selected
numeric kernel. Leaf-level base beam defaults to 32; moving one level toward
root divides by two with minimum one. Trees advance fairly. Ties use TreeKey,
PartitionKey, then RecordId. CPU-selected RaBitQ kernels may change boundary
candidates as specified in section 9.1. Search budget counts logical work
independent of cache/RPC batching:

- Tree Keys actually decoded and checked;
- distinct partition bodies visited, cache hit included;
- Leaf Entries read and considered, AllMatch included;
- Vector Records exactly reranked.

Exhaustion is set only when pending eligible work is prevented. Several dimensions may exhaust. New work consuming an exhausted dimension stops; already materialized permissible candidates finish. Deadline/cancellation is an error and returns no partial outcome.

### 12.1 RaBitQ candidate selection

For a leaf with `n` eligible entries, let
`r = min(n, max(checked_mul(2, k), 64))`. An empty leaf contributes nothing.
The leaf keeps the rough top `r` by estimated distance, uses the r-th smallest
upper endpoint as its overlap threshold, adds entries whose lower endpoint is
at most that threshold, and caps the result at
`min(checked_mul(4, r), remaining request rerank budget)` by lower bound,
estimated distance, then RecordId. Overflow is rejected during request
validation rather than wrapping.

After merging leaf sets, search uses the kth-smallest upper endpoint when at
least k candidates exist and positive infinity otherwise. It retains candidates
whose lower bound overlaps that threshold, sorts by estimated distance and
RecordId, truncates to the remaining request rerank budget, batch-reads Vector
Records, and exact-reranks. Any local overlap truncation sets
`rabitq_overlap_truncated`.

Exact final order is distance then RecordId bytes. Naturally observed duplicate RecordId membership is Corruption. Search does not proactively read Record Location for each candidate.

### 12.2 Intermediate-state traversal

Non-root Splitting/Draining sources and ReceivingSplit targets participate solely through current ChildEntries. They may have different parents after adjacent-level maintenance. ReceivingSplit and Merging search like normal same-level partitions and do not recursively expand a source/target family.

Root is special because its target slots temporarily replace parent references:

- root Splitting: search only root body; targets are empty/unexposed;
- root DrainingSplit: search root plus both target bodies at the same level;
- completed promotion: root is a Ready internal partition containing the two target ChildEntries.

The query deduplicates physical body work by `{TreeKey, PartitionKey}` and charges it once. Discovering one non-root child from two distinct parent ChildEntries is still Corruption.

## 13. Partition cache

The process-shared decoded-body cache defaults to 256 MiB; zero disables it. It contains complete internal ChildEntry bodies or complete leaf search bodies, including empty bodies. It excludes State, Header, Synopsis, Vector Record, and Payload. Write routing never uses it; a transaction-local cache may avoid repeated loads.

A new Header initializes `cache_epoch` to zero. For an existing partition, each
transaction that changes body bytes increments the epoch exactly once using
checked arithmetic, regardless of whether that transaction changes one entry
or both body kind and level during root promotion. State timestamps and
Synopsis-only changes do not increment it.

Search reads snapshot-visible Header before cache lookup:

- equal epoch: hit;
- cached older: miss and evict;
- cached newer: historical-snapshot miss, retain newer entry.

On miss, the same transaction scans and decodes the complete body and rechecks the same snapshot Header before publishing. Concurrent fills may duplicate work; there is no singleflight state. A cache hit still consumes logical partition budget, while Header validation I/O does not.

Eviction has one global byte capacity. Each observed tree level has independent
S3-FIFO small/main/ghost queues. When capacity is needed, the lowest level with
an eligible resident victim is chosen, then ordinary S3-FIFO chooses within
that level; empty levels are removed. Resident values, queue nodes, and ghost
keys all count toward the same capacity, so metadata is bounded. There are no
fixed per-level quotas or pinned roots; persistent leaf pressure may evict leaf
entries repeatedly while keeping small upper levels warm, which is the intended
level preference.

## 14. Foreground mutation protocol

Mutation routing begins without locks. The final typed IndexTxn update-protects the active matching Manifest, authoritative Record/Location as required, involved partition States/Headers/Synopses, and validates that the route is still legal. It then applies all source/target and source-record changes atomically.

Batch validation is all-or-nothing. Within one live transaction it may retain unaffected route decisions, clear stale transaction-local loads, reroute rejected records, and revalidate the full batch. A backend abort discards every route and retries the complete logical operation with capped exponential backoff and jitter, eight attempts by default. Exhaustion is ContentionExhausted. CommitOutcomeUnknown never enters the loop.

Insert commit unknown is returned directly; observing the same ID later cannot prove the record belongs to that insert. Idempotent full upsert/delete can be replayed by a caller, but KTANN does not automatically overwrite after unknown outcome.

## 15. Incremental tree shape

Every TreeKey begins with an empty searchable root at PartitionKey 1, created atomically with TreeManifest by its first mutation. The tree grows online only; initial import uses ordinary batch upserts. Fanout is fixed at two. Exact entry count drives thresholds. Root remains forever and never collapses.

### 15.1 Balanced K-means training

Split uses every source entry in one consistent read snapshot and performs training outside the transaction. Leaf training batch-reads original Vector Records and applies metric routing preprocessing plus ROT. Internal training uses full-f32 ChildEntry centroids. Missing source Vector Record is Corruption; the source remains searchable and the fixup retires.

Initialization is deterministic farthest pair:

1. entry farthest from source centroid;
2. entry farthest from the first seed;
3. distance ties by RecordId or Child PartitionKey.

Each Lloyd round sorts by distance difference to the two centroids, assigns exactly `floor(n/2)` left and the remainder right, and uses ID ties. Centroids accumulate in f64 and convert to finite f32. Assignment stability stops early; otherwise at most ten rounds. Training has no sampling or KTANN-specific CPU/memory cap, so abnormal source growth costs proportional resources.

## 16. Split state machine

Durable states are only Ready, Splitting, ReceivingSplit, DrainingSplit, and Merging. State stores its associated keys and state-start time. Header stores zero-based level, count, and epoch.

```text
Ready
  -> Splitting { left, right }
  -> install left ReceivingSplit
  -> install right ReceivingSplit
  -> DrainingSplit { left, right }
  -> bounded exact moves
  -> count == 0
  -> atomic completion
```

### 16.1 Start and target installation

Starting validates Ready and threshold, allocates/persists target IDs, and changes State. Splitting continues to accept foreground writes. Training snapshot may become stale; target centroids are routing models, not membership authority, and publication never validates a whole source epoch.

Targets install separately. First unique creation fixes the left centroid. A competing/recovering worker pins persisted left centroid when recomputing right; an existing right centroid also wins. Target creation always performs an update-protected read of source State in the same write transaction and requires it still be Splitting with this target in the corresponding slot. This point conflict prevents a worker that observed an old snapshot from recreating a target after source completion and deletion. For non-root, target creation and its parent ChildEntry insertion are one transaction that additionally validates:

- source still Splitting with that target ID;
- traversed parent still contains source ChildEntry and accepts a child.

Failure leaves a successfully installed other target intact for later recovery. Root targets instead occupy exclusive State slots without parent entries. Target State is `ReceivingSplit { source }`: searchable, accepts writes/moves and exact delete, but cannot split/merge.

Parent maintenance may move a ReceivingSplit ChildEntry without reading target state. Therefore targets may end up in different parents. Transition to Draining validates only source target IDs and both target States, not their current parent locations, source entries/count, or cache epoch.

### 16.2 Draining

New inserts route to the nearer target. An upsert whose Location still names source atomically deletes source membership and inserts target membership. Delete follows exact Location whether source or target. State changes cause whole-mutation retry.

Maintenance first reads a bounded source-ID page in a short snapshot, then opens a write transaction and re-reads source entry, Location, and Vector Record. A completed concurrent removal is skipped; a remaining mismatch is Corruption. Current normalized/rotated vector chooses persisted target centroid, tie by PartitionKey. The absolute RaBitQ7 bytes copy unchanged. Move atomically updates entries, Location, both counts/epochs, and target Synopsis.

No durable cursor exists. Each batch begins at the current smallest source key; successful moves delete that prefix. Internal drain uses current ChildEntry centroid and does not read child state, Location, Record, or Synopsis.

### 16.3 Completion

Exact count zero is sufficient. For a non-root source, completion first performs
an exact root-down traversal to the source's parent level and scans every
ChildEntry at that level in bounded pages, comparing PartitionKey rather than
centroid distance. It requires exactly one incoming source edge; none or more
than one is Corruption. This exceptional completion scan consumes maintenance
work, not Search Budget, and may fail on deadline/snapshot expiry.

One short transaction then update-protects and validates DrainingSplit, zero
count, and the discovered parent ChildEntry, promotes both targets to Ready,
and switches topology:

- non-root: remove source ChildEntry from the currently observed parent and delete full source prefix;
- root: convert PartitionKey 1 in place to a Ready internal root containing the two target ChildEntries and increment level.

A stale non-root parent observation fails validation and repeats the exact
parent-level scan in a fresh snapshot. No tombstone or extra completion state
remains.

## 17. Merge state machine

Only a non-root Ready partition, leaf or internal, below the minimum exact count
can start merge. No non-Ready intermediate-state partition starts another
structural operation, and a Merging source never reverts. Same-level routing
first selects another Ready candidate. The start write transaction then
update-protects and validates source State/Header, the observed source parent
ChildEntry, and the selected target State before changing source to Merging.
Any change discards the route and retries from a fresh snapshot.

Merging remains searchable and supports exact delete, but accepts no new insert. Upsert still located there must relocate to a Ready same-level target; absence after bounded retry is ContentionExhausted.

Each maintenance batch performs ordinary same-level centroid routing, skips source and non-Ready candidates, chooses the nearest Ready target, and update-protects target State in the write transaction. A conflict discards the route and selects again. There is no fixed target; different entries/batches may use different targets and targets may exceed the split threshold.

Leaf and internal drain use the same two-phase exact movement as split. Leaf re-reads Record/Location for routing and copies RaBitQ7; internal routes full-f32 child centroid without child-state reads. Source Synopsis never shrinks.

If no Ready target exists, the searchable Merging source remains, the fixup
retires, and later stalled access may restart it. It never creates a target or
reverts Ready. At count zero, completion uses the same exact bounded
parent-level scan as split. One transaction update-protects and validates
Merging, zero count, and the unique discovered parent ChildEntry, removes that
entry, and clears the source prefix. A conflict repeats discovery. No target
state changes and no tombstone remains.

## 18. Maintenance runtime

One Runtime-wide bounded in-memory queue serves every Index. Fixup identity is:

```text
(LogicalIndexId, TreeKey, PartitionKey, Split | Merge)
```

The queue deduplicates pending/running identities. Overflow drops admission, records a metric, and never blocks foreground work. A Ready partition crossing its threshold enqueues immediately. Observing an intermediate state does not immediately enqueue: only if the local dedupe map has no identical work and the persisted state-start time is stalled may access cooperate.

Clock is an internal injectable seam. Production wall clock only writes Unix epoch nanosecond state timestamps; Tokio monotonic time governs deadlines and backoff. Invalid/unrepresentable wall time prevents a state transition. Checked `now - started >= timeout` determines stalled; future timestamps are not stalled. Tests use fake time.

A fixup retries at most eight times by default, then retires. Queue loss and
retry exhaustion do not affect correctness because persistent state remains
searchable. There is no durable scan, cluster-wide convergence guarantee, or
owner/lease. Convergence is conditional on the intermediate partition being
rediscovered, maintenance being admitted repeatedly, a legal target existing,
and backend operations eventually succeeding.

Normal search may enqueue merge for an actually visited small Ready non-root partition. Import uses backlog-aware waves to avoid the split-versus-ingest conflict storm caused by every access attempting the same intermediate fixup.

### 18.1 ImportSession

`Index::import_session(ImportOptions)` creates one non-cloneable session bound to one Index without persistent state. `submit(&mut self, Vec<Mutation>)` is exactly one normal atomic batch and returns ordered outcomes. Internally, the session bounds in-flight batches and pauses waves at the Index's known maintenance high watermark.

`finish()` waits for accepted mutations and this Runtime's known backlog for the Index, returning the first unhandled failure. It does not scan the index or claim another process's queue is empty. Drop cancels work not yet started and returns immediately; already-committing batches finish with no observer. Callers requiring known completion must call finish.

## 19. Metrics, tracing, and privacy

KTANN emits through `metrics` and `tracing` facades and defines no custom sink interface. Metrics labels are bounded enums only: backend, operation, outcome, partition level/state, fixup kind, and budget dimension. IndexName, IDs, TreeKey, PartitionKey, RecordId, field values, vectors, and payloads are forbidden metric labels.

Tracing may contain LogicalIndexId, PartitionKey, and a stable TreeKey hash. It does not record raw IndexName, TreeKey, RecordId, field values, vector, or payload. Error Display follows the same privacy rule.

Required observations include:

- mutation/search latency and outcome;
- retry, conflict, commit-unknown, and transaction-too-large counts;
- search logical budget usage/exhaustion and cache hit/miss reason;
- cache bytes/entries/eviction by level;
- maintenance admission, drop, backlog, retries, state duration, and completion;
- Synopsis Bloom saturation/pruning classification;
- RocksDB semaphore wait and blocking-call duration;
- import wave/backpressure duration.

The observability contract fixes events, units, and bounded label dimensions,
not individual metric names or a public tracing-span hierarchy. Implementations
use one consistent `ktann.*` namespace. Tests audit metric label keys and
representative cardinality.

## 20. Verification and corruption policy

Hot paths trust atomically established projections. Search does not read Record Location per candidate or reevaluate predicates on Vector Record. It reports naturally observed duplicate records, child references, illegal formats/states, or source/location/entry mismatch as Corruption.

`verify()` is a read-only bounded audit over exactly one consistent snapshot. It checks:

- key/value decoding and format combinations;
- root and ChildEntry reachability;
- exactly one incoming reference per ordinary non-root partition;
- exact child and leaf membership uniqueness;
- Header count and body kind/level invariants;
- Record–Location–Leaf correspondence;
- Synopsis conservatism.

It may run with foreground work because the snapshot is fixed. It never combines multiple snapshots. Snapshot expiry, deadline, or cancellation returns an error rather than an advisory report. Large FoundationDB indexes may require an offline copy/backend snapshot to finish.

VerifyOptions contains OperationOptions plus three positive bounds:

- `max_issues`, default 100 and hard cap 10,000;
- `max_objects`, default 1,000,000 and hard cap 100,000,000;
- `max_memory_bytes`, default 64 MiB and hard cap 1 GiB.

Reaching any bound stops and returns `complete=false`. Object count includes
every decoded manifest, record projection, partition metadata value, and body
entry examined. All resident verifier state, including hash tables, sort runs,
and buffered keys, is charged to the memory bound; ordered merge scans are used
where possible. The verifier never spills persistent temporary state into the
index. `VerifyReport` contains `complete`, collected issues, and counts of each
logical object examined. Each issue has a non-exhaustive coarse kind:
`InvalidEncoding`, `InvalidTopology`, `MembershipMismatch`, `MetadataMismatch`,
or `SynopsisViolation`. It may include LogicalIndexId, TreeKey hash,
PartitionKey, and RecordId when applicable, but never data values. Detailed
internal causes remain diagnostic source/text rather than an expanding public
enum. There is no continuation, sampling, repair, or automatic repair-on-read.

The public shape is:

```rust
#[non_exhaustive]
pub enum VerifyIssueKind {
    InvalidEncoding,
    InvalidTopology,
    MembershipMismatch,
    MetadataMismatch,
    SynopsisViolation,
}

#[non_exhaustive]
pub struct VerifyIssue {
    pub kind: VerifyIssueKind,
    pub logical_index_id: u64,
    pub tree_key_hash: Option<u128>,
    pub partition_key: Option<u64>,
    pub record_id: Option<Bytes>,
}

#[non_exhaustive]
pub struct VerifyCounts {
    pub objects: u64,
    pub manifests: u64,
    pub records: u64,
    pub partitions: u64,
    pub entries: u64,
}

#[non_exhaustive]
pub struct VerifyReport {
    pub complete: bool,
    pub issues: Vec<VerifyIssue>,
    pub counts: VerifyCounts,
}
```

`Index::verify(VerifyOptions) -> Result<VerifyReport>` is the only verification
entry point. VerifyOptions uses private fields, `Default`, and consuming setters
for operation control and its three limits.

## 21. Persistent format scope

The core versions Manifest and typed-value codecs; each adapter independently
versions its physical key encoding. These versions are interpreted only while
opening an index through the same backend adapter. V1 supports no mixed-format
operation, in-place migration, or cross-backend data movement. Reusing codec
code does not define an export, import, or byte-compatibility contract.

Immutable configuration changes require a newly created Index and application-level data copy. Drop/recreate under one IndexName gets a new LogicalIndexId. No rollback mechanism mutates a built v1 index into an older format; rollback is to keep using the old index until a replacement has been validated and traffic switched externally.

## 22. Validation plan

### 22.1 Backend contract

The same black-box behavioral suite runs independently against deterministic
memory, FoundationDB, and RocksDB adapters and covers snapshot isolation,
update-protected conflicts, unique insert, half-open paged scans and strict
`next_start` progress, transaction/hard limits, range-clear capability
declaration, error classification, and commit-unknown behavior. It never
compares adapter storage encodings.

Memory backend is test-only and injects deterministic conflict schedules, unknown commits, and crash points. It is not a public persistence option.

### 22.2 Correctness and recovery

- Deterministic scheduled histories establish linearizability of insert/upsert/delete/batch mutation.
- Real backends run randomized high-concurrency histories followed by verify.
- Every persistent state transition has crash points before/after commit.
- Every structural chunk has conflict and commit-unknown injection.
- Restart and demand-driven recovery must preserve searchability, exact membership, and no orphan references; under repeated rediscovery/admission and eventual backend success it converges when a legal target exists.
- Adjacent-level maintenance tests move ReceivingSplit targets between parents while the child split advances.
- Root Draining search tests all three bodies and budget accounting.
- Drop/create/old-handle histories test ID non-reuse and Manifest ID validation.

### 22.3 Property tests

- Random nullable typed records and legal predicates prove the Synopsis possible-truth mask contains every actual SQL truth value.
- NoMatch never has a matching entry; AllMatch has only matching entries.
- Memcomparable TreeKey order matches typed comparator and generated prefix ranges are complete.
- Codec round trips, unknown versions, truncated lengths, f64 zero canonicalization, and RaBitQ bit padding fail closed.
- Balanced K-means is deterministic, produces exact sizes, and respects ID ties.
- RaBitQ7 codec vectors and scalar-oracle differential tests cover bit layout,
  reconstruction metadata, every distance bound, SIMD error, interval coverage,
  and exact-rerank selection.

### 22.4 Recall and performance

Fixed public data and synthetic distributions compare against brute-force top-k for unfiltered search, selective filters, broad TreeKey fan-out, incremental churn, and split/merge intermediate states. Reports include Recall@k and all budget usage; v1 first establishes reproducible baselines and does not claim an unsupported recall SLA.

Benchmarks separately measure point insert/upsert/delete, atomic batch mutation, cached/uncached search, selective pre-filter, import, split/merge, and RocksDB blocking saturation. Results include throughput, p50/p95/p99, RPC/IO, conflicts/retries, CPU, peak memory, and write amplification.

Import contention has a dedicated benchmark with continuous upsert, frequent split, maintenance backlog backpressure, and concurrent search. It must demonstrate that ImportSession avoids a fixup lock-conflict storm and does not indefinitely starve queries.

All random/property/concurrency/benchmark runs record seeds. CI uses a fixed seed set; nightly expands it. Failures print directly replayable seeds.

## 23. Delivery sequence

Implementation begins only after the design frontier is empty and this document is marked implementation-ready.

1. Establish workspace, core types/codecs, deterministic memory backend, and backend contract suite.
2. Implement typed storage and FoundationDB adapter, then record mutation and point get.
3. Implement predicates/Synopsis and tree creation/routing.
4. Implement search, RaBitQ7, exact rerank, and cache.
5. Implement split/merge state machines and maintenance runtime.
6. Implement RocksDB adapter and blocking-resource controls.
7. Implement lifecycle drop, ImportSession, verify, metrics/tracing, and full validation/benchmarks.

Each phase preserves one coherent end state; no alternate legacy path or placeholder production quantizer remains.

## 24. Design review status

The design frontier is empty. The final feasibility and contradiction review
covered the public API, persistent invariants, transaction adapters, numeric
codec, search budgets, cache, split/merge recovery, lifecycle, verification,
and ADR consistency. No implementation-blocking decision remains.

## 25. Evidence and rationale

The validation plan is the executable evidence for this design: backend
contract tests establish transaction semantics; model-based histories and crash
injection establish membership and recovery; property tests establish predicate
and codec safety; and benchmarks measure recall, latency, contention, memory,
and write amplification. No external implementation or private source tree is
part of KTANN's specification or validation contract.

The ADR set records the non-obvious trade-offs behind the design and is
normative when explaining why; this document is normative for the combined
behavior and implementation contract.
