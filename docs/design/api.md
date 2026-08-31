# Public API, Configuration, and Errors

Status: Implementation-ready

This module owns KTANN's caller-visible Rust contract. Storage representation,
search internals, and maintenance states are defined by their own modules.

## 1. Handles and lifecycle

`Runtime<B>` owns one backend, process-local configuration, admission state,
caches, and maintenance workers. `Index<B>` is a cheap cloneable handle to one
Active Logical Index. Static backend polymorphism keeps native transaction
lifetimes explicit.

The primary operations are:

```rust
impl<B: Backend> Runtime<B> {
    pub fn new(backend: B, config: RuntimeConfig) -> Result<Self>;
    pub async fn create_index(&self, name: &str, config: IndexConfig) -> Result<Index<B>>;
    pub async fn open_index(&self, name: &str) -> Result<Index<B>>;
    pub async fn drop_index(&self, name: &str) -> Result<()>;
    pub async fn shutdown(&self) -> Result<()>;
}

impl<B: Backend> Index<B> {
    pub async fn insert(&self, record: Record) -> Result<()>;
    pub async fn upsert(&self, record: Record) -> Result<UpsertResult>;
    pub async fn delete(&self, id: Bytes) -> Result<bool>;
    pub async fn batch_mutate(&self, mutations: Vec<Mutation>)
        -> Result<Vec<MutationOutcome>>;
    pub async fn get(&self, id: Bytes, options: GetOptions) -> Result<Option<StoredRecord>>;
    pub async fn batch_get(&self, ids: Vec<Bytes>, options: GetOptions)
        -> Result<Vec<Option<StoredRecord>>>;
    pub async fn search(&self, request: SearchRequest) -> Result<SearchOutcome>;
    pub fn import_session(&self, options: ImportOptions) -> Result<ImportSession<B>>;
    pub async fn verify(&self, options: VerifyOptions) -> Result<VerifyReport>;
}
```

Every operation has a companion `_with_control` form accepting
`OperationOptions` in addition to its ordinary request/options argument; the
simple form uses default operation control. `verify` is the one exception:
its deadline and cancellation control ride inside `VerifyOptions`, which the
single form takes in full. Runtime construction requires an
active Tokio multi-thread runtime, validates all process configuration, and
immediately starts maintenance workers. Successful Runtime shutdown drains
admitted work, awaits the backend's native-resource shutdown hook, and only then
releases the backend.

Index Names are `1..=255` UTF-8 bytes and are compared by their original bytes;
they are not normalized. The fixed bound permits create admission to prove
physical-key limits on both v1 backends.

Create is idempotent for the same name and configuration after an unknown commit
outcome. Open rejects a Dropping index, unsupported format, backend mismatch, or
configuration mismatch. Drop is idempotent and follows the storage lifecycle.

## 2. Records and mutations

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

Record IDs are opaque `1..=256` bytes. Payloads are opaque `0..=64 KiB` bytes.
All vector components are finite; dimension must equal the Manifest dimension.
Fields are positional and must exactly match the schema. Insert is
insert-if-absent and returns `RecordAlreadyExists` when the ID exists. Upsert is
last-write-wins full replacement. `payload: None` deletes an old payload;
`Some(empty)` stores an existing empty payload. Delete is idempotent and returns
whether a record existed. Duplicate Record IDs in one mutation batch are
invalid.

An empty batch succeeds with an empty result. A nonempty mutation vector is one
atomic transaction and is never split. Outcomes correspond to inputs in order;
an item failure returns one operation error with the input position and no
partial outcomes. Validation is completed before the write transaction begins.

`StoredRecord` returns the ID, vector, typed fields, and a closed payload
projection: `NotLoaded`, `Absent`, or `Present(Bytes)`. Batch get is same-order
and same-length; duplicate IDs return repeated values while storage may load
once. Record Location is never public.

Point reads open one consistent read snapshot and validate the persisted
Manifest first: an Active Manifest with the handle's exact immutable identity
proceeds, a Dropping Manifest returns `IndexDropping`, a missing Manifest
returns `IndexNotFound`, and any other mismatch is `Corruption`. Each read then
loads the requested Record Group — the Vector Record and Record Location pair,
plus the Opaque Payload when requested — from the same snapshot. An absent
Record ID means neither the Record nor the Location exists; a pair with only
one side, or a payload without its record, is `Corruption`. Record IDs are
validated before admission, an empty batch succeeds with an empty result, and
backend key and batch limits surface `LimitExceeded`. Cancellation and deadline
apply to the whole read through the shared operation-control contract.

## 3. Schema and predicates

```rust
pub struct FieldId(pub u16);
pub enum Metric { L2, Cosine, InnerProduct }
pub enum DataType { Bool, I64, F64, String }

#[non_exhaustive]
pub enum SynopsisConfig {
    MinMax,
    MinMaxBloom { expected_distinct: NonZeroU32, false_positive_rate: f64 },
}

#[non_exhaustive]
pub enum Value { Null, Bool(bool), I64(i64), F64(f64), String(String) }
pub enum CompareOp { Eq, NotEq, Lt, LessOrEqual, Gt, GreaterOrEqual }

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

`FieldSchema` has private fields and constructor/builders:
`new(name, data_type)`, `nullable()`, and `with_synopsis(config)`. Its default is
non-null `MinMax`. Every field, including every Tree Key field, maintains its
configured leaf synopsis. Public configuration structs use private fields,
`#[non_exhaustive]`, `Default` where meaningful, and consuming builders; they do
not expose serde or public struct literals.

Field names are nonempty, at most 255 UTF-8 bytes, unique by case-sensitive
bytes, and unnormalized. Strings are at most 1 KiB and compare by raw UTF-8
bytes. F64 values are finite and canonicalize negative zero to positive zero.
`Value::Null` is the sole NULL representation.

Predicate evaluation uses SQL three-valued logic and returns a record only when
the expression is TRUE. Compare-to-NULL and NULL in `IN` are invalid.
`And([])` is TRUE; `Or([])` and `In([])` are FALSE. Limits are 1,024 AST nodes,
depth 64, and 1,024 values per `IN`.

## 4. Immutable and runtime configuration

`IndexConfig` persists dimension, metric, ordered fields, ordered unique
non-null Tree Key FieldIds, and minimum/maximum partition entries. Limits are:

| Item | Contract |
| --- | --- |
| Dimension | `1..=16,384` |
| Fields | at most 16 |
| Bloom-enabled fields | at most 4 |
| Encoded Tree Key | schema worst case at most 8 KiB |
| Partition entries | `1 <= min`, `2 * min <= max <= 65,536`; defaults 16/128 |

RaBitQ7, binary fanout, Lloyd rounds, rotation algorithm, logical codecs, and
hard safety caps are fixed by format version 1, not caller options.

`RuntimeConfig` owns the foreground operation limit, cache bytes, worker count,
queue capacity, retry/backoff, maintenance transaction budgets, and default
search budgets. Adapter config owns backend resources such as RocksDB blocking
concurrency. Search options may only lower or override process defaults within
hard caps; changing them cannot alter index correctness.

The v1 defaults and caps are:

| Setting | Default | Hard cap / validation |
| --- | ---: | ---: |
| Running / waiting foreground operations | 1,024 each | 1..=65,536 each |
| Maintenance workers | `min(available_parallelism, 8)`, min 1 | zero disables background maintenance |
| Pending/running fixups | 1,024 | positive, at least worker count |
| Fixup / foreground attempts | 8 each | at least 1 |
| Partition cache | 256 MiB | zero disables; must fit `usize` |
| Scanned Tree Keys | 4,096 | 65,536 |
| Visited partitions | 1,024 | 16,384 |
| Visited Leaf Entries | 65,536 | 1,048,576 |
| Exact rerank candidates | `min(max(64,k+ceil(k/2)),65,536)` | Runtime ceiling 65,536; effective value at least `k` |
| Leaf beam size | 32 | 16,384 |
| Tree Key scan ranges | 1,024 | wider conservative fallback |
| Import maximum in-flight batches | `min(available_parallelism,4)`, min 1 | positive |
| Import backlog watermark | 2 | within queue capacity |

Stalled timeout defaults to checked
`max(1ms, 1s * max_partition_entries / 128)`. Retry backoff starts at 1 ms,
doubles to 100 ms, and applies full jitter in the current interval.

## 5. Search contract

`SearchRequest` contains a finite vector of exact dimension, `k`, an optional
Predicate, and SearchOptions. `k` is `1..=65,536`; the effective exact-rerank
budget is `max(64,k+ceil(k/2))` under the Runtime ceiling and must remain at
least `k`. SearchOptions may override Tree Key, partition, and Leaf Entry bounds and
the leaf-level base beam width per request; the beam is a traversal-quality
knob, not an accounted budget dimension, and the visited-partition budget still
bounds the work it schedules.

`SearchOutcome` contains ordered Search Hits, Search Budget usage, an exhaustive
set of exhausted dimensions, and `rabitq_overlap_truncated`. A hit contains only
Record ID and exact f64 distance. Payload and stored fields are loaded through
get/batch_get rather than search.

Search success means all returned hits are valid and exactly reranked; it does
not mean the global top-k was explored. There is no general completeness flag,
quality score, or continuation. Repeating an identical request with larger
budgets is the supported way to request more work, without a monotonic-result
guarantee across budgets. With the fixed scalar-f64 v1 kernel, identical
snapshot, codec, request, and budgets produce deterministic selection.

## 6. Operation control and errors

`OperationOptions` is a public value containing an optional monotonic
`std::time::Instant` deadline and optional cloneable `CancellationToken`.
Cancellation/deadline is checked before
admission, between bounded work units, and before starting commit. Once commit
starts it is not cancelled; the real result wins.

Errors are non-exhaustive and preserve a diagnostic source. Stable kinds are:

- `InvalidArgument`, `IndexAlreadyExists`, `IndexNotFound`, `IndexDropping`,
  `RecordAlreadyExists`, and `UnsupportedFormat`;
- `TransactionTooLarge`, `LimitExceeded`, `ContentionExhausted`,
  `CommitOutcomeUnknown`, and `IdExhausted`;
- `DeadlineExceeded`, `Cancelled`, and `RuntimeClosed`;
- `Backend` and `Corruption`.

Malformed input, schema mismatch, and non-finite caller-derived arithmetic are
InvalidArgument. Adapter-declared transaction admission failure is
TransactionTooLarge. A full bounded process-local admission queue is
LimitExceeded. Bounded whole-operation retry exhaustion is ContentionExhausted.
Invalid persistent encoding or an invariant mismatch is Corruption.

Display and Debug redact Index Name, raw Tree Key, Record ID, field values,
vectors, and payload. Cancellation never rewrites a successful commit or a
commit of unknown outcome.

## 7. Import and verification shapes

`ImportSession::submit(&mut self, Vec<Mutation>)` waits for local capacity,
admits exactly one ordinary atomic batch, and returns a unique process-local
`BatchToken`. Each session starts with one active batch and learns concurrency
from saturated clean completions and retryable conflicts up to its configured ceiling;
multiple accepted batches may execute concurrently within that learned bound.
`finish(self)` waits for all accepted work and returns ordered
`ImportBatchResult { token, result }` values in submission order. Dropping the
session cancels work not yet admitted to commit; committing work continues under
the Runtime's in-flight guard.

Verify returns a bounded report with `complete`, coarse issue kinds, safe
identifiers, and logical-object counts. Options bound issues, objects, memory,
deadline, and cancellation. Defaults/hard caps are 100/10,000 issues,
1,000,000/100,000,000 logical objects, and 64 MiB/1 GiB resident memory. There
is no continuation or repair API.

## 8. Contract validation

- Compile tests protect constructors, builder ownership, non-exhaustive enums,
  Send/Sync requirements, and redacted formatting.
- Focused tests cover record/schema/predicate boundaries, batch ordering,
  cancellation-before-commit, commit-result priority, search metadata, import
  token ordering, and lifecycle idempotency.
- Public tests assert behavior, not private cache, task, or codec structure.
