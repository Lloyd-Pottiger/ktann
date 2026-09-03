# Transactional Storage and Persistent Format

Status: Implementation-ready

This module owns the common backend contract, persistent identity and lifecycle,
logical key/value codecs, and typed atomic storage operations.

## 1. Backend contract

`Backend` uses GAT transaction types and stable RPITIT futures. A ReadTxn offers
snapshot `get`, same-order `batch_get`, bounded forward `scan`, and same-order
multi-range `batch_scan` — one independently paginated page per range from one
backend interaction. A WriteTxn
adds `get_for_update`, `batch_get_for_update`, `put`, unique `insert`, `delete`,
bounded `batch_mutate`, optional transactional `clear_range`, and consuming
commit/rollback.

Every adapter provides:

- one consistent read snapshot and atomic multi-key writes;
- read-your-writes behavior;
- update-protected point reads for conflict establishment;
- ordered, half-open forward scans with explicit item/byte bounds;
- hard limits, conservative admission budgets, and declared capabilities;
- an asynchronous native-resource shutdown hook invoked after Runtime drain;
- error classes `RetryableAbort`, `CommitOutcomeUnknown`, `LimitExceeded`,
  `Unsupported`, `Corruption`, and `Other`.

The common interface has no reverse scan, unbounded scan, implicit read renewal,
or cross-transaction logical snapshot. A scan may return one oversized first
item so callers can make progress; otherwise it never exceeds the requested
adapter byte bound. A page is explicitly terminal (the range is exhausted) or
non-terminal; a non-terminal page carries a `next_start` equal to the
byte-lexicographic successor of its last key — the smallest key strictly
greater than it within the backend's key-length limit — so resuming at that
bound returns every remaining key exactly once, with no skipped eligible key
and no duplicated returned key.

## 2. Transaction sizing and retry

Core plans logical work before starting a transaction and checks adapter
budgets while building mutations. The Admission Budget exposes the adapter's
physical key-prefix charge to operations that need an exact worst-case plan,
such as leaf relocation. Adapters re-check exact encoded keys, values, and
affected-data accounting. Exceeding a declared limit returns `LimitExceeded`,
not an unbounded internal retry.

RetryableAbort restarts the complete logical attempt against a fresh snapshot.
No transaction object survives an await outside its adapter-defined safe
section. CommitOutcomeUnknown is surfaced unless a lifecycle operation has an
idempotent persistent-state recovery rule.

## 3. Backend mappings

FoundationDB maps update-protected reads to conflict-establishing reads and
supports transactional logical range clear. Its adapter exposes actual database
limits and keeps write transactions short; snapshot expiry is a Backend error,
not a hidden four-second deadline.

RocksDB uses `OptimisticTransactionDB`. ReadTxn owns a Snapshot. WriteTxn enables
a transaction snapshot and binds every read option to it while retaining
read-your-writes. Point `get_for_update` establishes conflicts; state machines
never depend on range conflicts. WAL remains enabled and commits use
`sync=true`. One permit-bounded native thread actor owns each live snapshot or
write transaction from admission through native cleanup. Its capacity-one
channel serializes every synchronous call, commit, rollback, and destruction;
async tasks never run native cleanup or wait synchronously for it. Existing
transactions reuse their actor rather than reacquiring admission, so saturating
the configured live-actor limit delays only new transaction creation. A
backend shutdown hook waits for detached cleanup before Runtime releases the
adapter; direct users get the same barrier from consuming asynchronous adapter
shutdown. RocksDB v1 does not advertise transactional range clear.

## 4. Persistent identity and lifecycle

A Backend Namespace contains name mappings and never-reused Logical Index IDs.
Create reserves an ID and atomically inserts the name mapping and Active
Manifest. ID gaps are valid. Drop transitions the Manifest to Dropping before
deleting data; all ordinary operations update-protect and validate Active state.

FoundationDB may atomically clear the complete data range and remove the
Dropping Manifest. Without transactional range clear, core deletes bounded
pages of logical keys while preserving the Dropping Manifest, then atomically
removes the empty Manifest and name mapping. Unknown outcomes recover by reading
the lifecycle records.

## 5. Logical keyspace

The core defines a versioned logical namespace for:

- allocator and Index Name mapping;
- Index Manifest;
- Vector Record, Opaque Payload, and Record Location;
- Tree Manifest directory entries;
- Partition Header, immutable Centroid, Synopsis, and transition State;
- Leaf Entry and Child Entry.

Every data key begins with Logical Index ID, so drop owns one contiguous logical
range. Tree-local keys embed the canonical encoded Tree Key and Partition Key;
there is no Tree ID. Physical adapters add their own bounded prefix without a
second unbounded escaping pass.

Logical key codecs are versioned independently from value codecs. Version 1
specifies exact type tags, integer endianness, tuple escaping, terminators, and
field ordering in codec source plus checked-in golden vectors. The Tree Key
codec is memcomparable: byte ordering exactly matches typed comparison and
supports field-prefix half-open ranges. Decoders reject unknown tags, duplicate
fields, noncanonical values, nonzero padding, and trailing bytes.

## 6. Persistent values

The Index Manifest stores lifecycle state, format and codec versions, immutable
configuration, Logical Index ID, RaBitQ rotation seed, and exact Bloom
parameters. A Tree Manifest is the directory entry, root reference, and
Partition Key allocator high-water mark for one Tree Key. Reservation allocates
fixed ranges (default 1,024) through an update-protected manifest; unused keys
remain gaps.

Partition Header stores level (1 for a leaf), exact entry count, cache epoch,
and the small Partition State discriminator needed for traversal; level alone
determines whether the partition contains Leaf or Child Entries. Transition
payloads store the
source/target references and state-start time required to resume a transition;
structural drain and paged deletion restart from the current prefix beginning
and persist no cursor.
Leaf Entries contain Record ID, typed filter fields, and absolute RaBitQ7 bytes;
Child Entries contain child Partition Key and immutable centroid projection.

All persistent algorithms that affect bytes are format protocol:

- the seeded Givens rotation uses the persisted 32 bytes as the ChaCha8 256-bit
  key, with little-endian words, block counter zero, and stream identifier zero.
  It generates three Fisher-Yates permutations of dimension indexes. For a
  Fisher-Yates bound `n`, it consumes little-endian u32 output, sets
  `zone = floor(2^32 / n) * n`, rejects values at least `zone`, and uses
  `value % n`; thus bounded draws have no modulo bias. Adjacent indexes are
  paired in generated order. Each pair applies
  `(x, y) -> ((x + y) * c, (x - y) * c)` with
  `c = f32::from_bits(0x3f3504f3)`; an odd final index is unchanged for that
  round. Each arithmetic step rounds as IEEE-754 f32 without fused contraction;
- Bloom uses XXH3-128 with the v1 domain seed `0x4b54414e4e01b100`
  over the canonical non-NULL typed-value bytes. The low 64 digest bits are
  `h1` and the high 64 bits are `h2`; they drive wrapping double hashing
  `h1 + i*h2`, followed by unsigned modulo the persisted bit count and
  LSB-first bit numbering. V1 uses one probe and, for expected distinct count
  `n` and target false-positive rate `p`, derives `m = ceil(n / p)` bits. At
  most `n` bits can be occupied at the configured cardinality, so the uniform
  hash false-positive bound is directly `n / m <= p`, without relying on
  independent double-hash probes. Creation rejects `m > 2^32 - 1` or the
  existing 64-KiB complete-Synopsis limit;
- RaBitQ7 uses the exact layout defined by the search design.

These constants and steps must be written next to codec golden vectors before
the first format is emitted. “Implementation-defined” randomness or hashing is
not permitted because processes and restarts must produce equivalent summaries
and codes.

## 7. Typed atomic operations

Only this module may compose raw logical keys. It exposes typed operations for:

- validating/opening manifests and tree directory pages;
- reserving persistent IDs;
- reading records, locations, headers, states, synopses, and entries;
- atomically changing record membership and exact metadata;
- installing and advancing split/merge states;
- atomically relocating Leaf Entries and Child Entries under a typed structural
  movement protocol;
- bounded deletion of one partition prefix or full index range.

Algorithm modules do not hand-build keys or partially update counts/synopses.
Natural decode or cross-value invariant failures return Corruption.

Relocation update-protects the source, every actual target, and any split family
whose state authorizes the move. The allowed combinations are exact:

- `DrainingSplit` source to its own `ReceivingSplit` targets;
- `DrainingSplit` source to a same-level `Ready` corrective target;
- same-level `Ready` source to the named split family's `ReceivingSplit`
  targets;
- `Merging` source to same-level `Ready` targets.

The transaction validates Header/State agreement, levels, split-family
identity, and entry identity before mutation. Leaf relocation uniquely inserts
the target entry, deletes the source entry, repoints Record Location, and
updates source/target counts and epochs plus the target Synopsis. Child Entry
relocation uniquely inserts under the new parent, deletes under the old parent,
and updates both parent counts and epochs, preserving one incoming edge in
every committed state. A corrective internal move also rejects a batch that
would remove the Ready source's final Child Entry. Corrective target capacity
and encoded Backend mutation budgets are rechecked at this topology boundary;
a concurrent transition conflicts and the caller replans from a fresh
snapshot.

## 8. Partition deletion

Partition removal has one common correctness order:

1. install or retain the terminal transition state that keeps obsolete data
   unreachable or safely covered;
2. if transactional range clear is available, clear the full partition prefix
   in the final atomic topology transaction;
3. otherwise delete bounded point-key pages while the terminal state remains;
4. after an empty page proves the prefix empty, atomically perform the final
   topology switch and remove Header, State, and remaining metadata.

Split and merge may not describe an atomic “delete full prefix” on an adapter
that lacks transactional range clear.

For a split or merge source, the exact zero Header count is itself the
emptiness proof: the drained prefix holds only its fixed metadata keys, so
steps 3 and 4 collapse into the final transaction — bounded point deletes of
those keys commit atomically with the topology switch, and the terminal state
never outlives it. The paged form remains for removals without an exact-count
proof, such as index drop.

## 9. Verification

Backend contract tests run unchanged against a deterministic test backend,
FoundationDB, and RocksDB. They cover snapshot consistency, read-your-writes,
conflicts, unique insertion, gap-free scan pagination across item and byte
boundaries, empty ranges, oversized values, exact-boundary exhaustion, batched
multi-range scans with independent per-range pagination, limits,
rollback, commit outcome, range-clear capability, and durability mappings.

Codec tests use golden bytes, ordering properties, malformed/noncanonical input,
and cross-process deterministic vectors for rotation, Bloom, Tree Key, values,
and RaBitQ7. Model tests assert that typed atomic operations preserve exact
membership under conflicts and injected unknown outcomes.
