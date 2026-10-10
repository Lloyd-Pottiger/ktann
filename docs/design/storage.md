# Transactional Storage and Persistent Format

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

Memory uses a structurally shared ordered map with immutable read snapshots
and optimistic write transactions. Each new backend is one isolated Backend
Namespace; clones share its keyspace. Commits validate protected point reads,
including absent-key ABA changes, and atomically publish only the staged writes
on the latest committed root. A short-lived mutex serializes snapshot
registration and commits. There is no IO, persistence, eviction, or total-memory
quota. Live snapshots retain old data; applications control their lifetime.
Conflict history is reclaimed when writers finish and capped at 100,000 key
references and 8 MiB of key bytes. Writers with protected reads older than that
window abort with `RetryableAbort`. The adapter enforces 10,000-byte keys,
100,000-byte values, and budgets of 10,000 mutations and 1 MiB per transaction.
It does not advertise transactional range clear, so index drop uses bounded
point deletes. See [ADR 0024](../adr/0024-production-memory-adapter.md).

FoundationDB maps update-protected reads to conflict-establishing reads and
supports transactional logical range clear. Its adapter exposes actual database
limits and keeps write transactions short; snapshot expiry is a Backend error,
not a hidden four-second deadline.
ReadTxn disables native read-your-writes bookkeeping because it cannot mutate;
its pinned snapshot is unchanged. WriteTxn retains native read-your-writes.

Snapshot `batch_get` may combine two equal-length keys with consecutive final
bytes into one native range request, notably adjacent Record/Location values.
It fetches at most two rows, returns only the requested keys, and point-reads
missing targets if intervening keys or a short page prevented completion.
Order, duplicates, absent values, and the transaction snapshot are preserved.
The combined read remains bounded even when key extensions intervene. Maximum
length keys and update-protected reads use ordinary point reads, preserving
native key limits and exact write-conflict scope. Range values retain their
native owner rather than copying value buffers.

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

Offline refinement preserves the Active Manifest, Partition Keys and topology.
Record relocation preserves exact membership, payloads and capacity constraints
in bounded transactions. Centroid replacement atomically updates the centroid,
incoming parent projection and parent Header cache epoch. The operation has no
additional persistent lifecycle state.

FoundationDB may atomically clear the complete data range and remove the
Dropping Manifest. Without transactional range clear, core deletes bounded
pages of logical keys while preserving the Dropping Manifest, then atomically
removes the empty Manifest and name mapping. Unknown outcomes recover by reading
the lifecycle records.

## 5. Logical keyspace

The core defines one logical namespace for:

- allocator and Index Name mapping;
- Index Manifest;
- Vector Record, Opaque Payload, and Record Location;
- Tree Manifest directory entries;
- Partition Header, serving-immutable Centroid, Synopsis, and transition State;
- Leaf Entry and Child Entry.

Every data key begins with Logical Index ID, so drop owns one contiguous logical
range. Tree-local keys embed the canonical encoded Tree Key and Partition Key;
there is no Tree ID. Physical adapters add their own bounded prefix without a
second unbounded escaping pass.

Logical keys begin with their namespace or index scope tag. The canonical layout
specifies exact type tags, integer endianness, tuple escaping, terminators, and
field ordering in codec source plus checked-in golden vectors. The Tree Key
codec is memcomparable: byte ordering exactly matches typed comparison and
supports field-prefix half-open ranges. Decoders reject unknown tags, duplicate
fields, noncanonical values, nonzero padding, and trailing bytes.

## 6. Persistent values

The Index Manifest stores the persistent format version (`FORMAT_VERSION = 2`),
lifecycle state, immutable configuration, Logical Index ID, RaBitQ rotation
seed, and exact Bloom parameters. The Persistent Format covers Logical Keys,
stored values, adapter physical keys, and algorithms that determine persisted
bytes. Loading a Manifest validates its format version; an unsupported version
returns `UnsupportedFormat`.

Each stored value consists of a one-byte type tag followed by its payload.
Wrong key/value pairings and malformed or noncanonical encodings return
`Corruption`, including in allocator and Index Name values.

A Tree Manifest is the directory entry, root reference, and
Partition Key allocator high-water mark for one Tree Key. Reservation allocates
fixed ranges (default 1,024) through an update-protected manifest; unused keys
remain gaps.

Partition Header stores level (1 for a leaf), exact entry count, cache epoch,
and the small Partition State discriminator needed for traversal; level alone
determines whether the partition contains Leaf or Child Entries. Transition
payloads store the source/target references required to resume a transition and
`started_at_unix_millis`, an unsigned 64-bit Unix-epoch millisecond timestamp
used for recovery age checks and diagnostic metrics. Zero denotes an unavailable timestamp. Structural drain
and paged deletion restart from the
current prefix beginning.
Leaf Entries contain Record ID, typed filter fields, and absolute RaBitQ7 bytes;
Child Entries contain child Partition Key and a centroid projection matching the
child centroid. Serving centroids are immutable; offline refinement replaces the
centroid and incoming projection atomically.

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

Codec golden vectors fix these constants and steps. Processes and restarts
must produce equivalent summaries and codes; implementation-defined randomness
or hashing is not permitted.

## 7. Typed atomic operations

Only this module may compose raw logical keys. It exposes typed operations for:

- validating/opening manifests and tree directory pages;
- reserving persistent IDs;
- reading records, locations, headers, states, synopses, and entries;
- atomically changing record membership and exact metadata;
- installing and advancing split/merge states;
- bounded deletion of one partition prefix or full index range.

Algorithm modules do not hand-build keys or partially update counts/synopses.
Natural decode or cross-value invariant failures return Corruption.

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

Backend contract tests run unchanged against Memory, FoundationDB, and RocksDB.
Core tests use Memory's optional `test-support` controls for faults, replay,
resource ceilings, and simulated restart; there is no separate in-memory
transaction implementation. Default adapter builds omit this instrumentation.
The tests cover snapshot consistency, read-your-writes,
conflicts, unique insertion, gap-free scan pagination across item and byte
boundaries, empty ranges, oversized values, exact-boundary exhaustion, batched
multi-range scans with independent per-range pagination, limits,
rollback, commit outcome, range-clear capability, and durability mappings.

Codec tests use golden bytes, ordering properties, malformed/noncanonical input,
and cross-process deterministic vectors for rotation, Bloom, Tree Key, values,
and RaBitQ7. Model tests assert that typed atomic operations preserve exact
membership under conflicts and injected unknown outcomes.


## 10. Bulk Build records

Bulk Build reservations add lifecycle byte `2` (Building; Active remains `0`,
Dropping `1`) and index-owned key kind `0x05` for the Build Descriptor. Descriptor
value tag `0x0d` contains construction version u32, a length-prefixed absolute
UTF-8 source path (maximum 4096 bytes), a length-prefixed 89-byte input artifact
manifest, min/max entries u32, and sample/memory/scratch ceilings u64, all big
endian. Bootstrap transactions may access the descriptor alongside the Manifest
for atomic reservation. Serving transactions still validate an Active Manifest.
See [ADR 0022](../adr/0022-resumable-bulk-build.md).

The index-owned Build Progress key (`0x06`, value tag `0x0e`) stores one sized
89-byte Serving Artifact manifest, a positive u64 load epoch, and a phase byte:

- `0` Loading: u64 committed entries and a 32-byte artifact-prefix SHA-256.
- `1` Loaded: no additional fields; the complete artifact was read through EOF.
- `2` Validating: sized backend cursor (at most 16 KiB), u64 compared entries,
  and a 32-byte artifact-prefix SHA-256.
- `3` Validated: no additional fields; the backend scan and artifact EOF checks
  both completed. Publication still requires an Active Manifest transaction.

Unfinished counts cannot exceed the artifact count; zero entries require the
artifact-header digest. Counts/digests alone never imply completion. Terminal
phases derive their total count/digest from the artifact. These control bytes
are validated but excluded from serving membership ledgers. Debug output redacts
locators/cursors. Unknown phases, malformed/trailing bytes and unused key/value
kinds fail closed. No compatibility layer is retained.
See the implemented protocol in [Bulk Build](bulk-build.md).


Namespace key `[0, 2] || LogicalIndexId:u64be` stores Build Workspace (tag `0x0f`).
It survives index-prefix removal. Its canonical value contains a sized absolute
UTF-8 root (at most 3900 bytes), nine u64 resource/admission/hard-limit values,
a nonzero 32-byte ownership token, positive u64 epoch, two optional accepted
artifact descriptors (flag, epoch, sized 89-byte manifest), and a bounded failure
code. Accepted Forest/Serving kinds and epoch bounds are validated; Serving
requires Forest. All integer encodings are big endian.

See [ADR 0022](../adr/0022-resumable-bulk-build.md).


Namespace key `[0,3] || LogicalIndexId:u64be` stores Build Schedule, tag `0x11`:
sized UTF-8 Index Name (1..=255 bytes), the shared Worker Options encoding (sized
root and seven u64 resource/admission values), a 32-byte owner token, and u64 UTC
expiry milliseconds. Zero token means unowned; an unowned nonzero expiry is a
retry delay. Nonzero owner requires nonzero expiry. The token and options are
redacted in Debug. Index-bound transactions may access only their own namespace
Build Schedule to protect mutations against automatic takeover. Queue removal
is separate from index-prefix deletion. All values are canonical and reject
truncation, trailing bytes, and invalid ownership encodings.

## 11. Bulk Build files

Input, Forest and Serving Artifacts use an independently versioned immutable
file format. An 89-byte manifest contains eight magic/version bytes, a one-byte
kind, a 32-byte binding, two big-endian u64s (item count and data-file bytes),
and a 32-byte SHA-256. The data header repeats magic/kind/binding. Each bounded
frame contains a big-endian u32 body length, body bytes and a body SHA-256.
The manifest hashes the whole file; reordering valid frames fails verification.
Paths are excluded from identity, so identical bytes may be relocated.

- Input kind `0` preserves original-order records using canonical record,
  vector and field primitives, payload presence and bounded payload bytes.
  Its binding covers source schema shape. Metric, Tree Key field selection,
  partition thresholds and Synopsis policy belong to the consuming index.
- Forest kind `2` frames contain a u32 Tree Key length, canonical Tree Key and
  partition plan. Plans encode key, level, count, full-f32 centroid and
  length-prefixed entry identities. The binding covers the source identity,
  ordered Tree Key field selection, metric, persisted seed, construction
  version and resource options. Trees are canonical; partitions are ordered
  child-before-parent with each root last.
- Serving kind `3` frames contain a u32 key length, key bytes and value bytes.
  Keys are strictly ordered and unique. The binding covers source/forest
  identities, immutable Index Manifest, target hard limits and sort options.
  Manifest, name mapping, allocator and build bookkeeping are excluded.

Readers validate identity, canonical bodies, framing, count and whole-file EOF.
Forest readers additionally check local allocation, occupancy and tree closure;
Serving readers check index ownership, codecs, order and adapter limits. These
checks prove file validity, not worker authority or complete backend contents.
The [build protocol](bulk-build.md#artifact-sealing-and-workspace-ownership)
owns sealing, acceptance and recovery; exact backend validation precedes
publication.
