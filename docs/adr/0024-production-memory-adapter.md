# Support an ephemeral production Memory adapter

KTANN supports temporary and rebuildable indexes without a native storage
library or external service. Provide `ktann-memory` as a production adapter
beside `ktann-rocksdb` and `ktann-foundationdb`. This supersedes only ADR 0002's
restriction that in-memory backends exist exclusively in tests. It does not
change the backend-neutral transactional interface or persistent formats.

A new `MemoryBackend` creates an isolated Backend Namespace; cloning shares
the same keyspace. Data survives while the backend is retained, but never
survives process restart. No persistence, eviction, or interchange protocol is
implied. The adapter provides snapshots, atomic commits, read-your-writes,
ordered bounded scans, and optimistic point-read conflicts, including ABA.
Core algorithms, exact membership, and maintenance protocols are unchanged.

Use `imbl`'s structurally shared ordered map instead of copying a complete
`BTreeMap` for a transaction. Snapshots share immutable nodes; transactions
stage changes privately, and commits apply changed keys to the latest root
under a mutex. Conflict metadata is bounded and old write snapshots may need
to retry. Read snapshots retain their data until released. Individual mutation
and transaction limits bound foreground commit work; callers still own dataset
size and snapshot lifetimes. Conflict-history reclamation and releasing large
snapshots can require work proportional to the retained data being freed.

Use this same implementation in core functional and fault-injection tests.
The optional `test-support` feature adds explicit commit outcomes, crash replay,
resource instrumentation, and simulated adapter limits/capabilities at the
actual transaction and commit boundaries. Default builds compile out these
controls and their state. The former deterministic test backend and its shared
wrapper are removed, so tests exercise the production snapshot, staging,
conflict-validation, and publication paths. Durable restart remains only a
test simulation, never a persistence promise. The adapter also runs the shared
backend contract and real-vector recall/lifecycle suites.
