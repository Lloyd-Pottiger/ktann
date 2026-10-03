# Unpublished capacity-refined construction

Status: proposed alternative implementation; not selected for integration.

The existing Import Session intentionally performs ordinary concurrent mutation
batches into a serving index. It cannot own published partitions exclusively,
and routing centroids and their cached Child Entry projections are immutable.
Rewriting those centroids after `finish` would violate snapshot and concurrent
mutation contracts. The offline quality experiment therefore cannot become an
online optimize operation.

Add a distinct `Runtime::build_index` operation. It reserves the requested name
and a fresh Logical Index ID in a durable `Building { owner }` lifecycle. The
owner is a 128-bit OS-random nonce, created before the reservation attempt and
persisted with the manifest. Concurrent allocator transactions can tentatively
select the same ID; the nonce distinguishes an ambiguous reservation from a
competing builder with identical name and configuration.

A Building index has no public Index handle. Create/open return `IndexBuilding`.
Each bounded staging transaction update-protects and compares the complete
Building manifest, including owner. Deterministic key/value writes are replayed
on unknown staging outcomes. Each record group (record, location, leaf entry and
optional payload) is indivisible within one staging transaction; a record group
that cannot fit the adapter fails construction unpublished. The final transaction compares the same identity
and changes Building to Active, after auditing the staged topology and membership
through the ordinary verification implementation in one snapshot. The guarded
foreground commit boundary applies only to that publication. Unknown publication
is recovered only by reading the exact Active identity.

Cancellation, interruption or failure can leave a durable Building index. The
caller explicitly uses `drop_index(name)` to clean it up before retrying. Drop
changes Building to Dropping and uses the existing bounded cleanup protocol.
Every staging and publication transaction conflicts with that transition, so an
old builder cannot resurrect a dropped index or write into a new same-name
index. There is no timeout ownership, automatic error cleanup, automatic
resumption, or process-local exclusivity assumption. An ambiguous operation does
not automatically remove a competing builder's data.

Construction first uses the existing deterministic balanced binary training to
produce a power-of-two number of leaves. Initial leaf planning targets
`max(maximum / 2, 2 * minimum)` entries, while refinement and publication retain
the configured maximum. The minimum clamp preserves legal non-root occupancy
under balanced subdivision, including dense-capacity configurations. Internal
fanout continues to use the configured maximum. At one million records with
minimum 16 and maximum 512 this initializes 4096 leaves rather than 2048. This choice
provides more granularity and space for moves; it is an experiment in combined
initialization, granularity and slack, not proof that capacity rejection caused
previous performance differences. The full-data 4096 comparison must include
same-builder zero-round and refined controls. Zero refinement
rounds is the construction control. Each bounded local round selects at most 32
nearest leaf centroids, proposes each record's best positive-gain move, applies
proposals in descending gain order under exact minimum/maximum counts, then
recomputes metric-correct means. Cosine means are spherical. Internal routing is
built bottom-up from the final leaves with balanced capacity-limited fanout.
All partitions are Ready at publication; membership is exactly one location and
one searchable leaf entry per record. Existing foreground behavior is unchanged.

Planning runs on a cooperatively cancelled blocking task under the existing
foreground admission bound. Dropping its async owner cancels numerical work;
checkpoints also observe the caller token and deadline.

The complete source set is resident. An explicit byte limit covers caller data,
not workspace: preprocessed vectors, binary-training copies, memberships,
centroids, proposals, topology rows and verification state add memory. Exact
neighbor selection costs O(leaves² × dimension) per round; record proposals cost
O(records × neighbor count × dimension). Bounded rounds limit repetition, not
construction latency. The complete audit has the existing one-snapshot lifetime
constraint and a 1 GiB resident audit limit. A backend that cannot complete it
returns an error and leaves Building unpublished. Large FoundationDB construction
therefore remains unproved.

This extends ADR 0017 with a non-serving construction lifecycle and ADR 0015
with a separate initial-construction path; ordinary incremental topology and
immutable published centroids remain their existing contracts. The option is
not a production performance recommendation until same-builder zero-round,
refined, and ordinary-import end-to-end measurements establish the tradeoffs.
