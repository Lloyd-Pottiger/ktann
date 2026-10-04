# ADR 0025: Offline capacity refinement of a settled index

- Status: proposed
- Date: 2026-10-04

## Context

Ordinary import builds a valid searchable tree through foreground mutations and
Structure Maintenance. After import settles, routing centroids and leaf membership
can still offer poor recall at bounded search beams. Construction-time experiments
with a separate balanced builder changed initialization and lifecycle together;
those measurements cannot establish the effect of refining the ordinary tree.
KTANN has no stable release and needs no compatibility machinery for this change.

## Decision

Add `Index::refine(RefineOptions) -> Result<()>` as an offline preparation step
after ordinary import finishes and topology settles, before the index serves.
The caller excludes all other in-flight and new operations on this index, including
maintenance from other runtimes. The operation rejects local queued/running
fixups and non-Ready topology. Idle default workers may remain running; Runtime
reopen is unnecessary. No new Building state, owner nonce or publication protocol
is introduced.

The operation captures the existing tree, preserves all Partition Keys and
parent-child topology, and performs at most five local relocation rounds. Each
leaf examines at most 32 neighboring centroids. Positive-gain proposals respect
partition capacity constraints. Metric-correct means, including spherical cosine
means, replace leaf and then internal centroids. Each centroid update atomically
replaces its incoming parent projection and increments the parent Header's cache epoch.
Zero rounds recomputes means without relocation.

Record relocations use bounded atomic transactions preserving one Record Location
and one searchable leaf entry per record, with fields and opaque payload retained.
Centroid/projection/epoch coherence and exact membership remain valid after every
commit. This explicitly narrows the serving-immutable centroid contract: centroids
may change only in caller-exclusive offline refinement. Ordinary foreground
mutations and Structure Maintenance retain their existing contracts.

`RefineOptions::new(input_bytes)` requires a positive input byte limit and defaults
to two rounds and 32 neighbors. Builder methods select 0..=5 rounds, 1..=32
neighbors and operation deadline/cancellation. The byte bound covers resident
vectors, IDs, centroids and topology representation; numerical workspace and move
lists add memory. Bounded rounds limit repetition, not total latency or RSS.

## Failure and validation

Cancellation, errors and unknown commit outcomes may retain some completed moves
or centroid updates in a valid Active index. Refinement is not atomic as a whole
and offers no resume protocol. A caller requiring a fresh preparation can drop
and rebuild through ordinary import. There is no full prepublication audit;
existing `Index::verify` remains an explicit single-snapshot verification tool.

Benchmarks must import, settle and drain maintenance, refine exclusively, then
measure search. Construction totals include import, convergence and refinement
wall/CPU/backend costs; reports retain round, neighbor and input-bound provenance.
Compare ordinary import, zero-round centroid recomputation and positive-round
refinement on the same data and parameters. Historical bulk-builder measurements
remain research evidence for their original pipeline only. Production performance
or recall claims require measurements of this new pipeline.
