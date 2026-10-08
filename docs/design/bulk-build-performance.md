# Bulk Build Throughput Redesign

Status: **Draft for the wider redesign.** Byte-preserving IO improvements are
implemented; algorithm replacement and the end-to-end throughput target remain open.
This document records qualified IO changes and proposes further work; the
implemented lifecycle remains specified in
[Resumable Bulk Build](bulk-build.md) and ADRs 0027/0028.

## Problem and evidence

Bulk Build must shorten the time from receiving input to a searchable index,
while preserving index correctness and competitive search quality. Resumability
alone does not satisfy this goal. The present implementation is bounded and
recoverable, but repeatedly materializes full vectors during construction.

Local measurements use RocksDB, Cohere 1M, 768-dimensional cosine vectors,
top-100 queries, all 1,000 queries, and fresh databases. Input batches contain
50 records with one loader; query concurrency is 1/5/10/20 for 30 seconds each.
These are sequential local observations, not confidence intervals.

| Implementation | Receipt / online insertion | Optimize | Total | Recall@100, beam 128 |
| --- | ---: | ---: | ---: | ---: |
| Online, c083e16, two runs | 749–842 s | <0.04 s | 749–842 s | 92.14–92.17% |
| Original bulk, c083e16, two runs | 428–433 s | 627–629 s | 1,054–1,061 s | 89.07% |
| Bulk, 81c7f57, one run | 439 s | 460 s | 899 s | 89.07% |

The latest bulk result is still slower than both observed online runs. An updated
same-revision online comparison is required before claiming a precise speedup.
The original diagnostic run attributed approximately 26 s to snapshot creation,
381 s to forest construction, 147 s to serving artifact creation, 37 s to backend
loading, and 37 s to validation/publication/cleanup. These phase proportions are
historical; they are not a profile of 81c7f57.

Original logical scratch writes were approximately 264 GB for a 3.135 GB source.
Removing redundant ID sorts reduced measured tree writes from 168.66 GB to
119.54 GB. Increasing IO buffers reduced time without reducing logical writes.
At the old 95%-recall threshold, online beam 192 achieved about 204 QPS; bulk
needed beam 256 and achieved 117–129 QPS. This sampled parameter grid does not
establish an exact equal-recall frontier, but rules out treating equal beam as
equal quality.

Reproduction evidence is stored locally under
`/Users/lloyd/projects/ktann/.benchmark-data/results/`:
`vdbbench-ab-20261007`, `bulk-sort-20261008`, and
`bulk-sort-buffer-20261008`. Each contains reports and raw results. The accepted
buffer change has an unresolved full-run peak-RSS increase of 0.30–0.45 GB;
that remains a limitation of the current evidence.

## Mechanism diagnosis

* `ForestArtifact::build` in `src/bulk/forest.rs` sorts full-vector payloads by
  Record ID for global duplicate detection, then by TreeKey for construction.
  Duplicate rejection is required; carrying vectors through both sorts is not.
* `Workspace::group` in `src/construction.rs` samples a group, trains a binary
  split, externally sorts full vectors by the distance difference, and writes
  full-vector children. Every binary level repeats dimension-dependent IO.
  Terminal groups also need canonical ID order. The original profile spent far
  more time on sorting/IO than sample training.
* `ServingArtifact::build` in `src/bulk/serving.rs` joins source records with
  forest memberships and sorts encoded serving records. Publication requires
  complete, validated data, but not every current intermediate representation.
* `Staging::finish` in `benchmarks/src/bridge/bulk.rs` converts a staging file to
  an Input Snapshot only after receipt finishes. Construction then starts.
* The VectorDBBench client creates Python float lists, validates components,
  and JSON-encodes each insert; the bridge decodes them. Native staging took
  only about 5 s in the earlier diagnostic run. The remaining receipt time
  requires separate measurement; JSON alone has not been isolated as its cause.
* The scheduler assigns whole jobs. Single-job forest construction is sequential;
  adding scheduler hosts does not divide one large tree among them.

Receipt alone currently takes 413–439 s. Even free construction limits speedup
against the observed 749–842 s online baseline to roughly 1.7–2.0x. A 3x
end-to-end target therefore requires improving receipt too. Applying the same
transport improvement to online changes the comparison baseline as well.

## Proposed boundary and data flow

Keep the existing reservation, fenced ownership, hidden index, validation and
atomic publication contracts. Change the physical construction pipeline behind
those boundaries. Do not add another scheduler state machine to accelerate it.

1. **Receive and seal once.** Give the benchmark bridge a bounded binary vector
   batch representation, shared by online and bulk modes. Preserve finite-value,
   dimension, ID, frame-size, partial-insert and error semantics. Measure client
   conversion, encoding, IPC, native decoding and native work separately. Evolve
   snapshot writing so bulk ingestion can write an unsealed snapshot directly,
   then seal it only after count and integrity validation. Incomplete input is
   never a valid build source. Avoid the extra staging-to-snapshot rewrite.
2. **Separate vectors from assignments.** Within the builder, use immutable vector
   storage and compact ordinals/references. Detect duplicate IDs globally using
   compact metadata. Partition and reorder references rather than full vector
   payloads. Arbitrary TreeKeys, IDs and payloads remain supported; fixed-width
   Cohere vectors are a prototype workload, not a new public restriction.
3. **Prototype a better partitioner.** Compare balanced binary reference
   partitioning with deterministic multiway assignment/refinement. The former
   isolates IO savings; the latter tests reduced depth and improved quality.
   Capacity bounds, tiny groups, duplicate vectors, skewed TreeKeys and stable
   tie-breaking must be explicit. Do not select a branching factor or training
   algorithm solely from build time. Final centroids must describe all members.
4. **Produce serving data from final assignments.** Once leaf membership is
   stable, scan vectors and encode serving records with their final keys. Keep
   required key ordering and synopsis validation; remove intermediate joins or
   materializations only where the final assignment representation makes them
   redundant. Quantization must use the same finalized centroid as search.
5. **Add bounded local parallelism only after serial efficiency.** Independent
   groups can share immutable vectors. One job-wide admission budget must cover
   vectors, metadata, scratch buffers and all workers together. Avoid multiplying
   a per-worker memory limit. Cross-host subdivision is outside this proposal.

Immutable vector storage is not free IO: random reference traversal of an mmap
can fault repeatedly and exceed the intended resident-memory budget. The
out-of-core path must batch/block accesses and measure physical IO and RSS;
`mmap` is not itself evidence of bounded memory or speed.

## Constraints and open decisions

The primary path must handle input far larger than RAM, including terabyte-scale
input, within an explicitly configured working-memory budget. Memory must not
scale with the entire vector matrix or all record assignments. The existing
256 MiB construction setting remains the starting qualification target; a larger
default is not assumed. Small inputs may benefit from caching but cannot require
a separate resident-only algorithm for correctness or acceptable operation.

The next prototype therefore uses bounded blocks and streaming assignment files.
Read vectors sequentially in source/block order and merge their assignments in
that order. Bound the number of active centroids and per-group samples; spill
assignment metadata and process groups in batches when those bounds are reached.
Measure additional full scans caused by group batching. Avoid one full source
scan per partition. A controlled blockwise vector redistribution can be preferable
to repeated random gathers or repeated global scans: the requirement is low,
bounded IO amplification, not literally forbidding every vector copy.

Algorithm selection must compare scan/redistribution bytes, seek behavior,
training CPU, assignment metadata and scratch peaks as input and group counts
grow. Multiway partitioning may reduce full-vector passes, but centroid-distance
work and quality can offset that benefit. It remains a hypothesis to prototype.

A 1 TB workload is a scalability contract, not a completed local benchmark.
Qualify the external-memory path with input substantially exceeding its allowed
memory, then model passes and scratch capacity at 1 TB and validate on suitable
hardware before claiming that scale. Include all resident samples, centroids,
metadata and concurrent workers in admission; account separately for backend
cache and observed OS residency.

Preserve global ID uniqueness, exactly-once membership, partition capacity,
canonical persistent encoding, deterministic construction for fixed options,
bounded scratch use, fencing and atomic visibility. New tree bytes need not match
the old algorithm. Algorithm identity in immutable descriptors must change when
semantics change; an old in-progress descriptor must be explicitly rejected or
handled by its matching implementation, never silently resumed with new semantics.
There is no stable-release compatibility requirement.

Partial preparation remains disposable under the existing coarse retry model.
Only sealed artifacts may be reused; cancellation and stale-owner publication
must remain safe. Caching vectors does not justify publishing partial state or adding checkpoints
to every recursive split.

The precise out-of-core access strategy, partition algorithm, and achievable
speed/quality frontier remain unresolved. This is consequently
not an implementation-ready algorithm specification.

## Acceptance and staged validation

First profile receipt and implement a small construction prototype against the
same sealed source. Record vector bytes read/written, metadata bytes, CPU,
wall time, page faults, peak RSS, scratch peak and output cardinality. Separate
construction memory from total-process memory and filesystem-cache effects.

Use a proposed 3x end-to-end speedup as the stretch target, not an established
result. The minimum performance gate is faster loading than online on the same
revision and transport with comparable search quality. Re-measure online after
shared ingress changes. No stage-only timing may stand in for the full
receive-to-Active interval.

Compare recall/latency/QPS curves at matched recall thresholds, including 92%
and 95%, plus fixed-beam results. Cover Cohere 1M and SIFT1M, small input,
multiple/skewed TreeKeys, and memory-constrained spill behavior. Do not accept
faster construction by hiding a material query-cost or resource regression.

Before integration, verify global duplicate rejection, exact membership,
deterministic output, capacity and centroid correctness; then run existing
artifact, serving, recovery and backend publication tests. Add focused fault
coverage where input sealing or preparation behavior changes. Retain distinct
behavioral guarantees, but remove superseded full-vector sort machinery once the
replacement passes these gates. Only after a replacement algorithm meets throughput and quality goals should it
replace the implemented grouping design. Independent IO changes that preserve
identical index bytes may land after their own throughput and resource gates;
they do not establish that the wider redesign is complete.

## First prototype result (2026-10-09)

An isolated, output-preserving prototype replaced full-vector split sorts with
compact distance/ID key sorts and a sequential rescan/scatter. It projected keys
directly into bounded sorting buffers and retained the 256 MiB construction
budget. Against 81c7f57 on the same Cohere 1M source, a fresh baseline took
255.871 s; two candidate runs took 187.146 and 188.699 s (1.36x forest speedup).
Tree logical writes fell from 119.535 GB to 55.733 GB. Forest data and manifests
were byte-identical. This isolates unnecessary merge traffic without changing
search quality; it does not demonstrate end-to-end speedup.

Complementary qualification prevents unconditional adoption. With 1,503 scalar
vectors, 256-byte IDs and a 512 KiB budget, candidate scratch peak increased from
953,904 to 1,214,424 bytes. Keeping the original vectors during final key merging
can cost more when keys dominate row size. Some subsecond synthetic timings also
regressed; they require repetitions before estimating a stable effect. Across
12 synthetic cases, all emitted plans remained byte-identical for cosine, L2 and
inner product, repeated vectors, reverse input, and different spill budgets.

Whole-probe RSS was about 428 MB for baseline and candidates, while macOS peak
footprint increased by 16–46 MB, with cause unresolved. These metrics are not the
construction buffer ceiling. Existing 52 focused tests, Clippy, formatting and
independent review completed; the review's scratch concern was reproduced.

The prototype is **not integrated**. It is retained as experimental evidence at
`/Users/lloyd/projects/ktann/.benchmark-data/results/bulk-compact-split-20261008/`,
including `REPORT.md`, `prototype.patch`, probe sources and raw measurements.
The next design must reduce vector passes and assignment work together; replacing
payload sorts with key sorts alone neither meets the several-fold throughput
goal nor removes the need for resource/quality qualification.


## Qualified IO changes (2026-10-09)

The retained implementation preserves version 1 grouping and exact forest bytes.
It projects split keys to fixed-size distance/ordinal rows, preserving canonical
ID order through stable sequential scatters and eliminating final ID sorts. For
very small vector rows it retains full-row sorting, so keeping original vectors
during key sorting cannot recreate the scalar/long-ID scratch regression above.
The generic artifact sorter selects two to eight merge inputs within its existing
memory budget and consolidates smaller remaining runs first. No new public knob
or full-dataset resident structure is introduced.

With 256 MiB construction memory and the same Cohere 1M source, a clean serial
baseline/candidate forest comparison measured 244.645→163.580 s (1.50x); a separate
candidate execution in the native RocksDB harness measured 164.635 s. Tree writes
fell 119.535→52.635 GB and global-sort writes 41.261→18.546 GB. Tree scratch peak
remained 6.730 GB and sort peak 9.272 GB. Data and manifest were byte-identical.
System CPU fell 149.02→74.85 s; user CPU stayed 86.10→84.68 s. RSS fell 421.6→340.6 MB
and macOS peak footprint stayed 290.2→291.7 MB. These process metrics are distinct
from the 256 MiB construction working-buffer reservation.

The same native RocksDB harness measured sealed-source-to-Active time at
412.888 s baseline versus 315.153 s candidate (23.7% reduction, 1.31x). Its separate
forest executions were 248.382 s and 164.635 s. This interval excludes receipt
and snapshot writing. All 1,000 top-100 queries retain exact recall
89.074% / 93.123% / 95.341% at beams 128/192/256. Serial query throughput differs
only slightly; it is neither a demonstrated query improvement nor canonical
concurrent VectorDBBench QPS. Whole native build/query RSS rose 1.466→1.542 GB,
while peak macOS footprint fell 1.616→1.531 GB; do not claim a whole-process
memory reduction from these mixed residency measures.

A previous candidate run under low free disk space took 434.105 s and was briefly
profiled. It is recorded but excluded from the clean timing pair; write/unlink
calls dominated the sampled interval, without proving disk capacity alone caused
the difference. Disposable compiler caches and rejected experimental databases
were removed before the clean serial comparison.

Three alternating baseline/candidate repetitions of 21 complementary cases cover
all metrics, variable IDs, scalar and 768-dimensional vectors, small and roomy
budgets, projection thresholds, duplicates in vector values, and long IDs. All
plans and peak scratch counts match; projected cases reduce writes, while
scalar fallback cases retain the original work. Independent review found no
actionable issues. Full publication/query measurements are recorded alongside
raw evidence under
`/Users/lloyd/projects/ktann/.benchmark-data/results/bulk-io-20261009/`.

## Rejected multiway grouping experiments

A sampled binary routing tree fused up to eight scatter destinations. It reduced
forest writes but raised scratch peak 6.730→7.531 GB, changed partition occupancy,
and reduced Cohere recall at all tested beams. A flat eight-centroid Lloyd
variant further increased training time; its forest took 480.220 s and likewise
missed the quality goal. Neither algorithm is retained in production. Their
patches, query results, profiles and limitations are preserved under
`/Users/lloyd/projects/ktann/.benchmark-data/results/bulk-multiway-20261009/`.

The evidence rejects these particular implementations, not multiway construction
in general. Further algorithm work must improve total training/assignment/IO
cost and matched-recall query cost together. The implemented IO changes do not
resolve Bulk Build's lower default-beam recall or demonstrate a 3x end-to-end
speedup over online import.
