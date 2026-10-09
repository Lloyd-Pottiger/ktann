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

The original JSON receipt path took 413–439 s. Even free construction limits speedup
against the observed 749–842 s online baseline to roughly 1.7–2.0x. A 3x
end-to-end target therefore required improving receipt too; the streaming input
section below records that change. Applying the same
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

## Streaming input capture and binary transport (2026-10-09)

The benchmark adapter uses the same binary float32 insert transport for online
and bulk modes. It retains the 50-record batch and 8 MiB frame bounds and awaits
each acknowledgement. Protocol version 2 rejects stale version-1 clients;
there is no dual-format fallback for inserts. NumPy performs bounded conversion
and finite-value validation, replacing Python scalar conversion/check loops and
JSON decimal float serialization. The Rust decoder validates count, dimension
and exact body length before allocating vectors.

`InputSnapshotWriter` incrementally validates and encodes canonical records and
updates their integrity hashes as batches arrive. Appending consumes the writer;
an error cannot subsequently seal a partial batch. EOF only flushes, syncs and
seals the source. Failed/cancelled sources remain caller-owned and unsealed.
`InputSnapshot::create` delegates to the same implementation. This removes the
raw temporary file and the subsequent full read/encode/rewrite, while preserving
source bytes and immutable publication rules.

Two alternating baseline/candidate measurements used 100,000 real Cohere vectors
preloaded outside the timed section, fresh RocksDB databases and no profiler:
receipt fell 39.69/39.43 s to 4.37/4.46 s, about 9x faster; receipt plus Optimize
fell 70.07/70.35 s to 32.36/32.79 s. Source manifests are identical across all
four runs. A separate profile attributed 29.3 s to JSON encoding and substantial
additional time to Python per-component checks; profiling timings are not used
as the speedup denominator. Request bodies fell from 1.535 GB to 308 MB for
100,000 vectors. Native decode fell 2.70 s to 0.013 s. Canonical source encoding
moves into receipt, so the near-zero snapshot sealing time is not free work.

Clean-run RSS varied: baseline 545–722 MB, candidate 855–872 MB. Additional phase
memory maps showed received physical footprint 72.8/68.7 MiB and post-build peak
physical footprint 345.3/372.8 MiB, with comparable post-build resident totals
699.2/693.0 MiB and substantial empty malloc regions in both. This rules out
retaining the full received dataset in that run, but does not justify claiming a
whole-process memory reduction or attributing all RSS variation to one cause.
The source writer retains only bounded IO/record buffers.

A same-revision canonical Cohere 1M comparison (JSON/raw staging versus
binary/canonical capture) measured:

| Metric | Before | After |
| --- | ---: | ---: |
| Receipt | 413.007 s | 62.926 s |
| Optimize | 353.230 s | 328.803 s |
| Total load | 766.237 s | 391.729 s |
| Recall@100 | 0.8907 | 0.8907 |
| Maximum QPS | 288.857 | 287.342 |
| Serial p95 / p99 | 23.8 / 24.4 ms | 28.6 / 30.5 ms |
| Native peak RSS | 1.839 GB | 2.007 GB |

The load time is 48.9% shorter (1.956x). The full source manifests match exactly.
Native prepare/load is 290.361/292.375 s; snapshot finalization is 26.499/0.018 s,
with canonical encoding now counted during receipt. This demonstrates the
receipt improvement and measures its full bulk benefit, but one full pair does
not isolate the higher RSS or serial tail latency. No memory reduction or query
latency improvement is claimed. These are two bulk runs with the same tree
algorithm, not a completed comparison with online import.

This change starts input preparation during receipt. Global ID/tree sorting and
final partition training still begin after source sealing. Moving those stages
before EOF requires an explicit contract for reusable sorted input or a new
partitioning algorithm; the incremental writer alone does not supply it.
Evidence, profiles, source patches and raw runs are under
`/Users/lloyd/projects/ktann/.benchmark-data/results/bulk-ingress-20261009/`.

The same binary transport was also exercised by a canonical online Cohere 1M
run. Its insert worker finished in 238.04 s, but Optimize stopped showing
progress with 133 transitional partitions and was interrupted. It supplies no
completed online load, recall or QPS result. The bridge's existing rediscovery
probes repeatedly select the first 32 pending centroids with leaf beam 1;
approximate routing need not visit their target partitions. Review confirmed
this liveness gap, but did not establish it as the cause of this particular
plateau. At that point, large online completion and the bulk-versus-online comparison
were unverified; the completed follow-up is recorded below. Raw logs and the unsuccessful recovery diagnostic are retained in
the evidence directory; small online process tests passed.


## Compact global uniqueness sorting (2026-10-09)

The forest previously carried full vectors through global ID sorting before
sorting them again by Tree Key. It now sorts IDs alone and sequentially spools
the existing tree projection under the same scratch quota. Source payloads are
decoded once. Buffered IDs are checked without writing a run; spilled IDs retain
bounded external merge sorting. No training, membership, artifact identity or
memory-budget change is involved.

A serial Cohere1M forest-only comparison against `c65eed6`, using the same sealed
source, seed, sample 256, 64 MiB sort and 256 MiB construction budgets, measured:

| Metric | Before | After |
| --- | ---: | ---: |
| Forest build | 177.670 s | 171.480 s |
| Global sort writes | 18.546 GB | 12.360 GB |
| Global sort peak scratch | 9.272 GB | 9.270 GB |
| Tree construction writes | 52.635 GB | 52.635 GB |
| Native CPU, user + system | 164.53 s | 162.36 s |
| Maximum resident memory | 350.323 MB | 350.159 MB |

The complete forest manifests are byte-identical. Sort writes fall 33.4%, but
forest wall time falls only 3.5%; this is not a several-times speedup or a new
end-to-end VectorDBBench result. Unchanged tree construction still accounts for
most temporary writes. An intermediate candidate measured 170.463 s with the
same forest, illustrating that the small timing difference is not a precise
performance guarantee.

Complementary alternating before/after/after/before runs use variable-length
IDs and eight-dimensional vectors: 100k records without payload, and 20k with
16 KiB payload per record. Final forest bytes match in all runs. Low-dimensional
wall times are 1.232/1.243 s before and 1.234/1.219 s after; payload times are
2.180/2.179 s before and 2.177/2.179 s after. Sort writes decline from 30.10 to
29.90 MB and 6.02 to 5.98 MB respectively. An intermediate implementation that
always wrote the ID run increased these writes and was superseded.

Validation: 242 core unit tests (one existing ignored), 14 artifact tests,
32 serving tests, four bridge tests and seven real client process tests pass;
workspace all-target/all-feature Clippy and formatting pass. Fresh independent
reviews found no actionable issues. Reproduction tools, binary hashes, patches,
raw time/resource reports and intermediate experiments are archived under
`/Users/lloyd/projects/ktann/.benchmark-data/results/bulk-projection-20261009/`.


## Complete online readiness comparison (2026-10-09)

Approximate centroid probes cannot guarantee that cold maintenance sources are
visited. A committed two-split, identical-vector regression leaves an oversized
cold source and reproduces this gap. The benchmark bridge now reads bounded
Header snapshots and directly advances up to 32 split/merge sources per round,
using the existing authority-revalidating state machines. Progress starts the
next round immediately; idle/stalled/deferred-error rounds wait one second.
The header scan ceiling remains 262,144 allocated slots. Online import uses the
existing foreground retry policy with 32 attempts (previously 8), retaining its
bounded backoff and returning unknown insert outcomes without replay.

The final canonical RocksDB/Cohere1M run completed: 1M records, no actionable or
transitional partitions, maximum leaf size 512. Query concurrency was 1/5/10/20
for 30 seconds each, using the same default search budget as the earlier bulk
run. The completed measurements are:

| Metric | Earlier binary-ingress bulk | Online with readiness recovery |
| --- | ---: | ---: |
| Receipt / insertion | 62.926 s | 222.286 s |
| Optimize | 328.803 s | 827.685 s |
| Complete load | 391.729 s | 1049.971 s |
| Recall@100 | 0.8907 | 0.8875 |
| Maximum QPS | 287.342 | 314.464 |
| Serial p95 / p99 | 28.6 / 30.5 ms | 26.3 / 27.7 ms |
| Native peak RSS | 2.007 GB | 5.802 GB |

For these completed runs, bulk reaches the benchmark's stable readiness state
2.68x faster with similar measured recall, while online query throughput and
tail latency are better. This is one online run compared with the earlier bulk
run, not repeated evidence of a universal speedup or a fresh end-to-end run of
the compact-ID change. The two forests' algorithms and partition counts differ;
previous online query-quality observations also differ from this run. Neither
an index-quality improvement nor the historical quality gap's general resolution
is established. Online's insertion completion alone is not stable readiness;
this benchmark does not measure the latency/quality of searching while online
maintenance is still active.

The native partition-cache configuration is 4 GiB in both modes; forest working
memory remains separately bounded. Whole-run CPU, including all benchmark
queries, is 948.671/3532.180 seconds, with different query counts; this is not a
build-only CPU comparison. The online report has zero failed/unknown commits
and 23,545 retryable commit aborts. Its final header snapshot is a readiness
audit, not a full backend integrity verification.

Two preceding recovery runs are excluded: the first exhausted the old eight
foreground attempts at 970,400 inserted records; the next was intentionally
stopped because its driver imposed a fixed one-second wait even after progress.
Their logs remain available. All four bridge tests, including cold-source
convergence and full small-fixture integrity verification, and all seven real
client process tests pass. Independent review found no actionable issues in
both the recovery path and its work-conserving scheduling follow-up.

Raw canonical reports, native diagnostics, failure logs, binary identity and
review notes are under
`/Users/lloyd/projects/ktann/.benchmark-data/results/bulk-recovery-20261009/`.
The final binary identity is also recorded in the sibling
`bulk-projection-20261009/final-binaries.sha256`.


## Budgeted in-memory groups (2026-10-09)

The remaining construction cost included repeatedly writing and decoding groups
that already fit the configured memory reservation. Such a group now loads once
and uses in-place distance/ID selection over disjoint child slices. Only the
bounded training sample is copied; terminal rows are restored to canonical ID
order and consumed by the same emitter as external construction. Larger groups
keep the existing external path. Memory options, logical construction version,
source validation and publication rules are unchanged.

Against `d868e41`, a serial Cohere1M forest comparison with the same sealed source,
rotation seed, sample 256, 64 MiB sorting and 256 MiB construction budgets gives:

| Metric | Before | After |
| --- | ---: | ---: |
| Forest build | 170.689 s | 115.044 s |
| Tree scratch writes | 52.635 GB | 30.838 GB |
| Global sort writes | 12.360 GB | 12.360 GB |
| Peak tree scratch | 6.730 GB | 6.730 GB |
| Native CPU, user + system | 159.16 s | 108.77 s |
| Maximum RSS | 356.778 MB | 344.867 MB |
| macOS peak physical footprint | 335.627 MB | 338.510 MB |

Complete forest manifests are identical. Forest elapsed time falls 32.6% (1.48x),
tree writes fall 41.4%, and CPU falls 31.7%. RSS and physical footprint move in
opposite directions by small amounts; no general memory reduction is claimed.

Three alternating baseline/candidate repetitions across 21 complementary cases
cover all three metrics, short and maximum-length IDs, repeated vectors, row
projection thresholds, and small/roomy budgets. Every plan byte matches; peak
scratch is unchanged in every case. Per-case median elapsed reductions range
from about 24% to 65%, with fewer scratch writes throughout. A regression test
also compares minimum-valid memory (one sort row, forcing external groups)
against resident construction, checking centroid bits, IDs and partition keys.

Validation passes: 243 core unit tests (one existing ignored), 14 artifact tests,
32 serving tests, seven real client process tests, workspace all-target/all-feature
Clippy, formatting and independent read-only review. Reproduction sources,
binary hashes, source patch, raw timings and review notes are archived under
`/Users/lloyd/projects/ktann/.benchmark-data/results/bulk-resident-20261009/`.
This physical-work change does not start partition training before input EOF.

A fresh serial canonical VectorDBBench pair on RocksDB, including receipt,
publication and all configured query stages, completed successfully:

| Metric | Before (`d868e41`) | After |
| --- | ---: | ---: |
| Receipt | 65.480 s | 65.768 s |
| Optimize (construction through publication) | 368.109 s | 313.891 s |
| Total load to Ready | 433.589 s | 379.660 s |
| Recall@100 | 0.8907 | 0.8907 |
| Maximum QPS | 257.749 | 266.614 |
| Serial p50 / p95 / p99 | 23.6 / 30.6 / 33.3 ms | 23.7 / 26.7 / 30.9 ms |
| Whole-run native CPU | 908.407 s | 906.618 s |
| Whole-run maximum RSS | 1.339 GB | 1.597 GB |

Both reports confirm Ready with 1,000,000 records and identical source SHA,
rotation seed, configuration and construction budget. Total load falls 12.4%
(1.14x); Optimize falls 14.7%. This is one fresh complete pair, not a repeated
end-to-end speedup claim. Historical full-run timings above are not substituted
for this baseline. Completed baseline-generated database/workspace data was
reclaimed before candidate timing to restore disk headroom; reports and source
identity were retained.

Whole-run RSS rises by 258 MB (19.2%). This measurement includes publication,
backend/cache allocations and timed query stages with different query counts;
it does not isolate construction memory. The separate forest measurement above
shows no corresponding RSS increase, and the construction reservation is
unchanged. These results do not establish equal whole-process memory use or a
query-speed improvement. Whole-run CPU likewise is not build-only CPU. Raw
reports and identity assertions are in `full-baseline/`, `full-candidate/` and
`full-comparison.json` under the report directory above.
