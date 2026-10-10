# Bulk Build Performance

Status: **Qualified locally on RocksDB/Cohere1M; wider validation remains open.**
The latest same-revision comparison measured 2.85–3.98x faster loading to Ready,
with similar diagnostic query throughput around 95.3% recall. These are serial
observations on one development host, not confidence intervals or an SLA.

The lifecycle, fencing, resource ownership and publication contracts are defined
in [Resumable Bulk Build](bulk-build.md), [ADR 0027](../adr/0027-bulk-workspace-and-publication.md)
and [ADR 0028](../adr/0028-automatic-bulk-scheduling.md). This note describes the
physical pipeline and its performance evidence. Measurements below predate the
current simplification of Serving scratch runs; no new speedup or memory result
is attributed to that cleanup.

## Current pipeline and resource contract

- **Receipt:** Online and bulk use the same bounded binary float32 transport:
  50 records per batch, 8 MiB frames, count/dimension/body-length validation and
  acknowledgements supplying backpressure. Unknown online commit outcomes are
  returned without replay. Bulk acknowledgements mean capture, not publication
  or per-batch durability.
- **Source preparation:** `PreparedInputWriter` preserves canonical complete
  records in original order while sorting compact ID rows and Tree Key/vector
  projections. Two row buffers share one IO/merge reservation and scratch quota;
  their IO is serial. EOF flushes, syncs and seals the source. A failed consuming
  append cannot subsequently seal a partial batch. Prepared work is one-use,
  bound to source/configuration/options, and disposable; ordinary recovery
  rebuilds it from the sealed source.
- **Forest:** Global duplicate rejection precedes partition emission. Construction
  retains deterministic binary grouping and final centroids over every assigned
  member. Resident groups operate within the existing working-memory budget;
  larger groups spill. External splitting uses distance/ordinal metadata and
  stable sequential scatters, retaining full-row sorting when small vector rows
  would otherwise increase scratch use. Sort fan-in is bounded and memory charged.
  Partition training and metric normalization begin after EOF.
- **Serving:** External joins establish exact source/leaf membership and
  parent/child relationships. Encoded rows are grouped by their final serving key;
  record groups stream to the artifact, while tree rows are sorted once and merged
  with an ordered Synopsis run. Original vectors, fields and payloads use core
  codecs; Leaf Entries use the online absolute RaBitQ7 codes. Output keys and values
  obey backend hard limits. No all-record assignment map is retained in RAM.
- **Load and publication:** Fenced, bounded transactions load hidden serving data.
  Exact sealed-backend validation precedes atomic publication and workspace
  reclamation. Optimized receipt/construction cannot weaken those proofs.
- **Readiness:** The benchmark audits bounded Header snapshots and directly advances
  up to 32 split/merge sources per round. Approximate centroid probes do not
  guarantee visitation of cold sources. The scan ceiling is 262,144 allocated
  slots; idle/stalled/deferred-error rounds wait one second. Online foreground
  retries are bounded at 32 attempts, with backoff and no replay of uncertain
  insert outcomes.

SHA-256 uses sha2 0.11 runtime CPU detection with a software fallback
([upstream documentation](https://docs.rs/crate/sha2/0.11.0)). Artifact and source
identities, persistent bytes and publication proofs are unchanged by CPU-specific
hash acceleration. Hardware gains are architecture dependent.

Working-buffer budgets are not process-RSS ceilings. Construction, sort and
Serving allocations must remain bounded independently of total dataset size;
backend cache, allocator reservation and OS residency must also be measured.
The external-memory path is required, but terabyte scale and workloads larger
than physical RAM have not been qualified on suitable hardware. `mmap` or compact
references alone do not prove bounded residency or lower IO: random gathers can
repeatedly fault, and long IDs can dominate metadata cost.

## Latest complete online comparison

The 2026-10-10 measurement-only follow-up uses production revision `2ead212`, one
frozen bridge binary and identical dataset identities on an Apple M1 Pro / 16 GiB
host: RocksDB, Cohere1M, 768-dimensional cosine, one loader and batches of 50.
Canonical VectorDBBench query concurrency is 1/5/10/20 for 30 seconds each.
Shared settings include eight maintenance workers and a 4 GiB partition cache;
bulk separately uses 256 MiB construction and 128 MiB Serving working budgets.

Runs occurred serially in Bulk A / online / Bulk B order, with diagnostics
between runs and no overlapping build or benchmark. Online uses direct
`batch_mutate` under [ADR 0025](../adr/0025-caller-owned-online-batch-submission.md).
Both modes reached one million records, no actionable or transitional partitions,
and maximum leaf size at most 512. This readiness audit neither proves global
clustering optimality nor replaces full backend integrity validation.

| Canonical metric | Bulk A | Online batches | Bulk B |
| --- | ---: | ---: | ---: |
| Receipt / insertion (s) | 57.703 | 266.447 | 53.576 |
| Optimize to Ready (s) | 176.092 | 663.543 | 272.327 |
| Complete load to Ready (s) | 233.795 | 929.989 | 325.902 |
| Recall@100, beam 128 | 0.8907 | 0.8848 | 0.8907 |
| Maximum canonical QPS, beam 128 | 270.422 | 309.326 | 262.525 |
| Serial p95 (ms) | 30.4 | 25.1 | 25.5 |
| Sampled full-run RSS peak (GB, decimal) | 1.899 | 5.155 | 1.690 |

Both bulk source manifests match byte for byte. The observed load-time ratio is
**2.85–3.98x** in bulk's favor; variation between the identical bulk runs remains
visible. Only one online run was measured. Canonical JSON, rather than diagnostic
replay, supplies the load/QPS results.

The first capture wrapper made an unsupported post-run request and exited with
an error after upstream VectorDBBench completed successfully. Its canonical JSON
and continuous samples remain valid, but its native summary is missing. Later
wrappers completed and verified Ready through the supported health operation.

### Query quality at comparable recall

Diagnostics reopen the exact completed indices and evaluate all 1,000 ground-truth
queries at common beams. Bulk B at beam 256 reproduces every Bulk A result hash.
Other runtime/search settings match; calibration does not rebuild the indices.

| Beam | Bulk Recall@100 | Online Recall@100 | Bulk fixed-count QPS | Online fixed-count QPS |
| --- | ---: | ---: | ---: | ---: |
| 128 | 0.89074 | 0.88483 | — | — |
| 192 | 0.93123 | 0.92455 | — | — |
| 256 | 0.95341 | 0.94551 | 141.42 | 160.93 |
| 288 | 0.96083 | 0.95306 | 118.49 | 138.84 |

Throughput uses 4,000 fixed queries at concurrency 20 after a 1,000-query cache
warmup, with identical per-worker assignments. These timings are separate from
canonical VectorDBBench QPS. At approximately **95.3% recall**, bulk beam 256
reaches **141.42 QPS** and online beam 288 **138.84 QPS**; recalls differ by 0.035
percentage points. This supports similar throughput in this quality neighborhood,
not a query speedup or optimal ANN frontier. At equal beam, bulk has higher recall
and lower throughput; equal beam is not equal quality.

## Memory evidence and limitations

Controlled SHA 0.10.9/0.11 replay keeps the persisted index, queries, assignments
and concurrency fixed. Each variant runs 19,000 queries: 1,000 serial warmup,
then concurrency 1/5/10/20, including three 4,000-query rounds at concurrency 20.
A cold-cache control immediately runs concurrency 20 on the same 1,000-query
corpus three times. All 41,000 result hashes match and backend mutation counts
remain zero. MB below is decimal.

| Diagnostic process | Cache MB | Final live malloc MB | Final reserved malloc MB | Native peak RSS MB |
| --- | ---: | ---: | ---: | ---: |
| SHA 0.10.9, fixed 19K queries | 886.435 | 1106.626 | 1358.971 | 1354.007 |
| SHA 0.11, fixed 19K queries | 886.435 | 1106.643 | 1363.165 | 1285.587 |
| SHA 0.11, cold concurrency 20, 3K queries | 886.435 | 1106.633 | 1400.914 | 1392.919 |

Cache and live heap plateau after warmup. Paired final allocator reservations
vary by about 4 MiB and live allocations by about 17 KB: these controls do not
reproduce a SHA-upgrade query-heap regression or accumulating per-query live
heap. RSS alone is not an allocation measure; `vmmap` also records compressed
and swapped pages. Cold concurrent loading adds 702 scans and about 56.13 MB of
reads, supporting it as a transient-memory contributor. A diagnostic allocator
pressure-relief request released zero bytes; no workaround is retained.

Reopening the exact Bulk B database in a fresh process reproduces 1,000 query
results without writes. Live heap falls from 1,254,773,728 to 1,106,592,288 bytes.
The disappearing 141 allocations of 1 MiB account for almost all of the
148,181,440-byte difference; three RocksDB MemTables become one. Saved options
specify 1 MiB arena blocks and a 128 MiB write-history target. RocksDB 10.4.2
retains flushed memtables for optimistic transaction conflict checking, consistent
with this load-lifetime memory. Shrinking history can cause optimistic commit
failures and is not a free memory improvement. Dirty-plus-swapped malloc-zone
fragmentation falls from 262.6M to 107.4M after reopening.

These controls identify contributors, but do not uniquely explain the earlier
1.87 GB versus 1.24–1.57 GB whole-run peak difference or establish full-build
memory parity. No leak fix, cache-budget change, history reduction or allocator
tuning follows from this evidence. Diagnostic probes remain outside production.
The restored production bridge hash matches the frozen canonical binary.

## Retained changes and rejected tradeoffs

Historical measurements explain the current mechanisms; they are not additional
samples of the latest online comparison.

| Qualified mechanism | Isolated evidence | Qualification boundary |
| --- | --- | --- |
| Bounded metadata splits and stable scatter | Forest 244.645→163.580 s; tree writes 119.535→52.635 GB; sort writes 41.261→18.546 GB; identical manifests | Sealed-source-to-Active 412.888→315.153 s excludes receipt; whole native build/query RSS 1.466→1.542 GB |
| Incremental source + common binary ingress | Complete bulk load 766.237→391.729 s; snapshot finalization 26.499→0.018 s; identical source | One bulk pair, no online baseline; native peak RSS 1.839→2.007 GB |
| Compact global ID uniqueness | Forest 177.670→171.480 s; sort writes 18.546→12.360 GB; identical manifests | 3.5% stage reduction, not a several-times end-to-end result; low-dimension/payload controls approximately flat |
| Budgeted resident groups | Forest 170.689→115.044 s; tree writes 52.635→30.838 GB | Complete bulk pair 433.589→379.660 s; whole-run RSS 1.339→1.597 GB |
| Receipt-time paired sort preparation | Complete bulk load 402.806→340.482 s; identical source and forest; no budget raised | One full pair; whole-run RSS 1.617→1.889 GB; complementary cases preserve bytes and scratch bounds |
| SHA runtime acceleration | Same-source/forest Serving 89.895→53.610 s; CPU 82.29→47.78 s; identical 4,031,515,788-byte artifact and 3,010,259 keys | Scratch writes remain 27.991 GB, peak 12.526 GB; Apple CPU result |

For SHA acceleration, alternating frozen baseline / candidate / baseline complete
loads took 371.233 / 294.558 / 391.499 s, reducing time by 20.7%–24.8%. Ready
peak RSS was 392.905 / 428.245 / 431.096 MB, while whole-run peaks were
1238.614 / 1868.677 / 1572.192 MB. The last baseline and candidate completed
21,823/21,840 queries, so query count alone does not explain the peak difference.
Whole-run CPU/RSS and variable-count timed QPS do not isolate build costs.

Rejected alternatives remain useful constraints:

- A compact distance/ID split prototype preserved bytes and cut forest writes,
  but scalar vectors with 256-byte IDs at a 512 KiB budget raised scratch peak
  from 953,904 to 1,214,424 bytes. The retained design avoids that case with
  full-row sorting; compact metadata is not universally smaller.
- Fusing up to eight scatter destinations raised scratch peak 6.730→7.531 GB,
  changed occupancy and reduced recall. Flat eight-centroid Lloyd grouping took
  480.220 s and also missed the quality goal. These implementations are rejected;
  multiway construction in general remains an open research direction.
- An ID/ordinal membership join added a second complete source scan. With software
  SHA it increased Serving time 89.895→98.673 s. With sha2 0.11 it reached
  47.755–49.471 s, but peak RSS rose to 300.237–319.193 MB. Its modest speed gain
  did not justify the extra scan, memory and software-CPU regression; it is absent
  from production.

Future algorithm changes must improve training, assignment, IO, matched-recall
query cost and resources together. Keep global uniqueness, exact membership,
capacity, canonical bytes, deterministic fixed-option output, bounded scratch,
fencing and atomic visibility. There is no stable-release compatibility
requirement, but immutable in-progress descriptors must never silently resume
under changed construction semantics.

## Reproduction and evidence

Build the production bridge from the revision being measured:

```sh
cargo build --release -p ktann-benchmarks --bin ktann-vdbbench-bridge
```

Use a fresh database/socket/report location per run. To select bulk, add
`--bulk-workspace /absolute/new/directory`; omit it for online. From the installed
`/Users/lloyd/projects/VectorDBBench` checkout, the common Cohere workload is:

```sh
vectordbbench ktann --socket-path /tmp/ktann-bench.sock \
  --dataset-identity cohere-1m --case-type Performance768D1M \
  --load-concurrency 1 --insert-batch-size 50 --k 100
```

See [benchmark instructions](../../benchmarks/README.md#vectordbbench)
for bridge startup, process tests, diagnostics and FoundationDB setup. The client
checkout's `vectordb_bench/backend/clients/ktann/README.md` owns CLI usage.
Historical wrappers, frozen binary hashes and configuration receipts in the
following archives are needed to reproduce the exact measured settings; a
current binary run is a new measurement. Construction-only and complete SIFT
probes are documented in [the benchmark README](../../benchmarks/README.md).
They do not replace the canonical end-to-end comparison.

All archives are local under
`/Users/lloyd/projects/ktann/.benchmark-data/results/`:

| Archive | Evidence |
| --- | --- |
| `bulk-memory-import-20261010/` | Latest canonical JSON, RSS traces, quality calibration, same-index replay, heap/vmmap, RocksDB options, frozen hashes and diagnostic scripts; `summary.json`, `verification.json`, `memory-verification.json` |
| `bulk-serving-20261010/` | SHA CPU sample, source/forest reuse probe, alternating canonical runs, compact-join rejection and validation logs |
| `bulk-pipeline-20261009/` | Receipt-time preparation, complementary spill/metric/ID matrix, final hashes and full reports; failed zero-query run excluded from complete comparisons |
| `bulk-resident-20261009/` | Budgeted resident-group probes, `full-baseline/`, `full-candidate/`, `full-comparison.json` |
| `bulk-projection-20261009/` | Compact global uniqueness, payload/low-dimension controls and byte-identity checks |
| `bulk-ingress-20261009/` | Binary ingress and streaming source comparisons; interrupted online run is not completion evidence |
| `bulk-recovery-20261009/` | Earlier completed online readiness comparison and cold-source regression |
| `bulk-multiway-20261009/` | Rejected multiway implementations, profiles, query results and patches |
| `bulk-io-20261009/` | Qualified IO/scatter comparison and native sealed-source-to-Active probe |
| `bulk-compact-split-20261008/` | Initial rejected prototype, scalar/long-ID scratch regression, `REPORT.md` and patch |
| `vdbbench-ab-20261007/`, `bulk-sort-20261008/`, `bulk-sort-buffer-20261008/` | Early baselines and sort-buffer evidence; historical RSS increase remains unassigned |

Generated database/workspace data may be reclaimed after diagnostics; manifests,
reports and reproduction evidence remain. Raw records distinguish canonical QPS,
fixed-query diagnostics, Ready high-water RSS and whole-run RSS.

## Remaining qualification

Repeat online and bulk runs on an otherwise idle host and compare recall/latency/
QPS curves at matched quality, including 92% and 95%, rather than equal beam alone.
Qualify Cohere1M and SIFT1M, small input, multiple/skewed Tree Keys, long IDs and
constrained-memory spill paths. Larger-than-memory input, terabyte-scale hardware,
FoundationDB, other CPUs/software hashing and a broader ANN frontier remain open.
A 3x universal end-to-end target is not established by the local ratio above.

Any new Serving cleanup requires its own byte/contract tests and resource evidence.
Run focused artifact, exact-join, load/recovery/publication tests plus formatting
and workspace Clippy; preserve distinct fault and corruption guarantees. Measure
physical IO, scratch peak, CPU, page faults and RSS alongside full receive-to-Ready
and query quality. A faster stage cannot establish a whole-system improvement or
justify materially worse query cost or resources.
