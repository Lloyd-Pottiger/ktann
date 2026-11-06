# Leaf candidate selection experiments, 2026-11-06

No end-to-end improvement was demonstrated. All three experimental code changes
were reverted. The retained artifacts are this report and the
[measurements, patches and profile excerpts](leaf-selection-2026-11-06.json).

## Scope and baseline

Baseline: `fb4bdd872589c2093430e5a27556931939849067`, after rebasing the earlier
HNSW work and fixing the RaBitQ property generator. This experiment investigates
the approximate-selection stage; it does not change routing quality parameters.

Release build, RocksDB, synthetic clustered L2 data: 32,768 vectors, dimension 16,
64 queries, 1,000 warmup and 20,000 measured searches per process. Eight concurrent
queries, four Tokio workers, 64 MiB partition cache. The frozen tree has 3,215
leaves of at most 16 entries and one internal root. HNSW uses M=12,
efConstruction=100, efSearch=200, seed=7; the leaf beam is 128. This deliberately
wide-root fixture is a diagnostic, not a representative production capacity test.

The older fixture under `.benchmark-data/hnsw-visited/fixture` failed the current
binary's reuse verification (`complete=false, issues=100`). The cause was not
diagnosed here. It was left intact; a fresh fixture was constructed with the
current baseline. All reported comparisons use that same new fixture, which
passed reuse and final verification. Its dataset checksum and topology match
the earlier workload. No verification was disabled.

Each trial used a separately copied baseline binary and a candidate binary,
without concurrent Cargo builds or other benchmark runs from this session.
Run order was A/B/B/A for the scheduling trial and A/B/B/A/A/B for the other two.
Unrelated host activity was not controlled. The JSON records binary SHA-256,
environment, arguments, patches, every run, and aggregate medians. The runner's
`git_revision` is read at execution time and can say `-dirty` even for the saved
clean-baseline binary; use the artifact hashes and recorded baseline revision.

## Profiling

GNU gprofng timer sampling (`collect app -p hi`, following child processes) on the
baseline found approximately 2.0% of collected CPU samples in RaBitQ interval
scoring, 3.5% in `Bytes` ID cloning, and 11.3% in `syscall`. Caller attribution put
92% of the ID-clone samples under candidate admission, and 49% of the syscall
samples under Tokio parking/wakeup. Allocation, RocksDB reads and partition
reference validation were also visible. These are sampling observations, not
precise per-stage CPU accounting: the profile includes setup, warmup and
verification, and attribution/sampling can be incomplete. Timed A/B runs did
not run under the profiler.

Approximate selection averaged about 1.8 ms and exact reranking about 0.63 ms in
the unprofiled baseline. These elapsed-time stages include storage and scheduling
waits. They do not establish that leaf scoring itself consumes 1.8 ms of CPU.

## Experiments

| Trial | Baseline QPS | Candidate QPS | Change | Baseline/candidate P95 ms | Baseline/candidate CPU seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Yield every 16 bodies instead of 4 | 2882.5 | 2912.8 | +1.05% | 3.808 / 3.807 | 33.170 / 32.714 |
| Query-owned contiguous ID buffer | 2871.7 | 2846.2 | -0.89% | 3.782 / 3.914 | 32.794 / 32.673 |
| Reuse visit buffer; stream prefetch requests | 2891.9 | 2897.9 | +0.21% | 3.747 / 3.736 | 32.806 / 32.944 |

Values are medians across two baseline/candidate processes for the first trial
and three each for the others; every CPU value covers 20,000 measured searches.
Individual runs overlap, pairwise changes are inconsistent, and effects are
small compared with observed process-to-process spread. These small samples do
not support a statistical equivalence claim or a reliable speedup.

1. **Scheduling:** retain four-body IO batches, but explicitly yield once per
   four batches. Per-entry yields remain. The candidate reduces explicit
   scheduling boundaries, but does not demonstrate an end-to-end win. Reverted;
   the existing fairness policy remains unchanged.
2. **Identity collection:** replace the per-eligible-entry `Bytes` clone with
   copies into one byte buffer plus end offsets. Duplicate validation borrows
   complete byte slices, preserving screened and evicted IDs; accounting changes
   to cover buffer capacities and the larger borrowed-slice hash slots. This
   avoids shared ID reference updates but introduces copying and a different
   memory layout. Throughput did not improve. Reverted before broader long-ID,
   memory and correctness validation; no benefit is claimed for those paths.
3. **Temporary buffers:** retain the four-entry prepared-visit vector between
   groups, draining it after each group, and pass prefetch requests as an iterator
   instead of materializing another vector. For the 128-leaf warm beam, the code
   removes 63 temporary vector allocations per search (31 visit buffers and 32
   request buffers); this is a source-level count, not an allocator measurement.
   IO grouping, yields and snapshot validation remain unchanged. The observed
   throughput difference is negligible, so the performance candidate was reverted.

All 16 unprofiled runs accepted all searches without errors and had mean
recall@10 = **98.126%**. Their sorted per-search recall multisets, logical budgets,
visited-entry summaries, cache counts, backend IO counters and HNSW work counters
match exactly. Recall arrays are sorted for comparison because concurrent
completion order varies; this does not prove exact result-ID parity. Each search
visits 129 partitions and roughly 1,385 Leaf Entries and reranks 64 records.
Physical read bytes were zero. Peak RSS was not captured by this profile.

## Reproduction and remaining evidence

Build the baseline with:

```sh
cargo build --release -p ktann-benchmarks --bin ktann-bench
```

Copy the binary before applying an experiment. The companion JSON contains each
rejected patch against the baseline and the complete CLI argument array. Create
a separate RocksDB fixture using those arguments with `--warmup-operations 0`,
`--measured-operations 1`, `--query-concurrency 1` and a new `--rocksdb-path`.
Subsequent runs use the same path plus `--reuse-index true` and distinct `--output`
paths. Rebuild and copy one candidate at a time, restoring the baseline between
patches. Profile separately from timing:

```sh
gprofng collect app -p hi -o baseline.er /path/to/baseline run ...
gprofng display text -limit 30 -functions baseline.er
gprofng display text -limit 10 -csingle syscall baseline.er
```

Raw runs, copied executables, scripts and profiler data remain locally under
`.benchmark-data/leaf-selection/`; the committed JSON preserves the summarized
evidence and patches without requiring those ignored files.

All candidates compiled in release mode. The benchmark verifies the fixture
before and after every run. No full Rust test or Clippy gate was run for the
rejected implementations, and they must not be treated as production-ready.
After restoration, production code and tests match the baseline byte-for-byte;
only reports and their README link remain changed.

The next useful evidence is a matched-recall workload with realistic dimensions,
leaf sizes and filters, followed by separation of backend service time from
queueing. This tiny-leaf fixture magnifies per-leaf metadata and coordination
costs. The current measurements do not justify replacing HNSW, relaxing
corruption checks, changing recall budgets, or claiming these micro-changes close
the performance gap with Milvus.
