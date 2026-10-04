# Warmed-cache search CPU measurements

The resident CPU path can occupy nearly all eight logical CPUs on this host.
Exact reranking now prepares the caller query once and reuses the dimension and
finiteness guarantees established by manifest-bound persistent record decoding.
Cosine query norms are computed once; record zero norms, ownership/field checks,
snapshot consistency, budgets, and scalar-f64 arithmetic remain unchanged. Empty
reranks skip preparation.

## Workload and reproduction

- macOS 26.6.2, 8 logical CPUs, Rust 1.99.0, optimized release binary.
- In-memory backend, 5000 deterministic records in one flat leaf, no writes or
  topology maintenance during measurement, 512 MiB partition-cache capacity.
- 32 warmed query vectors; `k=100`; exactly 125 reranked records per query.
- Final baseline/candidate runs alternate B/C/C/B. Medians below use two runs
  per binary; builds/tests ran outside the measured intervals. The final raw logs
  use the `ktann-search-cpu-retained-` prefix.
- Cosine: 768 dimensions, 4000 queries per concurrency; L2: 128 dimensions,
  24000 queries per concurrency. Every result checks ordered IDs and exact
  distance bits against warmup, and baseline/candidate aggregate checksums match.

```sh
cargo build --release --example search_cpu
# Archive this executable before modifying the library, then archive the candidate.
./baseline 768 4000 1,4,16 5000 cosine 100
./candidate 768 4000 1,4,16 5000 cosine 100
./baseline 128 24000 4,16 5000 l2 100
./candidate 128 24000 4,16 5000 l2 100
```

The baseline library is commit `01926b55e1425d250e94294026ff824091ec6400`, with
only the benchmark example added. Local binaries and raw measurements are
archived in
`/Users/lloyd/projects/ktann/.benchmark-data/results/search-cpu-20261004-31d4`.
Baseline SHA-256: `b6ab4966fc499bf3f7264617efcc04de86b10920c2d1985f4fa75d176b8900c3`.
Candidate SHA-256: `6d92ca21643cd1e8ae6e9e0ad32cc6c9dcd18e7359504ab3ff6842264faf0219`.

## Results

| Metric / concurrency | QPS before → after | CPU µs/query before → after | p50 ms before → after | p99 ms before → after |
| --- | ---: | ---: | ---: | ---: |
| Cosine / 1 | 504 → 529 | 1983.0 → 1891.1 | 1.964 → 1.872 | 2.181 → 2.094 |
| Cosine / 4 | 1905 → 1989 | 2089.2 → 2002.5 | 2.066 → 1.982 | 2.366 → 2.255 |
| Cosine / 16 | 2864 → 3042 | 2506.9 → 2405.2 | 2.180 → 2.090 | 20.726 → 18.657 |
| L2 / 4 | 5125 → 5204 | 777.8 → 766.0 | 0.754 → 0.744 | 1.008 → 0.960 |
| L2 / 16 | 7616 → 7645 | 959.0 → 957.3 | 0.847 → 0.857 | 9.062 → 8.603 |

Cosine QPS improves 4.5–6.2%, CPU time/query falls 4.1–4.6%, and p50 falls
about 4–5% in the final runs. At concurrency 16, the final candidate occupies
7.32 CPU cores on average. This demonstrates a gain for this resident Cosine
workload.

Low-dimensional L2 has much less repeated norm work to remove. Its concurrency-4
CPU reduction is approximately 1.5%; concurrency-16 differences are within
noise. Tail latency varies with scheduling: longer 48000-query runs of the
scoring change had overlapping baseline/candidate p99 ranges. No general L2
tail-latency improvement is established. Increasing concurrency improves
throughput but also exposes queuing; these results are not a latency SLA.

## Correctness and limits

Focused library, search, codec, data-driven API, and ground-truth tests pass;
formatting and `cargo clippy -p ktann --all-targets --all-features -- -D warnings`
pass. Independent review confirmed the manifest-bound decoder enforces record
dimension and finite components before the scorer runs. Numeric oracle coverage
and added Vector Record NaN/Infinity corruption cases protect that contract.

This is a CPU-isolation fixture, not a million-vector quality curve. It does not
measure recall independently, memory bandwidth, or production adapter gains.
KTANN's Partition Cache contains decoded Leaf/Child Entries, not original
Vector Records. Reranking still reads record bodies and locations from the
backend snapshot, even with a warm Partition Cache. FoundationDB retains RPC
cost; its documented 512 MiB server cache alone cannot hold roughly 3 GiB of
Cohere original vectors. RocksDB separately bounds its native transaction pool
by available parallelism by default. Production 1M measurements must include
backend cache residency, admission/RPC waits, process and server CPU, recall,
and IO counters before changing cache policy or concurrency.
