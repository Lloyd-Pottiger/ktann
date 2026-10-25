# Adaptive write routing: validation findings

Graph-enabled indexes route writes exactly below **1024 nodes per internal
body** and use local HNSW at or above that threshold. The decision uses each
body's snapshot Header, not the index record count. Query behavior and budgets
are unchanged. See [ADR 0031](../docs/adr/0031-adaptive-local-graph-write-routing.md).

## Cohere validation

These are historical RocksDB measurements from 2026-10-25, collected before
removing the temporary experiment runners and instrumentation. The measured
implementation is pinned by `environment.json` in the local artifact directory
`.benchmark-data/results/write-routing-default-1024-20261025/`; commit
`0e6c18df2b22774cd909196a2291127277cd11d0` retains the validation sources.
The exact control used the pinned binary from
`.benchmark-data/results/write-routing-threshold-20261025/` with experimental
flags unset. Cleanup has not been remeasured on Cohere 1M.

Every comparison cloned the same mature topology and used the same batches:
Cohere 768-dimensional cosine, M16/efConstruction200/seed7, leaf bounds 16:128,
internal bounds 64:8192, two refinement passes, batch32, write beam4/ef64 and
two maintenance workers. Deterministic vector swaps preserve the dataset
multiset. Total time includes foreground writes and maintenance drain; Verify
and query evaluation are outside timing. Builds and tests did not overlap
measurements. The 1M topology starts with a three-child root and three large
internal bodies, the largest containing 5,150 nodes.

| Fixture | Internal nodes | Updates | Exact total | Adaptive total | Change |
|---|---:|---:|---:|---:|---:|
| 16K warm | 199 | 4 × 8,192 | 4.139s | 4.257s | +2.9% |
| 16K cold | 199 | 4 × 8,192 | 4.169s | 4.162s | −0.2% |
| 64K cold | 834 | 2 × 8,192 | 4.416s | 4.312s | −2.4% |
| 128K cold | 1,652 | 2 × 8,192 | 7.098s | 3.886s | −45.3% |
| 1M cold start | up to 5,150 | 8 × 32,768 | 651.351s | 193.196s | −70.3% |

Small/medium results are medians of three alternating repetitions per arm;
1M is one fresh eight-round pair. Small exact-only differences should not be
interpreted as an algorithmic speedup. Earlier threshold trials found HNSW
regressed at 413 nodes and improved at 793–834 nodes; 1024 is a conservative
cutover. The 1M first cold round improved from 83.043s to 25.114s.

For 1M, foreground time fell from 649.032s to 187.357s, CPU from 725.248s to
267.622s, and logical reads from 101.569GB to 55.992GB. Maintenance drain rose
from 2.319s to 5.839s and is included in total time. Small-root routes stayed
exact while large-body routes used HNSW.

## Write quality and recall

All 26 completed validation runs passed Verify after every round and ended
without pending transitional states. The 1M pair ended with 13,409 versus
13,407 leaves, three internal parents and one root; both respected the maximum
leaf size of 128. Maximum parent sizes were 5,264 versus 5,262.

Query evaluation was fixed at beam 128 / ef 64, rerank 125, k 100, query rows 100–199,
distance budget 65,536 and candidate budget 16,384. Initial recall was 82.60%;
final recall was **82.63% for both arms**. Intermediate adaptive-minus-exact
mean differences were 0.00–0.04 percentage points. A paired query bootstrap
(seed7, 3,000 resamples) gave a final 95% interval of [−0.05, +0.05] percentage
points. This supports write-quality parity for the measured workload, not a
universal recall or long-term drift guarantee.

Earlier 90%+ results used different operating points: the October 17 record
used leaf512/beam128 (92.422%, 500 queries); the October 19 record used
leaf128/beam435/ef435 (91.01%, a different query sample). They are not directly
comparable to this leaf128/beam128/ef64 write-routing control. Rerank remains125.

## Why the retained optimizations matter

Write search reuses workspace and scores two to four neighbors together while
preserving adjacency consumption and candidate ordering. Queries retain their
scalar path. Graph preparation supplies both persistence records and the native
graph; known-success commits can publish that graph into the existing bounded
cache, avoiding immediate rereads and reconstruction.

Structural maintenance discovers the incoming parent exactly after acquiring
sorted body guards and refreshing its snapshot. A captured parent proof avoids
a second full discovery during commit, while the commit rechecks the complete
parent set, update-protects its Headers and protects the actual incoming entry.
In the October 24 three-run mechanism comparison, proof reuse reduced total
write-plus-drain time from 19.777s to 18.406s, CPU from 37.749s to 27.292s and
logical reads from 24.301GB to 13.118GB; foreground time remained about 5.1s.

Cancellation, memory admission, cache accounting, source identity/epoch checks
and commit fences remain production requirements. There is no second graph
cache, reverse pointer, persistent-format change or runtime experiment switch.

## Limits

These observations cover a stationary multiset, local RocksDB and two
maintenance workers. They do not establish performance under distribution
drift, unbounded growth, more writers/workers, multiple processes or
FoundationDB. Historical artifacts and source commits preserve the experiment
provenance; temporary runners, forced-miss switches and phase counters are not
part of the maintained implementation.
