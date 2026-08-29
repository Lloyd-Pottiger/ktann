# Import admission calibration

This calibration selects backend-neutral Import Session and Structure
Maintenance defaults from complete whole-lifecycle behavior. It does not treat
submitted throughput, queue depth, or retry count alone as success.

## Method

Each cell ran the release `full` `import-to-search-lifecycle` SIFTsmall scenario
three times on the same host on 2026-08-29. Runs used fresh Backend Namespaces,
10,000 base vectors, 100 queries, 128 dimensions, `k=10`, a 128-entry Split
Threshold, and two convergence workers. The second repetition reversed cell
order to reduce systematic host drift. The same matrix ran on RocksDB and the
documented local FoundationDB 7.3 installation.

Every process exited successfully, accepted 10,000 of 10,000 records, reported
zero batch failures, verified a converged topology, shut the Runtime down
cleanly, and reported zero Search Budget exhaustion. A cell still fails
calibration when any immediate, stable-cold, or stable-warm recall differs from
the fixed-concurrency baseline. No cell produced `ContentionExhausted`, partial
import, timeout, or non-convergence.

Values below are the three-run mean followed by coefficient of variation in
parentheses. Recall tables show the range of per-run stable-warm means and the
lowest per-query recall observed in any search phase.

## Matrix

| ID | Batch records | Concurrency ceiling | Maintenance workers | Backlog watermark | Purpose |
|---|---:|---:|---:|---:|---|
| A | 25 | 1 | 2 | 2 | small batch, serial ceiling |
| B | 25 | 4 | 2 | 2 | small batch, adaptive ceiling |
| C | 50 | 1 | 2 | 2 | selected batch, serial ceiling |
| D | 50 | 2 | 2 | 2 | selected batch, intermediate ceiling |
| E | 50 | 4 | 2 | 2 | selected defaults |
| F | 100 | 1 | 2 | 2 | large batch, serial ceiling |
| G | 100 | 4 | 2 | 2 | large batch, adaptive ceiling |
| H | 50 | 4 | 1 | 1 | one worker, strict backlog gate |
| I | 50 | 4 | 1 | 2 | one worker, coexistence gate |
| J | 50 | 4 | 2 | 1 | two workers, strict backlog gate |

## Correctness and search quality

| ID | Backend | Stable recall mean range | Recall floor | Partitions mean [range] | Result |
|---|---|---:|---:|---:|---|
| A | FoundationDB | 1.000–1.000 | 1.000 | 434.0 [434–434] | pass |
| A | RocksDB | 0.997–1.000 | 0.900 | 455.3 [434–498] | failed recall |
| B | FoundationDB | 1.000–1.000 | 1.000 | 434.0 [434–434] | pass |
| B | RocksDB | 0.997–0.997 | 0.900 | 494.0 [490–498] | failed recall |
| C | FoundationDB | 1.000–1.000 | 1.000 | 442.0 [442–442] | pass |
| C | RocksDB | 1.000–1.000 | 1.000 | 442.0 [442–442] | pass |
| D | FoundationDB | 1.000–1.000 | 1.000 | 442.0 [442–442] | pass |
| D | RocksDB | 1.000–1.000 | 1.000 | 444.7 [442–450] | pass |
| E | FoundationDB | 1.000–1.000 | 1.000 | 442.0 [442–442] | pass |
| E | RocksDB | 1.000–1.000 | 1.000 | 442.0 [442–442] | pass |
| F | FoundationDB | 0.999–0.999 | 0.900 | 454.0 [454–454] | failed recall |
| F | RocksDB | 0.999–0.999 | 0.900 | 454.0 [454–454] | failed recall |
| G | FoundationDB | 0.999–0.999 | 0.900 | 454.0 [454–454] | failed recall |
| G | RocksDB | 0.999–0.999 | 0.900 | 454.0 [454–454] | failed recall |
| H | FoundationDB | 0.999–0.999 | 0.900 | 439.3 [434–442] | failed recall |
| H | RocksDB | 0.999–0.999 | 0.900 | 439.3 [434–442] | failed recall |
| I | FoundationDB | 1.000–1.000 | 1.000 | 442.0 [442–442] | pass |
| I | RocksDB | 1.000–1.000 | 1.000 | 442.0 [442–442] | pass |
| J | FoundationDB | 0.999–0.999 | 0.900 | 436.7 [434–442] | failed recall |
| J | RocksDB | 0.999–0.999 | 0.900 | 439.3 [434–442] | failed recall |

Cells A, B, F, G, H, and J are failed configurations, not performance
candidates. Their resource measurements remain below to show why faster import
or fewer retries cannot override the recall contract.

## Lifecycle and latency

### RocksDB

| ID | Case wall s | Case CPU s | Import wall s | Import CPU s | Submit p95 ms | Convergence s |
|---|---:|---:|---:|---:|---:|---:|
| A | 5.336 (29.8%) | 9.366 (61.4%) | 1.768 (35.2%) | 2.602 (42.1%) | 13.0 (53.6%) | 2.465 (88.7%) |
| B | 7.060 (4.0%) | 16.557 (4.3%) | 0.841 (3.2%) | 1.026 (2.6%) | 2.8 (5.0%) | 5.232 (5.0%) |
| C | 4.741 (3.7%) | 6.125 (3.8%) | 2.689 (6.1%) | 4.147 (5.4%) | 31.3 (10.4%) | 0.931 (0.4%) |
| D | 4.622 (5.7%) | 5.982 (5.2%) | 2.573 (8.8%) | 4.003 (7.1%) | 31.2 (14.8%) | 0.935 (1.5%) |
| E | 4.339 (4.8%) | 5.572 (5.0%) | 2.374 (8.4%) | 3.685 (7.1%) | 28.9 (17.1%) | 0.903 (1.8%) |
| F | 4.447 (9.1%) | 5.810 (9.7%) | 2.416 (12.5%) | 3.752 (11.6%) | 48.8 (14.7%) | 0.935 (4.4%) |
| G | 4.610 (5.8%) | 6.009 (6.1%) | 2.544 (7.7%) | 3.913 (7.2%) | 54.5 (6.6%) | 0.955 (3.2%) |
| H | 4.417 (0.6%) | 4.737 (0.7%) | 2.363 (0.3%) | 2.746 (0.3%) | 31.5 (6.3%) | 0.944 (1.1%) |
| I | 4.924 (10.4%) | 6.114 (10.6%) | 2.821 (14.4%) | 4.087 (13.2%) | 31.7 (16.3%) | 0.954 (4.0%) |
| J | 4.110 (4.8%) | 4.682 (5.3%) | 2.094 (7.2%) | 2.726 (7.1%) | 23.0 (11.5%) | 0.919 (2.1%) |

### FoundationDB

| ID | Case wall s | Case CPU s | Import wall s | Import CPU s | Submit p95 ms | Convergence s |
|---|---:|---:|---:|---:|---:|---:|
| A | 23.482 (0.5%) | 7.980 (0.9%) | 18.441 (0.6%) | 5.453 (1.4%) | 140.6 (0.9%) | 2.068 (1.9%) |
| B | 23.652 (2.3%) | 8.139 (3.3%) | 18.422 (1.4%) | 5.522 (3.0%) | 141.0 (1.1%) | 2.119 (7.7%) |
| C | 23.536 (1.8%) | 8.562 (3.5%) | 18.585 (2.3%) | 6.073 (5.1%) | 199.6 (1.7%) | 2.011 (1.1%) |
| D | 23.429 (0.3%) | 8.427 (0.7%) | 18.465 (0.4%) | 5.939 (0.9%) | 197.3 (2.3%) | 2.007 (0.8%) |
| E | 23.325 (0.2%) | 8.338 (0.3%) | 18.379 (0.3%) | 5.862 (0.4%) | 197.6 (0.6%) | 1.983 (0.6%) |
| F | 22.429 (0.2%) | 8.067 (0.1%) | 17.481 (0.1%) | 5.524 (0.1%) | 355.7 (1.6%) | 1.993 (2.2%) |
| G | 22.419 (0.6%) | 8.051 (0.9%) | 17.472 (0.6%) | 5.515 (1.3%) | 356.0 (0.4%) | 1.997 (2.1%) |
| H | 23.879 (0.1%) | 6.771 (0.2%) | 18.932 (0.3%) | 4.264 (0.7%) | 245.6 (1.1%) | 1.992 (1.0%) |
| I | 25.129 (0.4%) | 8.743 (0.2%) | 20.149 (0.3%) | 6.260 (0.3%) | 198.0 (0.8%) | 2.054 (2.0%) |
| J | 22.507 (0.5%) | 6.715 (0.3%) | 17.528 (0.8%) | 4.219 (0.4%) | 172.2 (1.7%) | 2.026 (2.8%) |

## Logical Backend IO and Fixup activity

### RocksDB

| ID | Commits | Retries | Read tx | Write tx | Point reads | Scans | Items read | Read MiB | Mutations | Mutation MiB | Fixup steps | Drained entries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 1115.0 (45.0%) | 122.0 (57.6%) | 929.7 (70.2%) | 1237.0 (46.2%) | 129795.0 (49.5%) | 1147.3 (70.5%) | 131678.3 (66.2%) | 38.6 (68.2%) | 82407.7 (34.3%) | 13.0 (25.3%) | 571.7 (70.3%) | 9280.7 (70.5%) |
| B | 406.7 (0.1%) | 28.3 (6.7%) | 10.3 (9.1%) | 435.0 (0.5%) | 39553.0 (0.7%) | 4.7 (10.1%) | 8415.0 (2.3%) | 1.4 (3.4%) | 43099.7 (0.5%) | 8.5 (0.3%) | 4.7 (10.1%) | 85.3 (17.7%) |
| C | 1290.0 (0.0%) | 175.0 (3.6%) | 1417.0 (0.0%) | 1465.0 (0.4%) | 186853.7 (0.8%) | 1468.0 (0.6%) | 183173.0 (0.6%) | 52.0 (0.5%) | 118864.7 (1.0%) | 18.6 (1.2%) | 872.0 (0.0%) | 14224.0 (0.0%) |
| D | 1293.3 (0.4%) | 166.7 (6.1%) | 1421.7 (0.5%) | 1460.0 (0.4%) | 184273.7 (1.9%) | 1460.3 (1.0%) | 182106.3 (1.1%) | 51.8 (0.8%) | 116885.3 (2.1%) | 18.1 (2.8%) | 874.7 (0.4%) | 14268.0 (0.4%) |
| E | 1290.0 (0.0%) | 165.3 (5.0%) | 1417.0 (0.0%) | 1455.3 (0.6%) | 184582.3 (1.3%) | 1455.7 (0.7%) | 181768.0 (0.9%) | 51.6 (0.8%) | 117192.3 (1.3%) | 18.2 (1.6%) | 872.0 (0.0%) | 14223.3 (0.0%) |
| F | 1181.0 (0.1%) | 123.0 (4.0%) | 1409.0 (0.1%) | 1304.7 (0.3%) | 175628.7 (0.9%) | 1231.3 (0.5%) | 161500.3 (0.6%) | 45.6 (0.6%) | 121683.0 (1.0%) | 19.1 (1.2%) | 865.0 (0.2%) | 14253.3 (0.1%) |
| G | 1180.3 (0.1%) | 126.0 (1.7%) | 1408.7 (0.1%) | 1307.3 (0.1%) | 177565.7 (0.3%) | 1236.0 (0.2%) | 162471.7 (0.2%) | 45.8 (0.3%) | 123107.7 (0.3%) | 19.4 (0.4%) | 864.3 (0.1%) | 14242.7 (0.2%) |
| H | 1276.7 (0.4%) | 5.7 (8.3%) | 1400.3 (0.5%) | 1282.3 (0.4%) | 143124.7 (0.2%) | 1254.7 (0.3%) | 154104.7 (0.3%) | 44.6 (0.3%) | 87958.7 (0.2%) | 12.4 (0.2%) | 861.3 (0.4%) | 14052.7 (0.5%) |
| I | 1290.0 (0.0%) | 173.3 (6.8%) | 1417.0 (0.0%) | 1463.3 (0.8%) | 191618.3 (1.8%) | 1491.7 (1.2%) | 186393.3 (1.3%) | 52.8 (1.2%) | 122071.7 (1.9%) | 19.2 (2.5%) | 872.0 (0.0%) | 14225.0 (0.0%) |
| J | 1276.7 (0.4%) | 27.7 (12.3%) | 1400.3 (0.5%) | 1304.3 (0.6%) | 143296.0 (0.2%) | 1257.3 (0.4%) | 154428.3 (0.3%) | 44.7 (0.4%) | 88105.3 (0.3%) | 12.4 (0.2%) | 861.3 (0.4%) | 14053.7 (0.5%) |

### FoundationDB

| ID | Commits | Retries | Read tx | Write tx | Point reads | Scans | Items read | Read MiB | Mutations | Mutation MiB | Fixup steps | Drained entries |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 1470.0 (0.0%) | 250.3 (2.8%) | 1391.0 (0.0%) | 1720.3 (0.4%) | 183885.7 (0.7%) | 1816.0 (0.7%) | 200625.3 (0.6%) | 59.5 (0.6%) | 107775.0 (0.7%) | 16.5 (1.0%) | 856.0 (0.0%) | 13905.3 (0.0%) |
| B | 1469.0 (0.1%) | 250.7 (0.7%) | 1389.7 (0.1%) | 1719.7 (0.2%) | 183833.0 (0.4%) | 1811.7 (0.6%) | 200606.0 (0.3%) | 59.5 (0.4%) | 107803.3 (0.3%) | 16.5 (0.4%) | 855.0 (0.2%) | 13894.3 (0.1%) |
| C | 1290.0 (0.0%) | 231.0 (0.9%) | 1417.0 (0.0%) | 1521.0 (0.1%) | 196339.3 (0.3%) | 1538.3 (0.2%) | 190994.7 (0.2%) | 54.3 (0.2%) | 124862.0 (0.3%) | 19.8 (0.3%) | 872.0 (0.0%) | 14224.0 (0.0%) |
| D | 1290.0 (0.0%) | 234.0 (0.9%) | 1417.0 (0.0%) | 1524.0 (0.1%) | 196955.7 (0.2%) | 1542.7 (0.2%) | 191430.7 (0.1%) | 54.5 (0.1%) | 125264.0 (0.2%) | 19.9 (0.3%) | 872.0 (0.0%) | 14223.0 (0.0%) |
| E | 1290.0 (0.0%) | 232.3 (0.9%) | 1417.0 (0.0%) | 1522.3 (0.1%) | 196915.7 (0.1%) | 1541.3 (0.2%) | 191484.7 (0.2%) | 54.5 (0.2%) | 125189.3 (0.1%) | 19.9 (0.1%) | 872.0 (0.0%) | 14223.0 (0.0%) |
| F | 1180.0 (0.0%) | 187.0 (1.9%) | 1408.0 (0.0%) | 1368.0 (0.3%) | 178574.7 (0.2%) | 1295.0 (0.3%) | 165899.0 (0.2%) | 47.5 (0.4%) | 123420.0 (0.2%) | 19.5 (0.2%) | 864.0 (0.0%) | 14242.0 (0.0%) |
| G | 1181.3 (0.1%) | 182.0 (2.8%) | 1409.3 (0.1%) | 1364.0 (0.3%) | 178122.0 (0.3%) | 1291.7 (0.4%) | 165558.0 (0.3%) | 47.3 (0.5%) | 123042.3 (0.3%) | 19.5 (0.4%) | 865.3 (0.1%) | 14264.0 (0.1%) |
| H | 1276.7 (0.4%) | 6.3 (7.4%) | 1399.7 (0.4%) | 1283.0 (0.3%) | 143249.7 (0.1%) | 1255.3 (0.3%) | 154130.3 (0.3%) | 44.6 (0.3%) | 87669.3 (0.2%) | 12.4 (0.0%) | 861.3 (0.4%) | 14053.7 (0.5%) |
| I | 1290.0 (0.0%) | 220.3 (0.2%) | 1417.0 (0.0%) | 1510.3 (0.0%) | 205256.7 (0.0%) | 1557.3 (0.0%) | 195564.7 (0.0%) | 55.2 (0.0%) | 131138.7 (0.1%) | 21.1 (0.1%) | 872.0 (0.0%) | 14225.0 (0.0%) |
| J | 1273.3 (0.4%) | 57.3 (6.4%) | 1395.3 (0.4%) | 1330.7 (0.6%) | 143606.3 (0.2%) | 1285.7 (0.4%) | 155902.0 (0.4%) | 45.5 (0.5%) | 87825.7 (0.2%) | 12.4 (0.0%) | 858.7 (0.4%) | 14005.3 (0.5%) |

## Selection

Only C, D, and E preserved the recall and topology contract on both adapters
within the batch-size and ceiling sweep. E had the lowest mean lifecycle wall,
import wall, and CPU cost of those valid cells on both adapters. Its ceiling of
four never forced four-way concurrency: every run increased once to two,
observed contention, and returned to one.

The maintenance matrix isolates the confounded signals. H and J made the queue
look quiet and reduced retries dramatically, but the strict watermark-one gate
changed stable topology and failed recall on both adapters. I preserved recall
with one worker but cost more than E. The engine therefore defaults to adaptive
ceiling four and backlog watermark two. The full lifecycle scenario retains the
validated 50-record batches and two maintenance workers as configuration
guidance; Runtime worker count remains host-scaled, and batch size remains
caller-controlled API input rather than an engine default derived from tree
size.
