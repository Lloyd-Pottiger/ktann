# Current Cohere1M k=100 baseline and attribution

Base revision: 3a62957c458a87b5b708af58ea1975c679e54ac2.
Isolated worktree: <experiment-worktree>.
No production changes or optimization candidates in this experiment.

Use the persistent supplied Cohere1M dataset and its 1,000 distinct held-out queries,
k=100 cosine, RocksDB, eight executor workers, four concurrent timed queries,
512 MiB partition cache, write beam eight, leaf threshold128, import batch50,
import in-flight ceiling4, backlog watermark2, two maintenance workers.
Native persistent Import Session is intentionally different from the older bridge's
per-batch sessions and serial input. Do not report direct speedup against that run.

Search budgets are 1 Tree Key, 1,024 partitions, 65,536 Leaf Entries, and the
unchanged k-derived rerank ceiling150. The Tree Key limit differs from default4096
but this workload has exactly one tree. Beam64/96/128/192/256/384 is the only search
quality variable. Warmup1,000 and measured2,000 requests per point for timing;
repeat the normal reuse sweep twice. Repetition is not extra recall samples.

First run small10k/100-query data with max partition32 and beams1/8/32 to prove
three-level traversal and collector parity against recomputed subset ground truth.
Then build a new1M fixture and collect import plus full convergence/verification.
Fresh-build search timings are preparation only: the first logical digest is
recorded after that sweep. Accepted timing sweeps and attribution must REUSE the
fixture, check complete integrity and dataset identity, and match logical KV hashes
before and after each run. Original construction values carried into reuse reports
describe the original build, not a new import.

Timing binary excludes diagnostic hooks through cfg. Diagnostic binary executes
one query at a time, with a collector-disabled control immediately before the
collector-enabled search. IDs, exact distance bits, all usage/exhaustion values,
and overlap truncation must match. Collect nested ground-truth membership at
visited/local interval/local cap/global interval/global cap/final output; assert
stage counts agree with normal usage. Query-by-query losses sum to recall deficit.
Reference-neighbor membership in visited entries gives the traversal recall ceiling
without reloading/scoring every visited full vector. This is an upper bound under
the supplied ground truth, not an alternative measured search implementation.
No instrumented duration is admissible performance evidence.

Keep source patch, collector source, lockfile, executable hashes, commands and env,
run-level host process snapshots, reports and logs. Do not build or run another
benchmark during timed windows. Full million-vector data and fixtures remain in
the main repository's ignored .benchmark-data directory.

Independent review: no actionable findings for reuse-based timing and diagnostics.
It identified the fresh-build hashing limitation, addressed by excluding that
search timing and using subsequent reuse-only baseline sweeps.

Small-screen correction: the first diagnostic run failed report validation because
normal engine metrics include both control and traced searches. The diagnostic
validator now expects twice as many budget samples while keeping one recall sample
per request. The repeated small run passed all300 per-query equivalence and loss
checks and preserved the logical digest. Failed v1 logs, patch and binaries remain
for provenance. This change is outside timed search and does not change production
selection. Core tests:224 passed/1 ignored; final benchmark tests:38 passed.

Before the second timing run, fix its beam order with Python Random(20260926):
96,192,128,256,384,64. Compare by beam key, not point position. This varies temporal
ordering without changing queries, warmup, concurrency, budgets or index contents.
