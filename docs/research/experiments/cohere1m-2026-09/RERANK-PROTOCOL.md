# Rerank budget experiment

2026-09-26. Fixed Cohere1M index and logical SHA256 from baseline; k=100,
beam=384, 1 GiB partition cache, RocksDB. No algorithm or production default change.
Independent variable: runtime exact rerank budget 150/125/100, through existing API.
All other runtime search budgets preserved. Runtime changes may also affect upstream
candidate caps; stage attribution will account for this, not just final exact scoring.

Run serial diagnostic controls at all three budgets, one thousand distinct queries
in the same order; compare per-query truth hits and six stages. Diagnostic timings
are ineligible. Each diagnostic query validates collector-disabled vs enabled parity.
Then run normal timing binary 150/125/100/100/125/150; 1000 warmup,
2000 measured operations (1000 distinct queries repeated), four concurrent searches.
No concurrent builds or workloads. Report both repetitions, QPS, p95/p99, CPU/query,
logical read bytes/query and exact candidate count. Do not equate rerank work reduction
with whole-query speedup. A candidate is acceptable at this operating point only
if recall is preserved; report query regressions even when average recall is unchanged.
No generalization to other datasets, k, filters, backends or tree shapes.

Verify full index and logical contents before/after reuse. Preserve executable,
source patch, invocation, environment and result hashes. All artifacts are isolated;
retain production defaults until broader evidence justifies changing them.
