# Use one-probe Bloom sizing for an explicit false-positive bound

MinMaxBloom uses one XXH3 probe and derives `m = ceil(n / p)` persisted bits
from expected distinct count `n` and target false-positive rate `p`. At the
configured cardinality, at most `n` bits can be occupied, so a uniformly hashed
absent value has false-positive probability at most `n / m <= p`. This direct
bound avoids the repeated-probe behavior that makes the usual independent-probe
approximation unsafe for small double-hashed Bloom summaries.

The cost is a larger Synopsis than an optimally sized multi-probe Bloom filter.
Logical Index creation rejects configurations whose complete encoded Synopsis
would exceed 64 KiB instead of weakening the requested rate. Inserting more
than the expected number of distinct values may saturate the Bloom summary and
weaken pruning, but remains conservative; it never becomes a correctness or
split condition. The bit count and single hash count are persisted v1 format
parameters and must be reproduced canonically when reopening the Index
Manifest.
