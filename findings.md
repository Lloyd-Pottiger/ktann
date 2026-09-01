# ANN quality findings

## Scope

This investigation targets issue #126 and the observed Cohere 1M cosine
quality regression. The working target is recall@10 above 95% with
`leaf_beam_size = 8`.

## Reproduced measurements

The large RocksDB benchmark uses one tree, three searchable levels, and the
same 1M-vector Cohere and SIFT datasets. The default exact-rerank limit is 64.

| dataset / metric | beam 1 | beam 4 | beam 8 | beam 16 | beam 32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SIFT1M / L2 recall@10 | 21.39% | 50.10% | 66.07% | 80.16% | 90.61% |
| Cohere1M / cosine recall@10 | 6.76% | 19.41% | 30.12% | 42.55% | 56.21% |

The SIFT run visits about 11,868 leaf entries at beam 32; Cohere visits about
10,310. The search and leaf-entry budgets are not exhausted, so the low
Cohere score is not explained by the configured traversal work limit.

Increasing the exact-rerank ceiling to 1,000 did not improve Cohere:

| beam | recall@10 | mean rerank candidates |
| ---: | ---: | ---: |
| 1 | 6.90% | 127.9 |
| 4 | 16.88% | 242.2 |
| 16 | 39.68% | 405.0 |
| 32 | 54.17% | 483.5 |

This confirms that the dominant loss happens before exact reranking.

## Root cause found

Cosine ingest preprocessing normalizes each vector before applying the
persistent rotation. Split training then computes each target centroid as the
arithmetic mean of its normalized routing vectors. Such a mean is generally
not unit-norm.

The shared routing implementation nevertheless compared a query with the raw
centroid using `-dot(query, centroid)`. This gives a higher-score advantage to
the centroid with the larger norm, even when its direction is worse. It is
not cosine distance between the query and the split centroid and can distort
both split draining and search descent. The same routing helper is also used
by merge redirects and other maintenance decisions.

An earlier local experiment normalized cosine centroids before routing. With
the default rerank=64 it raised Cohere recall from 56.21% to 83.70% at beam 32
(beam 16 rose from 42.55% to 74.89%). The committed spherical-training
implementation reproduced this improvement in a fresh run: 63.75% at beam 8
and 84.41% at beam 32 with rerank=64.

The post-fix rerank=1000 run reached 64.35% at beam 8 and 83.80% at beam 32.
The small change confirms that reranking is not the remaining dominant loss;
the beam=8 target is still unmet and traversal/tree coverage needs a separate
investigation.

## Planned correction and validation

Cosine split centroids are now normalized in f64 after every trained cluster
mean and before persistence. The query and leaf records are already normalized
by the preprocessing invariant, so routing remains a single dot product. This
avoids a second query norm calculation in the split-training hot loop and
makes the unit-centroid invariant explicit. A zero centroid remains zero so
degenerate split states remain deterministic and searchable.

Validation completed:

1. numeric routing tests for scale invariance and opposite directions;
2. split-training tests proving cosine centroids are used with the corrected
   routing semantics;
3. the existing test suite;
4. a fresh Cohere beam sweep, including beam 8 and a high rerank ceiling, to
   separate routing quality from rerank truncation.

The benchmark runner keeps beam 8 in the large-profile sweep. The rerank=1000
ceiling was a temporary diagnostic-only API change and has been restored to
the normal k-derived ceiling of 64.

## Follow-up experiments

### Traversal allocation is not the whole remaining loss

With the normalized centroids, the default global level beam reached about
63.75% recall at beam 8 and 83.70% at beam 32. Removing the level scaling and
using the same beam at every level changed those results only to about 64.80%
and 83.66%.

A per-parent beam experiment did more work: beam 8 visited about 11,932 leaf
entries and reached 73.95%, while beam 32 visited about 95,852 leaf entries
and reached 92.50%. The extra work improved quality, but beam 8 still missed
the target and the result was worse than the global beam 32 run at a similar
leaf-entry count. The global competition is therefore a quality trade-off,
not the root cause of the remaining failure.

### Weighted internal centroids were not a demonstrated fix

One candidate loaded each Child Entry's child Header and weighted its
centroid by `entry_count` during internal split training. This is only an
approximation: an internal Header counts direct Child Entries, not descendant
Vector Records. A fresh 1M Cohere import with this candidate produced 63.59%
at beam 8 and 83.77% at beam 32, indistinguishable from the normalized
baseline, while adding Header reads to training. The candidate was removed.

### InnerProduct does not have the same cosine normalization bug

InnerProduct intentionally preserves vector scale during preprocessing,
trains from unnormalized centroids, and ranks by `-dot`; exact reranking uses
the same definition. The numeric routing tests cover this contract. A
clustered 5K InnerProduct experiment reached 100% recall at beam 32. A
skewed InnerProduct experiment reached only 34.10%, while the same skewed
L2 experiment reached 93.70%. This is evidence of a broader centroid/tree
model limitation for skewed MIPS data, not evidence that InnerProduct
centroids should be normalized. Normalizing them would change the metric.

The committed cosine normalization fix is useful, but these experiments do
not justify closing #126: the beam-8 Cohere target remains unmet. The next
high-value investigation is the quality of immutable internal routing
centroids over the online split history, ideally with per-level routing
diagnostics or a descendant-aggregate experiment before changing the public
beam semantics.

## Follow-up experiments

The post-normalization 1M baseline remains far below the target. On a fresh
`max_partition_entries=512` tree with 2,753 leaves, eight level-2 partitions,
and one root, the Cohere curve was:

| beam | recall@10 | mean visited leaf entries |
| ---: | ---: | ---: |
| 8 | 64.28% | 2,956 |
| 32 | 84.01% | 11,846 |
| 64 | 90.56% | 23,740 |
| 128 | 95.16% | 47,528 |

Several structural/training candidates did not solve the problem:

- Five deterministic balanced-K-means seed pairs, selecting the lowest
  balanced objective, reached 65.15% at beam 8. Import took about 833 seconds,
  versus about 798 seconds for the single-seed run. The small gain does not
  justify the extra training complexity or meet the target, so the candidate
  was removed.
- Reducing the leaf capacity to 128 produced 11,075 leaves and reached only
  51.79% at beam 8 while visiting 737 leaf entries. Its import took about 848
  seconds, so smaller leaves are not a quality fix for this tree model.
- On a valid difficult 100k slice (`max_partition_entries=64`, query rows
  900--999), balanced training reached 0.1% at beam 8. Five seed pairs reached
  0.0%, and unbalanced nearest-cluster assignment reached 0.1%. These results
  reinforce that neither seed restarts nor removing the 50/50 constraint is the
  missing invariant.

Some early reduced-dataset runs incorrectly reported 100% because the temporary
runner cleared the base vectors before recomputing local brute-force truth. The
runner was corrected before the results above were accepted; those earlier
numbers are discarded and are not evidence about KTANN quality.

The remaining high-value direction is a better persistent routing model for
the online tree—likely a bulk-build or explicitly maintained descendant
aggregate—not a larger rerank candidate set, smaller leaves, or more local
K-means seeds. No additional production implementation change is justified by
the follow-up measurements yet.
