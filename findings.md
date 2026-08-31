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
