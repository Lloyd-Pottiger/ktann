# ANN quality findings

## Scope

This investigation targets issue #126 and the observed Cohere 1M cosine
quality regression. The working target is recall@10 above 95% with
`leaf_beam_size = 8`.

## Reproduced measurements

The large RocksDB benchmark uses one tree, three searchable levels, and the
same 1M-vector Cohere and SIFT datasets. The default exact-rerank limit is 64.

| dataset / metric (before centroid fix) | beam 1 | beam 4 | beam 8 | beam 16 | beam 32 |
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

## Committed correction and validation

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

## Write-time beam

Foreground inserts and upserts now accept a RuntimeConfig write beam through
the same routing path used by Import Session. The production default is eight;
`with_write_beam_size` remains available for controlled quality/cost tuning. A
wider beam does not duplicate membership: every mutation still validates and
commits one Record Location and one Leaf Entry.

The write beam is global within each tree level, matching the search
traversal's semantics. For example, if four candidate parents expose eight
children, write beam four retains the four nearest children for that vector
across all four parents; it does not retain four under every parent. A focused
three-level topology test catches this distinction. This also prevents the
effective work from growing as beam squared at the next level.

A full Cohere 1M run with `max_partition_entries=512` and write beam four
completed successfully on 2026-09-02. It produced 2,774 partitions and the
following search curve:

| search beam | recall@10 | mean visited leaf entries |
| ---: | ---: | ---: |
| 1 | 27.10% | 370 |
| 4 | 52.32% | 1,478 |
| 8 | 65.81% | 2,954 |
| 16 | 76.51% | 5,906 |
| 32 | 84.74% | 11,809 |

The comparable earlier write-beam-one run reached 64.28% at search beam 8
and 84.01% at search beam 32. This is a directional improvement, not a
controlled same-process comparison: the imported topology had 2,774 versus
about 2,762 partitions, and the write-beam-four import took 1,236 seconds.
The target of greater than 95% recall at search beam 8 is therefore still not
met. Write beam four remains a useful lower-cost comparison, while the Runtime
default is now write beam eight. The full-run report is
`.benchmark-data/results/diagnose-cohere-1m-write-beam4-2026-09-02.json`.

A follow-up full Cohere 1M run with `max_partition_entries=128` and write beam
eight completed on the same host. It produced 11,147 partitions; the import
took 1,566 seconds and the search curve was:

| search beam | recall@10 | mean visited leaf entries |
| ---: | ---: | ---: |
| 16 | 66.53% | 1,482 |
| 32 | 76.22% | 2,960 |
| 54 | 82.39% | 4,996 |
| 128 | 90.50% | 11,835 |

Even search beam 128 remains below 95%, so increasing write beam alone does
not explain or resolve the remaining quality loss. The run report is
`.benchmark-data/results/cohere-1m-max128-write-beam8-search-16-32-54-128-2026-09-02.json`.

Given the quality-first goal, the production defaults are now write beam 8,
leaf search beam 128, and maximum partition size 128. This deliberately trades
import and query cost for a stronger routing baseline; it is not a claim that
the current index reaches the 95% target. Import throughput is tracked
separately because the write beam makes the import path materially more costly.

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

The committed cosine normalization fix and the write-time beam are useful, but
these experiments do not justify closing #126: the beam-8 Cohere target
remains unmet. The next high-value investigation is the quality of immutable
internal routing centroids over the online split history, ideally with
per-level routing diagnostics or a descendant-aggregate experiment.

### Leaf capacity and equal-work comparison

The post-normalization 1M baseline remains far below the target. On a fresh
`max_partition_entries=512` tree with 2,753 leaves, eight level-2 partitions,
and one root, the Cohere curve was:

| beam | recall@10 | mean visited leaf entries |
| ---: | ---: | ---: |
| 8 | 64.28% | 2,956 |
| 32 | 84.01% | 11,846 |
| 64 | 90.56% | 23,740 |
| 128 | 95.16% | 47,528 |

The fair `max=128` run produced 11,025 leaves, 122 level-2 partitions, and one
root. Its beam=8 point reached 50.95% at 738 entries; beam=32 reached 71.74%
at 2,953 entries. The measured report is
`.benchmark-data/results/diagnose-cohere-1m-max128-beam8-32-2026-09-01.json`.

Several structural/training candidates did not solve the problem:

- Five deterministic balanced-K-means seed pairs, selecting the lowest
  balanced objective, reached 65.15% at beam 8. Import took about 833 seconds,
  versus about 798 seconds for the single-seed run. The small gain does not
  justify the extra training complexity or meet the target, so the candidate
  was removed.
- Reducing the leaf capacity to 128 produced about 11,000 leaves. The earlier
  `max=128, beam=8` point reached 51.79% while visiting only 737 leaf entries;
  that is a lower-work point and is not a fair comparison with `max=512,
  beam=8`. A fresh equal-work sweep measured `max=128, beam=32` at 71.74%
  recall and 2,953 visited leaf entries, versus 64.28% and 2,956 entries for
  `max=512, beam=8`. Smaller leaves therefore improve recall at matched leaf
  work, but they do not approach the 95% target and their import took about
  964 seconds.
The earlier bounded query-window reports that showed 100% recall were invalid:
the runner discarded the base vectors before computing the local exact truth,
so an empty truth set was treated as perfect recall. The runner now computes
truth before releasing the imported vectors. Those old reports must not be
used as quality evidence.

A corrected paired 100k Cohere run (`max_partition_entries=64`, query rows
900--999) measured write beam one versus four:

| write beam | search beam 8 recall@10 | mean visited leaf entries | import seconds |
| ---: | ---: | ---: | ---: |
| 1 | 47.60% | 364 | 49.1 |
| 4 | 52.20% | 371 | 55.5 |

The write beam four run improved this slice by 4.6 percentage points at search
beam 8, with about 13% more import time and nearly the same search work. This
supports keeping the option available, but the slice is not a replacement for
the full 1M quality result.

The remaining high-value direction is a better persistent routing model for
the online tree—likely a bulk-build or explicitly maintained descendant
aggregate—not a larger rerank candidate set. Smaller leaves remain a useful
quality/work trade-off, but neither leaf-capacity changes nor write beam four
close the beam-8 quality gap.
