# Filter Predicates, Tree Key Planning, and Partition Synopses

This module owns Filter Predicate semantics, conservative Tree Key range
planning, and Partition Synopsis classification. It describes the implemented
contract, not a proposed extension.

[Public API](api.md#3-schema-and-predicates) owns caller-visible types and
validation limits. [Search](search.md) owns snapshot acquisition, budgeted tree
enumeration, traversal, candidate selection, and exact reranking.
[Storage](storage.md) owns field and synopsis encodings;
[maintenance](maintenance.md) owns atomic membership changes.

## 1. Goals and invariants

Filtering must preserve exact eligibility while allowing approximate neighbor
selection. Its invariants are:

1. Every returned hit makes the complete Filter Predicate TRUE. FALSE and
   UNKNOWN never qualify.
2. Tree Key planning and synopsis pruning are conservative: neither may exclude
   a record that could make the predicate TRUE within the admitted search work.
3. Leaf Entry filter fields agree with the authoritative Vector Record. A
   mutation changes both projections atomically.
4. Predicate evaluation and pruning use the same typed comparison and NULL
   semantics, including arbitrary nesting of `Not`.
5. One search evaluates routing, synopses, and entries from the same backend
   snapshot. Malformed persisted values fail closed as `Corruption`.
6. Predicate compilation, range expansion, and tree enumeration are bounded.

Filtering does not guarantee global filtered top-k, a minimum hit count, or a
recall/latency SLA. The Rust API accepts a typed expression tree, not SQL text.
SQL parsing, field-to-field comparisons, LIKE, regex, full-text, array operators,
and predicates over Opaque Payload bytes are outside this contract.

## 2. Schema binding and validation

Fields belong to the immutable Index Configuration. `FieldId` is a zero-based
position in its ordered schema; Record field values use the same positions.
Field names describe the schema and are not runtime predicate lookup keys.
Every record supplies one value for every declared field. Fields are non-nullable
unless declared nullable; Tree Key fields must be non-nullable.

Predicates support `Compare`, `In`, `And`, `Or`, `Not`, `IsNull`, and `IsNotNull`.
Comparisons use `Eq`, `NotEq`, `Lt`, `LessOrEqual`, `Gt`, or `GreaterOrEqual`.
Ranges are conjunctions of comparisons; exclusion is negated membership.
Omitting the predicate imposes no filter.

Literal types must match the declared field type without implicit coercion.
Booleans order false before true; integers and finite floats use numeric order;
strings use raw UTF-8 byte order without normalization or case-folding. Negative
floating-point zero is canonicalized to positive zero. Comparison literals
cannot be NULL, and `In` lists cannot contain NULL.

`SearchRequest::with_predicate` attaches an expression. Schema-dependent
validation occurs during explicit request validation or search submission.
`CompiledPredicate::compile` validates the expression, binds referenced fields,
and compiles membership lists into sorted, deduplicated sets. The public
[API limits](api.md#3-schema-and-predicates) bound expression nodes, depth,
list length, and literal sizes; invalid caller expressions return
`InvalidArgument`. Stored field/type mismatches return `Corruption` rather than
being treated as predicate misses.

## 3. Three-valued evaluation

Evaluation follows SQL TRUE/FALSE/UNKNOWN semantics. Only TRUE qualifies.

| Expression | Non-NULL field | NULL field |
| --- | --- | --- |
| Comparison against a non-NULL literal | TRUE or FALSE | UNKNOWN |
| Nonempty `In` | TRUE or FALSE | UNKNOWN |
| `IsNull` | FALSE | TRUE |
| `IsNotNull` | TRUE | FALSE |

FALSE dominates `And`; TRUE dominates `Or`; otherwise UNKNOWN propagates.
`Not(TRUE)` is FALSE, `Not(FALSE)` is TRUE, and `Not(UNKNOWN)` is UNKNOWN.
`And([])` is TRUE; `Or([])` and empty `In` are FALSE. Empty `In` is FALSE even
for a NULL field, so its negation is TRUE.

For example, with nullable `score`, negating `score >= 0.8` excludes NULL scores.
Including them requires `IsNull(score) OR NOT(score >= 0.8)` (logical notation,
not a SQL parser interface). Compare-to-NULL is invalid; NULL checks must use
`IsNull` or `IsNotNull`.

## 4. Conservative Tree Key planning

Any declared field may participate in a predicate. Tree Key fields additionally
form the ordered memcomparable tuple that selects one tree for each record.
An empty Tree Key schema yields one tree and still supports filtering.

Equality and bounded `In` on leading Tree Key fields derive exact prefix ranges;
representable ranges on the next field narrow them. For `[tenant, region]`, a
tenant constraint can narrow a prefix, whereas a region-only condition generally
requires broader enumeration. Predicates not captured by these ranges remain
part of exact leaf evaluation.

Range expansion uses the Runtime's configured bound (default 1,024 ranges).
If disjoint expansion exceeds that bound, planning widens conservatively instead
of dropping alternatives. Search enumerates the resulting directory ranges in
canonical order under its independent Tree Key budget. Exhaustion can leave
matching trees unexplored and is reported by search; it is not a predicate miss.
Tree enumeration finishes before partition traversal, preserving deterministic
budget accounting and bounded materialization.

## 5. Partition Synopsis classification

Every field, including Tree Key fields, maintains its configured leaf synopsis:

- MinMax holds conservative non-NULL extrema and NULL/non-NULL presence flags.
- MinMaxBloom adds a fixed-size Bloom summary for equality and `In` pruning.

The compiler propagates sets of possible truth values through the expression.
Classification is `NoMatch` only when TRUE is impossible, `AllMatch` only when
TRUE is the sole possibility, and `MayMatch` otherwise. `Not` complements each
possible truth value; it must not simply invert a boolean pruning decision.
An empty partition is `NoMatch`.

Insertion expands extrema, presence flags, and Bloom bits. Deletion and
replacement never shrink source summaries, so historical values may weaken
pruning. New split targets start from canonical empty synopses. Each actual
Leaf Entry movement expands only its target synopsis, atomically with entry
and membership changes; no whole-source union or completion rescan is required.
Bloom saturation and retained extrema affect work, not eligibility correctness.
Storage owns Bloom sizing, hashing, and encoded-size limits.

## 6. Search integration and resource behavior

Search compiles the predicate, derives Tree Key ranges, and classifies admitted
leaf synopses before candidate admission. `NoMatch` can prune a leaf; `AllMatch`
proves every entry qualifies; `MayMatch` requires exact Leaf Entry evaluation.
No original Vector Record or Opaque Payload read is required to evaluate the
filter projection. Selected candidates still load original vectors for exact
reranking.

Filtering cannot hide malformed data in a loaded leaf body: decoding and
validation remain required even for entries rejected by the predicate. There is
no separate Leaf Entry scan budget; admitted partition sizes determine scan work.
Selective predicates may still require many tree or partition visits when routing
or summaries cannot prune them. Membership-list sorting and deduplication are
bounded by public expression limits; conservative range fallback bounds planning
without changing predicate semantics.

Search reports budget usage, exhausted dimensions, and RaBitQ overlap truncation.
Every returned hit matches, but fewer than `k` hits can result from insufficient
matches or approximate traversal and candidate budgets. Larger budgets or a wider
beam may explore more candidates without a monotonic-result guarantee. Hits
contain Record ID and exact distance; stored fields and payloads remain point-read
projections rather than search result fields.

## 7. Implementation and validation

| Contract | Implementation and evidence |
| --- | --- |
| Typed values, schema binding, and input limits | [`Predicate::validate` and `Value`](../../src/api/schema.rs), request validation in [`SearchRequest`](../../src/api/search.rs) |
| Exact three-valued evaluation and compiled membership | [`CompiledPredicate`](../../src/search/predicate.rs), with independent truth-oracle properties |
| Conservative Tree Key ranges and bounded fallback | [`search::plan`](../../src/search/plan.rs) and its planning tests |
| NULL, `Not`, Bloom saturation, and sound synopsis classification | [`search::predicate` tests](../../src/search/predicate.rs), including `null_truth_sets_and_not_produce_safe_classifications` and `bloom_misses_prune_but_saturation_stays_conservative` |
| Persisted synopsis representation | [`storage::values::synopsis`](../../src/storage/values/synopsis.rs) and storage codec tests |
| End-to-end filtering and tree selection | [`search.kddt`](../../tests/datadriven/search.kddt) and [`fanout.kddt`](../../tests/datadriven/fanout.kddt), executed by [`e2e.rs`](../../tests/e2e.rs) |

Validation covers empty expressions, NULLs, type boundaries, nested negation,
range widening, stale summaries, and malformed stored fields. Exact evaluation
uses an independent SQL truth oracle; synopsis properties prove that `NoMatch`
never rejects a qualifying entry and `AllMatch` never admits a nonqualifying one.
The rationale is recorded in [ADR 0004](../adr/0004-sharded-kmeans-forest.md),
[ADR 0005](../adr/0005-strong-leaf-prefilter.md), and
[ADR 0021](../adr/0021-one-probe-bloom-bound.md).
