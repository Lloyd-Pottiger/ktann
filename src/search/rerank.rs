//! Exact Vector Record reranking for globally selected Leaf Entries.
//!
//! Only SQL TRUE entries qualify. Record loads use one snapshot and verify
//! authoritative ownership and filter fields. Exact f64 distances use original,
//! unrotated vectors; ties use unsigned lexicographic Record ID order.

use bytes::Bytes;

use crate::api::{Error, ErrorKind, Result, SearchHit};
use crate::storage::ReadLogicalTxn;
use crate::storage::backend::ReadOps;

use super::candidates::LeafCandidate;
use super::numeric::{ExactDistance, VectorKernel, compare_finite};

/// The number of Vector Records loaded per backend batch.
///
/// Each record reads two logical keys (record body and Record Location), so
/// one batch issues at most 512 encoded keys — comfortably below backend batch
/// ceilings — and bounds the number of decoded records held in flight.
const RECORD_LOAD_BATCH: usize = 256;

/// Exact hits and the number of authoritative Vector Records read.
///
/// Global candidate admission owns the rerank budget; this stage reports the
/// actual reads through `SearchBudgetUsage::exact_rerank_candidates`.
pub(crate) struct ExactRerankOutcome {
    hits: Vec<SearchHit>,
    exact_rerank_candidates: u32,
}

impl ExactRerankOutcome {
    /// Returns the hits ordered by exact distance and Record ID bytes.
    #[cfg(test)]
    pub(crate) fn hits(&self) -> &[SearchHit] {
        &self.hits
    }

    /// Consumes the outcome and returns its hits.
    pub(crate) fn into_hits(self) -> Vec<SearchHit> {
        self.hits
    }

    /// Returns how many Vector Records were read and exactly reranked.
    pub(crate) const fn exact_rerank_candidates(&self) -> u32 {
        self.exact_rerank_candidates
    }
}

/// Loads original Vector Records and exactly reranks the selected candidates.
///
/// `candidates` are the filtered, globally selected candidates in deterministic
/// rough order (rough distance, then Record ID); `query` is the original
/// unrotated caller vector. Every load reads from `txn`'s one consistent
/// snapshot in bounded batches.
///
/// Global candidate admission has already bounded this work by the rerank
/// budget. Every supplied candidate is loaded exactly once.
///
/// The stage fails closed. A fully absent record group, a Record ID that does
/// not match the requested candidate, a Record Location that disagrees with
/// the candidate's source Tree Key or Leaf Partition, record fields that
/// diverge from the candidate's Leaf Entry projection are Corruption. Traversal
/// has already validated that candidate Record IDs are unique.
///
/// Final ordering sorts by the exact ranking value and then unsigned
/// lexicographic Record ID bytes, truncates to `k`, and builds Search Hits
/// carrying the caller-visible exact f64 distance. The result is deterministic
/// for identical inputs.
pub(crate) async fn exact_rerank<T: ReadOps>(
    txn: &mut ReadLogicalTxn<'_, T>,
    kernel: &VectorKernel,
    query: &[f32],
    candidates: Vec<LeafCandidate>,
    k: usize,
    check: &(impl Fn() -> Result<()> + Sync),
) -> Result<ExactRerankOutcome> {
    if k == 0 {
        return Err(Error::invalid_argument());
    }

    let load_count = candidates.len();
    let mut scored: Vec<(Bytes, ExactDistance)> = Vec::with_capacity(load_count);
    if load_count != 0 {
        let exact_query = kernel.prepare_exact_query(query)?;
        for (batch_index, batch) in candidates.chunks(RECORD_LOAD_BATCH).enumerate() {
            check()?;
            if batch_index > 0 {
                super::cooperate(check).await?;
            }
            let ids = batch
                .iter()
                .map(|candidate| candidate.record_id().clone())
                .collect();
            let groups = txn.read_record_groups(ids, false).await?;
            for (ordinal, (candidate, group)) in batch.iter().zip(groups).enumerate() {
                if ordinal > 0 && ordinal % 32 == 0 {
                    super::cooperate(check).await?;
                }
                let group = group.ok_or_else(|| Error::new(ErrorKind::Corruption))?;
                let record = group.record();
                if record.record_id() != candidate.record_id()
                    || group.location() != candidate.location()
                    || record.fields() != candidate.fields()
                {
                    return Err(Error::new(ErrorKind::Corruption));
                }
                let distance = exact_query.distance_validated(record.vector())?;
                scored.push((candidate.record_id().clone(), distance));
            }
        }
    }

    scored.sort_unstable_by(|left, right| {
        compare_finite(left.1.ranking(), right.1.ranking()).then_with(|| left.0.cmp(&right.0))
    });
    scored.truncate(k);

    let mut hits = Vec::with_capacity(scored.len());
    for (id, distance) in scored {
        // The persistent codecs bound Record ID length and the kernel
        // guarantees a finite distance, so construction can only fail on
        // corrupted persistent state.
        hits.push(
            SearchHit::new(id, distance.distance())
                .map_err(|_| Error::new(ErrorKind::Corruption))?,
        );
    }

    Ok(ExactRerankOutcome {
        hits,
        exact_rerank_candidates: u32::try_from(load_count)
            .map_err(|_| Error::new(ErrorKind::LimitExceeded))?,
    })
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;

    use bytes::Bytes;

    use crate::api::{
        CompareOp, DataType, ErrorKind, FieldId, FieldSchema, IndexConfig, LogicalIndexId, Metric,
        PartitionKey, Predicate, Result, Value,
    };
    use crate::storage::ReadLogicalTxn;
    use crate::storage::keys::{self, TreeKey};
    use crate::storage::test_support::MockReadTxn;
    use crate::storage::values::{
        BloomParameters, IndexLifecycle, IndexManifest, PersistentValue, RecordLocation,
        ValueCodec, VectorRecord,
    };

    use super::super::numeric::VectorKernel;
    use super::super::predicate::CompiledPredicate;
    use super::super::rabitq::test_approximate_distance;
    use super::{ExactRerankOutcome, LeafCandidate, exact_rerank};

    const DIMENSION: usize = 3;
    const SEED: [u8; 32] = [7; 32];
    const QUERY: [f32; DIMENSION] = [3.5, -2.0, 7.25];

    #[derive(Clone)]
    struct TestRecord {
        id: String,
        vector: [f32; DIMENSION],
        bucket: i64,
        tag: Option<&'static str>,
        score: Option<f64>,
    }

    impl TestRecord {
        fn fields(&self) -> Vec<Value> {
            vec![
                Value::I64(self.bucket),
                self.tag
                    .map_or(Value::Null, |tag| Value::String(tag.to_owned())),
                self.score.map_or(Value::Null, Value::F64),
            ]
        }
    }

    fn manifest(metric: Metric) -> IndexManifest {
        let fields = vec![
            FieldSchema::new("bucket", DataType::I64).expect("valid field"),
            FieldSchema::new("tag", DataType::String)
                .expect("valid field")
                .nullable(),
            FieldSchema::new("score", DataType::F64)
                .expect("valid field")
                .nullable(),
        ];
        let bloom = fields
            .iter()
            .map(|field| BloomParameters::derive(field.synopsis()).expect("valid synopsis"))
            .collect();
        let config = IndexConfig::new(DIMENSION, metric)
            .expect("valid config")
            .with_fields(fields)
            .expect("valid fields")
            .with_tree_key_fields(vec![FieldId(0)])
            .expect("valid tree key fields");
        IndexManifest::new(
            IndexLifecycle::Active,
            LogicalIndexId::new(1).expect("valid id"),
            config,
            SEED,
            bloom,
        )
        .expect("valid manifest")
    }

    /// A deterministic, replayable record set with NULLs and boundary values.
    fn fixture_records(count: i32) -> Vec<TestRecord> {
        let mut state = 0x2545_f491_u32;
        let mut next = move || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            // [-10, 10) at 24-bit granularity; the +1 keeps every vector nonzero.
            (state >> 8) as f32 / 16_777_216.0 * 20.0 - 10.0
        };
        (0..count)
            .map(|index| TestRecord {
                id: format!("r{index:02}"),
                vector: [next() + 1.0, next(), next()],
                bucket: i64::from(index % 4),
                tag: match index % 3 {
                    0 => None,
                    1 => Some("red"),
                    _ => Some("blue"),
                },
                score: match index % 4 {
                    0 => None,
                    1 => Some(0.0),
                    2 => Some(1.5),
                    _ => Some(-2.5),
                },
            })
            .collect()
    }

    fn tree_key(bucket: i64) -> TreeKey {
        TreeKey::encode(&[DataType::I64], &[Value::I64(bucket)]).expect("canonical tree key")
    }

    fn leaf(value: u64) -> PartitionKey {
        PartitionKey::new(value).expect("valid partition key")
    }

    fn candidate(record: &TestRecord, rough: f64) -> LeafCandidate {
        LeafCandidate::from_leaf(
            Bytes::copy_from_slice(record.id.as_bytes()),
            record.fields().into_boxed_slice(),
            test_approximate_distance(rough),
            RecordLocation::new(tree_key(record.bucket), leaf(1)),
        )
    }

    fn record_item(manifest: &IndexManifest, record: &TestRecord) -> (Vec<u8>, Vec<u8>) {
        let id = Bytes::copy_from_slice(record.id.as_bytes());
        let key = keys::record_key(manifest.logical_index_id(), &id).expect("record key");
        let value = ValueCodec::for_index(manifest)
            .encode(&PersistentValue::VectorRecord(VectorRecord::new(
                id,
                record.vector.to_vec(),
                record.fields(),
            )))
            .expect("encode record");
        (key, value)
    }

    fn location_item(
        manifest: &IndexManifest,
        record: &TestRecord,
        leaf: u64,
    ) -> (Vec<u8>, Vec<u8>) {
        let id = Bytes::copy_from_slice(record.id.as_bytes());
        let key = keys::location_key(manifest.logical_index_id(), &id).expect("location key");
        let value = ValueCodec::for_index(manifest)
            .encode(&PersistentValue::RecordLocation(RecordLocation::new(
                tree_key(record.bucket),
                self::leaf(leaf),
            )))
            .expect("encode location");
        (key, value)
    }

    fn record_group_data(
        manifest: &IndexManifest,
        records: &[TestRecord],
    ) -> Vec<(Vec<u8>, Vec<u8>)> {
        records
            .iter()
            .flat_map(|record| {
                [
                    record_item(manifest, record),
                    location_item(manifest, record, 1),
                ]
            })
            .collect()
    }

    /// A rerank fixture mock: the rerank stage only point-reads record groups;
    /// it never scans.
    fn mock_txn(items: Vec<(Vec<u8>, Vec<u8>)>) -> MockReadTxn {
        MockReadTxn::new(items).with_failing_scans()
    }

    async fn rerank(
        manifest: &IndexManifest,
        txn: MockReadTxn,
        candidates: Vec<LeafCandidate>,
        k: usize,
    ) -> Result<ExactRerankOutcome> {
        let mut txn = ReadLogicalTxn::for_index(txn, manifest);
        let kernel =
            VectorKernel::new(DIMENSION, manifest.config().metric(), SEED).expect("valid kernel");
        exact_rerank(&mut txn, &kernel, &QUERY, candidates, k, &|| Ok(())).await
    }

    fn compare(left: f64, right: f64) -> Ordering {
        if left < right {
            Ordering::Less
        } else if left > right {
            Ordering::Greater
        } else {
            Ordering::Equal
        }
    }

    fn oracle_topk(metric: Metric, records: &[&TestRecord], k: usize) -> Vec<(Bytes, f64)> {
        let mut scored: Vec<(Bytes, f64)> = records
            .iter()
            .map(|record| {
                (
                    Bytes::copy_from_slice(record.id.as_bytes()),
                    crate::search::test_support::exact_distance(metric, &QUERY, &record.vector),
                )
            })
            .collect();
        scored.sort_by(|left, right| compare(left.1, right.1).then_with(|| left.0.cmp(&right.0)));
        scored.truncate(k);
        scored
    }

    fn assert_hits(outcome: &ExactRerankOutcome, expected: &[(Bytes, f64)]) {
        assert_eq!(outcome.hits().len(), expected.len());
        for (hit, (id, distance)) in outcome.hits().iter().zip(expected) {
            assert_eq!(hit.id(), id);
            assert_eq!(hit.distance(), *distance);
        }
    }

    fn candidates(records: &[TestRecord]) -> Vec<LeafCandidate> {
        records
            .iter()
            .enumerate()
            .map(|(index, record)| candidate(record, index as f64))
            .collect()
    }

    #[tokio::test]
    async fn rerank_matches_the_brute_force_oracle_for_every_metric() {
        let records = fixture_records(24);
        for metric in [Metric::L2, Metric::InnerProduct, Metric::Cosine] {
            let manifest = manifest(metric);
            let data = record_group_data(&manifest, &records);
            let expected_reads = records.len() as u32;
            let outcome = rerank(&manifest, mock_txn(data), candidates(&records), 5)
                .await
                .expect("rerank succeeds");
            let expected = oracle_topk(metric, &records.iter().collect::<Vec<_>>(), 5);
            assert_hits(&outcome, &expected);
            assert_eq!(outcome.exact_rerank_candidates(), expected_reads);
        }
    }

    #[tokio::test]
    async fn filtered_rerank_matches_the_filtered_oracle_for_every_metric() {
        let records = fixture_records(24);
        let predicates = [
            // Numeric boundary: scores are exactly 0.0, 1.5, or -2.5.
            (
                Predicate::Compare {
                    field: FieldId(2),
                    op: CompareOp::GreaterOrEqual,
                    value: Value::F64(1.5),
                },
                Box::new(|record: &TestRecord| record.score.is_some_and(|s| s >= 1.5))
                    as Box<dyn Fn(&TestRecord) -> bool>,
            ),
            // Nested boolean: (tag = 'red' OR score < 0) AND score IS NOT NULL.
            (
                Predicate::And(vec![
                    Predicate::Or(vec![
                        Predicate::Compare {
                            field: FieldId(1),
                            op: CompareOp::Eq,
                            value: Value::String("red".to_owned()),
                        },
                        Predicate::Compare {
                            field: FieldId(2),
                            op: CompareOp::Lt,
                            value: Value::F64(0.0),
                        },
                    ]),
                    Predicate::IsNotNull(FieldId(2)),
                ]),
                Box::new(|record: &TestRecord| {
                    (record.tag == Some("red") || record.score.is_some_and(|s| s < 0.0))
                        && record.score.is_some()
                }),
            ),
        ];

        for metric in [Metric::L2, Metric::InnerProduct, Metric::Cosine] {
            let manifest = manifest(metric);
            for (predicate, oracle) in &predicates {
                let compiled =
                    CompiledPredicate::compile(predicate.clone(), manifest.config().fields())
                        .expect("compile");
                let filtered: Vec<_> = candidates(&records)
                    .into_iter()
                    .filter(|candidate| compiled.matches(candidate.fields()).expect("filter"))
                    .collect();
                let expected_reads = filtered.len() as u32;
                let data = record_group_data(&manifest, &records);
                let outcome = rerank(&manifest, mock_txn(data), filtered, 4)
                    .await
                    .expect("rerank succeeds");
                let qualifying: Vec<&TestRecord> = records.iter().filter(|r| oracle(r)).collect();
                let expected = oracle_topk(metric, &qualifying, 4);
                assert_hits(&outcome, &expected);
                assert_eq!(outcome.exact_rerank_candidates(), expected_reads);
            }
        }
    }

    #[test]
    fn predicate_boundaries_match_a_three_valued_oracle() {
        let manifest = manifest(Metric::L2);
        let records = vec![
            TestRecord {
                id: "nulls".to_owned(),
                vector: [1.0, 0.0, 0.0],
                bucket: 0,
                tag: None,
                score: None,
            },
            TestRecord {
                id: "empty".to_owned(),
                vector: [0.0, 1.0, 0.0],
                bucket: 1,
                tag: Some(""),
                score: Some(0.0),
            },
            TestRecord {
                id: "red".to_owned(),
                vector: [0.0, 0.0, 1.0],
                bucket: 2,
                tag: Some("red"),
                score: Some(1.5),
            },
            TestRecord {
                id: "blue".to_owned(),
                vector: [1.0, 1.0, 0.0],
                bucket: 3,
                tag: Some("blue"),
                score: Some(-1.5),
            },
            TestRecord {
                id: "red-neg".to_owned(),
                vector: [1.0, 0.0, 1.0],
                bucket: 4,
                tag: Some("red"),
                score: Some(-1.5),
            },
        ];
        let compare = |field: u16, op: CompareOp, value: Value| Predicate::Compare {
            field: FieldId(field),
            op,
            value,
        };
        let cases: Vec<(Predicate, Vec<&str>)> = vec![
            // Boundary equality and inequality include the exact boundary.
            (
                compare(2, CompareOp::LessOrEqual, Value::F64(0.0)),
                vec!["empty", "blue", "red-neg"],
            ),
            (
                compare(1, CompareOp::Eq, Value::String(String::new())),
                vec!["empty"],
            ),
            (
                compare(1, CompareOp::NotEq, Value::String(String::new())),
                vec!["red", "blue", "red-neg"],
            ),
            // NOT over a NULL-comparing expression is still UNKNOWN.
            (
                Predicate::Not(Box::new(compare(
                    1,
                    CompareOp::Eq,
                    Value::String("red".to_owned()),
                ))),
                vec!["empty", "blue"],
            ),
            (Predicate::IsNull(FieldId(2)), vec!["nulls"]),
            (
                Predicate::IsNotNull(FieldId(1)),
                vec!["empty", "red", "blue", "red-neg"],
            ),
            // Nested boolean over both fields.
            (
                Predicate::And(vec![
                    Predicate::Or(vec![
                        compare(1, CompareOp::Eq, Value::String("red".to_owned())),
                        compare(2, CompareOp::Lt, Value::F64(0.0)),
                    ]),
                    Predicate::IsNotNull(FieldId(2)),
                ]),
                vec!["red", "blue", "red-neg"],
            ),
        ];

        for (predicate, expected) in cases {
            let compiled =
                CompiledPredicate::compile(predicate, manifest.config().fields()).expect("compile");
            let admitted: Vec<_> = candidates(&records)
                .into_iter()
                .filter(|candidate| compiled.matches(candidate.fields()).expect("filter"))
                .collect();
            let admitted: Vec<&[u8]> = admitted
                .iter()
                .map(|candidate| candidate.record_id().as_ref())
                .collect();
            assert_eq!(
                admitted,
                expected.iter().map(|id| id.as_bytes()).collect::<Vec<_>>()
            );
        }
    }

    #[tokio::test]
    async fn a_missing_record_group_is_corruption() {
        let manifest = manifest(Metric::L2);
        let records = fixture_records(24);
        let outcome = rerank(
            &manifest,
            mock_txn(vec![]),
            vec![candidate(&records[0], 0.0)],
            1,
        )
        .await;
        assert_eq!(
            outcome.err().map(|error| error.kind()),
            Some(ErrorKind::Corruption)
        );
    }

    #[tokio::test]
    async fn a_mismatched_record_location_is_corruption() {
        let manifest = manifest(Metric::L2);
        let records = fixture_records(24);

        // The authoritative Record Location names a different Leaf Partition.
        let data = vec![
            record_item(&manifest, &records[0]),
            location_item(&manifest, &records[0], 2),
        ];
        let outcome = rerank(
            &manifest,
            mock_txn(data),
            vec![candidate(&records[0], 0.0)],
            1,
        )
        .await;
        assert_eq!(
            outcome.err().map(|error| error.kind()),
            Some(ErrorKind::Corruption)
        );

        // The authoritative Record Location names a different Tree Key.
        let mut moved = records[1].clone();
        moved.bucket += 1;
        let data = vec![
            record_item(&manifest, &records[1]),
            location_item(&manifest, &moved, 1),
        ];
        let outcome = rerank(
            &manifest,
            mock_txn(data),
            vec![candidate(&records[1], 0.0)],
            1,
        )
        .await;
        assert_eq!(
            outcome.err().map(|error| error.kind()),
            Some(ErrorKind::Corruption)
        );
    }

    #[tokio::test]
    async fn record_fields_diverging_from_the_projection_are_corruption() {
        let manifest = manifest(Metric::L2);
        let records = fixture_records(24);
        let data = record_group_data(&manifest, &records[..1]);

        let mut divergent = records[0].clone();
        divergent.tag = Some("forged");
        let candidate = LeafCandidate::from_leaf(
            Bytes::copy_from_slice(divergent.id.as_bytes()),
            divergent.fields().into_boxed_slice(),
            test_approximate_distance(0.0),
            RecordLocation::new(tree_key(divergent.bucket), leaf(1)),
        );
        let outcome = rerank(&manifest, mock_txn(data), vec![candidate], 1).await;
        assert_eq!(
            outcome.err().map(|error| error.kind()),
            Some(ErrorKind::Corruption)
        );
    }

    #[tokio::test]
    async fn a_record_id_mismatch_is_corruption() {
        let manifest = manifest(Metric::L2);
        let records = fixture_records(24);

        // The stored record value encodes another Record ID under this key.
        let id = Bytes::copy_from_slice(records[0].id.as_bytes());
        let key = keys::record_key(manifest.logical_index_id(), &id).expect("record key");
        let value = ValueCodec::for_index(&manifest)
            .encode(&PersistentValue::VectorRecord(VectorRecord::new(
                Bytes::from_static(b"other"),
                records[0].vector.to_vec(),
                records[0].fields(),
            )))
            .expect("encode record");
        let data = vec![(key, value), location_item(&manifest, &records[0], 1)];
        let outcome = rerank(
            &manifest,
            mock_txn(data),
            vec![candidate(&records[0], 0.0)],
            1,
        )
        .await;
        assert_eq!(
            outcome.err().map(|error| error.kind()),
            Some(ErrorKind::Corruption)
        );
    }

    #[tokio::test]
    async fn empty_candidates_do_not_load_records() {
        let manifest = manifest(Metric::L2);
        let outcome = rerank(&manifest, mock_txn(vec![]), vec![], 1)
            .await
            .expect("rerank succeeds");
        assert!(outcome.hits().is_empty());
        assert_eq!(outcome.exact_rerank_candidates(), 0);
    }

    #[tokio::test]
    async fn record_loads_respect_bounded_backend_batches() {
        let manifest = manifest(Metric::L2);
        // Force a full batch and a partial batch. The mock admits one batch
        // of record/location pairs, but rejects one unbounded read.
        let records = fixture_records(super::RECORD_LOAD_BATCH as i32 + 6);
        let data = record_group_data(&manifest, &records);
        let mut txn = mock_txn(data);
        txn.max_batch_size = 2 * super::RECORD_LOAD_BATCH;
        let outcome = rerank(&manifest, txn, candidates(&records), 5)
            .await
            .expect("batched rerank succeeds");
        let expected = oracle_topk(Metric::L2, &records.iter().collect::<Vec<_>>(), 5);
        assert_hits(&outcome, &expected);
        assert_eq!(outcome.exact_rerank_candidates(), records.len() as u32);
    }

    #[tokio::test]
    async fn identical_distances_order_by_record_id_bytes_deterministically() {
        let manifest = manifest(Metric::L2);
        let records: Vec<TestRecord> = ["b", "aa", "a", "ab"]
            .into_iter()
            .enumerate()
            .map(|(index, id)| TestRecord {
                id: id.to_owned(),
                vector: [1.0, 2.0, 3.0],
                bucket: index as i64,
                tag: None,
                score: None,
            })
            .collect();
        let data = record_group_data(&manifest, &records);

        let expected_ids: Vec<&[u8]> = [b"a".as_slice(), b"aa", b"ab", b"b"].to_vec();
        let mut previous = None;
        for _ in 0..2 {
            let outcome = rerank(&manifest, mock_txn(data.clone()), candidates(&records), 4)
                .await
                .expect("rerank succeeds");
            let ids: Vec<&[u8]> = outcome.hits().iter().map(|hit| hit.id().as_ref()).collect();
            assert_eq!(ids, expected_ids);
            let distances: Vec<f64> = outcome.hits().iter().map(|hit| hit.distance()).collect();
            assert!(distances.windows(2).all(|pair| pair[0] == pair[1]));
            if let Some(previous) = previous {
                assert_eq!(distances, previous, "identical inputs rerank identically");
            }
            previous = Some(distances);
        }
    }

    #[tokio::test]
    async fn rerank_rejects_a_zero_result_limit() {
        let manifest = manifest(Metric::L2);
        let outcome = rerank(&manifest, mock_txn(vec![]), vec![], 0).await;
        assert_eq!(
            outcome.err().map(|error| error.kind()),
            Some(ErrorKind::InvalidArgument)
        );
    }
}
