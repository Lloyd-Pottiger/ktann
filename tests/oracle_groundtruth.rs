//! Cross-checks the brute-force oracle against the published siftsmall
//! ground truth (INRIA TEXMEX), independently of the engine: every published
//! top-100 neighbor of a query must appear in the oracle's own top-100.

#[allow(dead_code)]
mod support;

use std::collections::BTreeSet;

use bytes::Bytes;
use ktann::api::Metric;
use support::oracle::{self, Model, ModelRecord};

#[test]
fn siftsmall_ground_truth_matches_oracle() {
    let base = support::dataset::generate("file:siftsmall_base.fvecs", 128, 0);
    let mut model = Model::new();
    for (id, vector) in base.ids.iter().zip(&base.vectors) {
        model.insert(
            id.clone(),
            ModelRecord {
                vector: vector.clone(),
                fields: Vec::new().into_boxed_slice(),
            },
        );
    }
    let queries = support::dataset::generate("file:siftsmall_query.fvecs", 128, 0);
    let ground_truth = support::dataset::read_ivecs_fixture("siftsmall_groundtruth.ivecs");
    assert_eq!(ground_truth.len(), queries.vectors.len());

    // Twenty queries keep the brute-force cost bounded in debug builds while
    // any systematic distance error would surface in the first one.
    for (ordinal, (query, expected)) in queries
        .vectors
        .iter()
        .zip(&ground_truth)
        .enumerate()
        .take(20)
    {
        assert_eq!(expected.len(), 100, "query {ordinal} ground truth width");
        let truth = oracle::truth(&model, Metric::L2, query, 100, &|_: &ModelRecord| true);
        let predicted: BTreeSet<&Bytes> = truth.iter().map(|(id, _)| id).collect();
        assert_eq!(predicted.len(), 100, "query {ordinal} oracle width");
        for neighbor in expected {
            let index = usize::try_from(*neighbor).expect("non-negative neighbor index");
            let id = &base.ids[index];
            assert!(
                predicted.contains(id),
                "query {ordinal}: oracle top-100 missed published neighbor {neighbor}"
            );
        }
    }
}
