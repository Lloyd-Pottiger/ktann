//! Cross-checks the brute-force oracle against the published siftsmall
//! ground truth (INRIA TEXMEX), independently of the engine: every published
//! top-100 neighbor of a query must appear in the oracle's own top-100.

#[allow(dead_code)]
mod support;

#[path = "support/fixtures.rs"]
mod fixtures;

use std::collections::BTreeSet;

use bytes::Bytes;
use ktann::api::Metric;
use support::oracle;

#[test]
fn siftsmall_ground_truth_matches_oracle() {
    let directory = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/datadriven/data");
    let base = fixtures::read_vectors(&directory, "siftsmall_base.fvecs");
    let queries = fixtures::read_vectors(&directory, "siftsmall_query.fvecs");
    let ground_truth = fixtures::read_ivecs(&directory, "siftsmall_groundtruth.ivecs");
    let ids: Vec<_> = (0..base.len())
        .map(|id| Bytes::from(format!("r{id:06}")))
        .collect();
    assert_eq!(ground_truth.len(), queries.len());

    // Twenty queries keep the brute-force cost bounded in debug builds while
    // any systematic distance error would surface in the first one.
    for (ordinal, (query, expected)) in queries.iter().zip(&ground_truth).enumerate().take(20) {
        assert_eq!(expected.len(), 100, "query {ordinal} ground truth width");
        let truth = oracle::truth_vectors(&ids, &base, Metric::L2, query, 100);
        let predicted: BTreeSet<&Bytes> = truth.iter().map(|(id, _)| id).collect();
        assert_eq!(predicted.len(), 100, "query {ordinal} oracle width");
        for neighbor in expected {
            let index = usize::try_from(*neighbor).expect("non-negative neighbor index");
            let id = &ids[index];
            assert!(
                predicted.contains(id),
                "query {ordinal}: oracle top-100 missed published neighbor {neighbor}"
            );
        }
    }
}
