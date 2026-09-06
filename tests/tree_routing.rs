//! Initial tree routing and empty-root growth contract tests.

use bytes::Bytes;
use ktann::api::{ErrorKind, PartitionKey};
use ktann::maintenance::routing::{
    Route, route_leaf, route_leaf_for_write, route_leaf_for_write_with_beam,
};
use ktann::storage::backend::{Backend, WriteTxn};
use ktann::storage::keys::{self, LogicalKey, TreeKey};
use ktann::storage::tree_manifest;
use ktann::storage::values::{
    ChildEntry, IndexManifest, PartitionCentroid, PartitionHeader, PartitionState,
    PartitionTransition, PersistentValue, TreeManifest,
};

use support::DeterministicBackend;
use support::builders::{create_committed_tree, id, manifest, pk, read_txn, tree_key, write_txn};

#[allow(dead_code)]
mod support;

fn header_key_at(tree_key: &TreeKey, partition: PartitionKey) -> LogicalKey {
    LogicalKey::Header {
        index: id(7),
        tree_key: tree_key.clone(),
        partition,
    }
}

fn edge_key_at(tree_key: &TreeKey, parent: PartitionKey, child: PartitionKey) -> LogicalKey {
    LogicalKey::ChildEntry {
        index: id(7),
        tree_key: tree_key.clone(),
        partition: parent,
        child,
    }
}

fn state_key_at(tree_key: &TreeKey, partition: PartitionKey) -> LogicalKey {
    LogicalKey::State {
        index: id(7),
        tree_key: tree_key.clone(),
        partition,
    }
}

/// The State value of a stable partition. Every fixture `header` is paired
/// with one because routing validates both authority values of every visited
/// partition against each other.
fn ready_state() -> PersistentValue {
    PersistentValue::PartitionState(PartitionTransition::Ready {
        started_at_unix_millis: 0,
    })
}

fn header(level: u32, count: u32) -> PersistentValue {
    PersistentValue::PartitionHeader(
        PartitionHeader::new(level, count, 0, PartitionState::Ready).expect("header"),
    )
}

fn edge(child: PartitionKey, centroid: f32) -> PersistentValue {
    PersistentValue::ChildEntry(ChildEntry::new(child, vec![centroid]))
}

/// Commits one hand-built topology. Tests install grown shapes directly; the
/// Header entry counts must be exact because routing proves every scanned
/// Child Entry set against them.
async fn write_topology(
    backend: &DeterministicBackend,
    manifest: &IndexManifest,
    values: Vec<(LogicalKey, PersistentValue)>,
) {
    let mut txn = write_txn(backend, manifest).await;
    for (key, value) in values {
        txn.put(key, value).await.expect("put topology");
    }
    txn.commit().await.expect("commit topology");
}

/// Seeds the grown root shape: root PK 1 at level 2 with leaf children PK 2
/// (centroid 0.0) and PK 3 (centroid 10.0).
async fn seed_grown_root(backend: &DeterministicBackend, manifest: &IndexManifest, key: &TreeKey) {
    create_committed_tree(backend, manifest, key).await;
    write_topology(
        backend,
        manifest,
        vec![
            (header_key_at(key, pk(1)), header(2, 2)),
            (header_key_at(key, pk(2)), header(1, 0)),
            (header_key_at(key, pk(3)), header(1, 0)),
            (state_key_at(key, pk(2)), ready_state()),
            (state_key_at(key, pk(3)), ready_state()),
            (edge_key_at(key, pk(1), pk(2)), edge(pk(2), 0.0)),
            (edge_key_at(key, pk(1), pk(3)), edge(pk(3), 10.0)),
        ],
    )
    .await;
}

/// Replaces the grown root shape with the result of a root split: the root
/// rises to level 3 and both leaf edges move under two new level-2 internal
/// parents, PK 4 (centroid 1.0) and PK 6 (centroid 10.0), keeping every
/// internal fanout at exactly two and every descent exactly one level.
async fn move_edge_under_new_parent(
    backend: &DeterministicBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
) {
    let mut txn = write_txn(backend, manifest).await;
    txn.delete(edge_key_at(key, pk(1), pk(2)))
        .await
        .expect("delete moved edge");
    txn.delete(edge_key_at(key, pk(1), pk(3)))
        .await
        .expect("delete sibling edge");
    for (key, value) in [
        (header_key_at(key, pk(1)), header(3, 2)),
        (header_key_at(key, pk(4)), header(2, 2)),
        (header_key_at(key, pk(6)), header(2, 2)),
        (header_key_at(key, pk(5)), header(1, 0)),
        (header_key_at(key, pk(7)), header(1, 0)),
        (state_key_at(key, pk(4)), ready_state()),
        (state_key_at(key, pk(5)), ready_state()),
        (state_key_at(key, pk(6)), ready_state()),
        (state_key_at(key, pk(7)), ready_state()),
        (edge_key_at(key, pk(1), pk(4)), edge(pk(4), 1.0)),
        (edge_key_at(key, pk(1), pk(6)), edge(pk(6), 10.0)),
        (edge_key_at(key, pk(4), pk(2)), edge(pk(2), 0.0)),
        (edge_key_at(key, pk(4), pk(5)), edge(pk(5), 2.0)),
        (edge_key_at(key, pk(6), pk(3)), edge(pk(3), 10.0)),
        (edge_key_at(key, pk(6), pk(7)), edge(pk(7), 12.0)),
    ] {
        txn.put(key, value).await.expect("put moved topology");
    }
    txn.commit().await.expect("commit move");
}

#[tokio::test]
async fn absent_tree_routes_to_none_for_reads() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);

    let route = route_leaf(&mut read_txn(&backend, &manifest).await, &key, &[1.0])
        .await
        .expect("route");
    assert_eq!(route, None);
}

#[tokio::test]
async fn first_insert_creates_the_tree_and_routes_to_the_empty_root() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);

    let route = {
        let mut txn = write_txn(&backend, &manifest).await;
        let route = route_leaf_for_write(&mut txn, &key, &[1.0], 100)
            .await
            .expect("write route");
        txn.commit().await.expect("commit");
        route
    };
    assert_eq!(route.leaf(), pk(1));
    assert_eq!(route.parent(), None);
    assert_eq!(
        route.leaf_header(),
        PartitionHeader::new(1, 0, 0, PartitionState::Ready).expect("header")
    );

    // The committed empty root is immediately routable for reads.
    let route_after = route_leaf(&mut read_txn(&backend, &manifest).await, &key, &[9.0])
        .await
        .expect("read route");
    assert_eq!(route_after, Some(route));

    let directory =
        tree_manifest::read_tree_manifest(&mut read_txn(&backend, &manifest).await, &key)
            .await
            .expect("read manifest")
            .expect("tree exists");
    assert_eq!(directory.root(), pk(1));
    assert_eq!(directory.partition_key_high_water(), pk(1));
}

#[tokio::test]
async fn invalid_vectors_are_rejected_before_the_tree_is_created() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);

    let mut txn = write_txn(&backend, &manifest).await;
    let error = route_leaf_for_write(&mut txn, &key, &[1.0, 2.0], 100)
        .await
        .expect_err("wrong dimension");
    assert_eq!(error.kind(), ErrorKind::InvalidArgument);
    txn.rollback().await;

    // Validation precedes storage work, so no tree exists afterwards either.
    let route = route_leaf(&mut read_txn(&backend, &manifest).await, &key, &[1.0])
        .await
        .expect("route");
    assert_eq!(route, None);
}

#[tokio::test]
async fn concurrent_first_inserts_install_one_tree_and_reroute() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);

    let mut first = write_txn(&backend, &manifest).await;
    let mut second = write_txn(&backend, &manifest).await;
    let first_route = route_leaf_for_write(&mut first, &key, &[1.0], 100)
        .await
        .expect("first route");
    let second_route = route_leaf_for_write(&mut second, &key, &[1.0], 101)
        .await
        .expect("second route");
    assert_eq!(first_route, second_route);

    first.commit().await.expect("first creation commits");
    let error = second
        .commit()
        .await
        .expect_err("racing creation conflicts");
    assert_eq!(error.kind(), ErrorKind::RetryableAbort);

    // The retried attempt observes the winner's tree and routes to the same
    // empty root without installing anything.
    let mut retry = write_txn(&backend, &manifest).await;
    let retried = route_leaf_for_write(&mut retry, &key, &[1.0], 102)
        .await
        .expect("retried route");
    retry.commit().await.expect("retry commits");
    assert_eq!(retried, first_route);
}

#[tokio::test]
async fn grown_root_routes_to_the_nearest_child_and_breaks_ties() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    seed_grown_root(&backend, &manifest, &key).await;

    let cases = [
        (1.0_f32, pk(2)),
        (9.0, pk(3)),
        // An exact distance tie resolves to the smaller Partition Key.
        (5.0, pk(2)),
    ];
    for (query, expected_leaf) in cases {
        let route = route_leaf(&mut read_txn(&backend, &manifest).await, &key, &[query])
            .await
            .expect("route")
            .expect("tree exists");
        assert_eq!(route.leaf(), expected_leaf, "query {query}");
        assert_eq!(route.parent(), Some(pk(1)));
        assert_eq!(route.leaf_header().level(), 1);
    }

    // The write path agrees with the read path on the same topology.
    let mut txn = write_txn(&backend, &manifest).await;
    let route = route_leaf_for_write(&mut txn, &key, &[9.0], 200)
        .await
        .expect("write route");
    assert_eq!(route.leaf(), pk(3));
    assert_eq!(route.parent(), Some(pk(1)));
    txn.rollback().await;
}

#[tokio::test]
async fn write_beam_keeps_a_second_internal_path_in_play() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    create_committed_tree(&backend, &manifest, &key).await;
    write_topology(
        &backend,
        &manifest,
        vec![
            (header_key_at(&key, pk(1)), header(3, 2)),
            (header_key_at(&key, pk(4)), header(2, 2)),
            (header_key_at(&key, pk(6)), header(2, 2)),
            (header_key_at(&key, pk(2)), header(1, 0)),
            (header_key_at(&key, pk(3)), header(1, 0)),
            (header_key_at(&key, pk(7)), header(1, 0)),
            (header_key_at(&key, pk(8)), header(1, 0)),
            (state_key_at(&key, pk(4)), ready_state()),
            (state_key_at(&key, pk(6)), ready_state()),
            (state_key_at(&key, pk(2)), ready_state()),
            (state_key_at(&key, pk(3)), ready_state()),
            (state_key_at(&key, pk(7)), ready_state()),
            (state_key_at(&key, pk(8)), ready_state()),
            (edge_key_at(&key, pk(1), pk(4)), edge(pk(4), 0.0)),
            (edge_key_at(&key, pk(1), pk(6)), edge(pk(6), 100.0)),
            (edge_key_at(&key, pk(4), pk(2)), edge(pk(2), 0.0)),
            (edge_key_at(&key, pk(4), pk(3)), edge(pk(3), 100.0)),
            (edge_key_at(&key, pk(6), pk(7)), edge(pk(7), 10.0)),
            (edge_key_at(&key, pk(6), pk(8)), edge(pk(8), 11.0)),
        ],
    )
    .await;

    // Top-1 greedily follows the centroid-0 branch and cannot see the leaf at
    // centroid 10 under the other root child.
    let mut greedy = write_txn(&backend, &manifest).await;
    let greedy_route = route_leaf_for_write(&mut greedy, &key, &[9.0], 200)
        .await
        .expect("greedy write route");
    assert_eq!(greedy_route.leaf(), pk(2));
    greedy.rollback().await;

    // A wider write beam retains both root branches and chooses the genuinely
    // nearer terminal leaf.
    let mut beam = write_txn(&backend, &manifest).await;
    let beam_route = route_leaf_for_write_with_beam(&mut beam, &key, &[9.0], 200, 4)
        .await
        .expect("beam write route");
    assert_eq!(beam_route.leaf(), pk(7));
    assert_eq!(beam_route.parent(), Some(pk(6)));
    beam.rollback().await;
}

#[tokio::test]
async fn write_beam_is_global_across_parents_at_each_level() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    create_committed_tree(&backend, &manifest, &key).await;

    let mut values = vec![
        (header_key_at(&key, pk(1)), header(4, 2)),
        (header_key_at(&key, pk(2)), header(3, 4)),
        (header_key_at(&key, pk(3)), header(3, 4)),
        (state_key_at(&key, pk(2)), ready_state()),
        (state_key_at(&key, pk(3)), ready_state()),
        (edge_key_at(&key, pk(1), pk(2)), edge(pk(2), 0.0)),
        (edge_key_at(&key, pk(1), pk(3)), edge(pk(3), 100.0)),
    ];
    for (parent, children, leaf_start, centroid) in [
        (pk(2), [pk(10), pk(11), pk(12), pk(13)], 30_u64, 9.0_f32),
        (pk(3), [pk(20), pk(21), pk(22), pk(23)], 34_u64, 20.0_f32),
    ] {
        for (offset, child) in children.into_iter().enumerate() {
            values.push((header_key_at(&key, child), header(2, 1)));
            values.push((state_key_at(&key, child), ready_state()));
            values.push((
                edge_key_at(&key, parent, child),
                edge(
                    child,
                    if parent == pk(2) {
                        100.0 + offset as f32
                    } else {
                        offset as f32
                    },
                ),
            ));
            let leaf = pk(leaf_start + offset as u64);
            values.push((header_key_at(&key, leaf), header(1, 0)));
            values.push((state_key_at(&key, leaf), ready_state()));
            values.push((edge_key_at(&key, child, leaf), edge(leaf, centroid)));
        }
    }
    write_topology(&backend, &manifest, values).await;

    // Top-1 follows the root's nearest branch and reaches the leaf at 9.
    let mut greedy = write_txn(&backend, &manifest).await;
    let greedy_route = route_leaf_for_write(&mut greedy, &key, &[9.0], 200)
        .await
        .expect("greedy write route");
    assert_eq!(greedy_route.leaf(), pk(30));
    greedy.rollback().await;

    // At write beam 8, the root retains both branches, but the next level
    // keeps only the four nearest children globally. All four therefore come
    // from the second root branch, matching search's per-level beam policy.
    let mut beam = write_txn(&backend, &manifest).await;
    let beam_route = route_leaf_for_write_with_beam(&mut beam, &key, &[9.0], 200, 8)
        .await
        .expect("beam write route");
    assert_eq!(beam_route.leaf(), pk(34));
    beam.rollback().await;
}

#[tokio::test]
async fn write_beam_rejects_duplicate_incoming_child_references() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    create_committed_tree(&backend, &manifest, &key).await;
    write_topology(
        &backend,
        &manifest,
        vec![
            (header_key_at(&key, pk(1)), header(3, 2)),
            (header_key_at(&key, pk(2)), header(2, 1)),
            (header_key_at(&key, pk(3)), header(2, 1)),
            (header_key_at(&key, pk(4)), header(1, 0)),
            (state_key_at(&key, pk(2)), ready_state()),
            (state_key_at(&key, pk(3)), ready_state()),
            (state_key_at(&key, pk(4)), ready_state()),
            (edge_key_at(&key, pk(1), pk(2)), edge(pk(2), 0.0)),
            (edge_key_at(&key, pk(1), pk(3)), edge(pk(3), 10.0)),
            (edge_key_at(&key, pk(2), pk(4)), edge(pk(4), 0.0)),
            (edge_key_at(&key, pk(3), pk(4)), edge(pk(4), 10.0)),
        ],
    )
    .await;

    let mut txn = write_txn(&backend, &manifest).await;
    let error = route_leaf_for_write_with_beam(&mut txn, &key, &[5.0], 200, 8)
        .await
        .expect_err("duplicate child reference must fail closed");
    assert_eq!(error.kind(), ErrorKind::Corruption);
    txn.rollback().await;
}

#[tokio::test]
async fn descent_decrements_levels_through_ordinary_internal_partitions() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    create_committed_tree(&backend, &manifest, &key).await;
    write_topology(
        &backend,
        &manifest,
        vec![
            (header_key_at(&key, pk(1)), header(3, 2)),
            (header_key_at(&key, pk(2)), header(2, 2)),
            (header_key_at(&key, pk(3)), header(2, 2)),
            (header_key_at(&key, pk(4)), header(1, 0)),
            (header_key_at(&key, pk(5)), header(1, 0)),
            (header_key_at(&key, pk(6)), header(1, 0)),
            (header_key_at(&key, pk(7)), header(1, 0)),
            (state_key_at(&key, pk(2)), ready_state()),
            (state_key_at(&key, pk(3)), ready_state()),
            (state_key_at(&key, pk(4)), ready_state()),
            (state_key_at(&key, pk(5)), ready_state()),
            (state_key_at(&key, pk(6)), ready_state()),
            (state_key_at(&key, pk(7)), ready_state()),
            (edge_key_at(&key, pk(1), pk(2)), edge(pk(2), 0.0)),
            (edge_key_at(&key, pk(1), pk(3)), edge(pk(3), 10.0)),
            (edge_key_at(&key, pk(2), pk(4)), edge(pk(4), 0.0)),
            (edge_key_at(&key, pk(2), pk(5)), edge(pk(5), 4.0)),
            (edge_key_at(&key, pk(3), pk(6)), edge(pk(6), 8.0)),
            (edge_key_at(&key, pk(3), pk(7)), edge(pk(7), 12.0)),
        ],
    )
    .await;

    let route = route_leaf(&mut read_txn(&backend, &manifest).await, &key, &[3.5])
        .await
        .expect("route")
        .expect("tree exists");
    assert_eq!(route.leaf(), pk(5));
    assert_eq!(route.parent(), Some(pk(2)));

    let route = route_leaf(&mut read_txn(&backend, &manifest).await, &key, &[11.0])
        .await
        .expect("route")
        .expect("tree exists");
    assert_eq!(route.leaf(), pk(7));
    assert_eq!(route.parent(), Some(pk(3)));
}

#[tokio::test]
async fn a_stale_observation_is_replaced_by_fresh_routing() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    seed_grown_root(&backend, &manifest, &key).await;

    // The first observation carries PK 1 as the parent of leaf PK 2.
    let stale: Route = route_leaf(&mut read_txn(&backend, &manifest).await, &key, &[1.0])
        .await
        .expect("route")
        .expect("tree exists");
    assert_eq!(stale.parent(), Some(pk(1)));

    // A root split then moves the leaf's incoming edge down one level.
    move_edge_under_new_parent(&backend, &manifest, &key).await;

    // A fresh route is deterministic under the new topology: the leaf is
    // unchanged but the carried parent is.
    let fresh = route_leaf(&mut read_txn(&backend, &manifest).await, &key, &[1.0])
        .await
        .expect("route")
        .expect("tree exists");
    assert_eq!(fresh.leaf(), pk(2));
    assert_eq!(fresh.parent(), Some(pk(4)));
}

#[tokio::test]
async fn a_concurrent_edge_move_conflicts_and_reroutes_deterministically() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    seed_grown_root(&backend, &manifest, &key).await;

    // One attempt routes and update-protects the leaf Header and the observed
    // incoming edge.
    let mut attempt = write_txn(&backend, &manifest).await;
    let route = route_leaf_for_write(&mut attempt, &key, &[1.0], 300)
        .await
        .expect("write route");
    assert_eq!(route.parent(), Some(pk(1)));

    // A concurrent topology change moves the protected edge and commits.
    move_edge_under_new_parent(&backend, &manifest, &key).await;

    // The stale attempt's commit conflicts and must retry.
    let error = attempt.commit().await.expect_err("moved edge conflicts");
    assert_eq!(error.kind(), ErrorKind::RetryableAbort);

    // The retried attempt reroutes deterministically under the current
    // topology: same leaf, new carried parent, and it commits cleanly.
    let mut retried = write_txn(&backend, &manifest).await;
    let rerouted = route_leaf_for_write(&mut retried, &key, &[1.0], 301)
        .await
        .expect("rerouted");
    assert_eq!(rerouted.leaf(), pk(2));
    assert_eq!(rerouted.parent(), Some(pk(4)));
    retried.commit().await.expect("retry commits");
}

#[tokio::test]
async fn a_concurrent_leaf_header_change_conflicts() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    seed_grown_root(&backend, &manifest, &key).await;

    let mut attempt = write_txn(&backend, &manifest).await;
    route_leaf_for_write(&mut attempt, &key, &[1.0], 400)
        .await
        .expect("write route");

    // A concurrent mutation commits a new count on the routed leaf.
    let mut concurrent = write_txn(&backend, &manifest).await;
    concurrent
        .put(
            header_key_at(&key, pk(2)),
            PersistentValue::PartitionHeader(
                PartitionHeader::new(1, 1, 1, PartitionState::Ready).expect("header"),
            ),
        )
        .await
        .expect("put header");
    concurrent.commit().await.expect("concurrent commits");

    let error = attempt.commit().await.expect_err("header change conflicts");
    assert_eq!(error.kind(), ErrorKind::RetryableAbort);
}

#[tokio::test]
async fn a_missing_root_header_is_corruption() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);

    {
        let mut txn = write_txn(&backend, &manifest).await;
        route_leaf_for_write(&mut txn, &key, &[1.0], 500)
            .await
            .expect("create");
        txn.commit().await.expect("commit");
    }

    // Remove the root Header through the raw seam.
    let mut raw = backend.begin_write().await.expect("begin write");
    raw.delete(Bytes::from(keys::header_key(id(7), &key, pk(1))))
        .await
        .expect("raw delete");
    raw.commit().await.expect("commit");

    let error = route_leaf(&mut read_txn(&backend, &manifest).await, &key, &[1.0])
        .await
        .expect_err("missing root header");
    assert_eq!(error.kind(), ErrorKind::Corruption);
}

#[tokio::test]
async fn a_child_at_the_wrong_level_is_corruption() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    create_committed_tree(&backend, &manifest, &key).await;
    write_topology(
        &backend,
        &manifest,
        vec![
            (header_key_at(&key, pk(1)), header(2, 2)),
            // A level-2 child of a level-2 root is not a leaf and cannot be
            // descended into; the exact one-level step is violated.
            (header_key_at(&key, pk(2)), header(2, 0)),
            (header_key_at(&key, pk(3)), header(1, 0)),
            (state_key_at(&key, pk(2)), ready_state()),
            (state_key_at(&key, pk(3)), ready_state()),
            (edge_key_at(&key, pk(1), pk(2)), edge(pk(2), 0.0)),
            (edge_key_at(&key, pk(1), pk(3)), edge(pk(3), 10.0)),
        ],
    )
    .await;

    let error = route_leaf(&mut read_txn(&backend, &manifest).await, &key, &[1.0])
        .await
        .expect_err("wrong child level");
    assert_eq!(error.kind(), ErrorKind::Corruption);
}

#[tokio::test]
async fn a_self_loop_is_corruption_instead_of_a_cycle() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    create_committed_tree(&backend, &manifest, &key).await;
    write_topology(
        &backend,
        &manifest,
        vec![
            (header_key_at(&key, pk(1)), header(2, 2)),
            (header_key_at(&key, pk(2)), header(1, 0)),
            (state_key_at(&key, pk(2)), ready_state()),
            // The root references itself; descending would revisit PK 1.
            (edge_key_at(&key, pk(1), pk(1)), edge(pk(1), 0.0)),
            (edge_key_at(&key, pk(1), pk(2)), edge(pk(2), 10.0)),
        ],
    )
    .await;

    let error = route_leaf(&mut read_txn(&backend, &manifest).await, &key, &[1.0])
        .await
        .expect_err("self loop");
    assert_eq!(error.kind(), ErrorKind::Corruption);
}

#[tokio::test]
async fn an_internal_partition_must_match_its_exact_header_count() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();

    // One child with an exact count of one: legal for a stable internal
    // partition (an unbalanced drain can leave a single-child target).
    let one_child = tree_key(1);
    create_committed_tree(&backend, &manifest, &one_child).await;
    write_topology(
        &backend,
        &manifest,
        vec![
            (header_key_at(&one_child, pk(1)), header(2, 1)),
            (header_key_at(&one_child, pk(2)), header(1, 0)),
            (state_key_at(&one_child, pk(2)), ready_state()),
            (edge_key_at(&one_child, pk(1), pk(2)), edge(pk(2), 0.0)),
        ],
    )
    .await;
    let route = route_leaf(&mut read_txn(&backend, &manifest).await, &one_child, &[1.0])
        .await
        .expect("one exact child routes")
        .expect("tree exists");
    assert_eq!(route.leaf(), pk(2));

    // More entries than the exact count: the bounded scan disproves the
    // Header.
    let too_many = tree_key(2);
    create_committed_tree(&backend, &manifest, &too_many).await;
    write_topology(
        &backend,
        &manifest,
        vec![
            (header_key_at(&too_many, pk(1)), header(2, 2)),
            (header_key_at(&too_many, pk(2)), header(1, 0)),
            (header_key_at(&too_many, pk(3)), header(1, 0)),
            (header_key_at(&too_many, pk(4)), header(1, 0)),
            (edge_key_at(&too_many, pk(1), pk(2)), edge(pk(2), 0.0)),
            (edge_key_at(&too_many, pk(1), pk(3)), edge(pk(3), 5.0)),
            (edge_key_at(&too_many, pk(1), pk(4)), edge(pk(4), 10.0)),
        ],
    )
    .await;
    let error = route_leaf(&mut read_txn(&backend, &manifest).await, &too_many, &[1.0])
        .await
        .expect_err("more entries than the exact count");
    assert_eq!(error.kind(), ErrorKind::Corruption);

    // Fewer entries than the exact count, and an empty stable internal
    // partition, are both Corruption.
    let too_few = tree_key(3);
    create_committed_tree(&backend, &manifest, &too_few).await;
    write_topology(
        &backend,
        &manifest,
        vec![
            (header_key_at(&too_few, pk(1)), header(2, 2)),
            (header_key_at(&too_few, pk(2)), header(1, 0)),
            (edge_key_at(&too_few, pk(1), pk(2)), edge(pk(2), 0.0)),
        ],
    )
    .await;
    let error = route_leaf(&mut read_txn(&backend, &manifest).await, &too_few, &[1.0])
        .await
        .expect_err("fewer entries than the exact count");
    assert_eq!(error.kind(), ErrorKind::Corruption);
}

#[tokio::test]
async fn a_splitting_leaf_root_still_accepts_writes() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);

    {
        let mut txn = write_txn(&backend, &manifest).await;
        route_leaf_for_write(&mut txn, &key, &[1.0], 600)
            .await
            .expect("create");
        txn.commit().await.expect("commit");
    }
    // A Splitting leaf root still holds its complete entry set and accepts
    // foreground writes until draining starts.
    write_topology(
        &backend,
        &manifest,
        vec![
            (
                header_key_at(&key, pk(1)),
                PersistentValue::PartitionHeader(
                    PartitionHeader::new(1, 0, 0, PartitionState::Splitting).expect("header"),
                ),
            ),
            (
                state_key_at(&key, pk(1)),
                PersistentValue::PartitionState(PartitionTransition::Splitting {
                    left: pk(2),
                    right: pk(3),
                    started_at_unix_millis: 0,
                }),
            ),
        ],
    )
    .await;

    let route = route_leaf(&mut read_txn(&backend, &manifest).await, &key, &[1.0])
        .await
        .expect("splitting routes")
        .expect("tree exists");
    assert_eq!(route.leaf(), pk(1));
    assert_eq!(route.parent(), None);

    let mut txn = write_txn(&backend, &manifest).await;
    let route = route_leaf_for_write(&mut txn, &key, &[1.0], 601)
        .await
        .expect("write route");
    assert_eq!(route.leaf(), pk(1));
    txn.commit().await.expect("write commits");
}

#[tokio::test]
async fn a_merging_partition_is_corruption_until_merge_exists() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);

    {
        let mut txn = write_txn(&backend, &manifest).await;
        route_leaf_for_write(&mut txn, &key, &[1.0], 600)
            .await
            .expect("create");
        txn.commit().await.expect("commit");
    }
    // Merging is unreachable before the merge state machine (#31); routing
    // fails closed rather than guessing at a traversal rule.
    write_topology(
        &backend,
        &manifest,
        vec![
            (
                header_key_at(&key, pk(1)),
                PersistentValue::PartitionHeader(
                    PartitionHeader::new(1, 0, 0, PartitionState::Merging).expect("header"),
                ),
            ),
            (
                state_key_at(&key, pk(1)),
                PersistentValue::PartitionState(PartitionTransition::Merging {
                    started_at_unix_millis: 0,
                }),
            ),
        ],
    )
    .await;

    let error = route_leaf(&mut read_txn(&backend, &manifest).await, &key, &[1.0])
        .await
        .expect_err("merging is unreachable");
    assert_eq!(error.kind(), ErrorKind::Corruption);
}

#[tokio::test]
async fn a_draining_header_without_matching_state_is_corruption() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);

    {
        let mut txn = write_txn(&backend, &manifest).await;
        route_leaf_for_write(&mut txn, &key, &[1.0], 600)
            .await
            .expect("create");
        txn.commit().await.expect("commit");
    }
    // The Header claims DrainingSplit but the State value disagrees.
    write_topology(
        &backend,
        &manifest,
        vec![(
            header_key_at(&key, pk(1)),
            PersistentValue::PartitionHeader(
                PartitionHeader::new(1, 0, 0, PartitionState::DrainingSplit).expect("header"),
            ),
        )],
    )
    .await;

    let error = route_leaf(&mut read_txn(&backend, &manifest).await, &key, &[1.0])
        .await
        .expect_err("header/state disagreement");
    assert_eq!(error.kind(), ErrorKind::Corruption);
}

#[tokio::test]
async fn a_ready_header_with_a_draining_state_is_corruption() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    create_committed_tree(&backend, &manifest, &key).await;

    // The Header claims Ready but the State value is mid-split: every
    // descent hop validates the pair against each other, whatever the
    // Header's discriminator would imply on its own.
    write_topology(
        &backend,
        &manifest,
        vec![(
            state_key_at(&key, pk(1)),
            PersistentValue::PartitionState(PartitionTransition::DrainingSplit {
                left: pk(2),
                right: pk(3),
                started_at_unix_millis: 0,
            }),
        )],
    )
    .await;

    let error = route_leaf(&mut read_txn(&backend, &manifest).await, &key, &[1.0])
        .await
        .expect_err("header/state disagreement");
    assert_eq!(error.kind(), ErrorKind::Corruption);
}

#[tokio::test]
async fn a_receiving_split_root_is_corruption() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    create_committed_tree(&backend, &manifest, &key).await;

    // The root is the one searchable entry point: it is never a split target.
    write_topology(
        &backend,
        &manifest,
        vec![
            (
                header_key_at(&key, pk(1)),
                PersistentValue::PartitionHeader(
                    PartitionHeader::new(1, 0, 0, PartitionState::ReceivingSplit).expect("header"),
                ),
            ),
            (
                state_key_at(&key, pk(1)),
                PersistentValue::PartitionState(PartitionTransition::ReceivingSplit {
                    source: pk(9),
                    started_at_unix_millis: 0,
                }),
            ),
        ],
    )
    .await;

    let error = route_leaf(&mut read_txn(&backend, &manifest).await, &key, &[1.0])
        .await
        .expect_err("receiving root on the read path");
    assert_eq!(error.kind(), ErrorKind::Corruption);

    let mut txn = write_txn(&backend, &manifest).await;
    let error = route_leaf_for_write(&mut txn, &key, &[1.0], 600)
        .await
        .expect_err("receiving root on the write path");
    assert_eq!(error.kind(), ErrorKind::Corruption);
    txn.rollback().await;
}

#[tokio::test]
async fn malformed_child_entry_bytes_are_corruption() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    seed_grown_root(&backend, &manifest, &key).await;

    // Overwrite one Child Entry with garbage bytes through the raw seam.
    let mut raw = backend.begin_write().await.expect("begin write");
    raw.put(
        Bytes::from(keys::child_entry_key(id(7), &key, pk(1), pk(2))),
        Bytes::from(vec![0xde, 0xad, 0xbe, 0xef]),
    )
    .await
    .expect("raw put");
    raw.commit().await.expect("commit");

    let error = route_leaf(&mut read_txn(&backend, &manifest).await, &key, &[1.0])
        .await
        .expect_err("garbage child entry");
    assert_eq!(error.kind(), ErrorKind::Corruption);
}

#[tokio::test]
async fn allocation_exhaustion_is_stable_and_the_tree_stays_routable() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);

    {
        let mut txn = write_txn(&backend, &manifest).await;
        route_leaf_for_write(&mut txn, &key, &[1.0], 700)
            .await
            .expect("create");
        txn.commit().await.expect("commit");
    }

    // Raise the allocator high-water mark to the end of the keyspace.
    {
        let mut txn = write_txn(&backend, &manifest).await;
        txn.put(
            LogicalKey::TreeManifest {
                index: id(7),
                tree_key: key.clone(),
            },
            PersistentValue::TreeManifest(
                TreeManifest::new(pk(1), pk(u64::MAX)).expect("exhausted allocator"),
            ),
        )
        .await
        .expect("raise high water");
        txn.commit().await.expect("commit");
    }

    // Topology growth cannot allocate another Partition Key and fails closed.
    let mut txn = write_txn(&backend, &manifest).await;
    let error = tree_manifest::reserve_partition_keys(&mut txn, &key, 1)
        .await
        .expect_err("space exhausted");
    assert_eq!(error.kind(), ErrorKind::IdExhausted);
    txn.rollback().await;

    // The existing tree remains routable.
    let route = route_leaf(&mut read_txn(&backend, &manifest).await, &key, &[1.0])
        .await
        .expect("route")
        .expect("tree exists");
    assert_eq!(route.leaf(), pk(1));
    assert_eq!(route.parent(), None);
}

#[tokio::test]
async fn a_drain_redirect_survives_edges_moved_to_different_parents() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    create_committed_tree(&backend, &manifest, &key).await;

    // Two splits drain concurrently at adjacent levels: leaf PK 3 drains into
    // PK 4 and PK 5, and the root PK 1 drains into PK 8 and PK 9. The root's
    // drain has already moved the source edge (to PK 3) into PK 8 while the
    // redirect target's edge (to PK 4) still sits in PK 1 — exactly the
    // edge-separated window ADR 0014 permits. The redirect must validate the
    // source's DrainingSplit slot, not a stale parent edge.
    let index = id(7);
    let state = |partition: PartitionKey, transition: PartitionTransition| {
        (
            LogicalKey::State {
                index,
                tree_key: key.clone(),
                partition,
            },
            PersistentValue::PartitionState(transition),
        )
    };
    let header_with_state =
        |partition: PartitionKey, level: u32, count: u32, kind: PartitionState| {
            (
                LogicalKey::Header {
                    index,
                    tree_key: key.clone(),
                    partition,
                },
                PersistentValue::PartitionHeader(
                    PartitionHeader::new(level, count, 0, kind).expect("header"),
                ),
            )
        };
    let mut values = vec![
        header_with_state(pk(1), 2, 3, PartitionState::DrainingSplit),
        state(
            pk(1),
            PartitionTransition::DrainingSplit {
                left: pk(8),
                right: pk(9),
                started_at_unix_millis: 10,
            },
        ),
        header_with_state(pk(8), 2, 2, PartitionState::ReceivingSplit),
        state(
            pk(8),
            PartitionTransition::ReceivingSplit {
                source: pk(1),
                started_at_unix_millis: 10,
            },
        ),
        header_with_state(pk(9), 2, 0, PartitionState::ReceivingSplit),
        state(
            pk(9),
            PartitionTransition::ReceivingSplit {
                source: pk(1),
                started_at_unix_millis: 10,
            },
        ),
        header_with_state(pk(3), 1, 0, PartitionState::DrainingSplit),
        state(
            pk(3),
            PartitionTransition::DrainingSplit {
                left: pk(4),
                right: pk(5),
                started_at_unix_millis: 20,
            },
        ),
    ];
    for target in [pk(4), pk(5)] {
        values.push(header_with_state(
            target,
            1,
            0,
            PartitionState::ReceivingSplit,
        ));
        values.push(state(
            target,
            PartitionTransition::ReceivingSplit {
                source: pk(3),
                started_at_unix_millis: 20,
            },
        ));
    }
    values.push((
        LogicalKey::Centroid {
            index,
            tree_key: key.clone(),
            partition: pk(4),
        },
        PersistentValue::PartitionCentroid(PartitionCentroid::new(vec![0.1])),
    ));
    values.push((
        LogicalKey::Centroid {
            index,
            tree_key: key.clone(),
            partition: pk(5),
        },
        PersistentValue::PartitionCentroid(PartitionCentroid::new(vec![9.9])),
    ));
    // The draining root's remaining edges, plus the moved pair.
    values.push((edge_key_at(&key, pk(1), pk(4)), edge(pk(4), 0.2)));
    values.push((edge_key_at(&key, pk(1), pk(2)), edge(pk(2), 20.0)));
    values.push((edge_key_at(&key, pk(1), pk(6)), edge(pk(6), 30.0)));
    values.push((edge_key_at(&key, pk(8), pk(3)), edge(pk(3), 0.0)));
    values.push((edge_key_at(&key, pk(8), pk(5)), edge(pk(5), 9.9)));
    write_topology(&backend, &manifest, values).await;

    // Routing x = 0.05 descends the root's draining family to PK 3's edge in
    // PK 8, then redirects to the nearer target PK 4 — whose edge lives in PK
    // 1, a different parent body. The write route must commit.
    let mut txn = write_txn(&backend, &manifest).await;
    let route = route_leaf_for_write(&mut txn, &key, &[0.05], 800)
        .await
        .expect("write route through edge-separated families");
    assert_eq!(route.leaf(), pk(4));
    txn.commit().await.expect("redirect route validates");
}
