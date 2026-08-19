//! Initial tree routing and empty-root growth contract tests.

use bytes::Bytes;
use ktann::api::{
    DataType, ErrorKind, FieldId, FieldSchema, IndexConfig, LogicalIndexId, Metric, PartitionKey,
    Value,
};
use ktann::maintenance::routing::{Route, route_leaf, route_leaf_for_write};
use ktann::storage::backend::{Backend, WriteTxn};
use ktann::storage::keys::{self, LogicalKey, TreeKey};
use ktann::storage::values::{
    ChildEntry, IndexLifecycle, IndexManifest, PartitionHeader, PartitionState, PersistentValue,
    TreeManifest,
};
use ktann::storage::{ReadLogicalTxn, WriteLogicalTxn, tree_manifest};

use support::DeterministicBackend;

#[allow(dead_code)]
mod support;

fn id(value: u64) -> LogicalIndexId {
    LogicalIndexId::new(value).expect("test Logical Index ID is nonzero")
}

fn pk(value: u64) -> PartitionKey {
    PartitionKey::new(value).expect("test Partition Key is nonzero")
}

/// A one-dimensional L2 index: rotation is the identity at dimension 1, so
/// routing distances are plain squared differences and fixtures need no
/// numeric setup.
fn manifest() -> IndexManifest {
    let config = IndexConfig::new(1, Metric::L2)
        .expect("valid config")
        .with_fields(vec![FieldSchema::new("a", DataType::I64).expect("field")])
        .expect("valid fields")
        .with_tree_key_fields(vec![FieldId(0)])
        .expect("valid tree key fields");
    IndexManifest::new(IndexLifecycle::Active, id(7), config, [7; 32], vec![None])
        .expect("valid manifest")
}

fn tree_key(value: i64) -> TreeKey {
    TreeKey::encode(&[DataType::I64], &[Value::I64(value)]).expect("canonical key")
}

async fn write_txn<'b, 'm>(
    backend: &'b DeterministicBackend,
    manifest: &'m IndexManifest,
) -> WriteLogicalTxn<'m, <DeterministicBackend as Backend>::WriteTxn<'b>> {
    let raw = backend.begin_write().await.expect("begin write");
    WriteLogicalTxn::for_index(
        raw,
        manifest,
        backend.hard_limits(),
        backend.admission_budget(),
    )
    .expect("bind manifest")
}

async fn read_txn<'b, 'm>(
    backend: &'b DeterministicBackend,
    manifest: &'m IndexManifest,
) -> ReadLogicalTxn<'m, <DeterministicBackend as Backend>::ReadTxn<'b>> {
    let raw = backend.begin_read().await.expect("begin read");
    ReadLogicalTxn::for_index(raw, manifest).expect("bind manifest")
}

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

fn header(level: u32) -> PersistentValue {
    PersistentValue::PartitionHeader(
        PartitionHeader::new(level, 0, 0, PartitionState::Ready).expect("header"),
    )
}

fn edge(child: PartitionKey, centroid: f32) -> PersistentValue {
    PersistentValue::ChildEntry(ChildEntry::new(child, vec![centroid]))
}

/// Commits one hand-built stable topology. The split state machines (#10) do
/// not exist yet, so tests install grown shapes directly; only the state the
/// routing contract reads is seeded.
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

/// Installs the tree's Tree Manifest and initial leaf root so fixtures can
/// grow the root shape from a committed empty root.
async fn create_committed_tree(
    backend: &DeterministicBackend,
    manifest: &IndexManifest,
    key: &TreeKey,
) {
    let mut txn = write_txn(backend, manifest).await;
    tree_manifest::create_tree(&mut txn, key, 0)
        .await
        .expect("create tree");
    txn.commit().await.expect("commit tree");
}

/// Seeds the grown root shape: root PK 1 at level 2 with leaf children PK 2
/// (centroid 0.0) and PK 3 (centroid 10.0).
async fn seed_grown_root(backend: &DeterministicBackend, manifest: &IndexManifest, key: &TreeKey) {
    create_committed_tree(backend, manifest, key).await;
    write_topology(
        backend,
        manifest,
        vec![
            (header_key_at(key, pk(1)), header(2)),
            (header_key_at(key, pk(2)), header(1)),
            (header_key_at(key, pk(3)), header(1)),
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
        (header_key_at(key, pk(1)), header(3)),
        (header_key_at(key, pk(4)), header(2)),
        (header_key_at(key, pk(6)), header(2)),
        (header_key_at(key, pk(5)), header(1)),
        (header_key_at(key, pk(7)), header(1)),
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
async fn descent_decrements_levels_through_ordinary_internal_partitions() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();
    let key = tree_key(1);
    create_committed_tree(&backend, &manifest, &key).await;
    write_topology(
        &backend,
        &manifest,
        vec![
            (header_key_at(&key, pk(1)), header(3)),
            (header_key_at(&key, pk(2)), header(2)),
            (header_key_at(&key, pk(3)), header(2)),
            (header_key_at(&key, pk(4)), header(1)),
            (header_key_at(&key, pk(5)), header(1)),
            (header_key_at(&key, pk(6)), header(1)),
            (header_key_at(&key, pk(7)), header(1)),
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
            (header_key_at(&key, pk(1)), header(2)),
            // A level-2 child of a level-2 root is not a leaf and cannot be
            // descended into; the exact one-level step is violated.
            (header_key_at(&key, pk(2)), header(2)),
            (header_key_at(&key, pk(3)), header(1)),
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
            (header_key_at(&key, pk(1)), header(2)),
            (header_key_at(&key, pk(2)), header(1)),
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
async fn a_stable_internal_partition_must_have_exactly_two_children() {
    let backend = DeterministicBackend::default();
    let manifest = manifest();

    // One child: fanout below two.
    let one_child = tree_key(1);
    create_committed_tree(&backend, &manifest, &one_child).await;
    write_topology(
        &backend,
        &manifest,
        vec![
            (header_key_at(&one_child, pk(1)), header(2)),
            (header_key_at(&one_child, pk(2)), header(1)),
            (edge_key_at(&one_child, pk(1), pk(2)), edge(pk(2), 0.0)),
        ],
    )
    .await;
    let error = route_leaf(&mut read_txn(&backend, &manifest).await, &one_child, &[1.0])
        .await
        .expect_err("one child");
    assert_eq!(error.kind(), ErrorKind::Corruption);

    // Three children: fanout above two, detected from a bounded scan.
    let three_children = tree_key(2);
    create_committed_tree(&backend, &manifest, &three_children).await;
    write_topology(
        &backend,
        &manifest,
        vec![
            (header_key_at(&three_children, pk(1)), header(2)),
            (header_key_at(&three_children, pk(2)), header(1)),
            (header_key_at(&three_children, pk(3)), header(1)),
            (header_key_at(&three_children, pk(4)), header(1)),
            (edge_key_at(&three_children, pk(1), pk(2)), edge(pk(2), 0.0)),
            (edge_key_at(&three_children, pk(1), pk(3)), edge(pk(3), 5.0)),
            (
                edge_key_at(&three_children, pk(1), pk(4)),
                edge(pk(4), 10.0),
            ),
        ],
    )
    .await;
    let error = route_leaf(
        &mut read_txn(&backend, &manifest).await,
        &three_children,
        &[1.0],
    )
    .await
    .expect_err("three children");
    assert_eq!(error.kind(), ErrorKind::Corruption);
}

#[tokio::test]
async fn a_non_ready_partition_is_corruption_until_maintenance_exists() {
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
    write_topology(
        &backend,
        &manifest,
        vec![(
            header_key_at(&key, pk(1)),
            PersistentValue::PartitionHeader(
                PartitionHeader::new(1, 0, 0, PartitionState::Splitting).expect("header"),
            ),
        )],
    )
    .await;

    let error = route_leaf(&mut read_txn(&backend, &manifest).await, &key, &[1.0])
        .await
        .expect_err("non-ready state");
    assert_eq!(error.kind(), ErrorKind::Corruption);
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
