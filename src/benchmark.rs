//! Untimed benchmark diagnostics, excluded from production builds.

/// Resolves truth membership in one snapshot of a caller-frozen graph fixture.
/// This bounded, untimed diagnostic is deliberately outside ordinary search.
pub(crate) async fn truth_leaves<B: crate::storage::backend::Backend>(
    backend: &B,
    index: crate::api::LogicalIndexId,
    ids: &[bytes::Bytes],
) -> crate::api::Result<Vec<(crate::api::PartitionKey, u64)>> {
    use crate::api::{Error, ErrorKind};
    use crate::storage::backend::ReadOps;
    use crate::storage::keys::{self, LogicalKey};
    use crate::storage::values::{PersistentValue, ValueCodec};
    if ids.len() > 65_536 {
        return Err(Error::new(ErrorKind::LimitExceeded));
    }
    let corrupt = || Error::new(ErrorKind::Corruption);
    let mut txn = backend.begin_read().await?;
    let raw = txn
        .get(keys::manifest_key(index).into())
        .await?
        .ok_or_else(corrupt)?;
    let PersistentValue::IndexManifest(manifest) =
        ValueCodec::bootstrap().decode(&LogicalKey::Manifest(index), raw)?
    else {
        return Err(corrupt());
    };
    if manifest.config().local_graph().is_none() || !manifest.config().tree_key_fields().is_empty()
    {
        return Err(Error::invalid_argument());
    }
    let codec = ValueCodec::for_index(&manifest);
    let mut counts = std::collections::BTreeMap::new();
    let mut result = Vec::with_capacity(ids.len());
    for id in ids {
        let raw = txn
            .get(keys::location_key(index, id)?.into())
            .await?
            .ok_or_else(corrupt)?;
        let PersistentValue::RecordLocation(location) = codec.decode(
            &LogicalKey::Location {
                index,
                id: id.clone(),
            },
            raw,
        )?
        else {
            return Err(corrupt());
        };
        let leaf = location.leaf();
        let count = if let Some(count) = counts.get(&leaf) {
            *count
        } else {
            let raw = txn
                .get(keys::header_key(index, location.tree_key(), leaf).into())
                .await?
                .ok_or_else(corrupt)?;
            let PersistentValue::PartitionHeader(header) = codec.decode(
                &LogicalKey::Header {
                    index,
                    tree_key: location.tree_key().clone(),
                    partition: leaf,
                },
                raw,
            )?
            else {
                return Err(corrupt());
            };
            if header.level() != 1 {
                return Err(corrupt());
            }
            counts.insert(leaf, header.entry_count());
            header.entry_count()
        };
        result.push((leaf, u64::from(count)));
    }
    Ok(result)
}

/// Untimed quality diagnostics for one graph-index query. Retains only truth IDs
/// and selected leaf identities, never a second copy of the candidate pool.
#[derive(Clone, Debug)]
pub struct SearchDiagnostics {
    pub(crate) truth: std::collections::HashSet<bytes::Bytes>,
    /// True neighbors present in scanned leaf bodies before RaBitQ selection.
    pub neighbors_in_leaves: usize,
    /// True neighbors surviving per-leaf overlap selection.
    pub neighbors_after_local_selection: usize,
    /// True neighbors admitted to exact reranking.
    pub neighbors_reranked: usize,
    /// Actual scanned leaves; used for global centroid top-beam coverage.
    pub leaves: Vec<crate::api::PartitionKey>,
}

impl SearchDiagnostics {
    pub(crate) fn new(truth: Vec<bytes::Bytes>) -> Self {
        Self {
            truth: truth.into_iter().collect(),
            neighbors_in_leaves: 0,
            neighbors_after_local_selection: 0,
            neighbors_reranked: 0,
            leaves: Vec::new(),
        }
    }
}

/// Streams a canonical leaf-membership/centroid fingerprint.
/// Partition identities and raw membership bytes are included;
/// internal grouping and graph parameters are deliberately excluded.
pub(crate) async fn leaf_layout_fingerprint<B: crate::storage::backend::Backend>(
    backend: &B,
    index: crate::api::LogicalIndexId,
) -> crate::api::Result<[u8; 32]> {
    use crate::api::{Error, ErrorKind, PartitionKey};
    use crate::storage::backend::{ReadOps, ScanLimits};
    use crate::storage::keys::{self, KeyRange, LogicalKey, TreeKey};
    use crate::storage::values::{PersistentValue, ValueCodec};
    use sha2::{Digest, Sha256};
    let corrupt = || Error::new(ErrorKind::Corruption);
    let mut txn = backend.begin_read().await?;
    let raw = txn
        .get(keys::manifest_key(index).into())
        .await?
        .ok_or_else(corrupt)?;
    let PersistentValue::IndexManifest(manifest) =
        ValueCodec::bootstrap().decode(&LogicalKey::Manifest(index), raw)?
    else {
        return Err(corrupt());
    };
    if manifest.config().local_graph().is_none() || !manifest.config().tree_key_fields().is_empty()
    {
        return Err(Error::invalid_argument());
    }
    let codec = ValueCodec::for_index(&manifest);
    let tree_key = TreeKey::encode(&[], &[])?;
    let raw = txn
        .get(keys::tree_manifest_key(index, &tree_key).into())
        .await?
        .ok_or_else(corrupt)?;
    let PersistentValue::TreeManifest(tree) = codec.decode(
        &LogicalKey::TreeManifest {
            index,
            tree_key: tree_key.clone(),
        },
        raw,
    )?
    else {
        return Err(corrupt());
    };
    if tree.partition_key_high_water().get() > 262_144 {
        return Err(Error::new(ErrorKind::LimitExceeded));
    }
    let mut hash = Sha256::new();
    hash.update(b"ktann-benchmark-leaf-layout-v1");
    for id in 1..=tree.partition_key_high_water().get() {
        let partition = PartitionKey::new(id)?;
        let Some(raw) = txn
            .get(keys::header_key(index, &tree_key, partition).into())
            .await?
        else {
            continue;
        };
        let PersistentValue::PartitionHeader(header) = codec.decode(
            &LogicalKey::Header {
                index,
                tree_key: tree_key.clone(),
                partition,
            },
            raw,
        )?
        else {
            return Err(corrupt());
        };
        if header.level() != 1 {
            continue;
        }
        hash.update(id.to_le_bytes());
        hash.update(header.entry_count().to_le_bytes());
        let centroid = txn
            .get(keys::centroid_key(index, &tree_key, partition).into())
            .await?
            .ok_or_else(corrupt)?;
        hash.update((centroid.len() as u64).to_le_bytes());
        hash.update(centroid);
        let mut range = keys::leaf_entry_range(index, &tree_key, partition);
        loop {
            let page = txn
                .scan(
                    &range,
                    ScanLimits {
                        item_limit: 256,
                        byte_limit: 1 << 20,
                    },
                )
                .await?;
            for item in page.items() {
                let LogicalKey::LeafEntry { id, .. } = keys::decode_key(&[], item.key())? else {
                    return Err(corrupt());
                };
                hash.update((id.len() as u64).to_le_bytes());
                hash.update(id);
                hash.update((item.value().len() as u64).to_le_bytes());
                hash.update(item.value());
            }
            let Some(next) = page.next_start() else {
                break;
            };
            range = KeyRange::new(next.to_vec(), range.end().to_vec());
        }
    }
    Ok(hash.finalize().into())
}
