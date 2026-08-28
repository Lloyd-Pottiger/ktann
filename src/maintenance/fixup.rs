//! Actionable Structure Maintenance discovery and execution dispatch.
//!
//! A partition is worth offering only when its committed Header can advance:
//! an oversized `Ready` partition can split, an undersized non-root `Ready`
//! partition can begin a merge, and durable split/merge source states can
//! resume. Healthy `Ready` partitions and `ReceivingSplit` targets cannot
//! advance and stay out of the process-local queue.
//!
//! Execution reads the Header and State once, then dispatches directly to the
//! owning state machine. This keeps the queue a lossy rediscovery hint while
//! avoiding separate split and merge preflight transactions.

use crate::api::{IndexConfig, PartitionKey, Result};
use crate::runtime::lifecycle::RetryPolicy;
use crate::runtime::reads;
use crate::storage::backend::Backend;
use crate::storage::keys::TreeKey;
use crate::storage::topology;
use crate::storage::values::{IndexManifest, PartitionHeader, PartitionState};

use super::{merge, split};

/// The state machine selected by one committed Partition Header.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Action {
    Split,
    Merge,
}

/// One directly dispatched Fixup step.
pub(crate) enum Advance {
    /// The partition no longer has actionable work.
    Idle,
    /// The split state machine ran one bounded step.
    Split(Result<split::Advance>),
    /// The merge state machine ran one bounded step.
    Merge(Result<merge::Advance>),
}

/// Returns whether one committed Header identifies actionable maintenance.
///
/// The stable root never merges. Intermediate source states remain eligible
/// regardless of their current count so queue loss or worker retirement can
/// be recovered by a later relevant access.
pub(crate) fn is_actionable(
    config: &IndexConfig,
    partition: PartitionKey,
    header: PartitionHeader,
) -> bool {
    action(config, partition, header).is_some()
}

/// Reads one authority pair and runs the matching state machine directly.
pub(crate) async fn advance<B: Backend>(
    backend: &B,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    partition: PartitionKey,
    started_at_unix_millis: u64,
    retry: &RetryPolicy,
) -> Result<Advance> {
    let mut read = reads::open_validated_read(backend, manifest).await?;
    let pair =
        topology::read_authority_pair(&mut read, manifest.logical_index_id(), tree_key, partition)
            .await?;
    drop(read);
    let Some(authority) = pair else {
        return Ok(Advance::Idle);
    };
    let header = authority.0;

    match action(manifest.config(), partition, header) {
        Some(Action::Split) => Ok(Advance::Split(
            split::advance_observed(
                backend,
                manifest,
                tree_key,
                partition,
                started_at_unix_millis,
                retry,
                authority,
            )
            .await,
        )),
        Some(Action::Merge) => Ok(Advance::Merge(
            merge::advance_observed(
                backend,
                manifest,
                tree_key,
                partition,
                started_at_unix_millis,
                retry,
                authority,
            )
            .await,
        )),
        None => Ok(Advance::Idle),
    }
}

/// Classifies one Header without reading any additional persistent state.
fn action(
    config: &IndexConfig,
    partition: PartitionKey,
    header: PartitionHeader,
) -> Option<Action> {
    match header.state() {
        PartitionState::Ready if header.entry_count() > config.max_partition_entries() => {
            Some(Action::Split)
        }
        PartitionState::Ready
            if partition != topology::root_partition()
                && header.entry_count() < config.min_partition_entries() =>
        {
            Some(Action::Merge)
        }
        PartitionState::Splitting | PartitionState::DrainingSplit => Some(Action::Split),
        PartitionState::Merging => Some(Action::Merge),
        PartitionState::Ready | PartitionState::ReceivingSplit => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::Metric;

    fn config() -> IndexConfig {
        IndexConfig::new(1, Metric::L2)
            .expect("config")
            .with_partition_entries(2, 8)
            .expect("partition entries")
    }

    fn header(count: u32, state: PartitionState) -> PartitionHeader {
        PartitionHeader::new(1, count, 0, state).expect("header")
    }

    #[test]
    fn eligibility_excludes_stable_non_actionable_partitions() {
        let config = config();
        let root = topology::root_partition();
        let child = PartitionKey::new(2).expect("child");

        assert!(!is_actionable(
            &config,
            root,
            header(0, PartitionState::Ready)
        ));
        assert!(!is_actionable(
            &config,
            root,
            header(8, PartitionState::Ready)
        ));
        assert!(!is_actionable(
            &config,
            child,
            header(9, PartitionState::ReceivingSplit),
        ));
    }

    #[test]
    fn eligibility_keeps_threshold_crossings_and_source_states() {
        let config = config();
        let root = topology::root_partition();
        let child = PartitionKey::new(2).expect("child");

        assert!(is_actionable(
            &config,
            root,
            header(9, PartitionState::Ready)
        ));
        assert!(is_actionable(
            &config,
            child,
            header(1, PartitionState::Ready)
        ));
        for state in [
            PartitionState::Splitting,
            PartitionState::DrainingSplit,
            PartitionState::Merging,
        ] {
            assert!(is_actionable(&config, child, header(4, state)));
        }
    }
}
