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

use std::time::Duration;

use crate::api::{IndexConfig, PartitionKey, Result};
use crate::runtime::RetryPolicy;
use crate::runtime::reads;
use crate::storage::backend::Backend;
use crate::storage::keys::TreeKey;
use crate::storage::topology;
use crate::storage::values::{IndexManifest, PartitionHeader, PartitionState, PartitionTransition};

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

/// Timing and retry policy for one maintenance step.
pub(crate) struct StepPolicy<'a> {
    pub(crate) now_unix_millis: u64,
    /// Applies to rediscovery; an already progressing worker has no age gate.
    pub(crate) recovery_timeout: Option<Duration>,
    pub(crate) retry: &'a RetryPolicy,
}

/// Unknown time permits recovery; a known future state waits for the clock.
fn recovery_due(state: PartitionTransition, now: u64, timeout: Duration) -> bool {
    if state.state() == PartitionState::Ready {
        return true;
    }
    let started = state.started_at_unix_millis();
    now == 0
        || started == 0
        || now
            .checked_sub(started)
            .is_some_and(|age| Duration::from_millis(age) >= timeout)
}

/// Reads one authority pair and runs the matching state machine directly.
pub(crate) async fn advance<B: Backend>(
    backend: &B,
    manifest: &IndexManifest,
    tree_key: &TreeKey,
    partition: PartitionKey,
    policy: StepPolicy<'_>,
) -> Result<Advance> {
    let (read, pair) = reads::open_authority_read(backend, manifest, tree_key, partition).await?;
    drop(read);
    let Some(authority) = pair else {
        return Ok(Advance::Idle);
    };
    let header = authority.0;
    if policy
        .recovery_timeout
        .is_some_and(|timeout| !recovery_due(authority.1, policy.now_unix_millis, timeout))
    {
        return Ok(Advance::Idle);
    }
    let started_at_unix_millis = policy.now_unix_millis;
    let retry = policy.retry;

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
    fn recovery_uses_state_age_with_explicit_clock_boundaries() {
        let timeout = Duration::from_millis(100);
        for (now, started, due) in [
            (1099, 1000, false),
            (1100, 1000, true),
            (1101, 1000, true),
            (999, 1000, false),
            (0, 1000, true),
            (1000, 0, true),
        ] {
            for state in [
                PartitionTransition::Splitting {
                    left: PartitionKey::new(2).unwrap(),
                    right: PartitionKey::new(3).unwrap(),
                    started_at_unix_millis: started,
                },
                PartitionTransition::DrainingSplit {
                    left: PartitionKey::new(2).unwrap(),
                    right: PartitionKey::new(3).unwrap(),
                    started_at_unix_millis: started,
                },
                PartitionTransition::Merging {
                    started_at_unix_millis: started,
                },
            ] {
                assert_eq!(
                    recovery_due(state, now, timeout),
                    due,
                    "{state:?}, now={now}"
                );
            }
        }
        assert!(recovery_due(
            PartitionTransition::Ready {
                started_at_unix_millis: 1000
            },
            1000,
            timeout,
        ));
        assert!(!recovery_due(
            PartitionTransition::Merging {
                started_at_unix_millis: 1000
            },
            1000,
            Duration::from_nanos(1),
        ));
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
