//! The independent resource limits of one verification (ADR 0019).
//!
//! Every visited logical object, every resident buffer byte, and every
//! reported issue charges this ledger. Any exhausted limit stops the audit;
//! the caller maps that to `complete: false`. All arithmetic is checked or
//! saturating.

use crate::api::VerifyOptions;

/// The bounded resource ledger of one audit.
pub(super) struct Limits {
    objects_remaining: u64,
    issues_remaining: usize,
    memory_used: u64,
    memory_limit: u64,
}

impl Limits {
    pub(super) fn new(options: &VerifyOptions) -> Self {
        Self {
            objects_remaining: options.object_limit(),
            issues_remaining: options.issue_limit(),
            memory_used: 0,
            memory_limit: options.memory_limit_bytes(),
        }
    }

    /// Charges one visited key–value pair; `false` exhausts the audit.
    pub(super) fn charge_object(&mut self) -> bool {
        match self.objects_remaining.checked_sub(1) {
            Some(remaining) => {
                self.objects_remaining = remaining;
                true
            }
            None => false,
        }
    }

    /// Reserves `bytes` of resident state; `false` exceeds the memory limit.
    pub(super) fn charge_memory(&mut self, bytes: u64) -> bool {
        match self.memory_used.checked_add(bytes) {
            Some(used) if used <= self.memory_limit => {
                self.memory_used = used;
                true
            }
            _ => false,
        }
    }

    /// Releases previously reserved resident state.
    pub(super) fn release_memory(&mut self, bytes: u64) {
        self.memory_used = self.memory_used.saturating_sub(bytes);
    }

    /// Reserves one reported-issue slot; `false` exhausts the audit.
    pub(super) fn take_issue_slot(&mut self) -> bool {
        match self.issues_remaining.checked_sub(1) {
            Some(remaining) => {
                self.issues_remaining = remaining;
                true
            }
            None => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn limits(objects: u64, issues: usize, memory: u64) -> Limits {
        Limits::new(
            &VerifyOptions::default()
                .with_object_limit(objects)
                .expect("object limit")
                .with_issue_limit(issues)
                .expect("issue limit")
                .with_memory_limit_bytes(memory)
                .expect("memory limit"),
        )
    }

    #[test]
    fn every_limit_exhausts_exactly() {
        let mut limits = limits(2, 1, 8);
        assert!(limits.charge_object());
        assert!(limits.charge_object());
        assert!(!limits.charge_object());

        assert!(limits.take_issue_slot());
        assert!(!limits.take_issue_slot());

        assert!(limits.charge_memory(8));
        assert!(!limits.charge_memory(1));
        limits.release_memory(8);
        assert!(limits.charge_memory(8));
        // Saturating release never underflows.
        limits.release_memory(u64::MAX);
        assert!(limits.charge_memory(8));
    }
}
