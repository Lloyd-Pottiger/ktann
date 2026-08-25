//! Process CPU and resident-memory observations for isolated scenario workers.

use std::mem::MaybeUninit;

/// One cumulative process-resource observation.
#[derive(Clone, Copy, Debug)]
pub struct ResourceSnapshot {
    /// Cumulative user plus system CPU consumed by the worker process.
    cpu_seconds: f64,
    /// Process lifetime resident-set high-water mark in normalized bytes.
    peak_rss_bytes: u64,
}

impl ResourceSnapshot {
    /// Reads cumulative user/system CPU and the process peak resident set.
    ///
    /// # Errors
    ///
    /// Returns an error when the operating system rejects `getrusage`.
    pub fn capture() -> Result<Self, String> {
        getrusage()
    }

    /// Returns CPU consumed between two snapshots.
    #[must_use]
    pub fn cpu_seconds_since(self, earlier: Self) -> f64 {
        (self.cpu_seconds - earlier.cpu_seconds).max(0.0)
    }

    /// Returns peak resident bytes for this isolated worker process.
    #[must_use]
    pub const fn peak_rss_bytes(self) -> u64 {
        self.peak_rss_bytes
    }
}

/// Calls the Unix process-accounting API behind one documented safe wrapper.
#[expect(
    unsafe_code,
    reason = "getrusage is the portable Unix API for process CPU and peak RSS"
)]
fn getrusage() -> Result<ResourceSnapshot, String> {
    let mut usage = MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: `usage` points to writable storage for one `libc::rusage`; a
    // successful call fully initializes it before `assume_init`.
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, usage.as_mut_ptr()) };
    if status != 0 {
        return Err(format!(
            "getrusage failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    // SAFETY: the successful getrusage call above initialized every field.
    let usage = unsafe { usage.assume_init() };
    let user = timeval_seconds(usage.ru_utime);
    let system = timeval_seconds(usage.ru_stime);
    let raw_rss = u64::try_from(usage.ru_maxrss).unwrap_or_default();
    // Darwin reports bytes; Linux and the other supported CI Unix targets
    // report KiB. Each worker runs one scenario, so this peak is isolated even
    // though setup allocations necessarily contribute to process high-water.
    let peak_rss_bytes = if cfg!(target_os = "macos") {
        raw_rss
    } else {
        raw_rss.saturating_mul(1_024)
    };
    Ok(ResourceSnapshot {
        cpu_seconds: user + system,
        peak_rss_bytes,
    })
}

/// Converts the libc seconds/microseconds pair without integer truncation.
fn timeval_seconds(value: libc::timeval) -> f64 {
    value.tv_sec as f64 + value.tv_usec as f64 / 1_000_000.0
}
