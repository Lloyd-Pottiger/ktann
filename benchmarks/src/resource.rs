//! Process CPU and resident-memory observations for isolated scenario workers.

use std::mem::MaybeUninit;

/// One cumulative process-resource observation.
#[derive(Clone, Copy, Debug)]
pub struct ResourceSnapshot {
    /// Cumulative user plus system CPU consumed by the worker process.
    cpu_seconds: f64,
    /// Process lifetime resident-set high-water mark in normalized bytes.
    peak_rss_bytes: u64,
    /// Physical bytes charged to this process by the operating system.
    disk_io_bytes: Option<(u64, u64)>,
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

    /// Returns timed physical read/write bytes where the OS exposes them.
    #[must_use]
    pub fn disk_io_bytes_since(self, earlier: Self) -> Option<(u64, u64)> {
        self.disk_io_bytes
            .zip(earlier.disk_io_bytes)
            .map(|(after, before)| {
                (
                    after.0.saturating_sub(before.0),
                    after.1.saturating_sub(before.1),
                )
            })
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
        disk_io_bytes: disk_io_bytes()?,
    })
}

/// Converts the libc seconds/microseconds pair without integer truncation.
fn timeval_seconds(value: libc::timeval) -> f64 {
    value.tv_sec as f64 + value.tv_usec as f64 / 1_000_000.0
}

/// Darwin process accounting distinguishes physical IO from logical reads.
#[cfg(target_os = "macos")]
#[expect(
    unsafe_code,
    reason = "proc_pid_rusage is Darwin's physical IO accounting API"
)]
fn disk_io_bytes() -> Result<Option<(u64, u64)>, String> {
    let mut usage = MaybeUninit::<libc::rusage_info_v2>::zeroed();
    // SAFETY: the flavor selects rusage_info_v2, matching the writable buffer.
    let status = unsafe {
        libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V2,
            usage.as_mut_ptr().cast(),
        )
    };
    if status != 0 {
        return Err(format!(
            "proc_pid_rusage failed: {}",
            std::io::Error::last_os_error()
        ));
    }
    // SAFETY: a successful call initialized the flavor-specific buffer.
    let usage = unsafe { usage.assume_init() };
    Ok(Some((
        usage.ri_diskio_bytesread,
        usage.ri_diskio_byteswritten,
    )))
}

/// Linux procfs reports storage-layer IO, excluding page-cache hits.
#[cfg(target_os = "linux")]
fn disk_io_bytes() -> Result<Option<(u64, u64)>, String> {
    let contents = std::fs::read_to_string("/proc/self/io")
        .map_err(|error| format!("read process IO: {error}"))?;
    let value = |key: &str| -> Result<u64, String> {
        contents
            .lines()
            .find_map(|line| line.strip_prefix(key))
            .ok_or_else(|| format!("missing process IO field {key}"))?
            .trim()
            .parse()
            .map_err(|error| format!("parse process IO: {error}"))
    };
    Ok(Some((value("read_bytes:")?, value("write_bytes:")?)))
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn disk_io_bytes() -> Result<Option<(u64, u64)>, String> {
    Ok(None)
}
