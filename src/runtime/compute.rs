//! Cancellation and admission ownership for bounded foreground CPU tasks.

use super::{CancellationToken, check_control};
use crate::api::{Error, ErrorKind, Result};

/// Keeps heavy CPU work off Tokio and cancels it when its owning future drops.
/// Existing foreground admission bounds the number of these owned tasks.
pub(super) async fn controlled_compute<T: Send + 'static>(
    options: crate::api::OperationOptions,
    keep_alive: impl Send + 'static,
    work: impl FnOnce(&crate::api::OperationOptions) -> Result<T> + Send + 'static,
) -> Result<T> {
    let cancellation = options
        .cancellation()
        .map(CancellationToken::child_token)
        .unwrap_or_default();
    let _guard = cancellation.clone().drop_guard();
    let options = options.with_cancellation(cancellation);
    tokio::task::spawn_blocking(move || {
        // Retain admission even after the owning async future is cancelled.
        let _keep_alive = keep_alive;
        check_control(&options)?;
        work(&options)
    })
    .await
    .map_err(|error| Error::with_source(ErrorKind::Other, error))?
}
