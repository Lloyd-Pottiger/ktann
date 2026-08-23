//! The public handle to one Active Logical Index.

use std::fmt;
use std::sync::Arc;

use bytes::Bytes;

use crate::maintenance::mutation;
use crate::observe::labels::Operation;
use crate::runtime::{OperationContext, RuntimeInner};
use crate::runtime::{lifecycle, reads, search, verify};
use crate::storage::backend::Backend;
use crate::storage::values::{IndexLifecycle, IndexManifest};

use super::{
    Error, ErrorKind, GetOptions, ImportOptions, ImportSession, IndexConfig, IndexName,
    LogicalIndexId, Mutation, MutationOutcome, OperationOptions, Record, Result, SearchOutcome,
    SearchRequest, StoredRecord, UpsertResult, VerifyOptions, VerifyReport, validate_id,
    validate_ids, validate_mutations,
};

/// A cheap cloneable handle to one Active Logical Index.
///
/// The handle retains the owning Runtime and the exact Index Manifest opened
/// by [`Runtime::create_index`](crate::runtime::Runtime::create_index) or
/// [`Runtime::open_index`](crate::runtime::Runtime::open_index). It never
/// retargets to a newer Logical Index created under the same Index Name.
pub struct Index<B: Backend> {
    runtime: Arc<RuntimeInner<B>>,
    name: IndexName,
    manifest: Arc<IndexManifest>,
}

impl<B: Backend> Index<B> {
    pub(crate) fn new(
        runtime: Arc<RuntimeInner<B>>,
        name: IndexName,
        manifest: IndexManifest,
    ) -> Result<Self> {
        if manifest.lifecycle() != IndexLifecycle::Active {
            return Err(Error::new(ErrorKind::IndexDropping));
        }
        Ok(Self {
            runtime,
            name,
            manifest: Arc::new(manifest),
        })
    }

    /// Returns the caller-chosen Index Name used to open this handle.
    #[must_use]
    pub fn name(&self) -> &IndexName {
        &self.name
    }

    /// Returns the never-reused Logical Index ID.
    #[must_use]
    pub fn logical_index_id(&self) -> LogicalIndexId {
        self.manifest.logical_index_id()
    }

    /// Returns the immutable Index Manifest configuration.
    #[must_use]
    pub fn config(&self) -> &IndexConfig {
        self.manifest.config()
    }

    /// Inserts one Vector Record only when its Record ID is absent.
    ///
    /// The record is validated against this index's immutable configuration
    /// before any storage work. The atomic commit installs the Vector Record,
    /// its Record Location, the routed Leaf Entry, and every affected exact
    /// count and synopsis together; an existing Record ID fails with
    /// [`ErrorKind::RecordAlreadyExists`]. A retryable topology or write
    /// conflict restarts the whole operation from a fresh snapshot under the
    /// Runtime's bounded contention policy, and a commit of unknown outcome is
    /// returned as [`ErrorKind::CommitOutcomeUnknown`] without a retry.
    pub async fn insert(&self, record: Record) -> Result<()> {
        self.insert_with_control(record, OperationOptions::default())
            .await
    }

    /// Inserts one Vector Record with explicit operation control.
    pub async fn insert_with_control(
        &self,
        record: Record,
        operation_options: OperationOptions,
    ) -> Result<()> {
        let mut mutations = vec![Mutation::Insert(record)];
        self.validate(&mut mutations)?;
        let outcomes = self
            .run_mutations(Operation::Insert, mutations, operation_options)
            .await?;
        match outcomes.first() {
            Some(MutationOutcome::Inserted) => Ok(()),
            _ => Err(Error::new(ErrorKind::Backend)),
        }
    }

    /// Creates or fully replaces one Vector Record, last-write-wins.
    ///
    /// The returned [`UpsertResult`] reports whether an existing Vector Record
    /// was replaced. Replacement is an atomic move: a changed Tree Key or a
    /// newly routed leaf relocates the Leaf Entry in the same commit, `None`
    /// payload deletes an old payload, and `Some(empty)` stores an empty
    /// payload, which stays distinct from absence. Validation, retry, and
    /// commit-outcome behavior match [`insert`](Self::insert).
    pub async fn upsert(&self, record: Record) -> Result<UpsertResult> {
        self.upsert_with_control(record, OperationOptions::default())
            .await
    }

    /// Creates or fully replaces one Vector Record with explicit operation
    /// control.
    pub async fn upsert_with_control(
        &self,
        record: Record,
        operation_options: OperationOptions,
    ) -> Result<UpsertResult> {
        let mut mutations = vec![Mutation::Upsert(record)];
        self.validate(&mut mutations)?;
        let outcomes = self
            .run_mutations(Operation::Upsert, mutations, operation_options)
            .await?;
        match outcomes.first() {
            Some(MutationOutcome::Upserted { replaced: true }) => Ok(UpsertResult::Replaced),
            Some(MutationOutcome::Upserted { replaced: false }) => Ok(UpsertResult::Created),
            _ => Err(Error::new(ErrorKind::Backend)),
        }
    }

    /// Idempotently deletes one Vector Record by Record ID.
    ///
    /// Returns whether a Vector Record existed. Delete follows the exact
    /// stored Record Location and atomically removes the record, location,
    /// Leaf Entry, and any Opaque Payload. Validation, retry, and
    /// commit-outcome behavior match [`insert`](Self::insert).
    pub async fn delete(&self, id: Bytes) -> Result<bool> {
        self.delete_with_control(id, OperationOptions::default())
            .await
    }

    /// Deletes one Vector Record with explicit operation control.
    pub async fn delete_with_control(
        &self,
        id: Bytes,
        operation_options: OperationOptions,
    ) -> Result<bool> {
        let mut mutations = vec![Mutation::Delete(id)];
        self.validate(&mut mutations)?;
        let outcomes = self
            .run_mutations(Operation::Delete, mutations, operation_options)
            .await?;
        match outcomes.first() {
            Some(MutationOutcome::Deleted { existed }) => Ok(*existed),
            _ => Err(Error::new(ErrorKind::Backend)),
        }
    }

    /// Applies one atomic mutation batch.
    ///
    /// A nonempty batch is one atomic transaction and is never split: every
    /// mutation commits or none does. Outcomes correspond to inputs in order.
    /// Validation of the whole batch completes before storage work; duplicate
    /// Record IDs are invalid. An empty batch succeeds with an empty result.
    /// Any item failure returns one operation error carrying the input
    /// position and no partial outcomes. Retry and commit-outcome behavior
    /// match [`insert`](Self::insert).
    pub async fn batch_mutate(&self, mutations: Vec<Mutation>) -> Result<Vec<MutationOutcome>> {
        self.batch_mutate_with_control(mutations, OperationOptions::default())
            .await
    }

    /// Applies one atomic mutation batch with explicit operation control.
    pub async fn batch_mutate_with_control(
        &self,
        mut mutations: Vec<Mutation>,
        operation_options: OperationOptions,
    ) -> Result<Vec<MutationOutcome>> {
        self.validate(&mut mutations)?;
        if mutations.is_empty() {
            return Ok(Vec::new());
        }
        self.run_mutations(Operation::BatchMutate, mutations, operation_options)
            .await
    }

    /// Opens a bounded Import Session on this index.
    ///
    /// The session admits ordinary atomic mutation batches under bounded
    /// concurrency; see [`ImportSession`] for the admission, ordering,
    /// cancellation, and shutdown contract. `options.in_flight_batches()`
    /// overrides the Runtime's configured in-flight batch limit.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::RuntimeClosed`] when the Runtime is shutting down
    /// or has shut down.
    pub fn import_session(&self, options: ImportOptions) -> Result<ImportSession<B>> {
        if !self.runtime.is_accepting() {
            return Err(Error::new(ErrorKind::RuntimeClosed));
        }
        let in_flight_batches = options
            .in_flight_batches()
            .unwrap_or_else(|| self.runtime.config().import_in_flight_batches());
        Ok(ImportSession::new(self.clone(), in_flight_batches))
    }

    /// Runs one bounded approximate search over one consistent snapshot.
    ///
    /// The request is validated against this index's immutable configuration
    /// before admission: the vector dimension must match, the Filter Predicate
    /// must be schema-correct, and the effective Search Budgets resolve from
    /// the Runtime defaults and the request overrides within hard caps, with
    /// the exact-rerank budget at least `k`. Manifest validation, Tree Key
    /// enumeration, traversal, filtering, Vector Record loading, and exact
    /// reranking then read from one consistent backend snapshot, and partition
    /// bodies are served by the Runtime's snapshot-validated Partition Cache.
    ///
    /// The outcome reports ordered exact-distance hits, actual budget usage,
    /// every budget dimension that prevented eligible work, and per-leaf
    /// RaBitQ overlap truncation. Success may return fewer than `k` hits and
    /// deliberately makes no exact-global-top-k, completeness, continuation,
    /// or monotonic-across-budgets guarantee: resubmitting the same request
    /// with larger budgets asks for more work without promising a superset of
    /// earlier hits. Cancellation and deadline apply to the whole operation;
    /// search never commits, so they surface as errors, never partial state.
    pub async fn search(&self, request: SearchRequest) -> Result<SearchOutcome> {
        self.search_with_control(request, OperationOptions::default())
            .await
    }

    /// Runs one bounded approximate search with explicit operation control.
    pub async fn search_with_control(
        &self,
        request: SearchRequest,
        operation_options: OperationOptions,
    ) -> Result<SearchOutcome> {
        let config = self.runtime.config();
        let prepared = search::PreparedSearch::new(
            &self.manifest,
            config.default_search_budgets(),
            config.tree_key_scan_ranges(),
            request,
        )?;
        let manifest = Arc::clone(&self.manifest);
        let cache = self.runtime.partition_cache();
        let (outcome, maintenance) = self
            .run_foreground(
                Operation::Search,
                operation_options,
                move |mut context| async move {
                    search::search(&mut context, &cache, &manifest, prepared).await
                },
            )
            .await?;
        // The search is the relevant access that rediscovers cold split and
        // merge states; offering them is best-effort and loss-safe.
        if !maintenance.is_empty() {
            self.runtime.offer_fixups(&self.manifest, maintenance);
        }
        Ok(outcome)
    }

    /// Runs one bounded, read-only verification of this index.
    ///
    /// The audit validates the persisted Active Manifest and then checks one
    /// consistent backend snapshot: canonical encodings, tree reachability
    /// and unique incoming references, exact Header counts and legal State
    /// references, Record–Location–Leaf membership, Leaf Entry projection
    /// agreement, conservative Synopses, and allocator high-water marks. It
    /// never mutates or repairs persistent data, and there is no
    /// continuation, spill, sampling, or repair mode.
    ///
    /// [`VerifyOptions`] bounds the reported issues, the visited logical
    /// objects, and the resident memory independently, and carries the
    /// deadline and cancellation control other operations take through their
    /// `_with_control` companion. Reaching any limit stops the audit and
    /// returns the collected issues with [`VerifyReport::complete`] set to
    /// `false`; only a complete report is conclusive. Cancellation,
    /// deadline, and snapshot failure return errors rather than a partial
    /// cross-snapshot conclusion, so on a backend with a short snapshot
    /// lifetime — FoundationDB — a large audit must run against an offline
    /// copy or another instance able to hold the snapshot.
    ///
    /// Issues are deliberately coarse and redacted: they carry only the
    /// Logical Index ID, a stable hash of the Tree Key, the Partition Key,
    /// and an optional Record ID — never raw Tree Keys, vectors, payloads,
    /// or filter values.
    pub async fn verify(&self, options: VerifyOptions) -> Result<VerifyReport> {
        options.validate()?;
        let manifest = Arc::clone(&self.manifest);
        let operation_options = options.operation_options().clone();
        self.run_foreground(
            Operation::Verify,
            operation_options,
            move |mut context| async move { verify::verify(&mut context, &manifest, options).await },
        )
        .await
    }

    /// Validates one mutation batch against this index's immutable
    /// configuration before any storage work.
    pub(crate) fn validate(&self, mutations: &mut [Mutation]) -> Result<()> {
        let config = self.manifest.config();
        validate_mutations(mutations, config.dimension(), config.fields())
    }

    /// Returns the owning Runtime's shared state.
    pub(crate) fn runtime(&self) -> &Arc<RuntimeInner<B>> {
        &self.runtime
    }

    /// Runs one foreground operation observed under this index's identity.
    pub(crate) async fn run_foreground<T, F, Fut>(
        &self,
        operation: Operation,
        options: OperationOptions,
        work: F,
    ) -> Result<T>
    where
        T: Send + 'static,
        F: FnOnce(OperationContext<B>) -> Fut + Send + 'static,
        Fut: Future<Output = Result<T>> + Send + 'static,
    {
        let index = Some(self.manifest.logical_index_id());
        self.runtime
            .run_foreground(operation, index, options, work)
            .await
    }

    /// Runs one validated mutation batch under foreground admission.
    ///
    /// A committed batch's maintenance discoveries — split candidates,
    /// shrunken leaves, and draining sources rerouted around — are offered to
    /// the Runtime's bounded Fixup queue after success; losing them never
    /// affects correctness.
    pub(crate) async fn run_mutations(
        &self,
        operation: Operation,
        mutations: Vec<Mutation>,
        operation_options: OperationOptions,
    ) -> Result<Vec<MutationOutcome>> {
        let retry = lifecycle::RetryPolicy::from_config(self.runtime.config());
        let manifest = Arc::clone(&self.manifest);
        let report = self
            .run_foreground(
                operation,
                operation_options,
                move |mut context| async move {
                    mutation::mutate(&mut context, &manifest, &mutations, retry, operation).await
                },
            )
            .await?;
        if !report.maintenance.is_empty() {
            self.runtime
                .offer_fixups(&self.manifest, report.maintenance);
        }
        Ok(report.outcomes)
    }

    /// Reads one Vector Record by Record ID.
    ///
    /// The read validates the persisted Active Manifest and the requested
    /// Record Group from one consistent backend snapshot. It returns
    /// `Ok(None)` when the Record ID is absent. A partial Record Group or a
    /// Manifest that no longer matches this handle fails with
    /// [`ErrorKind::Corruption`]; a Dropping Logical Index fails with
    /// [`ErrorKind::IndexDropping`] and a dropped one with
    /// [`ErrorKind::IndexNotFound`]. The Opaque Payload projection is closed
    /// by [`GetOptions`].
    pub async fn get(&self, id: Bytes, options: GetOptions) -> Result<Option<StoredRecord>> {
        self.get_with_control(id, options, OperationOptions::default())
            .await
    }

    /// Reads one Vector Record with explicit operation control.
    pub async fn get_with_control(
        &self,
        id: Bytes,
        options: GetOptions,
        operation_options: OperationOptions,
    ) -> Result<Option<StoredRecord>> {
        validate_id(&id)?;
        let manifest = Arc::clone(&self.manifest);
        self.run_foreground(
            Operation::Get,
            operation_options,
            move |mut context| async move {
                reads::get_record(&mut context, &manifest, id, options.includes_payload()).await
            },
        )
        .await
    }

    /// Reads Vector Records by Record ID in one bounded batch.
    ///
    /// The result is same-order and same-length with the input: `Ok(None)`
    /// marks an absent Record ID and duplicate Record IDs repeat their value.
    /// An empty batch succeeds with an empty result. All Record Groups and the
    /// validated Manifest are read from one consistent backend snapshot, and
    /// the backend's batch and key limits, cancellation, and deadline apply to
    /// the whole operation.
    pub async fn batch_get(
        &self,
        ids: Vec<Bytes>,
        options: GetOptions,
    ) -> Result<Vec<Option<StoredRecord>>> {
        self.batch_get_with_control(ids, options, OperationOptions::default())
            .await
    }

    /// Reads Vector Records with explicit operation control.
    pub async fn batch_get_with_control(
        &self,
        ids: Vec<Bytes>,
        options: GetOptions,
        operation_options: OperationOptions,
    ) -> Result<Vec<Option<StoredRecord>>> {
        validate_ids(&ids)?;
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let manifest = Arc::clone(&self.manifest);
        self.run_foreground(
            Operation::BatchGet,
            operation_options,
            move |mut context| async move {
                reads::batch_get_records(&mut context, &manifest, ids, options.includes_payload())
                    .await
            },
        )
        .await
    }
}

impl<B: Backend> Clone for Index<B> {
    fn clone(&self) -> Self {
        Self {
            runtime: Arc::clone(&self.runtime),
            name: self.name.clone(),
            manifest: Arc::clone(&self.manifest),
        }
    }
}

impl<B: Backend> fmt::Debug for Index<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Index")
            .field("name", &"[REDACTED]")
            .field("logical_index_id", &self.manifest.logical_index_id())
            .field("lifecycle", &self.manifest.lifecycle())
            .finish()
    }
}
