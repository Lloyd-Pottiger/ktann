//! Bounded approximate-search requests and outcomes.

use std::fmt;
use std::sync::Arc;

use bytes::Bytes;

use super::{Error, FieldSchema, Predicate, Result, validate_id};

const MAX_K: usize = 65_536;
const MAX_SCANNED_TREE_KEYS: u32 = 65_536;
const MAX_VISITED_PARTITIONS: u32 = 16_384;
const MAX_VISITED_LEAF_ENTRIES: u32 = 1_048_576;
const MAX_EXACT_RERANK_CANDIDATES: u32 = 65_536;
/// Beam width is counted in partitions, so anything wider than the visited
/// partition hard cap is guaranteed to exhaust that budget instead.
const MAX_LEAF_BEAM_SIZE: u32 = MAX_VISITED_PARTITIONS;

/// Concrete limits applied to one Search.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct SearchBudgets {
    scanned_tree_keys: u32,
    visited_partitions: u32,
    visited_leaf_entries: u32,
    exact_rerank_candidates: u32,
}

impl Default for SearchBudgets {
    fn default() -> Self {
        Self {
            scanned_tree_keys: 4_096,
            visited_partitions: 1_024,
            visited_leaf_entries: 65_536,
            exact_rerank_candidates: 65_536,
        }
    }
}

impl SearchBudgets {
    /// Constructs positive concrete Search Budgets.
    pub fn new(
        scanned_tree_keys: u32,
        visited_partitions: u32,
        visited_leaf_entries: u32,
        exact_rerank_candidates: u32,
    ) -> Result<Self> {
        let budgets = Self {
            scanned_tree_keys,
            visited_partitions,
            visited_leaf_entries,
            exact_rerank_candidates,
        };
        budgets.validate_hard_caps()?;
        Ok(budgets)
    }

    pub(crate) fn validate_hard_caps(&self) -> Result<()> {
        if self.scanned_tree_keys == 0
            || self.scanned_tree_keys > MAX_SCANNED_TREE_KEYS
            || self.visited_partitions == 0
            || self.visited_partitions > MAX_VISITED_PARTITIONS
            || self.visited_leaf_entries == 0
            || self.visited_leaf_entries > MAX_VISITED_LEAF_ENTRIES
            || self.exact_rerank_candidates == 0
            || self.exact_rerank_candidates > MAX_EXACT_RERANK_CANDIDATES
        {
            return Err(Error::invalid_argument());
        }
        Ok(())
    }

    /// Returns the scanned Tree Key limit.
    #[must_use]
    pub const fn scanned_tree_keys(self) -> u32 {
        self.scanned_tree_keys
    }

    /// Returns the visited partition limit.
    #[must_use]
    pub const fn visited_partitions(self) -> u32 {
        self.visited_partitions
    }

    /// Returns the visited Leaf Entry limit.
    #[must_use]
    pub const fn visited_leaf_entries(self) -> u32 {
        self.visited_leaf_entries
    }

    /// Returns the exact-rerank candidate limit.
    #[must_use]
    pub const fn exact_rerank_candidates(self) -> u32 {
        self.exact_rerank_candidates
    }
}

/// Optional per-request overrides of Runtime Search Budgets and traversal
/// width.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub struct SearchOptions {
    scanned_tree_keys: Option<u32>,
    visited_partitions: Option<u32>,
    visited_leaf_entries: Option<u32>,
    exact_rerank_candidates: Option<u32>,
    leaf_beam_size: Option<u32>,
}

impl SearchOptions {
    /// Overrides the positive scanned Tree Key budget.
    pub fn with_scanned_tree_keys(mut self, value: u32) -> Result<Self> {
        validate_override(value, MAX_SCANNED_TREE_KEYS)?;
        self.scanned_tree_keys = Some(value);
        Ok(self)
    }

    /// Overrides the positive visited partition budget.
    pub fn with_visited_partitions(mut self, value: u32) -> Result<Self> {
        validate_override(value, MAX_VISITED_PARTITIONS)?;
        self.visited_partitions = Some(value);
        Ok(self)
    }

    /// Overrides the positive visited Leaf Entry budget.
    pub fn with_visited_leaf_entries(mut self, value: u32) -> Result<Self> {
        validate_override(value, MAX_VISITED_LEAF_ENTRIES)?;
        self.visited_leaf_entries = Some(value);
        Ok(self)
    }

    /// Overrides the positive exact-rerank candidate budget.
    pub fn with_exact_rerank_candidates(mut self, value: u32) -> Result<Self> {
        validate_override(value, MAX_EXACT_RERANK_CANDIDATES)?;
        self.exact_rerank_candidates = Some(value);
        Ok(self)
    }

    /// Overrides the positive leaf-level base beam width.
    ///
    /// The beam is a traversal-quality knob, not an accounted budget
    /// dimension: wider beams visit more partitions, all still charged to the
    /// visited-partition budget. When unset, the leaf-level base beam defaults
    /// to 32 (design `search.md` section 6).
    pub fn with_leaf_beam_size(mut self, value: u32) -> Result<Self> {
        validate_override(value, MAX_LEAF_BEAM_SIZE)?;
        self.leaf_beam_size = Some(value);
        Ok(self)
    }

    /// Returns the leaf-level base beam width override.
    #[must_use]
    pub const fn leaf_beam_size(self) -> Option<u32> {
        self.leaf_beam_size
    }

    /// Resolves overrides against Runtime defaults and validates them for `k`.
    pub fn resolve(self, defaults: SearchBudgets, k: usize) -> Result<SearchBudgets> {
        defaults.validate_hard_caps()?;
        let exact_default = default_exact_rerank(k)?;
        let budgets = SearchBudgets {
            scanned_tree_keys: self.scanned_tree_keys.unwrap_or(defaults.scanned_tree_keys),
            visited_partitions: self
                .visited_partitions
                .unwrap_or(defaults.visited_partitions),
            visited_leaf_entries: self
                .visited_leaf_entries
                .unwrap_or(defaults.visited_leaf_entries),
            exact_rerank_candidates: self
                .exact_rerank_candidates
                .unwrap_or(defaults.exact_rerank_candidates.min(exact_default)),
        };
        budgets.validate_hard_caps()?;
        let k = u32::try_from(k).map_err(|_| Error::invalid_argument())?;
        if budgets.exact_rerank_candidates < k {
            return Err(Error::invalid_argument());
        }
        Ok(budgets)
    }
}

fn validate_override(value: u32, maximum: u32) -> Result<()> {
    if value == 0 || value > maximum {
        Err(Error::invalid_argument())
    } else {
        Ok(())
    }
}

fn default_exact_rerank(k: usize) -> Result<u32> {
    if !(1..=MAX_K).contains(&k) {
        return Err(Error::invalid_argument());
    }
    let value = k
        .checked_mul(4)
        .ok_or_else(Error::invalid_argument)?
        .clamp(100, MAX_K);
    u32::try_from(value).map_err(|_| Error::invalid_argument())
}

/// One owned approximate-search request.
#[derive(Clone)]
#[non_exhaustive]
pub struct SearchRequest {
    vector: Arc<[f32]>,
    k: usize,
    predicate: Option<Predicate>,
    options: SearchOptions,
}

impl SearchRequest {
    /// Creates a request with no Filter Predicate and Runtime Search Budgets.
    pub fn new(vector: impl Into<Arc<[f32]>>, k: usize) -> Result<Self> {
        let vector = vector.into();
        if !(1..=MAX_K).contains(&k) || vector.iter().any(|value| !value.is_finite()) {
            return Err(Error::invalid_argument());
        }
        Ok(Self {
            vector,
            k,
            predicate: None,
            options: SearchOptions::default(),
        })
    }

    /// Adds an exact typed Filter Predicate.
    #[must_use]
    pub fn with_predicate(mut self, predicate: Predicate) -> Self {
        self.predicate = Some(predicate);
        self
    }

    /// Sets request Search Budget overrides.
    #[must_use]
    pub const fn with_options(mut self, options: SearchOptions) -> Self {
        self.options = options;
        self
    }

    /// Validates dimension, Filter Predicate, and effective Search Budgets.
    pub fn validate(
        &mut self,
        dimension: usize,
        fields: &[FieldSchema],
        defaults: SearchBudgets,
    ) -> Result<SearchBudgets> {
        // Vector finiteness is guaranteed by `SearchRequest::new` and the
        // vector is immutable, so only the index-dependent shape is checked.
        if self.vector.len() != dimension {
            return Err(Error::invalid_argument());
        }
        if let Some(predicate) = &mut self.predicate {
            predicate.validate(fields)?;
        }
        self.options.resolve(defaults, self.k)
    }

    /// Returns the finite query vector.
    #[must_use]
    pub fn vector(&self) -> &[f32] {
        &self.vector
    }

    /// Returns the maximum requested hit count.
    #[must_use]
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Returns the optional exact Filter Predicate.
    #[must_use]
    pub const fn predicate(&self) -> Option<&Predicate> {
        self.predicate.as_ref()
    }

    /// Returns the Search Budget overrides.
    #[must_use]
    pub const fn options(&self) -> SearchOptions {
        self.options
    }
}

impl fmt::Debug for SearchRequest {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SearchRequest")
            .field("vector", &"[REDACTED]")
            .field("k", &self.k)
            .field("predicate", &self.predicate.as_ref().map(|_| "[REDACTED]"))
            .field("options", &self.options)
            .finish()
    }
}

/// One exactly reranked Search Hit.
#[derive(Clone)]
#[non_exhaustive]
pub struct SearchHit {
    id: Bytes,
    distance: f64,
}

impl SearchHit {
    /// Creates a hit from a valid Record ID and finite exact distance.
    pub fn new(id: Bytes, distance: f64) -> Result<Self> {
        validate_id(&id)?;
        if !distance.is_finite() {
            return Err(Error::invalid_argument());
        }
        Ok(Self { id, distance })
    }

    /// Returns the opaque Record ID.
    #[must_use]
    pub const fn id(&self) -> &Bytes {
        &self.id
    }

    /// Returns the exact distance computed from the original vector.
    #[must_use]
    pub const fn distance(&self) -> f64 {
        self.distance
    }
}

impl fmt::Debug for SearchHit {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SearchHit")
            .field("id", &"[REDACTED]")
            .field("distance", &self.distance)
            .finish()
    }
}

/// Logical work charged to each Search Budget dimension.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub struct SearchBudgetUsage {
    /// Tree Keys decoded and checked from the directory.
    pub scanned_tree_keys: u32,
    /// Distinct partition bodies logically visited, including cache hits.
    pub visited_partitions: u32,
    /// Leaf Entries read and considered under the exact Filter Predicate.
    pub visited_leaf_entries: u32,
    /// Vector Records read and exactly reranked.
    pub exact_rerank_candidates: u32,
}

/// Whether eligible work was prevented by each depleted Search Budget.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
#[non_exhaustive]
pub struct SearchBudgetExhaustion {
    /// Scanned Tree Key budget prevented eligible work.
    pub scanned_tree_keys: bool,
    /// Visited partition budget prevented eligible work.
    pub visited_partitions: bool,
    /// Visited Leaf Entry budget prevented eligible work.
    pub visited_leaf_entries: bool,
    /// Exact-rerank candidate budget prevented eligible work.
    pub exact_rerank_candidates: bool,
}

/// The bounded, deterministic result of approximate Search.
#[derive(Clone, Debug, Default)]
#[non_exhaustive]
pub struct SearchOutcome {
    /// Exactly reranked hits ordered by distance and Record ID.
    pub hits: Vec<SearchHit>,
    /// Logical work charged to Search Budgets.
    pub usage: SearchBudgetUsage,
    /// Every Search Budget dimension that prevented eligible work.
    pub exhausted: SearchBudgetExhaustion,
    /// Whether a per-leaf RaBitQ overlap cap discarded a qualifying overlap.
    pub rabitq_overlap_truncated: bool,
}
