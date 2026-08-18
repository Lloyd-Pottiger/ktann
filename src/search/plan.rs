//! Predicate-to-Tree-Key range planning and bounded directory enumeration.
//!
//! Planning derives one conservative, deterministic set of disjoint half-open
//! directory ranges from a Filter Predicate. Leading Tree Key fields narrowed
//! to exact points extend byte prefixes; at most one later field narrows with
//! a representable typed half-open interval, and fields beyond that stay
//! unbounded. When exact disjoint expansion would exceed the configured range
//! limit the plan widens conservatively and relies on exact predicate
//! evaluation later. Every constrained field keeps a typed membership check,
//! because memcomparable String prefix ranges may cover values extended by a
//! leading `0x00` byte.
//!
//! Enumeration pages the directory forward in canonical key order, counts
//! every decoded Tree Key against one global scanned-Tree-Key budget, and
//! materializes only keys counted inside that budget, so an unlimited number
//! of stored trees cannot cause unbounded query memory or read-ahead.

#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "the bounded search operation consumes the planner and enumeration (#30)"
    )
)]

use std::cmp::Ordering;

use crate::api::{
    CompareOp, DataType, Error, ErrorKind, FieldId, LogicalIndexId, MAX_STRING_BYTES, Predicate,
    Result, Value, typed_order,
};
use crate::storage::backend::{ReadOps, ScanLimits};
use crate::storage::keys::{self, KeyRange, LogicalKey, TreeKey};
use crate::storage::values::{IndexManifest, PersistentValue, TreeManifest};
use crate::storage::{LogicalRange, LogicalScanCursor, ReadLogicalTxn};

/// One planned tree to traverse: its canonical Tree Key and Tree Manifest.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct EnumeratedTree {
    tree_key: TreeKey,
    manifest: TreeManifest,
}

impl EnumeratedTree {
    /// Returns the canonical Tree Key.
    #[must_use]
    pub(crate) fn tree_key(&self) -> &TreeKey {
        &self.tree_key
    }

    /// Returns the Tree Manifest directory entry.
    #[must_use]
    pub(crate) const fn manifest(&self) -> TreeManifest {
        self.manifest
    }
}

/// The result of bounded directory enumeration.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TreeKeyEnumeration {
    trees: Vec<EnumeratedTree>,
    scanned_tree_keys: u32,
    scanned_tree_key_budget_exhausted: bool,
}

impl TreeKeyEnumeration {
    /// Returns the materialized trees in canonical Tree Key order.
    #[must_use]
    pub(crate) fn trees(&self) -> &[EnumeratedTree] {
        &self.trees
    }

    /// Returns the number of Tree Keys decoded and checked.
    #[must_use]
    pub(crate) const fn scanned_tree_keys(&self) -> u32 {
        self.scanned_tree_keys
    }

    /// Returns whether the scanned-Tree-Key budget prevented eligible work.
    #[must_use]
    pub(crate) const fn scanned_tree_key_budget_exhausted(&self) -> bool {
        self.scanned_tree_key_budget_exhausted
    }
}

/// One conservative, ordered, disjoint Tree Manifest directory range plan.
#[derive(Clone, Debug)]
pub(crate) struct TreeKeyPlan {
    ranges: Vec<KeyRange>,
    checks: Vec<FieldCheck>,
    types: Box<[DataType]>,
}

impl TreeKeyPlan {
    /// Returns the ordered disjoint directory ranges to page.
    #[must_use]
    pub(crate) fn ranges(&self) -> &[KeyRange] {
        &self.ranges
    }

    /// Returns whether the predicate can match no Tree Key at all.
    #[must_use]
    pub(crate) fn is_empty(&self) -> bool {
        self.ranges.is_empty()
    }

    /// Returns whether `tree_key` lies inside every constrained field's set.
    ///
    /// Malformed or noncanonical encodings fail closed. A plan without ranges
    /// can match no Tree Key at all.
    pub(crate) fn accepts(&self, tree_key: &TreeKey) -> Result<bool> {
        if self.ranges.is_empty() {
            return Ok(false);
        }
        if self.checks.is_empty() {
            return Ok(true);
        }
        let values = tree_key.values(&self.types)?;
        for check in &self.checks {
            let value = values.get(check.ordinal).ok_or_else(corruption)?;
            if !check.contains(value)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// Plans the Tree Manifest directory ranges implied by a Filter Predicate.
///
/// `range_limit` bounds the disjoint range expansion; wider plans stay
/// conservative. A `None` predicate plans the complete directory. The
/// predicate must reference existing fields with matching typed values;
/// invalid references fail as [`ErrorKind::InvalidArgument`].
pub(crate) fn plan_tree_keys(
    manifest: &IndexManifest,
    predicate: Option<&Predicate>,
    range_limit: u32,
) -> Result<TreeKeyPlan> {
    if range_limit == 0 {
        return Err(Error::invalid_argument());
    }
    let schema = TreeSchema::new(manifest);
    let constraints = match predicate {
        Some(predicate) => derive(predicate, &schema)?,
        None => Constraints::full(&schema.types),
    };
    let index = manifest.logical_index_id();

    let mut drafts = vec![Draft::prefix()];
    let mut ranges = Vec::new();
    for ordinal in 0..schema.field_count {
        let Some(set) = &constraints.sets[ordinal] else {
            break;
        };
        if set.intervals.is_empty() {
            return Ok(TreeKeyPlan {
                ranges: Vec::new(),
                checks: Vec::new(),
                types: schema.types.clone(),
            });
        }
        let draft_count = u64::try_from(drafts.len()).map_err(|_| limit_exceeded())?;
        let interval_count = u64::try_from(set.intervals.len()).map_err(|_| limit_exceeded())?;
        let product = draft_count
            .checked_mul(interval_count)
            .ok_or_else(limit_exceeded)?;
        if product > u64::from(range_limit) {
            break;
        }
        if set.is_all_points()? {
            let mut next = Vec::with_capacity(product as usize);
            for draft in &drafts {
                for interval in &set.intervals {
                    let Some(value) = set.point_value(interval)? else {
                        return Err(corruption());
                    };
                    next.push(Draft::with_point(&draft.prefix, value));
                }
            }
            drafts = next;
        } else {
            for draft in &drafts {
                for interval in &set.intervals {
                    ranges.push(materialize_range(
                        index,
                        &schema.types,
                        &draft.prefix,
                        interval.lo.as_ref(),
                        interval.hi.as_ref(),
                    )?);
                }
            }
            drafts.clear();
            break;
        }
    }
    for draft in &drafts {
        ranges.push(materialize_range(
            index,
            &schema.types,
            &draft.prefix,
            None,
            None,
        )?);
    }
    // String upper bounds widen to the byte successor of the bound encoding,
    // so a following range that extends that encoding can overlap. Merging
    // restores order and disjointness without losing any key.
    ranges = merge_ranges(ranges);

    let checks = constraints
        .sets
        .iter()
        .enumerate()
        .filter_map(|(ordinal, set)| {
            set.as_ref().map(|set| FieldCheck {
                ordinal,
                intervals: set.intervals.clone(),
            })
        })
        .collect();
    Ok(TreeKeyPlan {
        ranges,
        checks,
        types: schema.types.clone(),
    })
}

/// Enumerates Tree Manifests under one plan and one global key budget.
///
/// Pages advance in canonical key order with an item bound of the remaining
/// key budget and the caller's byte bound, so no page reads beyond the
/// remaining budget. Every decoded directory key counts one against `budget`;
/// keys outside the plan's typed checks are counted but not materialized.
/// Materialization stops as soon as the budget reaches zero, and
/// [`TreeKeyEnumeration::scanned_tree_key_budget_exhausted`] reports whether
/// eligible work remained.
pub(crate) async fn enumerate_tree_keys<T: ReadOps>(
    txn: &mut ReadLogicalTxn<'_, T>,
    manifest: &IndexManifest,
    plan: &TreeKeyPlan,
    budget: u32,
    page_limits: ScanLimits,
) -> Result<TreeKeyEnumeration> {
    let mut trees = Vec::new();
    let mut remaining = budget;
    let mut exhausted = false;
    'ranges: for raw in plan.ranges() {
        let range = LogicalRange::tree_manifest_plan(manifest, raw.clone());
        let mut cursor: Option<LogicalScanCursor> = None;
        loop {
            if remaining == 0 {
                exhausted = true;
                break 'ranges;
            }
            let page = txn
                .scan(
                    &range,
                    cursor.as_ref(),
                    ScanLimits {
                        item_limit: page_limits.item_limit.min(remaining as usize),
                        byte_limit: page_limits.byte_limit,
                    },
                )
                .await?;
            for item in page.items() {
                let LogicalKey::TreeManifest { tree_key, .. } = item.key() else {
                    return Err(corruption());
                };
                let PersistentValue::TreeManifest(tree_manifest) = item.value() else {
                    return Err(corruption());
                };
                remaining -= 1;
                if plan.accepts(tree_key)? {
                    trees.push(EnumeratedTree {
                        tree_key: tree_key.clone(),
                        manifest: *tree_manifest,
                    });
                }
            }
            match page.next_cursor() {
                Some(next) => cursor = Some(next.clone()),
                None => break,
            }
            if remaining == 0 {
                exhausted = true;
                break 'ranges;
            }
        }
    }
    Ok(TreeKeyEnumeration {
        trees,
        scanned_tree_keys: budget - remaining,
        scanned_tree_key_budget_exhausted: exhausted,
    })
}

/// The ordered Tree Key schema of one Logical Index.
struct TreeSchema {
    field_count: usize,
    types: Box<[DataType]>,
    ordinal_by_schema_pos: Vec<Option<usize>>,
}

impl TreeSchema {
    fn new(manifest: &IndexManifest) -> Self {
        let (types, field_count) = manifest.tree_key_types();
        let mut ordinal_by_schema_pos = vec![None; manifest.config().fields().len()];
        for (ordinal, field_id) in manifest.config().tree_key_fields().iter().enumerate() {
            ordinal_by_schema_pos[usize::from(field_id.0)] = Some(ordinal);
        }
        Self {
            field_count,
            types: types[..field_count].into(),
            ordinal_by_schema_pos,
        }
    }

    /// Returns the Tree Key ordinal of a schema field, after validating that
    /// the field exists.
    fn ordinal(&self, field: FieldId) -> Result<Option<usize>> {
        self.ordinal_by_schema_pos
            .get(usize::from(field.0))
            .copied()
            .ok_or_else(Error::invalid_argument)
    }
}

/// A conservative per-field constraint map plus soundness tracking.
///
/// `None` means the full domain; `Some` holds a conservative superset of the
/// values that field can take in a matching Tree Key. `exact` records whether
/// the possible Tree Key set equals the per-field product exactly, and
/// `tree_pure` records whether every leaf constrains Tree Key fields, so the
/// predicate's truth is a deterministic function of the Tree Key alone. A
/// negation can only complement when both hold, and even then only when the
/// product constrains at most one field.
struct Constraints {
    types: Box<[DataType]>,
    sets: Vec<Option<FieldSet>>,
    exact: bool,
    tree_pure: bool,
}

impl Constraints {
    fn full(types: &[DataType]) -> Self {
        Self {
            types: types.into(),
            sets: vec![None; types.len()],
            exact: true,
            tree_pure: true,
        }
    }

    /// The full domain for a leaf that references a non-Tree-Key field.
    fn non_tree(types: &[DataType]) -> Self {
        Self {
            types: types.into(),
            sets: vec![None; types.len()],
            exact: true,
            tree_pure: false,
        }
    }

    fn wide(types: &[DataType]) -> Self {
        Self {
            types: types.into(),
            sets: vec![None; types.len()],
            exact: false,
            tree_pure: false,
        }
    }

    fn impossible(types: &[DataType]) -> Self {
        Self {
            sets: types.iter().map(|ty| Some(FieldSet::empty(*ty))).collect(),
            types: types.into(),
            exact: true,
            tree_pure: true,
        }
    }

    fn empty_union(types: &[DataType]) -> Self {
        Self {
            sets: types.iter().map(|ty| Some(FieldSet::empty(*ty))).collect(),
            types: types.into(),
            exact: true,
            tree_pure: true,
        }
    }

    fn intersect(&mut self, other: &Self) {
        for (left, right) in self.sets.iter_mut().zip(&other.sets) {
            match (left.as_ref(), right.as_ref()) {
                (_, None) => {}
                (None, Some(set)) => *left = Some(set.clone()),
                (Some(left_set), Some(right_set)) => *left = Some(left_set.intersect(right_set)),
            }
        }
    }

    fn union(&mut self, other: &Self) {
        for (left, right) in self.sets.iter_mut().zip(&other.sets) {
            match (left.as_ref(), right.as_ref()) {
                (Some(left_set), Some(right_set)) => *left = Some(left_set.union(right_set)),
                _ => *left = None,
            }
        }
    }

    /// Complements one exact single-field tree-pure product, or widens.
    ///
    /// A per-field complement of a product constraining several fields
    /// excludes keys where only some fields fall outside their sets, so it is
    /// not sound; those negations widen to the full domain and rely on exact
    /// predicate evaluation later.
    fn negated(&self) -> Self {
        let constrained: Vec<(usize, &FieldSet)> = self
            .sets
            .iter()
            .enumerate()
            .filter_map(|(ordinal, set)| set.as_ref().map(|set| (ordinal, set)))
            .collect();
        match constrained.as_slice() {
            [] => Self::impossible(&self.types),
            [(ordinal, set)] => {
                let mut negated = Constraints::full(&self.types);
                negated.sets[*ordinal] = Some(set.complement());
                negated
            }
            _ => Self::wide(&self.types),
        }
    }
}

/// Derives one sound per-field constraint map from a validated predicate.
fn derive(predicate: &Predicate, schema: &TreeSchema) -> Result<Constraints> {
    match predicate {
        Predicate::And(children) => {
            let mut result = Constraints::full(&schema.types);
            for child in children {
                let child_constraints = derive(child, schema)?;
                result.intersect(&child_constraints);
                result.exact &= child_constraints.exact;
                result.tree_pure &= child_constraints.tree_pure;
            }
            Ok(result)
        }
        Predicate::Or(children) => {
            let mut result = Constraints::empty_union(&schema.types);
            for child in children {
                let child_constraints = derive(child, schema)?;
                result.union(&child_constraints);
                result.tree_pure &= child_constraints.tree_pure;
            }
            result.exact = false;
            Ok(result)
        }
        Predicate::Not(child) => {
            let child_constraints = derive(child, schema)?;
            if child_constraints.exact && child_constraints.tree_pure {
                Ok(child_constraints.negated())
            } else {
                Ok(Constraints::wide(&schema.types))
            }
        }
        Predicate::Compare { field, op, value } => {
            if let Some(ordinal) = schema.ordinal(*field)? {
                let ty = schema.types[ordinal];
                check_typed(ty, value)?;
                let mut constraints = Constraints::full(&schema.types);
                constraints.sets[ordinal] = Some(FieldSet::from_compare(ty, *op, value)?);
                Ok(constraints)
            } else {
                // Any Tree Key admits a non-tree field assignment that can
                // make the comparison TRUE, so the possible set is the full
                // domain, exactly.
                Ok(Constraints::non_tree(&schema.types))
            }
        }
        Predicate::In { field, values } => {
            if let Some(ordinal) = schema.ordinal(*field)? {
                let ty = schema.types[ordinal];
                let mut intervals = Vec::with_capacity(values.len());
                for value in values {
                    check_typed(ty, value)?;
                    intervals.push(Interval {
                        lo: Some(value.clone()),
                        hi: next_value(ty, value)?,
                    });
                }
                let mut constraints = Constraints::full(&schema.types);
                constraints.sets[ordinal] = Some(FieldSet::normalize(ty, intervals));
                Ok(constraints)
            } else {
                Ok(Constraints::non_tree(&schema.types))
            }
        }
        Predicate::IsNull(field) => {
            if let Some(ordinal) = schema.ordinal(*field)? {
                let mut constraints = Constraints::full(&schema.types);
                constraints.sets[ordinal] = Some(FieldSet::empty(schema.types[ordinal]));
                Ok(constraints)
            } else {
                Ok(Constraints::non_tree(&schema.types))
            }
        }
        Predicate::IsNotNull(field) => {
            if schema.ordinal(*field)?.is_some() {
                Ok(Constraints::full(&schema.types))
            } else {
                Ok(Constraints::non_tree(&schema.types))
            }
        }
    }
}

/// One half-open typed interval `[lo, hi)` over one Tree Key field domain.
#[derive(Clone, Debug)]
struct Interval {
    lo: Option<Value>,
    hi: Option<Value>,
}

/// A canonical sorted set of disjoint merged intervals over one domain.
#[derive(Clone, Debug)]
struct FieldSet {
    ty: DataType,
    intervals: Vec<Interval>,
}

impl FieldSet {
    fn empty(ty: DataType) -> Self {
        Self {
            ty,
            intervals: Vec::new(),
        }
    }

    fn normalize(ty: DataType, mut intervals: Vec<Interval>) -> Self {
        intervals.retain(|interval| !interval_is_empty(ty, interval));
        intervals.sort_by(cmp_intervals);
        let mut merged: Vec<Interval> = Vec::with_capacity(intervals.len());
        for interval in intervals {
            match merged.last_mut() {
                Some(last) if overlaps_or_adjacent(last, &interval) => {
                    last.hi = max_hi(last.hi.take(), interval.hi);
                }
                _ => merged.push(interval),
            }
        }
        Self {
            ty,
            intervals: merged,
        }
    }

    fn from_compare(ty: DataType, op: CompareOp, value: &Value) -> Result<Self> {
        let next = next_value(ty, value)?;
        let intervals = match op {
            CompareOp::Eq => vec![Interval {
                lo: Some(value.clone()),
                hi: next,
            }],
            CompareOp::NotEq => {
                let mut intervals = vec![Interval {
                    lo: None,
                    hi: Some(value.clone()),
                }];
                // No successor exists above the domain maximum, so the upper
                // half of the complement is empty there.
                if let Some(next) = next {
                    intervals.push(Interval {
                        lo: Some(next),
                        hi: None,
                    });
                }
                intervals
            }
            CompareOp::Lt => vec![Interval {
                lo: None,
                hi: Some(value.clone()),
            }],
            CompareOp::LessOrEqual => vec![Interval { lo: None, hi: next }],
            CompareOp::Gt => match next {
                Some(next) => vec![Interval {
                    lo: Some(next),
                    hi: None,
                }],
                // Strictly greater than the domain maximum matches nothing.
                None => vec![],
            },
            CompareOp::GreaterOrEqual => vec![Interval {
                lo: Some(value.clone()),
                hi: None,
            }],
        };
        Ok(Self::normalize(ty, intervals))
    }

    fn union(&self, other: &Self) -> Self {
        Self::normalize(
            self.ty,
            self.intervals
                .iter()
                .chain(&other.intervals)
                .cloned()
                .collect(),
        )
    }

    fn intersect(&self, other: &Self) -> Self {
        let mut intervals = Vec::new();
        for left in &self.intervals {
            for right in &other.intervals {
                intervals.push(Interval {
                    lo: max_lo(left.lo.clone(), right.lo.clone()),
                    hi: min_hi(left.hi.clone(), right.hi.clone()),
                });
            }
        }
        Self::normalize(self.ty, intervals)
    }

    fn complement(&self) -> Self {
        if self.intervals.is_empty() {
            return Self::normalize(self.ty, vec![Interval { lo: None, hi: None }]);
        }
        let mut intervals = Vec::with_capacity(self.intervals.len() + 1);
        let mut cursor: Option<Value> = None;
        for interval in &self.intervals {
            // The piece below the interval is empty when both bounds are
            // unbounded, and the piece above the last interval exists only
            // when its upper bound is bounded.
            if cursor.is_some() || interval.lo.is_some() {
                intervals.push(Interval {
                    lo: cursor,
                    hi: interval.lo.clone(),
                });
            }
            cursor = interval.hi.clone();
        }
        if cursor.is_some() {
            intervals.push(Interval {
                lo: cursor,
                hi: None,
            });
        }
        Self::normalize(self.ty, intervals)
    }

    fn is_all_points(&self) -> Result<bool> {
        self.intervals
            .iter()
            .map(|interval| self.point_value(interval))
            .collect::<Result<Vec<_>>>()
            .map(|points| points.iter().all(Option::is_some))
    }

    fn point_value(&self, interval: &Interval) -> Result<Option<Value>> {
        let Some(lo) = interval.lo.as_ref() else {
            return Ok(None);
        };
        let next = next_value(self.ty, lo)?;
        Ok(match (&interval.hi, next) {
            (Some(hi), Some(next)) if typed_order(hi, &next) == Some(Ordering::Equal) => {
                Some(lo.clone())
            }
            (None, None) => Some(lo.clone()),
            _ => None,
        })
    }
}

/// One constrained Tree Key field's typed membership check.
#[derive(Clone, Debug)]
struct FieldCheck {
    ordinal: usize,
    intervals: Vec<Interval>,
}

impl FieldCheck {
    fn contains(&self, value: &Value) -> Result<bool> {
        for interval in &self.intervals {
            if let Some(lo) = &interval.lo {
                if bound_order(value, lo) == Ordering::Less {
                    return Ok(false);
                }
            }
            match &interval.hi {
                Some(hi) if bound_order(value, hi) == Ordering::Less => return Ok(true),
                None => return Ok(true),
                Some(_) => {}
            }
        }
        Ok(false)
    }
}

/// One range draft: the leading exact field values expanded so far. The next
/// field's typed interval bounds ride on the expansion loop, not the draft.
struct Draft {
    prefix: Vec<Value>,
}

impl Draft {
    fn prefix() -> Self {
        Self { prefix: Vec::new() }
    }

    fn with_point(prefix: &[Value], value: Value) -> Self {
        let mut extended = prefix.to_vec();
        extended.push(value);
        Self { prefix: extended }
    }
}

/// Materializes one draft as a directory byte range.
///
/// The end is the byte successor of the prefix plus upper bound when present,
/// so a String upper bound conservatively includes extended values that the
/// plan's typed checks reject later.
fn materialize_range(
    index: LogicalIndexId,
    types: &[DataType],
    prefix_values: &[Value],
    lo: Option<&Value>,
    hi: Option<&Value>,
) -> Result<KeyRange> {
    let prefix = TreeKey::encode(&types[..prefix_values.len()], prefix_values)?;
    let ty = types.get(prefix_values.len());
    let lower = match lo {
        Some(value) => Some(encode_scalar(ty, value)?),
        None => None,
    };
    let upper = match hi {
        Some(value) => Some(encode_scalar(ty, value)?),
        None => None,
    };
    Ok(keys::tree_manifest_plan_range(
        index,
        prefix.as_bytes(),
        lower.as_deref(),
        upper.as_deref(),
    ))
}

/// Merges overlapping or adjacent byte ranges, preserving order.
fn merge_ranges(mut ranges: Vec<KeyRange>) -> Vec<KeyRange> {
    let mut merged: Vec<KeyRange> = Vec::with_capacity(ranges.len());
    for range in ranges.drain(..) {
        match merged.last_mut() {
            Some(last) if last.end() >= range.start() => {
                if range.end() > last.end() {
                    *last = KeyRange::new(last.start().to_vec(), range.end().to_vec());
                }
            }
            _ => merged.push(range),
        }
    }
    merged
}

/// Encodes one scalar field bound using the Tree Key codec.
fn encode_scalar(ty: Option<&DataType>, value: &Value) -> Result<Vec<u8>> {
    let ty = ty.ok_or_else(Error::invalid_argument)?;
    Ok(
        TreeKey::encode(std::slice::from_ref(ty), std::slice::from_ref(value))?
            .as_bytes()
            .to_vec(),
    )
}

fn interval_is_empty(ty: DataType, interval: &Interval) -> bool {
    match (&interval.lo, &interval.hi) {
        (Some(lo), Some(hi)) => bound_order(lo, hi) != Ordering::Less,
        (None, Some(hi)) => bound_order(&min_value(ty), hi) != Ordering::Less,
        _ => false,
    }
}

fn cmp_intervals(left: &Interval, right: &Interval) -> Ordering {
    cmp_lower(&left.lo, &right.lo).then_with(|| cmp_upper(&left.hi, &right.hi))
}

fn cmp_lower(left: &Option<Value>, right: &Option<Value>) -> Ordering {
    match (left, right) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Less,
        (Some(_), None) => Ordering::Greater,
        (Some(left), Some(right)) => bound_order(left, right),
    }
}

fn cmp_upper(left: &Option<Value>, right: &Option<Value>) -> Ordering {
    match (left, right) {
        (None, None) => Ordering::Equal,
        (None, Some(_)) => Ordering::Greater,
        (Some(_), None) => Ordering::Less,
        (Some(left), Some(right)) => bound_order(left, right),
    }
}

/// Whether `[last.lo, last.hi)` and `[next.lo, next.hi)` overlap or touch.
fn overlaps_or_adjacent(last: &Interval, next: &Interval) -> bool {
    match (&next.lo, &last.hi) {
        (_, None) | (None, _) => true,
        (Some(lo), Some(hi)) => bound_order(lo, hi) != Ordering::Greater,
    }
}

fn max_lo(left: Option<Value>, right: Option<Value>) -> Option<Value> {
    match (left, right) {
        (None, other) | (other, None) => other,
        (Some(left), Some(right)) => match bound_order(&left, &right) {
            Ordering::Less => Some(right),
            _ => Some(left),
        },
    }
}

fn min_hi(left: Option<Value>, right: Option<Value>) -> Option<Value> {
    match (left, right) {
        (None, other) | (other, None) => other,
        (Some(left), Some(right)) => match bound_order(&left, &right) {
            Ordering::Greater => Some(right),
            _ => Some(left),
        },
    }
}

fn max_hi(left: Option<Value>, right: Option<Value>) -> Option<Value> {
    match (left, right) {
        (None, _) | (_, None) => None,
        (Some(left), Some(right)) => match bound_order(&left, &right) {
            Ordering::Less => Some(right),
            _ => Some(left),
        },
    }
}

/// Orders two validated same-domain values.
///
/// Constraint values pass [`check_typed`] against one field type, so
/// cross-domain or non-finite values never reach this helper.
fn bound_order(left: &Value, right: &Value) -> Ordering {
    typed_order(left, right).unwrap_or(Ordering::Equal)
}

/// The typed successor of a canonical non-NULL value, when one exists.
fn next_value(ty: DataType, value: &Value) -> Result<Option<Value>> {
    Ok(match (ty, value) {
        (DataType::Bool, Value::Bool(false)) => Some(Value::Bool(true)),
        (DataType::I64, Value::I64(value)) => value.checked_add(1).map(Value::I64),
        (DataType::F64, Value::F64(value)) => next_finite(*value).map(Value::F64),
        (DataType::String, Value::String(value)) => {
            next_string(value, MAX_STRING_BYTES)?.map(Value::String)
        }
        _ => None,
    })
}

/// The smallest String value strictly greater than `value`, when one exists
/// within `max_len` bytes.
///
/// Strings are ordered by UTF-8 bytes. Below the length ceiling the successor
/// appends a NUL byte. At the ceiling, the successor keeps the longest prefix
/// whose final character can still grow: either the last byte of a multi-byte
/// character increments within its continuation range, or one character is
/// replaced by the smallest greater character. The construction always yields
/// valid UTF-8; a failure to do so is an internal error and fails closed.
fn next_string(value: &str, max_len: usize) -> Result<Option<String>> {
    let bytes = value.as_bytes();
    if bytes.len() < max_len {
        let mut next = value.to_owned();
        next.push('\0');
        return Ok(Some(next));
    }
    for k in (0..bytes.len()).rev() {
        let (lead_pos, char_len) = char_span(bytes, k);
        for candidate in (u16::from(bytes[k]) + 1)..=0xFF {
            let candidate = candidate as u8;
            let tail: &[u8] = if lead_pos < k {
                // `v[k]` is a continuation byte: the character must stay
                // intact, so the candidate has to remain a legal
                // continuation and the remaining positions complete with
                // minimal continuation bytes.
                if candidate > 0xBF {
                    break;
                }
                let position = k - lead_pos;
                if position == 1 && !first_continuation_valid(bytes[lead_pos], candidate) {
                    continue;
                }
                &[0x80; 3][..char_len - position - 1]
            } else {
                let Some(tail) = minimal_tail(candidate) else {
                    continue;
                };
                tail
            };
            if k + 1 + tail.len() <= max_len {
                let mut next = Vec::with_capacity(k + 1 + tail.len());
                next.extend_from_slice(&bytes[..k]);
                next.push(candidate);
                next.extend_from_slice(tail);
                return String::from_utf8(next).map(Some).map_err(|_| corruption());
            }
        }
    }
    Ok(None)
}

/// The lead-byte position and byte length of the character containing byte
/// `k` of a valid UTF-8 string.
fn char_span(bytes: &[u8], k: usize) -> (usize, usize) {
    let mut lead_pos = k;
    while lead_pos > 0 && bytes[lead_pos] & 0xC0 == 0x80 {
        lead_pos -= 1;
    }
    let char_len = match bytes[lead_pos] {
        0x00..=0x7F => 1,
        0xC2..=0xDF => 2,
        0xE0..=0xEF => 3,
        _ => 4,
    };
    (lead_pos, char_len)
}

/// Whether `byte` is a legal continuation at the first continuation position
/// of lead byte `lead`; later positions accept every continuation byte.
fn first_continuation_valid(lead: u8, byte: u8) -> bool {
    match lead {
        0xE0 => byte >= 0xA0,
        0xED => byte <= 0x9F,
        0xF0 => byte >= 0x90,
        0xF4 => byte <= 0x8F,
        _ => (0x80..=0xBF).contains(&byte),
    }
}

/// The minimal continuation tail completing a character that starts with lead
/// byte `lead`, or `None` when `lead` cannot start a character.
fn minimal_tail(lead: u8) -> Option<&'static [u8]> {
    match lead {
        0x00..=0x7F => Some(&[]),
        0xC2..=0xDF => Some(&[0x80]),
        0xE0 => Some(&[0xA0, 0x80]),
        0xE1..=0xEF => Some(&[0x80, 0x80]),
        0xF0 => Some(&[0x90, 0x80, 0x80]),
        0xF1..=0xF3 => Some(&[0x80, 0x80, 0x80]),
        0xF4 => Some(&[0x80, 0x80, 0x80]),
        _ => None,
    }
}

/// The typed minimum of one scalar domain.
fn min_value(ty: DataType) -> Value {
    match ty {
        DataType::Bool => Value::Bool(false),
        DataType::I64 => Value::I64(i64::MIN),
        DataType::F64 => Value::F64(f64::MIN),
        DataType::String => Value::String(String::new()),
    }
}

/// The next representable finite f64, computed with explicit bit arithmetic
/// so the planner stays on MSRV 1.85 without `f64::next_up`.
fn next_finite(value: f64) -> Option<f64> {
    let bits = value.to_bits();
    let next = if value >= 0.0 {
        bits.checked_add(1)?
    } else {
        bits.checked_sub(1)?
    };
    let mut candidate = f64::from_bits(next);
    if candidate == 0.0 {
        candidate = 0.0;
    }
    candidate.is_finite().then_some(candidate)
}

/// Rejects predicate values outside a Tree Key field's canonical domain.
fn check_typed(ty: DataType, value: &Value) -> Result<()> {
    let valid = match (ty, value) {
        (DataType::Bool, Value::Bool(_)) | (DataType::I64, Value::I64(_)) => true,
        (DataType::F64, Value::F64(value)) => {
            value.is_finite() && (*value != 0.0 || value.to_bits() == 0.0_f64.to_bits())
        }
        (DataType::String, Value::String(value)) => value.len() <= MAX_STRING_BYTES,
        _ => false,
    };
    if valid {
        Ok(())
    } else {
        Err(Error::invalid_argument())
    }
}

fn corruption() -> Error {
    Error::new(ErrorKind::Corruption)
}

fn limit_exceeded() -> Error {
    Error::new(ErrorKind::LimitExceeded)
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::ops::Bound;

    use bytes::Bytes;
    use proptest::prelude::*;

    use crate::api::{
        CompareOp, DataType, FieldId, FieldSchema, IndexConfig, LogicalIndexId, Metric,
        PartitionKey, Value,
    };
    use crate::storage::ReadLogicalTxn;
    use crate::storage::backend::{ReadOps, ScanItem, ScanLimits, ScanPage};
    use crate::storage::keys::{self, KeyRange, TreeKey};
    use crate::storage::values::{
        BloomParameters, IndexLifecycle, IndexManifest, PersistentValue, TreeManifest, ValueCodec,
    };

    use super::*;

    fn id(value: u64) -> LogicalIndexId {
        LogicalIndexId::new(value).expect("test Logical Index ID is nonzero")
    }

    fn pk(value: u64) -> PartitionKey {
        PartitionKey::new(value).expect("test Partition Key is nonzero")
    }

    fn test_manifest(fields: Vec<FieldSchema>, tree_fields: Vec<FieldId>) -> IndexManifest {
        let bloom = fields
            .iter()
            .map(|field| BloomParameters::derive(field.synopsis()).expect("valid synopsis"))
            .collect();
        let config = IndexConfig::new(1, Metric::L2)
            .expect("valid config")
            .with_fields(fields)
            .expect("valid fields")
            .with_tree_key_fields(tree_fields)
            .expect("valid tree key fields");
        IndexManifest::new(IndexLifecycle::Active, id(1), config, [0; 32], bloom)
            .expect("valid manifest")
    }

    fn i64_field(name: &str) -> FieldSchema {
        FieldSchema::new(name, DataType::I64).expect("valid field")
    }

    fn string_field(name: &str) -> FieldSchema {
        FieldSchema::new(name, DataType::String).expect("valid field")
    }

    pub(super) fn i64_value(value: i64) -> Value {
        Value::I64(value)
    }

    pub(super) fn string_value(value: &str) -> Value {
        Value::String(value.to_owned())
    }

    pub(super) fn compare(field: u16, op: CompareOp, value: Value) -> Predicate {
        Predicate::Compare {
            field: FieldId(field),
            op,
            value,
        }
    }

    /// The expected directory range for one planned field expansion.
    fn expected_range(
        index: LogicalIndexId,
        types: &[DataType],
        prefix_values: &[Value],
        lo: Option<&Value>,
        hi: Option<&Value>,
    ) -> KeyRange {
        let prefix = TreeKey::encode(&types[..prefix_values.len()], prefix_values)
            .expect("canonical prefix");
        let encode = |ty: DataType, value: &Value| {
            TreeKey::encode(std::slice::from_ref(&ty), std::slice::from_ref(value))
                .expect("canonical bound")
                .as_bytes()
                .to_vec()
        };
        let ty = types.get(prefix_values.len()).copied();
        let lower = lo.map(|value| encode(ty.expect("bound implies a field type"), value));
        let upper = hi.map(|value| encode(ty.expect("bound implies a field type"), value));
        keys::tree_manifest_plan_range(index, prefix.as_bytes(), lower.as_deref(), upper.as_deref())
    }

    fn assert_ordered_and_disjoint(plan: &TreeKeyPlan) {
        for pair in plan.ranges().windows(2) {
            assert!(
                pair[0].end() <= pair[1].start(),
                "ranges must be ordered and disjoint"
            );
        }
    }

    // --- Planning ---

    #[test]
    fn no_predicate_and_no_tree_fields_plan_the_complete_directory() {
        let manifest = test_manifest(vec![], vec![]);
        let plan = plan_tree_keys(&manifest, None, 1_024).expect("plan");
        assert_eq!(plan.ranges().len(), 1);
        assert_eq!(plan.ranges()[0], keys::tree_manifest_range(id(1)));
        assert!(!plan.is_empty());
        assert!(
            plan.accepts(&TreeKey::encode(&[], &[]).expect("empty key"))
                .expect("accepts")
        );
    }

    #[test]
    fn equality_derives_one_string_point_prefix() {
        let manifest = test_manifest(vec![string_field("a")], vec![FieldId(0)]);
        let plan = plan_tree_keys(
            &manifest,
            Some(&compare(0, CompareOp::Eq, string_value("x"))),
            1_024,
        )
        .expect("plan");
        let expected =
            keys::tree_manifest_prefix_range(id(1), &[DataType::String], &[string_value("x")])
                .expect("prefix range");
        assert_eq!(plan.ranges(), &[expected]);
        assert_ordered_and_disjoint(&plan);
        for (value, accepted) in [
            ("x", true),
            ("xy", false),
            ("y", false),
            ("", false),
            ("x\u{0}", false),
        ] {
            let tree_key = TreeKey::encode(&[DataType::String], &[string_value(value)])
                .expect("canonical key");
            assert_eq!(
                plan.accepts(&tree_key).expect("accepts"),
                accepted,
                "{value:?}"
            );
        }
    }

    #[test]
    fn in_merges_adjacent_values_and_orders_points() {
        let manifest = test_manifest(vec![i64_field("a")], vec![FieldId(0)]);
        let plan = plan_tree_keys(
            &manifest,
            Some(&Predicate::In {
                field: FieldId(0),
                values: vec![i64_value(2), i64_value(1), i64_value(3)],
            }),
            1_024,
        )
        .expect("plan");
        // The adjacent points 1, 2, 3 merge into one half-open interval.
        let expected = expected_range(
            id(1),
            &[DataType::I64],
            &[],
            Some(&i64_value(1)),
            Some(&i64_value(4)),
        );
        assert_eq!(plan.ranges(), &[expected]);
        for value in [1, 2, 3] {
            let tree_key =
                TreeKey::encode(&[DataType::I64], &[i64_value(value)]).expect("canonical key");
            assert!(plan.accepts(&tree_key).expect("accepts"));
        }
        let tree_key = TreeKey::encode(&[DataType::I64], &[i64_value(4)]).expect("canonical key");
        assert!(!plan.accepts(&tree_key).expect("accepts"));
    }

    #[test]
    fn bounded_range_on_the_next_field_narrows_a_point_prefix() {
        let manifest = test_manifest(
            vec![i64_field("a"), i64_field("b")],
            vec![FieldId(0), FieldId(1)],
        );
        let predicate = Predicate::And(vec![
            compare(0, CompareOp::Eq, i64_value(5)),
            compare(1, CompareOp::GreaterOrEqual, i64_value(10)),
            compare(1, CompareOp::Lt, i64_value(20)),
        ]);
        let plan = plan_tree_keys(&manifest, Some(&predicate), 1_024).expect("plan");
        let expected = expected_range(
            id(1),
            &[DataType::I64, DataType::I64],
            &[i64_value(5)],
            Some(&i64_value(10)),
            Some(&i64_value(20)),
        );
        assert_eq!(plan.ranges(), &[expected]);
        assert_ordered_and_disjoint(&plan);
        for (key, accepted) in [
            ([5, 10], true),
            ([5, 19], true),
            ([5, 20], false),
            ([4, 10], false),
            ([6, 15], false),
            ([5, 9], false),
        ] {
            let values = key.map(i64_value).to_vec();
            let tree_key =
                TreeKey::encode(&[DataType::I64, DataType::I64], &values).expect("canonical key");
            assert_eq!(
                plan.accepts(&tree_key).expect("accepts"),
                accepted,
                "{key:?}"
            );
        }
    }

    #[test]
    fn a_range_on_an_earlier_field_blocks_later_narrowing() {
        let manifest = test_manifest(
            vec![i64_field("a"), i64_field("b")],
            vec![FieldId(0), FieldId(1)],
        );
        let predicate = Predicate::And(vec![
            compare(0, CompareOp::GreaterOrEqual, i64_value(1)),
            compare(0, CompareOp::Lt, i64_value(10)),
            compare(1, CompareOp::Eq, i64_value(7)),
        ]);
        let plan = plan_tree_keys(&manifest, Some(&predicate), 1_024).expect("plan");
        let expected = expected_range(
            id(1),
            &[DataType::I64, DataType::I64],
            &[],
            Some(&i64_value(1)),
            Some(&i64_value(10)),
        );
        assert_eq!(plan.ranges(), &[expected]);
        // The b == 7 constraint stays as a typed check on the widened range.
        for (key, accepted) in [
            ([1, 7], true),
            ([9, 7], true),
            ([1, 8], false),
            ([10, 7], false),
            ([0, 7], false),
        ] {
            let values = key.map(i64_value).to_vec();
            let tree_key =
                TreeKey::encode(&[DataType::I64, DataType::I64], &values).expect("canonical key");
            assert_eq!(
                plan.accepts(&tree_key).expect("accepts"),
                accepted,
                "{key:?}"
            );
        }
    }

    #[test]
    fn widening_keeps_the_typed_checks_when_expansion_exceeds_the_limit() {
        let manifest = test_manifest(vec![i64_field("a")], vec![FieldId(0)]);
        let predicate = Predicate::In {
            field: FieldId(0),
            values: vec![i64_value(1), i64_value(3), i64_value(5)],
        };
        let plan = plan_tree_keys(&manifest, Some(&predicate), 2).expect("plan");
        assert_eq!(plan.ranges(), &[keys::tree_manifest_range(id(1))]);
        for value in [1, 3, 5] {
            let tree_key =
                TreeKey::encode(&[DataType::I64], &[i64_value(value)]).expect("canonical key");
            assert!(plan.accepts(&tree_key).expect("accepts"));
        }
        for value in [2, 4] {
            let tree_key =
                TreeKey::encode(&[DataType::I64], &[i64_value(value)]).expect("canonical key");
            assert!(!plan.accepts(&tree_key).expect("accepts"));
        }
    }

    #[test]
    fn or_over_different_fields_widens_to_the_complete_directory() {
        let manifest = test_manifest(
            vec![i64_field("a"), i64_field("b")],
            vec![FieldId(0), FieldId(1)],
        );
        let predicate = Predicate::Or(vec![
            compare(0, CompareOp::Eq, i64_value(1)),
            compare(1, CompareOp::Eq, i64_value(2)),
        ]);
        let plan = plan_tree_keys(&manifest, Some(&predicate), 1_024).expect("plan");
        assert_eq!(plan.ranges(), &[keys::tree_manifest_range(id(1))]);
        let values = [i64_value(1), i64_value(2)];
        let tree_key =
            TreeKey::encode(&[DataType::I64, DataType::I64], &values).expect("canonical key");
        assert!(plan.accepts(&tree_key).expect("accepts"));
    }

    #[test]
    fn negation_of_one_field_complements_exactly() {
        let manifest = test_manifest(vec![i64_field("a")], vec![FieldId(0)]);
        let plan = plan_tree_keys(
            &manifest,
            Some(&Predicate::Not(Box::new(compare(
                0,
                CompareOp::Eq,
                i64_value(5),
            )))),
            1_024,
        )
        .expect("plan");
        // The two complement halves are unbounded toward the domain ends, so
        // the ranges tile the directory and merge into the complete range.
        assert_eq!(plan.ranges(), &[keys::tree_manifest_range(id(1))]);
        assert_ordered_and_disjoint(&plan);
        for (value, accepted) in [
            (i64::MIN, true),
            (4, true),
            (5, false),
            (6, true),
            (i64::MAX, true),
        ] {
            let tree_key =
                TreeKey::encode(&[DataType::I64], &[i64_value(value)]).expect("canonical key");
            assert_eq!(
                plan.accepts(&tree_key).expect("accepts"),
                accepted,
                "{value}"
            );
        }
    }

    #[test]
    fn negation_of_multiple_fields_widens_conservatively() {
        let manifest = test_manifest(
            vec![i64_field("a"), i64_field("b")],
            vec![FieldId(0), FieldId(1)],
        );
        let predicate = Predicate::Not(Box::new(Predicate::And(vec![
            compare(0, CompareOp::Eq, i64_value(1)),
            compare(1, CompareOp::Eq, i64_value(2)),
        ])));
        let plan = plan_tree_keys(&manifest, Some(&predicate), 1_024).expect("plan");
        assert_eq!(plan.ranges(), &[keys::tree_manifest_range(id(1))]);
    }

    #[test]
    fn is_null_on_a_tree_field_plans_nothing() {
        let manifest = test_manifest(vec![i64_field("a")], vec![FieldId(0)]);
        let plan =
            plan_tree_keys(&manifest, Some(&Predicate::IsNull(FieldId(0))), 1_024).expect("plan");
        assert!(plan.is_empty());
        assert!(plan.ranges().is_empty());
    }

    #[test]
    fn is_not_null_on_a_tree_field_is_unbounded() {
        let manifest = test_manifest(vec![i64_field("a")], vec![FieldId(0)]);
        let plan = plan_tree_keys(&manifest, Some(&Predicate::IsNotNull(FieldId(0))), 1_024)
            .expect("plan");
        assert_eq!(plan.ranges(), &[keys::tree_manifest_range(id(1))]);
    }

    #[test]
    fn an_empty_in_plans_nothing() {
        let manifest = test_manifest(vec![i64_field("a")], vec![FieldId(0)]);
        let plan = plan_tree_keys(
            &manifest,
            Some(&Predicate::In {
                field: FieldId(0),
                values: vec![],
            }),
            1_024,
        )
        .expect("plan");
        assert!(plan.is_empty());
    }

    #[test]
    fn a_conjunction_with_a_non_tree_leaf_keeps_the_pure_part() {
        let manifest = test_manifest(
            vec![
                i64_field("a"),
                FieldSchema::new("other", DataType::I64)
                    .expect("field")
                    .nullable(),
            ],
            vec![FieldId(0)],
        );
        let predicate = Predicate::And(vec![
            compare(0, CompareOp::Eq, i64_value(5)),
            Predicate::IsNull(FieldId(1)),
        ]);
        let plan = plan_tree_keys(&manifest, Some(&predicate), 1_024).expect("plan");
        let expected = keys::tree_manifest_prefix_range(id(1), &[DataType::I64], &[i64_value(5)])
            .expect("prefix range");
        assert_eq!(plan.ranges(), &[expected]);
    }

    #[test]
    fn a_disjunction_with_a_non_tree_leaf_widens() {
        let manifest = test_manifest(
            vec![
                i64_field("a"),
                FieldSchema::new("other", DataType::I64)
                    .expect("field")
                    .nullable(),
            ],
            vec![FieldId(0)],
        );
        let predicate = Predicate::Or(vec![
            compare(0, CompareOp::Eq, i64_value(5)),
            Predicate::IsNull(FieldId(1)),
        ]);
        let plan = plan_tree_keys(&manifest, Some(&predicate), 1_024).expect("plan");
        assert_eq!(plan.ranges(), &[keys::tree_manifest_range(id(1))]);
    }

    #[test]
    fn negation_of_an_impure_conjunction_widens() {
        let manifest = test_manifest(
            vec![
                i64_field("a"),
                FieldSchema::new("other", DataType::I64)
                    .expect("field")
                    .nullable(),
            ],
            vec![FieldId(0)],
        );
        let predicate = Predicate::Not(Box::new(Predicate::And(vec![
            compare(0, CompareOp::Eq, i64_value(5)),
            Predicate::IsNull(FieldId(1)),
        ])));
        let plan = plan_tree_keys(&manifest, Some(&predicate), 1_024).expect("plan");
        assert_eq!(plan.ranges(), &[keys::tree_manifest_range(id(1))]);
    }

    #[test]
    fn invalid_field_references_and_wrong_types_fail_validation() {
        let manifest = test_manifest(
            vec![i64_field("a"), string_field("b")],
            vec![FieldId(0), FieldId(1)],
        );
        assert_eq!(
            plan_tree_keys(
                &manifest,
                Some(&compare(7, CompareOp::Eq, i64_value(1))),
                1_024,
            )
            .expect_err("unknown field")
            .kind(),
            ErrorKind::InvalidArgument
        );
        assert_eq!(
            plan_tree_keys(
                &manifest,
                Some(&compare(1, CompareOp::Eq, i64_value(1))),
                1_024,
            )
            .expect_err("wrong type")
            .kind(),
            ErrorKind::InvalidArgument
        );
        assert_eq!(
            plan_tree_keys(&manifest, None, 0)
                .expect_err("zero range limit")
                .kind(),
            ErrorKind::InvalidArgument
        );
    }

    #[test]
    fn extreme_integer_bounds_plan_emptily_or_exactly() {
        let manifest = test_manifest(vec![i64_field("a")], vec![FieldId(0)]);
        assert!(
            plan_tree_keys(
                &manifest,
                Some(&compare(0, CompareOp::Gt, i64_value(i64::MAX))),
                1_024,
            )
            .expect("plan")
            .is_empty()
        );
        assert!(
            plan_tree_keys(
                &manifest,
                Some(&compare(0, CompareOp::Lt, i64_value(i64::MIN))),
                1_024,
            )
            .expect("plan")
            .is_empty()
        );
        let plan = plan_tree_keys(
            &manifest,
            Some(&compare(0, CompareOp::GreaterOrEqual, i64_value(i64::MAX))),
            1_024,
        )
        .expect("plan");
        let expected = expected_range(
            id(1),
            &[DataType::I64],
            &[],
            Some(&i64_value(i64::MAX)),
            None,
        );
        assert_eq!(plan.ranges(), &[expected]);
        let tree_key =
            TreeKey::encode(&[DataType::I64], &[i64_value(i64::MAX)]).expect("canonical key");
        assert!(plan.accepts(&tree_key).expect("accepts"));
        let tree_key =
            TreeKey::encode(&[DataType::I64], &[i64_value(i64::MAX - 1)]).expect("canonical key");
        assert!(!plan.accepts(&tree_key).expect("accepts"));
    }

    #[test]
    fn boolean_points_cover_both_values_and_merge_to_full() {
        let manifest = test_manifest(
            vec![FieldSchema::new("a", DataType::Bool).expect("field")],
            vec![FieldId(0)],
        );
        let plan = plan_tree_keys(
            &manifest,
            Some(&Predicate::In {
                field: FieldId(0),
                values: vec![Value::Bool(true), Value::Bool(false)],
            }),
            1_024,
        )
        .expect("plan");
        // Adjacent Boolean points merge into one interval covering the whole
        // two-value domain; materialized it starts at the false encoding,
        // which is also every directory key's first possible byte here.
        assert_eq!(plan.ranges().len(), 1);
        for value in [false, true] {
            let tree_key =
                TreeKey::encode(&[DataType::Bool], &[Value::Bool(value)]).expect("canonical key");
            assert!(plan.accepts(&tree_key).expect("accepts"));
        }
    }

    #[test]
    fn f64_boundaries_use_the_memcomparable_successor() {
        let manifest = test_manifest(
            vec![FieldSchema::new("a", DataType::F64).expect("field")],
            vec![FieldId(0)],
        );
        let plan = plan_tree_keys(
            &manifest,
            Some(&compare(0, CompareOp::Eq, Value::F64(1.0))),
            1_024,
        )
        .expect("plan");
        // The equality point becomes a complete prefix range over the
        // memcomparable encoding of 1.0.
        let expected =
            keys::tree_manifest_prefix_range(id(1), &[DataType::F64], &[Value::F64(1.0)])
                .expect("prefix range");
        assert_eq!(plan.ranges(), &[expected]);
        let tree_key =
            TreeKey::encode(&[DataType::F64], &[Value::F64(1.0)]).expect("canonical key");
        assert!(plan.accepts(&tree_key).expect("accepts"));
        let next = Value::F64(f64::from_bits(1.0_f64.to_bits() + 1));
        let tree_key = TreeKey::encode(&[DataType::F64], &[next]).expect("canonical key");
        assert!(!plan.accepts(&tree_key).expect("accepts"));
    }

    #[test]
    fn string_upper_bounds_cover_extensions_conservatively() {
        let manifest = test_manifest(vec![string_field("a")], vec![FieldId(0)]);
        let plan = plan_tree_keys(
            &manifest,
            Some(&compare(0, CompareOp::Lt, string_value("b"))),
            1_024,
        )
        .expect("plan");
        let expected = expected_range(
            id(1),
            &[DataType::String],
            &[],
            None,
            Some(&string_value("b")),
        );
        assert_eq!(plan.ranges(), &[expected]);
        for value in ["", "a", "a\u{0}", "aa"] {
            let tree_key = TreeKey::encode(&[DataType::String], &[string_value(value)])
                .expect("canonical key");
            assert!(plan.accepts(&tree_key).expect("accepts"), "{value:?}");
        }
        for value in ["b", "b\u{0}", "c"] {
            let tree_key = TreeKey::encode(&[DataType::String], &[string_value(value)])
                .expect("canonical key");
            assert!(!plan.accepts(&tree_key).expect("accepts"), "{value:?}");
        }
    }

    #[test]
    fn max_length_strings_have_no_successor_and_stay_exact_points() {
        let manifest = test_manifest(vec![string_field("a")], vec![FieldId(0)]);
        let long = "x".repeat(MAX_STRING_BYTES);
        let plan = plan_tree_keys(
            &manifest,
            Some(&compare(0, CompareOp::Eq, string_value(&long))),
            1_024,
        )
        .expect("plan");
        let tree_key =
            TreeKey::encode(&[DataType::String], &[string_value(&long)]).expect("canonical key");
        assert!(plan.accepts(&tree_key).expect("accepts"));
        let shorter = "x".repeat(MAX_STRING_BYTES - 1);
        let tree_key =
            TreeKey::encode(&[DataType::String], &[string_value(&shorter)]).expect("canonical key");
        assert!(!plan.accepts(&tree_key).expect("accepts"));
    }

    #[test]
    fn equality_on_every_field_plans_one_complete_prefix() {
        let manifest = test_manifest(
            vec![i64_field("a"), i64_field("b")],
            vec![FieldId(0), FieldId(1)],
        );
        let predicate = Predicate::And(vec![
            compare(0, CompareOp::Eq, i64_value(1)),
            compare(1, CompareOp::Eq, i64_value(2)),
        ]);
        let plan = plan_tree_keys(&manifest, Some(&predicate), 1_024).expect("plan");
        let expected = keys::tree_manifest_prefix_range(
            id(1),
            &[DataType::I64, DataType::I64],
            &[i64_value(1), i64_value(2)],
        )
        .expect("prefix range");
        assert_eq!(plan.ranges(), &[expected]);
        for (key, accepted) in [([1, 2], true), ([1, 3], false), ([2, 2], false)] {
            let values = key.map(i64_value).to_vec();
            let tree_key =
                TreeKey::encode(&[DataType::I64, DataType::I64], &values).expect("canonical key");
            assert_eq!(
                plan.accepts(&tree_key).expect("accepts"),
                accepted,
                "{key:?}"
            );
        }
    }

    #[test]
    fn next_finite_matches_ieee_successor_semantics() {
        assert_eq!(
            next_finite(0.0),
            Some(f64::from_bits(0x0000_0000_0000_0001))
        );
        assert_eq!(next_finite(f64::MAX), None);
        assert_eq!(
            next_finite(-f64::MAX),
            Some(f64::from_bits(0xFFEF_FFFF_FFFF_FFFE))
        );
        // The successor of the least negative value is canonical positive zero.
        assert_eq!(next_finite(-f64::from_bits(1)), Some(0.0));
        assert_eq!(
            next_finite(-1.0),
            Some(f64::from_bits(0xBFEF_FFFF_FFFF_FFFF))
        );
        assert!(next_finite(f64::MIN_POSITIVE).expect("finite") > f64::MIN_POSITIVE);
    }

    #[test]
    fn next_string_appends_nul_below_the_length_ceiling() {
        assert_eq!(
            next_string("É", MAX_STRING_BYTES).expect("successor"),
            Some("É\0".to_owned())
        );
    }

    #[test]
    fn next_string_grows_within_a_multibyte_character_at_the_ceiling() {
        // "É" is [0xC3, 0x89]; at a two-byte ceiling the successor increments
        // the continuation byte to [0xC3, 0x8A], which is U+00CA.
        assert_eq!(
            next_string("É", 2).expect("successor"),
            Some("Ê".to_owned())
        );
    }

    #[test]
    fn next_string_skips_the_surrogate_gap() {
        // U+D7FF is [0xED, 0x9F, 0xBF]; the next Unicode scalar value is
        // U+E000, because surrogates are not valid UTF-8.
        assert_eq!(
            next_string("\u{D7FF}", 3).expect("successor"),
            Some("\u{E000}".to_owned())
        );
    }

    #[test]
    fn next_string_reports_none_for_the_greatest_ceiling_length_string() {
        assert_eq!(next_string("\u{10FFFF}", 4).expect("successor"), None);
    }

    #[test]
    fn next_string_reports_none_when_only_longer_characters_remain() {
        // Every character above U+007F needs at least two bytes.
        assert_eq!(next_string("\u{7F}", 1).expect("successor"), None);
    }

    // --- Enumeration ---

    struct MockReadTxn {
        data: BTreeMap<Vec<u8>, Vec<u8>>,
    }

    impl MockReadTxn {
        fn new(items: impl IntoIterator<Item = (Vec<u8>, Vec<u8>)>) -> Self {
            Self {
                data: items.into_iter().collect(),
            }
        }
    }

    impl ReadOps for MockReadTxn {
        async fn get(&mut self, key: Bytes) -> Result<Option<Bytes>> {
            Ok(self.data.get(key.as_ref()).cloned().map(Bytes::from))
        }

        async fn batch_get(&mut self, keys: Vec<Bytes>) -> Result<Vec<Option<Bytes>>> {
            Ok(keys
                .iter()
                .map(|key| self.data.get(key.as_ref()).cloned().map(Bytes::from))
                .collect())
        }

        async fn scan(&mut self, range: &KeyRange, limits: ScanLimits) -> Result<ScanPage> {
            if limits.item_limit == 0 || limits.byte_limit == 0 {
                return Err(Error::invalid_argument());
            }
            let mut iter = self
                .data
                .range::<[u8], _>((Bound::Included(range.start()), Bound::Excluded(range.end())))
                .peekable();
            let mut items = Vec::new();
            let mut bytes = 0_usize;
            while let Some((key, value)) = iter.peek() {
                let size = key.len() + value.len();
                if items.is_empty() && size > limits.byte_limit {
                    let (key, value) = iter.next().expect("peeked item exists");
                    items.push(ScanItem::new(
                        Bytes::copy_from_slice(key),
                        Bytes::copy_from_slice(value),
                    ));
                    break;
                }
                if items.len() >= limits.item_limit || bytes + size > limits.byte_limit {
                    break;
                }
                let (key, value) = iter.next().expect("peeked item exists");
                items.push(ScanItem::new(
                    Bytes::copy_from_slice(key),
                    Bytes::copy_from_slice(value),
                ));
                bytes += size;
            }
            if iter.peek().is_some() {
                ScanPage::continued(items, keys::MAX_TREE_KEY_BYTES + 64)
            } else {
                Ok(ScanPage::terminal(items))
            }
        }
    }

    fn directory_item(
        manifest: &IndexManifest,
        types: &[DataType],
        values: &[Value],
    ) -> (Vec<u8>, Vec<u8>) {
        let tree_key = TreeKey::encode(types, values).expect("canonical key");
        let key = keys::tree_manifest_key(manifest.logical_index_id(), &tree_key);
        let value = ValueCodec::for_index(manifest)
            .encode(&PersistentValue::TreeManifest(
                TreeManifest::new(pk(1), pk(1)).expect("valid tree manifest"),
            ))
            .expect("encode");
        (key, value)
    }

    async fn enumerate(
        manifest: &IndexManifest,
        plan: &TreeKeyPlan,
        budget: u32,
        data: Vec<(Vec<u8>, Vec<u8>)>,
    ) -> TreeKeyEnumeration {
        let mut txn =
            ReadLogicalTxn::for_index(MockReadTxn::new(data), manifest).expect("bind manifest");
        enumerate_tree_keys(
            &mut txn,
            manifest,
            plan,
            budget,
            ScanLimits {
                item_limit: 1_024,
                byte_limit: 1 << 20,
            },
        )
        .await
        .expect("enumerate")
    }

    fn tree_values(enumeration: &TreeKeyEnumeration, types: &[DataType]) -> Vec<Vec<Value>> {
        enumeration
            .trees()
            .iter()
            .map(|tree| tree.tree_key().values(types).expect("canonical values"))
            .collect()
    }

    #[tokio::test]
    async fn enumeration_counts_decoded_keys_against_one_global_budget() {
        let manifest = test_manifest(vec![i64_field("a")], vec![FieldId(0)]);
        let types = [DataType::I64];
        let data: Vec<(Vec<u8>, Vec<u8>)> = (1..=5)
            .map(|value| directory_item(&manifest, &types, &[i64_value(value)]))
            .collect();
        let plan = plan_tree_keys(&manifest, None, 1_024).expect("plan");

        let short = enumerate(&manifest, &plan, 2, data.clone()).await;
        assert_eq!(short.scanned_tree_keys(), 2);
        assert!(short.scanned_tree_key_budget_exhausted());
        assert_eq!(
            tree_values(&short, &types),
            vec![vec![i64_value(1)], vec![i64_value(2)]]
        );
        for tree in short.trees() {
            assert_eq!(tree.manifest().root(), pk(1));
            assert_eq!(tree.manifest().partition_key_high_water(), pk(1));
        }

        let roomy = enumerate(&manifest, &plan, 10, data.clone()).await;
        assert_eq!(roomy.scanned_tree_keys(), 5);
        assert!(!roomy.scanned_tree_key_budget_exhausted());
        assert_eq!(
            tree_values(&roomy, &types),
            (1..=5)
                .map(|value| vec![i64_value(value)])
                .collect::<Vec<_>>()
        );

        let exact = enumerate(&manifest, &plan, 5, data).await;
        assert_eq!(exact.scanned_tree_keys(), 5);
        assert!(!exact.scanned_tree_key_budget_exhausted());
    }

    #[tokio::test]
    async fn enumeration_materializes_plan_eligible_keys_only() {
        let manifest = test_manifest(vec![string_field("a")], vec![FieldId(0)]);
        let types = [DataType::String];
        let data = vec![
            directory_item(&manifest, &types, &[string_value("a")]),
            directory_item(&manifest, &types, &[string_value("a\u{0}")]),
            directory_item(&manifest, &types, &[string_value("ab")]),
        ];
        let plan = plan_tree_keys(
            &manifest,
            Some(&compare(0, CompareOp::Eq, string_value("a"))),
            1_024,
        )
        .expect("plan");
        let enumeration = enumerate(&manifest, &plan, 10, data).await;
        // The NUL-extended value lies inside the byte prefix range and is
        // decoded and counted, but the typed check rejects it; "ab" lies
        // outside the range and is never read.
        assert_eq!(enumeration.scanned_tree_keys(), 2);
        assert_eq!(
            tree_values(&enumeration, &types),
            vec![vec![string_value("a")]]
        );
        assert!(!enumeration.scanned_tree_key_budget_exhausted());
    }

    #[tokio::test]
    async fn enumeration_merges_multiple_ranges_in_canonical_order() {
        let manifest = test_manifest(vec![i64_field("a")], vec![FieldId(0)]);
        let types = [DataType::I64];
        let data: Vec<(Vec<u8>, Vec<u8>)> = (1..=4)
            .map(|value| directory_item(&manifest, &types, &[i64_value(value)]))
            .collect();
        let plan = plan_tree_keys(
            &manifest,
            Some(&Predicate::In {
                field: FieldId(0),
                values: vec![i64_value(3), i64_value(1)],
            }),
            1_024,
        )
        .expect("plan");
        let enumeration = enumerate(&manifest, &plan, 10, data).await;
        assert_eq!(
            tree_values(&enumeration, &types),
            vec![vec![i64_value(1)], vec![i64_value(3)]]
        );
        assert_eq!(enumeration.scanned_tree_keys(), 2);
        assert!(!enumeration.scanned_tree_key_budget_exhausted());
    }

    #[tokio::test]
    async fn crossing_a_range_boundary_with_no_budget_reports_exhaustion() {
        let manifest = test_manifest(vec![i64_field("a")], vec![FieldId(0)]);
        let types = [DataType::I64];
        let data: Vec<(Vec<u8>, Vec<u8>)> = (1..=2)
            .map(|value| directory_item(&manifest, &types, &[i64_value(value)]))
            .collect();
        let plan = plan_tree_keys(
            &manifest,
            Some(&Predicate::In {
                field: FieldId(0),
                values: vec![i64_value(1), i64_value(2)],
            }),
            1_024,
        )
        .expect("plan");
        let enumeration = enumerate(&manifest, &plan, 1, data).await;
        assert_eq!(enumeration.scanned_tree_keys(), 1);
        assert!(enumeration.scanned_tree_key_budget_exhausted());
        assert_eq!(tree_values(&enumeration, &types), vec![vec![i64_value(1)]]);
    }

    #[tokio::test]
    async fn widened_plans_still_filter_and_stay_bounded() {
        let manifest = test_manifest(vec![i64_field("a")], vec![FieldId(0)]);
        let types = [DataType::I64];
        let data: Vec<(Vec<u8>, Vec<u8>)> = (1..=3)
            .map(|value| directory_item(&manifest, &types, &[i64_value(value)]))
            .collect();
        let plan = plan_tree_keys(
            &manifest,
            Some(&Predicate::In {
                field: FieldId(0),
                values: vec![i64_value(1), i64_value(2)],
            }),
            1,
        )
        .expect("plan");
        let enumeration = enumerate(&manifest, &plan, 10, data).await;
        assert_eq!(enumeration.scanned_tree_keys(), 3);
        assert_eq!(
            tree_values(&enumeration, &types),
            vec![vec![i64_value(1)], vec![i64_value(2)]]
        );
        assert!(!enumeration.scanned_tree_key_budget_exhausted());
    }

    #[tokio::test]
    async fn enumeration_pages_forward_with_one_item_pages() {
        let manifest = test_manifest(vec![i64_field("a")], vec![FieldId(0)]);
        let types = [DataType::I64];
        let data: Vec<(Vec<u8>, Vec<u8>)> = (1..=3)
            .map(|value| directory_item(&manifest, &types, &[i64_value(value)]))
            .collect();
        let plan = plan_tree_keys(&manifest, None, 1_024).expect("plan");
        let mut txn =
            ReadLogicalTxn::for_index(MockReadTxn::new(data), &manifest).expect("bind manifest");
        let enumeration = enumerate_tree_keys(
            &mut txn,
            &manifest,
            &plan,
            10,
            ScanLimits {
                item_limit: 1,
                byte_limit: 1 << 20,
            },
        )
        .await
        .expect("enumerate");
        assert_eq!(
            tree_values(&enumeration, &types),
            (1..=3)
                .map(|value| vec![i64_value(value)])
                .collect::<Vec<_>>()
        );
    }

    #[tokio::test]
    async fn an_empty_directory_enumerates_nothing() {
        let manifest = test_manifest(vec![i64_field("a")], vec![FieldId(0)]);
        let plan = plan_tree_keys(&manifest, None, 1_024).expect("plan");
        let enumeration = enumerate(&manifest, &plan, 10, Vec::new()).await;
        assert_eq!(enumeration.scanned_tree_keys(), 0);
        assert!(enumeration.trees().is_empty());
        assert!(!enumeration.scanned_tree_key_budget_exhausted());
    }

    #[tokio::test]
    async fn enumeration_fails_closed_on_a_wrong_value_kind() {
        let manifest = test_manifest(vec![i64_field("a")], vec![FieldId(0)]);
        let types = [DataType::I64];
        let (key, _) = directory_item(&manifest, &types, &[i64_value(1)]);
        let garbage = vec![0x00, 0x00];
        let plan = plan_tree_keys(&manifest, None, 1_024).expect("plan");
        let mut txn = ReadLogicalTxn::for_index(MockReadTxn::new(vec![(key, garbage)]), &manifest)
            .expect("bind manifest");
        let error = enumerate_tree_keys(
            &mut txn,
            &manifest,
            &plan,
            10,
            ScanLimits {
                item_limit: 1_024,
                byte_limit: 1 << 20,
            },
        )
        .await
        .expect_err("garbage value");
        assert_eq!(error.kind(), ErrorKind::Corruption);
    }

    #[tokio::test]
    async fn enumeration_fails_closed_on_a_malformed_directory_key() {
        let manifest = test_manifest(vec![i64_field("a")], vec![FieldId(0)]);
        let plan = plan_tree_keys(&manifest, None, 1_024).expect("plan");
        // The bare directory prefix is not a canonical Tree Key for a String
        // schema: the decoder must reject it rather than skip it.
        let key = keys::tree_manifest_range(id(1)).start().to_vec();
        let value = ValueCodec::for_index(&manifest)
            .encode(&PersistentValue::TreeManifest(
                TreeManifest::new(pk(1), pk(1)).expect("valid tree manifest"),
            ))
            .expect("encode");
        let mut txn = ReadLogicalTxn::for_index(MockReadTxn::new(vec![(key, value)]), &manifest)
            .expect("bind manifest");
        let error = enumerate_tree_keys(
            &mut txn,
            &manifest,
            &plan,
            10,
            ScanLimits {
                item_limit: 1_024,
                byte_limit: 1 << 20,
            },
        )
        .await
        .expect_err("malformed key");
        assert_eq!(error.kind(), ErrorKind::Corruption);
    }

    #[tokio::test]
    async fn a_lower_bounded_only_interval_enumerates_everything_above() {
        let manifest = test_manifest(vec![i64_field("a")], vec![FieldId(0)]);
        let types = [DataType::I64];
        let data: Vec<(Vec<u8>, Vec<u8>)> = [5_i64, 10, 15, 20]
            .iter()
            .map(|value| directory_item(&manifest, &types, &[i64_value(*value)]))
            .collect();
        let plan = plan_tree_keys(
            &manifest,
            Some(&compare(0, CompareOp::GreaterOrEqual, i64_value(10))),
            1_024,
        )
        .expect("plan");
        let enumeration = enumerate(&manifest, &plan, 10, data).await;
        assert_eq!(
            tree_values(&enumeration, &types),
            vec![
                vec![i64_value(10)],
                vec![i64_value(15)],
                vec![i64_value(20)]
            ]
        );
        assert!(!enumeration.scanned_tree_key_budget_exhausted());
    }

    #[tokio::test]
    async fn not_equal_enumerates_both_sides_of_the_excluded_point() {
        let manifest = test_manifest(vec![i64_field("a")], vec![FieldId(0)]);
        let types = [DataType::I64];
        let data: Vec<(Vec<u8>, Vec<u8>)> = [4_i64, 5, 6, 7]
            .iter()
            .map(|value| directory_item(&manifest, &types, &[i64_value(*value)]))
            .collect();
        let plan = plan_tree_keys(
            &manifest,
            Some(&compare(0, CompareOp::NotEq, i64_value(5))),
            1_024,
        )
        .expect("plan");
        let enumeration = enumerate(&manifest, &plan, 10, data).await;
        assert_eq!(
            tree_values(&enumeration, &types),
            vec![vec![i64_value(4)], vec![i64_value(6)], vec![i64_value(7)]]
        );
    }

    #[tokio::test]
    async fn a_string_lower_bound_enumerates_every_greater_value() {
        let manifest = test_manifest(vec![string_field("a")], vec![FieldId(0)]);
        let types = [DataType::String];
        let data: Vec<(Vec<u8>, Vec<u8>)> = ["a", "b", "c"]
            .iter()
            .map(|value| directory_item(&manifest, &types, &[string_value(value)]))
            .collect();
        let plan = plan_tree_keys(
            &manifest,
            Some(&compare(0, CompareOp::Gt, string_value("a"))),
            1_024,
        )
        .expect("plan");
        let enumeration = enumerate(&manifest, &plan, 10, data).await;
        assert_eq!(
            tree_values(&enumeration, &types),
            vec![vec![string_value("b")], vec![string_value("c")]]
        );
    }

    // --- Property tests ---

    fn property_schema() -> Vec<FieldSchema> {
        vec![
            i64_field("a"),
            string_field("b"),
            FieldSchema::new("other", DataType::I64)
                .expect("field")
                .nullable(),
        ]
    }

    pub(super) fn property_manifest() -> IndexManifest {
        test_manifest(property_schema(), vec![FieldId(0), FieldId(1)])
    }

    fn property_types() -> [DataType; 2] {
        [DataType::I64, DataType::String]
    }

    fn small_i64() -> impl Strategy<Value = Value> {
        prop::sample::select(vec![i64::MIN, -2, -1, 0, 1, 2, 7, i64::MAX]).prop_map(i64_value)
    }

    fn small_string() -> impl Strategy<Value = Value> {
        prop::sample::select(vec![
            String::new(),
            "a".to_owned(),
            "aa".to_owned(),
            "b".to_owned(),
            "z".to_owned(),
            "a\u{0}".to_owned(),
            "x".repeat(MAX_STRING_BYTES),
        ])
        .prop_map(|value| string_value(&value))
    }

    fn compare_op() -> impl Strategy<Value = CompareOp> {
        prop::sample::select(vec![
            CompareOp::Eq,
            CompareOp::NotEq,
            CompareOp::Lt,
            CompareOp::LessOrEqual,
            CompareOp::Gt,
            CompareOp::GreaterOrEqual,
        ])
    }

    fn atom() -> impl Strategy<Value = Predicate> {
        prop_oneof![
            (compare_op(), small_i64()).prop_map(|(op, value)| compare(0, op, value)),
            (compare_op(), small_string()).prop_map(|(op, value)| compare(1, op, value)),
            prop::collection::vec(small_i64(), 0..=3).prop_map(|values| Predicate::In {
                field: FieldId(0),
                values
            }),
            prop::collection::vec(small_string(), 0..=3).prop_map(|values| Predicate::In {
                field: FieldId(1),
                values
            }),
            prop::sample::select(vec![
                Predicate::IsNull(FieldId(0)),
                Predicate::IsNotNull(FieldId(0)),
                Predicate::IsNull(FieldId(1)),
                Predicate::IsNotNull(FieldId(1)),
            ]),
        ]
    }

    /// Conjunctive tree-pure predicates derive exact single-field products.
    fn conjunctive() -> impl Strategy<Value = Predicate> {
        let leaf = prop_oneof![
            atom(),
            atom().prop_map(|child| Predicate::Not(Box::new(child))),
        ];
        prop_oneof![
            leaf.clone(),
            prop::collection::vec(leaf, 0..=3).prop_map(Predicate::And),
        ]
    }

    fn pure_predicate() -> impl Strategy<Value = Predicate> {
        let leaf = prop_oneof![
            atom(),
            atom().prop_map(|child| Predicate::Not(Box::new(child))),
        ];
        let inner = prop_oneof![
            leaf.clone(),
            prop::collection::vec(leaf, 0..=3).prop_map(Predicate::And),
        ];
        inner.prop_recursive(2, 8, 2, |deeper| {
            prop_oneof![
                deeper
                    .clone()
                    .prop_map(|child| Predicate::Not(Box::new(child))),
                prop::collection::vec(deeper.clone(), 0..=2).prop_map(Predicate::Or),
                prop::collection::vec(deeper, 0..=3).prop_map(Predicate::And),
            ]
        })
    }

    fn tree_key_values() -> impl Strategy<Value = Vec<Value>> {
        (small_i64(), small_string()).prop_map(|(a, b)| vec![a, b])
    }

    pub(super) fn oracle(predicate: &Predicate, key_values: &[Value]) -> bool {
        match predicate {
            Predicate::And(children) => children.iter().all(|child| oracle(child, key_values)),
            Predicate::Or(children) => children.iter().any(|child| oracle(child, key_values)),
            Predicate::Not(child) => !oracle(child, key_values),
            Predicate::Compare { field, op, value } => {
                let stored = &key_values[usize::from(field.0)];
                let ordering = typed_order(stored, value).expect("same domain");
                match op {
                    CompareOp::Eq => ordering == Ordering::Equal,
                    CompareOp::NotEq => ordering != Ordering::Equal,
                    CompareOp::Lt => ordering == Ordering::Less,
                    CompareOp::LessOrEqual => ordering != Ordering::Greater,
                    CompareOp::Gt => ordering == Ordering::Greater,
                    CompareOp::GreaterOrEqual => ordering != Ordering::Less,
                }
            }
            Predicate::In { field, values } => values.iter().any(|candidate| {
                typed_order(&key_values[usize::from(field.0)], candidate) == Some(Ordering::Equal)
            }),
            Predicate::IsNull(_) => false,
            Predicate::IsNotNull(_) => true,
        }
    }

    proptest! {
        #[test]
        fn plans_are_ordered_disjoint_and_never_miss_a_match(
            predicate in pure_predicate(),
            key in tree_key_values(),
        ) {
            let manifest = property_manifest();
            let plan = plan_tree_keys(&manifest, Some(&predicate), 8).expect("plan");
            assert_ordered_and_disjoint(&plan);
            let tree_key = TreeKey::encode(&property_types(), &key).expect("canonical key");
            if oracle(&predicate, &key) {
                prop_assert!(plan.accepts(&tree_key).expect("accepts"));
            }
        }

        #[test]
        fn every_oracle_match_lies_inside_a_planned_range(
            predicate in pure_predicate(),
            key in tree_key_values(),
        ) {
            let manifest = property_manifest();
            let plan = plan_tree_keys(&manifest, Some(&predicate), 8).expect("plan");
            let tree_key = TreeKey::encode(&property_types(), &key).expect("canonical key");
            if oracle(&predicate, &key) {
                // Enumeration only scans the planned ranges, so every matching
                // Tree Key must byte-order inside at least one of them.
                let directory_key = keys::tree_manifest_key(manifest.logical_index_id(), &tree_key);
                let covered = plan.ranges().iter().any(|range| {
                    range.start() <= directory_key.as_slice()
                        && directory_key.as_slice() < range.end()
                });
                prop_assert!(covered, "matching key outside every planned range");
            }
        }

        #[test]
        fn conjunctive_plans_agree_exactly_with_the_oracle(
            predicate in conjunctive(),
            key in tree_key_values(),
        ) {
            let manifest = property_manifest();
            let plan = plan_tree_keys(&manifest, Some(&predicate), 8).expect("plan");
            let tree_key = TreeKey::encode(&property_types(), &key).expect("canonical key");
            prop_assert_eq!(plan.accepts(&tree_key).expect("accepts"), oracle(&predicate, &key));
        }
    }
}

#[cfg(test)]
mod exhaustive {
    use super::tests::{compare, i64_value, oracle, property_manifest, string_value};
    use crate::api::{CompareOp, DataType, FieldId, Predicate, Value};
    use crate::storage::keys::{self, TreeKey};

    use super::plan_tree_keys;

    fn format_predicate(predicate: &Predicate) -> String {
        match predicate {
            Predicate::And(children) => format!(
                "And[{}]",
                children
                    .iter()
                    .map(format_predicate)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Predicate::Or(children) => format!(
                "Or[{}]",
                children
                    .iter()
                    .map(format_predicate)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Predicate::Not(child) => format!("Not({})", format_predicate(child)),
            Predicate::Compare { field, op, value } => {
                format!(
                    "Compare(field {}, {op:?}, {})",
                    field.0,
                    format_value(value)
                )
            }
            Predicate::In { field, values } => format!(
                "In(field {}, [{}])",
                field.0,
                values
                    .iter()
                    .map(format_value)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Predicate::IsNull(field) => format!("IsNull({})", field.0),
            Predicate::IsNotNull(field) => format!("IsNotNull({})", field.0),
        }
    }

    fn format_value(value: &Value) -> String {
        match value {
            Value::Null => "NULL".to_owned(),
            Value::Bool(value) => format!("bool({value})"),
            Value::I64(value) => format!("i64({value})"),
            Value::F64(value) => format!("f64({value})"),
            Value::String(value) => format!("str({value:?})"),
        }
    }

    #[test]
    fn small_domain_plans_never_miss_a_match() {
        let manifest = property_manifest();
        let types = [DataType::I64, DataType::String];
        let i64_domain = [i64::MIN, -1, 0, 1, 2, i64::MAX];
        let string_domain = [
            String::new(),
            "a".to_owned(),
            "a\u{0}".to_owned(),
            "b".to_owned(),
            "z".to_owned(),
            "x".repeat(1024),
        ];

        let mut atoms = Vec::new();
        for value in i64_domain {
            let value = i64_value(value);
            for op in [
                CompareOp::Eq,
                CompareOp::NotEq,
                CompareOp::Lt,
                CompareOp::LessOrEqual,
                CompareOp::Gt,
                CompareOp::GreaterOrEqual,
            ] {
                atoms.push(compare(0, op, value.clone()));
            }
        }
        for value in &string_domain {
            let value = string_value(value);
            for op in [
                CompareOp::Eq,
                CompareOp::NotEq,
                CompareOp::Lt,
                CompareOp::LessOrEqual,
                CompareOp::Gt,
                CompareOp::GreaterOrEqual,
            ] {
                atoms.push(compare(1, op, value.clone()));
            }
        }
        for values in [
            vec![],
            vec![i64_value(0)],
            vec![i64_value(0), i64_value(1)],
            vec![i64_value(1), i64_value(2)],
        ] {
            atoms.push(Predicate::In {
                field: FieldId(0),
                values,
            });
        }
        for values in [
            vec![],
            vec![string_value("a")],
            vec![string_value("a"), string_value("a\u{0}")],
        ] {
            atoms.push(Predicate::In {
                field: FieldId(1),
                values,
            });
        }
        atoms.extend([
            Predicate::IsNull(FieldId(0)),
            Predicate::IsNotNull(FieldId(0)),
            Predicate::IsNull(FieldId(1)),
            Predicate::IsNotNull(FieldId(1)),
        ]);

        let mut leaves = atoms.clone();
        for atom in &atoms {
            leaves.push(Predicate::Not(Box::new(atom.clone())));
        }
        let mut predicates = leaves.clone();
        for left in &leaves {
            for right in &leaves {
                predicates.push(Predicate::And(vec![left.clone(), right.clone()]));
                predicates.push(Predicate::Or(vec![left.clone(), right.clone()]));
            }
        }
        for left in &leaves {
            predicates.push(Predicate::Not(Box::new(Predicate::And(vec![
                left.clone(),
                left.clone(),
            ]))));
        }

        let mut conjunctive = leaves.clone();
        for left in &leaves {
            for right in &leaves {
                conjunctive.push(Predicate::And(vec![left.clone(), right.clone()]));
            }
        }
        for predicate in &conjunctive {
            let plan = plan_tree_keys(&manifest, Some(predicate), 8).expect("plan");
            for a in i64_domain {
                for s in &string_domain {
                    let key = vec![i64_value(a), string_value(s)];
                    let expected = oracle(predicate, &key);
                    let tree_key = TreeKey::encode(&types, &key).expect("canonical key");
                    let accepted = plan.accepts(&tree_key).expect("accepts");
                    assert_eq!(
                        accepted,
                        expected,
                        "inexact conjunctive plan: {} for key [{}, {:?}]",
                        format_predicate(predicate),
                        a,
                        s
                    );
                }
            }
        }
        for predicate in &predicates {
            let plan = plan_tree_keys(&manifest, Some(predicate), 8).expect("plan");
            for a in i64_domain {
                for s in &string_domain {
                    let key = vec![i64_value(a), string_value(s)];
                    let expected = oracle(predicate, &key);
                    let tree_key = TreeKey::encode(&types, &key).expect("canonical key");
                    let accepted = plan.accepts(&tree_key).expect("accepts");
                    assert!(
                        accepted || !expected,
                        "missed a match: {} for key [{}, {:?}]",
                        format_predicate(predicate),
                        a,
                        s
                    );
                    if expected {
                        // Enumeration only scans the planned ranges, so every
                        // matching Tree Key must byte-order inside one of them.
                        let directory_key =
                            keys::tree_manifest_key(manifest.logical_index_id(), &tree_key);
                        assert!(
                            plan.ranges().iter().any(|range| {
                                range.start() <= directory_key.as_slice()
                                    && directory_key.as_slice() < range.end()
                            }),
                            "matching key outside every planned range: {} for key [{}, {:?}]",
                            format_predicate(predicate),
                            a,
                            s
                        );
                    }
                }
            }
        }
    }
}
