# Foreground Mutation and Structure Maintenance

Status: Implementation-ready

This module owns record routing, exact foreground membership changes, binary
K-means tree shape, and persistent split/merge state machines.

## 1. Foreground mutation protocol

Each attempt starts from a fresh write transaction and Active Manifest check.
For every Record ID, storage reads the existing Record and Record Location with
update protection. Upsert validates and computes the new Tree Key, routes through
the current searchable topology, and locks affected Headers/States. Delete uses
the exact stored location rather than approximate routing.

The atomic commit changes:

- Vector Record and Record Location;
- old/new Leaf Entries;
- exact source/target Header counts and cache epochs;
- every affected field synopsis, including Tree Key field synopses.

Tree Key changes are an atomic move between trees. A vector-changing upsert
re-encodes RaBitQ7; structural movement copies the absolute code unchanged.
Whole-attempt retries re-read topology and membership. There is no record
revision, stale membership cleanup, or repair branch.

After commit, one mutation batch coalesces its changed partitions and offers
only actionable final Headers: oversized Ready partitions, undersized
non-root Ready partitions, and durable split/merge source states. Healthy
Ready partitions and ReceivingSplit targets are not offered because their
current state cannot advance independently. Failure to enqueue never changes
the mutation result or correctness.

## 2. Tree shape

Each Tree Key lazily installs one initial leaf root with its Tree Manifest.
Leaf partitions are level 1; partitions above level 1 contain Child Entries.
The level therefore determines the partition kind without a second persistent
discriminator. Internal fanout is exactly two. Partition identities are never
reused. An ordinary non-root split allocates and exposes two new targets, then
removes the source at completion. Root Partition Key 1 remains stable and is
converted in place to an internal root only after its entries drain, so every committed state
has one searchable entry point.

Split training reads the complete source through one consistent snapshot and
runs at most ten deterministic Lloyd rounds outside the write transaction.
Initial centroids use deterministic farthest-pair seeding with canonical
tie-breakers. Each round orders entries by distance difference and assigns
exactly `floor(n/2)` to the left cluster, with identity tie-breaking. Assignment
stability stops early. Training output is not persistent authority; published
target centroids are routing models and concurrent source writes need not
restart training. Because Splitting accepts foreground writes, the source can
legally shrink below two entries before exposure; training then emits
degenerate centroids — the single entry replicated, or zero vectors when the
source is empty — so every committed split state can always advance.

Merge is eligible for a Ready non-root partition below the configured minimum.
The worker reselects a legal target for each bounded batch under current
topology rather than persisting or relying on a stale target. Roots never
collapse.

## 3. State-machine rules

Durable states are only `Ready`, `Splitting { left, right }`,
`ReceivingSplit { source }`, `DrainingSplit { left, right }`, and `Merging`.
State stores associated Partition Keys, codec version, and state-start time; it
stores no drain cursor, owner, lease, or fixed merge target. A fixup transaction
update-protects the relevant State, Headers, incoming references, and entry keys
it changes. Each step is bounded by adapter budgets. Structural drain starts
every batch from the current smallest source entry, and successful movement
deletes that prefix. Repeating a completed or conflicting step is harmless: it
either sees the next state or aborts and replans.

Traversal behavior is defined for every state. A source entry remains covered
until the same transaction installs its target copy and updates all exact
metadata. Entries never have zero or two authoritative leaf memberships in a
committed state.

## 4. Split

### 4.1 Start

A short start transaction update-protects a Ready partition above maximum,
revalidates its threshold, reserves never-reused target Partition Keys, and writes
`Splitting { left, right }` on the source. Each target is then independently
unique-created as `ReceivingSplit { source }` with its persisted centroid and,
for a non-root split, its Child Entry. Target creation update-protects the source
State and current incoming topology so a stale worker cannot recreate a target
after completion. Once both targets exist, one transaction changes the source
to `DrainingSplit { left, right }`. Splitting continues to accept foreground
writes; ReceivingSplit accepts writes and movement but cannot start its own
split or merge.

### 4.2 Drain

Each maintenance iteration reads a bounded source-entry page, then opens a short
write transaction and re-reads each entry and its authoritative routing data.
For every remaining entry it deterministically chooses a target, uniquely
inserts the target entry, deletes the source entry, and updates both exact
counts and cache epochs. Leaf movement also updates Record Location and target
Synopsis. A concurrently removed entry is skipped; any remaining membership
mismatch is Corruption.

The leaf page first derives the largest batch whose exact worst-case relocation
charge fits the current Backend Admission Budget. The charge uses the
Manifest's dimension, fields and Bloom parameters, the current Tree Key, codec
key/value sizes, adapter key-prefix overhead, and the operation's worst target
distribution: two persisted targets plus the bounded corrective candidate set
for split, or one distinct Ready target per entry for
merge. It then caps that safe bound at one quarter of the configured split
threshold, with a floor of eight, limiting conflict rollback and failure blast
radius relative to the index's ordinary partition size. Internal movement
retains a 128-entry contention cap, reduced when necessary by an exact encoded
Child Entry and Header mutation-byte calculation for the current dimension and
Backend Admission Budget.

Drain placement normally chooses the nearer persisted target centroid. Exact
remaining and target counts reserve the last entries needed for each target to
reach the configured minimum when the source has enough entries, crediting
entries already redirected into the targets. Thus duplicate-heavy routing
cannot immediately merge a small target back into its oversized source, while
ordinary inputs retain the distance-based placement learned by balanced
training. If concurrent deletes make both minima unattainable, nearest routing
continues and the normal merge protocol converges the undersized result.

Before each ordinary drain batch, a non-root split performs one bounded
same-level corrective pass. Leaf routing vectors are current Vector Records;
internal routing vectors are the immutable full-f32 centroids already stored in
Child Entries. The same strict-improvement rule and Partition Key tie-break
serve both kinds. A source entry may move to an existing Ready sibling only
when that sibling is strictly closer than both split targets and doing so still
leaves enough source entries to attain both target minima. Conversely, each
Ready sibling or cousin in the bounded candidate beam contributes at most one
128-entry, 1-MiB entry page. Each pass rotates through at most four candidates,
scans at most 512 entry envelopes, and screens a deterministic aggregate
sample of at most eight routing vectors; successive drain rounds cover
different candidates and page positions. The first page with qualifying entries supplies a
mutation-budget-sized batch to a ReceivingSplit target only when that target
is strictly closer than the current owner's immutable centroid. Configured
target capacity is enforced both by planning and at the atomic topology
boundary. An internal Ready source always retains at least one Child Entry, so
its incoming edge can never route to an empty body.

Corrective discovery has its own fixed maintenance beam of 16 and does not
change foreground routing. Starting at the root, each level expands at most 16
searchable partition bodies; each body contributes at most one 128-entry,
1-MiB Child Entry page, and only the nearest 16 child paths survive. The final
level admits at most 16 nonempty Ready candidates, excluding the split source
and targets. Corrective pulls rotate through up to four candidates that can
donate an entry; an internal donor must contain at least two Child Entries. A
fully exposed transitional split family is expanded as its
source and both targets after validating the family relationship. A partially
exposed ReceivingSplit target redirects to its still-complete Splitting source
body. A root split has no sibling and skips correction. Existing Child Entry
centroids are never refreshed.

New inserts route to the nearer target. Upsert relocates a source membership;
delete follows exact Record Location. Traversal visits the source plus both
targets while draining, so the state remains searchable if no worker runs again.

### 4.3 Complete

Exact source Header count zero is the sole proof that structural draining is
complete; no entry rescan is performed. For a non-root source, completion finds
and validates its unique incoming Child Entry. With transactional range clear,
one final transaction switches topology, promotes both targets to Ready, and
removes the source prefix. Without that capability, the exact zero count proves
the entry ranges empty without a rescan — the source prefix holds only its
fixed metadata keys — so the same final transaction revalidates zero count,
removes the incoming source edge and those fixed metadata keys with bounded
point deletes, and promotes the already exposed targets. Root completion
converts Partition Key 1 in place to a Ready internal root containing the two
target Child Entries.

## 5. Merge

A worker encountering an eligible Ready partition locks its incoming reference
and validates that at least one legal same-level Ready target exists before
changing the source to Merging.

Merging stores no fixed target and no drain cursor. Each bounded batch performs
ordinary same-level routing, skips the source and non-Ready candidates, and
selects the nearest current Ready target with canonical tie-breakers. Different
entries may move to different targets, and a target may cross the split
threshold. Transactions use the same atomic insert/delete, count, epoch,
synopsis, and Record Location rules as split. The source remains searchable
throughout and supports exact delete, but accepts no new insert. An upsert whose
Location still names the source atomically relocates to a current Ready target;
if none exists after bounded retry it returns ContentionExhausted.

After exact source count reaches zero, cleanup follows the same capability
branch as non-root split: transactional full-prefix clear when available;
otherwise the exact zero count proves the entry ranges empty, so the final
transaction removes the incoming reference and the source's fixed metadata
keys with bounded point deletes. No target state changes and no tombstone
remains.

If no legal target exists before merge begins, no state starts. If targets later
disappear, the searchable Merging source remains and later access retries; it
never creates a target or reverts to Ready.

## 6. Contention and recovery

Foreground mutations never wait for a whole split/merge. They may conflict with
one bounded fixup and retry from a fresh snapshot. Fixups do not hold locks
across transactions or reserve a durable owner.

State timestamps diagnose stalls but do not grant ownership. A future timestamp
is not considered stalled. Any process may advance an encountered state after
bounded admission. Conditional convergence requires rediscovery, repeated
admission, eventual backend success, and a legal merge target; correctness does
not require convergence.

Natural mismatches—wrong location, missing authoritative entry, duplicate
incoming reference, impossible count, or malformed state—return Corruption and
do not trigger automatic repair.

## 7. Validation

- State-machine tables test every state, allowed transition, traversal rule,
  retry point, and adapter capability branch.
- Model histories interleave mutation, split, merge, root transitions, process
  loss, queue loss, conflicts, and unknown outcomes while asserting exact
  membership and searchability after every commit.
- Focused tests cover Tree Key moves, monotonic synopsis expansion, allocator gaps,
  deterministic training ties, target reselection, terminal paged deletion, and
  cold intermediate states.
- Contention benchmarks measure transaction retries and write amplification;
  they do not weaken invariants to improve throughput.
