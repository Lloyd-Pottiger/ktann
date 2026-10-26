# Reuse committed internal bodies during foreground routing

The Foreground Mutation preparation phase may reuse and fill the Runtime's
existing Partition Cache for internal bodies. It still reads each partition's
Header and State from its own consistent transaction snapshot, and reuses a
body only when its identity, kind and cache epoch match that Header.

Foreground cache fills are restricted to grouped pre-mutation routing. At that point no
internal body changes have been staged: lazy Tree Key creation can create only
a new leaf root. All cached Child Entries therefore describe committed snapshot
data even if the later mutation aborts. General-purpose write routing, which can
run after arbitrary staged writes, does not use this cache. Leaf bodies, mutable
State and update-protected membership validation are not cached by routing.

A miss retains the existing lockstep bounded scans. Complete bodies must match
the snapshot Header's exact count before publication; an over-count fails early.
The existing byte capacity and eviction policy govern retained bodies, and the
sum of one wave's prospective fill buffers is also bounded by that capacity.
Exact routing streams plain bodies when retention is disabled or the body exceeds
cache capacity. HNSW routing requires a complete validated graph body within the
routing working-byte limit even when it cannot retain that body in the cache.
Historical snapshots cannot replace a newer cached epoch. Both paths share the
same cache; they need no pinned root or persistent cache metadata.

Commit safety remains at the existing owner: routing update-protects each
selected leaf Header and incoming Child Entry. Concurrent topology changes
therefore abort and retry the complete mutation; a cache hit never substitutes
for that validation. A cache miss or eviction changes work, not routing choices.
