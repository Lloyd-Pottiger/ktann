#include "ktann-usearch/src/lib.rs.h"
#include "native/graph.h"
// Upstream gates exception-enabled call frames on NDEBUG. Cancellation must
// unwind these frames in every profile, including builds setting CXXFLAGS.
#ifdef NDEBUG
#undef NDEBUG
#endif
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif
#include "usearch/index.hpp"
#if defined(__clang__) || defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
#include <algorithm>
#include <cmath>
#include <limits>
#include <new>
#include <stdexcept>

namespace ktann_usearch {
namespace {
using namespace unum::usearch;

// Each allocation carries its original account, so scratch and graph storage
// can be destroyed on a different thread. Mutation of an account is exclusive.
struct Account {
    std::size_t limit, used, peak, allocations = 0;
    Account(std::size_t limit, std::size_t metadata) : limit(limit), used(metadata), peak(metadata) {
        if (metadata > limit) throw std::runtime_error("limit");
    }
};

thread_local Account* active_account = nullptr;

struct AllocationScope {
    Account* previous;
    explicit AllocationScope(Account& account) : previous(active_account) { active_account = &account; }
    ~AllocationScope() { active_account = previous; }
};

struct AllocationHeader { Account* account; std::size_t charge; };

// USearch expects a non-throwing allocator returning null on failure. Charge
// payload, 64-byte alignment/header, and 64 bytes of allocator overhead. These
// are allocation bounds, not an assertion about OS pages or process RSS.
template <typename T> struct Allocator {
    using value_type = T;
    T* allocate(std::size_t n) noexcept {
        static_assert(alignof(T) <= 64, "unsupported alignment");
        if (!active_account || n > (SIZE_MAX - 191) / sizeof(T)) return nullptr;
        std::size_t bytes = (n * sizeof(T) + 63) & ~std::size_t(63);
        std::size_t charge = bytes + 128;
        auto& account = *active_account;
        if (charge > account.limit - account.used) return nullptr;
        auto* raw = static_cast<char*>(::operator new(bytes + 64, std::align_val_t(64), std::nothrow));
        if (!raw) return nullptr;
        new (raw) AllocationHeader{&account, charge};
        account.used += charge;
        ++account.allocations;
        account.peak = std::max(account.peak, account.used);
        return reinterpret_cast<T*>(raw + 64);
    }
    void deallocate(T* pointer, std::size_t) noexcept {
        if (!pointer) return;
        auto* raw = reinterpret_cast<char*>(pointer) - 64;
        auto* header = reinterpret_cast<AllocationHeader*>(raw);
        header->account->used -= header->charge;
        ::operator delete(raw, std::align_val_t(64));
    }
};

using Engine = index_gt<double, std::uint64_t, std::uint32_t, Allocator<char>>;

void prepare(Engine::context_t& context, std::size_t n, std::size_t m) {
    // At most n unique queued nodes. Reserve the worst-case visit lookahead
    // too, so upstream reserve calls cannot trigger query-time growth.
    if (!context.top_candidates.reserve(std::max(n, std::size_t(1))) ||
        !context.next_candidates.reserve(std::max(n, std::size_t(1))) ||
        !context.visits.reserve(n + 2 * m + 1))
        throw std::runtime_error("limit");
}

void checkpoint(MetricContext const& context) {
    if (!proceed(context)) throw std::runtime_error("cancelled");
}

// Only this callback computes distances; exported hits retain the same values.
// Rust callbacks complete normally; cancellation unwinds C++ frames only.
struct Metric {
    MetricContext const& context;
    std::size_t calls = 0;
    static std::uint32_t ordinal(std::uint32_t node) { return node; }
    template <typename T> static std::uint32_t ordinal(T const& node) { return get_slot(node); }
    template <typename A, typename B> double operator()(A const& a, B const& b) {
        if ((calls++ & 63) == 0) checkpoint(context);
        double value = measure(context, ordinal(a), ordinal(b));
        if (!std::isfinite(value)) throw std::runtime_error("invalid");
        return value;
    }
};
}

struct GraphState {
    // Destruction order matters: the account must outlive all engine storage.
    Account account;
    Engine engine;
    GraphState(std::uint32_t m, std::size_t limit)
        : account(limit, sizeof(GraphState) + sizeof(NativeGraph) + 128),
          engine(unum::usearch::index_config_t(m, 2 * m)) {}
};

struct WorkspaceState {
    Account account;
    Engine::context_t context;
    explicit WorkspaceState(std::size_t limit)
        : account(limit, sizeof(WorkspaceState) + sizeof(NativeWorkspace) + 128) {}
};

NativeGraph::NativeGraph(std::unique_ptr<GraphState> value) : state(std::move(value)) {}
NativeGraph::~NativeGraph() = default;
NativeWorkspace::NativeWorkspace(std::unique_ptr<WorkspaceState> value) : state(std::move(value)) {}
NativeWorkspace::~NativeWorkspace() = default;

std::unique_ptr<NativeWorkspace> workspace(std::uint32_t n, std::uint32_t m, std::size_t limit) {
    auto state = std::make_unique<WorkspaceState>(limit);
    AllocationScope scope(state->account);
    prepare(state->context, n, m);
    return std::make_unique<NativeWorkspace>(std::move(state));
}

std::unique_ptr<NativeGraph> build(std::uint32_t n, std::uint32_t m, std::uint32_t ef,
    std::uint64_t seed, std::size_t limit, MetricContext const& context) {
    checkpoint(context);
    auto state = std::make_unique<GraphState>(m, limit);
    AllocationScope scope(state->account);
    auto& engine = state->engine;
    if (n) {
        if (!engine.reserve({n, 1})) throw std::runtime_error("limit");
        auto& scratch = engine.build_context();
        prepare(scratch, n, m);
        scratch.level_generator = seed;
        Metric metric{context};
        for (std::uint32_t node = 0; node != n; ++node) {
            checkpoint(context);
            auto added = engine.add(node, node, metric, unum::usearch::index_update_config_t{ef, 0});
            if (!added) throw std::runtime_error("limit");
        }

        // Native reverse-link pruning can disconnect duplicate-heavy graphs.
        // Audit directed reachability. Only a disconnected graph gains the
        // canonical directed cycle; replacing at most one edge per full node
        // preserves the 2*M degree cap and makes all nodes reachable.
        unum::usearch::buffer_gt<std::uint32_t, Allocator<std::uint32_t>> queue(n);
        if (!queue) throw std::runtime_error("limit");
        auto& visited = scratch.visits;
        visited.clear();
        queue[0] = engine.entry_slot();
        visited.set(queue[0]);
        std::size_t tail = 1, edges = 0;
        for (std::size_t head = 0; head != tail; ++head) {
            checkpoint(context);
            for (auto neighbor : engine.neighbors(queue[head], 0)) {
                if ((edges++ & 63) == 0) checkpoint(context);
                auto node = static_cast<std::uint32_t>(neighbor.slot);
                if (!visited.set(node)) queue[tail++] = node;
            }
        }
        if (tail != n) {
            for (std::uint32_t node = 0; node != n; ++node) {
                checkpoint(context);
                engine.ensure_successor(node, (node + 1) % n);
            }
        }
        engine.release_build_context();
    }
    checkpoint(context);
    return std::make_unique<NativeGraph>(std::move(state));
}

Stats NativeGraph::search(std::uint32_t ef, MetricContext const& context, NativeWorkspace& workspace,
    rust::Slice<Hit> output) const {
    checkpoint(context);
    if (output.empty()) return {};
    AllocationScope scope(workspace.state->account);
    auto& scratch = workspace.state->context;
    Metric metric{context};
    std::uint32_t query = state->engine.size();
    if (output.size() == query) {
        // Full coverage needs one score per node. Fill once and sort once;
        // upstream's sorted-buffer insertion would move O(n²) hits here.
        for (std::uint32_t node = 0; node != query; ++node)
            output[node] = Hit{node, metric(query, node)};
        std::size_t comparisons = 0;
        std::sort(output.begin(), output.end(), [&](Hit const& a, Hit const& b) {
            if ((comparisons++ & 63) == 0) checkpoint(context);
            return a.distance < b.distance || (a.distance == b.distance && a.node < b.node);
        });
        checkpoint(context);
        return Stats{metric.calls, 0};
    }
    auto result = state->engine.search(query, output.size(), metric,
        unum::usearch::index_search_config_t{ef, 0, false},
        unum::usearch::dummy_predicate_t{}, unum::usearch::dummy_prefetch_t{}, &scratch);
    if (!result) throw std::runtime_error("limit");
    if (result.size() != output.size()) throw std::runtime_error("incomplete graph search");
    for (std::size_t i = 0; i != output.size(); ++i) {
        if ((i & 63) == 0) checkpoint(context);
        output[i] = Hit{static_cast<std::uint32_t>(result[i].member.slot), result[i].distance};
    }
    checkpoint(context);
    return Stats{metric.calls, result.visited_members};
}

// Pin the generic native format consumed by the safe Rust validator. A USearch
// upgrade changing the header or scalar layout requires a codec audit.
static_assert(sizeof(unum::usearch::index_serialized_header_t) == 40);
static_assert(offsetof(unum::usearch::index_serialized_header_t, entry_slot) == 32);

std::size_t NativeGraph::encoded_size() const { return state->engine.serialized_length(); }

void NativeGraph::save(rust::Slice<std::uint8_t> output, MetricContext const& context) const {
    checkpoint(context);
    std::size_t position = 0;
    bool cancelled = false;
    // Upstream stream functions are noexcept. Callbacks return false rather than
    // throw; the bridge reports cancellation only after native frames return.
    auto result = state->engine.save_to_stream([&](void const* source, std::size_t size) {
        if (size > output.size() - position) return false;
        auto* bytes = static_cast<std::uint8_t const*>(source);
        for (std::size_t copied = 0; copied != size;) {
            if (!proceed(context)) { cancelled = true; return false; }
            auto count = std::min(size - copied, std::size_t(65536));
            std::memcpy(output.data() + position, bytes + copied, count);
            copied += count;
            position += count;
        }
        return true;
    });
    if (cancelled) { result.error.release(); throw std::runtime_error("cancelled"); }
    if (!result || position != output.size()) {
        result.error.release();
        throw std::runtime_error("native serialization failed");
    }
    checkpoint(context);
}

// Called only after Graph::decode validates the same immutable borrowed bytes.
std::unique_ptr<NativeGraph> restore(rust::Slice<const std::uint8_t> bytes,
    std::uint32_t n, std::uint32_t m, std::size_t limit, MetricContext const& context) {
    checkpoint(context);
    auto state = std::make_unique<GraphState>(m, limit);
    AllocationScope scope(state->account);
    std::size_t position = 0;
    bool cancelled = false;
    auto result = state->engine.load_from_stream([&](void* target, std::size_t size) {
        if (size > bytes.size() - position) return false;
        auto* output = static_cast<std::uint8_t*>(target);
        for (std::size_t copied = 0; copied != size;) {
            if (!proceed(context)) { cancelled = true; return false; }
            auto count = std::min(size - copied, std::size_t(65536));
            std::memcpy(output + copied, bytes.data() + position, count);
            copied += count;
            position += count;
        }
        return true;
    });
    if (cancelled) { result.error.release(); throw std::runtime_error("cancelled"); }
    if (!result) throw std::runtime_error(result.error.release());
    if (position != bytes.size() || state->engine.size() != n)
        throw std::runtime_error("native deserialization failed");
    state->engine.release_build_context();
    checkpoint(context);
    return std::make_unique<NativeGraph>(std::move(state));
}

std::size_t NativeGraph::resident_bytes() const { return state->account.used; }
std::size_t NativeGraph::peak_bytes() const { return state->account.peak; }
std::uint32_t NativeGraph::entry() const { return state->engine.entry_slot(); }
std::uint32_t NativeGraph::level(std::uint32_t node) const { return state->engine.level_of(node); }
std::size_t NativeWorkspace::allocated_bytes() const { return state->account.used; }
std::size_t NativeWorkspace::allocation_count() const { return state->account.allocations; }

std::size_t NativeGraph::neighbors(std::uint32_t node, std::uint32_t level, rust::Slice<std::uint32_t> output) const {
    std::size_t i = 0;
    for (auto neighbor : state->engine.neighbors(node, level)) output[i++] = neighbor.slot;
    return i;
}
}
