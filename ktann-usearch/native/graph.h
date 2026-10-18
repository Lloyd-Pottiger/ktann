#pragma once
#include "rust/cxx.h"
#include <cstdint>
#include <memory>

namespace ktann_usearch {
struct MetricContext;
struct Hit;
struct Stats;
struct GraphState;
struct WorkspaceState;

class NativeWorkspace {
  public:
    explicit NativeWorkspace(std::unique_ptr<WorkspaceState> state);
    ~NativeWorkspace();
    std::size_t allocated_bytes() const;
    std::size_t allocation_count() const;
    std::unique_ptr<WorkspaceState> state;
};

class NativeGraph {
  public:
    explicit NativeGraph(std::unique_ptr<GraphState> state);
    ~NativeGraph();
    Stats search(std::uint32_t ef, MetricContext const&, NativeWorkspace&, rust::Slice<Hit>) const;
    std::size_t resident_bytes() const;
    std::size_t peak_bytes() const;
    std::uint32_t entry() const;
    std::uint32_t level(std::uint32_t node) const;
    std::size_t neighbors(std::uint32_t node, std::uint32_t level, rust::Slice<std::uint32_t>) const;
    bool repaired() const;
    std::unique_ptr<GraphState> state;
};

std::unique_ptr<NativeGraph> build(std::uint32_t n, std::uint32_t m, std::uint32_t ef,
    std::uint64_t seed, std::size_t limit, MetricContext const&);
std::unique_ptr<NativeWorkspace> workspace(std::uint32_t n, std::uint32_t m, std::size_t limit);
}
