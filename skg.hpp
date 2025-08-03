#ifndef SKG_SKG_HPP_
#define SKG_SKG_HPP_

#include <cstdint>
#include <cmath>
#include <algorithm>
#include <array>
#include <concepts>
#include <memory>
#include <span>
#include <stdexcept>
#include <limits>
#include <ranges>
#include <cmath> 
#include <string_view>
#include <concepts>

namespace skg {

template<typename T>
concept Numeric = std::integral<T> || std::floating_point<T>;

template<typename T>
concept EdgeWeight = std::floating_point<T>;

inline constexpr std::size_t kDefaultScalingFactor = 15;
inline constexpr std::size_t kDefaultEdgeFactor = 20;
inline constexpr float kDefaultProbability = 0.25f;
inline constexpr std::size_t kMaxProcesses = 1024;

// probability matrix for Kronecker graph generation
struct ProbabilityMatrix {
    std::array<float, 4> probabilities{kDefaultProbability, kDefaultProbability, 
                                       kDefaultProbability, kDefaultProbability};
    std::int64_t expected_edges{0};
    
    [[nodiscard]] constexpr float a() const noexcept { return probabilities[0]; }
    [[nodiscard]] constexpr float b() const noexcept { return probabilities[1]; }
    [[nodiscard]] constexpr float c() const noexcept { return probabilities[2]; }
    [[nodiscard]] constexpr float d() const noexcept { return probabilities[3]; }
    
    constexpr void set_a(float value) { probabilities[0] = value; }
    constexpr void set_b(float value) { probabilities[1] = value; }
    constexpr void set_c(float value) { probabilities[2] = value; }
    constexpr void set_d(float value) { 
        probabilities[3] = 1.0f - (probabilities[0] + probabilities[1] + probabilities[2]); 
    }
    
    [[nodiscard]] constexpr bool IsValid() const noexcept {
        float sum = probabilities[0] + probabilities[1] + probabilities[2] + probabilities[3];
        return std::abs(sum - 1.0f) < 1e-6f && 
               std::ranges::all_of(probabilities, [](float p) { return p >= 0.0f && p <= 1.0f; });
    }
    
    auto operator<=>(const ProbabilityMatrix&) const = default;
};

template<EdgeWeight W = float>
struct Edge {
    std::int64_t source{0};
    std::int64_t target{0};
    W weight{0};
    
    Edge() = default;
    Edge(std::int64_t s, std::int64_t t, W w) : source(s), target(t), weight(w) {}
    
    auto operator<=>(const Edge&) const = default;
};

using DefaultEdge = Edge<float>;

#ifdef GPU_ENABLED
extern "C" {
    [[nodiscard]] DefaultEdge* CreateEdgeListGPU(
        std::span<const std::int64_t> edge_distribution,
        std::int64_t process_edges,
        std::int64_t nodes_per_process,
        std::int32_t num_processes,
        const std::array<float, 4>& probability_matrix,
        std::unique_ptr<std::int32_t[]>& node_edge_count,
        float& execution_time_ms
    );
}
#endif

// compressed sparse row matrix representation
template<typename IndexType = std::int64_t, EdgeWeight WeightType = float>
class CSRMatrix {
public:
    using index_type = IndexType;
    using weight_type = WeightType;
    
private:
    std::unique_ptr<index_type[]> row_ptr_;
    std::unique_ptr<index_type[]> col_indices_;
    std::unique_ptr<weight_type[]> values_;
    index_type num_nodes_;
    index_type num_edges_;
    
public:
    CSRMatrix() = default;
    
    CSRMatrix(index_type nodes, index_type edges)
        : num_nodes_(nodes), num_edges_(edges) {
        row_ptr_ = std::make_unique<index_type[]>(nodes + 1);
        col_indices_ = std::make_unique<index_type[]>(edges);
        values_ = std::make_unique<weight_type[]>(edges);
    }
    
    // move semantics
    CSRMatrix(CSRMatrix&&) noexcept = default;
    CSRMatrix& operator=(CSRMatrix&&) noexcept = default;
    
    // must delete copy semantics to prevent accidental copies
    CSRMatrix(const CSRMatrix&) = delete;
    CSRMatrix& operator=(const CSRMatrix&) = delete;
    
    [[nodiscard]] constexpr index_type NumNodes() const noexcept { return num_nodes_; }
    [[nodiscard]] constexpr index_type NumEdges() const noexcept { return num_edges_; }
    
    [[nodiscard]] std::span<index_type> RowPointers() noexcept {
        return {row_ptr_.get(), static_cast<std::size_t>(num_nodes_ + 1)};
    }
    
    [[nodiscard]] std::span<const index_type> RowPointers() const noexcept {
        return {row_ptr_.get(), static_cast<std::size_t>(num_nodes_ + 1)};
    }
    
    [[nodiscard]] std::span<index_type> ColumnIndices() noexcept {
        return {col_indices_.get(), static_cast<std::size_t>(num_edges_)};
    }
    
    [[nodiscard]] std::span<const index_type> ColumnIndices() const noexcept {
        return {col_indices_.get(), static_cast<std::size_t>(num_edges_)};
    }
    
    [[nodiscard]] std::span<weight_type> Values() noexcept {
        return {values_.get(), static_cast<std::size_t>(num_edges_)};
    }
    
    [[nodiscard]] std::span<const weight_type> Values() const noexcept {
        return {values_.get(), static_cast<std::size_t>(num_edges_)};
    }
    
    [[nodiscard]] auto GetNeighbors(index_type node) const {
        struct NeighborIterator {
            const index_type* col_ptr;
            const weight_type* val_ptr;
            index_type current;
            index_type end;
            
            struct NeighborInfo {
                index_type target;
                weight_type weight;
            };
            
            [[nodiscard]] bool operator!=(const NeighborIterator& other) const {
                return current != other.current;
            }
            
            NeighborIterator& operator++() {
                ++current;
                return *this;
            }
            
            [[nodiscard]] NeighborInfo operator*() const {
                return {col_ptr[current], val_ptr[current]};
            }
        };
        
        struct NeighborRange {
            const index_type* col_ptr;
            const weight_type* val_ptr;
            index_type start;
            index_type end;
            
            [[nodiscard]] NeighborIterator begin() const {
                return {col_ptr, val_ptr, start, end};
            }
            
            [[nodiscard]] NeighborIterator end() const {
                return {col_ptr, val_ptr, end, end};
            }
        };
        
        return NeighborRange{
            col_indices_.get(),
            values_.get(),
            row_ptr_[node],
            row_ptr_[node + 1]
        };
    }
    
    [[nodiscard]] static CSRMatrix CreateFromEdgeList(
        std::span<const Edge<weight_type>> edges,
        index_type num_nodes);
};

struct PerformanceStats {
    double current_time{0.0};
    double average_time{0.0};
    double min_time{std::numeric_limits<double>::max()};
    double max_time{0.0};
    
    void Update(double time) noexcept {
        current_time = time;
        min_time = std::min(min_time, time);
        max_time = std::max(max_time, time);
    }
    
    [[nodiscard]] std::string ToString() const;
};

struct GraphConfig {
    std::size_t scaling_factor{kDefaultScalingFactor};
    std::size_t edge_factor{kDefaultEdgeFactor};
    ProbabilityMatrix probability_matrix{};
    bool use_gpu{false};
    std::string_view output_format{"binary"};
    
    [[nodiscard]] constexpr std::int64_t NumNodes() const noexcept {
        return 1LL << scaling_factor;
    }
    
    [[nodiscard]] constexpr std::int64_t ExpectedEdges() const noexcept {
        return NumNodes() * static_cast<std::int64_t>(edge_factor);
    }
};

class StochasticKroneckerGenerator {
private:
    GraphConfig config_;
    std::int32_t mpi_rank_;
    std::int32_t mpi_size_;
    
public:
    explicit StochasticKroneckerGenerator(GraphConfig config, 
                                         std::int32_t rank = 0, 
                                         std::int32_t size = 1)
        : config_(std::move(config)), mpi_rank_(rank), mpi_size_(size) {
        if ((mpi_size_ & (mpi_size_ - 1)) != 0) {
            throw std::invalid_argument("Number of MPI processes must be a power of 2");
        }
    }
    
    [[nodiscard]] std::unique_ptr<CSRMatrix<>> Generate();
    [[nodiscard]] std::vector<std::int64_t> CalculateEdgeDistribution() const;
    
private:
    [[nodiscard]] std::unique_ptr<DefaultEdge[]> GenerateEdges(
        std::span<const std::int64_t> edge_distribution,
        std::unique_ptr<std::int32_t[]>& node_edge_count) const;
};

}  

#endif  