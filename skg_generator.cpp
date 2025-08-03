#include "skg.hpp"
#include <algorithm>
#include <execution>
#include <format>
#include <numeric>
#include <random>
#include <ranges>
#include <vector>
#include <mpi.h>

namespace skg {

namespace {

thread_local std::mt19937_64 g_random_engine{
    std::random_device{}() + static_cast<unsigned>(MPI_Wtime() * 1000000)
};

// helper: generate Kronecker edges using recursion
[[nodiscard]] DefaultEdge GenerateKroneckerEdge(
    std::int64_t dimension,
    std::int64_t start_index,
    const std::array<float, 4>& cumulative_probs) {
    
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::int64_t row = 0;
    std::int64_t col = 0;
    std::int64_t current_dim = dimension;
    const auto k = static_cast<int>(std::log2(dimension));
    
    for (int level = 0; level < k; ++level) {
        const float random_value = dist(g_random_engine);
        
        const auto quadrant = static_cast<int>(
            std::ranges::lower_bound(cumulative_probs, random_value) - cumulative_probs.begin()
        );
        
        const int row_bit = quadrant >> 1;
        const int col_bit = quadrant & 1;
        
        current_dim >>= 1;
        row += current_dim * row_bit;
        col += current_dim * col_bit;
    }
    
    return {row, start_index + col, dist(g_random_engine)};
}

// block allocation for edge distribution
[[nodiscard]] std::unique_ptr<std::int64_t[]> AllocateBlockMatrix(
    std::size_t rows, std::size_t cols) {
    
    auto matrix = std::make_unique<std::int64_t[]>(rows * cols);
    std::fill_n(matrix.get(), rows * cols, 0);
    return matrix;
}

} 

std::string PerformanceStats::ToString() const {
    #ifdef __cpp_lib_format
    return std::format("avg={:.5f}s, min={:.5f}s, max={:.5f}s",
                      average_time, min_time, max_time);
    #else
    char buffer[256];
    std::snprintf(buffer, sizeof(buffer), 
                 "avg=%.5fs, min=%.5fs, max=%.5fs",
                 average_time, min_time, max_time);
    return std::string(buffer);
    #endif
}

std::vector<std::int64_t> StochasticKroneckerGenerator::CalculateEdgeDistribution() const {
    const auto num_processes = static_cast<std::size_t>(mpi_size_);
    const auto log_processes = static_cast<int>(std::log2(num_processes));
    
    auto edge_blocks = AllocateBlockMatrix(num_processes, num_processes);
    const std::int64_t total_edges = config_.ExpectedEdges();
    
    std::fill_n(edge_blocks.get(), num_processes * num_processes, total_edges);
    
    for (int level = 0; level < log_processes; ++level) {
        const std::size_t num_blocks = 1ULL << level;
        const std::size_t grid_size = 1ULL << (log_processes - 1 - level);
        
        std::for_each(std::execution::par_unseq,
                     std::views::iota(0UZ, num_blocks).begin(),
                     std::views::iota(0UZ, num_blocks).end(),
                     [&](std::size_t block_i) {
            const std::size_t block_row = block_i * grid_size * 2;
            
            for (std::size_t block_j = 0; block_j < num_blocks; ++block_j) {
                const std::size_t block_col = block_j * grid_size * 2;
                
                for (int sub_i = 0; sub_i < 2; ++sub_i) {
                    const std::size_t grid_row = block_row + grid_size * sub_i;
                    
                    for (int sub_j = 0; sub_j < 2; ++sub_j) {
                        const std::size_t grid_col = block_col + grid_size * sub_j;
                        const float prob = config_.probability_matrix.probabilities[sub_i * 2 + sub_j];
                        
                        for (std::size_t x = 0; x < grid_size; ++x) {
                            for (std::size_t y = 0; y < grid_size; ++y) {
                                const std::size_t idx = (grid_row + x) * num_processes + (grid_col + y);
                                edge_blocks[idx] = static_cast<std::int64_t>(
                                    std::round(edge_blocks[idx] * prob)
                                );
                            }
                        }
                    }
                }
            }
        });
    }
    
    // edge distribution for current rank
    std::vector<std::int64_t> distribution(num_processes);
    std::copy_n(&edge_blocks[mpi_rank_ * num_processes], num_processes, distribution.begin());
    
    return distribution;
}

std::unique_ptr<DefaultEdge[]> StochasticKroneckerGenerator::GenerateEdges(
    std::span<const std::int64_t> edge_distribution,
    std::unique_ptr<std::int32_t[]>& node_edge_count) const {
    
    const std::int64_t nodes_per_process = config_.NumNodes() / mpi_size_;
    const std::int64_t total_edges = std::reduce(edge_distribution.begin(), edge_distribution.end());
    
    auto edges = std::make_unique<DefaultEdge[]>(total_edges);
    node_edge_count = std::make_unique<std::int32_t[]>(nodes_per_process);
    std::fill_n(node_edge_count.get(), nodes_per_process, 0);
    
    std::array<float, 4> cumulative_probs{};
    std::partial_sum(config_.probability_matrix.probabilities.begin(),
                    config_.probability_matrix.probabilities.end(),
                    cumulative_probs.begin());
    
    std::int64_t edge_index = 0;
    
    for (int block = 0; block < mpi_size_; ++block) {
        const std::int64_t block_edges = edge_distribution[mpi_size_ - 1 - block];
        const std::int64_t start_index = nodes_per_process * (mpi_size_ - 1 - block);
        
        if (block_edges > 10000) {
            std::vector<DefaultEdge> temp_edges(block_edges);
            
            std::for_each(std::execution::par_unseq,
                         temp_edges.begin(), temp_edges.end(),
                         [&](DefaultEdge& edge) {
                edge = GenerateKroneckerEdge(nodes_per_process, start_index, cumulative_probs);
            });
            for (const auto& edge : temp_edges) {
                edges[edge_index++] = edge;
                node_edge_count[edge.source]++;
            }
        } else {
            for (std::int64_t i = 0; i < block_edges; ++i) {
                edges[edge_index] = GenerateKroneckerEdge(
                    nodes_per_process, start_index, cumulative_probs
                );
                node_edge_count[edges[edge_index].source]++;
                ++edge_index;
            }
        }
    }
    
    return edges;
}

std::unique_ptr<CSRMatrix<>> StochasticKroneckerGenerator::Generate() {
    auto edge_distribution = CalculateEdgeDistribution();
    const std::int64_t process_edges = std::reduce(edge_distribution.begin(), 
                                                   edge_distribution.end());
    const std::int64_t nodes_per_process = config_.NumNodes() / mpi_size_;
    
    std::unique_ptr<std::int32_t[]> node_edge_count;
    std::unique_ptr<DefaultEdge[]> edges;
    
    #ifdef GPU_ENABLED
    if (config_.use_gpu) {
        float gpu_time = 0.0f;
        edges.reset(CreateEdgeListGPU(
            edge_distribution,
            process_edges,
            nodes_per_process,
            mpi_size_,
            config_.probability_matrix.probabilities,
            node_edge_count,
            gpu_time
        ));
    }
    #endif
    
    if (!edges) {
        edges = GenerateEdges(edge_distribution, node_edge_count);
    }
    
    auto csr = std::make_unique<CSRMatrix<>>(nodes_per_process, process_edges);
    auto row_ptrs = csr->RowPointers();
    auto col_indices = csr->ColumnIndices();
    auto values = csr->Values();
    
    row_ptrs[0] = 0;
    std::partial_sum(node_edge_count.get(), 
                    node_edge_count.get() + nodes_per_process,
                    row_ptrs.begin() + 1);
    
    auto temp_counts = std::make_unique<std::int32_t[]>(nodes_per_process);
    std::copy_n(node_edge_count.get(), nodes_per_process, temp_counts.get());
    
    for (std::int64_t i = 0; i < process_edges; ++i) {
        const auto& edge = edges[i];
        const auto row = edge.source;
        temp_counts[row]--;
        const auto idx = row_ptrs[row] + temp_counts[row];
        col_indices[idx] = edge.target;
        values[idx] = edge.weight;
    }
    
    std::for_each(std::execution::par_unseq,
                 std::views::iota(std::int64_t{0}, nodes_per_process).begin(),
                 std::views::iota(std::int64_t{0}, nodes_per_process).end(),
                 [&](std::int64_t row) {
        const auto start = row_ptrs[row];
        const auto end = row_ptrs[row + 1];
        
        std::vector<std::pair<std::int64_t, float>> row_edges;
        row_edges.reserve(end - start);
        
        for (auto idx = start; idx < end; ++idx) {
            row_edges.emplace_back(col_indices[idx], values[idx]);
        }
        
        std::ranges::sort(row_edges);
        
        for (std::size_t i = 0; i < row_edges.size(); ++i) {
            col_indices[start + i] = row_edges[i].first;
            values[start + i] = row_edges[i].second;
        }
    });
    
    return csr;
}

template<EdgeWeight W>
CSRMatrix<std::int64_t, W> CSRMatrix<std::int64_t, W>::CreateFromEdgeList(
    std::span<const Edge<W>> edges,
    std::int64_t num_nodes) {
    std::vector<std::int64_t> edge_counts(num_nodes, 0);
    for (const auto& edge : edges) {
        edge_counts[edge.source]++;
    }
    
    CSRMatrix<std::int64_t, W> csr(num_nodes, edges.size());
    auto row_ptrs = csr.RowPointers();
    auto col_indices = csr.ColumnIndices();
    auto values = csr.Values();
    row_ptrs[0] = 0;
    std::partial_sum(edge_counts.begin(), edge_counts.end(), row_ptrs.begin() + 1);
    
    std::fill(edge_counts.begin(), edge_counts.end(), 0);
    
    for (const auto& edge : edges) {
        const auto idx = row_ptrs[edge.source] + edge_counts[edge.source]++;
        col_indices[idx] = edge.target;
        values[idx] = edge.weight;
    }
    
    return csr;
}

template class CSRMatrix<std::int64_t, float>;
template class CSRMatrix<std::int64_t, double>;

}  