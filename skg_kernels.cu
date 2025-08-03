#include "skg.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cooperative_groups.h>
#include <cub/cub.cuh>

namespace cg = cooperative_groups;

namespace skg {
namespace cuda {

constexpr int kBlockSize = 256;
constexpr int kWarpSize = 32;
constexpr int kMaxGridSize = 65535;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(error)); \
            cudaDeviceReset(); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

__device__ __forceinline__ DefaultEdge GenerateKroneckerEdgeDevice(
    curandState_t* state,
    int64_t dimension,
    int64_t start_index,
    const float* __restrict__ cumulative_probs) {
    
    int64_t row = 0;
    int64_t col = 0;
    int64_t current_dim = dimension;
    const int k = __popc(__brev(dimension - 1)) + 1;  
    
    #pragma unroll 4
    for (int level = 0; level < k; ++level) {
        const float random_value = curand_uniform(state);
        
        int quadrant = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (random_value >= cumulative_probs[i]) {
                quadrant = i + 1;
            }
        }
        quadrant = min(quadrant, 3);
        
        const int row_bit = quadrant >> 1;
        const int col_bit = quadrant & 1;
        
        current_dim >>= 1;
        row = __fma_rn(current_dim, row_bit, row);  // fused multiply-add
        col = __fma_rn(current_dim, col_bit, col);
    }
    
    return DefaultEdge{row, start_index + col, curand_uniform(state)};
}

__global__ void GenerateEdgesKernel(
    DefaultEdge* __restrict__ edges,
    int32_t* __restrict__ node_edge_count,
    const int64_t* __restrict__ edge_distribution,
    const float* __restrict__ probability_matrix,
    int64_t total_edges,
    int64_t nodes_per_process,
    int32_t num_processes,
    uint64_t seed) {
    
    cg::thread_block block = cg::this_thread_block();
    
    __shared__ float shared_cumulative_probs[4];
    __shared__ int64_t shared_edge_offsets[32];  
    
    if (block.thread_rank() < 4) {
        float cumsum = 0.0f;
        for (int i = 0; i <= block.thread_rank(); ++i) {
            cumsum += probability_matrix[i];
        }
        shared_cumulative_probs[block.thread_rank()] = cumsum;
    }
    
    if (block.thread_rank() < num_processes && block.thread_rank() < 32) {
        int64_t offset = 0;
        for (int i = 0; i <= block.thread_rank(); ++i) {
            offset += edge_distribution[num_processes - 1 - i];
        }
        shared_edge_offsets[block.thread_rank()] = offset;
    }
    
    block.sync();
    
    const int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t stride = gridDim.x * blockDim.x;
    
    curandState_t local_state;
    curand_init(seed, tid, 0, &local_state);
    
    for (int64_t edge_idx = tid; edge_idx < total_edges; edge_idx += stride) {
        int block_id = 0;
        for (int i = 0; i < min(num_processes, 32); ++i) {
            if (edge_idx < shared_edge_offsets[i]) {
                block_id = num_processes - 1 - i;
                break;
            }
        }
        
        const int64_t start_index = nodes_per_process * block_id;
        DefaultEdge edge = GenerateKroneckerEdgeDevice(
            &local_state,
            nodes_per_process,
            start_index,
            shared_cumulative_probs
        );
        edges[edge_idx] = edge;
        atomicAdd(&node_edge_count[edge.source], 1);
    }
}

class CudaMemoryPool {
private:
    std::vector<void*> allocations_;
    size_t total_allocated_{0};
    
public:
    ~CudaMemoryPool() {
        for (auto ptr : allocations_) {
            cudaFree(ptr);
        }
    }
    
    template<typename T>
    T* Allocate(size_t count) {
        size_t bytes = count * sizeof(T);
        T* ptr = nullptr;
        
        CUDA_CHECK(cudaMalloc(&ptr, bytes));
        allocations_.push_back(ptr);
        total_allocated_ += bytes;
        
        return ptr;
    }
    
    [[nodiscard]] size_t TotalAllocated() const { return total_allocated_; }
};

class EdgeGenerator {
private:
    int device_id_;
    cudaDeviceProp device_props_;
    std::vector<cudaStream_t> streams_;
    std::vector<cudaEvent_t> events_;
    
public:
    explicit EdgeGenerator(int device_id = 0) : device_id_(device_id) {
        CUDA_CHECK(cudaSetDevice(device_id_));
        CUDA_CHECK(cudaGetDeviceProperties(&device_props_, device_id_));
        
        const int num_streams = 4;
        streams_.resize(num_streams);
        events_.resize(num_streams);
        
        for (int i = 0; i < num_streams; ++i) {
            CUDA_CHECK(cudaStreamCreate(&streams_[i]));
            CUDA_CHECK(cudaEventCreate(&events_[i]));
        }
    }
    
    ~EdgeGenerator() {
        for (auto& stream : streams_) {
            cudaStreamDestroy(stream);
        }
        for (auto& event : events_) {
            cudaEventDestroy(event);
        }
    }
    
    [[nodiscard]] DefaultEdge* Generate(
        const std::vector<int64_t>& edge_distribution,
        int64_t total_edges,
        int64_t nodes_per_process,
        int32_t num_processes,
        const std::array<float, 4>& probability_matrix,
        std::unique_ptr<int32_t[]>& node_edge_count,
        float& execution_time_ms) {
        
        CudaMemoryPool memory_pool;
        
        auto* d_edges = memory_pool.Allocate<DefaultEdge>(total_edges);
        auto* d_node_edge_count = memory_pool.Allocate<int32_t>(nodes_per_process);
        auto* d_edge_distribution = memory_pool.Allocate<int64_t>(num_processes);
        auto* d_probability_matrix = memory_pool.Allocate<float>(4);
        
        CUDA_CHECK(cudaMemset(d_node_edge_count, 0, 
                              nodes_per_process * sizeof(int32_t)));
        
        CUDA_CHECK(cudaMemcpy(d_edge_distribution, edge_distribution.data(),
                              num_processes * sizeof(int64_t),
                              cudaMemcpyHostToDevice));
        
        CUDA_CHECK(cudaMemcpy(d_probability_matrix, probability_matrix.data(),
                              4 * sizeof(float),
                              cudaMemcpyHostToDevice));
        
        const int block_size = kBlockSize;
        const int grid_size = std::min(
            static_cast<int>((total_edges + block_size - 1) / block_size),
            device_props_.maxGridSize[0]
        );
        
        std::random_device rd;
        const uint64_t seed = rd();
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        
        CUDA_CHECK(cudaEventRecord(start));
        
        GenerateEdgesKernel<<<grid_size, block_size>>>(
            d_edges,
            d_node_edge_count,
            d_edge_distribution,
            d_probability_matrix,
            total_edges,
            nodes_per_process,
            num_processes,
            seed
        );
        
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        CUDA_CHECK(cudaEventElapsedTime(&execution_time_ms, start, stop));
        
        // Allocate host memory for results
        auto* h_edges = new DefaultEdge[total_edges];
        node_edge_count = std::make_unique<int32_t[]>(nodes_per_process);
        
        // Copy results back to host
        CUDA_CHECK(cudaMemcpy(h_edges, d_edges,
                              total_edges * sizeof(DefaultEdge),
                              cudaMemcpyDeviceToHost));
        
        CUDA_CHECK(cudaMemcpy(node_edge_count.get(), d_node_edge_count,
                              nodes_per_process * sizeof(int32_t),
                              cudaMemcpyDeviceToHost));
        
        // Cleanup events
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        
        return h_edges;
    }
};

}  // namespace cuda
}  // namespace skg

// C interface for integration
extern "C" {

skg::DefaultEdge* CreateEdgeListGPU(
    std::span<const std::int64_t> edge_distribution,
    std::int64_t process_edges,
    std::int64_t nodes_per_process,
    std::int32_t num_processes,
    const std::array<float, 4>& probability_matrix,
    std::unique_ptr<std::int32_t[]>& node_edge_count,
    float& execution_time_ms) {
    
    try {
        // from span to vector for CUDA
        std::vector<std::int64_t> edge_dist_vec(
            edge_distribution.begin(), 
            edge_distribution.end()
        );
        
        // form generator and generate edges
        skg::cuda::EdgeGenerator generator;
        return generator.Generate(
            edge_dist_vec,
            process_edges,
            nodes_per_process,
            num_processes,
            probability_matrix,
            node_edge_count,
            execution_time_ms
        );
    } catch (const std::exception& e) {
        std::cerr << "GPU edge generation failed: " << e.what() << "\n";
        return nullptr;
    }
}

}  // extern "C"