#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <math.h> // For log2

#define BLOCK_SIZE 1024

// Define a struct for edges for better readability
struct Edge {
    long v, u;
    float w;
};

// Error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Kernel to generate stochastic Kronecker graph
__global__ void stochastic_kronecker_graph_kernel(Edge *edge_list,
												long *block_edges, 
												long dim,
												long *start_idx,
												float *prob,
												float *c_prob,
												int k,
												int n,
												int npes) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curandStateXORWOW_t rand_state;
        curand_init(idx, 0, 0, &rand_state);
        float p;
        long mat_dim = dim, row = 0, col = 0;
        int p_idx, prow, pcol;

        for (int i = 0; i < k; i++) {
            p = curand_uniform(&rand_state);
            p_idx = (p < c_prob[0]) ? 0 : (p < c_prob[1]) ? 1 : (p < c_prob[2]) ? 2 : 3;
            prow = p_idx / 2;
            pcol = p_idx % 2;
            mat_dim /= 2;
            row += mat_dim * prow;
            col += mat_dim * pcol;
        }
        edge_list[idx].v = row;
        for (int i = 0; i < npes; i++) {
            if (block_edges[i] < idx) {
                edge_list[idx].u = start_idx[i] + col;
                break;
            }
        }
        edge_list[idx].w = p;
    }
}

// Host function to create edge list on GPU
Edge* create_edge_list_gpu(long *edges_dist,
							long pe_edges,
							long nodes_per_pe,
							int npes,
							float mat_prob[4],
							int **node_edge_count,
							float *time) {
    Edge *d_edge_list, *edge_list = (Edge*)malloc(pe_edges * sizeof(Edge));
    cudaCheckError(cudaMalloc((void**)&d_edge_list, pe_edges * sizeof(Edge)));

    cudaEvent_t start, stop;
    cudaCheckError(cudaEventCreate(&start));
    cudaCheckError(cudaEventCreate(&stop));

    *node_edge_count = (int*)calloc(nodes_per_pe, sizeof(int));
    dim3 dimBlock(BLOCK_SIZE);
    dim3 dimGrid((pe_edges - 1) / BLOCK_SIZE + 1);

    long *d_block_edges, *d_start_idx;
    float *d_prob, *d_c_prob;

    // Preprocessing for kernel execution
    float c_prob[4] = {mat_prob[0]};
    for (int i = 1; i < 4; i++) c_prob[i] = mat_prob[i] + c_prob[i - 1];

    // Allocate and copy to device
    cudaCheckError(cudaMalloc((void**)&d_c_prob, 4 * sizeof(float)));
    cudaCheckError(cudaMemcpy(d_c_prob, c_prob, 4 * sizeof(float), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMalloc((void**)&d_prob, 4 * sizeof(float)));
    cudaCheckError(cudaMemcpy(d_prob, mat_prob, 4 * sizeof(float), cudaMemcpyHostToDevice));

    long *block_edges = (long*)malloc(npes * sizeof(long));
    long *start_idx = (long*)malloc(npes * sizeof(long));
    long idx = 0;
    for (int i = 0; i < npes; i++) {
        block_edges[i] = idx;
        start_idx[i] = nodes_per_pe * i;
        idx += edges_dist[i];
    }

    cudaCheckError(cudaMalloc((void**)&d_block_edges, npes * sizeof(long)));
    cudaCheckError(cudaMalloc((void**)&d_start_idx, npes * sizeof(long)));
    cudaCheckError(cudaMemcpy(d_block_edges, block_edges, npes * sizeof(long), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_start_idx, start_idx, npes * sizeof(long), cudaMemcpyHostToDevice));

    // Execute kernel and measure time
    cudaCheckError(cudaEventRecord(start));
    stochastic_kronecker_graph_kernel<<<dimGrid, dimBlock>>>(d_edge_list,
															d_block_edges,
															nodes_per_pe, 
															d_start_idx, 
															d_prob,
															d_c_prob,
															log2(nodes_per_pe),
															pe_edges, npes);
    cudaCheckError(cudaEventRecord(stop));
    cudaCheckError(cudaDeviceSynchronize());
    cudaCheckError(cudaEventElapsedTime(time, start, stop));

    // Copy results back to host
    cudaCheckError(cudaMemcpy(edge_list,
					d_edge_list,
					pe_edges * sizeof(Edge),
					cudaMemcpyDeviceToHost));

    // Update node edge counts
    for (int i = 0; i < pe_edges; i++) (*node_edge_count)[edge_list[i].v]++;

    // Cleanup
    cudaFree(d_edge_list);
    cudaFree(d_c_prob);
    cudaFree(d_prob);
    cudaFree(d_block_edges);
    cudaFree(d_start_idx);
    free(block_edges);
    free(start_idx);

    return edge_list;
}
