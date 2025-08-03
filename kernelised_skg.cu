#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <stdio.h>
#include <math.h>
#include "skg.h"

#define BLOCK_SIZE 256

// error checking macro
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

__global__ void stochastic_kronecker_graph_kernel(Edge *edge_list,
                                                 long *cumulative_edges,
                                                 long nodes_per_pe,
                                                 long *start_idx,
                                                 float *c_prob,
                                                 int k,
                                                 long total_edges,
                                                 int npes,
                                                 unsigned int seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_edges) return;
    
    // initialize random state for this thread
    curandStateXORWOW_t rand_state;
    curand_init(seed + tid, 0, 0, &rand_state);
    
    // determine which PE block this edge belongs to
    int pe_block = 0;
    for (int i = 0; i < npes; i++) {
        if (tid < cumulative_edges[i]) {
            pe_block = npes - 1 - i;
            break;
        }
    }
    
    // generate the edge using Kronecker recursion
    long mat_dim = nodes_per_pe;
    long row = 0, col = 0;
    float p;
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
    
    edge_list[tid].v = row;
    edge_list[tid].u = start_idx[pe_block] + col;
    edge_list[tid].w = curand_uniform(&rand_state);
}

// host function to create edge list on GPU
extern "C" Edge* create_edge_list_gpu(long *edges_dist,
                                     long pe_edges,
                                     long nodes_per_pe,
                                     int npes,
                                     float mat_prob[4],
                                     int **node_edge_count,
                                     float *time) {
    // allocate host memory
    Edge *edge_list = (Edge*)malloc(pe_edges * sizeof(Edge));
    *node_edge_count = (int*)calloc(nodes_per_pe, sizeof(int));
    
    // prepare cumulative probabilities
    float c_prob[4];
    c_prob[0] = mat_prob[0];
    for (int i = 1; i < 4; i++) {
        c_prob[i] = mat_prob[i] + c_prob[i - 1];
    }
    
    // prepare cumulative edges and start indices
    long *cumulative_edges = (long*)malloc(npes * sizeof(long));
    long *start_idx = (long*)malloc(npes * sizeof(long));
    cumulative_edges[0] = edges_dist[npes - 1];
    start_idx[0] = nodes_per_pe * (npes - 1);
    
    for (int i = 1; i < npes; i++) {
        cumulative_edges[i] = cumulative_edges[i - 1] + edges_dist[npes - 1 - i];
        start_idx[i] = nodes_per_pe * (npes - 1 - i);
    }
    
    // allocate device memory
    Edge *d_edge_list;
    long *d_cumulative_edges, *d_start_idx;
    float *d_c_prob;
    
    cudaCheckError(cudaMalloc((void**)&d_edge_list, pe_edges * sizeof(Edge)));
    cudaCheckError(cudaMalloc((void**)&d_cumulative_edges, npes * sizeof(long)));
    cudaCheckError(cudaMalloc((void**)&d_start_idx, npes * sizeof(long)));
    cudaCheckError(cudaMalloc((void**)&d_c_prob, 4 * sizeof(float)));
    
    cudaCheckError(cudaMemcpy(d_cumulative_edges, cumulative_edges, npes * sizeof(long), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_start_idx, start_idx, npes * sizeof(long), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_c_prob, c_prob, 4 * sizeof(float), cudaMemcpyHostToDevice));
    
    cudaEvent_t start_event, stop_event;
    cudaCheckError(cudaEventCreate(&start_event));
    cudaCheckError(cudaEventCreate(&stop_event));
    
    int num_blocks = (pe_edges + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    cudaCheckError(cudaEventRecord(start_event));
    
    unsigned int seed = (unsigned int)clock();
    stochastic_kronecker_graph_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_edge_list,
        d_cumulative_edges,
        nodes_per_pe,
        d_start_idx,
        d_c_prob,
        (int)log2(nodes_per_pe),
        pe_edges,
        npes,
        seed
    );
    
    cudaCheckError(cudaEventRecord(stop_event));
    cudaCheckError(cudaEventSynchronize(stop_event));
    
    cudaCheckError(cudaEventElapsedTime(time, start_event, stop_event));
    
    cudaCheckError(cudaMemcpy(edge_list, d_edge_list, pe_edges * sizeof(Edge), cudaMemcpyDeviceToHost));
    
    for (long i = 0; i < pe_edges; i++) {
        (*node_edge_count)[edge_list[i].v]++;
    }
    
    cudaCheckError(cudaEventDestroy(start_event));
    cudaCheckError(cudaEventDestroy(stop_event));
    cudaCheckError(cudaFree(d_edge_list));
    cudaCheckError(cudaFree(d_cumulative_edges));
    cudaCheckError(cudaFree(d_start_idx));
    cudaCheckError(cudaFree(d_c_prob));
    
    free(cumulative_edges);
    free(start_idx);
    
    return edge_list;
}