#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "skg.h"
#include <stdio.h>
#define BLOCK 1024

__global__ 
void stochastic_Kronecker_graph_kernel(edge *edge_list, long *block_edges, long dim, long *start_idx, float *prob, float *c_prob, int k, int n, int npes)
{
	int idx;
	idx  = blockIdx.x * blockDim.x + threadIdx.x;
	if(idx < n)
	{
		curandStateXORWOW_t rand_state;
		curand_init(idx, 0, 0, &rand_state);
		float p;
		long mat_dim, row, col;
		int p_idx, prow, pcol, i;
		mat_dim = dim;
		row=0;
		col=0;
		for(i=0;i<k;i++)
		{
			p = curand_uniform(&rand_state);
			p_idx = p<c_prob[0]?0:(p<c_prob[1]?1:(p<c_prob[2]?2:3));
			prow = p_idx/2;
			pcol = p_idx%2;
			mat_dim/=2;
			row = row + mat_dim*prow;
			col = col + mat_dim*pcol;
		}
		edge_list[idx].v = row;
		for(int i=0;i<npes;i++)
			if(block_edges[i] < idx)
			{
				edge_list[idx].u = start_idx[i]+col;
				break;
			}
		edge_list[idx].w = p;
	}

}

edge* create_edge_list_gpu(long *edges_dist, long pe_edges, long nodes_per_pe, int npes, block *mat_prob, int **node_edge_count, float *time)
{
	edge *d_edge_list, *edge_list = (edge*)malloc((size_t)pe_edges*sizeof(edge));
	cudaMalloc((void**)&d_edge_list, sizeof(edge) * pe_edges);
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int i, log_dim; 
	*node_edge_count = (int*)calloc(nodes_per_pe, sizeof(int));
	dim3 DimBlock(BLOCK);
	dim3 DimGrid((pe_edges-1)/(BLOCK) + 1);
	long *block_edges, *d_block_edges, *d_start_idx, *start_idx, idx = 0;
	block_edges = (long*)malloc(sizeof(long)*npes);
	start_idx = (long*)malloc(sizeof(long)*npes);
	log_dim = log2(nodes_per_pe);
	float prob[] = {mat_prob->a, mat_prob->b, mat_prob->c, mat_prob->d}, c_prob[4], *d_prob, *d_c_prob;
	c_prob[0] = prob[0];
	for(i=1;i<4;i++) c_prob[i] = prob[i] + c_prob[i-1];
	cudaMalloc((void**)&d_c_prob, sizeof(float)*4);
	cudaMemcpy(d_c_prob, c_prob, sizeof(float)*4, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&d_prob, sizeof(float)*4);
	cudaMemcpy(d_prob, prob, sizeof(float)*4, cudaMemcpyHostToDevice);
	for(i=0;i<npes;i++)
	{
		block_edges[i] = idx;
		start_idx[i] = nodes_per_pe*i;
		idx += edges_dist[i];
	}
	cudaMalloc((void**)&d_block_edges, sizeof(float)*npes);
	cudaMalloc((void**)&d_start_idx, sizeof(float)*npes);
	cudaMemcpy(d_block_edges, block_edges, sizeof(float)*npes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_start_idx, start_idx, sizeof(float)*npes, cudaMemcpyHostToDevice);
	cudaEventRecord(start);
	stochastic_Kronecker_graph_kernel<<<DimGrid,DimBlock>>>(d_edge_list, d_block_edges, nodes_per_pe, d_start_idx, prob, d_c_prob, log_dim, pe_edges, npes);
	cudaEventRecord(stop);
	cudaDeviceSynchronize();
	cudaEventElapsedTime(time, start, stop);
	cudaMemcpy(edge_list, d_edge_list, sizeof(edge) * pe_edges, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	for(i=0;i<pe_edges;i++) node_edge_count[0][edge_list[i].v]++;
	//memory deallocation
	cudaFree(d_edge_list);
	cudaFree(d_c_prob);
	cudaFree(d_prob);
	cudaFree(d_block_edges);
	cudaFree(d_start_idx);
	return edge_list;
}