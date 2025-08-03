#pragma once
#include "skg.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <mpi.h>
#include "graph_io.h"

long** block_allocation(int n, int m) {
    size_t size = (size_t)n * m;
    long **pe_blocks = (long**)malloc(sizeof(long*) * n);
    pe_blocks[0] = (long*)malloc(sizeof(long) * size);
    for(int i = 1; i < n; i++)
        pe_blocks[i] = pe_blocks[0] + (size_t)i * m;
    return pe_blocks;
}

long* calculate_edge_distribution(int rank, int npes, Block *mat_prop) {
    long **pe_blocks = block_allocation(npes, npes);
    int i, j, k, l, m, x, y;
    int n = log2(npes);
    for(i = 0; i < npes; i++)
        for(j = 0; j < npes; j++)
            pe_blocks[i][j] = mat_prop->edges;
    
    int num_blocks, grid_size, block_row, block_col, grid_row, grid_col;
    float prob;
    
    for(i = 0; i < n; i++) {
        num_blocks = 1 << i;
        grid_size = 1 << (n - 1 - i);
        for(j = 0; j < num_blocks; j++) {
            block_row = j * grid_size * 2;
            for(k = 0; k < num_blocks; k++) {
                block_col = k * grid_size * 2;
                for(l = 0; l < 2; l++) {
                    grid_row = block_row + grid_size * l;
                    for(m = 0; m < 2; m++) {
                        grid_col = block_col + grid_size * m;
                        prob = l ? (m ? mat_prop->d : mat_prop->c) : (m ? mat_prop->b : mat_prop->a);
                        for(x = 0; x < grid_size; x++)
                            for(y = 0; y < grid_size; y++)
                                pe_blocks[grid_row + x][grid_col + y] = 
                                    (long)round(pe_blocks[grid_row + x][grid_col + y] * prob);
                    }
                }
            }
        }
    }
    
    long *edges_dist = (long*)malloc(sizeof(long) * npes);
    memcpy(edges_dist, pe_blocks[rank], (size_t)npes * sizeof(long));
    free(pe_blocks[0]);
    free(pe_blocks);
    return edges_dist;
}

long calculate_edges(long *edges_dist, int n) {
    int i;
    long edges = 0;
    for(i = 0; i < n; i++)
        edges += edges_dist[i];
    return edges;
}

void stochastic_kronecker_graph(Edge *edge_list,
                               long pe_edges,
                               long dim,
                               long start_idx,
                               float *prob,
                               float *c_prob,
                               int *node_edge_count) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    srand(start_idx + rank);
    
    long mat_dim, row, col, edge_count = 0;
    int p_idx, prow, pcol, i, k;
    float p;
    k = log2(dim);
    
    for(edge_count = 0; edge_count < pe_edges; edge_count++) {
        mat_dim = dim;
        row = 0;
        col = 0;
        for(i = 0; i < k; i++) {
            p = (float)rand() / RAND_MAX;
            p_idx = p < c_prob[0] ? 0 : (p < c_prob[1] ? 1 : (p < c_prob[2] ? 2 : 3));
            prow = p_idx / 2;
            pcol = p_idx % 2;
            mat_dim /= 2;
            row = row + mat_dim * prow;
            col = col + mat_dim * pcol;
        }
        edge_list[edge_count].v = row;
        edge_list[edge_count].u = start_idx + col;
        edge_list[edge_count].w = p;
        node_edge_count[row]++;
    }
}

Edge* create_edge_list(long *edges_dist,
                      long pe_edges,
                      long nodes_per_pe,
                      int npes,
                      Block *mat_prob,
                      int **node_edge_count) {
    Edge *edge_list = (Edge*)malloc((size_t)pe_edges * sizeof(Edge));
    int i;
    *node_edge_count = (int*)calloc(nodes_per_pe, sizeof(int));
    long block_edges, start_idx, idx = 0;
    float prob[] = {mat_prob->a, mat_prob->b, mat_prob->c, mat_prob->d}, c_prob[4];
    c_prob[0] = prob[0];
    for(i = 1; i < 4; i++) 
        c_prob[i] = prob[i] + c_prob[i-1];
    
    for(i = 0; i < npes; i++) {
        block_edges = edges_dist[npes - 1 - i];
        start_idx = nodes_per_pe * (npes - 1 - i);
        stochastic_kronecker_graph(&edge_list[idx],
                                  block_edges,
                                  nodes_per_pe,
                                  start_idx,
                                  prob,
                                  c_prob,
                                  *node_edge_count);
        idx += block_edges;
    }
    return edge_list;
}

CSRData* create_csr_data(Edge *edge_list,
                        int *node_edge_count,
                        long pe_edges,
                        long nodes_per_pe,
                        long* edge_count) {
    size_t n = sizeof(CSRData)
             + (nodes_per_pe + 1) * sizeof(long)
             + pe_edges * (sizeof(long) + sizeof(float));
    
    CSRData *csr_mat = (CSRData*)malloc(n);
    csr_mat->edges = pe_edges;
    csr_mat->nodes = nodes_per_pe;
    
    long i, idx;
    if(csr_mat == NULL) {
        printf("csr memory allocation failed (%ld B)\n", n);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    n = sizeof(CSRData);
    csr_mat->row_ptr = (long*)((char*)csr_mat + n);
    n += ((nodes_per_pe + 1) * sizeof(long));
    csr_mat->col_ptr = (long*)((char*)csr_mat + n);
    n += (pe_edges * sizeof(long));
    csr_mat->val_ptr = (float*)((char*)csr_mat + n);
    
    // edge_count
    long count = 0;
    for(i = 0; i < nodes_per_pe; i++) 
        count += node_edge_count[i];
    *edge_count = count;
    
    // row_ptr
    csr_mat->row_ptr[0] = 0;
    for(i = 1; i <= nodes_per_pe; i++)
        csr_mat->row_ptr[i] = csr_mat->row_ptr[i-1] + node_edge_count[i-1];
    
    for(i = 0; i < pe_edges; i++) {
        idx = edge_list[i].v;
        node_edge_count[idx]--;
        idx = csr_mat->row_ptr[idx] + node_edge_count[idx];
        csr_mat->col_ptr[idx] = edge_list[i].u;
        csr_mat->val_ptr[idx] = edge_list[i].w;
    }
    
    for(i = 0; i < nodes_per_pe; i++)
        std::sort(&csr_mat->col_ptr[csr_mat->row_ptr[i]],
                 &csr_mat->col_ptr[csr_mat->row_ptr[i+1]]);
    
    free(node_edge_count);
    free(edge_list);
    return csr_mat;
}

void print_edge_list(Edge *edge_list,
                    long pe_edges,
                    int *node_edge_count,
                    long nodes_per_pe,
                    long *edges_dist,
                    int npes) {
    printf("edges: %ld, nodes_per_pe: %ld\n", pe_edges, nodes_per_pe);
    for(int i = 0; i < pe_edges; i++) 
        printf("%d (%ld -> %ld)\n", i, edge_list[i].v, edge_list[i].u);
    printf("---------\n");
    for(int i = 0; i < nodes_per_pe; i++) 
        printf("%d -> %d\n", i, node_edge_count[i]);
    printf("---------\n");
    for(int i = 0; i < npes; i++) 
        printf("%d -> %ld\n", i, edges_dist[i]);
}

void print_csr(CSRData* csr_mat) {
    printf("nodes: %ld, edges: %ld\n", csr_mat->nodes, csr_mat->edges);
    for(int i = 0; i < csr_mat->nodes; i++) {
        printf("%d(%ld) -> ", i, csr_mat->row_ptr[i+1] - csr_mat->row_ptr[i]);
        for(int j = csr_mat->row_ptr[i]; j < csr_mat->row_ptr[i+1]; j++)
            printf("(%ld, %0.2f) ", csr_mat->col_ptr[j], csr_mat->val_ptr[j]);
        printf("\n");
    }
}

long PerformGraphOperations(long *edges_dist,
                           long pe_edges,
                           long nodes_per_pe,
                           int npes,
                           Block& mat_prob,
                           int rank,
                           TimeStats& graph_time,
                           TimeStats& csr_time,
                           TimeStats& io_time) {
    double t_start, t_end;
    Edge *edge_list = nullptr;
    int *node_edge_count = nullptr;
    CSRData *csr_mat = nullptr;
    long edge_count;
    
    // graph generation
    t_start = MPI_Wtime();
    #ifdef _GPU
        float gpu_time;
        float prob_array[4] = {mat_prob.a, mat_prob.b, mat_prob.c, mat_prob.d};
        edge_list = create_edge_list_gpu(edges_dist, pe_edges, nodes_per_pe, 
                                        npes, prob_array, &node_edge_count, &gpu_time);
        if (edge_list == nullptr) {
            if (rank == 0) {
                printf("GPU acceleration not available, using CPU implementation...\n");
            }
            edge_list = create_edge_list(edges_dist, pe_edges, nodes_per_pe, 
                                        npes, &mat_prob, &node_edge_count);
            t_end = MPI_Wtime();
            graph_time.t = t_end - t_start;
        } else {
            graph_time.t = gpu_time / 1000.0; 
        }
    #else
        edge_list = create_edge_list(edges_dist, pe_edges, nodes_per_pe, 
                                    npes, &mat_prob, &node_edge_count);
        t_end = MPI_Wtime();
        graph_time.t = t_end - t_start;
    #endif
    
    t_start = MPI_Wtime();
    csr_mat = create_csr_data(edge_list, node_edge_count, pe_edges, nodes_per_pe, &edge_count);
    t_end = MPI_Wtime();
    csr_time.t = t_end - t_start;
    
    t_start = MPI_Wtime();
    
    write_graph_text("graph.txt", csr_mat, rank, npes);
    write_graph_binary("graph.dat", csr_mat, rank, npes);
    write_graph_stats("graph_stats.txt", csr_mat, rank, npes);
    
    t_end = MPI_Wtime();
    io_time.t = t_end - t_start;
    
    // timing statistics
    MPI_Allreduce(&graph_time.t, &graph_time.avg_t, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&graph_time.t, &graph_time.min_t, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&graph_time.t, &graph_time.max_t, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    graph_time.avg_t /= npes;
    
    MPI_Allreduce(&csr_time.t, &csr_time.avg_t, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&csr_time.t, &csr_time.min_t, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&csr_time.t, &csr_time.max_t, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    csr_time.avg_t /= npes;
    
    MPI_Allreduce(&io_time.t, &io_time.avg_t, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&io_time.t, &io_time.min_t, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&io_time.t, &io_time.max_t, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
    io_time.avg_t /= npes;
    
    long total_edges;
    MPI_Allreduce(&edge_count, &total_edges, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD);
    
    free(csr_mat);
    free(edges_dist);
    
    return total_edges;
}