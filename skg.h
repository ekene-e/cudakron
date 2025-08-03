#pragma once
#include <math.h>
#include <algorithm>

typedef struct block_probability {
    float a, b, c, d;
    long edges;
} Block;

typedef struct edge_struct {
    // edge from v to u with weight w
    long v;
    long u;
    float w;
} Edge;

typedef struct csr_matrix {
    long *row_ptr;
    long *col_ptr;
    float *val_ptr;
    long edges, nodes;
} CSRData;

typedef struct time_stats {
    double t, avg_t, min_t, max_t;
} TimeStats;

#ifdef _GPU
Edge* create_edge_list_gpu(long* edges_dist,
                          long pe_edges,
                          long nodes_per_pe,
                          int npes,
                          float mat_prob[4],
                          int** node_edge_count,
                          float* time);
#endif