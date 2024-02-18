#pragma once
#include<math.h>
#include<algorithm>
typedef struct block_probablity
{
	float a, b, c, d;
	long edges;
}block;
typedef struct EDGE
{
	//edge from v to u with weight w
	long v;
	long u;
	float w;
}edge;
#ifdef _GPU
edge* create_edge_list_gpu(long*, long , long , int , block* , int**, float *);
#endif
typedef struct CSR_MATRIX
{
	long *row_ptr;
	long *col_ptr;
	float *val_ptr;
	long edges, nodes;
}csr_data;
typedef struct TIME
{
	double t, avg_t, min_t, max_t;
}time_stats;