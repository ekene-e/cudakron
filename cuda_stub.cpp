// CUDA stub implementation for when NVCC cannot be used (no MSVC on Windows)
#include "skg.h"
#include <iostream>
#include <cstdlib>

Edge* create_edge_list_gpu(long* edges_dist,
                          long pe_edges,
                          long nodes_per_pe,
                          int npes,
                          float mat_prob[4],
                          int** node_edge_count,
                          float* time) {
    std::cerr << "WARNING: GPU version was compiled without CUDA support!" << std::endl;
    std::cerr << "Install Visual Studio Build Tools to enable GPU acceleration." << std::endl;
    std::cerr << "Falling back to CPU implementation would require additional code." << std::endl;
    
    // minimal dummy implementation
    *time = 0.0f;
    *node_edge_count = (int*)calloc(nodes_per_pe, sizeof(int));
    Edge* edge_list = (Edge*)malloc(pe_edges * sizeof(Edge));
    
    // initialize with dummy data (achtung! this won't produce a valid graph)
    for(long i = 0; i < pe_edges; i++) {
        edge_list[i].v = 0;
        edge_list[i].u = 0;
        edge_list[i].w = 0.0f;
    }
    
    std::cerr << "ERROR: This is a stub implementation. Results will be incorrect!" << std::endl;
    std::cerr << "Please use the CPU version (skg.exe) instead." << std::endl;
    
    return edge_list;
}