#pragma once
#include "skg.h"
#include <stdio.h>
#include <mpi.h>

void write_graph_text(const char* filename, CSRData* csr_mat, int rank, int npes) {
    FILE* fp;
    
    if (rank == 0) {
        fp = fopen(filename, "w");
        if (!fp) {
            printf("Error: Cannot create file %s\n", filename);
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        fprintf(fp, "# Stochastic Kronecker Graph\n");
        fprintf(fp, "# Format: source_node target_node weight\n");
        fclose(fp);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    for (int p = 0; p < npes; p++) {
        if (rank == p) {
            fp = fopen(filename, "a");
            if (!fp) {
                printf("Rank %d: Cannot open file %s\n", rank, filename);
                MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
            }
            
            for (long i = 0; i < csr_mat->nodes; i++) {
                long global_node = rank * csr_mat->nodes + i;
                for (long j = csr_mat->row_ptr[i]; j < csr_mat->row_ptr[i+1]; j++) {
                    fprintf(fp, "%ld %ld %.6f\n", global_node, csr_mat->col_ptr[j], csr_mat->val_ptr[j]);
                }
            }
            
            fclose(fp);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    
    if (rank == 0) {
        printf("Graph saved to %s (text format)\n", filename);
    }
}

void write_graph_binary(const char* filename, CSRData* csr_mat, int rank, int npes) {
    MPI_File handle;
    MPI_Status status;
    int access_mode = MPI_MODE_CREATE | MPI_MODE_WRONLY;
    
    if (MPI_File_open(MPI_COMM_WORLD, filename, access_mode, MPI_INFO_NULL, &handle) != MPI_SUCCESS) {
        printf("[MPI process %d] Failure in opening file %s\n", rank, filename);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    
    if (rank == 0) {
        long header[3];
        long total_nodes = csr_mat->nodes * npes;
        long total_edges = 0;
        
        MPI_Reduce(&csr_mat->edges, &total_edges, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        
        header[0] = 0x4B524753; 
        header[1] = total_nodes;
        header[2] = total_edges;
        
        MPI_File_write(handle, header, 3, MPI_LONG, &status);
    } else {
        long edges = csr_mat->edges;
        MPI_Reduce(&edges, NULL, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    
    MPI_Offset offset = 3 * sizeof(long);  
    for (int p = 0; p < rank; p++) {
        // each process before us writes their edges
        // TODO: extend this to handle variable/dynamic nodes/edges per process
        offset += csr_mat->edges * (2 * sizeof(long) + sizeof(float));
    }
    
    MPI_File_seek(handle, offset, MPI_SEEK_SET);
    
    // edges
    for (long i = 0; i < csr_mat->nodes; i++) {
        long global_node = rank * csr_mat->nodes + i;
        for (long j = csr_mat->row_ptr[i]; j < csr_mat->row_ptr[i+1]; j++) {
            // source node
            MPI_File_write(handle, &global_node, 1, MPI_LONG, &status);
            // target node
            MPI_File_write(handle, &csr_mat->col_ptr[j], 1, MPI_LONG, &status);
            // weight
            MPI_File_write(handle, &csr_mat->val_ptr[j], 1, MPI_FLOAT, &status);
        }
    }
    
    MPI_File_close(&handle);
    
    if (rank == 0) {
        printf("Graph saved to %s (binary format)\n", filename);
    }
}

// graph statistics
void write_graph_stats(const char* filename, CSRData* csr_mat, int rank, int npes) {
    if (rank == 0) {
        FILE* fp = fopen(filename, "w");
        if (!fp) {
            printf("Error: Cannot create stats file %s\n", filename);
            return;
        }
        
        long total_nodes = csr_mat->nodes * npes;
        long total_edges = 0;
        long min_edges = csr_mat->edges;
        long max_edges = csr_mat->edges;
        
        MPI_Reduce(&csr_mat->edges, &total_edges, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&csr_mat->edges, &min_edges, 1, MPI_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&csr_mat->edges, &max_edges, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
        
        fprintf(fp, "Graph Statistics\n");
        fprintf(fp, "================\n");
        fprintf(fp, "Total Nodes: %ld\n", total_nodes);
        fprintf(fp, "Total Edges: %ld\n", total_edges);
        fprintf(fp, "Average Degree: %.2f\n", (double)total_edges / total_nodes);
        fprintf(fp, "Processes: %d\n", npes);
        fprintf(fp, "Nodes per Process: %ld\n", csr_mat->nodes);
        fprintf(fp, "Edges per Process: min=%ld, max=%ld, avg=%.2f\n", 
                min_edges, max_edges, (double)total_edges / npes);
        
        fclose(fp);
        printf("Statistics saved to %s\n", filename);
    } else {
        long edges = csr_mat->edges;
        MPI_Reduce(&edges, NULL, 1, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&edges, NULL, 1, MPI_LONG, MPI_MIN, 0, MPI_COMM_WORLD);
        MPI_Reduce(&edges, NULL, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    }
}