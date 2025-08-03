#include <iostream>
#include <iomanip>
#include <mpi.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include "skg.h"
#include "graph.h"

void PrintHelp(const std::string& option = "");
void SetParameters(int argc,
                  char** argv,
                  int& scalingFactor,
                  int& edgeFactor,
                  Block& matProb,
                  long& matSize);
void PrintParameters(const int& scalingFactor,
                    const long& matSize,
                    const Block& matProb,
                    const long& nodesPerPe,
                    const int& matBlocks,
                    const int edgeFactor,
                    const TimeStats& graphTime,
                    const TimeStats& csrTime,
                    const TimeStats& IOTime,
                    const long totalEdges);

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, npes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);

    if (!rank && (npes & (npes - 1)) != 0) {
        std::cerr << "Number of processes: (" << npes << ") is not a power of 2\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int scalingFactor = 15, edgeFactor = 20, matBlocks;
    long matSize, nodesPerPe, peEdges;
    Block matProb = {0.25, 0.25, 0.25, 0.25, 0};
    SetParameters(argc, argv, scalingFactor, edgeFactor, matProb, matSize);

    nodesPerPe = matSize / npes;
    matBlocks = matSize / nodesPerPe;
    
    long* edgesDist = calculate_edge_distribution(rank, npes, &matProb);
    peEdges = calculate_edges(edgesDist, npes);

    TimeStats graphTime, csrTime, IOTime;
    long totalEdges = PerformGraphOperations(edgesDist,
                                            peEdges,
                                            nodesPerPe,
                                            npes,
                                            matProb,
                                            rank,
                                            graphTime,
                                            csrTime,
                                            IOTime);

    if (!rank) {
        PrintParameters(scalingFactor,
                       matSize,
                       matProb,
                       nodesPerPe,
                       matBlocks,
                       edgeFactor,
                       graphTime,
                       csrTime,
                       IOTime,
                       totalEdges);
    }

    MPI_Finalize();
    return EXIT_SUCCESS;
}

void PrintHelp(const std::string& option) {
    if (option.empty() || option == "s") {
        std::cerr << "Invalid value for -s (requires a positive integer).\n";
    }
    if (option.empty() || option == "e") {
        std::cerr << "Invalid value for -e (requires a positive integer).\n";
    }
    if (option.empty() || option == "a" || option == "b" || option == "c") {
        std::cerr << "Invalid value for probability (requires a value between 0 and 1).\n";
    }
    
    if (option.empty()) {
        std::cerr << "Usage: program options\n"
                  << "  -s  scaling factor (default 15)\n"
                  << "  -e  edge factor (default 20)\n"
                  << "  -a  probability a (default 0.25)\n"
                  << "  -b  probability b (default 0.25)\n"
                  << "  -c  probability c (default 0.25)\n"
                  << "  -h  help\n";
    }
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

void SetParameters(int argc,
                  char** argv,
                  int& scalingFactor,
                  int& edgeFactor,
                  Block& matProb,
                  long& matSize) {
    int opt;
    while ((opt = getopt(argc, argv, "s:e:a:b:c:h")) != -1) {
        switch (opt) {
            case 's':
                scalingFactor = std::stoi(optarg);
                if (scalingFactor < 1) PrintHelp("s");
                break;
            case 'e':
                edgeFactor = std::stoi(optarg);
                if (edgeFactor < 1) PrintHelp("e");
                break;
            case 'a':
                matProb.a = std::stof(optarg);
                if (matProb.a < 0 || matProb.a > 1) PrintHelp("a");
                break;
            case 'b':
                matProb.b = std::stof(optarg);
                if (matProb.b < 0 || matProb.b > 1) PrintHelp("b");
                break;
            case 'c':
                matProb.c = std::stof(optarg);
                if (matProb.c < 0 || matProb.c > 1) PrintHelp("c");
                break;
            case 'h':
                PrintHelp();
                break;
            default:
                PrintHelp();
                break;
        }
    }
    
    matProb.d = 1.0 - (matProb.a + matProb.b + matProb.c);
    if (matProb.d < 0 || matProb.d > 1) {
        std::cerr << "Invalid probability matrix values (sum must equal 1).\n";
        PrintHelp();
    }
    
    matSize = 1L << scalingFactor;
    matProb.edges = matSize * edgeFactor;
}

void PrintParameters(const int& scalingFactor,
                    const long& matSize,
                    const Block& matProb,
                    const long& nodesPerPe,
                    const int& matBlocks,
                    const int edgeFactor,
                    const TimeStats& graphTime,
                    const TimeStats& csrTime,
                    const TimeStats& IOTime,
                    const long totalEdges) {
    std::cout << "\n=== Stochastic Kronecker Graph Generation ===\n\n";
    std::cout << "Parameters:\n";
    std::cout << "  Scaling factor: " << scalingFactor << "\n";
    std::cout << "  Matrix size: " << matSize << " x " << matSize << "\n";
    std::cout << "  Probability matrix: [a=" << std::fixed << std::setprecision(3) << matProb.a
              << ", b=" << matProb.b << ", c=" << matProb.c << ", d=" << matProb.d << "]\n";
    std::cout << "  Edge factor: " << edgeFactor << "\n";
    std::cout << "  Total edges: " << totalEdges << " (expected: " << matProb.edges << ")\n";
    std::cout << "  Nodes per process: " << nodesPerPe << "\n";
    std::cout << "  Matrix blocks: " << matBlocks << "\n\n";

    std::cout << "Performance:\n";
    #ifdef _GPU
    std::cout << "  GPU Graph generation: ";
    #else
    std::cout << "  CPU Graph generation: ";
    #endif
    std::cout << "avg=" << std::fixed << std::setprecision(5) << graphTime.avg_t
              << "s, min=" << graphTime.min_t << "s, max=" << graphTime.max_t << "s\n";

    std::cout << "  CSR conversion: ";
    std::cout << "avg=" << std::fixed << std::setprecision(5) << csrTime.avg_t
              << "s, min=" << csrTime.min_t << "s, max=" << csrTime.max_t << "s\n";

    std::cout << "  I/O operations: ";
    std::cout << "avg=" << std::fixed << std::setprecision(5) << IOTime.avg_t
              << "s, min=" << IOTime.min_t << "s, max=" << IOTime.max_t << "s\n";
    
    std::cout << "\n===========================================\n";
}