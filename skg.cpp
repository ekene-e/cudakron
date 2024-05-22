#include <iostream>
#include <mpi.h>
#include <unistd.h>
#include <cstdlib>
#include <cstring>
#include "skg.h"
#include "graph.h"

// Function prototypes
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

    // Ensure the number of processes is a power of 2
    if (!rank && (npes & (npes - 1)) != 0) {
        std::cerr << "Number of processes: (" << npes << ") is not a power of 2\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

    int scalingFactor = 15, edgeFactor = 20, matBlocks;
    long matSize, nodesPerPe, peEdges;
    Block matProb = {0.25, 0.25, 0.25, 0.25};
    SetParameters(argc, argv, scalingFactor, edgeFactor, matProb, matSize);

    nodesPerPe = matSize / npes;
    matBlocks = matSize / nodesPerPe;
    auto edgesDist = CalculateEdgeDistribution(rank, npes, matProb);
    peEdges = CalculateEdges(edgesDist, npes);

    int* nodeEdgeCount = nullptr;
    Edge* edgeList = nullptr;
    TimeStats graphTime, csrTime, IOTime;
    long totalEdges = PerformGraphOperations(edgeList,
                            nodeEdgeCount,
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
    // Add more error messages as necessary...
    
    if (option.empty()) {
        std::cerr << "Usage: program options\n"
                  << "-s scaling factor (default 15)\n"
                  << "-e edge factor (default 20)\n"
                  << "-a probability a (default 0.25)\n"
                  << "-b probability b (default 0.25)\n"
                  << "-c probability c (default 0.25)\n"
                  << "-h help\n";
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
            case 'a': // Similar checks for 'a', 'b', 'c'
                break;
            case 'h':
                PrintHelp();
                break;
            default:
                PrintHelp(std::string(1, opt));
                break;
        }
    }
    matProb.d = 1 - (matProb.a + matProb.b + matProb.c);
    if (matProb.d < 0 || matProb.d > 1) {
        std::cerr << "Invalid probability matrix values.\n";
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
    std::cout << "Scaling factor: " << scalingFactor << "\n";
    std::cout << "Matrix size: " << matSize << "\n";
    std::cout << "Probability matrix (a: " << std::fixed << std::setprecision(3) << matProb.a
              << ", b: " << matProb.b << ", c: " << matProb.c << ", d: " << matProb.d << ")\n";
    std::cout << "Total edges: " << totalEdges << " (" << matProb.edges << " expected, edge factor: " << edgeFactor << ")\n";
    std::cout << "Nodes per process (PE): " << nodesPerPe << "\n";
    std::cout << "Matrix blocks: " << matBlocks << "\n";

	std::cout << "Graph generation time: avg: " << std::fixed << std::setprecision(5) << graphTime.avg_t
			  << " s, min: " << graphTime.min_t << " s, max: " << graphTime.max_t << " s\n";

	std::cout << "CSR conversion time: avg: " << std::fixed << std::setprecision(5) << csrTime.avg_t
			  << " s, min: " << csrTime.min_t << " s, max: " << csrTime.max_t << " s\n";

	#ifdef _GPU
		std::cout << "GPU Graph generation time: avg: " << std::fixed << std::setprecision(5) << graphTime.avg_t
				<< " s, min: " << graphTime.min_t << " s, max: " << graphTime.max_t << " s\n";
	#else
		std::cout << "CPU Graph generation time: avg: " << std::fixed << std::setprecision(5) << graphTime.avg_t
				<< " s, min: " << graphTime.min_t << " s, max: " << graphTime.max_t << " s\n";
	#endif

    std::cout << "CSR conversion time: avg: " << std::fixed << std::setprecision(5) << csrTime.avg_t
              << " s, min: " << csrTime.min_t << " s, max: " << csrTime.max_t << " s\n";
    std::cout << "I/O time: avg: " << std::fixed << std::setprecision(5) << IOTime.avg_t
              << " s, min: " << IOTime.min_t << " s, max: " << IOTime.max_t << " s\n";

}
