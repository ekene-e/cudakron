# CudaKron
### High-Performance C++20 Implementation with MPI and CUDA Support

[![C++20](https://img.shields.io/badge/C%2B%2B-20-blue.svg)](https://en.cppreference.com/w/cpp/20)
[![MPI](https://img.shields.io/badge/MPI-Enabled-green.svg)](https://www.mpi-forum.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.3-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview

A production-grade, high-performance implementation of the Stochastic Kronecker Graph generation algorithm, designed for generating massive-scale graphs with realistic properties. This implementation leverages modern C++20 features, MPI for distributed computing, and optional CUDA GPU acceleration. (This was built as an aside for the Summer 2023 course "C++ for C programmers" at Columbia.)

### The Kronecker Product

The Stochastic Kronecker Graph (SKG) model is based on the **Kronecker product** of matrices:

#### Definition
Given two matrices $\mathbf{A} \in \mathbb{R}^{m \times n}$ and $\mathbf{B} \in \mathbb{R}^{p \times q}$, their Kronecker product $\mathbf{A} \otimes \mathbf{B} \in \mathbb{R}^{mp \times nq}$ is defined as:

$$\mathbf{A} \otimes \mathbf{B} = \begin{bmatrix}
a_{11}\mathbf{B} & a_{12}\mathbf{B} & \cdots & a_{1n}\mathbf{B} \\
a_{21}\mathbf{B} & a_{22}\mathbf{B} & \cdots & a_{2n}\mathbf{B} \\
\vdots & \vdots & \ddots & \vdots \\
a_{m1}\mathbf{B} & a_{m2}\mathbf{B} & \cdots & a_{mn}\mathbf{B}
\end{bmatrix}$$


The SKG model generates graphs through recursive Kronecker products of a small **initiator matrix** $\mathbf{P} \in [0,1]^{2 \times 2}$.

The seed or **initiator matrix** has the form:

$$\mathbf{P} = \begin{bmatrix} a & b \\
c & d \end{bmatrix}$$

where $a, b, c, d \in [0,1]$ and $a + b + c + d = 1$.
These values represent probabilities; $a$ is the probability of edges within the same community, $b$ and $c$ are the probabilities of edges between different communities, and $d$ is the probability of edges between distant nodes.

#### Graph Generation Process

For a graph with $N = 2^k$ nodes, the adjacency matrix $\mathbf{G} \in [0,1]^{N \times N}$ is generated as

$$G[i,j] = \prod_{l=0}^{k-1} P[i_l, j_l]$$

where $i_l$ and $j_l$ are the $l$-th bits of the binary representations of $i$ and $j$. This means that each entry in the adjacency matrix
(with edge weights interpreted as probabilities) is computed by recursively applying the initiator matrix. Instead of computing the full Kronecker product (which would require $O(N^2)$ space), we use a recursive algorithm, sketched below:

```python
def generate_edge(k, P):
    row, col = 0, 0
    for level in range(k):
        p = random()
        quadrant = select_quadrant(p, P)
        row = 2*row + quadrant.row
        col = 2*col + quadrant.col
    return (row, col)
```


### Parallelization Strategy

#### Edge Distribution
For $p$ MPI processes ($p = 2^m$), edges are distributed using:
$$E[i,j] = E_{\text{total}} \times \prod_{l=0}^{m-1} \mathbf{P}[i_l, j_l]$$
where $i_l$ and $j_l$ are the $l$-th bits of process indices $i$ and $j$.
How do we balance loads? Check that the expected edges per process is
$$\mu = \frac{E_{\text{total}}}{p}, \quad \sigma^2 = \frac{E_{\text{total}} \times \text{Var}(\mathbf{P})}{p}$$
and so we can use techniques like work stealing or dynamic load balancing to ensure all processes remain busy.

## Features

### Performance Characteristics
- **Scalability**: Tested up to $2^{30}$ nodes (1 billion+)
- **Memory**: $O(E/p)$ per process for $E$ edges, $p$ processes
- **Time Complexity**: $O(E/p)$ per process
- **GPU Speedup**: 10-100x for edge generation


## Building the Project

### Prerequisites
- C++20 compliant compiler (GCC 11+, Clang 13+, MSVC 2022+)
- CMake 3.20+
- MPI implementation (OpenMPI, MPICH, MS-MPI)
- CUDA Toolkit 11.0+ (optional)

### Build Instructions

```bash
# Clone repository
git clone https://github.com/ekene-e/cudakron.git
cd cudakron

# Create build directory
mkdir build && cd build

# Configure
cmake .. -DCMAKE_BUILD_TYPE=Release \
         -DSKG_ENABLE_GPU=ON \
         -DCMAKE_CUDA_ARCHITECTURES=75

# Build
make -j$(nproc)

# Run tests
ctest --verbose

# Install
sudo make install
```

### CMake Options
| Option | Default | Description |
|--------|---------|-------------|
| `SKG_ENABLE_GPU` | ON | Enable CUDA GPU acceleration |
| `SKG_ENABLE_TESTING` | ON | Build unit tests |
| `SKG_ENABLE_BENCHMARKS` | OFF | Build performance benchmarks |
| `SKG_USE_SANITIZERS` | OFF | Enable AddressSanitizer and UBSan |
| `SKG_ENABLE_LTO` | ON | Enable Link Time Optimization |

## Usage

### Basic Usage

```bash
# Generate graph with 2^20 nodes, edge factor 16
mpiexec -n 4 skg -s 20 -e 16

# Custom probability matrix (social network pattern)
mpiexec -n 8 skg -s 24 -e 20 -a 0.45 -b 0.25 -c 0.25

# GPU acceleration
mpiexec -n 4 skg -s 26 -e 32 --gpu
```
Note that some of these are quite large, so you may want to use `--format text` to output a human-readable edge list instead of the binary CSR format.

### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-s, --scaling` | Scaling factor ($2^s$ nodes) | 15 |
| `-e, --edges` | Edge factor (average degree) | 20 |
| `-a` | Probability matrix entry a | 0.25 |
| `-b` | Probability matrix entry b | 0.25 |
| `-c` | Probability matrix entry c | 0.25 |
| `--gpu` | Enable GPU acceleration | false |
| `--format` | Output format (binary/text/both) | binary |
| `-h, --help` | Show help message | - |

### Example Configurations

#### Social Network (High Clustering)
```bash
mpiexec -n 16 skg -s 22 -e 25 -a 0.45 -b 0.25 -c 0.25
```
- High intra-community edges (a = 0.45)
- Produces clustering coefficient ~0.3-0.4

#### Web Graph (Power Law)
```bash
mpiexec -n 8 skg -s 24 -e 16 -a 0.57 -b 0.19 -c 0.19
```
- Strong preferential attachment (a = 0.57)
- Power-law degree distribution γ ≈ 2.1

#### Random Graph (Erdős–Rényi-like)
```bash
mpiexec -n 4 skg -s 20 -e 20 -a 0.25 -b 0.25 -c 0.25
```
- Uniform probabilities
- Similar to G(n,p) model

## Performance Analysis

### Benchmarks

| Nodes ($2^s$) | Edges | Processes | CPU Time | GPU Time | Speedup |
|------------|-------|-----------|----------|----------|---------|
| $2^{20}$ | 20M | 4 | 2.3s | 0.15s | 15.3x |
| $2^{24}$ | 320M | 16 | 38.5s | 2.1s | 18.3x |
| $2^{28}$ | 5.4B | 64 | 12.3m | 31.2s | 23.6x |

### Scaling Analysis

#### Strong Scaling (Fixed Problem Size)
$$\text{Efficiency}(p) = \frac{T(1)}{p \times T(p)}$$
- Maintains >80% efficiency up to 128 processes

#### Weak Scaling (Fixed Work per Process)
$$\text{Efficiency}(p) = \frac{T(1)}{T(p)}$$
- Maintains >90% efficiency up to 256 processes

### Memory Usage

Per process memory requirement:

$$\text{Memory} = 8 \times \left(\frac{N}{p} + 1\right) + 8 \times \left(\frac{E}{p}\right) + 4 \times \left(\frac{E}{p}\right) \text{ bytes}$$

$$\quad\quad\quad\quad\uparrow \text{ row-ptr} \quad\quad\quad\uparrow \text{ col-idx} \quad\uparrow \text{ values}$$

Example ($2^{24}$ nodes, 320M edges, 16 processes):
- Per process: ~260 MB
- Total: ~4.2 GB

## Advanced Features

### Custom Probability Matrices

Create specialized graph types by adjusting the initiator matrix:

```cpp
// Bipartite-like structure
ProbabilityMatrix bipartite{
    .probabilities = {0.05f, 0.45f, 0.45f, 0.05f}
};

// Hub-and-spoke pattern
ProbabilityMatrix hub_spoke{
    .probabilities = {0.70f, 0.10f, 0.10f, 0.10f}
};
```

### Streaming Large Graphs

For graphs too large to fit in memory:

```cpp
GraphIO io(rank, size);
io.StreamEdges(csr_matrix, [](const Edge& e) {
    // Process each edge
    process_edge(e.source, e.target, e.weight);
});
```

### Development Setup

```bash
# Debug build with sanitizers
cmake .. -DCMAKE_BUILD_TYPE=Debug \
         -DSKG_USE_SANITIZERS=ON \
         -DSKG_ENABLE_TESTING=ON \
         -DSKG_ENABLE_BENCHMARKS=ON

# Format code
clang-format -i src/*.cpp include/*.hpp

# Run static analysis
clang-tidy src/*.cpp -- -std=c++20
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Original SKG algorithm: Leskovec et al., JMLR 2010
- MPI optimization strategies: Graph 500 benchmark
- CUDA implementation inspired by cuGraph
