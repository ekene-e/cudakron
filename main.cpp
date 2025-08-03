#include "skg.hpp"
#include "graph_io.hpp"
#include <chrono>
#include <cstdlib>
#include <exception>
#include <format>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <mpi.h>

namespace skg {

class MPIContext {
private:
    int rank_{0};
    int size_{1};
    
public:
    MPIContext(int argc, char* argv[]) {
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        MPI_Comm_rank(MPI_COMM_WORLD, &rank_);
        MPI_Comm_size(MPI_COMM_WORLD, &size_);
        
        if ((size_ & (size_ - 1)) != 0) {
            if (rank_ == 0) {
                std::cerr << "Error: Number of MPI processes (" << size_ 
                         << ") must be a power of 2\n";
            }
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
    }
    
    ~MPIContext() {
        MPI_Finalize();
    }
    
    [[nodiscard]] int Rank() const noexcept { return rank_; }
    [[nodiscard]] int Size() const noexcept { return size_; }
    [[nodiscard]] bool IsRoot() const noexcept { return rank_ == 0; }
};

class ArgumentParser {
private:
    GraphConfig config_;
    bool show_help_{false};
    std::string error_message_;
    
public:
    ArgumentParser() = default;
    
    bool Parse(int argc, char* argv[]) {
        std::vector<std::string_view> args(argv + 1, argv + argc);
        
        for (std::size_t i = 0; i < args.size(); ++i) {
            const auto& arg = args[i];
            
            if (arg == "-h" || arg == "--help") {
                show_help_ = true;
                return true;
            }
            
            if (i + 1 >= args.size()) {
                error_message_ = std::format("Option {} requires a value", arg);
                return false;
            }
            
            const auto& value = args[++i];
            
            try {
                if (arg == "-s" || arg == "--scaling") {
                    config_.scaling_factor = std::stoull(std::string(value));
                    if (config_.scaling_factor < 1 || config_.scaling_factor > 30) {
                        error_message_ = "Scaling factor must be between 1 and 30";
                        return false;
                    }
                } else if (arg == "-e" || arg == "--edges") {
                    config_.edge_factor = std::stoull(std::string(value));
                    if (config_.edge_factor < 1) {
                        error_message_ = "Edge factor must be positive";
                        return false;
                    }
                } else if (arg == "-a") {
                    config_.probability_matrix.set_a(std::stof(std::string(value)));
                } else if (arg == "-b") {
                    config_.probability_matrix.set_b(std::stof(std::string(value)));
                } else if (arg == "-c") {
                    config_.probability_matrix.set_c(std::stof(std::string(value)));
                    config_.probability_matrix.set_d(0);  
                } else if (arg == "--gpu") {
                    config_.use_gpu = true;
                    --i;  
                } else if (arg == "--format") {
                    config_.output_format = value;
                } else {
                    error_message_ = std::format("Unknown option: {}", arg);
                    return false;
                }
            } catch (const std::exception& e) {
                error_message_ = std::format("Invalid value for {}: {}", arg, e.what());
                return false;
            }
        }
        
        if (!config_.probability_matrix.IsValid()) {
            error_message_ = "Invalid probability matrix: sum must equal 1.0";
            return false;
        }
        
        return true;
    }
    
    [[nodiscard]] bool ShowHelp() const noexcept { return show_help_; }
    [[nodiscard]] const GraphConfig& Config() const noexcept { return config_; }
    [[nodiscard]] std::string_view ErrorMessage() const noexcept { return error_message_; }
    
    static void PrintHelp() {
        std::cout << R"(
Stochastic Kronecker Graph Generator 

Usage: mpiexec -n <processes> skg [options]

Options:
  -s, --scaling <int>    Scaling factor (2^s nodes) [default: 15]
  -e, --edges <int>      Edge factor (average degree) [default: 20]
  -a <float>             Probability matrix value a [default: 0.25]
  -b <float>             Probability matrix value b [default: 0.25]
  -c <float>             Probability matrix value c [default: 0.25]
                         (d is automatically calculated as 1 - a - b - c)
  --gpu                  Enable GPU acceleration (if available)
  --format <type>        Output format: binary, text, or both [default: binary]
  -h, --help             Show this help message

Requirements:
  - Number of MPI processes must be a power of 2
  - Probabilities must sum to 1.0 and be in range [0, 1]

Examples:
  mpiexec -n 4 skg -s 10 -e 16
  mpiexec -n 8 skg -s 15 -e 20 --gpu
  mpiexec -n 4 skg -s 12 -e 16 -a 0.57 -b 0.19 -c 0.19
)";
    }
};

class Timer {
private:
    using clock = std::chrono::high_resolution_clock;
    using time_point = clock::time_point;
    
    time_point start_;
    
public:
    Timer() : start_(clock::now()) {}
    
    void Reset() noexcept { start_ = clock::now(); }
    
    [[nodiscard]] double ElapsedSeconds() const noexcept {
        const auto duration = clock::now() - start_;
        return std::chrono::duration<double>(duration).count();
    }
};

class Application {
private:
    MPIContext mpi_context_;
    GraphConfig config_;
    PerformanceStats graph_generation_stats_;
    PerformanceStats csr_conversion_stats_;
    PerformanceStats io_stats_;
    
public:
    Application(int argc, char* argv[]) : mpi_context_(argc, argv) {
        ArgumentParser parser;
        
        if (!parser.Parse(argc, argv)) {
            if (mpi_context_.IsRoot()) {
                std::cerr << "Error: " << parser.ErrorMessage() << "\n";
                ArgumentParser::PrintHelp();
            }
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        }
        
        if (parser.ShowHelp()) {
            if (mpi_context_.IsRoot()) {
                ArgumentParser::PrintHelp();
            }
            MPI_Abort(MPI_COMM_WORLD, EXIT_SUCCESS);
        }
        
        config_ = parser.Config();
        
        #ifndef GPU_ENABLED
        if (config_.use_gpu && mpi_context_.IsRoot()) {
            std::cout << "Warning: GPU support not compiled. Using CPU implementation.\n";
            config_.use_gpu = false;
        }
        #endif
    }
    
    void Run() {
        if (mpi_context_.IsRoot()) {
            PrintConfiguration();
        }
        
        StochasticKroneckerGenerator generator(config_, 
                                              mpi_context_.Rank(), 
                                              mpi_context_.Size());
        
        Timer timer;
        auto csr_matrix = generator.Generate();
        const double generation_time = timer.ElapsedSeconds();
        
        timer.Reset();
        const double csr_time = 0.0;  
        
        timer.Reset();
        GraphIO io(mpi_context_.Rank(), mpi_context_.Size());
        
        if (config_.output_format == "text" || config_.output_format == "both") {
            io.WriteTextFormat("graph.txt", *csr_matrix);
        }
        
        if (config_.output_format == "binary" || config_.output_format == "both") {
            io.WriteBinaryFormat("graph.dat", *csr_matrix);
        }
        
        io.WriteStatistics("graph_stats.txt", *csr_matrix);
        const double io_time = timer.ElapsedSeconds();
        
        // Gather and print statistics
        GatherStatistics(generation_time, csr_time, io_time, *csr_matrix);
        
        if (mpi_context_.IsRoot()) {
            PrintResults(*csr_matrix);
        }
    }
    
private:
    void PrintConfiguration() const {
        std::cout << "\n╔═════════════════════════════════════════════════╗\n";
        std::cout << "║   Stochastic Kronecker Graph Generator            ║\n";
        std::cout << "╚═══════════════════════════════════════════════════╝\n\n";
        
        std::cout << "Configuration:\n";
        std::cout << "├─ Scaling Factor: " << config_.scaling_factor << "\n";
        std::cout << "├─ Matrix Size: " << config_.NumNodes() << " × " 
                  << config_.NumNodes() << "\n";
        std::cout << "├─ Edge Factor: " << config_.edge_factor << "\n";
        std::cout << "├─ Expected Edges: " << config_.ExpectedEdges() << "\n";
        std::cout << "├─ MPI Processes: " << mpi_context_.Size() << "\n";
        std::cout << "├─ GPU Acceleration: " 
                  << (config_.use_gpu ? "Enabled" : "Disabled") << "\n";
        std::cout << "└─ Probability Matrix: ["
                  << std::format("a={:.3f}, b={:.3f}, c={:.3f}, d={:.3f}",
                                config_.probability_matrix.a(),
                                config_.probability_matrix.b(),
                                config_.probability_matrix.c(),
                                config_.probability_matrix.d())
                  << "]\n\n";
    }
    
    void GatherStatistics(double gen_time, double csr_time, double io_time,
                         const CSRMatrix<>& matrix) {
        graph_generation_stats_.Update(gen_time);
        csr_conversion_stats_.Update(csr_time);
        io_stats_.Update(io_time);
        
        MPI_Allreduce(&gen_time, &graph_generation_stats_.average_time, 
                     1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        MPI_Allreduce(&gen_time, &graph_generation_stats_.min_time, 
                     1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
        MPI_Allreduce(&gen_time, &graph_generation_stats_.max_time, 
                     1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
        
        graph_generation_stats_.average_time /= mpi_context_.Size();
        
        if (csr_time > 0) {
            MPI_Allreduce(&csr_time, &csr_conversion_stats_.average_time,
                         1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            csr_conversion_stats_.average_time /= mpi_context_.Size();
        }
        
        MPI_Allreduce(&io_time, &io_stats_.average_time,
                     1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        io_stats_.average_time /= mpi_context_.Size();
    }
    
    void PrintResults(const CSRMatrix<>& matrix) const {
        std::int64_t total_edges = matrix.NumEdges();
        MPI_Allreduce(MPI_IN_PLACE, &total_edges, 1, MPI_LONG_LONG, 
                     MPI_SUM, MPI_COMM_WORLD);
        
        std::cout << "Performance Results:\n";
        std::cout << "├─ Graph Generation: " 
                  << graph_generation_stats_.ToString() << "\n";
        
        if (csr_conversion_stats_.average_time > 0) {
            std::cout << "├─ CSR Conversion: " 
                      << csr_conversion_stats_.ToString() << "\n";
        }
        
        std::cout << "├─ I/O Operations: " 
                  << io_stats_.ToString() << "\n";
        std::cout << "└─ Total Edges Generated: " << total_edges 
                  << " (expected: " << config_.ExpectedEdges() << ")\n\n";
        
        const double efficiency = static_cast<double>(total_edges) / 
                                 config_.ExpectedEdges() * 100.0;
        std::cout << "Generation Efficiency: " 
                  << std::format("{:.2f}%", efficiency) << "\n";
        
        if (std::abs(efficiency - 100.0) > 5.0) {
            std::cout << "Warning: Edge count deviation > 5%\n";
        }
        
        std::cout << "\n✓ Graph generation completed successfully.\n";
    }
};

}  

int main(int argc, char* argv[]) {
    try {
        skg::Application app(argc, argv);
        app.Run();
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << "\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    } catch (...) {
        std::cerr << "Unknown fatal error occurred\n";
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
}