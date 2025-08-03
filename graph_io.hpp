#ifndef SKG_GRAPH_IO_HPP_
#define SKG_GRAPH_IO_HPP_

#include "skg.hpp"
#include <filesystem>
#include <fstream>
#include <span>
#include <string_view>

namespace skg {
class GraphIO {
private:
    std::int32_t mpi_rank_;
    std::int32_t mpi_size_;
    
    static constexpr std::uint32_t kBinaryMagicNumber = 0x53474B52;  // "RKGS"
    static constexpr std::uint32_t kBinaryVersion = 1;
    
public:
    GraphIO(std::int32_t rank, std::int32_t size) 
        : mpi_rank_(rank), mpi_size_(size) {}
    
    void WriteTextFormat(std::filesystem::path filepath, 
                        const CSRMatrix<>& matrix) const;
    
    void WriteBinaryFormat(std::filesystem::path filepath, 
                          const CSRMatrix<>& matrix) const;
    
    void WriteStatistics(std::filesystem::path filepath, 
                        const CSRMatrix<>& matrix) const;
    
    [[nodiscard]] std::unique_ptr<CSRMatrix<>> ReadBinaryFormat(
        std::filesystem::path filepath) const;
    template<typename OutputIterator>
    void StreamEdges(const CSRMatrix<>& matrix, OutputIterator out) const {
        const auto row_ptrs = matrix.RowPointers();
        const auto col_indices = matrix.ColumnIndices();
        const auto values = matrix.Values();
        const std::int64_t base_node = mpi_rank_ * matrix.NumNodes();
        
        for (std::int64_t i = 0; i < matrix.NumNodes(); ++i) {
            const std::int64_t global_node = base_node + i;
            
            for (auto j = row_ptrs[i]; j < row_ptrs[i + 1]; ++j) {
                *out++ = DefaultEdge{global_node, col_indices[j], values[j]};
            }
        }
    }
    
private:
    struct BinaryHeader {
        std::uint32_t magic_number{kBinaryMagicNumber};
        std::uint32_t version{kBinaryVersion};
        std::int64_t total_nodes{0};
        std::int64_t total_edges{0};
        std::int32_t num_processes{0};
        std::int32_t reserved{0};  
        
        [[nodiscard]] bool IsValid() const noexcept {
            return magic_number == kBinaryMagicNumber && 
                   version == kBinaryVersion;
        }
    };
    
    static_assert(sizeof(BinaryHeader) == 32, "Header must be 32 bytes");
};

class ParallelFileWriter {
private:
    std::filesystem::path filepath_;
    MPI_File file_handle_{};
    bool is_open_{false};
    
public:
    explicit ParallelFileWriter(std::filesystem::path path) 
        : filepath_(std::move(path)) {}
    
    ~ParallelFileWriter() {
        if (is_open_) {
            Close();
        }
    }
    
    ParallelFileWriter(const ParallelFileWriter&) = delete;
    ParallelFileWriter& operator=(const ParallelFileWriter&) = delete;
    ParallelFileWriter(ParallelFileWriter&& other) noexcept;
    ParallelFileWriter& operator=(ParallelFileWriter&& other) noexcept;
    
    void Open();
    void Close();
    
    template<typename T>
    void Write(std::span<const T> data, MPI_Offset offset) {
        static_assert(std::is_trivially_copyable_v<T>, 
                     "Type must be trivially copyable for MPI I/O");
        
        if (!is_open_) {
            throw std::runtime_error("File not open");
        }
        
        MPI_Status status;
        MPI_File_write_at(file_handle_, offset, 
                         data.data(), 
                         static_cast<int>(data.size() * sizeof(T)),
                         MPI_BYTE, &status);
        
        int count;
        MPI_Get_count(&status, MPI_BYTE, &count);
        
        if (count != static_cast<int>(data.size() * sizeof(T))) {
            throw std::runtime_error("Failed to write all data");
        }
    }
    
    template<typename T>
    void WriteCollective(std::span<const T> data, MPI_Offset offset) {
        static_assert(std::is_trivially_copyable_v<T>, 
                     "Type must be trivially copyable for MPI I/O");
        
        if (!is_open_) {
            throw std::runtime_error("File not open");
        }
        
        MPI_Status status;
        MPI_File_write_at_all(file_handle_, offset,
                             data.data(),
                             static_cast<int>(data.size() * sizeof(T)),
                             MPI_BYTE, &status);
    }
};

}  // namespace skg

#endif  // SKG_GRAPH_IO_HPP_