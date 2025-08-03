CXX_STD ?= c++20

ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
else
    DETECTED_OS := Linux
endif

ifeq ($(DETECTED_OS),Windows)
    MPI_INC = C:\Program Files (x86)\Microsoft SDKs\MPI\Include
    MPI_LIB = C:\Program Files (x86)\Microsoft SDKs\MPI\Lib\x64
    
    CUDA_PATH = C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.3
    
    CXX = g++
    CXXFLAGS = -O3 -std=$(CXX_STD)
    
    MPI_CFLAGS = -I"$(MPI_INC)"
    MPI_LDFLAGS = -L"$(MPI_LIB)" -lmsmpi
    
    CUDA_INC = -I"$(CUDA_PATH)\include"
    CUDA_LIBS = -L"$(CUDA_PATH)\lib\x64" -lcudart -lcurand
    
    LDFLAGS = -mconsole
    
    NVCC_EXISTS := $(shell if exist "$(CUDA_PATH)\bin\nvcc.exe" echo yes)
    ifdef NVCC_EXISTS
        NVCC = "$(CUDA_PATH)\bin\nvcc.exe"
    endif
    
    CL_EXISTS := $(shell where cl 2>NUL)
    
    EXE = .exe
    RM = del /Q
else
    CXX = mpic++
    CXXFLAGS = -O3 -std=$(CXX_STD)
    MPI_CFLAGS = 
    MPI_LDFLAGS = 
    
    CUDA_PATH = /usr/local/cuda
    CUDA_INC = -I$(CUDA_PATH)/include
    CUDA_LIBS = -L$(CUDA_PATH)/lib64 -lcudart -lcurand
    
    LDFLAGS = 
    NVCC = nvcc
    NVCC_EXISTS := $(shell which nvcc 2>/dev/null)
    CL_EXISTS = yes
    
    EXE = 
    RM = rm -f
endif

GPU_ARCH = sm_75

all: skg$(EXE) skg_gpu$(EXE)

skg$(EXE): skg.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $< -o $@ $(MPI_LDFLAGS)
	@echo "CPU version built successfully!"

skg.o: skg.cpp skg.h graph.h graph_io.h
	$(CXX) $(CXXFLAGS) $(MPI_CFLAGS) -c skg.cpp -o $@

skg_gpu$(EXE): skg_gpu.o kernelised_skg.o
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $^ -o $@ $(MPI_LDFLAGS) $(CUDA_LIBS)
	@echo "GPU version built successfully!"

skg_gpu.o: skg.cpp skg.h graph.h graph_io.h
	$(CXX) $(CXXFLAGS) $(MPI_CFLAGS) $(CUDA_INC) -D_GPU -c skg.cpp -o $@

kernelised_skg.o: 
ifeq ($(DETECTED_OS),Windows)
    ifdef NVCC_EXISTS
        ifdef CL_EXISTS
			@echo "Compiling CUDA kernel..."
			$(NVCC) -c kernelised_skg.cu -o $@ -arch=$(GPU_ARCH)
        else
			@echo "Warning: NVCC requires MSVC (cl.exe). Using stub..."
			$(CXX) $(CXXFLAGS) -c cuda_stub.cpp -o $@
        endif
    else
		@echo "Warning: CUDA not found. Using stub..."
		$(CXX) $(CXXFLAGS) -c cuda_stub.cpp -o $@
    endif
else
    ifdef NVCC_EXISTS
		$(NVCC) -c kernelised_skg.cu -o $@ -arch=$(GPU_ARCH)
    else
		@echo "Warning: CUDA not found. Using stub..."
		$(CXX) $(CXXFLAGS) -c cuda_stub.cpp -o $@
    endif
endif

clean:
	-$(RM) *.o *.obj *.exe skg skg_gpu *.dat *.txt 2>NUL

test: skg$(EXE)
	mpiexec -n 4 skg$(EXE) -s 10 -e 16

test_gpu: skg_gpu$(EXE)
	mpiexec -n 4 skg_gpu$(EXE) -s 10 -e 16

info:
	@echo "===== Build Configuration ====="
	@echo "OS: $(DETECTED_OS)"
	@echo "Compiler: $(CXX)"
	@echo "C++ Standard: $(CXX_STD)"
	@echo "Flags: $(CXXFLAGS)"
	@echo "MPI Include: $(MPI_INC)"
	@echo "MPI Library: $(MPI_LIB)"
	@echo "CUDA Path: $(CUDA_PATH)"
	@echo "CUDA Available: $(NVCC_EXISTS)"
	@echo "MSVC Available: $(CL_EXISTS)"
	@echo "================================"

.PHONY: all clean test test_gpu info