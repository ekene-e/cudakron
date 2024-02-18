CFLAGS = -O3 -lm
MCC = mpic++
NVCC = nvcc
CUDA_LIB64 = -L/software/spackages/linux-rocky8-x86_64/gcc-9.5.0/cuda-11.6.2-er5txg5a4g3a7xzhmtvncdmgbzqcir2s/lib64
CUDA_FLAGS = -lcudart


all: build

build: skg skg_gpu

skg: skg.o
	$(MCC) $(CFLAGS) $< -o $@

skg.o: skg.cpp
	$(MCC) $(CFLAGS) $< -c -o $@

skg_gpu: skg_gpu.o skg_kernel.o
	$(MCC) $(CFLAGS) $< skg_kernel.o -o $@ $(CUDA_LIB64) $(CUDA_FLAGS)	

skg_gpu.o: skg.cpp
	$(MCC) $(CFLAGS) -D_GPU $< -c -o $@

skg_kernel.o: skg_kernel.cu
	$(NVCC) $< -c -o $@

clean:
	rm -rf skg skg_gpu core* *.o