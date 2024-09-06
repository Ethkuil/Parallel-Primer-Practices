#include "inc_cuda.h"

__global__ void inc_1thread_kernel(int *A, const size_t n) {
  for (auto i = 0; i < n; ++i) {
    ++A[i];
  }
}

void inc_cuda_1thread(int *A, const size_t n) {
  inc_1thread_kernel<<<1, 1>>>(A, n);
}

constexpr size_t BLOCK_SIZE = 1024;

__global__ void inc_multi_kernel(int *A, const size_t n) {
  for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x) {
    ++A[i];
  }
}

void inc_cuda_multi(int *A, const size_t n) {
  auto num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
  inc_multi_kernel<<<num_blocks, BLOCK_SIZE>>>(A, n);
}
