#include "inc_cuda.h"

__global__ void inc_1thread_kernel(int *A, const size_t n) {
  for (auto i = 0; i < n; ++i) {
    ++A[i];
  }
}

void inc_cuda_1thread(int *A, const size_t n) {
  inc_1thread_kernel<<<1, 1>>>(A, n);
}
