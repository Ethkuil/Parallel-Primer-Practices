#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include <algorithm>
#include <functional>

#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

#include "inc_cuda.h"
#include "inc_serial.h"

using IncImpl = void(int *A, const size_t n);

void verify_inc(int *before, int *after, size_t n) {
  for (int i = 0; i < n; ++i) {
    if (after[i] != before[i] + 1) {
      printf("Verification failed!\n");
      exit(1);
    }
  }
}

void test_inc(benchmark::State &state, IncImpl fn, bool is_gpu) {
  size_t n = state.range(0);

  // allocate
  int *A = new int[n];
  int *A_bak = new int[n];
  int *A_gpu;
  cudaMalloc(&A_gpu, sizeof(int) * n);

  // init
  for (int i = 0; i < n; ++i) {
    A[i] = rand();
  }
  std::copy_n(A, n, A_bak);
  cudaMemcpy(A_gpu, A, sizeof(int) * n, cudaMemcpyHostToDevice);

  std::function<void()> run_once =
      is_gpu ? std::function<void()>([&]() {
        fn(A_gpu, n);
        cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
          printf("CUDA error: %s\n", cudaGetErrorString(error));
          exit(1);
        }
      })
             : std::function<void()>([&]() { fn(A, n); });

  // verify
  run_once();
  if (is_gpu) {
    cudaMemcpy(A, A_gpu, sizeof(int) * n, cudaMemcpyDeviceToHost);
  }
  verify_inc(A_bak, A, n);

  run_once(); // warmup
  // benchmark
  for (auto _ : state) {
    run_once();
  }
  state.SetComplexityN(state.range(0));

  delete[] A;
  delete[] A_bak;
  cudaFree(A_gpu);
}

#define MY_BENCHMARK_CONFIG                                                    \
  Range(1 << 6, 1 << 18)->Unit(benchmark::kMicrosecond)
BENCHMARK_CAPTURE(test_inc, serial, inc_serial, false)
    ->MY_BENCHMARK_CONFIG->Complexity(benchmark::oN);
BENCHMARK_CAPTURE(test_inc, cuda_1thread, inc_cuda_1thread, true)
    ->MY_BENCHMARK_CONFIG->Complexity(benchmark::oN);
BENCHMARK_MAIN();
