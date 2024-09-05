#include <cstddef>
#include <cstdio>
#include <cstdlib>

#include <algorithm>

#include <benchmark/benchmark.h>

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

void test_inc(benchmark::State &state, IncImpl fn, size_t n) {
  // allocate
  int *A = new int[n];
  int *A_bak = new int[n];

  // init
  for (int i = 0; i < n; ++i) {
    A[i] = rand();
  }
  std::copy_n(A, n, A_bak);

  // verify
  fn(A, n);
  verify_inc(A_bak, A, n);

  fn(A, n);
  // benchmark
  for (auto _ : state) {
    fn(A, n);
  }

  delete[] A;
  delete[] A_bak;
}

constexpr size_t N = 1 << 18; // 64 * 64 * 64
BENCHMARK_CAPTURE(test_inc, serial, inc_serial, N)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_MAIN();
