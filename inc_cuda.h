#pragma once

#include <cstddef>

void inc_cuda_1thread(int *A, const size_t n);

void inc_cuda_multi(int *A, const size_t n);
