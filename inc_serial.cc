#include "inc_serial.h"

void inc_serial(int *A, const size_t n) {
  for (auto i = 0; i < n; ++i) {
    ++A[i];
  }
}
