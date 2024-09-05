cmake_minimum_required(VERSION 3.12)
project(practices VERSION 0.1.0 LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE Release)

include_directories(/usr/local/cuda/include)
find_package(benchmark REQUIRED)

add_executable(inc_test inc_test.cc inc_serial.cc inc_cuda.cu)
target_link_libraries(inc_test benchmark::benchmark)
