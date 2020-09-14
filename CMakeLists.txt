cmake_minimum_required(VERSION 3.17)
project(TensorScript)

include(CTest)

set(CXX_STANDARD 14)

find_package(CUDA)
find_package(CUDAToolkit)

cuda_add_executable(ts src/main.cpp)
target_link_libraries(ts CUDA::cudart)

include_directories(include)
add_subdirectory(test)