cmake_minimum_required(VERSION 2.8)
project( parallel_test )

find_package( OpenCV REQUIRED )

find_package(CUDA REQUIRED)
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_22,code=sm_22
    )

add_executable( parallel_test mycudatest.cpp )
target_link_libraries( parallel_test ${OpenCV_LIBS} )
