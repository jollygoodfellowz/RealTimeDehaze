cmake_minimum_required(VERSION 2.8)
project( display_image )

find_package( OpenCV REQUIRED )

find_package(CUDA REQUIRED)
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -O3 -gencode arch=compute_22,code=sm_22
    )

find_package(OpenMP)
if (OPENMP_FOUND)
	    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
	        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

option(GPU "GPU" ON)
option(NEW "NEW" ON)

if(${GPU} AND ${NEW})
	add_executable ( display_image Dehaze-gpu-update.cpp )
elseif(${GPU} AND NOT ${NEW})
	add_executable ( display_image Dehaze-gpu-original.cpp )
elseif(NOT ${GPU} AND ${NEW})
	add_executable ( display_image dehaze-update-nongpu.cpp )
else(NOT ${GPU} AND NOT ${NEW})
	add_executable ( display_image dehaze-original-nongpu.cpp )
endif(${GPU} AND ${NEW})

target_link_libraries( display_image ${OpenCV_LIBS} )
