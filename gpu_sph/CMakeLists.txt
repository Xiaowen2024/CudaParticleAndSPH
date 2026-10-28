cmake_minimum_required(VERSION 3.20)
project(sph_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

add_executable(sph_cuda src/main.cu src/sph.cu src/sph.cuh)
target_compile_options(sph_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
