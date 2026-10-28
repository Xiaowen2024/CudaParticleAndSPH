cmake_minimum_required(VERSION 3.20)
project(sph_cpu_baseline LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(sph_cpu
    src/main.cpp
    src/sph.cpp)
target_compile_options(sph_cpu PRIVATE -O3 -Wall -Wextra -Wpedantic)
