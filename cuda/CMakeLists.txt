cmake_minimum_required(VERSION 3.11)
project(hw2_3)
enable_language(CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED YES)

# Group number
set(GROUP_NAME "None" CACHE STRING "Your group name as it appears on bCourses (no spaces)")

# Packaging system
if (NOT ${GROUP_NAME} STREQUAL None)
    set(CPACK_GENERATOR TGZ)
    set(CPACK_PACKAGE_FILE_NAME "cs267${GROUP_NAME}_hw2_3")
    install(FILES gpu.cu DESTINATION .)
    install(FILES ${CPACK_PACKAGE_FILE_NAME}.pdf DESTINATION .)
    include(CPack)
endif ()

# We require the GNU compiler for this assignment.
if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Intel "PrgEnv-intel")
    set(Prg-Clang "PrgEnv-cray")
    message(WARNING
            "Must use GNU Compiler for submission. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
endif ()

function(configure_cuda_target target)
    add_executable(${target} ${ARGN})
    target_compile_features(${target} PRIVATE cxx_std_17)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    set_target_properties(${target} PROPERTIES CUDA_ARCHITECTURES "80")
endfunction()

# Same cell-list implementation, isolating the effects of SoA and periodic reordering.
configure_cuda_target(soa_reorder main.cu gpu.cu common.h)
target_compile_definitions(soa_reorder PRIVATE ENABLE_REORDER=1 REORDER_INTERVAL=10)
configure_cuda_target(soa_no_reorder main.cu gpu.cu common.h)
target_compile_definitions(soa_no_reorder PRIVATE ENABLE_REORDER=0)

function(add_soa_reorder_sweep interval)
    configure_cuda_target(soa_reorder_${interval} main.cu gpu.cu common.h)
    target_compile_definitions(soa_reorder_${interval} PRIVATE ENABLE_REORDER=1 REORDER_INTERVAL=${interval})
endfunction()

add_soa_reorder_sweep(1)
add_soa_reorder_sweep(2)
add_soa_reorder_sweep(5)
add_soa_reorder_sweep(20)
add_soa_reorder_sweep(50)
add_soa_reorder_sweep(100)
configure_cuda_target(aos_reorder main_aos.cu gpu_aos.cu common_aos.h)
target_compile_definitions(aos_reorder PRIVATE ENABLE_REORDER=1)
configure_cuda_target(aos_baseline main_aos.cu gpu_aos.cu common_aos.h)
target_compile_definitions(aos_baseline PRIVATE ENABLE_REORDER=0)

# Preserve the original executable name as an alias of the current SoA + reorder version.
configure_cuda_target(gpu main.cu gpu.cu common.h)
target_compile_definitions(gpu PRIVATE ENABLE_REORDER=1 REORDER_INTERVAL=10)
configure_cuda_target(gpu_float main.cu gpu.cu common.h)
target_compile_definitions(gpu_float PRIVATE PARTICLE_USE_FLOAT=1 ENABLE_REORDER=1 REORDER_INTERVAL=10)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_80")

# Copy the job scripts
configure_file(job-gpu job-gpu COPYONLY)
