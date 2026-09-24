cmake_minimum_required(VERSION 3.14)
project(hw2_1)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)

find_package(OpenMP QUIET)
if (NOT OpenMP_CXX_FOUND)
    if (APPLE AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        find_path(OPENMP_INCLUDE_DIR omp.h
                PATHS /opt/homebrew/opt/libomp/include /usr/local/opt/libomp/include)
        find_library(OPENMP_LIBRARY omp
                PATHS /opt/homebrew/opt/libomp/lib /usr/local/opt/libomp/lib)

        if (OPENMP_INCLUDE_DIR AND OPENMP_LIBRARY)
            add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)
            set_target_properties(OpenMP::OpenMP_CXX PROPERTIES
                    INTERFACE_COMPILE_OPTIONS "-Xpreprocessor;-fopenmp"
                    INTERFACE_INCLUDE_DIRECTORIES "${OPENMP_INCLUDE_DIR}"
                    INTERFACE_LINK_LIBRARIES "${OPENMP_LIBRARY}")
        else ()
            message(FATAL_ERROR
                    "OpenMP was not found. Install libomp with Homebrew, then rerun CMake.")
        endif ()
    else ()
        find_package(OpenMP REQUIRED)
    endif ()
endif ()

# Group number
set(GROUP_NAME "None" CACHE STRING "Your group name as it appears on bCourses (no spaces)")

# Packaging system
if (NOT ${GROUP_NAME} STREQUAL None)
    set(CPACK_GENERATOR TGZ)
    set(CPACK_PACKAGE_FILE_NAME "cs267${GROUP_NAME}_hw2_1")
    install(FILES serial.cpp openmp.cpp DESTINATION .)
    install(FILES ${CPACK_PACKAGE_FILE_NAME}.pdf DESTINATION .)
    include(CPack)
endif ()

# We require the GNU compiler for this assignment.
if (NOT "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
    set(Prg-Clang "PrgEnv-cray")
    message(WARNING
            "Must use GNU Compiler for submission. Make sure you ran:\n"
            "module swap ${Prg-${CMAKE_C_COMPILER_ID}} PrgEnv-gnu")
endif ()

add_executable(serial main.cpp serial.cpp common.h)

add_executable(openmp main.cpp openmp.cpp common.h)
target_link_libraries(openmp PRIVATE OpenMP::OpenMP_CXX)

# Copy the job scripts
configure_file(job-serial job-serial COPYONLY)
configure_file(job-openmp job-openmp COPYONLY)
