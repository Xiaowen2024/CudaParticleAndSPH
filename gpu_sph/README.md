# CUDA Smoothed-Particle Hydrodynamics

This directory contains a CUDA implementation of a 3D SPH fluid solver. The
CPU reference implementation is in ../cpu_sph.

## GPU pipeline

    cell count -> CUB prefix scan -> cell scatter -> spatial reorder
    -> true-neighbor count -> CUB prefix scan -> density + neighbor-list fill
    -> force -> integration and boundary constraints

Key implementation choices:

- Persistent device allocations; no per-step cudaMalloc or cudaFree.
- Structure-of-arrays particle state for position, velocity, force, density,
  and pressure.
- Reordering complete particle state into cell order every step, so particles
  inside a cell occupy contiguous ranges in memory.
- CSR cell and true-neighbor lists built with CUB exclusive scans.
- Explicit reuse of per-neighbor FP64 terms such as 1/r, 1/density, h-r, and
  h^2-r^2 in force evaluation.
- A domain that grows with --grid, preserving initial lattice spacing and
  preventing boundary-cell clustering at large particle counts.

## Build

On a CUDA machine:

    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
    cmake --build build -j

Run the one-million-particle benchmark:

    ./build/sph_cuda --grid 100 --steps 10 --neighbor-list 1

Use --neighbor-list 0 to run the direct 27-cell traversal baseline. The CSR
list mode is the default.

## Results

On an NVIDIA L40S, valid 1M-particle measurements were:

    direct cell traversal with arithmetic cleanup: 13.292 ms/step
    CSR true-neighbor list:                        7.897 ms/step
    CPU reference:                              2994.392 ms/step

The CSR neighbor list improves the optimized direct GPU implementation by
1.68x; the final CUDA solver is approximately 379x faster than the same-host
CPU reference for this configuration.

See benchmarks/GPU_VS_CPU.md for the short benchmark report and
benchmarks/OPTIMIZATION_JOURNAL.md for profiling data, design decisions, and
known limitations.
