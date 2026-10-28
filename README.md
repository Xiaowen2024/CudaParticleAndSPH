# Parallel Particle and CUDA SPH Simulations

This repository contains two related CUDA/parallel-computing projects:

- `openmp/` and `cuda/`: CS267 2D particle-simulation implementations.
- `cpu_sph/`: CPU reference implementation of a 3D Smoothed-Particle
  Hydrodynamics (SPH) fluid solver.
- `gpu_sph/`: CUDA SPH solver with a cell list, SoA state, spatial reordering,
  CUB scans, and a reusable CSR neighbor list.

The CUDA SPH solver is benchmarked at one million particles on an NVIDIA L40S.
It uses a scalable simulation domain, reducing the valid 1M timestep from
14.754 ms to 7.897 ms through arithmetic cleanup and neighbor-list reuse.
See `gpu_sph/benchmarks/OPTIMIZATION_JOURNAL.md` for profiling evidence and
the optimization history.

Each directory is a standalone CMake project. Build from the corresponding
directory, for example:

```bash
cmake -S openmp -B openmp/build
cmake --build openmp/build

cmake -S cuda -B cuda/build
cmake --build cuda/build

cmake -S gpu_sph -B gpu_sph/build -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build gpu_sph/build
./gpu_sph/build/sph_cuda --grid 100 --steps 10 --neighbor-list 1
```
