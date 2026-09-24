# OpenMP and CUDA Particle Simulation

This repository contains two CS267 particle-simulation implementations:

- `openmp/`: serial and OpenMP CPU implementations.
- `cuda/`: CUDA implementation with cell lists, SoA storage, and optional spatial reordering.

Each directory is a standalone CMake project. Build from the corresponding directory, for example:

```bash
cmake -S openmp -B openmp/build
cmake --build openmp/build

cmake -S cuda -B cuda/build
cmake --build cuda/build
```
