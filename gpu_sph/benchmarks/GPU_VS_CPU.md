# CPU and CUDA SPH benchmark

## Test setup

- Remote machine: Intel Xeon Gold 6338 (8 logical CPUs) with an NVIDIA L40S
  GPU (compute capability 8.9).
- CUDA: 13.0; CPU compiler: GCC 13.3; both builds use Release mode.
- Input: `--grid 17 --steps 100`, or `17^3 = 4913` particles.
- Precision: `double` in both implementations.
- CPU timing measures only `SphSimulation::run()`.
- GPU timing uses CUDA events around `SphGpuSimulation::run()`; it excludes host
  initialization, host/device copies, checksum generation, and validation.

## Repeated timing results

CPU milliseconds per timestep:

```text
37.063292
37.153992
37.170239
```

Median CPU time: **37.153992 ms/step**.

GPU milliseconds per timestep:

```text
2.903567
2.901759
2.900534
```

Median GPU time: **2.901759 ms/step** for the original AoS, indirect-cell-ID
implementation.

The same-machine median speedup is therefore **12.804x**.

## SoA and spatial-reorder update

The CUDA state was then converted from AoS to SoA and the complete particle
state is reordered into cell order after each cell-list rebuild.  Consequently,
the range `[cell_offsets[cell], cell_offsets[cell] + cell_counts[cell])` is a
contiguous range of particles and density/force no longer dereference
`particle_ids` for every neighbor.

Repeated GPU timing after that change (same input and machine):

```text
1.820844 ms/step
1.825871 ms/step
1.821988 ms/step
```

Median: **1.821988 ms/step**, or **1.59x** faster than the prior GPU version.
Nsight Systems measured the new `reorder_particles` kernel at only about
**0.0028 ms/step**; its cost is substantially smaller than the reduction in
the two neighbor kernels.

## One-million-particle benchmark

Before creating particles, both programs now expand each box dimension to at
least `2 * grid_length * particle_radius`. This keeps the initial lattice in
the domain while preserving particle spacing, mass, and smoothing length.
For `--grid 100`, the resulting domain is `4.245559 x 4.245559 x 4.245559`
and the GPU cell grid is `59 x 59 x 59 = 205379` cells.

GPU timing uses ten simulation steps per sample after warm-up:

```text
14.740096 ms/step
14.760576 ms/step
14.754486 ms/step
```

Median GPU time: **14.754486 ms/step**. CPU one-step samples were
`2998.484`, `2994.392`, and `2989.176 ms`, with a median of
**2994.392 ms/step**. The same-host 1M speedup is therefore **203.0x**.

NSYS at this valid 1M configuration attributes 72.7% of GPU kernel time to
force (10.756 ms/step) and 24.9% to density (3.680 ms/step). Reordering costs
only 0.240 ms/step (1.6%). NCU reports 86.82% compute throughput for force,
with only 0.36% DRAM throughput: the next optimization target remains FP64
force computation rather than global-memory bandwidth.

## CSR neighbor-list and force-arithmetic update

The GPU can now be run with `--neighbor-list 0|1`. The default (`1`) first
counts true radius-neighbors, performs a CUB exclusive scan, and stores their
IDs in a CSR list. Density fills that list while computing density; force then
reuses it instead of scanning 27 cells a second time. Force also explicitly
reuses `1/r`, `1/density`, `h-r`, and `h^2-r^2` within each neighbor iteration.

At 1M particles, three ten-step samples gave these medians:

```text
direct cells, optimized arithmetic: 13.292 ms/step
CSR neighbor list:                  7.897 ms/step
```

That is a **1.68x** speedup for CSR neighbor reuse, with the same 10-step
checksum (`521422836.199754`). NSYS shows why: force falls to 1.887 ms/step,
but list count and density/list-fill cost 2.259 and 3.335 ms/step respectively.
The total still wins because force no longer re-enumerates cell candidates.

## Execution checks

The CUDA build and all three compute stages completed on the L40S for:

```text
grid 1:  1 particle
grid 17: 4913 particles
grid 20: 8000 particles
```

For the default grid, the dense 3D cell grid has `21 x 7 x 7 = 1029` cells.
The cell-list validation confirmed that all particles were represented by the
cell counts after execution.

## Numerical comparison

After one step:

```text
CPU checksum: 2374402.648059
GPU checksum: 2374138.979539
relative difference: 0.000111  (about 0.0111%)
```

This is a useful early correctness signal: the GPU result is close, but it is
not expected to be bitwise identical to the CPU result.

After 100 steps, CPU checksum remains deterministic on this input, while GPU
checksums vary slightly across runs. The reason is that `atomicAdd` during
cell-list scatter can produce a different order inside a cell's `particle_ids`
range. Each thread then sums neighbors in a slightly different floating-point
order; SPH is nonlinear, so the small differences accumulate over many steps.

This behavior should be treated as a numerical-validation item, not as a
performance result. For stronger validation, next compare selected particle
positions/densities after a small number of steps and define an acceptable
relative-error threshold.

## Scope and next steps

The GPU solver currently performs:

```text
cell-list rebuild -> density/pressure -> force -> integration/boundary
```

The current high-value follow-ups are checksum/state validation and profiling
with Nsight Systems / Nsight Compute. Avoid comparing against the older local
macOS CPU timings: only the same-host comparison above is suitable for a
speedup claim.
