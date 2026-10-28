# CUDA SPH optimization journal

This is the detailed record of the CUDA SPH solver: the problems encountered,
their causes, the changes made, and the measurements that justified them.
The compact benchmark summary is in GPU_VS_CPU.md.

## Environment and timing method

- GPU: NVIDIA L40S, compute capability 8.9, CUDA 13.0.
- CPU: Intel Xeon Gold 6338, GCC 13.3, Release build.
- GPU time is measured with CUDA events around the simulation loop only.
  Initialization, allocation, checksums, validation, and host/device setup are
  excluded.
- CPU time is wall-clock time around SphSimulation::run().
- NSYS assigns time to CUDA kernels. NCU reports hardware utilization of a
  selected kernel. NCU replays kernels, so its program runtime is not used as
  a benchmark.

## CPU reference design

The original CPU solver has this structure:

    unordered_map<cell hash, vector<particle IDs>>
    -> vector<vector<int>> neighbor lists
    -> density/pressure loop
    -> force loop
    -> integration loop

This is valid CPU code, but vectors and unordered_map cannot be manipulated
from CUDA kernels. The GPU version needs global-memory arrays and an explicit
representation of variable-length lists.

## GPU cell-list foundation

The first GPU implementation replaced the CPU hash-map cells with a dense CSR
cell list:

    cell_counts[cell]     number of particles in the cell
    cell_offsets[cell]    exclusive-prefix-sum start of its range
    particle_ids[slot]    particle ID in that range

Each timestep builds it as:

    clear counts
    -> count_particles_per_cell using atomicAdd
    -> CUB ExclusiveSum, counts to offsets
    -> CUB ExclusiveSum, counts to cursors
    -> scatter_particle_ids using atomicAdd on cursors

All persistent GPU arrays and CUB scan temporary storage are allocated once in
initialize and freed in cleanup. This avoids the earlier risk of per-step
cudaMalloc/cudaFree overhead.

## Issue 1: AoS and indirect neighbor access

### Initial form

The first device layout was Array of Structures:

    Particle = position + velocity + force + density + pressure

Each record is 88 bytes. Density and force often needed only a subset of those
fields. Also, a cell range was contiguous only in particle_ids; it pointed
back to the original initialization-order Particle array.

### Fix: SoA plus spatial reorder

The device state was converted to these independent arrays:

    x, y, z
    vx, vy, vz
    fx, fy, fz
    density, pressure

After cell scatter, reorder_particles copies every field into a second SoA
buffer in cell order, then the two buffers are swapped. Therefore:

    [cell_offsets[c], cell_offsets[c] + cell_counts[c])

is a contiguous cell range in the actual particle data. Density and force can
directly use the slot as the particle index; particle_ids is only needed by
the reorder kernel.

On the earlier small-grid configuration this reduced median time from about
2.902 to 1.822 ms/step, a 1.59x improvement. These old small-grid values are
historical and should not be combined with later large-domain results.

### Numerical consequence

Atomic scatter does not guarantee a stable ordering of particles within a
cell. Reordering can therefore change floating-point summation order.
Long-running CPU/GPU comparisons should use numerical tolerances rather than
bitwise equality. An early one-step check had roughly 0.0111% aggregate
checksum difference; this validation should be repeated after the scalable
domain change.

## Issue 2: invalid one-million-particle benchmark

### Symptom

An early grid=100 run gave roughly 10.9 seconds per warm GPU step while the
CPU needed about 3 seconds. This was a configuration error, not evidence that
the GPU algorithm was slower.

### Cause

Grid=100 creates a 100 by 100 by 100 initial lattice, but the box was still:

    1.5 x 0.5 x 0.5

GPU cell_index clamps positions outside this box into boundary cells. Huge
fractions of the million particles landed in the same boundary cells. Every
particle scans 27 cells, so a single very full cell caused huge numbers of
rejected distance tests.

The CPU reference hash grid initially used actual position coordinates rather
than clamped dense-cell coordinates. At this point CPU and GPU no longer had
the same spatial partition, so that comparison was invalid.

NSYS diagnosed the bad configuration:

    density: about 14.76 seconds
    force:   about  9.62 seconds
    reorder: under 0.4 milliseconds

Thus the problem was not reorder or memory allocation. It was excessive
candidate-neighbor arithmetic.

### Fix: scalable domain

Both CPU and GPU call fit_domain_to_initial_lattice after parsing grid size:

    radius = cbrt(3 * mass / (4 * pi * rest_density))
    minimum extent = 2 * grid_length * radius
    each box axis = max(original axis, minimum extent)

This keeps particle spacing, particle mass, smoothing length, and intended
local density unchanged, while making the initial lattice valid.

For grid=100 the corrected configuration is:

    box = 4.245559 x 4.245559 x 4.245559
    cells = 59 x 59 x 59 = 205379 cells

This is the configuration used by all valid 1M results below.

## Valid 1M baseline after SoA and reorder

Before later arithmetic/list work:

    GPU: 14.754 ms/step
    CPU: 2994.392 ms/step
    GPU speedup: about 203x

NSYS:

    force:    10.756 ms/step, 72.7%
    density:   3.680 ms/step, 24.9%
    reorder:   0.240 ms/step,  1.6%

NCU showed:

- FP64 was the dominant pipeline, about 86.8% utilized.
- DRAM utilization was low, about 0.36%, while cache hit rates were high.
- No local-memory spilling occurred.
- The force kernel used 96 registers/thread, giving 33.3% theoretical
  occupancy; achieved occupancy was close to that bound.

This showed that shared memory, CUB scan tuning, CUDA Graphs, and launch
overhead were not the first priorities. The force arithmetic and repeated
neighbor traversal were.

## Issue 3: redundant FP64 operations in force

### Cause

Each accepted neighbor needed sqrt(r2), divisions by r and density, and
pressure, viscosity, normal, and color-laplacian polynomial terms. The source
recomputed values that are common to all four terms.

### Fix

Each neighbor iteration now computes and reuses:

    inv_r       = 1 / sqrt(r2)
    inv_density = 1 / neighbor_density
    h_minus_r   = h - r
    h2_minus_r2 = h^2 - r^2
    common      = mass * inv_density

The three force components use these values directly, and accumulations use
fma where applicable. Once r2 has passed the cutoff test, smoothing helpers
do not need to perform another cutoff test.

The direct-cell A/B mode remains available:

    ./sph_cuda --grid 100 --steps 10 --neighbor-list 0

At 1M this optimized direct mode measured 13.292 ms/step median.

## Issue 4: density and force re-enumerated neighbors

### Cause

Both density and force scanned the same 27 cell neighborhoods. Each scanned
candidate positions and discarded particles outside radius h. This was
especially wasteful after force had been identified as the dominant kernel.

### Fix: optional CSR true-neighbor list

The GPU now has:

    neighbor_counts[i]
    neighbor_offsets[i]
    neighbor_ids[]

The sequence is:

    1. count_neighbors:
       scan 27 cells and count only radius-valid neighbors.
    2. CUB ExclusiveSum:
       neighbor_counts to neighbor_offsets.
    3. density and fill:
       scan cells again, calculate density, and write accepted neighbor IDs.
    4. force:
       read only the stored true-neighbor range.

The default is neighbor-list mode. Direct mode is retained for A/B testing:

    --neighbor-list 1    CSR list, default
    --neighbor-list 0    direct cell traversal

The current implementation reserves 128 IDs per particle and records a GPU
overflow flag. Validation copies that flag and throws on overflow. This bound
is sufficient for the current uniform test, but it is not a universal
production-fluid capacity policy.

### Measured result

At 1M particles, three ten-step runs:

    direct cells plus arithmetic cleanup:
    13.297121
    13.290471
    13.291827 ms/step

    CSR neighbor list:
     7.887859
     7.904528
     7.896960 ms/step

Medians:

    direct: 13.292 ms/step
    CSR:     7.897 ms/step
    gain:    1.68x

The ten-step checksum was identical in this test:

    521422836.199754

NSYS after CSR conversion:

    density plus fill list: 3.335 ms/step
    count neighbors:        2.259 ms/step
    force from list:        1.887 ms/step
    reorder:                0.238 ms/step
    integration:            0.100 ms/step

The list construction costs time, but it reduces force from roughly 10.756 to
1.887 ms/step. The net result is strongly positive.

## End-to-end result

The meaningful progression under the same corrected 1M domain is:

    SoA plus reorder baseline:             14.754 ms/step
    direct plus arithmetic cleanup:        13.292 ms/step
    CSR list plus arithmetic cleanup:       7.897 ms/step

Therefore:

    14.754 / 7.897 = 1.87x end-to-end GPU improvement
    2994.392 / 7.897 = about 379x GPU speedup over CPU

Do not combine this 1.87x with the old small-grid 1.59x number: those were
different domain configurations.

## What is still worth doing

The solver is already in a strong state for this project. The next decisions
are no longer obvious free optimizations; each has a validation or complexity
cost.

### Highest-value experiments

1. Mixed precision or FP32. FP64 was the original force bottleneck and L40S
   has much higher FP32 throughput. Validate long-run particle positions,
   density, stability, and visual output before accepting it.

2. Amortize CSR list construction. After this optimization, list count plus
   density/fill is the main cost. A skin/Verlet-style list reused over multiple
   timesteps could help, but must rebuild before motion invalidates neighbors.

3. Stronger dynamic-capacity handling. A highly compressed fluid could exceed
   128 neighbors. A robust solver needs growth or a safe fallback without
   synchronizing the host every normal step.

4. Repeat CPU/GPU numerical validation at the corrected scalable domain.
   Define an accepted error threshold before mixed precision or temporal list
   reuse.

### Lower-priority experiments

- Block-size tests at 128, 256, and 512 threads. Register pressure may make
  lower occupancy experiments worse if spills occur.
- A cell-centric or warp-aware force mapping, after measuring per-cell
  neighbor-count variation.
- Shared memory only after proving repeated neighbor data reuse within a block.
  Current profiling does not show global bandwidth saturation.

## Reproducibility

Sync source:

    rsync -avz -e 'ssh -i ~/.ssh/id_ed25519' --exclude build --exclude .git \
      ./gpu_sph/ xw@89.169.120.233:~/gpu_sph/

Build:

    PATH=/usr/local/cuda-13.0/bin:$PATH cmake --build ~/gpu_sph/build -j$(nproc)

Run 1M default benchmark:

    cd ~/gpu_sph/build
    ./sph_cuda --grid 100 --steps 10 --neighbor-list 1

Relevant source files:

- gpu_sph/src/sph.cu
- gpu_sph/src/sph.cuh
- gpu_sph/src/main.cu
- cpu_sph/src/sph.cpp
- cpu_sph/src/main.cpp
