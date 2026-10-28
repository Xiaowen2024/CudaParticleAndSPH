# CPU baseline benchmark

## Scope

This benchmark measures only the SPH simulation pipeline:

1. rebuild the spatial hash grid;
2. discover neighbors in the 27 surrounding cells;
3. compute density and pressure;
4. compute pressure, viscosity, surface-tension, and gravity forces;
5. integrate velocity and position, then apply box boundaries.

It deliberately excludes NanoGUI, OpenGL, shader work, texture loading, mesh drawing,
and all display/event-loop time. The result is the CPU reference for a future CUDA
implementation.

## Build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Machine and build configuration

- Host: local macOS development machine
- Compiler: AppleClang 15.0.0
- Build type: Release (`-O3`)
- Numeric type: `double`
- Layout: AoS (`std::vector<Particle>`)
- Neighbor structure: `std::unordered_map` spatial hash
- Simulation parameters: the original project's liquid scene defaults, including
  smoothing length `h = 0.0726`, mass `0.04`, and rest density `998.29`.

## Results

### Original default scene size

Command:

```bash
./build/sph_cpu --grid 17 --steps 100
```

Particle count: `17^3 = 4913`.

Three runs produced:

```text
total_seconds    milliseconds_per_step
1.768559         17.685586
1.767613         17.676130
2.270877         22.708769
```

The first two runs are stable at about **17.68 ms/step**. The third run is a host-side
outlier, so the practical reference for this machine is **approximately 17.7 ms/step**
or **1.77 seconds per 100 steps**.

### Larger input

Command:

```bash
./build/sph_cpu --grid 20 --steps 50
```

Particle count: `20^3 = 8000`.

```text
total_seconds=1.677933
milliseconds_per_step=33.558663
```

This is **33.56 ms/step**.

## Interpretation

Runtime grows faster than linearly with particle count because each step performs a
variable number of neighbor interactions in addition to rebuilding the hash table.
For CUDA comparisons, record the same grid size, timestep count, precision, and physical
parameters; compare simulation-only time rather than rendering-inclusive wall time.
