# SPH CPU baseline

This is the headless CPU baseline extracted from the original water-simulation project.
GUI/rendering infrastructure (NanoGUI, OpenGL, shaders, textures, camera, mesh drawing, scene loading, and committed build artifacts) is intentionally excluded.

The remaining executable models the original SPH stages: spatial-cell construction, neighbor discovery, density/pressure, force computation, and integration with box boundaries.

Build and benchmark:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/sph_cpu --grid 17 --steps 100
```

`--grid N` creates `N^3` particles. This baseline is deliberately CPU-only and uses an AoS particle layout plus an `unordered_map` cell table, so it is a correctness and performance reference for a later GPU rewrite.
