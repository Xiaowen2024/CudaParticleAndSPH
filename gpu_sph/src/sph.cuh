#pragma once

#include <cstddef>
#include <vector>

struct Vec3 { double x, y, z; };

// Host-side particle representation used only for initialization and reporting.
struct Particle {
    Vec3 position;
    Vec3 velocity;
    Vec3 force;
    double density;
    double pressure;
};

// Device-resident structure-of-arrays state.  A kernel that only needs positions
// now loads x/y/z instead of an entire Particle record (88 bytes in the AoS form).
// All fields use the same particle index and must be reordered together.
struct ParticlesSoA {
    double* x = nullptr;
    double* y = nullptr;
    double* z = nullptr;
    double* vx = nullptr;
    double* vy = nullptr;
    double* vz = nullptr;
    double* fx = nullptr;
    double* fy = nullptr;
    double* fz = nullptr;
    double* density = nullptr;
    double* pressure = nullptr;
};

struct Parameters {
    int grid_length = 17;
    int steps = 100;
    double dt = 1.0 / 360.0;
    double box_x = 1.5, box_y = 0.5, box_z = 0.5;
    double smoothing_length = 0.0726;
    double rest_density = 998.29;
    double mass = 0.04;
    double pressure_stiffness = 5.0;
    double viscosity = 36.0;
    double surface_tension = 6.0;
    double surface_threshold = 5.0;
    double gravity = 10.0;
    double wall_reflection = -0.5;
    bool use_neighbor_list = true;

    // Precomputed once on the host; copied to kernels as part of Parameters.
    double poly6_coefficient = 0.0;
    double spiky_gradient_coefficient = 0.0;
    double viscosity_laplacian_coefficient = 0.0;
    double poly6_derivative_coefficient = 0.0;
};

// Host-side owner. Its pointers refer to device global memory.
struct SphGpuSimulation {
    ParticlesSoA gpu_particles_{};
    ParticlesSoA reordered_particles_{};
    int* cell_counts_ = nullptr;
    int* cell_cursors_ = nullptr;
    int* cell_offsets_ = nullptr;
    int* particle_cell_ = nullptr;
    int* particle_ids_ = nullptr;
    int* neighbor_counts_ = nullptr;
    int* neighbor_offsets_ = nullptr;
    int* neighbor_ids_ = nullptr;
    int* neighbor_overflow_ = nullptr;
    int neighbor_capacity_ = 0;
    void* scan_temp_storage_ = nullptr;
    std::size_t scan_temp_storage_bytes_ = 0;

    int num_particles_ = 0;
    int cells_x_ = 0, cells_y_ = 0, cells_z_ = 0;
    int num_cells_ = 0;
    int blocks_ = 0;
    Parameters parameters_{};

    SphGpuSimulation() = default;
    SphGpuSimulation(const SphGpuSimulation&) = delete;
    SphGpuSimulation& operator=(const SphGpuSimulation&) = delete;

    void initialize(const std::vector<Particle>& host_particles, Parameters parameters);
    void run();
    void step();
    void rebuild_cells();
    void build_neighbor_list();
    double run_timed();
    double checksum() const;
    void validate_cell_list() const;
    void cleanup();
};

std::vector<Particle> make_initial_particles(const Parameters& parameters);
// Keep the initial lattice inside the simulation domain when --grid changes.
void fit_domain_to_initial_lattice(Parameters& parameters);
