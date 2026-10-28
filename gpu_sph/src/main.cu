#include "sph.cuh"

#include <cstdlib>
#include <exception>
#include <iomanip>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    Parameters parameters;
    for (int i = 1; i < argc; ++i) {
        const std::string_view option(argv[i]);
        if (option == "--grid" && i + 1 < argc) parameters.grid_length = std::atoi(argv[++i]);
        else if (option == "--steps" && i + 1 < argc) parameters.steps = std::atoi(argv[++i]);
        else if (option == "--neighbor-list" && i + 1 < argc) parameters.use_neighbor_list = std::atoi(argv[++i]) != 0;
        else if (option == "-h" || option == "--help") {
            std::cout << "Usage: sph_cuda [--grid N] [--steps N] [--neighbor-list 0|1]\n";
            return 0;
        } else {
            std::cerr << "Unknown or incomplete option: " << option << '\n';
            return 1;
        }
    }
    if (parameters.grid_length <= 0 || parameters.steps <= 0) {
        std::cerr << "--grid and --steps must be positive.\n";
        return 1;
    }
    try {
        fit_domain_to_initial_lattice(parameters);
        SphGpuSimulation simulation;
        simulation.initialize(make_initial_particles(parameters), parameters);
        const double seconds = simulation.run_timed();
        simulation.validate_cell_list();
        std::cout << std::fixed << std::setprecision(6)
                  << "particles=" << simulation.num_particles_
                  << " steps=" << parameters.steps
                  << " box=" << parameters.box_x << 'x' << parameters.box_y << 'x' << parameters.box_z
                  << " neighbor_list=" << parameters.use_neighbor_list
                  << " gpu_total_seconds=" << seconds
                  << " gpu_milliseconds_per_step=" << 1000.0 * seconds / parameters.steps
                  << " checksum=" << simulation.checksum() << '\n';
        simulation.cleanup();
    } catch (const std::exception& error) {
        std::cerr << "CUDA setup failed: " << error.what() << '\n';
        return 1;
    }
}
