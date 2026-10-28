#include "sph.h"

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
    Parameters parameters;
    for (int i = 1; i < argc; ++i) {
        const std::string_view option(argv[i]);
        if (option == "--grid" && i + 1 < argc) parameters.grid_length = std::atoi(argv[++i]);
        else if (option == "--steps" && i + 1 < argc) parameters.steps = std::atoi(argv[++i]);
        else if (option == "-h" || option == "--help") {
            std::cout << "Usage: sph_cpu [--grid N] [--steps N]\n";
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
    fit_domain_to_initial_lattice(parameters);
    SphSimulation simulation(parameters);
    const auto start = std::chrono::steady_clock::now();
    simulation.run();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::cout << std::fixed << std::setprecision(6)
              << "particles=" << simulation.particle_count()
              << " steps=" << parameters.steps
              << " box=" << parameters.box_x << 'x' << parameters.box_y << 'x' << parameters.box_z
              << " total_seconds=" << seconds
              << " milliseconds_per_step=" << 1000.0 * seconds / parameters.steps
              << " checksum=" << simulation.checksum() << '\n';
}
