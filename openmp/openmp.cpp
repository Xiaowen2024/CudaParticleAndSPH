#include "common.h"

#include <algorithm>
#include <cmath>
#include <omp.h>
#include <vector>

// The domain is divided into cells whose side length is cutoff. A particle
// can only interact with particles in its own cell or one of its 8 neighbors.
static int cells_per_side;
static std::vector<std::vector<int>> cells;

static int cell_index(double coordinate) {
    int cell = static_cast<int>(coordinate / cutoff);
    // A particle on the upper boundary belongs to the last valid cell.
    return std::max(0, std::min(cell, cells_per_side - 1));
}

static void apply_force(particle_t& particle, const particle_t& neighbor) {
    const double dx = neighbor.x - particle.x;
    const double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;

    if (r2 > cutoff * cutoff)
        return;

    r2 = std::fmax(r2, min_r * min_r);
    const double r = std::sqrt(r2);
    const double coef = (1 - cutoff / r) / r2 / mass;
    particle.ax += coef * dx;
    particle.ay += coef * dy;
}

static void move(particle_t& p, double size) {
    p.vx += p.ax * dt;
    p.vy += p.ay * dt;
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    while (p.x < 0 || p.x > size) {
        p.x = p.x < 0 ? -p.x : 2 * size - p.x;
        p.vx = -p.vx;
    }
    while (p.y < 0 || p.y > size) {
        p.y = p.y < 0 ? -p.y : 2 * size - p.y;
        p.vy = -p.vy;
    }
}

void init_simulation(particle_t* /*parts*/, int /*num_parts*/, double size) {
    cells_per_side = static_cast<int>(std::ceil(size / cutoff));
    cells.resize(cells_per_side * cells_per_side);
}

void simulate_one_step(particle_t* parts, int num_parts, double size) {
    // Rebuild the spatial index after positions change. This is serial because
    // vector::clear and vector::push_back on the same cell are not thread-safe.
    #pragma omp single
    {
        for (auto& cell : cells)
            cell.clear();
        for (int i = 0; i < num_parts; ++i) {
            const int x = cell_index(parts[i].x);
            const int y = cell_index(parts[i].y);
            cells[y * cells_per_side + x].push_back(i);
        }
    }

    #pragma omp for schedule(static)
    for (int i = 0; i < num_parts; ++i) {
        particle_t& particle = parts[i];
        particle.ax = particle.ay = 0;

        const int x = cell_index(particle.x);
        const int y = cell_index(particle.y);
        for (int dy = -1; dy <= 1; ++dy) {
            const int neighbor_y = y + dy;
            if (neighbor_y < 0 || neighbor_y >= cells_per_side)
                continue;
            for (int dx = -1; dx <= 1; ++dx) {
                const int neighbor_x = x + dx;
                if (neighbor_x < 0 || neighbor_x >= cells_per_side)
                    continue;

                const auto& cell = cells[neighbor_y * cells_per_side + neighbor_x];
                for (int j : cell)
                    apply_force(particle, parts[j]);
            }
        }
    }

    #pragma omp for schedule(static)
    for (int i = 0; i < num_parts; ++i)
        move(parts[i], size);
}
