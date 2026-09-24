#include "common.h"

#include <algorithm>
#include <cmath>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <iostream>
#include <limits>
#include <unordered_set>
#include <vector>

#define NUM_THREADS 256

#ifndef ENABLE_REORDER
#define ENABLE_REORDER 1
#endif

#ifndef REORDER_INTERVAL
#define REORDER_INTERVAL 10
#endif

static int blks;
static int num_cells;
static int cells_per_side;
static int* particle_cell;
static int* cell_counts;
static int* cell_offsets;
static int* cell_cursors;
static int* particle_ids;
static void* d_temp_storage = nullptr;
static size_t temp_storage_bytes = 0;
static particles_soa_t reorder_buffer{};
static int step_number = 0;
static int reorder_interval = REORDER_INTERVAL;
static int analyze_reuse_at_step = -1;

__device__ int cell_index(particle_scalar_t x, particle_scalar_t y, int cells_per_side) {
    int x_bin = static_cast<int>(x / cutoff_value);
    int y_bin = static_cast<int>(y / cutoff_value);
    int clamped_x = x_bin < 0 ? 0 : (x_bin >= cells_per_side ? cells_per_side - 1 : x_bin);
    int clamped_y = y_bin < 0 ? 0 : (y_bin >= cells_per_side ? cells_per_side - 1 : y_bin);
    return clamped_y * cells_per_side + clamped_x;
}

__device__ void apply_force_gpu(particle_scalar_t x, particle_scalar_t y,
                                particle_scalar_t neighbor_x, particle_scalar_t neighbor_y,
                                particle_scalar_t& ax, particle_scalar_t& ay) {
    particle_scalar_t dx = neighbor_x - x;
    particle_scalar_t dy = neighbor_y - y;
    particle_scalar_t r2 = dx * dx + dy * dy;
    if (r2 > cutoff_value * cutoff_value)
        return;
    r2 = r2 > min_r_value * min_r_value ? r2 : min_r_value * min_r_value;
    particle_scalar_t r = sqrt(r2);
    particle_scalar_t coef = (1 - cutoff_value / r) / r2 / mass_value;
    ax += coef * dx;
    ay += coef * dy;
}

__global__ void add_cell_count(const particle_scalar_t* x, const particle_scalar_t* y,
                               int num_parts, int cells_per_side,
                               int* particle_cell, int* cell_counts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts)
        return;
    int cell_id = cell_index(x[tid], y[tid], cells_per_side);
    particle_cell[tid] = cell_id;
    atomicAdd(&cell_counts[cell_id], 1);
}

__global__ void assign_particle_slot(int num_parts, const int* particle_cell,
                                     int* cell_cursors, int* particle_ids) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts)
        return;
    int cell_id = particle_cell[tid];
    int slot = atomicAdd(&cell_cursors[cell_id], 1);
    particle_ids[slot] = tid;
}

__global__ void reorder_particles(particles_soa_t input, particles_soa_t output,
                                  const int* particle_ids, int num_parts) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot >= num_parts)
        return;
    int source = particle_ids[slot];
    output.x[slot] = input.x[source];
    output.y[slot] = input.y[source];
    output.vx[slot] = input.vx[source];
    output.vy[slot] = input.vy[source];
    output.ax[slot] = input.ax[source];
    output.ay[slot] = input.ay[source];
}

__global__ void compute_forces_gpu(particles_soa_t particles, int num_parts, int cells_per_side,
                                   const int* cell_counts,
                                   const int* cell_offsets, const int* particle_ids,
                                   bool particles_are_cell_sorted) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts)
        return;

    particle_scalar_t x = particles.x[tid];
    particle_scalar_t y = particles.y[tid];
    particle_scalar_t ax = 0.0;
    particle_scalar_t ay = 0.0;
    int cell_id = cell_index(x, y, cells_per_side);
    int cell_y = cell_id / cells_per_side;
    int cell_x = cell_id % cells_per_side;

    for (int dy = -1; dy <= 1; ++dy) {
        int neighbor_y = cell_y + dy;
        if (neighbor_y < 0 || neighbor_y >= cells_per_side)
            continue;
        for (int dx = -1; dx <= 1; ++dx) {
            int neighbor_x = cell_x + dx;
            if (neighbor_x < 0 || neighbor_x >= cells_per_side)
                continue;
            int neighbor_cell = neighbor_y * cells_per_side + neighbor_x;
            int start = cell_offsets[neighbor_cell];
            int count = cell_counts[neighbor_cell];
            for (int i = start; i < start + count; ++i) {
                int neighbor_id = particles_are_cell_sorted ? i : particle_ids[i];
                apply_force_gpu(x, y, particles.x[neighbor_id], particles.y[neighbor_id], ax, ay);
            }
        }
    }
    particles.ax[tid] = ax;
    particles.ay[tid] = ay;
}

__global__ void move_gpu(particles_soa_t particles, int num_parts, particle_scalar_t size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts)
        return;

    particle_scalar_t x = particles.x[tid];
    particle_scalar_t y = particles.y[tid];
    particle_scalar_t vx = particles.vx[tid] + particles.ax[tid] * dt_value;
    particle_scalar_t vy = particles.vy[tid] + particles.ay[tid] * dt_value;
    x += vx * dt_value;
    y += vy * dt_value;
    while (x < 0 || x > size) {
        x = x < 0 ? -x : 2 * size - x;
        vx = -vx;
    }
    while (y < 0 || y > size) {
        y = y < 0 ? -y : 2 * size - y;
        vy = -vy;
    }
    particles.x[tid] = x;
    particles.y[tid] = y;
    particles.vx[tid] = vx;
    particles.vy[tid] = vy;
}

void allocate_particles(particles_soa_t& parts, int num_parts) {
    size_t bytes = static_cast<size_t>(num_parts) * sizeof(particle_scalar_t);
    cudaMalloc(&parts.x, bytes); cudaMalloc(&parts.y, bytes);
    cudaMalloc(&parts.vx, bytes); cudaMalloc(&parts.vy, bytes);
    cudaMalloc(&parts.ax, bytes); cudaMalloc(&parts.ay, bytes);
}

void free_particles(particles_soa_t& parts) {
    cudaFree(parts.x); cudaFree(parts.y); cudaFree(parts.vx);
    cudaFree(parts.vy); cudaFree(parts.ax); cudaFree(parts.ay);
}

void swap_particles(particles_soa_t& left, particles_soa_t& right) {
    particles_soa_t temporary = left;
    left = right;
    right = temporary;
}

// Diagnostic only: compute reuse among the threads that make up each force-kernel
// CUDA block. For a sorted step, slots in [offset[cell], offset[cell] + count[cell])
// are exactly that cell's particles after reorder; otherwise particle_ids maps slots
// back to the current particle array.
void analyze_block_reuse_cpu(int num_parts, bool particles_are_cell_sorted) {
    std::vector<int> h_counts(num_cells);
    std::vector<int> h_offsets(num_cells);
    std::vector<int> h_particle_cell(num_parts);
    std::vector<int> h_particle_ids;

    cudaMemcpy(h_counts.data(), cell_counts, num_cells * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_offsets.data(), cell_offsets, num_cells * sizeof(int), cudaMemcpyDeviceToHost);
    if (!particles_are_cell_sorted) {
        h_particle_ids.resize(num_parts);
        cudaMemcpy(h_particle_cell.data(), particle_cell, num_parts * sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_particle_ids.data(), particle_ids, num_parts * sizeof(int), cudaMemcpyDeviceToHost);
    } else {
        // Reconstruct the cell of every post-reorder slot from the cell ranges.
        for (int cell = 0; cell < num_cells; ++cell)
            for (int slot = h_offsets[cell]; slot < h_offsets[cell] + h_counts[cell]; ++slot)
                h_particle_cell[slot] = cell;
    }

    unsigned long long total_requests = 0;
    unsigned long long total_unique = 0;
    int blocks_with_reuse_over_15 = 0;
    int blocks_with_reuse_over_2 = 0;
    double min_reuse = std::numeric_limits<double>::infinity();
    double max_reuse = 0.0;

    for (int block = 0; block < blks; ++block) {
        int first_tid = block * NUM_THREADS;
        int last_tid = std::min(first_tid + NUM_THREADS, num_parts);
        std::unordered_set<int> unique_neighbors;
        int block_requests = 0;

        for (int tid = first_tid; tid < last_tid; ++tid) {
            int cell_id = h_particle_cell[tid];
            int cell_y = cell_id / cells_per_side;
            int cell_x = cell_id % cells_per_side;
            for (int dy = -1; dy <= 1; ++dy) {
                int neighbor_y = cell_y + dy;
                if (neighbor_y < 0 || neighbor_y >= cells_per_side)
                    continue;
                for (int dx = -1; dx <= 1; ++dx) {
                    int neighbor_x = cell_x + dx;
                    if (neighbor_x < 0 || neighbor_x >= cells_per_side)
                        continue;
                    int neighbor_cell = neighbor_y * cells_per_side + neighbor_x;
                    int start = h_offsets[neighbor_cell];
                    int end = start + h_counts[neighbor_cell];
                    for (int slot = start; slot < end; ++slot) {
                        int neighbor_id = particles_are_cell_sorted ? slot : h_particle_ids[slot];
                        unique_neighbors.insert(neighbor_id);
                        ++block_requests;
                    }
                }
            }
        }

        int block_unique = static_cast<int>(unique_neighbors.size());
        if (block_unique == 0)
            continue;
        double reuse = static_cast<double>(block_requests) / block_unique;
        total_requests += block_requests;
        total_unique += block_unique;
        min_reuse = std::min(min_reuse, reuse);
        max_reuse = std::max(max_reuse, reuse);
        blocks_with_reuse_over_15 += reuse > 1.5;
        blocks_with_reuse_over_2 += reuse > 2.0;
    }

    std::cout << "\nBlock neighbor-reuse analysis (CPU diagnostic)\n"
              << "  step: " << step_number
              << ", layout: " << (particles_are_cell_sorted ? "post-reorder" : "indirect particle_ids") << '\n'
              << "  candidate requests: " << total_requests << '\n'
              << "  sum of per-block unique neighbor IDs: " << total_unique << '\n'
              << "  weighted reuse factor: "
              << (total_unique ? static_cast<double>(total_requests) / total_unique : 0.0) << '\n'
              << "  per-block reuse range: [" << min_reuse << ", " << max_reuse << "]\n"
              << "  blocks with reuse > 1.5: " << blocks_with_reuse_over_15 << " / " << blks << '\n'
              << "  blocks with reuse > 2.0: " << blocks_with_reuse_over_2 << " / " << blks << "\n\n";
}

void init_simulation(int num_parts, double size, int requested_reorder_interval, int analyze_reuse_step) {
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    step_number = 0;
    reorder_interval = requested_reorder_interval;
    analyze_reuse_at_step = analyze_reuse_step;
    cells_per_side = static_cast<int>(ceil(size / cutoff));
    num_cells = cells_per_side * cells_per_side;
    cudaMalloc(&particle_cell, num_parts * sizeof(int));
    cudaMalloc(&cell_counts, num_cells * sizeof(int));
    cudaMalloc(&cell_offsets, num_cells * sizeof(int));
    cudaMalloc(&cell_cursors, num_cells * sizeof(int));
    cudaMalloc(&particle_ids, num_parts * sizeof(int));
#if ENABLE_REORDER
    if (reorder_interval > 0)
        allocate_particles(reorder_buffer, num_parts);
#endif
    cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, cell_counts, cell_offsets, num_cells);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
}

void simulate_one_step(particles_soa_t& parts, int num_parts, double size) {
    cudaMemset(cell_counts, 0, num_cells * sizeof(int));
    add_cell_count<<<blks, NUM_THREADS>>>(parts.x, parts.y, num_parts, cells_per_side,
                                           particle_cell, cell_counts);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, cell_counts, cell_offsets, num_cells);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, cell_counts, cell_cursors, num_cells);
    assign_particle_slot<<<blks, NUM_THREADS>>>(num_parts, particle_cell, cell_cursors, particle_ids);
    bool particles_are_cell_sorted = ENABLE_REORDER && reorder_interval > 0 &&
                                    (step_number % reorder_interval == 0);
#if ENABLE_REORDER
    if (particles_are_cell_sorted) {
        reorder_particles<<<blks, NUM_THREADS>>>(parts, reorder_buffer, particle_ids, num_parts);
        swap_particles(parts, reorder_buffer);
    }
#endif
    // Run this only after the data have actually been written in cell-major order.
    // The CPU copies below synchronize with reorder_particles before inspecting it.
    if (step_number == analyze_reuse_at_step) {
        analyze_block_reuse_cpu(num_parts, particles_are_cell_sorted);
        analyze_reuse_at_step = -1;
    }
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, cells_per_side, cell_counts,
                                               cell_offsets, particle_ids, particles_are_cell_sorted);
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, static_cast<particle_scalar_t>(size));
    ++step_number;
}

void clean_simulation() {
    cudaFree(particle_cell);
    cudaFree(cell_counts);
    cudaFree(cell_offsets);
    cudaFree(cell_cursors);
    cudaFree(particle_ids);
    cudaFree(d_temp_storage);
#if ENABLE_REORDER
    if (reorder_interval > 0)
        free_particles(reorder_buffer);
#endif
}
