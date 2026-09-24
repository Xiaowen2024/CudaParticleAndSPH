#include "common_aos.h"

#include <cmath>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

#define NUM_THREADS 256

#ifndef ENABLE_REORDER
#define ENABLE_REORDER 0
#endif

static int blks;
static int num_cells;
static int cells_per_side;
static int* particle_cell;
static int* cell_counts;
static int* cell_offsets;
static int* cell_cursors;
static int* particle_ids;
static particle_t* reorder_buffer;
static void* d_temp_storage = nullptr;
static size_t temp_storage_bytes = 0;
static int step_number = 0;
static int reorder_interval = 10;

__device__ int cell_index(double x, double y, int cells_per_side) {
    int x_bin = static_cast<int>(x / cutoff);
    int y_bin = static_cast<int>(y / cutoff);
    int clamped_x = x_bin < 0 ? 0 : (x_bin >= cells_per_side ? cells_per_side - 1 : x_bin);
    int clamped_y = y_bin < 0 ? 0 : (y_bin >= cells_per_side ? cells_per_side - 1 : y_bin);
    return clamped_y * cells_per_side + clamped_x;
}

__device__ void apply_force_gpu(const particle_t& particle, const particle_t& neighbor,
                                double& ax, double& ay) {
    double dx = neighbor.x - particle.x;
    double dy = neighbor.y - particle.y;
    double r2 = dx * dx + dy * dy;
    if (r2 > cutoff * cutoff)
        return;
    r2 = r2 > min_r * min_r ? r2 : min_r * min_r;
    double r = sqrt(r2);
    double coef = (1 - cutoff / r) / r2 / mass;
    ax += coef * dx;
    ay += coef * dy;
}

__global__ void add_cell_count(const particle_t* particles, int num_parts, int cells_per_side,
                               int* particle_cell, int* cell_counts) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts)
        return;
    int cell_id = cell_index(particles[tid].x, particles[tid].y, cells_per_side);
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

__global__ void reorder_particles(const particle_t* input, particle_t* output,
                                  const int* particle_ids, int num_parts) {
    int slot = blockIdx.x * blockDim.x + threadIdx.x;
    if (slot < num_parts)
        output[slot] = input[particle_ids[slot]];
}

__global__ void compute_forces_gpu(particle_t* particles, int num_parts, int cells_per_side,
                                   const int* cell_counts, const int* cell_offsets,
                                   const int* particle_ids, bool particles_are_cell_sorted) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts)
        return;
    particle_t particle = particles[tid];
    double ax = 0.0;
    double ay = 0.0;
    int cell_id = cell_index(particle.x, particle.y, cells_per_side);
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
            int cell = neighbor_y * cells_per_side + neighbor_x;
            int start = cell_offsets[cell];
            int count = cell_counts[cell];
            for (int i = start; i < start + count; ++i) {
                int neighbor_id = particles_are_cell_sorted ? i : particle_ids[i];
                apply_force_gpu(particle, particles[neighbor_id], ax, ay);
            }
        }
    }
    particles[tid].ax = ax;
    particles[tid].ay = ay;
}

__global__ void move_gpu(particle_t* particles, int num_parts, double size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_parts)
        return;
    particle_t particle = particles[tid];
    particle.vx += particle.ax * dt;
    particle.vy += particle.ay * dt;
    particle.x += particle.vx * dt;
    particle.y += particle.vy * dt;
    while (particle.x < 0 || particle.x > size) {
        particle.x = particle.x < 0 ? -particle.x : 2 * size - particle.x;
        particle.vx = -particle.vx;
    }
    while (particle.y < 0 || particle.y > size) {
        particle.y = particle.y < 0 ? -particle.y : 2 * size - particle.y;
        particle.vy = -particle.vy;
    }
    particles[tid] = particle;
}

void init_simulation(int num_parts, double size, int requested_reorder_interval, bool analyze_reuse) {
    blks = (num_parts + NUM_THREADS - 1) / NUM_THREADS;
    step_number = 0;
    reorder_interval = requested_reorder_interval;
    cells_per_side = static_cast<int>(ceil(size / cutoff));
    num_cells = cells_per_side * cells_per_side;
    cudaMalloc(&particle_cell, num_parts * sizeof(int));
    cudaMalloc(&cell_counts, num_cells * sizeof(int));
    cudaMalloc(&cell_offsets, num_cells * sizeof(int));
    cudaMalloc(&cell_cursors, num_cells * sizeof(int));
    cudaMalloc(&particle_ids, num_parts * sizeof(int));
#if ENABLE_REORDER
    if (reorder_interval > 0)
        cudaMalloc(&reorder_buffer, num_parts * sizeof(particle_t));
#endif
    cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, cell_counts, cell_offsets, num_cells);
    cudaMalloc(&d_temp_storage, temp_storage_bytes);
}

void simulate_one_step(particle_t*& parts, int num_parts, double size) {
    cudaMemset(cell_counts, 0, num_cells * sizeof(int));
    add_cell_count<<<blks, NUM_THREADS>>>(parts, num_parts, cells_per_side, particle_cell, cell_counts);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, cell_counts, cell_offsets, num_cells);
    cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, cell_counts, cell_cursors, num_cells);
    assign_particle_slot<<<blks, NUM_THREADS>>>(num_parts, particle_cell, cell_cursors, particle_ids);
    bool particles_are_cell_sorted = ENABLE_REORDER && reorder_interval > 0 &&
                                    (step_number % reorder_interval == 0);
#if ENABLE_REORDER
    if (particles_are_cell_sorted) {
        reorder_particles<<<blks, NUM_THREADS>>>(parts, reorder_buffer, particle_ids, num_parts);
        particle_t* temporary = parts;
        parts = reorder_buffer;
        reorder_buffer = temporary;
    }
#endif
    compute_forces_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, cells_per_side, cell_counts, cell_offsets,
                                               particle_ids, particles_are_cell_sorted);
    move_gpu<<<blks, NUM_THREADS>>>(parts, num_parts, size);
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
        cudaFree(reorder_buffer);
#endif
}
