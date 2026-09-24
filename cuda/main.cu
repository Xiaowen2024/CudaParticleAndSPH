#include "common.h"

#include <chrono>
#include <cmath>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#ifndef REORDER_INTERVAL
#define REORDER_INTERVAL 10
#endif

void save(std::ofstream& fsave, const particles_soa_t& parts, int num_parts, double size) {
    static bool first = true;
    if (first) {
        fsave << num_parts << " " << size << std::endl;
        first = false;
    }
    for (int i = 0; i < num_parts; ++i)
        fsave << parts.x[i] << " " << parts.y[i] << std::endl;
    fsave << std::endl;
}

void init_particles(particles_soa_t parts, int num_parts, double size, int part_seed) {
    std::random_device rd;
    std::mt19937 gen(part_seed ? part_seed : rd());
    int sx = static_cast<int>(ceil(sqrt(static_cast<double>(num_parts))));
    int sy = (num_parts + sx - 1) / sx;

    std::vector<int> shuffle(num_parts);
    for (int i = 0; i < num_parts; ++i)
        shuffle[i] = i;

    for (int i = 0; i < num_parts; ++i) {
        std::uniform_int_distribution<int> rand_int(0, num_parts - i - 1);
        int j = rand_int(gen);
        int k = shuffle[j];
        shuffle[j] = shuffle[num_parts - i - 1];
        parts.x[i] = size * (1. + (k % sx)) / (1 + sx);
        parts.y[i] = size * (1. + (k / sx)) / (1 + sy);
        std::uniform_real_distribution<particle_scalar_t> rand_real(-1.0, 1.0);
        parts.vx[i] = rand_real(gen);
        parts.vy[i] = rand_real(gen);
        parts.ax[i] = 0.0;
        parts.ay[i] = 0.0;
    }
}

int find_arg_idx(int argc, char** argv, const char* option) {
    for (int i = 1; i < argc; ++i)
        if (strcmp(argv[i], option) == 0)
            return i;
    return -1;
}

int find_int_arg(int argc, char** argv, const char* option, int default_value) {
    int index = find_arg_idx(argc, argv, option);
    return (index >= 0 && index < argc - 1) ? std::stoi(argv[index + 1]) : default_value;
}

double find_double_arg(int argc, char** argv, const char* option, double default_value) {
    int index = find_arg_idx(argc, argv, option);
    return (index >= 0 && index < argc - 1) ? std::stod(argv[index + 1]) : default_value;
}

char* find_string_option(int argc, char** argv, const char* option, char* default_value) {
    int index = find_arg_idx(argc, argv, option);
    return (index >= 0 && index < argc - 1) ? argv[index + 1] : default_value;
}

void allocate_host_particles(particles_soa_t& parts, int num_parts) {
    parts.x = new particle_scalar_t[num_parts]; parts.y = new particle_scalar_t[num_parts];
    parts.vx = new particle_scalar_t[num_parts]; parts.vy = new particle_scalar_t[num_parts];
    parts.ax = new particle_scalar_t[num_parts]; parts.ay = new particle_scalar_t[num_parts];
}

void free_host_particles(particles_soa_t& parts) {
    delete[] parts.x; delete[] parts.y; delete[] parts.vx;
    delete[] parts.vy; delete[] parts.ax; delete[] parts.ay;
}

void allocate_device_particles(particles_soa_t& parts, int num_parts) {
    size_t bytes = static_cast<size_t>(num_parts) * sizeof(particle_scalar_t);
    cudaMalloc(&parts.x, bytes); cudaMalloc(&parts.y, bytes);
    cudaMalloc(&parts.vx, bytes); cudaMalloc(&parts.vy, bytes);
    cudaMalloc(&parts.ax, bytes); cudaMalloc(&parts.ay, bytes);
}

void free_device_particles(particles_soa_t& parts) {
    cudaFree(parts.x); cudaFree(parts.y); cudaFree(parts.vx);
    cudaFree(parts.vy); cudaFree(parts.ax); cudaFree(parts.ay);
}

void copy_host_to_device(const particles_soa_t& host, const particles_soa_t& device, int num_parts) {
    size_t bytes = static_cast<size_t>(num_parts) * sizeof(particle_scalar_t);
    cudaMemcpy(device.x, host.x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device.y, host.y, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device.vx, host.vx, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device.vy, host.vy, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device.ax, host.ax, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(device.ay, host.ay, bytes, cudaMemcpyHostToDevice);
}

void copy_device_to_host(const particles_soa_t& device, const particles_soa_t& host, int num_parts) {
    size_t bytes = static_cast<size_t>(num_parts) * sizeof(particle_scalar_t);
    cudaMemcpy(host.x, device.x, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(host.y, device.y, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(host.vx, device.vx, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(host.vy, device.vy, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(host.ax, device.ax, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(host.ay, device.ay, bytes, cudaMemcpyDeviceToHost);
}

int main(int argc, char** argv) {
    if (find_arg_idx(argc, argv, "-h") >= 0) {
        std::cout << "Options:\n-h: see this help\n-n <int>: set number of particles\n"
                  << "-o <filename>: set the output file name\n-s <int>: set particle initialization seed\n"
                  << "-d <double>: area per particle (default " << density << ")\n"
                  << "-r <int>: reorder interval; 0 disables reordering\n"
                  << "--analyze-reuse: analyze CPU block-level neighbor reuse at step 0\n"
                  << "--analyze-reuse-step <int>: analyze it at the specified simulation step\n";
        return 0;
    }

    char* savename = find_string_option(argc, argv, "-o", nullptr);
    std::ofstream fsave(savename);
    int num_parts = find_int_arg(argc, argv, "-n", 1000);
    int part_seed = find_int_arg(argc, argv, "-s", 0);
    double area_per_particle = find_double_arg(argc, argv, "-d", density);
    double size = sqrt(area_per_particle * num_parts);

    particles_soa_t host_parts{};
    particles_soa_t device_parts{};
    allocate_host_particles(host_parts, num_parts);
    init_particles(host_parts, num_parts, size, part_seed);
    allocate_device_particles(device_parts, num_parts);
    copy_host_to_device(host_parts, device_parts, num_parts);

    auto start_time = std::chrono::steady_clock::now();
    int reorder_interval = find_int_arg(argc, argv, "-r", REORDER_INTERVAL);
    int analyze_reuse_step = find_arg_idx(argc, argv, "--analyze-reuse") >= 0
                                 ? 0
                                 : find_int_arg(argc, argv, "--analyze-reuse-step", -1);
    init_simulation(num_parts, size, reorder_interval, analyze_reuse_step);
    for (int step = 0; step < nsteps; ++step) {
        simulate_one_step(device_parts, num_parts, size);
        if (fsave.good() && (step % savefreq) == 0) {
            copy_device_to_host(device_parts, host_parts, num_parts);
            save(fsave, host_parts, num_parts, size);
        }
    }
    cudaDeviceSynchronize();
    clean_simulation();
    auto end_time = std::chrono::steady_clock::now();

    cudaError_t error_code = cudaGetLastError();
    if (error_code != cudaSuccess) {
        std::cerr << "cudaGetLastError failed: " << cudaGetErrorString(error_code) << std::endl;
        return 1;
    }
    std::chrono::duration<double> elapsed = end_time - start_time;
    std::cout << "Simulation Time = " << elapsed.count() << " seconds for " << num_parts << " particles.\n";
    free_device_particles(device_parts);
    free_host_particles(host_parts);
}
