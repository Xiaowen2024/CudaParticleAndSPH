#include "common_aos.h"

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

void save(std::ofstream& output, const particle_t* parts, int num_parts, double size) {
    static bool first = true;
    if (first) {
        output << num_parts << " " << size << std::endl;
        first = false;
    }
    for (int i = 0; i < num_parts; ++i)
        output << parts[i].x << " " << parts[i].y << std::endl;
    output << std::endl;
}

void init_particles(particle_t* parts, int num_parts, double size, int part_seed) {
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
        parts[i].x = size * (1. + (k % sx)) / (1 + sx);
        parts[i].y = size * (1. + (k / sx)) / (1 + sy);
        std::uniform_real_distribution<double> rand_real(-1.0, 1.0);
        parts[i].vx = rand_real(gen);
        parts[i].vy = rand_real(gen);
        parts[i].ax = parts[i].ay = 0.0;
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

char* find_string_option(int argc, char** argv, const char* option, char* default_value) {
    int index = find_arg_idx(argc, argv, option);
    return (index >= 0 && index < argc - 1) ? argv[index + 1] : default_value;
}

int main(int argc, char** argv) {
    char* savename = find_string_option(argc, argv, "-o", nullptr);
    std::ofstream output(savename);
    int num_parts = find_int_arg(argc, argv, "-n", 1000);
    int part_seed = find_int_arg(argc, argv, "-s", 0);
    double size = sqrt(density * num_parts);

    particle_t* host_parts = new particle_t[num_parts];
    init_particles(host_parts, num_parts, size, part_seed);
    particle_t* device_parts;
    cudaMalloc(&device_parts, static_cast<size_t>(num_parts) * sizeof(particle_t));
    cudaMemcpy(device_parts, host_parts, static_cast<size_t>(num_parts) * sizeof(particle_t), cudaMemcpyHostToDevice);

    auto start = std::chrono::steady_clock::now();
    int reorder_interval = find_int_arg(argc, argv, "-r", REORDER_INTERVAL);
    init_simulation(num_parts, size, reorder_interval, false);
    for (int step = 0; step < nsteps; ++step) {
        simulate_one_step(device_parts, num_parts, size);
        if (output.good() && step % savefreq == 0) {
            cudaMemcpy(host_parts, device_parts, static_cast<size_t>(num_parts) * sizeof(particle_t), cudaMemcpyDeviceToHost);
            save(output, host_parts, num_parts, size);
        }
    }
    cudaDeviceSynchronize();
    clean_simulation();
    auto finish = std::chrono::steady_clock::now();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "cudaGetLastError failed: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }
    std::chrono::duration<double> elapsed = finish - start;
    std::cout << "Simulation Time = " << elapsed.count() << " seconds for " << num_parts << " particles.\n";
    cudaFree(device_parts);
    delete[] host_parts;
}
