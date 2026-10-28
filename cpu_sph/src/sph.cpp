#include "sph.h"

#include <algorithm>
#include <cmath>

namespace {
constexpr double kPi = 3.14159265358979323846;
}

Vec3& Vec3::operator+=(const Vec3& other) { x += other.x; y += other.y; z += other.z; return *this; }
Vec3& Vec3::operator-=(const Vec3& other) { x -= other.x; y -= other.y; z -= other.z; return *this; }
Vec3 operator+(Vec3 a, const Vec3& b) { return a += b; }
Vec3 operator-(Vec3 a, const Vec3& b) { return a -= b; }
Vec3 operator*(double s, Vec3 v) { return {s * v.x, s * v.y, s * v.z}; }
Vec3 operator*(Vec3 v, double s) { return s * v; }
Vec3 operator/(Vec3 v, double s) { return (1.0 / s) * v; }
double norm_squared(const Vec3& v) { return v.x * v.x + v.y * v.y + v.z * v.z; }
double norm(const Vec3& v) { return std::sqrt(norm_squared(v)); }

void fit_domain_to_initial_lattice(Parameters& parameters) {
    const double radius = std::cbrt(3.0 * parameters.mass / (4.0 * kPi * parameters.rest_density));
    const double lattice_extent = 2.0 * parameters.grid_length * radius;
    parameters.box_x = std::max(parameters.box_x, lattice_extent);
    parameters.box_y = std::max(parameters.box_y, lattice_extent);
    parameters.box_z = std::max(parameters.box_z, lattice_extent);
}

SphSimulation::SphSimulation(Parameters parameters) : p_(parameters) { initialize_particles(); }

void SphSimulation::initialize_particles() {
    particles_.reserve(static_cast<std::size_t>(p_.grid_length) * p_.grid_length * p_.grid_length);
    const double radius = std::cbrt(3.0 * p_.mass / (4.0 * kPi * p_.rest_density));
    const double spacing = 2.0 * radius;
    for (int y = 0; y < p_.grid_length; ++y)
        for (int x = 0; x < p_.grid_length; ++x)
            for (int z = 0; z < p_.grid_length; ++z)
                particles_.push_back({{x * spacing + radius, y * spacing + radius, z * spacing + radius}, {}, {}, 0.0, 0.0});
}

std::int64_t SphSimulation::cell_hash(const Vec3& position) const {
    const auto x = static_cast<std::int64_t>(std::floor(position.x / p_.smoothing_length));
    const auto y = static_cast<std::int64_t>(std::floor(position.y / p_.smoothing_length));
    const auto z = static_cast<std::int64_t>(std::floor(position.z / p_.smoothing_length));
    return 73856093LL * x ^ 19349663LL * y ^ 83492791LL * z;
}

void SphSimulation::rebuild_cells() {
    cells_.clear();
    cells_.reserve(particles_.size());
    for (int i = 0; i < static_cast<int>(particles_.size()); ++i)
        cells_[cell_hash(particles_[i].position)].push_back(i);
}

double SphSimulation::poly6(double r2) const {
    const double h2 = p_.smoothing_length * p_.smoothing_length;
    if (r2 > h2) return 0.0;
    const double t = h2 - r2;
    return 315.0 / (64.0 * kPi * std::pow(p_.smoothing_length, 9)) * t * t * t;
}
double SphSimulation::spiky_gradient(double r) const {
    if (r > p_.smoothing_length) return 0.0;
    const double t = p_.smoothing_length - r;
    return -45.0 / (kPi * std::pow(p_.smoothing_length, 6)) * t * t;
}
double SphSimulation::viscosity_laplacian(double r) const {
    if (r > p_.smoothing_length) return 0.0;
    return 45.0 / (kPi * std::pow(p_.smoothing_length, 6)) * (p_.smoothing_length - r);
}
double SphSimulation::poly6_gradient(double r) const {
    if (r > p_.smoothing_length) return 0.0;
    const double h2 = p_.smoothing_length * p_.smoothing_length;
    return -945.0 / (32.0 * kPi * std::pow(p_.smoothing_length, 9)) * (h2 - r * r) * (h2 - r * r);
}
double SphSimulation::poly6_laplacian(double r) const {
    if (r > p_.smoothing_length) return 0.0;
    const double h2 = p_.smoothing_length * p_.smoothing_length;
    return -945.0 / (32.0 * kPi * std::pow(p_.smoothing_length, 9)) * (h2 - r * r) * (3.0 * h2 - 7.0 * r * r);
}

void SphSimulation::step() {
    rebuild_cells();
    const int n = static_cast<int>(particles_.size());
    std::vector<std::vector<int>> neighbors(n);
    for (int i = 0; i < n; ++i) {
        const Vec3& pos = particles_[i].position;
        const int cx = static_cast<int>(std::floor(pos.x / p_.smoothing_length));
        const int cy = static_cast<int>(std::floor(pos.y / p_.smoothing_length));
        const int cz = static_cast<int>(std::floor(pos.z / p_.smoothing_length));
        for (int dx = -1; dx <= 1; ++dx) for (int dy = -1; dy <= 1; ++dy) for (int dz = -1; dz <= 1; ++dz) {
            const std::int64_t key = 73856093LL * (cx + dx) ^ 19349663LL * (cy + dy) ^ 83492791LL * (cz + dz);
            const auto found = cells_.find(key);
            if (found == cells_.end()) continue;
            for (int j : found->second)
                if (norm_squared(particles_[j].position - pos) <= p_.smoothing_length * p_.smoothing_length)
                    neighbors[i].push_back(j);
        }
    }

    for (int i = 0; i < n; ++i) {
        Particle& particle = particles_[i];
        particle.density = 0.0;
        for (int j : neighbors[i]) particle.density += p_.mass * poly6(norm_squared(particles_[j].position - particle.position));
        particle.density = std::max(particle.density, 1e-12);
        particle.pressure = p_.pressure_stiffness * (particle.density - p_.rest_density);
    }

    for (int i = 0; i < n; ++i) {
        Particle& particle = particles_[i];
        Vec3 pressure_force, viscosity_force, normal;
        double color_laplacian = 0.0;
        for (int j : neighbors[i]) {
            if (i == j) continue;
            const Particle& neighbor = particles_[j];
            const Vec3 delta = particle.position - neighbor.position;
            const double r = norm(delta);
            if (r <= 1e-12) continue;
            pressure_force += (-p_.mass * (particle.pressure + neighbor.pressure) / (2.0 * neighbor.density) * spiky_gradient(r) / r) * delta;
            viscosity_force += (p_.viscosity * p_.mass / neighbor.density * viscosity_laplacian(r)) * (neighbor.velocity - particle.velocity);
            normal += (p_.mass / neighbor.density * poly6_gradient(r) / r) * delta;
            color_laplacian += p_.mass / neighbor.density * poly6_laplacian(r);
        }
        const double normal_length = norm(normal);
        const Vec3 surface_force = normal_length >= p_.surface_threshold ? (-p_.surface_tension * color_laplacian / normal_length) * normal : Vec3{};
        particle.force = pressure_force + viscosity_force + surface_force + particle.density * Vec3{0.0, -p_.gravity, 0.0};
    }

    for (Particle& particle : particles_) {
        particle.velocity += (p_.dt / particle.density) * particle.force;
        particle.position += p_.dt * particle.velocity;
        constrain_to_box(particle);
    }
}

void SphSimulation::constrain_to_box(Particle& particle) const {
    auto constrain = [this](double& position, double& velocity, double maximum) {
        if (position < 0.0) { position = 0.0; velocity *= p_.wall_reflection; }
        if (position > maximum) { position = maximum; velocity *= p_.wall_reflection; }
    };
    constrain(particle.position.x, particle.velocity.x, p_.box_x);
    constrain(particle.position.y, particle.velocity.y, p_.box_y);
    constrain(particle.position.z, particle.velocity.z, p_.box_z);
}

void SphSimulation::run() { for (int step_index = 0; step_index < p_.steps; ++step_index) step(); }
double SphSimulation::checksum() const {
    double sum = 0.0;
    for (const Particle& particle : particles_) sum += particle.position.x + particle.position.y + particle.position.z + particle.density;
    return sum;
}
