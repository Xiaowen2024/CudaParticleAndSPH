#pragma once

#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

struct Vec3 {
    double x = 0.0, y = 0.0, z = 0.0;
    Vec3() = default;
    Vec3(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}
    Vec3& operator+=(const Vec3& other);
    Vec3& operator-=(const Vec3& other);
};
Vec3 operator+(Vec3 a, const Vec3& b);
Vec3 operator-(Vec3 a, const Vec3& b);
Vec3 operator*(double s, Vec3 v);
Vec3 operator*(Vec3 v, double s);
Vec3 operator/(Vec3 v, double s);
double norm(const Vec3& v);
double norm_squared(const Vec3& v);

struct Particle {
    Vec3 position;
    Vec3 velocity;
    Vec3 force;
    double density = 0.0;
    double pressure = 0.0;
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
};

// Keep the initial lattice inside the simulation domain when --grid changes.
void fit_domain_to_initial_lattice(Parameters& parameters);

class SphSimulation {
  public:
    explicit SphSimulation(Parameters parameters);
    void step();
    void run();
    std::size_t particle_count() const { return particles_.size(); }
    double checksum() const;

  private:
    using Cell = std::vector<int>;
    Parameters p_;
    std::vector<Particle> particles_;
    std::unordered_map<std::int64_t, Cell> cells_;

    void initialize_particles();
    void rebuild_cells();
    std::int64_t cell_hash(const Vec3& position) const;
    double poly6(double r2) const;
    double spiky_gradient(double r) const;
    double viscosity_laplacian(double r) const;
    double poly6_gradient(double r) const;
    double poly6_laplacian(double r) const;
    void constrain_to_box(Particle& particle) const;
};
