#ifndef __CS267_COMMON_H__
#define __CS267_COMMON_H__

// Program Constants
#define nsteps   1000
#define savefreq 10
#define density  0.0005
#define mass     0.01
#define cutoff   0.01
#define min_r    (cutoff / 100)
#define dt       0.0005

#if PARTICLE_USE_FLOAT
using particle_scalar_t = float;
#else
using particle_scalar_t = double;
#endif

constexpr particle_scalar_t cutoff_value = static_cast<particle_scalar_t>(cutoff);
constexpr particle_scalar_t min_r_value = static_cast<particle_scalar_t>(min_r);
constexpr particle_scalar_t mass_value = static_cast<particle_scalar_t>(mass);
constexpr particle_scalar_t dt_value = static_cast<particle_scalar_t>(dt);

// Struct of arrays: every field is a separate contiguous array of num_parts doubles.
struct particles_soa_t {
    particle_scalar_t* x;
    particle_scalar_t* y;
    particle_scalar_t* vx;
    particle_scalar_t* vy;
    particle_scalar_t* ax;
    particle_scalar_t* ay;
};

void init_simulation(int num_parts, double size, int reorder_interval, int analyze_reuse_step);
void simulate_one_step(particles_soa_t& parts, int num_parts, double size);
void clean_simulation();

#endif
