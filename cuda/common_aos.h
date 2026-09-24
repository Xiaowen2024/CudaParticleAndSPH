#ifndef __CS267_COMMON_AOS_H__
#define __CS267_COMMON_AOS_H__

#define nsteps   1000
#define savefreq 10
#define density  0.0005
#define mass     0.01
#define cutoff   0.01
#define min_r    (cutoff / 100)
#define dt       0.0005

struct particle_t {
    double x, y;
    double vx, vy;
    double ax, ay;
};

void init_simulation(int num_parts, double size, int reorder_interval, bool analyze_reuse);
void simulate_one_step(particle_t*& parts, int num_parts, double size);
void clean_simulation();

#endif
