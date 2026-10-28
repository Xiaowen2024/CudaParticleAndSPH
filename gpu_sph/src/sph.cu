#include "sph.cuh"
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
constexpr int kThreadsPerBlock = 256;
constexpr int kMaxNeighborsPerParticle = 128;
constexpr double kPi = 3.14159265358979323846;
void check_cuda(cudaError_t s, const char* op) { if (s != cudaSuccess) throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(s)); }
void precompute_kernel_coefficients(Parameters& p) {
    const double h6 = std::pow(p.smoothing_length, 6.0), h9 = std::pow(p.smoothing_length, 9.0);
    p.poly6_coefficient = 315.0 / (64.0 * kPi * h9);
    p.spiky_gradient_coefficient = -45.0 / (kPi * h6);
    p.viscosity_laplacian_coefficient = 45.0 / (kPi * h6);
    p.poly6_derivative_coefficient = -945.0 / (32.0 * kPi * h9);
}
void allocate_particles(ParticlesSoA& q, int n) {
    const size_t b = size_t(n) * sizeof(double);
    check_cuda(cudaMalloc(&q.x,b),"malloc x"); check_cuda(cudaMalloc(&q.y,b),"malloc y"); check_cuda(cudaMalloc(&q.z,b),"malloc z");
    check_cuda(cudaMalloc(&q.vx,b),"malloc vx"); check_cuda(cudaMalloc(&q.vy,b),"malloc vy"); check_cuda(cudaMalloc(&q.vz,b),"malloc vz");
    check_cuda(cudaMalloc(&q.fx,b),"malloc fx"); check_cuda(cudaMalloc(&q.fy,b),"malloc fy"); check_cuda(cudaMalloc(&q.fz,b),"malloc fz");
    check_cuda(cudaMalloc(&q.density,b),"malloc density"); check_cuda(cudaMalloc(&q.pressure,b),"malloc pressure");
}
void free_particles(ParticlesSoA& q) {
    cudaFree(q.x); q.x=nullptr; cudaFree(q.y); q.y=nullptr; cudaFree(q.z); q.z=nullptr;
    cudaFree(q.vx); q.vx=nullptr; cudaFree(q.vy); q.vy=nullptr; cudaFree(q.vz); q.vz=nullptr;
    cudaFree(q.fx); q.fx=nullptr; cudaFree(q.fy); q.fy=nullptr; cudaFree(q.fz); q.fz=nullptr;
    cudaFree(q.density); q.density=nullptr; cudaFree(q.pressure); q.pressure=nullptr;
}
void copy_initial_particles(const std::vector<Particle>& h, ParticlesSoA q) {
    const int n = int(h.size()); const size_t b = size_t(n) * sizeof(double);
    std::vector<double> x(n),y(n),z(n),vx(n),vy(n),vz(n),fx(n),fy(n),fz(n),d(n),p(n);
    for (int i=0;i<n;++i) {
        x[i]=h[i].position.x; y[i]=h[i].position.y; z[i]=h[i].position.z;
        vx[i]=h[i].velocity.x; vy[i]=h[i].velocity.y; vz[i]=h[i].velocity.z;
        fx[i]=h[i].force.x; fy[i]=h[i].force.y; fz[i]=h[i].force.z; d[i]=h[i].density; p[i]=h[i].pressure;
    }
    check_cuda(cudaMemcpy(q.x, x.data(), b, cudaMemcpyHostToDevice), "copy x");
    check_cuda(cudaMemcpy(q.y, y.data(), b, cudaMemcpyHostToDevice), "copy y");
    check_cuda(cudaMemcpy(q.z, z.data(), b, cudaMemcpyHostToDevice), "copy z");
    check_cuda(cudaMemcpy(q.vx, vx.data(), b, cudaMemcpyHostToDevice), "copy vx");
    check_cuda(cudaMemcpy(q.vy, vy.data(), b, cudaMemcpyHostToDevice), "copy vy");
    check_cuda(cudaMemcpy(q.vz, vz.data(), b, cudaMemcpyHostToDevice), "copy vz");
    check_cuda(cudaMemcpy(q.fx, fx.data(), b, cudaMemcpyHostToDevice), "copy fx");
    check_cuda(cudaMemcpy(q.fy, fy.data(), b, cudaMemcpyHostToDevice), "copy fy");
    check_cuda(cudaMemcpy(q.fz, fz.data(), b, cudaMemcpyHostToDevice), "copy fz");
    check_cuda(cudaMemcpy(q.density, d.data(), b, cudaMemcpyHostToDevice), "copy density");
    check_cuda(cudaMemcpy(q.pressure, p.data(), b, cudaMemcpyHostToDevice), "copy pressure");
}
__device__ int cell_index(double x,double y,double z,int nx,int ny,int nz,double h) {
    int cx=int(x/h), cy=int(y/h), cz=int(z/h);
    cx=cx<0?0:(cx>=nx?nx-1:cx); cy=cy<0?0:(cy>=ny?ny-1:cy); cz=cz<0?0:(cz>=nz?nz-1:cz);
    return cx + nx*(cy + ny*cz);
}
__device__ void cell_coordinates(int cell,int nx,int ny,int& x,int& y,int& z) {
    const int per_layer=nx*ny; z=cell/per_layer; const int r=cell-z*per_layer; y=r/nx; x=r-y*nx;
}
__device__ double poly6(double r2,const Parameters& p) {
    const double h2=p.smoothing_length*p.smoothing_length; if(r2>h2)return 0.; const double t=h2-r2; return p.poly6_coefficient*t*t*t;
}

__global__ void count_particles_per_cell(ParticlesSoA q,int n,int nx,int ny,int nz,double h,int* particle_cell,int* counts) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    const int c=cell_index(q.x[i],q.y[i],q.z[i],nx,ny,nz,h); particle_cell[i]=c; atomicAdd(&counts[c],1);
}
__global__ void scatter_particle_ids(int n,const int* particle_cell,int* cursors,int* ids) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; ids[atomicAdd(&cursors[particle_cell[i]],1)]=i;
}
__global__ void reorder_particles(ParticlesSoA a,ParticlesSoA b,const int* ids,int n) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return; const int j=ids[i];
    b.x[i]=a.x[j]; b.y[i]=a.y[j]; b.z[i]=a.z[j]; b.vx[i]=a.vx[j]; b.vy[i]=a.vy[j]; b.vz[i]=a.vz[j];
    b.fx[i]=a.fx[j]; b.fy[i]=a.fy[j]; b.fz[i]=a.fz[j]; b.density[i]=a.density[j]; b.pressure[i]=a.pressure[j];
}
__global__ void count_neighbors(ParticlesSoA q,int n,const int* counts,const int* offsets,int nx,int ny,int nz,Parameters p,int* neighbor_counts) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    const double px=q.x[i],py=q.y[i],pz=q.z[i],h2=p.smoothing_length*p.smoothing_length;
    const int cell=cell_index(px,py,pz,nx,ny,nz,p.smoothing_length); int cx,cy,cz; cell_coordinates(cell,nx,ny,cx,cy,cz); int total=0;
    for(int dz=-1;dz<=1;++dz) for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx) {
        const int x=cx+dx,y=cy+dy,z=cz+dz; if(x<0||x>=nx||y<0||y>=ny||z<0||z>=nz)continue;
        const int c=x+nx*(y+ny*z),end=offsets[c]+counts[c];
        for(int j=offsets[c];j<end;++j) { const double rx=q.x[j]-px,ry=q.y[j]-py,rz=q.z[j]-pz; if(rx*rx+ry*ry+rz*rz<=h2)++total; }
    }
    neighbor_counts[i]=total;
}
__global__ void calculate_density_pressure_and_fill_neighbors(ParticlesSoA q,int n,const int* counts,const int* offsets,int nx,int ny,int nz,Parameters p,
                                                               const int* neighbor_offsets,int* neighbor_ids,int neighbor_capacity,int* overflow) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    const double px=q.x[i],py=q.y[i],pz=q.z[i],h2=p.smoothing_length*p.smoothing_length;
    const int cell=cell_index(px,py,pz,nx,ny,nz,p.smoothing_length); int cx,cy,cz; cell_coordinates(cell,nx,ny,cx,cy,cz);
    int neighbor_index=neighbor_offsets[i]; double density=0.;
    for(int dz=-1;dz<=1;++dz) for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx) {
        const int x=cx+dx,y=cy+dy,z=cz+dz; if(x<0||x>=nx||y<0||y>=ny||z<0||z>=nz)continue;
        const int c=x+nx*(y+ny*z),end=offsets[c]+counts[c];
        for(int j=offsets[c];j<end;++j) {
            const double rx=q.x[j]-px,ry=q.y[j]-py,rz=q.z[j]-pz,r2=rx*rx+ry*ry+rz*rz;
            if(r2<=h2) { if(neighbor_index<neighbor_capacity)neighbor_ids[neighbor_index]=j; else atomicExch(overflow,1); ++neighbor_index; density+=p.mass*poly6(r2,p); }
        }
    }
    q.density[i]=fmax(density,1e-12); q.pressure[i]=p.pressure_stiffness*(q.density[i]-p.rest_density);
}
__global__ void calculate_density_pressure(ParticlesSoA q,int n,const int* counts,const int* offsets,int nx,int ny,int nz,Parameters p) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    const double px=q.x[i],py=q.y[i],pz=q.z[i],h2=p.smoothing_length*p.smoothing_length;
    const int cell=cell_index(px,py,pz,nx,ny,nz,p.smoothing_length); int cx,cy,cz; cell_coordinates(cell,nx,ny,cx,cy,cz); double density=0.;
    for(int dz=-1;dz<=1;++dz) for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx) {
        const int x=cx+dx,y=cy+dy,z=cz+dz; if(x<0||x>=nx||y<0||y>=ny||z<0||z>=nz)continue;
        const int c=x+nx*(y+ny*z), end=offsets[c]+counts[c];
        for(int j=offsets[c];j<end;++j) { const double rx=q.x[j]-px,ry=q.y[j]-py,rz=q.z[j]-pz,r2=rx*rx+ry*ry+rz*rz; if(r2<=h2)density+=p.mass*poly6(r2,p); }
    }
    q.density[i]=fmax(density,1e-12); q.pressure[i]=p.pressure_stiffness*(q.density[i]-p.rest_density);
}
__global__ void calculate_forces(ParticlesSoA q,int n,const int* counts,const int* offsets,int nx,int ny,int nz,Parameters p) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    const double px=q.x[i],py=q.y[i],pz=q.z[i],vx=q.vx[i],vy=q.vy[i],vz=q.vz[i],density=q.density[i],pressure_i=q.pressure[i],h2=p.smoothing_length*p.smoothing_length;
    const int cell=cell_index(px,py,pz,nx,ny,nz,p.smoothing_length); int cx,cy,cz; cell_coordinates(cell,nx,ny,cx,cy,cz);
    double pfx=0.,pfy=0.,pfz=0.,vfx=0.,vfy=0.,vfz=0.,nxv=0.,nyv=0.,nzv=0.,color_lap=0.;
    for(int dz=-1;dz<=1;++dz) for(int dy=-1;dy<=1;++dy) for(int dx=-1;dx<=1;++dx) {
        const int x=cx+dx,y=cy+dy,z=cz+dz; if(x<0||x>=nx||y<0||y>=ny||z<0||z>=nz)continue;
        const int c=x+nx*(y+ny*z), end=offsets[c]+counts[c];
        for(int j=offsets[c];j<end;++j) {
            if(j==i)continue; const double rx=px-q.x[j],ry=py-q.y[j],rz=pz-q.z[j],r2=rx*rx+ry*ry+rz*rz;
            if(r2<=1e-24||r2>h2)continue; const double r=sqrt(r2), inv_r=1./r, inv_d=1./q.density[j], h_minus_r=p.smoothing_length-r, h2_minus_r2=h2-r2;
            const double common=p.mass*inv_d;
            const double ps=-.5*common*(pressure_i+q.pressure[j])*p.spiky_gradient_coefficient*h_minus_r*h_minus_r*inv_r;
            pfx=fma(ps,rx,pfx); pfy=fma(ps,ry,pfy); pfz=fma(ps,rz,pfz);
            const double vs=p.viscosity*common*p.viscosity_laplacian_coefficient*h_minus_r;
            vfx+=vs*(q.vx[j]-vx); vfy+=vs*(q.vy[j]-vy); vfz+=vs*(q.vz[j]-vz);
            const double ns=common*p.poly6_derivative_coefficient*h2_minus_r2*h2_minus_r2*inv_r;
            nxv=fma(ns,rx,nxv); nyv=fma(ns,ry,nyv); nzv=fma(ns,rz,nzv);
            color_lap+=common*p.poly6_derivative_coefficient*h2_minus_r2*(3.*h2-7.*r2);
        }
    }
    const double normal=sqrt(nxv*nxv+nyv*nyv+nzv*nzv), ss=normal>=p.surface_threshold?-p.surface_tension*color_lap/normal:0.;
    q.fx[i]=pfx+vfx+ss*nxv; q.fy[i]=pfy+vfy+ss*nyv-density*p.gravity; q.fz[i]=pfz+vfz+ss*nzv;
}
__global__ void calculate_forces_from_neighbors(ParticlesSoA q,int n,const int* neighbor_counts,const int* neighbor_offsets,const int* neighbor_ids,int neighbor_capacity,Parameters p) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    const double px=q.x[i],py=q.y[i],pz=q.z[i],vx=q.vx[i],vy=q.vy[i],vz=q.vz[i],density=q.density[i],pressure_i=q.pressure[i],h2=p.smoothing_length*p.smoothing_length;
    double pfx=0.,pfy=0.,pfz=0.,vfx=0.,vfy=0.,vfz=0.,nxv=0.,nyv=0.,nzv=0.,color_lap=0.;
    const int end=min(neighbor_offsets[i]+neighbor_counts[i],neighbor_capacity);
    for(int slot=neighbor_offsets[i];slot<end;++slot) {
        const int j=neighbor_ids[slot]; if(j==i)continue;
        const double rx=px-q.x[j],ry=py-q.y[j],rz=pz-q.z[j],r2=rx*rx+ry*ry+rz*rz;
        if(r2<=1e-24)continue;
        const double r=sqrt(r2),inv_r=1./r,inv_d=1./q.density[j],h_minus_r=p.smoothing_length-r,h2_minus_r2=h2-r2,common=p.mass*inv_d;
        const double ps=-.5*common*(pressure_i+q.pressure[j])*p.spiky_gradient_coefficient*h_minus_r*h_minus_r*inv_r;
        pfx=fma(ps,rx,pfx); pfy=fma(ps,ry,pfy); pfz=fma(ps,rz,pfz);
        const double vs=p.viscosity*common*p.viscosity_laplacian_coefficient*h_minus_r;
        vfx+=vs*(q.vx[j]-vx); vfy+=vs*(q.vy[j]-vy); vfz+=vs*(q.vz[j]-vz);
        const double ns=common*p.poly6_derivative_coefficient*h2_minus_r2*h2_minus_r2*inv_r;
        nxv=fma(ns,rx,nxv); nyv=fma(ns,ry,nyv); nzv=fma(ns,rz,nzv);
        color_lap+=common*p.poly6_derivative_coefficient*h2_minus_r2*(3.*h2-7.*r2);
    }
    const double normal=sqrt(nxv*nxv+nyv*nyv+nzv*nzv),ss=normal>=p.surface_threshold?-p.surface_tension*color_lap/normal:0.;
    q.fx[i]=pfx+vfx+ss*nxv; q.fy[i]=pfy+vfy+ss*nyv-density*p.gravity; q.fz[i]=pfz+vfz+ss*nzv;
}
__device__ void constrain(double& x,double& v,double max,double reflection) { if(x<0.){x=0.;v*=reflection;} if(x>max){x=max;v*=reflection;} }
__global__ void update_velocity_position(ParticlesSoA q,int n,Parameters p) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n)return;
    double vx=q.vx[i]+p.dt/q.density[i]*q.fx[i],vy=q.vy[i]+p.dt/q.density[i]*q.fy[i],vz=q.vz[i]+p.dt/q.density[i]*q.fz[i];
    double x=q.x[i]+p.dt*vx,y=q.y[i]+p.dt*vy,z=q.z[i]+p.dt*vz;
    constrain(x,vx,p.box_x,p.wall_reflection); constrain(y,vy,p.box_y,p.wall_reflection); constrain(z,vz,p.box_z,p.wall_reflection);
    q.x[i]=x;q.y[i]=y;q.z[i]=z;q.vx[i]=vx;q.vy[i]=vy;q.vz[i]=vz;
}
} // namespace

std::vector<Particle> make_initial_particles(const Parameters& p) {
    std::vector<Particle> r; r.reserve(size_t(p.grid_length)*p.grid_length*p.grid_length);
    const double radius=std::cbrt(3.*p.mass/(4.*kPi*p.rest_density)), spacing=2.*radius;
    for(int y=0;y<p.grid_length;++y) for(int x=0;x<p.grid_length;++x) for(int z=0;z<p.grid_length;++z)
        r.push_back({{x*spacing+radius,y*spacing+radius,z*spacing+radius},{},{},0.,0.});
    return r;
}
void fit_domain_to_initial_lattice(Parameters& p) {
    const double radius=std::cbrt(3.*p.mass/(4.*kPi*p.rest_density));
    const double lattice_extent=2.*p.grid_length*radius;
    p.box_x=std::max(p.box_x,lattice_extent);
    p.box_y=std::max(p.box_y,lattice_extent);
    p.box_z=std::max(p.box_z,lattice_extent);
}
void SphGpuSimulation::initialize(const std::vector<Particle>& h, Parameters p) {
    cleanup(); precompute_kernel_coefficients(p); parameters_=p; num_particles_=int(h.size());
    cells_x_=int(std::ceil(p.box_x/p.smoothing_length)); cells_y_=int(std::ceil(p.box_y/p.smoothing_length)); cells_z_=int(std::ceil(p.box_z/p.smoothing_length));
    num_cells_=cells_x_*cells_y_*cells_z_; blocks_=(num_particles_+kThreadsPerBlock-1)/kThreadsPerBlock;
    allocate_particles(gpu_particles_,num_particles_); allocate_particles(reordered_particles_,num_particles_);
    check_cuda(cudaMalloc(&cell_counts_,num_cells_*sizeof(int)),"malloc counts"); check_cuda(cudaMalloc(&cell_cursors_,num_cells_*sizeof(int)),"malloc cursors");
    check_cuda(cudaMalloc(&cell_offsets_,num_cells_*sizeof(int)),"malloc offsets"); check_cuda(cudaMalloc(&particle_cell_,num_particles_*sizeof(int)),"malloc cells");
    check_cuda(cudaMalloc(&particle_ids_,num_particles_*sizeof(int)),"malloc ids"); copy_initial_particles(h,gpu_particles_);
    check_cuda(cub::DeviceScan::ExclusiveSum(nullptr,scan_temp_storage_bytes_,cell_counts_,cell_offsets_,num_cells_),"query cell scan");
    if(parameters_.use_neighbor_list) {
        neighbor_capacity_=num_particles_*kMaxNeighborsPerParticle;
        check_cuda(cudaMalloc(&neighbor_counts_,num_particles_*sizeof(int)),"malloc neighbor counts");
        check_cuda(cudaMalloc(&neighbor_offsets_,num_particles_*sizeof(int)),"malloc neighbor offsets");
        check_cuda(cudaMalloc(&neighbor_ids_,size_t(neighbor_capacity_)*sizeof(int)),"malloc neighbor ids");
        check_cuda(cudaMalloc(&neighbor_overflow_,sizeof(int)),"malloc neighbor overflow");
        size_t neighbor_scan_bytes=0;
        check_cuda(cub::DeviceScan::ExclusiveSum(nullptr,neighbor_scan_bytes,neighbor_counts_,neighbor_offsets_,num_particles_),"query neighbor scan");
        scan_temp_storage_bytes_=std::max(scan_temp_storage_bytes_,neighbor_scan_bytes);
    }
    check_cuda(cudaMalloc(&scan_temp_storage_,scan_temp_storage_bytes_),"malloc scan temp");
}
void SphGpuSimulation::rebuild_cells() {
    check_cuda(cudaMemset(cell_counts_,0,num_cells_*sizeof(int)),"clear counts");
    count_particles_per_cell<<<blocks_,kThreadsPerBlock>>>(gpu_particles_,num_particles_,cells_x_,cells_y_,cells_z_,parameters_.smoothing_length,particle_cell_,cell_counts_); check_cuda(cudaGetLastError(),"count launch");
    check_cuda(cub::DeviceScan::ExclusiveSum(scan_temp_storage_,scan_temp_storage_bytes_,cell_counts_,cell_offsets_,num_cells_),"scan offsets");
    check_cuda(cub::DeviceScan::ExclusiveSum(scan_temp_storage_,scan_temp_storage_bytes_,cell_counts_,cell_cursors_,num_cells_),"scan cursors");
    scatter_particle_ids<<<blocks_,kThreadsPerBlock>>>(num_particles_,particle_cell_,cell_cursors_,particle_ids_); check_cuda(cudaGetLastError(),"scatter launch");
    reorder_particles<<<blocks_,kThreadsPerBlock>>>(gpu_particles_,reordered_particles_,particle_ids_,num_particles_); check_cuda(cudaGetLastError(),"reorder launch");
    std::swap(gpu_particles_,reordered_particles_);
}
void SphGpuSimulation::build_neighbor_list() {
    if(!parameters_.use_neighbor_list)return;
    check_cuda(cudaMemset(neighbor_overflow_,0,sizeof(int)),"clear neighbor overflow");
    count_neighbors<<<blocks_,kThreadsPerBlock>>>(gpu_particles_,num_particles_,cell_counts_,cell_offsets_,cells_x_,cells_y_,cells_z_,parameters_,neighbor_counts_); check_cuda(cudaGetLastError(),"neighbor count launch");
    check_cuda(cub::DeviceScan::ExclusiveSum(scan_temp_storage_,scan_temp_storage_bytes_,neighbor_counts_,neighbor_offsets_,num_particles_),"scan neighbor offsets");
}
void SphGpuSimulation::step() {
    rebuild_cells();
    if(parameters_.use_neighbor_list) {
        build_neighbor_list();
        calculate_density_pressure_and_fill_neighbors<<<blocks_,kThreadsPerBlock>>>(gpu_particles_,num_particles_,cell_counts_,cell_offsets_,cells_x_,cells_y_,cells_z_,parameters_,neighbor_offsets_,neighbor_ids_,neighbor_capacity_,neighbor_overflow_); check_cuda(cudaGetLastError(),"density/list fill launch");
        calculate_forces_from_neighbors<<<blocks_,kThreadsPerBlock>>>(gpu_particles_,num_particles_,neighbor_counts_,neighbor_offsets_,neighbor_ids_,neighbor_capacity_,parameters_); check_cuda(cudaGetLastError(),"force list launch");
    } else {
        calculate_density_pressure<<<blocks_,kThreadsPerBlock>>>(gpu_particles_,num_particles_,cell_counts_,cell_offsets_,cells_x_,cells_y_,cells_z_,parameters_); check_cuda(cudaGetLastError(),"density launch");
        calculate_forces<<<blocks_,kThreadsPerBlock>>>(gpu_particles_,num_particles_,cell_counts_,cell_offsets_,cells_x_,cells_y_,cells_z_,parameters_); check_cuda(cudaGetLastError(),"force launch");
    }
    update_velocity_position<<<blocks_,kThreadsPerBlock>>>(gpu_particles_,num_particles_,parameters_); check_cuda(cudaGetLastError(),"move launch");
}
void SphGpuSimulation::run() { for(int i=0;i<parameters_.steps;++i)step(); }
double SphGpuSimulation::run_timed() {
    cudaEvent_t start,stop; check_cuda(cudaEventCreate(&start),"create start"); check_cuda(cudaEventCreate(&stop),"create stop"); check_cuda(cudaEventRecord(start),"record start");
    run(); check_cuda(cudaEventRecord(stop),"record stop"); check_cuda(cudaEventSynchronize(stop),"sync stop"); float ms=0.; check_cuda(cudaEventElapsedTime(&ms,start,stop),"elapsed");
    check_cuda(cudaEventDestroy(start),"destroy start"); check_cuda(cudaEventDestroy(stop),"destroy stop"); return double(ms)/1000.;
}
double SphGpuSimulation::checksum() const {
    std::vector<double> x(num_particles_),y(num_particles_),z(num_particles_),d(num_particles_); const size_t b=size_t(num_particles_)*sizeof(double);
    check_cuda(cudaMemcpy(x.data(),gpu_particles_.x,b,cudaMemcpyDeviceToHost),"copy checksum x"); check_cuda(cudaMemcpy(y.data(),gpu_particles_.y,b,cudaMemcpyDeviceToHost),"copy checksum y");
    check_cuda(cudaMemcpy(z.data(),gpu_particles_.z,b,cudaMemcpyDeviceToHost),"copy checksum z"); check_cuda(cudaMemcpy(d.data(),gpu_particles_.density,b,cudaMemcpyDeviceToHost),"copy checksum density");
    double sum=0.; for(int i=0;i<num_particles_;++i)sum+=x[i]+y[i]+z[i]+d[i]; return sum;
}
void SphGpuSimulation::validate_cell_list() const {
    std::vector<int> c(num_cells_); check_cuda(cudaMemcpy(c.data(),cell_counts_,num_cells_*sizeof(int),cudaMemcpyDeviceToHost),"copy counts");
    long long total=0; for(int v:c)total+=v; if(total!=num_particles_)throw std::runtime_error("cell list validation failed");
    if(parameters_.use_neighbor_list) { int overflow=0; check_cuda(cudaMemcpy(&overflow,neighbor_overflow_,sizeof(int),cudaMemcpyDeviceToHost),"copy neighbor overflow"); if(overflow)throw std::runtime_error("neighbor-list capacity exceeded"); }
    std::cout<<"cell_list: particles="<<num_particles_<<" cells="<<num_cells_<<" dimensions="<<cells_x_<<'x'<<cells_y_<<'x'<<cells_z_<<'\n';
}
void SphGpuSimulation::cleanup() {
    free_particles(gpu_particles_); free_particles(reordered_particles_); cudaFree(cell_counts_);cell_counts_=nullptr;cudaFree(cell_cursors_);cell_cursors_=nullptr;
    cudaFree(cell_offsets_);cell_offsets_=nullptr;cudaFree(particle_cell_);particle_cell_=nullptr;cudaFree(particle_ids_);particle_ids_=nullptr;
    cudaFree(neighbor_counts_);neighbor_counts_=nullptr;cudaFree(neighbor_offsets_);neighbor_offsets_=nullptr;cudaFree(neighbor_ids_);neighbor_ids_=nullptr;cudaFree(neighbor_overflow_);neighbor_overflow_=nullptr;neighbor_capacity_=0;
    cudaFree(scan_temp_storage_);scan_temp_storage_=nullptr;scan_temp_storage_bytes_=0;num_particles_=num_cells_=blocks_=0;
}
