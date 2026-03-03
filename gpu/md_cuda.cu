// md_cuda.cu

#include <cuda_runtime.h>
#include <cub/cub.cuh>

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>
#include <cstdlib>
#include <algorithm>

#define CUDA_CHECK(call) do {                                   \
    cudaError_t err = (call);                                   \
    if (err != cudaSuccess) {                                   \
        std::cerr << "CUDA error: " << cudaGetErrorString(err)  \
                  << " at " << __FILE__ << ":" << __LINE__      \
                  << "\n";                                      \
        std::exit(1);                                           \
    }                                                           \
} while(0)

using real = double;

//parameters
static constexpr int  N_req      = 1000000;
static constexpr real density    = (real)0.8;
static constexpr real dt         = (real)0.001;
static constexpr int  n_steps    = 1000;
static constexpr int  save_every = 10;

static constexpr int  DUP_MAX = 8;

//device helpers 
__device__ __forceinline__ int wrap_int_d(int q, int k) {
    q %= k;
    if (q < 0) q += k;
    return q;
}

__device__ __forceinline__ int block_index_d(int bx, int by, int bz, int k) {
    return (bx * k + by) * k + bz;
}

__device__ __forceinline__ void minimum_image_d(real& dx, real& dy, real& dz, real L, real halfL) {
    if (dx >  halfL) dx -= L;
    if (dx < -halfL) dx += L;
    if (dy >  halfL) dy -= L;
    if (dy < -halfL) dy += L;
    if (dz >  halfL) dz -= L;
    if (dz < -halfL) dz += L;
}

__device__ __forceinline__ int block_coord_d(real x, real s, int kblocks) {
    int b = (int)floor(x / s);
    if (b < 0) b = 0;
    if (b >= kblocks) b = kblocks - 1;
    return b;
}

__device__ __forceinline__ void wrap_position_d(real& x, real L) {
    x -= L * floor(x / L);
    if (x < (real)0.0) x += L;
    if (x >= L) x = nextafter(L, (real)0.0);
}

__device__ __forceinline__
int axis_blocks_d(real coord, real s, real rc, int kblocks, int out[2]) {
    if (kblocks <= 1) {
        out[0] = 0;
        return 1;
    }

    int q = (int)floor(coord / s);
    if (q < 0) q = 0;
    if (q >= kblocks) q = kblocks - 1;

    real local = coord - (real)q * s;
    out[0] = q;

    if (local < rc) {
        out[1] = wrap_int_d(q - 1, kblocks);
        return 2;
    }
    if (local >= (s - rc)) {
        out[1] = wrap_int_d(q + 1, kblocks);
        return 2;
    }
    return 1;
}

//verlet kernels 
__global__ void copy_accel_kernel(
    int p,
    const real* __restrict__ ax,
    const real* __restrict__ ay,
    const real* __restrict__ az,
    real* __restrict__ ax_old,
    real* __restrict__ ay_old,
    real* __restrict__ az_old
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < p) {
        ax_old[i] = ax[i];
        ay_old[i] = ay[i];
        az_old[i] = az[i];
    }
}

__global__ void verlet_position_kernel(
    int p,
    real* __restrict__ x,
    real* __restrict__ y,
    real* __restrict__ z,
    const real* __restrict__ vx,
    const real* __restrict__ vy,
    const real* __restrict__ vz,
    const real* __restrict__ ax,
    const real* __restrict__ ay,
    const real* __restrict__ az,
    real dt_,
    real L
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < p) {
        x[i] += vx[i] * dt_ + (real)0.5 * ax[i] * dt_ * dt_;
        y[i] += vy[i] * dt_ + (real)0.5 * ay[i] * dt_ * dt_;
        z[i] += vz[i] * dt_ + (real)0.5 * az[i] * dt_ * dt_;

        wrap_position_d(x[i], L);
        wrap_position_d(y[i], L);
        wrap_position_d(z[i], L);
    }
}

__global__ void verlet_velocity_kernel(
    int p,
    real* __restrict__ vx,
    real* __restrict__ vy,
    real* __restrict__ vz,
    const real* __restrict__ ax_old,
    const real* __restrict__ ay_old,
    const real* __restrict__ az_old,
    const real* __restrict__ ax,
    const real* __restrict__ ay,
    const real* __restrict__ az,
    real dt_
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < p) {
        vx[i] += (real)0.5 * (ax_old[i] + ax[i]) * dt_;
        vy[i] += (real)0.5 * (ay_old[i] + ay[i]) * dt_;
        vz[i] += (real)0.5 * (az_old[i] + az[i]) * dt_;
    }
}

//list building 
__global__ void count_blocks_dup_kernel(
    int p,
    const real* __restrict__ x,
    const real* __restrict__ y,
    const real* __restrict__ z,
    int kblocks,
    real s,
    real rc,
    int* __restrict__ blockCount,
    int* __restrict__ particleHome
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p) return;

    int xs[2], ys[2], zs[2];
    int nx = axis_blocks_d(x[i], s, rc, kblocks, xs);
    int ny = axis_blocks_d(y[i], s, rc, kblocks, ys);
    int nz = axis_blocks_d(z[i], s, rc, kblocks, zs);

    int home = block_index_d(xs[0], ys[0], zs[0], kblocks);
    particleHome[i] = home;

    for (int ix = 0; ix < nx; ++ix)
        for (int iy = 0; iy < ny; ++iy)
            for (int iz = 0; iz < nz; ++iz) {
                int b = block_index_d(xs[ix], ys[iy], zs[iz], kblocks);
                atomicAdd(&blockCount[b], 1);
            }
}

__global__ void set_blockstart_last_from_counts_kernel(
    int nb,
    const int* __restrict__ blockStart,
    const int* __restrict__ blockCount,
    int* __restrict__ blockStartOut
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        int last = nb - 1;
        int total = blockStart[last] + blockCount[last];
        blockStartOut[nb] = total;
    }
}

__global__ void init_write_kernel(int n, const int* __restrict__ start, int* __restrict__ write) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) write[i] = start[i];
}

__global__ void scatter_blocks_dup_kernel(
    int p,
    const real* __restrict__ x,
    const real* __restrict__ y,
    const real* __restrict__ z,
    int kblocks,
    real s,
    real rc,
    int* __restrict__ blockWrite,
    int* __restrict__ blockList
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p) return;

    int xs[2], ys[2], zs[2];
    int nx = axis_blocks_d(x[i], s, rc, kblocks, xs);
    int ny = axis_blocks_d(y[i], s, rc, kblocks, ys);
    int nz = axis_blocks_d(z[i], s, rc, kblocks, zs);

    for (int ix = 0; ix < nx; ++ix)
        for (int iy = 0; iy < ny; ++iy)
            for (int iz = 0; iz < nz; ++iz) {
                int b = block_index_d(xs[ix], ys[iy], zs[iz], kblocks);
                int pos = atomicAdd(&blockWrite[b], 1);
                blockList[pos] = i;
            }
}

__global__ void count_home_kernel(
    int p,
    const int* __restrict__ particleHome,
    int* __restrict__ homeCount
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p) return;
    atomicAdd(&homeCount[particleHome[i]], 1);
}

__global__ void set_home_last_kernel(int nb, int p, int* homeStart) {
    if (threadIdx.x == 0 && blockIdx.x == 0) homeStart[nb] = p;
}

__global__ void scatter_home_kernel(
    int p,
    const int* __restrict__ particleHome,
    int* __restrict__ homeWrite,
    int* __restrict__ homeList
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p) return;

    int b = particleHome[i];
    int pos = atomicAdd(&homeWrite[b], 1);
    homeList[pos] = i;
}

//Forces 
__global__ void compute_forces_blockcentric_kernel(
    int nb,
    real* __restrict__ ax,
    real* __restrict__ ay,
    real* __restrict__ az,
    const real* __restrict__ x,
    const real* __restrict__ y,
    const real* __restrict__ z,
    real L,
    real halfL,
    real rCut2,
    const int* __restrict__ blockStart,
    const int* __restrict__ blockList,
    const int* __restrict__ homeStart,
    const int* __restrict__ homeList
) {
    int b = (int)blockIdx.x;
    if (b >= nb) return;

    int c_begin = blockStart[b];
    int c_end   = blockStart[b + 1];

    int h_begin = homeStart[b];
    int h_end   = homeStart[b + 1];

    if (h_begin == h_end || c_begin == c_end) return;

    const real tiny = (real)1e-12;

    for (int hi = h_begin + (int)threadIdx.x; hi < h_end; hi += (int)blockDim.x) {
        int i = homeList[hi];

        real xi = x[i], yi = y[i], zi = z[i];
        real aix = 0, aiy = 0, aiz = 0;

        for (int idx = c_begin; idx < c_end; ++idx) {
            int j = blockList[idx];
            if (j == i) continue;

            real dx = xi - x[j];
            real dy = yi - y[j];
            real dz = zi - z[j];
            minimum_image_d(dx, dy, dz, L, halfL);

            real r2 = dx*dx + dy*dy + dz*dz;
            if (r2 > rCut2 || r2 < tiny) continue;

            real inv_r2  = (real)1.0 / r2;
            real inv_r6  = inv_r2 * inv_r2 * inv_r2;
            real inv_r12 = inv_r6 * inv_r6;

            real coef = (real)24.0 * inv_r2 * ((real)2.0 * inv_r12 - inv_r6);

            aix += coef * dx;
            aiy += coef * dy;
            aiz += coef * dz;
        }

        ax[i] = aix;
        ay[i] = aiy;
        az[i] = aiz;
    }
}

//Diagnostics
__global__ void compute_pe_blockcentric_per_particle_kernel(
    int nb,
    const real* __restrict__ x,
    const real* __restrict__ y,
    const real* __restrict__ z,
    real L,
    real halfL,
    real rCut2,
    const int* __restrict__ blockStart,
    const int* __restrict__ blockList,
    const int* __restrict__ homeStart,
    const int* __restrict__ homeList,
    real* __restrict__ pe_i
) {
    int b = (int)blockIdx.x;
    if (b >= nb) return;

    int c_begin = blockStart[b];
    int c_end   = blockStart[b + 1];

    int h_begin = homeStart[b];
    int h_end   = homeStart[b + 1];

    if (h_begin == h_end || c_begin == c_end) return;

    const real tiny = (real)1e-12;

    const real inv_rc2  = (real)1.0 / rCut2;
    const real inv_rc6  = inv_rc2 * inv_rc2 * inv_rc2;
    const real inv_rc12 = inv_rc6 * inv_rc6;
    const real Uc = (real)4.0 * (inv_rc12 - inv_rc6);

    for (int hi = h_begin + (int)threadIdx.x; hi < h_end; hi += (int)blockDim.x) {
        int i = homeList[hi];

        real xi = x[i], yi = y[i], zi = z[i];
        real pei = 0;

        for (int idx = c_begin; idx < c_end; ++idx) {
            int j = blockList[idx];
            if (j == i) continue;

            real dx = xi - x[j];
            real dy = yi - y[j];
            real dz = zi - z[j];
            minimum_image_d(dx, dy, dz, L, halfL);

            real r2 = dx*dx + dy*dy + dz*dz;
            if (r2 > rCut2 || r2 < tiny) continue;

            real inv_r2  = (real)1.0 / r2;
            real inv_r6  = inv_r2 * inv_r2 * inv_r2;
            real inv_r12 = inv_r6 * inv_r6;

            real U = (real)4.0 * (inv_r12 - inv_r6) - Uc;
            pei += (real)0.5 * U;
        }

        pe_i[i] = pei;
    }
}

__global__ void compute_ke_p_per_particle_kernel(
    int p,
    const real* __restrict__ vx,
    const real* __restrict__ vy,
    const real* __restrict__ vz,
    real* __restrict__ ke_i,
    real* __restrict__ px_i,
    real* __restrict__ py_i,
    real* __restrict__ pz_i
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= p) return;

    real vxi = vx[i];
    real vyi = vy[i];
    real vzi = vz[i];

    ke_i[i] = (real)0.5 * (vxi*vxi + vyi*vyi + vzi*vzi);
    px_i[i] = vxi;
    py_i[i] = vyi;
    pz_i[i] = vzi;
}

//main
int main() {
    CUDA_CHECK(cudaSetDevice(0));
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    auto t0 = std::chrono::steady_clock::now();

    real V = (real)N_req / density;
    real L = std::cbrt(V);
    real halfL = (real)0.5 * L;

    int n = (int)std::ceil(std::cbrt((real)N_req));
    real a = L / (real)n;

    real rCut = std::pow((real)2.0, (real)1.0 / (real)6.0);
    real rCut2 = rCut * rCut;

    
    int kblocks = (int)std::floor(L / ((real)2.0 * rCut));
    if (kblocks < 1) kblocks = 1;
    real s = L / (real)kblocks;

    int nb = kblocks * kblocks * kblocks;

    std::vector<real> x(N_req), y(N_req), z(N_req);
    std::vector<real> vx(N_req), vy(N_req), vz(N_req);

    real Temperature0 = (real)1.0;
    real vmag0 = std::sqrt((real)3.0 * Temperature0);

    std::mt19937 rng(12345);
    std::normal_distribution<real> G((real)0.0, (real)1.0);

    int p = 0;
    real vcmx = 0, vcmy = 0, vcmz = 0;

    for (int i = 0; i < n && p < N_req; ++i) {
        for (int j = 0; j < n && p < N_req; ++j) {
            for (int k = 0; k < n && p < N_req; ++k) {
                x[p] = ((real)i + (real)0.5) * a;
                y[p] = ((real)j + (real)0.5) * a;
                z[p] = ((real)k + (real)0.5) * a;

                real gx = G(rng), gy = G(rng), gz = G(rng);
                real rr = std::sqrt(gx*gx + gy*gy + gz*gz);
                if (rr < (real)1e-12) continue;

                gx /= rr; gy /= rr; gz /= rr;
                vx[p] = vmag0 * gx;
                vy[p] = vmag0 * gy;
                vz[p] = vmag0 * gz;

                vcmx += vx[p];
                vcmy += vy[p];
                vcmz += vz[p];

                ++p;
            }
        }
    }

    x.resize(p); y.resize(p); z.resize(p);
    vx.resize(p); vy.resize(p); vz.resize(p);

    if (p == 0) {
        std::cerr << "No particles initialized.\n";
        return 1;
    }

    vcmx /= (real)p; vcmy /= (real)p; vcmz /= (real)p;
    for (int i = 0; i < p; ++i) {
        vx[i] -= vcmx;
        vy[i] -= vcmy;
        vz[i] -= vcmz;
    }

    real *d_x=nullptr, *d_y=nullptr, *d_z=nullptr;
    real *d_vx=nullptr, *d_vy=nullptr, *d_vz=nullptr;
    real *d_ax=nullptr, *d_ay=nullptr, *d_az=nullptr;
    real *d_ax_old=nullptr, *d_ay_old=nullptr, *d_az_old=nullptr;

    CUDA_CHECK(cudaMalloc(&d_x,  p*sizeof(real)));
    CUDA_CHECK(cudaMalloc(&d_y,  p*sizeof(real)));
    CUDA_CHECK(cudaMalloc(&d_z,  p*sizeof(real)));
    CUDA_CHECK(cudaMalloc(&d_vx, p*sizeof(real)));
    CUDA_CHECK(cudaMalloc(&d_vy, p*sizeof(real)));
    CUDA_CHECK(cudaMalloc(&d_vz, p*sizeof(real)));

    CUDA_CHECK(cudaMalloc(&d_ax, p*sizeof(real)));
    CUDA_CHECK(cudaMalloc(&d_ay, p*sizeof(real)));
    CUDA_CHECK(cudaMalloc(&d_az, p*sizeof(real)));

    CUDA_CHECK(cudaMalloc(&d_ax_old, p*sizeof(real)));
    CUDA_CHECK(cudaMalloc(&d_ay_old, p*sizeof(real)));
    CUDA_CHECK(cudaMalloc(&d_az_old, p*sizeof(real)));

    int *d_blockCount=nullptr, *d_blockStart=nullptr, *d_blockWrite=nullptr, *d_blockList=nullptr;
    int *d_particleHome=nullptr;

    CUDA_CHECK(cudaMalloc(&d_blockCount, nb*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_blockStart, (nb+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_blockWrite, nb*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_particleHome, p*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_blockList, (size_t)DUP_MAX * (size_t)p * sizeof(int)));

    int *d_homeCount=nullptr, *d_homeStart=nullptr, *d_homeWrite=nullptr, *d_homeList=nullptr;
    CUDA_CHECK(cudaMalloc(&d_homeCount, nb*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_homeStart, (nb+1)*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_homeWrite, nb*sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_homeList, p*sizeof(int)));

    real *d_pe_i=nullptr, *d_pe_sum=nullptr;
    real *d_ke_i=nullptr, *d_px_i=nullptr, *d_py_i=nullptr, *d_pz_i=nullptr;
    real *d_ke_sum=nullptr, *d_px_sum=nullptr, *d_py_sum=nullptr, *d_pz_sum=nullptr;

    CUDA_CHECK(cudaMalloc(&d_pe_i,   p*sizeof(real)));
    CUDA_CHECK(cudaMalloc(&d_pe_sum, sizeof(real)));

    CUDA_CHECK(cudaMalloc(&d_ke_i, p*sizeof(real)));
    CUDA_CHECK(cudaMalloc(&d_px_i, p*sizeof(real)));
    CUDA_CHECK(cudaMalloc(&d_py_i, p*sizeof(real)));
    CUDA_CHECK(cudaMalloc(&d_pz_i, p*sizeof(real)));

    CUDA_CHECK(cudaMalloc(&d_ke_sum, sizeof(real)));
    CUDA_CHECK(cudaMalloc(&d_px_sum, sizeof(real)));
    CUDA_CHECK(cudaMalloc(&d_py_sum, sizeof(real)));
    CUDA_CHECK(cudaMalloc(&d_pz_sum, sizeof(real)));

    CUDA_CHECK(cudaMemcpyAsync(d_x,  x.data(),  p*sizeof(real), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_y,  y.data(),  p*sizeof(real), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_z,  z.data(),  p*sizeof(real), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_vx, vx.data(), p*sizeof(real), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_vy, vy.data(), p*sizeof(real), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_vz, vz.data(), p*sizeof(real), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    void* d_scanTemp=nullptr;
    size_t scanTempBytes=0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, scanTempBytes, d_blockCount, d_blockStart, nb, stream));
    CUDA_CHECK(cudaMalloc(&d_scanTemp, scanTempBytes));

    void* d_scanTempHome=nullptr;
    size_t scanTempHomeBytes=0;
    CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, scanTempHomeBytes, d_homeCount, d_homeStart, nb, stream));
    CUDA_CHECK(cudaMalloc(&d_scanTempHome, scanTempHomeBytes));

    void* d_reduceTemp=nullptr;
    size_t reduceTempBytes = 0, tmpBytes = 0;
    CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, tmpBytes, d_pe_i, d_pe_sum, p, stream));
    reduceTempBytes = std::max(reduceTempBytes, tmpBytes);
    CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, tmpBytes, d_ke_i, d_ke_sum, p, stream));
    reduceTempBytes = std::max(reduceTempBytes, tmpBytes);
    CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, tmpBytes, d_px_i, d_px_sum, p, stream));
    reduceTempBytes = std::max(reduceTempBytes, tmpBytes);
    CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, tmpBytes, d_py_i, d_py_sum, p, stream));
    reduceTempBytes = std::max(reduceTempBytes, tmpBytes);
    CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, tmpBytes, d_pz_i, d_pz_sum, p, stream));
    reduceTempBytes = std::max(reduceTempBytes, tmpBytes);
    CUDA_CHECK(cudaMalloc(&d_reduceTemp, reduceTempBytes));

    int threads = 256;
    int gridP = (p + threads - 1) / threads;
    int gridB = (nb + threads - 1) / threads;

    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    double ms_build = 0.0;
    double ms_force = 0.0;
    double ms_diag  = 0.0;

    auto time_begin = [&](){ CUDA_CHECK(cudaEventRecord(ev0, stream)); };
    auto time_end_add = [&](double& acc_ms){
        CUDA_CHECK(cudaEventRecord(ev1, stream));
        CUDA_CHECK(cudaEventSynchronize(ev1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
        acc_ms += (double)ms;
    };

    auto build_lists_gpu = [&](){
        time_begin();

        CUDA_CHECK(cudaMemsetAsync(d_blockCount, 0, nb*sizeof(int), stream));

        count_blocks_dup_kernel<<<gridP, threads, 0, stream>>>(
            p, d_x, d_y, d_z, kblocks, s, rCut, d_blockCount, d_particleHome
        );
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            d_scanTemp, scanTempBytes, d_blockCount, d_blockStart, nb, stream
        ));

        set_blockstart_last_from_counts_kernel<<<1,1,0,stream>>>(
            nb, d_blockStart, d_blockCount, d_blockStart
        );
        CUDA_CHECK(cudaGetLastError());

        init_write_kernel<<<gridB, threads, 0, stream>>>(nb, d_blockStart, d_blockWrite);
        CUDA_CHECK(cudaGetLastError());

        scatter_blocks_dup_kernel<<<gridP, threads, 0, stream>>>(
            p, d_x, d_y, d_z, kblocks, s, rCut, d_blockWrite, d_blockList
        );
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cudaMemsetAsync(d_homeCount, 0, nb*sizeof(int), stream));
        count_home_kernel<<<gridP, threads, 0, stream>>>(p, d_particleHome, d_homeCount);
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
            d_scanTempHome, scanTempHomeBytes, d_homeCount, d_homeStart, nb, stream
        ));

        set_home_last_kernel<<<1,1,0,stream>>>(nb, p, d_homeStart);
        CUDA_CHECK(cudaGetLastError());

        init_write_kernel<<<gridB, threads, 0, stream>>>(nb, d_homeStart, d_homeWrite);
        CUDA_CHECK(cudaGetLastError());

        scatter_home_kernel<<<gridP, threads, 0, stream>>>(p, d_particleHome, d_homeWrite, d_homeList);
        CUDA_CHECK(cudaGetLastError());

        time_end_add(ms_build);
    };

    auto compute_forces_gpu = [&](){
        time_begin();
        compute_forces_blockcentric_kernel<<<nb, 256, 0, stream>>>(
            nb,
            d_ax, d_ay, d_az,
            d_x, d_y, d_z,
            L, halfL, rCut2,
            d_blockStart, d_blockList,
            d_homeStart, d_homeList
        );
        CUDA_CHECK(cudaGetLastError());
        time_end_add(ms_force);
    };

    auto compute_pe_gpu = [&](real& PE_out){
        CUDA_CHECK(cudaMemsetAsync(d_pe_i, 0, p*sizeof(real), stream));

        compute_pe_blockcentric_per_particle_kernel<<<nb, 256, 0, stream>>>(
            nb,
            d_x, d_y, d_z,
            L, halfL, rCut2,
            d_blockStart, d_blockList,
            d_homeStart, d_homeList,
            d_pe_i
        );
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cub::DeviceReduce::Sum(
            d_reduceTemp, reduceTempBytes,
            d_pe_i, d_pe_sum, p, stream
        ));
        CUDA_CHECK(cudaMemcpyAsync(&PE_out, d_pe_sum, sizeof(real), cudaMemcpyDeviceToHost, stream));
    };

    auto compute_ke_p_gpu = [&](real& KE_out, real& Px_out, real& Py_out, real& Pz_out){
        compute_ke_p_per_particle_kernel<<<gridP, threads, 0, stream>>>(
            p, d_vx, d_vy, d_vz, d_ke_i, d_px_i, d_py_i, d_pz_i
        );
        CUDA_CHECK(cudaGetLastError());

        CUDA_CHECK(cub::DeviceReduce::Sum(d_reduceTemp, reduceTempBytes, d_ke_i, d_ke_sum, p, stream));
        CUDA_CHECK(cub::DeviceReduce::Sum(d_reduceTemp, reduceTempBytes, d_px_i, d_px_sum, p, stream));
        CUDA_CHECK(cub::DeviceReduce::Sum(d_reduceTemp, reduceTempBytes, d_py_i, d_py_sum, p, stream));
        CUDA_CHECK(cub::DeviceReduce::Sum(d_reduceTemp, reduceTempBytes, d_pz_i, d_pz_sum, p, stream));

        CUDA_CHECK(cudaMemcpyAsync(&KE_out, d_ke_sum, sizeof(real), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(&Px_out, d_px_sum, sizeof(real), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(&Py_out, d_py_sum, sizeof(real), cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(&Pz_out, d_pz_sum, sizeof(real), cudaMemcpyDeviceToHost, stream));
    };

    build_lists_gpu();
    compute_forces_gpu();
    CUDA_CHECK(cudaStreamSynchronize(stream));

    real KE=0, PE=0, Temp=0, Px=0, Py=0, Pz=0;
    time_begin();
    compute_ke_p_gpu(KE, Px, Py, Pz);
    compute_pe_gpu(PE);
    CUDA_CHECK(cudaStreamSynchronize(stream));
    time_end_add(ms_diag);

    Temp = (p > 0) ? ((real)2.0 * KE / ((real)3.0 * (real)p)) : 0;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "N=" << p
              << "  L=" << L
              << "  rCut=" << rCut
              << "  kblocks=" << kblocks
              << "  s=" << s
              << "  nb=" << nb
              << "\n";
    std::cout << "Initial: KE=" << KE << "  PE=" << PE << "  E=" << (KE+PE)
              << "  T=" << Temp
              << "  |P|=" << std::sqrt(Px*Px+Py*Py+Pz*Pz) << "\n\n";

    for (int step = 0; step < n_steps; ++step) {
        copy_accel_kernel<<<gridP, threads, 0, stream>>>(p, d_ax, d_ay, d_az, d_ax_old, d_ay_old, d_az_old);
        CUDA_CHECK(cudaGetLastError());

        verlet_position_kernel<<<gridP, threads, 0, stream>>>(
            p, d_x, d_y, d_z,
            d_vx, d_vy, d_vz,
            d_ax, d_ay, d_az,
            dt, L
        );
        CUDA_CHECK(cudaGetLastError());

        build_lists_gpu();
        compute_forces_gpu();

        verlet_velocity_kernel<<<gridP, threads, 0, stream>>>(
            p, d_vx, d_vy, d_vz,
            d_ax_old, d_ay_old, d_az_old,
            d_ax, d_ay, d_az,
            dt
        );
        CUDA_CHECK(cudaGetLastError());

        if ((step % save_every) == 0 || step == n_steps - 1) {
            time_begin();
            compute_ke_p_gpu(KE, Px, Py, Pz);
            compute_pe_gpu(PE);
            CUDA_CHECK(cudaStreamSynchronize(stream));
            time_end_add(ms_diag);

            Temp = (p > 0) ? ((real)2.0 * KE / ((real)3.0 * (real)p)) : 0;

            real Etot = KE + PE;
            real Pmag = std::sqrt(Px*Px + Py*Py + Pz*Pz);

            std::cout << "step=" << step
                      << "  t=" << (step * dt)
                      << "  KE=" << KE
                      << "  PE=" << PE
                      << "  E="  << Etot
                      << "  T="  << Temp
                      << "  |P|=" << Pmag
                      << "\n";
        }
    }

    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::cout << "\n--- GPU timing (accumulated) ---\n";
    std::cout << "build lists total (ms):      " << ms_build << "\n";
    std::cout << "forces total (ms):           " << ms_force << "\n";
    std::cout << "diagnostics total (ms):      " << ms_diag << " (only print steps)\n";
    double steps_d = (double)n_steps;
    std::cout << "avg build per step (ms):     " << (ms_build / steps_d) << "\n";
    std::cout << "avg forces per step (ms):    " << (ms_force / steps_d) << "\n";

    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));

    auto t1 = std::chrono::steady_clock::now();
    real seconds = (real)std::chrono::duration<double>(t1 - t0).count();
    std::cout << "\nRuntime (s): " << seconds << "\n";

    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_vx));
    CUDA_CHECK(cudaFree(d_vy));
    CUDA_CHECK(cudaFree(d_vz));
    CUDA_CHECK(cudaFree(d_ax));
    CUDA_CHECK(cudaFree(d_ay));
    CUDA_CHECK(cudaFree(d_az));
    CUDA_CHECK(cudaFree(d_ax_old));
    CUDA_CHECK(cudaFree(d_ay_old));
    CUDA_CHECK(cudaFree(d_az_old));

    CUDA_CHECK(cudaFree(d_blockCount));
    CUDA_CHECK(cudaFree(d_blockStart));
    CUDA_CHECK(cudaFree(d_blockWrite));
    CUDA_CHECK(cudaFree(d_blockList));
    CUDA_CHECK(cudaFree(d_particleHome));

    CUDA_CHECK(cudaFree(d_homeCount));
    CUDA_CHECK(cudaFree(d_homeStart));
    CUDA_CHECK(cudaFree(d_homeWrite));
    CUDA_CHECK(cudaFree(d_homeList));

    CUDA_CHECK(cudaFree(d_pe_i));
    CUDA_CHECK(cudaFree(d_pe_sum));

    CUDA_CHECK(cudaFree(d_ke_i));
    CUDA_CHECK(cudaFree(d_px_i));
    CUDA_CHECK(cudaFree(d_py_i));
    CUDA_CHECK(cudaFree(d_pz_i));

    CUDA_CHECK(cudaFree(d_ke_sum));
    CUDA_CHECK(cudaFree(d_px_sum));
    CUDA_CHECK(cudaFree(d_py_sum));
    CUDA_CHECK(cudaFree(d_pz_sum));

    CUDA_CHECK(cudaFree(d_scanTemp));
    CUDA_CHECK(cudaFree(d_scanTempHome));
    CUDA_CHECK(cudaFree(d_reduceTemp));

    CUDA_CHECK(cudaStreamDestroy(stream));
    return 0;
}