#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include <numeric>
#include <iomanip>
#include <algorithm>

using real = double;

const real dt = (real)0.001;
const int  n_steps = 1000;
const int  save_every = 10;

static inline int wrap_int(int q, int k) {
    q %= k;
    if (q < 0) q += k;
    return q;
}

static inline int block_index(int bx, int by, int bz, int k) {
    return (bx * k + by) * k + bz;
}

static inline void wrap_position(real& x, real L) {
    x -= L * std::floor(x / L);
}

static inline void minimum_image(real& dx, real& dy, real& dz, real L, real halfL) {
    if (dx >  halfL) dx -= L;
    if (dx < -halfL) dx += L;
    if (dy >  halfL) dy -= L;
    if (dy < -halfL) dy += L;
    if (dz >  halfL) dz -= L;
    if (dz < -halfL) dz += L;
}

static inline int axis_blocks(real coord, real s, real rc, int kblocks, int out[2]) {
    if (kblocks <= 1) {
        out[0] = 0;
        return 1;
    }

    int q = (int)std::floor(coord / s);
    if (q < 0) q = 0;
    if (q >= kblocks) q = kblocks - 1;

    real local = coord - (real)q * s;

    out[0] = q;

    if (local < rc) {
        out[1] = wrap_int(q - 1, kblocks);
        return 2;
    }

    if (local >= (s - rc)) {
        out[1] = wrap_int(q + 1, kblocks);
        return 2;
    }

    return 1;
}

void build_block_list(
    int p,
    const std::vector<real>& x,
    const std::vector<real>& y,
    const std::vector<real>& z,
    int kblocks,
    real s,
    real rc,
    std::vector<int>& blockCount,
    std::vector<int>& blockStart,
    std::vector<int>& blockList,
    std::vector<int>& particleBlock
) {
    const int nb = kblocks * kblocks * kblocks;

    blockCount.assign(nb, 0);
    particleBlock.assign(p, 0);

    for (int i = 0; i < p; ++i) {
        int xs[2], ys[2], zs[2];

        int nx = axis_blocks(x[i], s, rc, kblocks, xs);
        int ny = axis_blocks(y[i], s, rc, kblocks, ys);
        int nz = axis_blocks(z[i], s, rc, kblocks, zs);

        particleBlock[i] = block_index(xs[0], ys[0], zs[0], kblocks);

        for (int ix = 0; ix < nx; ++ix)
            for (int iy = 0; iy < ny; ++iy)
                for (int iz = 0; iz < nz; ++iz) {
                    int b = block_index(xs[ix], ys[iy], zs[iz], kblocks);
                    blockCount[b] += 1;
                }
    }

    blockStart.assign(nb + 1, 0);
    for (int b = 0; b < nb; ++b)
        blockStart[b + 1] = blockStart[b] + blockCount[b];

    const int m = blockStart[nb];

    blockList.assign(m, 0);
    std::vector<int> writePtr = blockStart;

    for (int i = 0; i < p; ++i) {
        int xs[2], ys[2], zs[2];

        int nx = axis_blocks(x[i], s, rc, kblocks, xs);
        int ny = axis_blocks(y[i], s, rc, kblocks, ys);
        int nz = axis_blocks(z[i], s, rc, kblocks, zs);

        for (int ix = 0; ix < nx; ++ix)
            for (int iy = 0; iy < ny; ++iy)
                for (int iz = 0; iz < nz; ++iz) {
                    int b = block_index(xs[ix], ys[iy], zs[iz], kblocks);
                    int pos = writePtr[b]++;
                    blockList[pos] = i;
                }
    }
}

void build_home_list(
    int p,
    int nb,
    const std::vector<int>& particleBlock,
    std::vector<int>& homeCount,
    std::vector<int>& homeStart,
    std::vector<int>& homeList
) {
    homeCount.assign(nb, 0);

    for (int i = 0; i < p; ++i)
        homeCount[particleBlock[i]]++;

    homeStart.assign(nb + 1, 0);
    for (int b = 0; b < nb; ++b)
        homeStart[b + 1] = homeStart[b] + homeCount[b];

    homeList.assign(p, 0);
    std::vector<int> writePtr = homeStart;

    for (int i = 0; i < p; ++i) {
        int b = particleBlock[i];
        homeList[writePtr[b]++] = i;
    }
}

void compute_forces_blockwise(
    int p,
    std::vector<real>& ax,
    std::vector<real>& ay,
    std::vector<real>& az,
    const std::vector<real>& x,
    const std::vector<real>& y,
    const std::vector<real>& z,
    int nb,
    real L,
    real halfL,
    real rCut2,
    const std::vector<int>& blockStart,
    const std::vector<int>& blockList,
    const std::vector<int>& homeStart,
    const std::vector<int>& homeList,
    real& potential_energy_out
) {
    const real tiny = (real)1e-12;

    const real inv_rc2  = (real)1.0 / rCut2;
    const real inv_rc6  = inv_rc2 * inv_rc2 * inv_rc2;
    const real inv_rc12 = inv_rc6 * inv_rc6;
    const real Uc = (real)4.0 * (inv_rc12 - inv_rc6);

    std::fill(ax.begin(), ax.end(), (real)0.0);
    std::fill(ay.begin(), ay.end(), (real)0.0);
    std::fill(az.begin(), az.end(), (real)0.0);
    potential_energy_out = (real)0.0;

    for (int b = 0; b < nb; ++b) {
        int c_begin = blockStart[b];
        int c_end   = blockStart[b + 1];

        int h_begin = homeStart[b];
        int h_end   = homeStart[b + 1];

        if (h_begin == h_end || c_begin == c_end)
            continue;

        for (int hi = h_begin; hi < h_end; ++hi) {
            int i = homeList[hi];

            real aix = 0, aiy = 0, aiz = 0;
            real pei = 0;

            for (int idx = c_begin; idx < c_end; ++idx) {
                int j = blockList[idx];
                if (j == i) continue;

                real dx = x[i] - x[j];
                real dy = y[i] - y[j];
                real dz = z[i] - z[j];
                minimum_image(dx, dy, dz, L, halfL);

                real r2 = dx*dx + dy*dy + dz*dz;
                if (r2 > rCut2 || r2 < tiny)
                    continue;

                real inv_r2  = (real)1.0 / r2;
                real inv_r6  = inv_r2 * inv_r2 * inv_r2;
                real inv_r12 = inv_r6 * inv_r6;

                real coef = (real)24.0 * inv_r2 * ((real)2.0 * inv_r12 - inv_r6);

                aix += coef * dx;
                aiy += coef * dy;
                aiz += coef * dz;

                real U = (real)4.0 * (inv_r12 - inv_r6) - Uc;
                pei += (real)0.5 * U;
            }

            ax[i] = aix;
            ay[i] = aiy;
            az[i] = aiz;
            potential_energy_out += pei;
        }
    }
}

static inline void compute_kinetic_momentum_temperature(
    int p,
    const std::vector<real>& vx,
    const std::vector<real>& vy,
    const std::vector<real>& vz,
    real& kinetic_out,
    real& temperature_out,
    real& px_out,
    real& py_out,
    real& pz_out
) {
    real K = 0, Px = 0, Py = 0, Pz = 0;

    for (int i = 0; i < p; ++i) {
        K += (real)0.5 * (vx[i]*vx[i] + vy[i]*vy[i] + vz[i]*vz[i]);
        Px += vx[i];
        Py += vy[i];
        Pz += vz[i];
    }

    kinetic_out = K;
    temperature_out = (p > 0) ? ((real)2.0 * K / ((real)3.0 * (real)p)) : 0;
    px_out = Px; py_out = Py; pz_out = Pz;
}

int main() {
    auto t0 = std::chrono::steady_clock::now();

    int N = 1000000;
    real density = (real)0.8;
    real V = (real)N / density;
    real L = std::cbrt(V);
    real halfL = (real)0.5 * L;

    int n = (int)std::ceil(std::cbrt((real)N));
    real a = L / (real)n;

    real rCut = std::pow((real)2.0, (real)1.0 / (real)6.0);
    real rCut2 = rCut * rCut;

    int kblocks = (int)std::floor(L / ((real)2.0 * rCut));
    if (kblocks < 1) kblocks = 1;
    real s = L / (real)kblocks;

    const int nb = kblocks * kblocks * kblocks;

    std::vector<real> x(N), y(N), z(N);
    std::vector<real> vx(N), vy(N), vz(N);
    std::vector<real> vxh(N), vyh(N), vzh(N);
    std::vector<real> ax(N), ay(N), az(N);

    real Temperature0 = (real)1.0;
    real vmag0 = std::sqrt((real)3.0 * Temperature0);

    std::mt19937 rng(12345);
    std::normal_distribution<real> G((real)0.0, (real)1.0);

    int p = 0;
    real vcmx = 0, vcmy = 0, vcmz = 0;

    for (int i = 0; i < n && p < N; ++i)
        for (int j = 0; j < n && p < N; ++j)
            for (int k = 0; k < n && p < N; ++k) {
                x[p] = ((real)i + (real)0.5) * a;
                y[p] = ((real)j + (real)0.5) * a;
                z[p] = ((real)k + (real)0.5) * a;

                real gx = G(rng), gy = G(rng), gz = G(rng);
                real r = std::sqrt(gx*gx + gy*gy + gz*gz);
                if (r < (real)1e-12) continue;

                gx /= r; gy /= r; gz /= r;
                vx[p] = vmag0 * gx;
                vy[p] = vmag0 * gy;
                vz[p] = vmag0 * gz;

                vcmx += vx[p];
                vcmy += vy[p];
                vcmz += vz[p];
                ++p;
            }

    x.resize(p); y.resize(p); z.resize(p);
    vx.resize(p); vy.resize(p); vz.resize(p);
    vxh.resize(p); vyh.resize(p); vzh.resize(p);
    ax.resize(p); ay.resize(p); az.resize(p);

    vcmx /= (real)p; vcmy /= (real)p; vcmz /= (real)p;
    for (int i = 0; i < p; ++i) {
        vx[i] -= vcmx;
        vy[i] -= vcmy;
        vz[i] -= vcmz;
    }

    std::vector<int> blockCount, blockStart, blockList, particleBlock;
    std::vector<int> homeCount, homeStart, homeList;

    build_block_list(p, x, y, z, kblocks, s, rCut,
                     blockCount, blockStart, blockList, particleBlock);
    build_home_list(p, nb, particleBlock,
                    homeCount, homeStart, homeList);

    real PE = 0;
    compute_forces_blockwise(p, ax, ay, az, x, y, z,
                             nb, L, halfL, rCut2,
                             blockStart, blockList,
                             homeStart, homeList, PE);

    for (int i = 0; i < p; ++i) {
        vxh[i] = vx[i] + (real)0.5 * ax[i] * dt;
        vyh[i] = vy[i] + (real)0.5 * ay[i] * dt;
        vzh[i] = vz[i] + (real)0.5 * az[i] * dt;
    }

    real KE=0, T=0, Px=0, Py=0, Pz=0;
    compute_kinetic_momentum_temperature(p, vx, vy, vz, KE, T, Px, Py, Pz);

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "N=" << p
              << "  L=" << L
              << "  rCut=" << rCut
              << "  kblocks=" << kblocks
              << "  s=" << s
              << "\n";
    std::cout << "Initial: KE=" << KE << "  PE=" << PE << "  E=" << (KE+PE)
              << "  T=" << T
              << "  |P|=" << std::sqrt(Px*Px+Py*Py+Pz*Pz) << "\n\n";

    for (int step = 0; step < n_steps; ++step) {
        for (int i = 0; i < p; ++i) {
            x[i] += vxh[i] * dt;
            y[i] += vyh[i] * dt;
            z[i] += vzh[i] * dt;

            wrap_position(x[i], L);
            wrap_position(y[i], L);
            wrap_position(z[i], L);
        }

        build_block_list(p, x, y, z, kblocks, s, rCut,
                         blockCount, blockStart, blockList, particleBlock);
        build_home_list(p, nb, particleBlock,
                        homeCount, homeStart, homeList);

        compute_forces_blockwise(p, ax, ay, az, x, y, z,
                                 nb, L, halfL, rCut2,
                                 blockStart, blockList,
                                 homeStart, homeList, PE);

        for (int i = 0; i < p; ++i) {
            vxh[i] += ax[i] * dt;
            vyh[i] += ay[i] * dt;
            vzh[i] += az[i] * dt;

            vx[i] = vxh[i] - (real)0.5 * ax[i] * dt;
            vy[i] = vyh[i] - (real)0.5 * ay[i] * dt;
            vz[i] = vzh[i] - (real)0.5 * az[i] * dt;
        }

        if (((step + 1) % save_every) == 0 || step == n_steps - 1) {
            compute_kinetic_momentum_temperature(p, vx, vy, vz,
                                                 KE, T, Px, Py, Pz);

            std::cout << "step=" << (step + 1)
                      << "  t=" << ((step + 1) * dt)
                      << "  KE=" << KE
                      << "  PE=" << PE
                      << "  E="  << (KE + PE)
                      << "  T="  << T
                      << "  |P|=" << std::sqrt(Px*Px + Py*Py + Pz*Pz)
                      << "\n";
        }
    }

    compute_kinetic_momentum_temperature(p, vx, vy, vz, KE, T, Px, Py, Pz);

    auto t1 = std::chrono::steady_clock::now();
    real seconds = (real)std::chrono::duration<double>(t1 - t0).count();

    std::cout << "\nFinal: KE=" << KE << "  PE=" << PE << "  E=" << (KE+PE)
              << "  T=" << T
              << "  |P|=" << std::sqrt(Px*Px+Py*Py+Pz*Pz) << "\n";
    std::cout << "Runtime (s): " << seconds << "\n";

    return 0;
}