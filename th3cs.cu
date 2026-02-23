// th3cs.cu
// nvcc -O3 -std=c++17 th3cs.cu 4splat.c -o th3cs -lineinfo

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

#ifndef __CUDACC__
#define __host__
#define __device__
#define __global__
#endif

// -------------------------------------------------------------------------
// 4Splat C API Declarations
// -------------------------------------------------------------------------
extern "C" {
typedef struct {
  float mu_x, sigma_x, mu_y, sigma_y, mu_z, sigma_z, mu_t, sigma_t, r, g, b,
      alpha;
} Splat4D;

typedef struct {
  uint32_t magic;
  uint8_t version[4];
  uint32_t width, height, depth, frames;
  uint32_t pSize;
  uint32_t flags;
} Splat4DHeader;

typedef struct {
  Splat4D *palette;
} Splat4DPalette;
typedef struct {
  uint64_t *index;
} Splat4DIndex;
typedef struct {
  uint32_t checksum;
  uint64_t idxoffset;
  uint32_t end;
} Splat4DFooter;

typedef struct {
  Splat4DHeader header;
  Splat4DPalette palette;
  Splat4DIndex index;
  Splat4DFooter footer;
} Splat4DVideo;

Splat4D create_splat4D(float mu_x, float sigma_x, float mu_y, float sigma_y,
                       float mu_z, float sigma_z, float mu_t, float sigma_t,
                       float r, float g, float b, float alpha);
Splat4DHeader create_splat4DHeader(uint32_t width, uint32_t height,
                                   uint32_t depth, uint32_t frames,
                                   uint32_t pSize, uint32_t flags);
Splat4DVideo create_splat4DVideo(Splat4DHeader header, Splat4D *splats,
                                 uint64_t *idxs);
bool write_splat4DVideo(FILE *fp, Splat4DVideo *v);
}

// -------------------------------------------------------------------------
// Simulation Physics and Kernels
// -------------------------------------------------------------------------

struct Params {
  int nx, ny, nz;
  float dx, dy, dz;
  float cfl;
  float u_ref;
  float R;
  float gamma_floor;
  float Twall;
  float tau_vib;
  float theta_v;
  float sdf_cx, sdf_cy, sdf_cz;
  float sdf_r;
  float inflow_r;
  float inflow_p;
  float inflow_u;
  float inflow_v;
  float inflow_w;
  int sponge_n;
  float sponge_strength;
  int sponge_out_n;
  float sponge_out_strength;
};

struct Cons {
  float r, mx, my, mz, Et, Ev;
};
struct Prim {
  float r, u, v, w, p, T, ev, Tv;
};

constexpr float RHO_P_FLOOR = 1e-30f;
constexpr float THERMAL_ENERGY_FLOOR = 1e-12f;
constexpr float DENOM_EPS = 1e-12f;
constexpr float NEWTON_TEMP_FLOOR = 1e-6f;
constexpr float WENO_EPS = 1e-6f;
constexpr float TAU_VIB_MIN = 1e-9f;
constexpr int WENO_HALO = 3;

// Error checking helper
static inline void ck(cudaError_t e, const char *m) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", m, cudaGetErrorString(e));
    std::exit(1);
  }
}

__device__ __constant__ Params P;

__device__ inline float sgnf(float x) { return (x > 0.f) - (x < 0.f); }
__device__ inline unsigned int f2u(float x) { return __float_as_uint(x); }
__device__ inline float u2f(unsigned int x) { return __uint_as_float(x); }

__device__ inline float rho_from_xi(float xi) { return __expf(xi); }
__device__ inline float p_from_lambda(float l) { return __expf(l); }
__device__ inline float evib_from_zeta(float z) { return __expf(z); }
__device__ inline float sinhf_dev(float x) { return sinhf(x); }
__device__ inline float asinhf_dev(float x) {
  float ax = fabsf(x);
  float t = logf(ax + sqrtf(ax * ax + 1.0f));
  return copysignf(t, x);
}

__device__ inline float vel_from_phi(float phi) {
  return P.u_ref * sinhf_dev(phi);
}
__device__ inline float phi_from_vel(float u) {
  return asinhf_dev(u / P.u_ref);
}
__device__ inline float clampf(float x, float a, float b) {
  return fminf(fmaxf(x, a), b);
}
__device__ inline float minmod(float a, float b) {
  return (a * b <= 0.f) ? 0.f : sgnf(a) * fminf(fabsf(a), fabsf(b));
}
__device__ inline float max3(float a, float b, float c) {
  return fmaxf(a, fmaxf(b, c));
}
__device__ inline float signed_denom_guard(float x) {
  float ax = fabsf(x);
  return copysignf(fmaxf(ax, DENOM_EPS), x);
}

__device__ inline int idx3(int x, int y, int z) {
  return (z * P.ny + y) * P.nx + x;
}
__device__ inline int wrapi(int i, int n) {
  i %= n;
  return (i < 0) ? i + n : i;
}

__device__ inline float xi_from_rho(float r) {
  return __logf(fmaxf(r, RHO_P_FLOOR));
}
__device__ inline float lambda_from_p(float p) {
  return __logf(fmaxf(p, RHO_P_FLOOR));
}
__device__ inline float zeta_from_evib(float e) {
  return __logf(fmaxf(e, RHO_P_FLOOR));
}

__device__ inline float sdf_sphere(float x, float y, float z) {
  float dx = x - P.sdf_cx;
  float dy = y - P.sdf_cy;
  float dz = z - P.sdf_cz;
  return sqrtf(dx * dx + dy * dy + dz * dz) - P.sdf_r;
}

__device__ inline bool cell_is_solid(const uint8_t *solid, int x, int y,
                                     int z) {
  if (x >= 0 && x < P.nx && y >= 0 && y < P.ny && z >= 0 && z < P.nz)
    return solid[idx3(x, y, z)] != 0;
  float X = (x + 0.5f) * P.dx;
  float Y = (y + 0.5f) * P.dy;
  float Z = (z + 0.5f) * P.dz;
  return sdf_sphere(X, Y, Z) < 0.f;
}

__device__ inline float Tv_from_evib_seed(float evib, float Tseed) {
  float Tv = fmaxf(P.Twall, fmaxf(Tseed, NEWTON_TEMP_FLOOR));
#pragma unroll
  for (int k = 0; k < 3; k++) {
    float a = P.theta_v / fmaxf(Tv, NEWTON_TEMP_FLOOR);
    float ea = __expf(a);
    float denom = fmaxf(ea - 1.f, NEWTON_TEMP_FLOOR);
    float f = (P.R * P.theta_v) / denom - evib;
    float df =
        (P.R * P.theta_v) * (ea * (P.theta_v / (Tv * Tv))) / (denom * denom);
    Tv = fmaxf(NEWTON_TEMP_FLOOR, Tv - f / fmaxf(df, DENOM_EPS));
  }
  return Tv;
}

__device__ inline float evib_eq(float T) {
  float a = P.theta_v / fmaxf(T, NEWTON_TEMP_FLOOR);
  float ea = __expf(a);
  float denom = fmaxf(ea - 1.f, NEWTON_TEMP_FLOOR);
  return (P.R * P.theta_v) / denom;
}

__device__ inline Prim log_to_prim_fast(float xi, float phix, float phiy,
                                        float phiz, float lam, float zet) {
  Prim q;
  q.r = rho_from_xi(xi);
  q.u = vel_from_phi(phix);
  q.v = vel_from_phi(phiy);
  q.w = vel_from_phi(phiz);
  q.p = p_from_lambda(lam);
  q.ev = evib_from_zeta(zet);
  q.T = q.p / (q.r * P.R);
  q.Tv = 0.f;
  return q;
}

__device__ inline Prim log_to_prim_full(float xi, float phix, float phiy,
                                        float phiz, float lam, float zet) {
  Prim q = log_to_prim_fast(xi, phix, phiy, phiz, lam, zet);
  q.Tv = Tv_from_evib_seed(q.ev, q.T);
  return q;
}

__device__ inline Cons prim_to_cons(const Prim &q) {
  Cons U;
  U.r = q.r;
  U.mx = q.r * q.u;
  U.my = q.r * q.v;
  U.mz = q.r * q.w;
  float ke = 0.5f * (q.u * q.u + q.v * q.v + q.w * q.w);
  float e_th = q.p / fmaxf((P.gamma_floor - 1.f) * q.r, RHO_P_FLOOR);
  U.Ev = q.r * q.ev;
  U.Et = q.r * (ke + e_th + q.ev);
  return U;
}

__device__ inline Prim cons_to_prim(const Cons &U) {
  Prim q;
  q.r = fmaxf(U.r, RHO_P_FLOOR);
  q.u = U.mx / q.r;
  q.v = U.my / q.r;
  q.w = U.mz / q.r;
  float ke = 0.5f * (q.u * q.u + q.v * q.v + q.w * q.w);
  float ev = fmaxf(U.Ev / q.r, 0.f);
  float e_tot = U.Et / q.r;
  float e_th = fmaxf(e_tot - ke - ev, THERMAL_ENERGY_FLOOR);
  q.p = fmaxf((P.gamma_floor - 1.f) * q.r * e_th, RHO_P_FLOOR);
  q.ev = ev;
  q.T = q.p / (q.r * P.R);
  q.Tv = Tv_from_evib_seed(q.ev, q.T);
  return q;
}

__device__ inline float soundspeed(const Prim &q) {
  return sqrtf(fmaxf(P.gamma_floor * q.p / q.r, DENOM_EPS));
}

__device__ inline Cons flux_x(const Prim &q) {
  Cons F;
  float u = q.u;
  float H = (q.p / q.r) + (0.5f * (q.u * q.u + q.v * q.v + q.w * q.w) + q.ev) +
            q.p / fmaxf((P.gamma_floor - 1.f) * q.r, RHO_P_FLOOR);
  F.r = q.r * u;
  F.mx = q.r * q.u * u + q.p;
  F.my = q.r * q.v * u;
  F.mz = q.r * q.w * u;
  F.Et = q.r * H * u;
  F.Ev = q.r * q.ev * u;
  return F;
}

__device__ inline Cons flux_y(const Prim &q) {
  Cons F;
  float v = q.v;
  float H = (q.p / q.r) + (0.5f * (q.u * q.u + q.v * q.v + q.w * q.w) + q.ev) +
            q.p / fmaxf((P.gamma_floor - 1.f) * q.r, RHO_P_FLOOR);
  F.r = q.r * v;
  F.mx = q.r * q.u * v;
  F.my = q.r * q.v * v + q.p;
  F.mz = q.r * q.w * v;
  F.Et = q.r * H * v;
  F.Ev = q.r * q.ev * v;
  return F;
}

__device__ inline Cons flux_z(const Prim &q) {
  Cons F;
  float w = q.w;
  float H = (q.p / q.r) + (0.5f * (q.u * q.u + q.v * q.v + q.w * q.w) + q.ev) +
            q.p / fmaxf((P.gamma_floor - 1.f) * q.r, RHO_P_FLOOR);
  F.r = q.r * w;
  F.mx = q.r * q.u * w;
  F.my = q.r * q.v * w;
  F.mz = q.r * q.w * w + q.p;
  F.Et = q.r * H * w;
  F.Ev = q.r * q.ev * w;
  return F;
}

__device__ inline float axis_velocity(const Prim &q, int axis) {
  if (axis == 0)
    return q.u;
  if (axis == 1)
    return q.v;
  return q.w;
}

__device__ inline float axis_crossflow_speed(const Prim &L, const Prim &R,
                                             int axis) {
  if (axis == 0)
    return (fabsf(L.v) + fabsf(R.v) + fabsf(L.w) + fabsf(R.w)) * 0.5f;
  if (axis == 1)
    return (fabsf(L.u) + fabsf(R.u) + fabsf(L.w) + fabsf(R.w)) * 0.5f;
  return (fabsf(L.u) + fabsf(R.u) + fabsf(L.v) + fabsf(R.v)) * 0.5f;
}

__device__ inline Cons axis_flux(const Prim &q, int axis) {
  if (axis == 0)
    return flux_x(q);
  if (axis == 1)
    return flux_y(q);
  return flux_z(q);
}

__device__ inline void fill_star_momentum(Cons &UStar, const Prim &q,
                                          float rStar, float sM, int axis) {
  if (axis == 0) {
    UStar.mx = rStar * sM;
    UStar.my = rStar * q.v;
    UStar.mz = rStar * q.w;
  } else if (axis == 1) {
    UStar.mx = rStar * q.u;
    UStar.my = rStar * sM;
    UStar.mz = rStar * q.w;
  } else {
    UStar.mx = rStar * q.u;
    UStar.my = rStar * q.v;
    UStar.mz = rStar * sM;
  }
}

__device__ inline Cons addC(const Cons &a, const Cons &b) {
  return {a.r + b.r,   a.mx + b.mx, a.my + b.my,
          a.mz + b.mz, a.Et + b.Et, a.Ev + b.Ev};
}
__device__ inline Cons subC(const Cons &a, const Cons &b) {
  return {a.r - b.r,   a.mx - b.mx, a.my - b.my,
          a.mz - b.mz, a.Et - b.Et, a.Ev - b.Ev};
}
__device__ inline Cons mulC(const Cons &a, float s) {
  return {a.r * s, a.mx * s, a.my * s, a.mz * s, a.Et * s, a.Ev * s};
}

__device__ inline float entropy_fix_speed(float s, float a_ref) {
  float d = 0.1f * a_ref;
  float as = fabsf(s);
  if (as >= d)
    return s;
  float sgn = (s >= 0.f) ? 1.f : -1.f;
  float sm = 0.5f * (as * as / fmaxf(d, DENOM_EPS) + d);
  return sgn * sm;
}

__device__ inline float shock_sensor(const Prim &L, const Prim &R) {
  float dp = fabsf(R.p - L.p) / fmaxf(R.p + L.p, DENOM_EPS);
  float dr = fabsf(R.r - L.r) / fmaxf(R.r + L.r, DENOM_EPS);
  float s = 0.5f * (dp + dr);
  return clampf(5.f * s, 0.f, 1.f);
}

__device__ inline Cons hllc_flux_axis(const Prim &L, const Prim &R, int axis) {
  float aL = soundspeed(L), aR = soundspeed(R);
  float unL = axis_velocity(L, axis);
  float unR = axis_velocity(R, axis);
  float sL = fminf(unL - aL, unR - aR);
  float sR = fmaxf(unL + aL, unR + aR);

  float aRef = fmaxf(aL, aR);
  sL = entropy_fix_speed(sL, aRef);
  sR = entropy_fix_speed(sR, aRef);

  Cons UL = prim_to_cons(L);
  Cons UR = prim_to_cons(R);
  Cons FL = axis_flux(L, axis);
  Cons FR = axis_flux(R, axis);

  if (sL >= 0.f)
    return FL;
  if (sR <= 0.f)
    return FR;

  float rL = L.r, rR = R.r;
  float pL = L.p, pR = R.p;

  float denom = signed_denom_guard(rL * (sL - unL) - rR * (sR - unR));
  float sM = (pR - pL + rL * unL * (sL - unL) - rR * unR * (sR - unR)) / denom;

  float pStarL = pL + rL * (sL - unL) * (sM - unL);
  float pStarR = pR + rR * (sR - unR) * (sM - unR);
  float pStar = 0.5f * (pStarL + pStarR);

  float vCarb = axis_crossflow_speed(L, R, axis);
  float align = clampf(1.f - vCarb / fmaxf(aRef, DENOM_EPS), 0.f, 1.f);
  float alpha = shock_sensor(L, R) * align;

  Cons FHLL;
  {
    Cons num = subC(mulC(FL, sR), mulC(FR, sL));
    Cons corr = mulC(subC(UR, UL), sL * sR);
    FHLL = mulC(addC(num, corr), 1.f / signed_denom_guard(sR - sL));
  }

  if (sM >= 0.f) {
    float starDenom = signed_denom_guard(sL - sM);
    float rStar = rL * (sL - unL) / starDenom;
    float EL = UL.Et;
    float EStar = ((sL - unL) * EL - pL * unL + pStar * sM) / starDenom;
    float EvStar = UL.Ev * (sL - unL) / starDenom;

    Cons UStar;
    UStar.r = rStar;
    fill_star_momentum(UStar, L, rStar, sM, axis);
    UStar.Et = EStar;
    UStar.Ev = EvStar;

    Cons FHLLC = addC(FL, mulC(subC(UStar, UL), sL));
    return addC(mulC(FHLLC, 1.f - alpha), mulC(FHLL, alpha));
  } else {
    float starDenom = signed_denom_guard(sR - sM);
    float rStar = rR * (sR - unR) / starDenom;
    float ER = UR.Et;
    float EStar = ((sR - unR) * ER - pR * unR + pStar * sM) / starDenom;
    float EvStar = UR.Ev * (sR - unR) / starDenom;

    Cons UStar;
    UStar.r = rStar;
    fill_star_momentum(UStar, R, rStar, sM, axis);
    UStar.Et = EStar;
    UStar.Ev = EvStar;

    Cons FHLLC = addC(FR, mulC(subC(UStar, UR), sR));
    return addC(mulC(FHLLC, 1.f - alpha), mulC(FHLL, alpha));
  }
}

__device__ inline Cons hllc_flux_x(const Prim &L, const Prim &R) {
  return hllc_flux_axis(L, R, 0);
}
__device__ inline Cons hllc_flux_y(const Prim &L, const Prim &R) {
  return hllc_flux_axis(L, R, 1);
}
__device__ inline Cons hllc_flux_z(const Prim &L, const Prim &R) {
  return hllc_flux_axis(L, R, 2);
}

__device__ inline void apply_wall(Prim &q) {
  float p_keep = fmaxf(q.p, RHO_P_FLOOR);
  q.u = 0.f;
  q.v = 0.f;
  q.w = 0.f;
  q.T = P.Twall;
  q.p = p_keep;
  q.r = fmaxf(q.p / (P.R * fmaxf(q.T, NEWTON_TEMP_FLOOR)), RHO_P_FLOOR);
  q.ev = evib_eq(P.Twall);
  q.Tv = P.Twall;
}

__device__ inline void atomicMaxFloat(float *addr, float val) {
  unsigned int *a = (unsigned int *)addr;
  unsigned int old = *a, assumed;
  while (u2f(old) < val) {
    assumed = old;
    old = atomicCAS(a, assumed, f2u(val));
    if (old == assumed)
      break;
  }
}

__device__ inline float weno5_left(float v0, float v1, float v2, float v3,
                                   float v4) {
  float p0 = (2.f * v0 - 7.f * v1 + 11.f * v2) * (1.f / 6.f);
  float p1 = (-1.f * v1 + 5.f * v2 + 2.f * v3) * (1.f / 6.f);
  float p2 = (2.f * v2 + 5.f * v3 - 1.f * v4) * (1.f / 6.f);

  float b0 = (13.f / 12.f) * (v0 - 2.f * v1 + v2) * (v0 - 2.f * v1 + v2) +
             0.25f * (v0 - 4.f * v1 + 3.f * v2) * (v0 - 4.f * v1 + 3.f * v2);
  float b1 = (13.f / 12.f) * (v1 - 2.f * v2 + v3) * (v1 - 2.f * v2 + v3) +
             0.25f * (v1 - v3) * (v1 - v3);
  float b2 = (13.f / 12.f) * (v2 - 2.f * v3 + v4) * (v2 - 2.f * v3 + v4) +
             0.25f * (3.f * v2 - 4.f * v3 + v4) * (3.f * v2 - 4.f * v3 + v4);

  float eps = WENO_EPS;
  float a0 = 0.1f / ((eps + b0) * (eps + b0));
  float a1 = 0.6f / ((eps + b1) * (eps + b1));
  float a2 = 0.3f / ((eps + b2) * (eps + b2));
  float s = a0 + a1 + a2;

  return (a0 / s) * p0 + (a1 / s) * p1 + (a2 / s) * p2;
}

__device__ inline float weno5_right(float v0, float v1, float v2, float v3,
                                    float v4) {
  return weno5_left(v4, v3, v2, v1, v0);
}

__device__ inline void prim_floor_fast(Prim &q) {
  q.r = fmaxf(q.r, RHO_P_FLOOR);
  q.p = fmaxf(q.p, RHO_P_FLOOR);
  q.ev = fmaxf(q.ev, 0.f);
  q.T = q.p / (q.r * P.R);
  q.Tv = 0.f;
}

__device__ inline void weno_face_from_6(const Prim &q0, const Prim &q1,
                                        const Prim &q2, const Prim &q3,
                                        const Prim &q4, const Prim &q5, Prim &L,
                                        Prim &R) {
  L.r = weno5_left(q0.r, q1.r, q2.r, q3.r, q4.r);
  L.u = weno5_left(q0.u, q1.u, q2.u, q3.u, q4.u);
  L.v = weno5_left(q0.v, q1.v, q2.v, q3.v, q4.v);
  L.w = weno5_left(q0.w, q1.w, q2.w, q3.w, q4.w);
  L.p = weno5_left(q0.p, q1.p, q2.p, q3.p, q4.p);
  L.ev = weno5_left(q0.ev, q1.ev, q2.ev, q3.ev, q4.ev);

  R.r = weno5_right(q1.r, q2.r, q3.r, q4.r, q5.r);
  R.u = weno5_right(q1.u, q2.u, q3.u, q4.u, q5.u);
  R.v = weno5_right(q1.v, q2.v, q3.v, q4.v, q5.v);
  R.w = weno5_right(q1.w, q2.w, q3.w, q4.w, q5.w);
  R.p = weno5_right(q1.p, q2.p, q3.p, q4.p, q5.p);
  R.ev = weno5_right(q1.ev, q2.ev, q3.ev, q4.ev, q5.ev);

  prim_floor_fast(L);
  prim_floor_fast(R);
}

__device__ inline Prim inflow_prim() {
  Prim q{};
  q.r = fmaxf(P.inflow_r, RHO_P_FLOOR);
  q.u = P.inflow_u;
  q.v = P.inflow_v;
  q.w = P.inflow_w;
  q.p = fmaxf(P.inflow_p, RHO_P_FLOOR);
  q.T = q.p / (q.r * P.R);
  q.ev = evib_eq(q.T);
  q.Tv = 0.f;
  return q;
}

__device__ inline Prim
outflow_prim_transmissive(const float *xi, const float *phix, const float *phiy,
                          const float *phiz, const float *lam, const float *zet,
                          int xghost, int y, int z) {
  int iR = idx3(P.nx - 1, y, z);
  Prim qR =
      log_to_prim_fast(xi[iR], phix[iR], phiy[iR], phiz[iR], lam[iR], zet[iR]);

  Prim q = qR;
  float aR = soundspeed(qR);
  float un = qR.u;

  if (un < 0.0f) {
    return inflow_prim();
  }

  if (un < aR) {
    float p_amb = fmaxf(P.inflow_p, RHO_P_FLOOR);
    float relax = 0.05f;
    q.p = fmaxf(q.p + relax * (p_amb - q.p), RHO_P_FLOOR);
  }

  q.r = fmaxf(q.r, RHO_P_FLOOR);
  q.p = fmaxf(q.p, RHO_P_FLOOR);
  q.ev = fmaxf(q.ev, 0.f);
  q.T = q.p / (q.r * P.R);
  q.Tv = 0.f;
  return q;
}

__device__ inline Prim prim_at_xbc(const float *xi, const float *phix,
                                   const float *phiy, const float *phiz,
                                   const float *lam, const float *zet,
                                   const uint8_t *solid, int x, int y, int z) {
  y = wrapi(y, P.ny);
  z = wrapi(z, P.nz);

  if (x < 0) {
    Prim q = inflow_prim();
    if (cell_is_solid(solid, x, y, z))
      apply_wall(q);
    return q;
  }

  if (x >= P.nx) {
    return outflow_prim_transmissive(xi, phix, phiy, phiz, lam, zet, x, y, z);
  }

  int i = idx3(x, y, z);
  Prim q = log_to_prim_fast(xi[i], phix[i], phiy[i], phiz[i], lam[i], zet[i]);
  if (cell_is_solid(solid, x, y, z))
    apply_wall(q);
  return q;
}

__device__ inline bool solid_at_xbc(const uint8_t *solid, int x, int y, int z) {
  y = wrapi(y, P.ny);
  z = wrapi(z, P.nz);
  return cell_is_solid(solid, x, y, z);
}

__global__ void k_build_solid_mask(uint8_t *solid) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  int z = (int)(blockIdx.z * blockDim.z + threadIdx.z);
  if (x >= P.nx || y >= P.ny || z >= P.nz)
    return;

  float X = (x + 0.5f) * P.dx;
  float Y = (y + 0.5f) * P.dy;
  float Z = (z + 0.5f) * P.dz;
  solid[idx3(x, y, z)] = (sdf_sphere(X, Y, Z) < 0.f) ? 1 : 0;
}

__device__ inline Prim mirror_wall(const Prim &q, int axis) {
  Prim g = q;
  if (axis == 0)
    g.u = -g.u;
  if (axis == 1)
    g.v = -g.v;
  if (axis == 2)
    g.w = -g.w;
  return g;
}

// Extracting Schlieren gradients for export (since it's an excellent
// visualization metric)
__global__ void k_schlieren_export(const float *xi, const float *phix,
                                   const float *phiy, const float *phiz,
                                   const float *lam, const float *zet,
                                   const uint8_t *solid, float *out) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  int z = (int)(blockIdx.z * blockDim.z + threadIdx.z);
  if (x >= P.nx || y >= P.ny || z >= P.nz)
    return;

  int i = idx3(x, y, z);

  if (solid[i]) {
    out[i] = 0.f;
    return;
  }

  int xm = x - 1, xp = x + 1;
  int ym = wrapi(y - 1, P.ny), yp = wrapi(y + 1, P.ny);
  int zm = wrapi(z - 1, P.nz), zp = wrapi(z + 1, P.nz);

  Prim qxm = prim_at_xbc(xi, phix, phiy, phiz, lam, zet, solid, xm, y, z);
  Prim qxp = prim_at_xbc(xi, phix, phiy, phiz, lam, zet, solid, xp, y, z);
  Prim qym = prim_at_xbc(xi, phix, phiy, phiz, lam, zet, solid, x, ym, z);
  Prim qyp = prim_at_xbc(xi, phix, phiy, phiz, lam, zet, solid, x, yp, z);
  Prim qzm = prim_at_xbc(xi, phix, phiy, phiz, lam, zet, solid, x, y, zm);
  Prim qzp = prim_at_xbc(xi, phix, phiy, phiz, lam, zet, solid, x, y, zp);

  float drdx = (qxp.r - qxm.r) / (2.0f * P.dx);
  float drdy = (qyp.r - qym.r) / (2.0f * P.dy);
  float drdz = (qzp.r - qzm.r) / (2.0f * P.dz);
  out[i] = sqrtf(drdx * drdx + drdy * drdy + drdz * drdz);
}

__device__ inline bool finite_dev(float x) { return isfinite(x); }

__global__ void k_init(float *xi, float *phix, float *phiy, float *phiz,
                       float *lam, float *zet, const uint8_t *solid) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  int z = (int)(blockIdx.z * blockDim.z + threadIdx.z);
  if (x >= P.nx || y >= P.ny || z >= P.nz)
    return;

  int i = idx3(x, y, z);
  float r = fmaxf(P.inflow_r, RHO_P_FLOOR);
  float p = fmaxf(P.inflow_p, RHO_P_FLOOR);

  Prim q{};
  q.r = r;
  q.p = p;
  q.u = 0.0f;
  q.v = 0.0f;
  q.w = 0.0f;
  q.T = q.p / (q.r * P.R);
  q.ev = evib_eq(q.T);

  if (solid[i]) {
    q.u = 0.f;
    q.v = 0.f;
    q.w = 0.f;
    q.T = P.Twall;
    q.p = p;
    q.r = fmaxf(q.p / (P.R * fmaxf(q.T, NEWTON_TEMP_FLOOR)), RHO_P_FLOOR);
    q.ev = evib_eq(q.T);
  }

  xi[i] = xi_from_rho(q.r);
  phix[i] = phi_from_vel(q.u);
  phiy[i] = phi_from_vel(q.v);
  phiz[i] = phi_from_vel(q.w);
  lam[i] = lambda_from_p(q.p);
  zet[i] = zeta_from_evib(q.ev);
}

__global__ void k_step(const float *xi, const float *phix, const float *phiy,
                       const float *phiz, const float *lam, const float *zet,
                       float *xi2, float *phix2, float *phiy2, float *phiz2,
                       float *lam2, float *zet2, float dt, float inflow_gain,
                       float *g_maxwavespeed, const uint8_t *solid) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  int z = (int)(blockIdx.z * blockDim.z + threadIdx.z);
  bool in_domain = (x < P.nx && y < P.ny && z < P.nz);

  const int sx = (int)blockDim.x + 2 * WENO_HALO;
  const int sy = (int)blockDim.y + 2 * WENO_HALO;
  const int sz = (int)blockDim.z + 2 * WENO_HALO;
  const int sxy = sx * sy;
  const int svol = sxy * sz;

  extern __shared__ unsigned char s_mem[];
  float *s_r = reinterpret_cast<float *>(s_mem);
  float *s_u = s_r + svol;
  float *s_v = s_u + svol;
  float *s_w = s_v + svol;
  float *s_p = s_w + svol;
  float *s_ev = s_p + svol;
  uint8_t *s_solid = reinterpret_cast<uint8_t *>(s_ev + svol);

  int tid = (int)((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x +
                  threadIdx.x);
  int tcount = (int)(blockDim.x * blockDim.y * blockDim.z);
  int bx0 = (int)(blockIdx.x * blockDim.x);
  int by0 = (int)(blockIdx.y * blockDim.y);
  int bz0 = (int)(blockIdx.z * blockDim.z);

  for (int t = tid; t < svol; t += tcount) {
    int lz = t / sxy;
    int rem = t - lz * sxy;
    int ly = rem / sx;
    int lx = rem - ly * sx;

    int gx = bx0 + lx - WENO_HALO;
    int gy = by0 + ly - WENO_HALO;
    int gz = bz0 + lz - WENO_HALO;
    int gyw = wrapi(gy, P.ny);
    int gzw = wrapi(gz, P.nz);

    bool is_solid = solid_at_xbc(solid, gx, gyw, gzw);
    Prim q;
    if (gx < 0) {
      q = inflow_prim();
    } else if (gx >= P.nx) {
      q = outflow_prim_transmissive(xi, phix, phiy, phiz, lam, zet, gx, gyw,
                                    gzw);
    } else {
      int gxc = (gx >= P.nx) ? (P.nx - 1) : gx;
      int gi = idx3(gxc, gyw, gzw);
      q = log_to_prim_fast(xi[gi], phix[gi], phiy[gi], phiz[gi], lam[gi],
                           zet[gi]);
    }
    if (is_solid)
      apply_wall(q);

    s_r[t] = q.r;
    s_u[t] = q.u;
    s_v[t] = q.v;
    s_w[t] = q.w;
    s_p[t] = q.p;
    s_ev[t] = q.ev;
    s_solid[t] = is_solid ? 1 : 0;
  }
  __syncthreads();

  if (!in_domain)
    return;

  int i = idx3(x, y, z);
  if (solid[i]) {
    xi2[i] = xi[i];
    phix2[i] = phix[i];
    phiy2[i] = phiy[i];
    phiz2[i] = phiz[i];
    lam2[i] = lam[i];
    zet2[i] = zet[i];
    return;
  }

  int lcx = (int)threadIdx.x + WENO_HALO;
  int lcy = (int)threadIdx.y + WENO_HALO;
  int lcz = (int)threadIdx.z + WENO_HALO;

  auto prim_sh = [&](int ox, int oy, int oz) {
    int j = ((lcz + oz) * sy + (lcy + oy)) * sx + (lcx + ox);
    Prim q;
    q.r = s_r[j];
    q.u = s_u[j];
    q.v = s_v[j];
    q.w = s_w[j];
    q.p = s_p[j];
    q.ev = s_ev[j];
    return q;
  };

  auto solid_sh = [&](int ox, int oy, int oz) {
    int j = ((lcz + oz) * sy + (lcy + oy)) * sx + (lcx + ox);
    return s_solid[j] != 0;
  };

  auto any_solid_line = [&](int ax, int ay, int az, int bx, int by, int bz) {
    int dx = (bx > ax) - (bx < ax);
    int dy = (by > ay) - (by < ay);
    int dz = (bz > az) - (bz < az);
    int n = max3(abs(bx - ax), abs(by - ay), abs(bz - az));
#pragma unroll
    for (int k = 0; k <= 6; k++) {
      if (k > n)
        break;
      if (solid_sh(ax + k * dx, ay + k * dy, az + k * dz))
        return true;
    }
    return false;
  };

  Prim q0 = prim_sh(0, 0, 0);

  Cons Fx_m, Fx_p;
  {
    Prim qx_m3 = prim_sh(-3, 0, 0), qx_m2 = prim_sh(-2, 0, 0),
         qx_m1 = prim_sh(-1, 0, 0);
    Prim qx_0 = q0;
    Prim qx_p1 = prim_sh(1, 0, 0), qx_p2 = prim_sh(2, 0, 0),
         qx_p3 = prim_sh(3, 0, 0);

    bool face_solid_m = solid_sh(-1, 0, 0) || solid_sh(0, 0, 0);
    bool stencil_solid_m = any_solid_line(-3, 0, 0, 2, 0, 0);

    if (face_solid_m) {
      Prim R = q0;
      Prim L = mirror_wall(R, 0);
      Fx_m = hllc_flux_x(L, R);
    } else if (stencil_solid_m) {
      Prim L = qx_m1, R = qx_0;
      prim_floor_fast(L);
      prim_floor_fast(R);
      Fx_m = hllc_flux_x(L, R);
    } else {
      Prim L, R;
      weno_face_from_6(qx_m3, qx_m2, qx_m1, qx_0, qx_p1, qx_p2, L, R);
      Fx_m = hllc_flux_x(L, R);
    }

    bool face_solid_p = solid_sh(0, 0, 0) || solid_sh(1, 0, 0);
    bool stencil_solid_p = any_solid_line(-2, 0, 0, 3, 0, 0);

    if (face_solid_p) {
      Prim L = q0;
      Prim R = mirror_wall(L, 0);
      Fx_p = hllc_flux_x(L, R);
    } else if (stencil_solid_p) {
      Prim L = qx_0, R = qx_p1;
      prim_floor_fast(L);
      prim_floor_fast(R);
      Fx_p = hllc_flux_x(L, R);
    } else {
      Prim L, R;
      weno_face_from_6(qx_m2, qx_m1, qx_0, qx_p1, qx_p2, qx_p3, L, R);
      Fx_p = hllc_flux_x(L, R);
    }
  }

  Cons Fy_m, Fy_p;
  {
    Prim qy_m3 = prim_sh(0, -3, 0), qy_m2 = prim_sh(0, -2, 0),
         qy_m1 = prim_sh(0, -1, 0);
    Prim qy_0 = q0;
    Prim qy_p1 = prim_sh(0, 1, 0), qy_p2 = prim_sh(0, 2, 0),
         qy_p3 = prim_sh(0, 3, 0);

    bool face_solid_m = solid_sh(0, -1, 0) || solid_sh(0, 0, 0);
    bool stencil_solid_m = any_solid_line(0, -3, 0, 0, 2, 0);

    if (face_solid_m) {
      Prim R = q0;
      Prim L = mirror_wall(R, 1);
      Fy_m = hllc_flux_y(L, R);
    } else if (stencil_solid_m) {
      Prim L = qy_m1, R = qy_0;
      prim_floor_fast(L);
      prim_floor_fast(R);
      Fy_m = hllc_flux_y(L, R);
    } else {
      Prim L, R;
      weno_face_from_6(qy_m3, qy_m2, qy_m1, qy_0, qy_p1, qy_p2, L, R);
      Fy_m = hllc_flux_y(L, R);
    }

    bool face_solid_p = solid_sh(0, 0, 0) || solid_sh(0, 1, 0);
    bool stencil_solid_p = any_solid_line(0, -2, 0, 0, 3, 0);

    if (face_solid_p) {
      Prim L = q0;
      Prim R = mirror_wall(L, 1);
      Fy_p = hllc_flux_y(L, R);
    } else if (stencil_solid_p) {
      Prim L = qy_0, R = qy_p1;
      prim_floor_fast(L);
      prim_floor_fast(R);
      Fy_p = hllc_flux_y(L, R);
    } else {
      Prim L, R;
      weno_face_from_6(qy_m2, qy_m1, qy_0, qy_p1, qy_p2, qy_p3, L, R);
      Fy_p = hllc_flux_y(L, R);
    }
  }

  Cons Fz_m, Fz_p;
  {
    Prim qz_m3 = prim_sh(0, 0, -3), qz_m2 = prim_sh(0, 0, -2),
         qz_m1 = prim_sh(0, 0, -1);
    Prim qz_0 = q0;
    Prim qz_p1 = prim_sh(0, 0, 1), qz_p2 = prim_sh(0, 0, 2),
         qz_p3 = prim_sh(0, 0, 3);

    bool face_solid_m = solid_sh(0, 0, -1) || solid_sh(0, 0, 0);
    bool stencil_solid_m = any_solid_line(0, 0, -3, 0, 0, 2);

    if (face_solid_m) {
      Prim R = q0;
      Prim L = mirror_wall(R, 2);
      Fz_m = hllc_flux_z(L, R);
    } else if (stencil_solid_m) {
      Prim L = qz_m1, R = qz_0;
      prim_floor_fast(L);
      prim_floor_fast(R);
      Fz_m = hllc_flux_z(L, R);
    } else {
      Prim L, R;
      weno_face_from_6(qz_m3, qz_m2, qz_m1, qz_0, qz_p1, qz_p2, L, R);
      Fz_m = hllc_flux_z(L, R);
    }

    bool face_solid_p = solid_sh(0, 0, 0) || solid_sh(0, 0, 1);
    bool stencil_solid_p = any_solid_line(0, 0, -2, 0, 0, 3);

    if (face_solid_p) {
      Prim L = q0;
      Prim R = mirror_wall(L, 2);
      Fz_p = hllc_flux_z(L, R);
    } else if (stencil_solid_p) {
      Prim L = qz_0, R = qz_p1;
      prim_floor_fast(L);
      prim_floor_fast(R);
      Fz_p = hllc_flux_z(L, R);
    } else {
      Prim L, R;
      weno_face_from_6(qz_m2, qz_m1, qz_0, qz_p1, qz_p2, qz_p3, L, R);
      Fz_p = hllc_flux_z(L, R);
    }
  }

  Cons U0 = prim_to_cons(q0);

  Cons dU;
  dU.r = -((Fx_p.r - Fx_m.r) / P.dx + (Fy_p.r - Fy_m.r) / P.dy +
           (Fz_p.r - Fz_m.r) / P.dz);
  dU.mx = -((Fx_p.mx - Fx_m.mx) / P.dx + (Fy_p.mx - Fy_m.mx) / P.dy +
            (Fz_p.mx - Fz_m.mx) / P.dz);
  dU.my = -((Fx_p.my - Fx_m.my) / P.dx + (Fy_p.my - Fy_m.my) / P.dy +
            (Fz_p.my - Fz_m.my) / P.dz);
  dU.mz = -((Fx_p.mz - Fx_m.mz) / P.dx + (Fy_p.mz - Fy_m.mz) / P.dy +
            (Fz_p.mz - Fz_m.mz) / P.dz);
  dU.Et = -((Fx_p.Et - Fx_m.Et) / P.dx + (Fy_p.Et - Fy_m.Et) / P.dy +
            (Fz_p.Et - Fz_m.Et) / P.dz);
  dU.Ev = -((Fx_p.Ev - Fx_m.Ev) / P.dx + (Fy_p.Ev - Fy_m.Ev) / P.dy +
            (Fz_p.Ev - Fz_m.Ev) / P.dz);

  Cons U1 = addC(U0, mulC(dU, dt));
  Prim q1 = cons_to_prim(U1);
  if (!finite_dev(q1.r) || !finite_dev(q1.p) || !finite_dev(q1.u) ||
      !finite_dev(q1.v) || !finite_dev(q1.w) || !finite_dev(q1.ev) ||
      q1.r <= 0.f || q1.p <= 0.f || q1.ev < 0.f) {
    q1 = inflow_prim();
  }

  float ev_eq = evib_eq(q1.T);
  q1.ev = fmaxf(q1.ev + (ev_eq - q1.ev) * (dt / fmaxf(P.tau_vib, TAU_VIB_MIN)),
                0.f);

  int nsp = (P.sponge_n > 0) ? P.sponge_n : 0;
  if (nsp > 0 && x < nsp) {
    float s = fminf(fmaxf(1.0f - (float)x / (float)nsp, 0.0f), 1.0f);
    float k = P.sponge_strength * (s * s);
    Prim tgt{};
    tgt.r = fmaxf(P.inflow_r, RHO_P_FLOOR);
    tgt.p = fmaxf(P.inflow_p, RHO_P_FLOOR);
    tgt.u = inflow_gain * P.inflow_u;
    tgt.v = inflow_gain * P.inflow_v;
    tgt.w = inflow_gain * P.inflow_w;
    tgt.T = tgt.p / (tgt.r * P.R);
    tgt.ev = evib_eq(tgt.T);

    q1.r = fmaxf(q1.r + k * (tgt.r - q1.r), RHO_P_FLOOR);
    q1.p = fmaxf(q1.p + k * (tgt.p - q1.p), RHO_P_FLOOR);
    q1.u += k * (tgt.u - q1.u);
    q1.v += k * (tgt.v - q1.v);
    q1.w += k * (tgt.w - q1.w);
    q1.ev = fmaxf(q1.ev + k * (tgt.ev - q1.ev), 0.f);
  }

  int nspo = (P.sponge_out_n > 0) ? P.sponge_out_n : 0;
  if (nspo > 0 && x >= (P.nx - nspo)) {
    float s =
        fminf(fmaxf((float)(x - (P.nx - nspo)) / (float)nspo, 0.0f), 1.0f);
    float k = P.sponge_out_strength * (s * s);
    Prim tgt{};
    tgt.r = fmaxf(P.inflow_r, RHO_P_FLOOR);
    tgt.p = fmaxf(P.inflow_p, RHO_P_FLOOR);
    tgt.u = 0.0f;
    tgt.v = 0.0f;
    tgt.w = 0.0f;
    tgt.T = tgt.p / (tgt.r * P.R);
    tgt.ev = evib_eq(tgt.T);

    q1.r = fmaxf(q1.r + k * (tgt.r - q1.r), RHO_P_FLOOR);
    q1.p = fmaxf(q1.p + k * (tgt.p - q1.p), RHO_P_FLOOR);
    q1.u += k * (tgt.u - q1.u);
    q1.v += k * (tgt.v - q1.v);
    q1.w += k * (tgt.w - q1.w);
    q1.ev = fmaxf(q1.ev + k * (tgt.ev - q1.ev), 0.f);
  }

  float a = soundspeed(q1);
  float ssum = (fabsf(q1.u) + a) / P.dx + (fabsf(q1.v) + a) / P.dy +
               (fabsf(q1.w) + a) / P.dz;
  if (finite_dev(ssum) && ssum > 0.f)
    atomicMaxFloat(g_maxwavespeed, ssum);

  xi2[i] = xi_from_rho(q1.r);
  phix2[i] = phi_from_vel(q1.u);
  phiy2[i] = phi_from_vel(q1.v);
  phiz2[i] = phi_from_vel(q1.w);
  lam2[i] = lambda_from_p(q1.p);
  zet2[i] = zeta_from_evib(q1.ev);
}

// -------------------------------------------------------------------------
// Main Extractor & 4Splat Encoder
// -------------------------------------------------------------------------

int main() {
  Params hp{};
  hp.nx = 64;
  hp.ny = 64;
  hp.nz = 64;
  hp.dx = 1.f / hp.nx;
  hp.dy = 1.f / hp.ny;
  hp.dz = 1.f / hp.nz;
  hp.cfl = 0.3333f;
  hp.u_ref = 10.f;
  hp.R = 10.f;
  hp.gamma_floor = 1.1f;
  hp.Twall = 0.02f;
  hp.tau_vib = 2e-4f;
  hp.theta_v = 0.2f;
  hp.sdf_cx = 0.5f;
  hp.sdf_cy = 0.5f;
  hp.sdf_cz = 0.5f;
  hp.sdf_r = 0.25f;
  hp.inflow_r = 0.02f;
  hp.inflow_p = 0.02f;
  hp.inflow_u = 100.0f;
  hp.inflow_v = 0.0f;
  hp.inflow_w = 0.0f;
  hp.sponge_n = 24;
  hp.sponge_strength = 0.05f;
  hp.sponge_out_n = 24;
  hp.sponge_out_strength = 0.05f;

  ck(cudaMemcpyToSymbol(P, &hp, sizeof(Params)), "MemcpyToSymbol(P)");

  size_t N = (size_t)hp.nx * (size_t)hp.ny * (size_t)hp.nz;
  size_t bytes = N * sizeof(float);

  float *d_xi, *d_phix, *d_phiy, *d_phiz, *d_lam, *d_zet;
  float *d_xi2, *d_phix2, *d_phiy2, *d_phiz2, *d_lam2, *d_zet2;
  float *d_vis, *d_maxs;
  uint8_t *d_solid;

  ck(cudaMalloc(&d_xi, bytes), "malloc");
  ck(cudaMalloc(&d_phix, bytes), "malloc");
  ck(cudaMalloc(&d_phiy, bytes), "malloc");
  ck(cudaMalloc(&d_phiz, bytes), "malloc");
  ck(cudaMalloc(&d_lam, bytes), "malloc");
  ck(cudaMalloc(&d_zet, bytes), "malloc");
  ck(cudaMalloc(&d_xi2, bytes), "malloc");
  ck(cudaMalloc(&d_phix2, bytes), "malloc");
  ck(cudaMalloc(&d_phiy2, bytes), "malloc");
  ck(cudaMalloc(&d_phiz2, bytes), "malloc");
  ck(cudaMalloc(&d_lam2, bytes), "malloc");
  ck(cudaMalloc(&d_zet2, bytes), "malloc");
  ck(cudaMalloc(&d_vis, bytes), "malloc vis");
  ck(cudaMalloc(&d_maxs, sizeof(float)), "malloc maxs");
  ck(cudaMalloc(&d_solid, N * sizeof(uint8_t)), "malloc solid");

  dim3 block(8, 8, 4);
  dim3 grid((hp.nx + block.x - 1) / block.x, (hp.ny + block.y - 1) / block.y,
            (hp.nz + block.z - 1) / block.z);

  size_t k_step_smem =
      (size_t)(block.x + 2 * WENO_HALO) * (size_t)(block.y + 2 * WENO_HALO) *
      (size_t)(block.z + 2 * WENO_HALO) * (6 * sizeof(float) + sizeof(uint8_t));
  ck(cudaFuncSetAttribute(k_step, cudaFuncAttributeMaxDynamicSharedMemorySize,
                          (int)k_step_smem),
     "smem attr");

  k_build_solid_mask<<<grid, block>>>(d_solid);
  k_init<<<grid, block>>>(d_xi, d_phix, d_phiy, d_phiz, d_lam, d_zet, d_solid);
  ck(cudaDeviceSynchronize(), "k_init sync");

  // Configure 4Splat Export
  int frames = 60;
  int steps_per_frame = 4;
  int pSize = 256;

  std::vector<float> h_sch(N);
  uint64_t total_voxels = (uint64_t)N * frames;
  std::vector<uint64_t> h_indices(total_voxels);
  std::vector<Splat4D> h_palette(pSize);

  // Populate colormap palette (Thermal/Heat map from black -> red -> yellow ->
  // white)
  for (int i = 0; i < pSize; i++) {
    float t_val = (float)i / (pSize - 1.0f);
    float r = std::min(1.0f, t_val * 2.5f);
    float g = std::max(0.0f, std::min(1.0f, t_val * 2.5f - 0.5f));
    float b = std::max(0.0f, std::min(1.0f, t_val * 2.5f - 1.5f));
    h_palette[i] = create_splat4D(0, 1, 0, 1, 0, 1, 0, 1, r, g, b, 1.0f);
  }

  float t = 1e-5f;
  float d_tau = 1e-3f;

  std::printf("Running Hypersonic CFD for %d frames...\n", frames);

  for (int f = 0; f < frames; f++) {
    float dt = 0.f;

    for (int s = 0; s < steps_per_frame; s++) {
      t *= expf(d_tau);
      dt = t * d_tau;
      float ramp = t / 0.02f;
      float inflow_gain = fminf(fmaxf(ramp, 0.f), 1.f);

      float zero = 0.f;
      ck(cudaMemcpy(d_maxs, &zero, sizeof(float), cudaMemcpyHostToDevice),
         "set maxs");

      k_step<<<grid, block, k_step_smem>>>(
          d_xi, d_phix, d_phiy, d_phiz, d_lam, d_zet, d_xi2, d_phix2, d_phiy2,
          d_phiz2, d_lam2, d_zet2, dt, inflow_gain, d_maxs, d_solid);

      float maxs = 0.f;
      ck(cudaMemcpy(&maxs, d_maxs, sizeof(float), cudaMemcpyDeviceToHost),
         "get maxs");

      float dt_cfl = hp.cfl / fmaxf(maxs, 1e-9f);
      if (dt > 1.10f * dt_cfl)
        d_tau *= 0.80f;
      else if (dt < 0.85f * dt_cfl)
        d_tau *= 1.10f;
      d_tau = fminf(fmaxf(d_tau, 1e-7f), 5e-2f);

      std::swap(d_xi, d_xi2);
      std::swap(d_phix, d_phix2);
      std::swap(d_phiy, d_phiy2);
      std::swap(d_phiz, d_phiz2);
      std::swap(d_lam, d_lam2);
      std::swap(d_zet, d_zet2);
    }

    // Extract Schlieren for this frame
    k_schlieren_export<<<grid, block>>>(d_xi, d_phix, d_phiy, d_phiz, d_lam,
                                        d_zet, d_solid, d_vis);
    ck(cudaMemcpy(h_sch.data(), d_vis, bytes, cudaMemcpyDeviceToHost),
       "copy vis");

    // Map Scalar to Palette
    float min_val = 1e30f, max_val = -1e30f;
    for (float v : h_sch) {
      min_val = fminf(min_val, v);
      max_val = fmaxf(max_val, v);
    }
    float range = fmaxf(max_val - min_val, 1e-12f);

    uint64_t offset = (uint64_t)f * N;
    for (int z = 0; z < hp.nz; z++) {
      for (int y = 0; y < hp.ny; y++) {
        for (int x = 0; x < hp.nx; x++) {
          int idx3d = (z * hp.ny + y) * hp.nx + x;
          float norm = (h_sch[idx3d] - min_val) / range;

          // Add slight gamma curve for visualization clarity
          norm = powf(norm, 0.65f);

          int pIdx = (int)(norm * 255.0f);
          pIdx = std::max(0, std::min(255, pIdx));
          h_indices[offset + idx3d] = pIdx;
        }
      }
    }
    std::printf("Frame %d/%d processed (t=%.6f)\n", f + 1, frames, t);
  }

  // 0x0004 sets Float32 Precision (0x04) and 8-bit Index Width (0x00)
  Splat4DHeader header =
      create_splat4DHeader(hp.nx, hp.ny, hp.nz, frames, pSize, 0x0004);
  Splat4DVideo video =
      create_splat4DVideo(header, h_palette.data(), h_indices.data());

  std::printf("Writing simulation video to tau_hypersonic.4spl...\n");
  FILE *fp = fopen("tau_hypersonic.4spl", "wb");
  if (fp) {
    write_splat4DVideo(fp, &video);
    fclose(fp);
    std::printf("Export Complete!\n");
  } else {
    std::fprintf(stderr, "Failed to open output file!\n");
  }

  ck(cudaFree(d_xi), "free");
  ck(cudaFree(d_phix), "free");
  ck(cudaFree(d_phiy), "free");
  ck(cudaFree(d_phiz), "free");
  ck(cudaFree(d_lam), "free");
  ck(cudaFree(d_zet), "free");
  ck(cudaFree(d_xi2), "free");
  ck(cudaFree(d_phix2), "free");
  ck(cudaFree(d_phiy2), "free");
  ck(cudaFree(d_phiz2), "free");
  ck(cudaFree(d_lam2), "free");
  ck(cudaFree(d_zet2), "free");
  ck(cudaFree(d_vis), "free");
  ck(cudaFree(d_maxs), "free");
  ck(cudaFree(d_solid), "free");

  return 0;
}