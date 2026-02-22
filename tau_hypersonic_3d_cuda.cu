// tau_hypersonic_3d_cuda.cu
// nvcc -O3 -std=c++17 tau_hypersonic_3d_cuda.cu -o tau3d -lineinfo

#include "raylib.h"
#include "rlgl.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <utility>
#include <vector>

#ifndef __CUDACC__
#define __host__
#define __device__
#define __global__
#endif

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
};

struct Cons {
  float r, mx, my, mz, Et, Ev;
};

struct Prim {
  float r, u, v, w, p, T, ev, Tv;
};

// Thermodynamic positivity floors: preserve realizable rho/p/e_th state during
// primitive <-> conservative conversion and reconstruction clipping.
constexpr float RHO_P_FLOOR = 1e-30f;
constexpr float THERMAL_ENERGY_FLOOR = 1e-12f;

// Denominator guards: avoid singular divisions while preserving equation sign
// where flux-wave relations depend on directional information.
constexpr float DENOM_EPS = 1e-12f;

// Vibrational temperature inversion control: keep Newton iterations in a
// physically admissible positive-temperature range.
constexpr float NEWTON_TEMP_FLOOR = 1e-6f;

// WENO smoothness epsilon: regularizes nonlinear weights near uniform states
// without coupling stencil behavior to thermo/Newton tolerances.
constexpr float WENO_EPS = 1e-6f;

// Source-term stiffness limiter for vibrational relaxation.
constexpr float TAU_VIB_MIN = 1e-9f;
constexpr int WENO_HALO = 3;

// helpers

static inline void ck(cudaError_t e, const char *m) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "%s: %s\n", m, cudaGetErrorString(e));
    std::exit(1);
  }
}

static inline Vector3 v3_add(Vector3 a, Vector3 b) {
  return {a.x + b.x, a.y + b.y, a.z + b.z};
}

static inline Vector3 v3_sub(Vector3 a, Vector3 b) {
  return {a.x - b.x, a.y - b.y, a.z - b.z};
}

static inline Vector3 v3_scale(Vector3 a, float s) {
  return {a.x * s, a.y * s, a.z * s};
}

static inline float v3_dot(Vector3 a, Vector3 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

static inline Vector3 v3_cross(Vector3 a, Vector3 b) {
  return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

static inline float v3_len(Vector3 a) { return sqrtf(v3_dot(a, a)); }

static inline Vector3 v3_norm(Vector3 a) {
  float L = v3_len(a);
  return (L > 1e-20f) ? v3_scale(a, 1.0f / L) : Vector3{0, 0, 0};
}

static inline Vector3 v3_rotate_axis_angle(Vector3 v, Vector3 axis, float ang) {
  axis = v3_norm(axis);
  float c = cosf(ang), s = sinf(ang);
  return v3_add(v3_add(v3_scale(v, c), v3_scale(v3_cross(axis, v), s)),
                v3_scale(axis, v3_dot(axis, v) * (1.0f - c)));
}

// device kernels

__device__ __constant__ Params P;

__device__ inline float sgnf(float x) { return (x > 0.f) - (x < 0.f); }

__device__ inline unsigned int f2u(float x) { return __float_as_uint(x); }

__device__ inline float u2f(unsigned int x) { return __uint_as_float(x); }

__device__ inline float rho_from_xi(float xi) { return __expf(xi); }

__device__ inline float p_from_lambda(float l) { return __expf(l); }

__device__ inline float evib_from_zeta(float z) { return __expf(z); }

__device__ __forceinline__ float sinhf_dev(float x) { return sinhf(x); }

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

__device__ inline bool cell_is_solid(const uint8_t *solid, int x, int y, int z) {
  if (x >= 0 && x < P.nx && y >= 0 && y < P.ny && z >= 0 && z < P.nz)
    return solid[idx3(x, y, z)] != 0;

  // Ghost lookups only happen near x boundaries; keep this fallback rare.
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

__device__ inline Prim log_to_prim(float xi, float phix, float phiy, float phiz,
                                   float lam, float zet) {
  Prim q;
  q.r = rho_from_xi(xi);
  q.u = vel_from_phi(phix);
  q.v = vel_from_phi(phiy);
  q.w = vel_from_phi(phiz);
  q.p = p_from_lambda(lam);
  q.ev = evib_from_zeta(zet);
  q.T = q.p / (q.r * P.R);
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

// MUSCL fallback reconstruction: use recon_pair_x only when intentionally
// running the 3-point minmod path. Prefer weno_face_from_6 for production
// high-order face reconstruction.
__device__ inline void recon_pair_x(const Prim &qm, const Prim &q0,
                                    const Prim &qp, Prim &L, Prim &R) {
  float dr = minmod(q0.r - qm.r, qp.r - q0.r);
  float du = minmod(q0.u - qm.u, qp.u - q0.u);
  float dv = minmod(q0.v - qm.v, qp.v - q0.v);
  float dw = minmod(q0.w - qm.w, qp.w - q0.w);
  float dp = minmod(q0.p - qm.p, qp.p - q0.p);
  float dev = minmod(q0.ev - qm.ev, qp.ev - q0.ev);

  L = q0;
  R = q0;
  L.r = q0.r - 0.5f * dr;
  R.r = q0.r + 0.5f * dr;
  L.u = q0.u - 0.5f * du;
  R.u = q0.u + 0.5f * du;
  L.v = q0.v - 0.5f * dv;
  R.v = q0.v + 0.5f * dv;
  L.w = q0.w - 0.5f * dw;
  R.w = q0.w + 0.5f * dw;
  L.p = q0.p - 0.5f * dp;
  R.p = q0.p + 0.5f * dp;
  L.ev = q0.ev - 0.5f * dev;
  R.ev = q0.ev + 0.5f * dev;

  L.r = fmaxf(L.r, RHO_P_FLOOR);
  R.r = fmaxf(R.r, RHO_P_FLOOR);
  L.p = fmaxf(L.p, RHO_P_FLOOR);
  R.p = fmaxf(R.p, RHO_P_FLOOR);
  L.ev = fmaxf(L.ev, 0.f);
  R.ev = fmaxf(R.ev, 0.f);

  L.T = L.p / (L.r * P.R);
  R.T = R.p / (R.r * P.R);
  L.Tv = Tv_from_evib_seed(L.ev, L.T);
  R.Tv = Tv_from_evib_seed(R.ev, R.T);
}

__device__ inline void apply_wall(Prim &q) {
  q.u = 0.f;
  q.v = 0.f;
  q.w = 0.f;
  q.T = P.Twall;
  q.p = fmaxf(q.r * P.R * q.T, RHO_P_FLOOR);
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

  float w0 = a0 / s;
  float w1 = a1 / s;
  float w2 = a2 / s;

  return w0 * p0 + w1 * p1 + w2 * p2;
}

__device__ inline float weno5_right(float v0, float v1, float v2, float v3,
                                    float v4) {
  return weno5_left(v4, v3, v2, v1, v0);
}

__device__ inline void prim_floor(Prim &q) {
  q.r = fmaxf(q.r, RHO_P_FLOOR);
  q.p = fmaxf(q.p, RHO_P_FLOOR);
  q.ev = fmaxf(q.ev, 0.f);
  q.T = q.p / (q.r * P.R);
  q.Tv = Tv_from_evib_seed(q.ev, q.T);
}

__device__ inline void weno_face_from_6(const Prim &q0, const Prim &q1,
                                        const Prim &q2, const Prim &q3,
                                        const Prim &q4, const Prim &q5, Prim &L,
                                        Prim &R) {
  // Interface between q2 (cell i) and q3 (cell i+1)
  // L uses q0..q4 (i-2..i+2), R uses q1..q5 (i-1..i+3)
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

  prim_floor(L);
  prim_floor(R);
}

__device__ inline Prim prim_at(const float *xi, const float *phix,
                               const float *phiy, const float *phiz,
                               const float *lam, const float *zet,
                               const uint8_t *solid, int x, int y, int z) {
  int i = idx3(x, y, z);
  Prim q = log_to_prim(xi[i], phix[i], phiy[i], phiz[i], lam[i], zet[i]);
  if (cell_is_solid(solid, x, y, z))
    apply_wall(q);
  return q;
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
  q.Tv = Tv_from_evib_seed(q.ev, q.T);
  return q;
}

__device__ inline Prim outflow_prim_characteristic(const float *xi,
                                                   const float *phix,
                                                   const float *phiy,
                                                   const float *phiz,
                                                   const float *lam,
                                                   const float *zet,
                                                   int xghost, int y, int z) {
  // nearest interior boundary state and one-cell-upwind state for
  // extrapolation slope
  int iR = idx3(P.nx - 1, y, z);
  Prim qR = log_to_prim(xi[iR], phix[iR], phiy[iR], phiz[iR], lam[iR], zet[iR]);

  Prim qL = qR;
  if (P.nx > 1) {
    int iL = idx3(P.nx - 2, y, z);
    qL = log_to_prim(xi[iL], phix[iL], phiy[iL], phiz[iL], lam[iL], zet[iL]);
  }

  int g = xghost - (P.nx - 1);
  float gf = (float)g;

  Prim qEx{};
  qEx.r = fmaxf(qR.r + gf * (qR.r - qL.r), RHO_P_FLOOR);
  qEx.u = qR.u + gf * (qR.u - qL.u);
  qEx.v = qR.v + gf * (qR.v - qL.v);
  qEx.w = qR.w + gf * (qR.w - qL.w);
  qEx.p = fmaxf(qR.p + gf * (qR.p - qL.p), RHO_P_FLOOR);
  qEx.T = qEx.p / (qEx.r * P.R);
  qEx.ev = fmaxf(qR.ev + gf * (qR.ev - qL.ev), 0.f);
  qEx.Tv = Tv_from_evib_seed(qEx.ev, qEx.T);

  // far-field target used for incoming characteristics
  Prim qT = inflow_prim();

  float rho_ref = fmaxf(qR.r, RHO_P_FLOOR);
  float a_ref = soundspeed(qR);
  float un = qR.u; // x-normal velocity at outflow

  // primitive perturbation relative to target, projected to 1D Euler
  // characteristics in x.
  float drho = qEx.r - qT.r;
  float du = qEx.u - qT.u;
  float dp = qEx.p - qT.p;

  float L1 = 0.5f * (dp / (a_ref * a_ref) - rho_ref * du / a_ref); // u-a
  float L5 = 0.5f * (dp / (a_ref * a_ref) + rho_ref * du / a_ref); // u+a
  float L2 = drho - dp / (a_ref * a_ref);                           // u
  float L3 = qEx.v - qT.v;                                          // u
  float L4 = qEx.w - qT.w;                                          // u
  float L6 = qEx.ev - qT.ev;                                        // u

  if (un - a_ref < 0.f)
    L1 = 0.f;
  if (un < 0.f) {
    L2 = 0.f;
    L3 = 0.f;
    L4 = 0.f;
    L6 = 0.f;
  }
  if (un + a_ref < 0.f)
    L5 = 0.f;

  Prim q{};
  q.r = fmaxf(qT.r + L1 + L2 + L5, RHO_P_FLOOR);
  q.u = qT.u + (L5 - L1) / fmaxf(rho_ref * a_ref, DENOM_EPS);
  q.p = fmaxf(qT.p + a_ref * a_ref * (L1 + L5), RHO_P_FLOOR);
  q.v = qT.v + L3;
  q.w = qT.w + L4;
  q.T = q.p / (q.r * P.R);
  q.ev = fmaxf(qT.ev + L6, 0.f);
  q.Tv = Tv_from_evib_seed(q.ev, q.T);
  return q;
}

__device__ inline Prim prim_at_xbc(const float *xi, const float *phix,
                                   const float *phiy, const float *phiz,
                                   const float *lam, const float *zet,
                                   const uint8_t *solid, int x, int y, int z) {
  // y,z periodic
  y = wrapi(y, P.ny);
  z = wrapi(z, P.nz);

  if (x < 0) {
    Prim q = inflow_prim();
    // still enforce internal solid if you want it to block inflow stencils
    if (cell_is_solid(solid, x, y, z))
      apply_wall(q);
    return q;
  }

  if (x >= P.nx)
    return outflow_prim_characteristic(xi, phix, phiy, phiz, lam, zet, x, y,
                                       z);

  int i = idx3(x, y, z);
  Prim q = log_to_prim(xi[i], phix[i], phiy[i], phiz[i], lam[i], zet[i]);

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

// global kernels
enum VisMode {
  VIS_SCHLIEREN_RHO = 0, // |∇ρ|
  VIS_LOG_RHO = 1,       // log(1+ρ)
  VIS_LOG_P = 2,         // log(1+p)
  VIS_SPEED = 3,         // |u|
  VIS_MACH = 4,          // |u|/a
  VIS_VORT_MAG = 5,      // |curl(u)|
  VIS_DIV = 6,           // ∇·u
  VIS_Q_CRITERION = 7,   // Q = 0.5(||Ω||^2 - ||S||^2)
  VIS_COUNT
};

__device__ inline float safe_log1pf_dev(float x) {
  return logf(1.0f + fmaxf(x, 0.0f));
}

__global__ void k_vis(const float *xi, const float *phix, const float *phiy,
                      const float *phiz, const float *lam, const float *zet,
                      const uint8_t *solid, float *out, int mode) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  int z = (int)(blockIdx.z * blockDim.z + threadIdx.z);
  if (x >= P.nx || y >= P.ny || z >= P.nz)
    return;

  int i = idx3(x, y, z);

  // mask solid interior so it doesn't dominate visuals
  if (solid[i]) {
    out[i] = 0.f;
    return;
  }

  // periodic y,z neighbors (x uses inflow/outflow helper)
  int xm = x - 1, xp = x + 1;
  int ym = wrapi(y - 1, P.ny), yp = wrapi(y + 1, P.ny);
  int zm = wrapi(z - 1, P.nz), zp = wrapi(z + 1, P.nz);

  // For anything needing primitives, use BC-aware accessor
  Prim q0 = prim_at_xbc(xi, phix, phiy, phiz, lam, zet, solid, x, y, z);

  if (mode == VIS_LOG_RHO) {
    out[i] = safe_log1pf_dev(q0.r);
    return;
  }
  if (mode == VIS_LOG_P) {
    out[i] = safe_log1pf_dev(q0.p);
    return;
  }
  if (mode == VIS_SPEED) {
    float s = sqrtf(q0.u * q0.u + q0.v * q0.v + q0.w * q0.w);
    out[i] = s;
    return;
  }
  if (mode == VIS_MACH) {
    float a = soundspeed(q0);
    float s = sqrtf(q0.u * q0.u + q0.v * q0.v + q0.w * q0.w);
    out[i] = s / fmaxf(a, DENOM_EPS);
    return;
  }

  // Need neighbors for derivatives (curl/div/Q/schlieren)
  Prim qxm = prim_at_xbc(xi, phix, phiy, phiz, lam, zet, solid, xm, y, z);
  Prim qxp = prim_at_xbc(xi, phix, phiy, phiz, lam, zet, solid, xp, y, z);
  Prim qym = prim_at_xbc(xi, phix, phiy, phiz, lam, zet, solid, x, ym, z);
  Prim qyp = prim_at_xbc(xi, phix, phiy, phiz, lam, zet, solid, x, yp, z);
  Prim qzm = prim_at_xbc(xi, phix, phiy, phiz, lam, zet, solid, x, y, zm);
  Prim qzp = prim_at_xbc(xi, phix, phiy, phiz, lam, zet, solid, x, y, zp);

  float inv2dx = 0.5f / P.dx;
  float inv2dy = 0.5f / P.dy;
  float inv2dz = 0.5f / P.dz;

  // velocity gradients
  float dudx = (qxp.u - qxm.u) * inv2dx;
  float dudy = (qyp.u - qym.u) * inv2dy;
  float dudz = (qzp.u - qzm.u) * inv2dz;

  float dvdx = (qxp.v - qxm.v) * inv2dx;
  float dvdy = (qyp.v - qym.v) * inv2dy;
  float dvdz = (qzp.v - qzm.v) * inv2dz;

  float dwdx = (qxp.w - qxm.w) * inv2dx;
  float dwdy = (qyp.w - qym.w) * inv2dy;
  float dwdz = (qzp.w - qzm.w) * inv2dz;

  if (mode == VIS_DIV) {
    out[i] = dudx + dvdy + dwdz;
    return;
  }

  // vorticity = curl(u)
  float wx = dwdy - dvdz;
  float wy = dudz - dwdx;
  float wz = dvdx - dudy;

  if (mode == VIS_VORT_MAG) {
    out[i] = sqrtf(wx * wx + wy * wy + wz * wz);
    return;
  }

  if (mode == VIS_Q_CRITERION) {
    // Decompose gradU into symmetric S and antisymmetric Ω
    // ||Ω||^2 = 2*(Ω12^2 + Ω13^2 + Ω23^2)
    float O12 = 0.5f * (dudy - dvdx);
    float O13 = 0.5f * (dudz - dwdx);
    float O23 = 0.5f * (dvdz - dwdy);
    float Om2 = 2.0f * (O12 * O12 + O13 * O13 + O23 * O23);

    // ||S||^2 = sum_{ij} Sij^2 (with S symmetric)
    float S11 = dudx;
    float S22 = dvdy;
    float S33 = dwdz;
    float S12 = 0.5f * (dudy + dvdx);
    float S13 = 0.5f * (dudz + dwdx);
    float S23 = 0.5f * (dvdz + dwdy);
    float Sm2 = (S11 * S11 + S22 * S22 + S33 * S33) +
                2.0f * (S12 * S12 + S13 * S13 + S23 * S23);

    out[i] = 0.5f * (Om2 - Sm2);
    return;
  }

  // default: schlieren of density |∇ρ|
  float drdx = (qxp.r - qxm.r) * inv2dx;
  float drdy = (qyp.r - qym.r) * inv2dy;
  float drdz = (qzp.r - qzm.r) * inv2dz;
  out[i] = sqrtf(drdx * drdx + drdy * drdy + drdz * drdz);
}

__global__ void k_init(float *xi, float *phix, float *phiy, float *phiz,
                       float *lam, float *zet, const uint8_t *solid) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  int z = (int)(blockIdx.z * blockDim.z + threadIdx.z);
  if (x >= P.nx || y >= P.ny || z >= P.nz)
    return;

  int i = idx3(x, y, z);

  // Ambient thermodynamic state = inflow (no initial pressure mismatch)
  float r = fmaxf(P.inflow_r, RHO_P_FLOOR);
  float p = fmaxf(P.inflow_p, RHO_P_FLOOR);

  // Start at rest everywhere to avoid an impulsive "sphere appears in a moving
  // flow"
  float u = 0.0f, v = 0.0f, w = 0.0f;

  Prim q{};
  q.r = r;
  q.p = p;
  q.u = u;
  q.v = v;
  q.w = w;
  q.T = q.p / (q.r * P.R);
  q.ev = evib_eq(q.T);
  q.Tv = Tv_from_evib_seed(q.ev, q.T);

  // Pre-apply the wall state in the solid so the solver doesn't "discover" it
  // at t=0
  float X = (x + 0.5f) * P.dx;
  float Y = (y + 0.5f) * P.dy;
  float Z = (z + 0.5f) * P.dz;
  if (solid[i]) {
    // Keep pressure matched to ambient to prevent an artificial blast
    // by choosing wall density consistent with (p = rho R Twall).
    q.u = 0.f;
    q.v = 0.f;
    q.w = 0.f;
    q.T = P.Twall;
    q.p = p;
    q.r = fmaxf(q.p / (P.R * fmaxf(q.T, NEWTON_TEMP_FLOOR)), RHO_P_FLOOR);
    q.ev = evib_eq(q.T);
    q.Tv = q.T;
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
    } else {
      int gxc = (gx >= P.nx) ? (P.nx - 1) : gx;
      int gi = idx3(gxc, gyw, gzw);
      q = log_to_prim(xi[gi], phix[gi], phiy[gi], phiz[gi], lam[gi], zet[gi]);
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
    q.T = 0.f;
    q.Tv = 0.f;
    return q;
  };

  auto solid_sh = [&](int ox, int oy, int oz) {
    int j = ((lcz + oz) * sy + (lcy + oy)) * sx + (lcx + ox);
    return s_solid[j] != 0;
  };

  Prim q0 = prim_sh(0, 0, 0);

  Cons Fx_m, Fx_p;
  {
    Prim qx_m3 = prim_sh(-3, 0, 0);
    Prim qx_m2 = prim_sh(-2, 0, 0);
    Prim qx_m1 = prim_sh(-1, 0, 0);
    Prim qx_0 = q0;
    Prim qx_p1 = prim_sh(1, 0, 0);
    Prim qx_p2 = prim_sh(2, 0, 0);
    Prim qx_p3 = prim_sh(3, 0, 0);

    bool solid_m = solid_sh(-1, 0, 0) || solid_sh(0, 0, 0);
    if (solid_m)
      Fx_m = {0, 0, 0, 0, 0, 0};
    else {
      Prim L, R;
      weno_face_from_6(qx_m3, qx_m2, qx_m1, qx_0, qx_p1, qx_p2, L, R);
      Fx_m = hllc_flux_x(L, R);
    }

    bool solid_p = solid_sh(0, 0, 0) || solid_sh(1, 0, 0);
    if (solid_p)
      Fx_p = {0, 0, 0, 0, 0, 0};
    else {
      Prim L, R;
      weno_face_from_6(qx_m2, qx_m1, qx_0, qx_p1, qx_p2, qx_p3, L, R);
      Fx_p = hllc_flux_x(L, R);
    }
  }

  Cons Fy_m, Fy_p;
  {
    Prim qy_m3 = prim_sh(0, -3, 0);
    Prim qy_m2 = prim_sh(0, -2, 0);
    Prim qy_m1 = prim_sh(0, -1, 0);
    Prim qy_0 = q0;
    Prim qy_p1 = prim_sh(0, 1, 0);
    Prim qy_p2 = prim_sh(0, 2, 0);
    Prim qy_p3 = prim_sh(0, 3, 0);

    bool solid_m = solid_sh(0, -1, 0) || solid_sh(0, 0, 0);
    if (solid_m)
      Fy_m = {0, 0, 0, 0, 0, 0};
    else {
      Prim L, R;
      weno_face_from_6(qy_m3, qy_m2, qy_m1, qy_0, qy_p1, qy_p2, L, R);
      Fy_m = hllc_flux_y(L, R);
    }

    bool solid_p = solid_sh(0, 0, 0) || solid_sh(0, 1, 0);
    if (solid_p)
      Fy_p = {0, 0, 0, 0, 0, 0};
    else {
      Prim L, R;
      weno_face_from_6(qy_m2, qy_m1, qy_0, qy_p1, qy_p2, qy_p3, L, R);
      Fy_p = hllc_flux_y(L, R);
    }
  }

  Cons Fz_m, Fz_p;
  {
    Prim qz_m3 = prim_sh(0, 0, -3);
    Prim qz_m2 = prim_sh(0, 0, -2);
    Prim qz_m1 = prim_sh(0, 0, -1);
    Prim qz_0 = q0;
    Prim qz_p1 = prim_sh(0, 0, 1);
    Prim qz_p2 = prim_sh(0, 0, 2);
    Prim qz_p3 = prim_sh(0, 0, 3);

    bool solid_m = solid_sh(0, 0, -1) || solid_sh(0, 0, 0);
    if (solid_m)
      Fz_m = {0, 0, 0, 0, 0, 0};
    else {
      Prim L, R;
      weno_face_from_6(qz_m3, qz_m2, qz_m1, qz_0, qz_p1, qz_p2, L, R);
      Fz_m = hllc_flux_z(L, R);
    }

    bool solid_p = solid_sh(0, 0, 0) || solid_sh(0, 0, 1);
    if (solid_p)
      Fz_p = {0, 0, 0, 0, 0, 0};
    else {
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

  float ev_eq = evib_eq(q1.T);
  q1.ev = fmaxf(q1.ev + (ev_eq - q1.ev) * (dt / fmaxf(P.tau_vib, TAU_VIB_MIN)), 0.f);
  q1.Tv = Tv_from_evib_seed(q1.ev, q1.T);

  // Sponge near inflow boundary (x small): gently relax toward ramped inflow
  // state. This prevents a startup blast and reduces reflections.
  int nsp = (P.sponge_n > 0) ? P.sponge_n : 0;
  if (nsp > 0 && x < nsp) {
    float s = 1.0f - (float)x / (float)nsp; // 1 at x=0 -> 0 at x=nsp
    s = fminf(fmaxf(s, 0.0f), 1.0f);
    float k = P.sponge_strength * (s * s); // stronger right at boundary

    Prim tgt{};
    tgt.r = fmaxf(P.inflow_r, RHO_P_FLOOR);
    tgt.p = fmaxf(P.inflow_p, RHO_P_FLOOR);
    tgt.u = inflow_gain * P.inflow_u;
    tgt.v = inflow_gain * P.inflow_v;
    tgt.w = inflow_gain * P.inflow_w;
    tgt.T = tgt.p / (tgt.r * P.R);
    tgt.ev = evib_eq(tgt.T);
    tgt.Tv = Tv_from_evib_seed(tgt.ev, tgt.T);

    // blend primitives (simple, effective for a sponge)
    q1.r = fmaxf(q1.r + k * (tgt.r - q1.r), RHO_P_FLOOR);
    q1.p = fmaxf(q1.p + k * (tgt.p - q1.p), RHO_P_FLOOR);
    q1.u = q1.u + k * (tgt.u - q1.u);
    q1.v = q1.v + k * (tgt.v - q1.v);
    q1.w = q1.w + k * (tgt.w - q1.w);
    q1.T = q1.p / (q1.r * P.R);
    q1.ev = fmaxf(q1.ev + k * (tgt.ev - q1.ev), 0.f);
    q1.Tv = Tv_from_evib_seed(q1.ev, q1.T);
  }

  float a = soundspeed(q1);
  float sx = (fabsf(q1.u) + a) / P.dx;
  float sy = (fabsf(q1.v) + a) / P.dy;
  float sz = (fabsf(q1.w) + a) / P.dz;
  atomicMaxFloat(g_maxwavespeed, sx + sy + sz);

  xi2[i] = xi_from_rho(q1.r);
  phix2[i] = phi_from_vel(q1.u);
  phiy2[i] = phi_from_vel(q1.v);
  phiz2[i] = phi_from_vel(q1.w);
  lam2[i] = lambda_from_p(q1.p);
  zet2[i] = zeta_from_evib(q1.ev);
}

__global__ void k_schlieren(const float *xi, float *out) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  int z = (int)(blockIdx.z * blockDim.z + threadIdx.z);
  if (x >= P.nx || y >= P.ny || z >= P.nz)
    return;

  int xm = x - 1, xp = x + 1;
  int ym = wrapi(y - 1, P.ny), yp = wrapi(y + 1, P.ny);
  int zm = wrapi(z - 1, P.nz), zp = wrapi(z + 1, P.nz);

  float rxm = (xm < 0) ? fmaxf(P.inflow_r, 1e-30f)
                       : rho_from_xi(xi[idx3(xm, y, z)]);
  float rxp = (xp >= P.nx) ? rho_from_xi(xi[idx3(P.nx - 1, y, z)])
                           : rho_from_xi(xi[idx3(xp, y, z)]);
  float rym = rho_from_xi(xi[idx3(x, ym, z)]);
  float ryp = rho_from_xi(xi[idx3(x, yp, z)]);
  float rzm = rho_from_xi(xi[idx3(x, y, zm)]);
  float rzp = rho_from_xi(xi[idx3(x, y, zp)]);

  float drdx = (rxp - rxm) / (2.f * P.dx);
  float drdy = (ryp - rym) / (2.f * P.dy);
  float drdz = (rzp - rzm) / (2.f * P.dz);

  float g = sqrtf(drdx * drdx + drdy * drdy + drdz * drdz);
  out[idx3(x, y, z)] = g;
}

__global__ void k_outflow_reflection_metric(const float *xi, const float *phix,
                                            const float *phiy,
                                            const float *phiz,
                                            const float *lam,
                                            const float *zet, float *g_max_dp,
                                            int nprobe) {
  int x = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int y = (int)(blockIdx.y * blockDim.y + threadIdx.y);
  int z = (int)(blockIdx.z * blockDim.z + threadIdx.z);
  if (x >= P.nx || y >= P.ny || z >= P.nz)
    return;

  int x0 = P.nx - ((nprobe > 1) ? nprobe : 1);
  if (x < x0)
    return;

  int i = idx3(x, y, z);
  Prim q = log_to_prim(xi[i], phix[i], phiy[i], phiz[i], lam[i], zet[i]);
  float p_ref = fmaxf(P.inflow_p, RHO_P_FLOOR);
  atomicMaxFloat(g_max_dp, fabsf(q.p - p_ref));
}

static inline float clamp01(float x) {
  return x < 0.f ? 0.f : (x > 1.f ? 1.f : x);
}

static inline float safe_log1p(float x) { return logf(1.0f + fmaxf(x, 0.0f)); }

static void slice_to_rgba(uint32_t *dst_rgba, const float *vol, int nx, int ny,
                          int nz, int zslice, bool log_scale, float a_gain) {
  zslice = (zslice < 0) ? 0 : (zslice >= nz ? (nz - 1) : zslice);
  const float *s = vol + (size_t)zslice * (size_t)nx * (size_t)ny;

  float mn = 1e30f, mx = -1e30f;
  for (int i = 0; i < nx * ny; i++) {
    float v = s[i];
    v = log_scale ? safe_log1p(v) : v;
    mn = fminf(mn, v);
    mx = fmaxf(mx, v);
  }
  float inv = 1.0f / fmaxf(mx - mn, 1e-20f);

  for (int i = 0; i < nx * ny; i++) {
    float v = s[i];
    v = log_scale ? safe_log1p(v) : v;
    float t = clamp01((v - mn) * inv);

    float a = clamp01(a_gain * (t * t)); // emphasize structures
    unsigned char c = (unsigned char)(t * 255.0f);
    unsigned char A = (unsigned char)(a * 255.0f);

    dst_rgba[i] = ((uint32_t)A << 24) | ((uint32_t)c << 16) |
                  ((uint32_t)c << 8) | (uint32_t)c;
  }
}

static void draw_slice_z(Texture2D tex, float z01, unsigned char alphaMul) {
  rlSetTexture(tex.id);
  rlBegin(RL_QUADS);
  rlColor4ub(255, 255, 255, alphaMul);

  rlTexCoord2f(0.0f, 1.0f);
  rlVertex3f(0.0f, 0.0f, z01);
  rlTexCoord2f(1.0f, 1.0f);
  rlVertex3f(1.0f, 0.0f, z01);
  rlTexCoord2f(1.0f, 0.0f);
  rlVertex3f(1.0f, 1.0f, z01);
  rlTexCoord2f(0.0f, 0.0f);
  rlVertex3f(0.0f, 1.0f, z01);

  rlEnd();
  rlSetTexture(0);
}

static void camera_orbit_pan_zoom(Camera3D *cam) {
  Vector2 md = GetMouseDelta();
  float wheel = GetMouseWheelMove();

  Vector3 forward = v3_norm(v3_sub(cam->target, cam->position));
  Vector3 right = v3_norm(v3_cross(forward, cam->up));
  Vector3 up = v3_norm(v3_cross(right, forward));

  if (wheel != 0.0f) {
    float dist = v3_len(v3_sub(cam->position, cam->target));
    float dz = dist * 0.12f * wheel;
    cam->position = v3_add(cam->position, v3_scale(forward, dz));
  }

  if (IsMouseButtonDown(MOUSE_BUTTON_RIGHT)) {
    float yaw = -md.x * 0.0045f;
    float pitch = -md.y * 0.0045f;

    Vector3 off = v3_sub(cam->position, cam->target);
    off = v3_rotate_axis_angle(off, cam->up, yaw);

    Vector3 f2 = v3_norm(v3_scale(off, -1.0f));
    Vector3 r2 = v3_norm(v3_cross(f2, cam->up));
    off = v3_rotate_axis_angle(off, r2, pitch);

    cam->position = v3_add(cam->target, off);
  }

  if (IsMouseButtonDown(MOUSE_BUTTON_MIDDLE)) {
    float dist = v3_len(v3_sub(cam->position, cam->target));
    float k = 0.0014f * dist;
    Vector3 pan = v3_add(v3_scale(right, -md.x * k), v3_scale(up, md.y * k));
    cam->position = v3_add(cam->position, pan);
    cam->target = v3_add(cam->target, pan);
  }
}

static void reset_sim(dim3 grid, dim3 block, float *d_xi, float *d_phix,
                      float *d_phiy, float *d_phiz, float *d_lam,
                      float *d_zet, const uint8_t *d_solid) {
  k_init<<<grid, block>>>(d_xi, d_phix, d_phiy, d_phiz, d_lam, d_zet, d_solid);
  ck(cudaGetLastError(), "k_init launch");
  ck(cudaDeviceSynchronize(), "k_init sync");
}

static const char *vis_name(int m) {
  switch (m) {
  case VIS_SCHLIEREN_RHO:
    return "schlieren(|grad rho|)";
  case VIS_LOG_RHO:
    return "log(1+rho)";
  case VIS_LOG_P:
    return "log(1+p)";
  case VIS_SPEED:
    return "speed(|u|)";
  case VIS_MACH:
    return "Mach(|u|/a)";
  case VIS_VORT_MAG:
    return "vorticity(|curl u|)";
  case VIS_DIV:
    return "divergence(div u)";
  case VIS_Q_CRITERION:
    return "Q-criterion";
  default:
    return "?";
  }
}

int main() {
  Params hp{};
  hp.nx = 64;
  hp.ny = 64;
  hp.nz = 64;
  hp.dx = 1.f / hp.nx;
  hp.dy = 1.f / hp.ny;
  hp.dz = 1.f / hp.nz;
  hp.cfl = 0.35f;
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

  ck(cudaMemcpyToSymbol(P, &hp, sizeof(Params)), "MemcpyToSymbol(P)");

  size_t N = (size_t)hp.nx * (size_t)hp.ny * (size_t)hp.nz;
  size_t bytes = N * sizeof(float);

  float *d_xi, *d_phix, *d_phiy, *d_phiz, *d_lam, *d_zet;
  float *d_xi2, *d_phix2, *d_phiy2, *d_phiz2, *d_lam2, *d_zet2;
  float *d_vis;
  float *d_maxs;
  float *d_reflect;
  uint8_t *d_solid;
  int vis_mode = VIS_SCHLIEREN_RHO;

  ck(cudaMalloc(&d_xi, bytes), "malloc xi");
  ck(cudaMalloc(&d_phix, bytes), "malloc phix");
  ck(cudaMalloc(&d_phiy, bytes), "malloc phiy");
  ck(cudaMalloc(&d_phiz, bytes), "malloc phiz");
  ck(cudaMalloc(&d_lam, bytes), "malloc lam");
  ck(cudaMalloc(&d_zet, bytes), "malloc zet");

  ck(cudaMalloc(&d_xi2, bytes), "malloc xi2");
  ck(cudaMalloc(&d_phix2, bytes), "malloc phix2");
  ck(cudaMalloc(&d_phiy2, bytes), "malloc phiy2");
  ck(cudaMalloc(&d_phiz2, bytes), "malloc phiz2");
  ck(cudaMalloc(&d_lam2, bytes), "malloc lam2");
  ck(cudaMalloc(&d_zet2, bytes), "malloc zet2");

  ck(cudaMalloc(&d_vis, bytes), "malloc vis");
  ck(cudaMalloc(&d_maxs, sizeof(float)), "malloc maxs");
  ck(cudaMalloc(&d_reflect, sizeof(float)), "malloc reflect");
  ck(cudaMalloc(&d_solid, N * sizeof(uint8_t)), "malloc solid mask");

  dim3 block(8, 8, 4);
  dim3 grid((hp.nx + block.x - 1) / block.x, (hp.ny + block.y - 1) / block.y,
            (hp.nz + block.z - 1) / block.z);
  size_t k_step_smem =
      (size_t)(block.x + 2 * WENO_HALO) * (size_t)(block.y + 2 * WENO_HALO) *
      (size_t)(block.z + 2 * WENO_HALO) *
      (6 * sizeof(float) + sizeof(uint8_t));
  ck(cudaFuncSetAttribute(k_step, cudaFuncAttributeMaxDynamicSharedMemorySize,
                          (int)k_step_smem),
     "k_step shared-memory attribute");

  k_build_solid_mask<<<grid, block>>>(d_solid);
  ck(cudaGetLastError(), "k_build_solid_mask launch");
  ck(cudaDeviceSynchronize(), "k_build_solid_mask sync");

  reset_sim(grid, block, d_xi, d_phix, d_phiy, d_phiz, d_lam, d_zet, d_solid);

  std::vector<float> h_sch(N);
  std::vector<uint32_t> h_rgba((size_t)hp.nx * (size_t)hp.ny);

  InitWindow(1280, 720, "tau3d volume");
  SetTargetFPS(99999);

  Camera3D cam{};
  cam.position = Vector3{1.8f, 1.2f, 1.8f};
  cam.target = Vector3{0.5f, 0.5f, 0.5f};
  cam.up = Vector3{0.0f, 1.0f, 0.0f};
  cam.fovy = 55.0f;
  cam.projection = CAMERA_PERSPECTIVE;

  Image img{};
  img.data = h_rgba.data();
  img.width = hp.nx;
  img.height = hp.ny;
  img.mipmaps = 1;
  img.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;

  std::vector<Texture2D> texZ(hp.nz);
  for (int z = 0; z < hp.nz; z++)
    texZ[z] = LoadTextureFromImage(img);

  std::vector<int> order(hp.nz);
  for (int z = 0; z < hp.nz; z++)
    order[z] = z;

  float t = 1e-5f;
  float d_tau = 1e-3f;

  int zslice = hp.nz / 2;
  bool paused = false;
  bool log_scale = true;
  float a_gain = 0.55f; // opacity
  int z_stride = 1; // 1=all slices, 2=half, 3=third
  const int steps_per_frame = 2;

  while (!WindowShouldClose()) {
    if (IsKeyPressed(KEY_SPACE))
      paused = !paused;
    if (IsKeyPressed(KEY_L))
      log_scale = !log_scale;
    if (IsKeyPressed(KEY_M))
      vis_mode = (vis_mode + 1) % VIS_COUNT;
    if (IsKeyPressed(KEY_R)) {
      t = 1e-5f;
      d_tau = 1e-3f;
      reset_sim(grid, block, d_xi, d_phix, d_phiy, d_phiz, d_lam, d_zet,
                d_solid);
    }
    if (IsKeyPressed(KEY_MINUS) || IsKeyPressed(KEY_KP_SUBTRACT))
      a_gain = fmaxf(0.05f, a_gain * 0.85f);
    if (IsKeyPressed(KEY_EQUAL) || IsKeyPressed(KEY_KP_ADD))
      a_gain = fminf(2.00f, a_gain * 1.18f);

    if (IsKeyPressed(KEY_ONE))
      z_stride = 1;
    if (IsKeyPressed(KEY_TWO))
      z_stride = 2;
    if (IsKeyPressed(KEY_THREE))
      z_stride = 3;
    if (IsKeyPressed(KEY_LEFT_BRACKET))
      zslice = (zslice - 1 + hp.nz) % hp.nz;
    if (IsKeyPressed(KEY_RIGHT_BRACKET))
      zslice = (zslice + 1) % hp.nz;

    float maxs = 0.f;
    float refl_dp = 0.f;
    float dt = 0.f;

    if (!paused) {
      for (int s = 0; s < steps_per_frame; s++) {
        t *= expf(d_tau);
        dt = t * d_tau;
        float ramp = t / 0.02f;
        float inflow_gain = fminf(fmaxf(ramp, 0.f), 1.f);

        float zero = 0.f;
        ck(cudaMemcpy(d_maxs, &zero, sizeof(float), cudaMemcpyHostToDevice),
           "set maxs");

        k_step<<<grid, block, k_step_smem>>>(d_xi, d_phix, d_phiy, d_phiz,
                                              d_lam, d_zet, d_xi2, d_phix2,
                                              d_phiy2, d_phiz2, d_lam2, d_zet2,
                                              dt, inflow_gain, d_maxs, d_solid);
        ck(cudaGetLastError(), "k_step launch");

        ck(cudaMemcpy(&maxs, d_maxs, sizeof(float), cudaMemcpyDeviceToHost),
           "get maxs");

        float dt_cfl = hp.cfl / fmaxf(maxs, 1e-9f);
        if (dt > dt_cfl)
          d_tau *= 0.5f;
        else if (dt < 0.25f * dt_cfl)
          d_tau *= 1.2f;

        std::swap(d_xi, d_xi2);
        std::swap(d_phix, d_phix2);
        std::swap(d_phiy, d_phiy2);
        std::swap(d_phiz, d_phiz2);
        std::swap(d_lam, d_lam2);
        std::swap(d_zet, d_zet2);
      }
    }

    k_vis<<<grid, block>>>(d_xi, d_phix, d_phiy, d_phiz, d_lam, d_zet,
                           d_solid, d_vis, vis_mode);
    ck(cudaGetLastError(), "k_vis launch");
    ck(cudaDeviceSynchronize(), "vis sync");

    ck(cudaMemcpy(h_sch.data(), d_vis, bytes, cudaMemcpyDeviceToHost),
       "copy vis");
    ck(cudaDeviceSynchronize(), "sch sync");

    {
      float zero = 0.f;
      ck(cudaMemcpy(d_reflect, &zero, sizeof(float), cudaMemcpyHostToDevice),
         "set reflect metric");
      k_outflow_reflection_metric<<<grid, block>>>(
          d_xi, d_phix, d_phiy, d_phiz, d_lam, d_zet, d_reflect, 6);
      ck(cudaGetLastError(), "k_outflow_reflection_metric launch");
      ck(cudaMemcpy(&refl_dp, d_reflect, sizeof(float), cudaMemcpyDeviceToHost),
         "get reflect metric");
    }

    for (int z = 0; z < hp.nz; z += z_stride) {
      slice_to_rgba(h_rgba.data(), h_sch.data(), hp.nx, hp.ny, hp.nz, z,
                    log_scale, a_gain);
      UpdateTexture(texZ[z], h_rgba.data());
    }

    BeginDrawing();
    ClearBackground(BLACK);
    camera_orbit_pan_zoom(&cam);
    BeginMode3D(cam);

    DrawCubeWires(Vector3{0.5f, 0.5f, 0.5f}, 1.0f, 1.0f, 1.0f,
                  Color{90, 90, 90, 255});

    rlEnableDepthTest();
    rlDisableBackfaceCulling();
    rlEnableColorBlend();

    unsigned char alphaMul = 255; // extra global alpha knob if you want
    for (int zi = 0; zi < hp.nz; zi += z_stride) {
      int z = order[zi];
      float zz = (z + 0.5f) / (float)hp.nz;
      draw_slice_z(texZ[z], zz, alphaMul);
    }

    EndMode3D();

    DrawRectangle(10, 10, 760, 90, Fade(BLACK, 0.55f));
    DrawText(TextFormat(
                 "t=%g  dt=%g  d_tau=%g  maxs=%g  outflow |dp|=%g  a_gain=%g  "
                 "stride=%d mode=%s",
                 t, dt, d_tau, maxs, refl_dp, a_gain, z_stride,
                 vis_name(vis_mode)),
             18, 18, 18, RAYWHITE);
    DrawText("RMB orbit  |  MMB pan  |  wheel zoom  |  SPACE pause  |  L log  "
             "|  R reset  |  +/- opacity  |  1/2/3 stride",
             18, 42, 18, RAYWHITE);

    EndDrawing();
  }

  for (int z = 0; z < hp.nz; z++)
    UnloadTexture(texZ[z]);
  CloseWindow();

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
  ck(cudaFree(d_reflect), "free");
  ck(cudaFree(d_solid), "free");

  return 0;
}
