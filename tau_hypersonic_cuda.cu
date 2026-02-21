// tau_hypersonic_cuda.cu
//
// nvcc -O3 -o tau_2d_hypersonic_cuda tau_hypersonic_cuda.cu -std=c++17 -lraylib
//
// Controls: SPACE pause, R reset, M mode
//
// View modes:
//   0 log(rho)
//   1 log(p)
//   2 speed
//   3 schlieren (|∇rho|)
//   4 vorticity (asinh(omega))
//   5 Mach
//   6 log(p/rho)

#ifndef TAU_HYPERSONIC_CUDA_NO_RAYLIB
#include "raylib.h"
#endif
#include <cuda_runtime.h>
#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W 1600
#define H 512
#define SCALE 1

#define EPS_RHO 1e-25
#define EPS_P 1e-25

constexpr double kPi = 3.14159265358979323846;

struct SimConfig {
  double gamma;
  double cfl;
  double visc_nu;
  double visc_rho;
  double visc_e;
  double inflow_mach;
  double geom_x0;
  double geom_cy;
  double geom_Rb;
  double geom_Rn;
  double geom_theta;
  int steps_per_frame;
};

__constant__ SimConfig d_cfg;

static inline void ck(cudaError_t e, const char *msg) {
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(e));
    exit(1);
  }
}
#define CK(x) ck((x), #x)

static void validate_reduction_launch_config(int threads,
                                             size_t shared_bytes_per_array) {
  if (threads <= 0 || (threads & (threads - 1)) != 0) {
    fprintf(stderr,
            "Invalid thread count %d: reductions require a positive power-of-two block size.\n",
            threads);
    exit(1);
  }

  int dev = 0;
  CK(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  CK(cudaGetDeviceProperties(&prop, dev));

  if (threads > prop.maxThreadsPerBlock) {
    fprintf(stderr,
            "Invalid thread count %d: device maxThreadsPerBlock is %d.\n",
            threads, prop.maxThreadsPerBlock);
    exit(1);
  }

  size_t shared_bytes = shared_bytes_per_array * (size_t)threads;
  if (shared_bytes > prop.sharedMemPerBlock) {
    fprintf(stderr,
            "Invalid shared memory request %zu bytes: device sharedMemPerBlock is %zu bytes.\n",
            shared_bytes, (size_t)prop.sharedMemPerBlock);
    exit(1);
  }
}

typedef struct {
  double *rho, *mx, *my, *E;
} Usoa;

typedef struct {
  double *rho, *mx, *my, *E;
} Csoa;

struct Cons {
  double rho, mx, my, E;
};

struct Prim {
  double rho, u, v, p;
};

struct FacePrim {
  Prim L, R;
};

// d_*
__device__ __forceinline__ int d_idx(int x, int y) { return y * W + x; }
__device__ __forceinline__ double d_fmax(double a, double b) {
  return a > b ? a : b;
}
__device__ __forceinline__ double d_fmin(double a, double b) {
  return a < b ? a : b;
}
__device__ __forceinline__ double d_fabs(double a) { return a < 0 ? -a : a; }
__device__ __forceinline__ double d_isfinite(double x) {
  return isfinite(x) ? 1.0 : 0.0;
}

// prims and cons
__device__ __forceinline__ Prim cons_to_prim(Cons c) {
  Prim p;
  double rho = d_fmax(c.rho, EPS_RHO);
  double inv = 1.0 / rho;
  double u = c.mx * inv;
  double v = c.my * inv;

  double kin = 0.5 * rho * (u * u + v * v);
  double eint = c.E - kin;
  double pr = (d_cfg.gamma - 1.0) * d_fmax(eint, EPS_P);

  p.rho = rho;
  p.u = u;
  p.v = v;
  p.p = pr;
  return p;
}

__device__ __forceinline__ Cons prim_to_cons(Prim p) {
  Cons c;
  double rho = d_fmax(p.rho, EPS_RHO);
  double pr = d_fmax(p.p, EPS_P);
  c.rho = rho;
  c.mx = rho * p.u;
  c.my = rho * p.v;
  c.E = pr / (d_cfg.gamma - 1.0) + 0.5 * rho * (p.u * p.u + p.v * p.v);
  return c;
}

__device__ __forceinline__ double sound_speed(Prim p) {
  return sqrt(d_cfg.gamma * d_fmax(p.p, EPS_P) / d_fmax(p.rho, EPS_RHO));
}

enum class Axis : int { X = 0, Y = 1 };

template <Axis AX>
__device__ __forceinline__ double prim_normal(const Prim &p) {
  if constexpr (AX == Axis::X) {
    return p.u;
  }
  return p.v;
}

template <Axis AX>
__device__ __forceinline__ double prim_tangent(const Prim &p) {
  if constexpr (AX == Axis::X) {
    return p.v;
  }
  return p.u;
}

template <Axis AX>
__device__ __forceinline__ Cons flux_axis(Cons c) {
  Prim p = cons_to_prim(c);
  Cons f;
  double un = prim_normal<AX>(p);
  f.rho = (AX == Axis::X) ? c.mx : c.my;
  f.mx = (AX == Axis::X) ? (c.mx * un + p.p) : (c.mx * un);
  f.my = (AX == Axis::X) ? (c.my * un) : (c.my * un + p.p);
  f.E = (c.E + p.p) * un;
  return f;
}

template <Axis AX>
__device__ __forceinline__ Cons flux_axis(Prim p) {
  return flux_axis<AX>(prim_to_cons(p));
}

__device__ __forceinline__ Cons flux_x(Cons c) { return flux_axis<Axis::X>(c); }

__device__ __forceinline__ Cons flux_x(Prim p) { return flux_axis<Axis::X>(p); }

__device__ __forceinline__ Cons flux_y(Cons c) { return flux_axis<Axis::Y>(c); }

__device__ __forceinline__ Cons flux_y(Prim p) { return flux_axis<Axis::Y>(p); }

__device__ __forceinline__ double minmod(double a, double b) {
  if (a * b <= 0.0)
    return 0.0;
  return (d_fabs(a) < d_fabs(b)) ? a : b;
}

__device__ __forceinline__ double mc_limiter(double dl, double dc, double dr) {
  double mm1 = minmod(dl, dr);
  double mm2 = minmod(dc, 2.0 * dl);
  double mm3 = minmod(dc, 2.0 * dr);
  return minmod(mm1, minmod(mm2, mm3));
}

__device__ __forceinline__ Prim inflow_state() {
  const double mach = d_cfg.inflow_mach;
  const double rho = 1.0;
  const double p = 1.0;
  double a = sqrt(d_cfg.gamma * p / rho);
  double u = mach * a;
  double v = 0;
  return Prim{rho, u, v, p};
}

__device__ __forceinline__ Cons load_cons(const Usoa U, int i) {
  return Cons{U.rho[i], U.mx[i], U.my[i], U.E[i]};
}

__device__ __forceinline__ void store_cons(Usoa U, int i, Cons c) {
  U.rho[i] = c.rho;
  U.mx[i] = c.mx;
  U.my[i] = c.my;
  U.E[i] = c.E;
}

__device__ __forceinline__ Cons load_cons(const Csoa A, int i) {
  return Cons{A.rho[i], A.mx[i], A.my[i], A.E[i]};
}

__device__ __forceinline__ void store_cons(Csoa A, int i, Cons c) {
  A.rho[i] = c.rho;
  A.mx[i] = c.mx;
  A.my[i] = c.my;
  A.E[i] = c.E;
}

__device__ __forceinline__ Prim wall_ghost_prim(Prim inside) {
  // Wall model: reflective no-slip ghost state built from the adjacent fluid
  // cell. We preserve thermodynamic state and flip velocity so the wall face
  // velocity is zero at the interface (instead of imposing a fixed stagnation
  // state unrelated to local interior conditions).
  return Prim{inside.rho, -inside.u, -inside.v, inside.p};
}

__device__ __forceinline__ Cons neighbor_or_wall(const Usoa U,
                                                 const uint8_t *mask, int x,
                                                 int y, int dx, int dy) {
  int xn = x + dx;
  int yn = y + dy;

  if (yn < 0)
    yn = 0;
  if (yn >= H)
    yn = H - 1;

  if (xn < 0) {
    return prim_to_cons(inflow_state());
  }
  if (xn >= W) {
    return load_cons(U, d_idx(W - 1, yn));
  }

  int j = d_idx(xn, yn);
  if (mask[j]) {
    return prim_to_cons(wall_ghost_prim(cons_to_prim(load_cons(U, d_idx(x, y)))));
  }
  return load_cons(U, j);
}

__device__ __forceinline__ Cons neighbor_for_diff(const Usoa U,
                                                  const uint8_t *mask, int xc,
                                                  int yc, int xn, int yn) {
  if (yn < 0)
    yn = 0;
  if (yn >= H)
    yn = H - 1;

  if (xn < 0) {
    return prim_to_cons(inflow_state());
  }
  if (xn >= W) {
    return load_cons(U, d_idx(W - 1, yn));
  }

  int j = d_idx(xn, yn);
  if (mask[j]) {
    return prim_to_cons(
        wall_ghost_prim(cons_to_prim(load_cons(U, d_idx(xc, yc)))));
  }
  return load_cons(U, j);
}

struct TileView {
  const double *rho;
  const double *mx;
  const double *my;
  const double *E;
  const uint8_t *mask;
  int tileW;
  int tileH;
  int x0;
  int y0;
};

__device__ __forceinline__ int tile_index(const TileView &tv, int lx, int ly) {
  return ly * tv.tileW + lx;
}

__device__ __forceinline__ bool tile_contains(const TileView &tv, int x, int y) {
  return x >= tv.x0 && x < (tv.x0 + tv.tileW) && y >= tv.y0 && y < (tv.y0 + tv.tileH);
}

__device__ __forceinline__ Cons tile_load_cons(const TileView &tv, int x, int y) {
  int li = tile_index(tv, x - tv.x0, y - tv.y0);
  return Cons{tv.rho[li], tv.mx[li], tv.my[li], tv.E[li]};
}

__device__ __forceinline__ uint8_t tile_load_mask(const TileView &tv, int x, int y) {
  int li = tile_index(tv, x - tv.x0, y - tv.y0);
  return tv.mask[li];
}

__device__ __forceinline__ Cons load_neighbor_or_wall_tiled(
    const Usoa U, const uint8_t *mask, const TileView &tv,
    const Prim &centerPrim, int xn, int yn) {
  if (yn < 0)
    yn = 0;
  if (yn >= H)
    yn = H - 1;

  if (xn < 0) {
    return prim_to_cons(inflow_state());
  }
  if (xn >= W) {
    return load_cons(U, d_idx(W - 1, yn));
  }

  uint8_t m = tile_contains(tv, xn, yn) ? tile_load_mask(tv, xn, yn)
                                        : mask[d_idx(xn, yn)];
  if (m) {
    return prim_to_cons(wall_ghost_prim(centerPrim));
  }
  return tile_contains(tv, xn, yn) ? tile_load_cons(tv, xn, yn)
                                   : load_cons(U, d_idx(xn, yn));
}

__device__ __forceinline__ void enforce_positive_faces(Prim &qm, const Prim &qc,
                                                       Prim &qp) {
  for (int it = 0; it < 8; it++) {
    int bad = 0;
    if (qm.rho <= EPS_RHO || qp.rho <= EPS_RHO)
      bad = 1;
    if (qm.p <= EPS_P || qp.p <= EPS_P)
      bad = 1;
    if (!bad)
      return;

    qm.rho = 0.5 * (qm.rho + qc.rho);
    qm.u = 0.5 * (qm.u + qc.u);
    qm.v = 0.5 * (qm.v + qc.v);
    qm.p = 0.5 * (qm.p + qc.p);

    qp.rho = 0.5 * (qp.rho + qc.rho);
    qp.u = 0.5 * (qp.u + qc.u);
    qp.v = 0.5 * (qp.v + qc.v);
    qp.p = 0.5 * (qp.p + qc.p);
  }
  qm.rho = d_fmax(qm.rho, EPS_RHO);
  qp.rho = d_fmax(qp.rho, EPS_RHO);
  qm.p = d_fmax(qm.p, EPS_P);
  qp.p = d_fmax(qp.p, EPS_P);
}

template <Axis AX>
__device__ __forceinline__ FacePrim reconstruct_axis(const Usoa U,
                                                     const uint8_t *mask,
                                                     int x, int y) {
  constexpr int DX = (AX == Axis::X) ? 1 : 0;
  constexpr int DY = (AX == Axis::X) ? 0 : 1;

  Cons Uc = load_cons(U, d_idx(x, y));
  Cons Um = neighbor_or_wall(U, mask, x, y, -DX, -DY);
  Cons Up = neighbor_or_wall(U, mask, x, y, +DX, +DY);

  Prim qm = cons_to_prim(Um);
  Prim qc = cons_to_prim(Uc);
  Prim qp = cons_to_prim(Up);

  double dl_rho = qc.rho - qm.rho, dr_rho = qp.rho - qc.rho;
  double dc_rho = 0.5 * (qp.rho - qm.rho);
  double s_rho = mc_limiter(dl_rho, dc_rho, dr_rho);

  double dl_u = qc.u - qm.u, dr_u = qp.u - qc.u;
  double dc_u = 0.5 * (qp.u - qm.u);
  double s_u = mc_limiter(dl_u, dc_u, dr_u);

  double dl_v = qc.v - qm.v, dr_v = qp.v - qc.v;
  double dc_v = 0.5 * (qp.v - qm.v);
  double s_v = mc_limiter(dl_v, dc_v, dr_v);

  double dl_p = qc.p - qm.p, dr_p = qp.p - qc.p;
  double dc_p = 0.5 * (qp.p - qm.p);
  double s_p = mc_limiter(dl_p, dc_p, dr_p);

  Prim qL{qc.rho - 0.5 * s_rho, qc.u - 0.5 * s_u, qc.v - 0.5 * s_v,
          qc.p - 0.5 * s_p};
  Prim qR{qc.rho + 0.5 * s_rho, qc.u + 0.5 * s_u, qc.v + 0.5 * s_v,
          qc.p + 0.5 * s_p};

  enforce_positive_faces(qL, qc, qR);
  return FacePrim{qL, qR};
}

__device__ __forceinline__ FacePrim reconstruct_x(const Usoa U,
                                                  const uint8_t *mask, int x,
                                                  int y) {
  return reconstruct_axis<Axis::X>(U, mask, x, y);
}

__device__ __forceinline__ FacePrim reconstruct_y(const Usoa U,
                                                  const uint8_t *mask, int x,
                                                  int y) {
  return reconstruct_axis<Axis::Y>(U, mask, x, y);
}

template <Axis AX>
__device__ __forceinline__ Prim half_step_predict_axis(Prim q, Cons dF,
                                                       double half_dt_dn) {
  (void)AX;
  Cons c = prim_to_cons(q);
  c.rho -= half_dt_dn * dF.rho;
  c.mx -= half_dt_dn * dF.mx;
  c.my -= half_dt_dn * dF.my;
  c.E -= half_dt_dn * dF.E;
  Prim out = cons_to_prim(c);
  out.rho = d_fmax(out.rho, EPS_RHO);
  out.p = d_fmax(out.p, EPS_P);
  return out;
}

__device__ __forceinline__ Prim half_step_predict_x(Prim q, double dF_rho,
                                                    double dF_mx, double dF_my,
                                                    double dF_E,
                                                    double half_dt_dx) {
  return half_step_predict_axis<Axis::X>(
      q, Cons{dF_rho, dF_mx, dF_my, dF_E}, half_dt_dx);
}

__device__ __forceinline__ Prim half_step_predict_y(Prim q, double dG_rho,
                                                    double dG_mx, double dG_my,
                                                    double dG_E,
                                                    double half_dt_dy) {
  return half_step_predict_axis<Axis::Y>(
      q, Cons{dG_rho, dG_mx, dG_my, dG_E}, half_dt_dy);
}

__device__ __forceinline__ Cons cons_sub(Cons a, Cons b) {
  return Cons{a.rho - b.rho, a.mx - b.mx, a.my - b.my, a.E - b.E};
}
__device__ __forceinline__ Cons cons_add(Cons a, Cons b) {
  return Cons{a.rho + b.rho, a.mx + b.mx, a.my + b.my, a.E + b.E};
}
__device__ __forceinline__ Cons cons_mul(double s, Cons a) {
  return Cons{s * a.rho, s * a.mx, s * a.my, s * a.E};
}

template <Axis AX>
__device__ __forceinline__ Cons hlle_axis(Cons UL, Cons UR) {
  Prim L = cons_to_prim(UL);
  Prim R = cons_to_prim(UR);
  double uL = prim_normal<AX>(L);
  double uR = prim_normal<AX>(R);
  double aL = sound_speed(L);
  double aR = sound_speed(R);
  double SL = d_fmin(uL - aL, uR - aR);
  double SR = d_fmax(uL + aL, uR + aR);

  Cons FL = flux_axis<AX>(UL);
  Cons FR = flux_axis<AX>(UR);

  if (SL >= 0.0)
    return FL;
  if (SR <= 0.0)
    return FR;

  double denom = SR - SL;
  if (d_fabs(denom) < 1e-14)
    return cons_mul(0.5, cons_add(FL, FR));

  Cons term1 = cons_mul(SR, FL);
  Cons term2 = cons_mul(-SL, FR);
  Cons term3 = cons_mul(SL * SR, cons_sub(UR, UL));
  return cons_mul(1.0 / denom, cons_add(cons_add(term1, term2), term3));
}

__device__ __forceinline__ Cons hlle_x(Cons UL, Cons UR) {
  return hlle_axis<Axis::X>(UL, UR);
}

__device__ __forceinline__ Cons hlle_y(Cons UL, Cons UR) {
  return hlle_axis<Axis::Y>(UL, UR);
}

template <Axis AX>
__device__ __forceinline__ Cons hllc_axis(Cons UL, Cons UR) {
  Prim L = cons_to_prim(UL);
  Prim R = cons_to_prim(UR);

  double unL = prim_normal<AX>(L);
  double unR = prim_normal<AX>(R);
  double utL = prim_tangent<AX>(L);
  double utR = prim_tangent<AX>(R);

  double aL = sound_speed(L);
  double aR = sound_speed(R);

  double SL = d_fmin(unL - aL, unR - aR);
  double SR = d_fmax(unL + aL, unR + aR);

  Cons FL = flux_axis<AX>(UL);
  Cons FR = flux_axis<AX>(UR);

  if (SL >= 0.0)
    return FL;
  if (SR <= 0.0)
    return FR;

  double rhoL = L.rho, rhoR = R.rho;
  double pL = L.p, pR = R.p;

  double num = pR - pL + rhoL * unL * (SL - unL) - rhoR * unR * (SR - unR);
  double den = rhoL * (SL - unL) - rhoR * (SR - unR);

  if (d_fabs(den) < 1e-14 || !isfinite(num) || !isfinite(den)) {
    return hlle_axis<AX>(UL, UR);
  }

  double SM = num / den;
  if (!isfinite(SM))
    return hlle_axis<AX>(UL, UR);

  double pStar = pL + rhoL * (SL - unL) * (SM - unL);
  pStar = d_fmax(pStar, EPS_P);

  double dLS = (SL - SM);
  double dRS = (SR - SM);
  if (d_fabs(dLS) < 1e-14 || d_fabs(dRS) < 1e-14) {
    return hlle_axis<AX>(UL, UR);
  }

  double rhoStarL = rhoL * (SL - unL) / dLS;
  double rhoStarR = rhoR * (SR - unR) / dRS;

  if (!(rhoStarL > 0.0) || !(rhoStarR > 0.0) || !isfinite(rhoStarL) ||
      !isfinite(rhoStarR)) {
    return hlle_axis<AX>(UL, UR);
  }

  double momNL = rhoStarL * SM;
  double momTL = rhoStarL * utL;
  double EL = UL.E;
  double EStarL = ((SL - unL) * EL - pL * unL + pStar * SM) / dLS;
  if (!isfinite(EStarL))
    return hlle_axis<AX>(UL, UR);
  Cons UStarL = (AX == Axis::X)
                    ? Cons{rhoStarL, momNL, momTL, EStarL}
                    : Cons{rhoStarL, momTL, momNL, EStarL};

  double momNR = rhoStarR * SM;
  double momTR = rhoStarR * utR;
  double ER = UR.E;
  double EStarR = ((SR - unR) * ER - pR * unR + pStar * SM) / dRS;
  if (!isfinite(EStarR))
    return hlle_axis<AX>(UL, UR);
  Cons UStarR = (AX == Axis::X)
                    ? Cons{rhoStarR, momNR, momTR, EStarR}
                    : Cons{rhoStarR, momTR, momNR, EStarR};

  if (SM >= 0.0) {
    Cons F;
    F.rho = FL.rho + SL * (UStarL.rho - UL.rho);
    F.mx = FL.mx + SL * (UStarL.mx - UL.mx);
    F.my = FL.my + SL * (UStarL.my - UL.my);
    F.E = FL.E + SL * (UStarL.E - UL.E);
    return F;
  }

  Cons F;
  F.rho = FR.rho + SR * (UStarR.rho - UR.rho);
  F.mx = FR.mx + SR * (UStarR.mx - UR.mx);
  F.my = FR.my + SR * (UStarR.my - UR.my);
  F.E = FR.E + SR * (UStarR.E - UR.E);
  return F;
}

__device__ __forceinline__ Cons hllc_x(Cons UL, Cons UR) {
  return hllc_axis<Axis::X>(UL, UR);
}

__device__ __forceinline__ Cons hllc_x(Prim qL, Prim qR) {
  return hllc_axis<Axis::X>(prim_to_cons(qL), prim_to_cons(qR));
}

__device__ __forceinline__ Cons hllc_y(Cons UL, Cons UR) {
  return hllc_axis<Axis::Y>(UL, UR);
}

__device__ __forceinline__ Cons hllc_y(Prim qB, Prim qT) {
  return hllc_axis<Axis::Y>(prim_to_cons(qB), prim_to_cons(qT));
}

// device helpers
__device__ __forceinline__ double clamp01(double t) {
  return t < 0.0 ? 0.0 : (t > 1.0 ? 1.0 : t);
}

__device__ __forceinline__ double len2(double x, double y) {
  return sqrt(x * x + y * y);
}

__device__ __forceinline__ double sdSegment(double px, double py, double ax,
                                            double ay, double bx, double by) {
  double abx = bx - ax, aby = by - ay;
  double apx = px - ax, apy = py - ay;
  double denom = abx * abx + aby * aby + 1e-30;
  double t = (apx * abx + apy * aby) / denom;
  t = clamp01(t);
  double qx = ax + t * abx, qy = ay + t * aby;
  return len2(px - qx, py - qy);
}

__device__ __forceinline__ double
sdSphereConeCapsule(double x, double y, double Rb, double Rn, double theta) {
  double r = d_fabs(y);

  double st = sin(theta);
  double ct = cos(theta);
  double tt = tan(theta);

  double xt = Rn * (1.0 - st);
  double rt = Rn * ct;

  double xb = xt + (Rb - rt) / d_fmax(tt, 1e-30);

  double rprof = 0.0;
  if (x < 0.0) {
    rprof = -1.0;
  } else if (x <= xt) {
    double dx = x - Rn;
    double inside = Rn * Rn - dx * dx;
    rprof = (inside > 0.0) ? sqrt(inside) : 0.0;
  } else if (x <= xb) {
    rprof = rt + (x - xt) * tt;
  } else {
    rprof = -1.0;
  }

  int inside = (x >= 0.0 && x <= xb && r <= rprof);

  double d_sphere = d_fabs(len2(x - Rn, r) - Rn);
  double d_cone = sdSegment(x, r, xt, rt, xb, Rb);
  double d_base = sdSegment(x, y, xb, -Rb, xb, +Rb);
  double d_rim = len2(x - xb, r - Rb);

  double d = d_sphere;
  if (d_cone < d)
    d = d_cone;
  if (d_base < d)
    d = d_base;
  if (d_rim < d)
    d = d_rim;

  return inside ? -d : d;
}

__device__ __forceinline__ uchar4 pack_rgba(uint8_t r, uint8_t g, uint8_t b) {
  return uchar4{r, g, b, 255};
}

__device__ __forceinline__ void get_color(double t, uint8_t &r, uint8_t &g,
                                          uint8_t &b) {
  if (t < 0)
    t = 0;
  if (t > 1)
    t = 1;
  double rr = 255.0 * d_fmin(1.0, d_fmax(0.0, 3.0 * t - 1.0));
  double gg = 255.0 * d_fmin(1.0, d_fmax(0.0, 2.0 - 4.0 * d_fabs(t - 0.5)));
  double bb = 255.0 * d_fmin(1.0, d_fmax(0.0, 2.0 - 3.0 * t));
  r = (uint8_t)rr;
  g = (uint8_t)gg;
  b = (uint8_t)bb;
}

__device__ __forceinline__ Prim sample_prim_bc(const Usoa U,
                                               const uint8_t *mask, int xc,
                                               int yc, int x, int y) {
  if (y < 0)
    y = 0;
  if (y >= H)
    y = H - 1;

  if (x < 0)
    return inflow_state();
  if (x >= W) {
    int i = d_idx(W - 1, y);
    return cons_to_prim(load_cons(U, i));
  }

  int i = d_idx(x, y);
  if (mask[i]) {
    // Keep render-side boundary sampling consistent with solver ghost states,
    // especially for gradient-based views (schlieren and vorticity).
    Prim interior = cons_to_prim(load_cons(U, d_idx(xc, yc)));
    return wall_ghost_prim(interior);
  }
  return cons_to_prim(load_cons(U, i));
}

__device__ __forceinline__ double spherecone_xb(double Rb, double Rn,
                                                double theta) {
  double st = sin(theta);
  double ct = cos(theta);
  double tt = tan(theta);
  double xt = Rn * (1.0 - st);
  double rt = Rn * ct;
  return xt + (Rb - rt) / d_fmax(tt, 1e-30);
}

// k_*
__global__ void k_init(Usoa U, uint8_t *mask) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = W * H;
  if (i >= N)
    return;

  int x = i % W;
  int y = i / W;

  double x0 = d_cfg.geom_x0;
  double cy = d_cfg.geom_cy;

  double Rb = d_cfg.geom_Rb;
  double Rn = d_cfg.geom_Rn;
  double theta = d_cfg.geom_theta;

  double X = (double)x - x0;
  double Y = (double)y - cy;

  double k_round = Rb;
  double xb = spherecone_xb(Rb, Rn, theta);
  double sd0 = sdSphereConeCapsule(X, Y, Rb, Rn, theta);
  double sd = sd0 - k_round;
  sd = d_fmax(sd, X - xb);
  uint8_t m = (sd < 0.0) ? 1 : 0;
  mask[i] = m;

  Prim inflow = inflow_state();
  Prim s = m ? Prim{inflow.rho, 0.0, 0.0, inflow.p} : inflow;
  store_cons(U, i, prim_to_cons(s));
}

__global__ void k_apply_inflow_left(Usoa U, const uint8_t *mask) {
  int y = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (y >= H)
    return;

  int i0 = d_idx(0, y);
  if (mask[i0])
    return;

  Prim infl = inflow_state();
  Prim pin{infl.rho, infl.u, infl.v, infl.p};
  store_cons(U, i0, prim_to_cons(pin));
}

__global__ void k_max_wavespeed_blocks(const Usoa U, const uint8_t *mask,
                                       double *blockMax) {
  extern __shared__ double smax[];
  int tid = threadIdx.x;
  int N = W * H;

  int i = (int)(blockIdx.x * blockDim.x + tid);
  double v = 1e-12;

  if (i < N && !mask[i]) {
    Prim p = cons_to_prim(load_cons(U, i));
    double a = sound_speed(p);
    double sx = d_fabs(p.u) + a;
    double sy = d_fabs(p.v) + a;
    v = (sx > sy) ? sx : sy;
    if (!isfinite(v))
      v = 1e-12;
  }

  smax[tid] = v;
  __syncthreads();

  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (tid < off) {
      double b = smax[tid + off];
      if (b > smax[tid])
        smax[tid] = b;
    }
    __syncthreads();
  }

  if (tid == 0)
    blockMax[blockIdx.x] = smax[0];
}

__global__ void k_reduce_block_max(const double *blockMax, int nBlocks,
                                   double *outMax) {
  extern __shared__ double smax[];
  int tid = threadIdx.x;
  double vmax = 1e-12;

  for (int i = tid; i < nBlocks; i += blockDim.x) {
    double v = blockMax[i];
    if (v > vmax)
      vmax = v;
  }

  smax[tid] = vmax;
  __syncthreads();

  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (tid < off) {
      double b = smax[tid + off];
      if (b > smax[tid])
        smax[tid] = b;
    }
    __syncthreads();
  }

  if (tid == 0)
    *outMax = smax[0];
}

__global__ void k_predict_face_states(Usoa U, const uint8_t *mask,
                                      Csoa xL_states, Csoa xR_states,
                                      Csoa yL_states, Csoa yR_states,
                                      double half_dt_dx,
                                      double half_dt_dy) {
  constexpr int HALO = 1;
  const int bdx = blockDim.x;
  const int bdy = blockDim.y;
  const int tileW = bdx + 2 * HALO;
  const int tileH = bdy + 2 * HALO;
  const int tileN = tileW * tileH;

  extern __shared__ unsigned char smem[];
  double *sRho = reinterpret_cast<double *>(smem);
  double *sMx = sRho + tileN;
  double *sMy = sMx + tileN;
  double *sE = sMy + tileN;
  uint8_t *sMask = reinterpret_cast<uint8_t *>(sE + tileN);

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = (int)(blockIdx.x * bdx + tx);
  int y = (int)(blockIdx.y * bdy + ty);

  int x0 = (int)(blockIdx.x * bdx) - HALO;
  int y0 = (int)(blockIdx.y * bdy) - HALO;

  int tflat = ty * bdx + tx;
  int tcount = bdx * bdy;
  for (int t = tflat; t < tileN; t += tcount) {
    int tlx = t % tileW;
    int tly = t / tileW;
    int gx = x0 + tlx;
    int gy = y0 + tly;

    int sx = gx;
    int sy = gy;
    if (sy < 0)
      sy = 0;
    if (sy >= H)
      sy = H - 1;
    if (sx < 0)
      sx = 0;
    if (sx >= W)
      sx = W - 1;

    int gi = d_idx(sx, sy);
    Cons c = load_cons(U, gi);
    sRho[t] = c.rho;
    sMx[t] = c.mx;
    sMy[t] = c.my;
    sE[t] = c.E;
    sMask[t] = mask[gi];
  }
  __syncthreads();

  int i = d_idx(x, y);
  int N = W * H;
  if (i >= N)
    return;

  TileView tv{sRho, sMx, sMy, sE, sMask, tileW, tileH, x0, y0};

  if (mask[i]) {
    Cons Uc = tile_load_cons(tv, x, y);
    store_cons(xL_states, i, Uc);
    store_cons(xR_states, i, Uc);
    store_cons(yL_states, i, Uc);
    store_cons(yR_states, i, Uc);
    return;
  }

  Cons Uc = tile_load_cons(tv, x, y);
  Prim qc = cons_to_prim(Uc);

  Cons Umx = load_neighbor_or_wall_tiled(U, mask, tv, qc, x - 1, y);
  Cons Upx = load_neighbor_or_wall_tiled(U, mask, tv, qc, x + 1, y);
  Prim qmX = cons_to_prim(Umx);
  Prim qpX = cons_to_prim(Upx);

  double dl_rho_x = qc.rho - qmX.rho, dr_rho_x = qpX.rho - qc.rho;
  double dc_rho_x = 0.5 * (qpX.rho - qmX.rho);
  double s_rho_x = mc_limiter(dl_rho_x, dc_rho_x, dr_rho_x);

  double dl_u_x = qc.u - qmX.u, dr_u_x = qpX.u - qc.u;
  double dc_u_x = 0.5 * (qpX.u - qmX.u);
  double s_u_x = mc_limiter(dl_u_x, dc_u_x, dr_u_x);

  double dl_v_x = qc.v - qmX.v, dr_v_x = qpX.v - qc.v;
  double dc_v_x = 0.5 * (qpX.v - qmX.v);
  double s_v_x = mc_limiter(dl_v_x, dc_v_x, dr_v_x);

  double dl_p_x = qc.p - qmX.p, dr_p_x = qpX.p - qc.p;
  double dc_p_x = 0.5 * (qpX.p - qmX.p);
  double s_p_x = mc_limiter(dl_p_x, dc_p_x, dr_p_x);

  FacePrim fpCx;
  fpCx.L = Prim{qc.rho - 0.5 * s_rho_x, qc.u - 0.5 * s_u_x,
                qc.v - 0.5 * s_v_x, qc.p - 0.5 * s_p_x};
  fpCx.R = Prim{qc.rho + 0.5 * s_rho_x, qc.u + 0.5 * s_u_x,
                qc.v + 0.5 * s_v_x, qc.p + 0.5 * s_p_x};
  enforce_positive_faces(fpCx.L, qc, fpCx.R);

  Cons fpCx_L = prim_to_cons(fpCx.L);
  Cons fpCx_R = prim_to_cons(fpCx.R);
  Cons fpCx_FL = flux_x(fpCx_L);
  Cons fpCx_FR = flux_x(fpCx_R);

  Prim qL = half_step_predict_x(
      fpCx.L, (fpCx_FR.rho - fpCx_FL.rho), (fpCx_FR.mx - fpCx_FL.mx),
      (fpCx_FR.my - fpCx_FL.my), (fpCx_FR.E - fpCx_FL.E), half_dt_dx);
  Prim qR = half_step_predict_x(
      fpCx.R, (fpCx_FR.rho - fpCx_FL.rho), (fpCx_FR.mx - fpCx_FL.mx),
      (fpCx_FR.my - fpCx_FL.my), (fpCx_FR.E - fpCx_FL.E), half_dt_dx);
  qL.rho = d_fmax(qL.rho, EPS_RHO);
  qL.p = d_fmax(qL.p, EPS_P);
  qR.rho = d_fmax(qR.rho, EPS_RHO);
  qR.p = d_fmax(qR.p, EPS_P);
  store_cons(xL_states, i, prim_to_cons(qL));
  store_cons(xR_states, i, prim_to_cons(qR));

  Cons Umy = load_neighbor_or_wall_tiled(U, mask, tv, qc, x, y - 1);
  Cons Upy = load_neighbor_or_wall_tiled(U, mask, tv, qc, x, y + 1);
  Prim qmY = cons_to_prim(Umy);
  Prim qpY = cons_to_prim(Upy);

  double dl_rho_y = qc.rho - qmY.rho, dr_rho_y = qpY.rho - qc.rho;
  double dc_rho_y = 0.5 * (qpY.rho - qmY.rho);
  double s_rho_y = mc_limiter(dl_rho_y, dc_rho_y, dr_rho_y);

  double dl_u_y = qc.u - qmY.u, dr_u_y = qpY.u - qc.u;
  double dc_u_y = 0.5 * (qpY.u - qmY.u);
  double s_u_y = mc_limiter(dl_u_y, dc_u_y, dr_u_y);

  double dl_v_y = qc.v - qmY.v, dr_v_y = qpY.v - qc.v;
  double dc_v_y = 0.5 * (qpY.v - qmY.v);
  double s_v_y = mc_limiter(dl_v_y, dc_v_y, dr_v_y);

  double dl_p_y = qc.p - qmY.p, dr_p_y = qpY.p - qc.p;
  double dc_p_y = 0.5 * (qpY.p - qmY.p);
  double s_p_y = mc_limiter(dl_p_y, dc_p_y, dr_p_y);

  FacePrim fpCy;
  fpCy.L = Prim{qc.rho - 0.5 * s_rho_y, qc.u - 0.5 * s_u_y,
                qc.v - 0.5 * s_v_y, qc.p - 0.5 * s_p_y};
  fpCy.R = Prim{qc.rho + 0.5 * s_rho_y, qc.u + 0.5 * s_u_y,
                qc.v + 0.5 * s_v_y, qc.p + 0.5 * s_p_y};
  enforce_positive_faces(fpCy.L, qc, fpCy.R);

  Cons fpCy_L = prim_to_cons(fpCy.L);
  Cons fpCy_R = prim_to_cons(fpCy.R);
  Cons fpCy_GL = flux_y(fpCy_L);
  Cons fpCy_GR = flux_y(fpCy_R);

  Prim qB = half_step_predict_y(
      fpCy.L, (fpCy_GR.rho - fpCy_GL.rho), (fpCy_GR.mx - fpCy_GL.mx),
      (fpCy_GR.my - fpCy_GL.my), (fpCy_GR.E - fpCy_GL.E), half_dt_dy);
  Prim qT = half_step_predict_y(
      fpCy.R, (fpCy_GR.rho - fpCy_GL.rho), (fpCy_GR.mx - fpCy_GL.mx),
      (fpCy_GR.my - fpCy_GL.my), (fpCy_GR.E - fpCy_GL.E), half_dt_dy);
  qB.rho = d_fmax(qB.rho, EPS_RHO);
  qB.p = d_fmax(qB.p, EPS_P);
  qT.rho = d_fmax(qT.rho, EPS_RHO);
  qT.p = d_fmax(qT.p, EPS_P);
  store_cons(yL_states, i, prim_to_cons(qB));
  store_cons(yR_states, i, prim_to_cons(qT));
}

__global__ void k_compute_xface_flux(Usoa U, const uint8_t *mask,
                                     Csoa xL_states, Csoa xR_states,
                                     Csoa xFlux) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int NF = (W + 1) * H;
  if (i >= NF)
    return;

  int fx = i % (W + 1);
  int y = i / (W + 1);
  int xl = fx - 1;
  int xr = fx;

  bool hasL = (xl >= 0) && !mask[d_idx(xl, y)];
  bool hasR = (xr < W) && !mask[d_idx(xr, y)];

  Cons UL{}, UR{};
  if (hasL && hasR) {
    UL = load_cons(xR_states, d_idx(xl, y));
    UR = load_cons(xL_states, d_idx(xr, y));
  } else if (hasR) {
    UL = neighbor_or_wall(U, mask, xr, y, -1, 0);
    UR = load_cons(xL_states, d_idx(xr, y));
  } else if (hasL) {
    UL = load_cons(xR_states, d_idx(xl, y));
    UR = neighbor_or_wall(U, mask, xl, y, +1, 0);
  } else {
    store_cons(xFlux, i, Cons{0.0, 0.0, 0.0, 0.0});
    return;
  }

  store_cons(xFlux, i, hllc_x(UL, UR));
}

__global__ void k_compute_yface_flux(Usoa U, const uint8_t *mask,
                                     Csoa yL_states, Csoa yR_states,
                                     Csoa yFlux) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int NF = W * (H + 1);
  if (i >= NF)
    return;

  int x = i % W;
  int fy = i / W;
  int yb = fy - 1;
  int yt = fy;

  bool hasB = (yb >= 0) && !mask[d_idx(x, yb)];
  bool hasT = (yt < H) && !mask[d_idx(x, yt)];

  Cons UB{}, UT{};
  if (hasB && hasT) {
    UB = load_cons(yR_states, d_idx(x, yb));
    UT = load_cons(yL_states, d_idx(x, yt));
  } else if (hasT) {
    UB = neighbor_or_wall(U, mask, x, yt, 0, -1);
    UT = load_cons(yL_states, d_idx(x, yt));
  } else if (hasB) {
    UB = load_cons(yR_states, d_idx(x, yb));
    UT = neighbor_or_wall(U, mask, x, yb, 0, +1);
  } else {
    store_cons(yFlux, i, Cons{0.0, 0.0, 0.0, 0.0});
    return;
  }

  store_cons(yFlux, i, hllc_y(UB, UT));
}

__global__ void k_step(Usoa U, Usoa Uout, const uint8_t *mask, Csoa xFlux,
                       Csoa yFlux, double dt, double dt_dx, double dt_dy) {
  constexpr int HALO = 2;
  const int bdx = blockDim.x;
  const int bdy = blockDim.y;
  const int tileW = bdx + 2 * HALO;
  const int tileH = bdy + 2 * HALO;
  const int tileN = tileW * tileH;

  extern __shared__ unsigned char smem[];
  double *sRho = reinterpret_cast<double *>(smem);
  double *sMx = sRho + tileN;
  double *sMy = sMx + tileN;
  double *sE = sMy + tileN;
  uint8_t *sMask = reinterpret_cast<uint8_t *>(sE + tileN);

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int x = (int)(blockIdx.x * bdx + tx);
  int y = (int)(blockIdx.y * bdy + ty);

  int x0 = (int)(blockIdx.x * bdx) - HALO;
  int y0 = (int)(blockIdx.y * bdy) - HALO;

  int tflat = ty * bdx + tx;
  int tcount = bdx * bdy;
  for (int t = tflat; t < tileN; t += tcount) {
    int tlx = t % tileW;
    int tly = t / tileW;
    int gx = x0 + tlx;
    int gy = y0 + tly;

    int sx = gx;
    int sy = gy;
    if (sy < 0)
      sy = 0;
    if (sy >= H)
      sy = H - 1;
    if (sx < 0)
      sx = 0;
    if (sx >= W)
      sx = W - 1;

    int gi = d_idx(sx, sy);
    Cons c = load_cons(U, gi);
    sRho[t] = c.rho;
    sMx[t] = c.mx;
    sMy[t] = c.my;
    sE[t] = c.E;
    sMask[t] = mask[gi];
  }
  __syncthreads();

  int i = d_idx(x, y);
  int N = W * H;
  if (i >= N)
    return;

  TileView tv{sRho, sMx, sMy, sE, sMask, tileW, tileH, x0, y0};

  if (mask[i]) {
    store_cons(Uout, i, tile_load_cons(tv, x, y));
    return;
  }

  Cons FxL = load_cons(xFlux, y * (W + 1) + x);
  Cons FxR = load_cons(xFlux, y * (W + 1) + (x + 1));
  Cons GyB = load_cons(yFlux, y * W + x);
  Cons GyT = load_cons(yFlux, (y + 1) * W + x);

  // Hyperbolic update
  Cons Uc = tile_load_cons(tv, x, y);
  Prim centerPrim = cons_to_prim(Uc);
  Cons Un = Uc;

  Un.rho -= dt_dx * (FxR.rho - FxL.rho);
  Un.mx -= dt_dx * (FxR.mx - FxL.mx);
  Un.my -= dt_dx * (FxR.my - FxL.my);
  Un.E -= dt_dx * (FxR.E - FxL.E);

  Un.rho -= dt_dy * (GyT.rho - GyB.rho);
  Un.mx -= dt_dy * (GyT.mx - GyB.mx);
  Un.my -= dt_dy * (GyT.my - GyB.my);
  Un.E -= dt_dy * (GyT.E - GyB.E);

  // Diffusion (25-pt via separable 5-tap 2nd derivative)
  {
    const double inv12 = 1.0 / 12.0;

    Cons xm2 = load_neighbor_or_wall_tiled(U, mask, tv, centerPrim, x - 2, y);
    Cons xm1 = load_neighbor_or_wall_tiled(U, mask, tv, centerPrim, x - 1, y);
    Cons xp1 = load_neighbor_or_wall_tiled(U, mask, tv, centerPrim, x + 1, y);
    Cons xp2 = load_neighbor_or_wall_tiled(U, mask, tv, centerPrim, x + 2, y);

    double d2x_rho =
        (-xm2.rho + 16.0 * xm1.rho - 30.0 * Uc.rho + 16.0 * xp1.rho - xp2.rho) *
        inv12;
    double d2x_mx =
        (-xm2.mx + 16.0 * xm1.mx - 30.0 * Uc.mx + 16.0 * xp1.mx - xp2.mx) *
        inv12;
    double d2x_my =
        (-xm2.my + 16.0 * xm1.my - 30.0 * Uc.my + 16.0 * xp1.my - xp2.my) *
        inv12;
    double d2x_E =
        (-xm2.E + 16.0 * xm1.E - 30.0 * Uc.E + 16.0 * xp1.E - xp2.E) * inv12;

    Cons ym2 = load_neighbor_or_wall_tiled(U, mask, tv, centerPrim, x, y - 2);
    Cons ym1 = load_neighbor_or_wall_tiled(U, mask, tv, centerPrim, x, y - 1);
    Cons yp1 = load_neighbor_or_wall_tiled(U, mask, tv, centerPrim, x, y + 1);
    Cons yp2 = load_neighbor_or_wall_tiled(U, mask, tv, centerPrim, x, y + 2);

    double d2y_rho =
        (-ym2.rho + 16.0 * ym1.rho - 30.0 * Uc.rho + 16.0 * yp1.rho - yp2.rho) *
        inv12;
    double d2y_mx =
        (-ym2.mx + 16.0 * ym1.mx - 30.0 * Uc.mx + 16.0 * yp1.mx - yp2.mx) *
        inv12;
    double d2y_my =
        (-ym2.my + 16.0 * ym1.my - 30.0 * Uc.my + 16.0 * yp1.my - yp2.my) *
        inv12;
    double d2y_E =
        (-ym2.E + 16.0 * ym1.E - 30.0 * Uc.E + 16.0 * yp1.E - yp2.E) * inv12;

    double lap_rho = d2x_rho + d2y_rho;
    double lap_mx = d2x_mx + d2y_mx;
    double lap_my = d2x_my + d2y_my;
    double lap_E = d2x_E + d2y_E;

    Un.rho += (d_cfg.visc_rho * dt) * lap_rho;
    Un.mx += (d_cfg.visc_nu * dt) * lap_mx;
    Un.my += (d_cfg.visc_nu * dt) * lap_my;
    Un.E += (d_cfg.visc_e * dt) * lap_E;
  }

  Un.rho = d_fmax(Un.rho, EPS_RHO);
  Prim pp = cons_to_prim(Un);
  if (pp.p <= EPS_P || !isfinite(pp.p) || !isfinite(pp.rho) ||
      !isfinite(pp.u) || !isfinite(pp.v)) {
    pp.rho = d_fmax(pp.rho, EPS_RHO);
    pp.p = d_fmax(pp.p, EPS_P);
    Un = prim_to_cons(pp);
  }

  store_cons(Uout, i, Un);
}

__global__ void k_render_vals(const Usoa U, const uint8_t *mask, int view_mode,
                              double *tmpVal, double *blockMin,
                              double *blockMax) {
  extern __shared__ double sdata[];
  double *smin = sdata;
  double *smax = sdata + blockDim.x;

  int tid = threadIdx.x;
  int i = (int)(blockIdx.x * blockDim.x + tid);
  int N = W * H;

  double v = 0.0;
  double mn = 1e300;
  double mx = -1e300;

  if (i < N && !mask[i]) {
    int x = i % W;
    int y = i / W;

    Prim p = cons_to_prim(load_cons(U, i));

    if (view_mode == 0) {
      v = log(p.rho);
    } else if (view_mode == 1) {
      v = log(p.p);
    } else if (view_mode == 2) {
      v = sqrt(p.u * p.u + p.v * p.v);
    } else if (view_mode == 3) {
      double rhoL = sample_prim_bc(U, mask, x, y, x - 1, y).rho;
      double rhoR = sample_prim_bc(U, mask, x, y, x + 1, y).rho;
      double rhoB = sample_prim_bc(U, mask, x, y, x, y - 1).rho;
      double rhoT = sample_prim_bc(U, mask, x, y, x, y + 1).rho;
      double gx = 0.5 * (rhoR - rhoL);
      double gy = 0.5 * (rhoT - rhoB);
      v = log(1e-12 + sqrt(gx * gx + gy * gy));
    } else if (view_mode == 4) {
      Prim pL = sample_prim_bc(U, mask, x, y, x - 1, y);
      Prim pR = sample_prim_bc(U, mask, x, y, x + 1, y);
      Prim pB = sample_prim_bc(U, mask, x, y, x, y - 1);
      Prim pT = sample_prim_bc(U, mask, x, y, x, y + 1);
      double dv_dx = 0.5 * (pR.v - pL.v);
      double du_dy = 0.5 * (pT.u - pB.u);
      double omega = dv_dx - du_dy;
      v = asinh(omega);
    } else if (view_mode == 5) {
      double a = sound_speed(p);
      double sp = sqrt(p.u * p.u + p.v * p.v);
      v = sp / d_fmax(a, 1e-30);
    } else {
      v = log(d_fmax(p.p / d_fmax(p.rho, EPS_RHO), 1e-30));
    }

    if (!isfinite(v))
      v = 0.0;

    tmpVal[i] = v;
    mn = v;
    mx = v;
  } else if (i < N) {
    tmpVal[i] = 0.0;
  }

  smin[tid] = mn;
  smax[tid] = mx;
  __syncthreads();

  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (tid < off) {
      double bmin = smin[tid + off];
      double bmax = smax[tid + off];
      if (bmin < smin[tid])
        smin[tid] = bmin;
      if (bmax > smax[tid])
        smax[tid] = bmax;
    }
    __syncthreads();
  }

  if (tid == 0) {
    blockMin[blockIdx.x] = smin[0];
    blockMax[blockIdx.x] = smax[0];
  }
}

__global__ void k_render_pixels(const uint8_t *mask, const double *tmpVal,
                                const double *minv, const double *invRange,
                                uchar4 *out) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = W * H;
  if (i >= N)
    return;

  if (mask[i]) {
    out[i] = pack_rgba(110, 110, 110);
    return;
  }

  double t = (tmpVal[i] - minv[0]) * invRange[0];
  uint8_t r, g, b;
  get_color(t, r, g, b);
  out[i] = pack_rgba(r, g, b);
}

__global__ void k_reduce_minmax(const double *inMin, const double *inMax,
                                double *outMin, double *outMax, int n) {
  extern __shared__ double sdata[];
  double *smin = sdata;
  double *smax = sdata + blockDim.x;

  int tid = threadIdx.x;
  int i0 = (int)(2 * blockIdx.x * blockDim.x + tid);

  double mn = 1e300;
  double mx = -1e300;

  if (i0 < n) {
    mn = inMin[i0];
    mx = inMax[i0];
  }

  int i1 = i0 + blockDim.x;
  if (i1 < n) {
    double mn1 = inMin[i1];
    double mx1 = inMax[i1];
    if (mn1 < mn)
      mn = mn1;
    if (mx1 > mx)
      mx = mx1;
  }

  smin[tid] = mn;
  smax[tid] = mx;
  __syncthreads();

  for (int off = blockDim.x / 2; off > 0; off >>= 1) {
    if (tid < off) {
      double bmin = smin[tid + off];
      double bmax = smax[tid + off];
      if (bmin < smin[tid])
        smin[tid] = bmin;
      if (bmax > smax[tid])
        smax[tid] = bmax;
    }
    __syncthreads();
  }

  if (tid == 0) {
    outMin[blockIdx.x] = smin[0];
    outMax[blockIdx.x] = smax[0];
  }
}

__global__ void k_compute_inv_range(const double *minv, const double *maxv,
                                    double *invRange) {
  double range = d_fmax(maxv[0] - minv[0], 1e-30);
  invRange[0] = 1.0 / range;
}

// general helpers
static void alloc_Us(Usoa *U, int N) {
  CK(cudaMalloc(&U->rho, N * sizeof(double)));
  CK(cudaMalloc(&U->mx, N * sizeof(double)));
  CK(cudaMalloc(&U->my, N * sizeof(double)));
  CK(cudaMalloc(&U->E, N * sizeof(double)));
}

static void free_Us(Usoa *U) {
  cudaFree(U->rho);
  cudaFree(U->mx);
  cudaFree(U->my);
  cudaFree(U->E);
  U->rho = U->mx = U->my = U->E = nullptr;
}

static void free_cuda_ptr(void **ptr) {
  if (!ptr || !*ptr)
    return;
  cudaError_t e = cudaFree(*ptr);
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA warning during cleanup: cudaFree failed: %s\n",
            cudaGetErrorString(e));
  }
  *ptr = nullptr;
}

static void alloc_Cs(Csoa *A, int N) {
  CK(cudaMalloc(&A->rho, N * sizeof(double)));
  CK(cudaMalloc(&A->mx, N * sizeof(double)));
  CK(cudaMalloc(&A->my, N * sizeof(double)));
  CK(cudaMalloc(&A->E, N * sizeof(double)));
}

static void free_Cs(Csoa *A) {
  cudaFree(A->rho);
  cudaFree(A->mx);
  cudaFree(A->my);
  cudaFree(A->E);
  A->rho = A->mx = A->my = A->E = nullptr;
}

static inline void swap_Us(Usoa *a, Usoa *b) {
  double *t;
  t = a->rho;
  a->rho = b->rho;
  b->rho = t;
  t = a->mx;
  a->mx = b->mx;
  b->mx = t;
  t = a->my;
  a->my = b->my;
  b->my = t;
  t = a->E;
  a->E = b->E;
  b->E = t;
}

static SimConfig default_config() {
  SimConfig cfg{};
  cfg.gamma = 1.1;
  cfg.cfl = 0.25;
  cfg.visc_nu = 5e-2;
  cfg.visc_rho = 5e-2;
  cfg.visc_e = 2e-2;
  cfg.inflow_mach = 25.0;
  cfg.geom_x0 = (double)W / 12.0;
  cfg.geom_cy = (double)H / 2.0;
  cfg.geom_Rb = (double)H / 12.0;
  cfg.geom_Rn = (double)H / 24.0;
  cfg.geom_theta = kPi / 4.0;
  cfg.steps_per_frame = 2;
  return cfg;
}

static void print_usage(const char *argv0) {
  fprintf(stderr,
          "Usage: %s [--mach M] [--gamma G] [--cfl C] [--visc-nu NU]\n"
          "          [--visc-rho MU] [--visc-e K] [--steps-per-frame N]\n"
          "          [--geom-x0 X0] [--geom-cy CY] [--geom-rb RB]\n"
          "          [--geom-rn RN] [--geom-theta THETA]\n",
          argv0);
}

static bool parse_double_flag(const char *name, const char *value, double *out) {
  char *end = nullptr;
  double v = strtod(value, &end);
  if (!end || *end != '\0' || !isfinite(v)) {
    fprintf(stderr, "Invalid value for %s: %s\n", name, value);
    return false;
  }
  *out = v;
  return true;
}

static bool parse_int_flag(const char *name, const char *value, int *out) {
  char *end = nullptr;
  errno = 0;
  long v = strtol(value, &end, 10);
  if (!end || *end != '\0' || errno == ERANGE || v < INT_MIN || v > INT_MAX) {
    fprintf(stderr, "Invalid value for %s: %s\n", name, value);
    return false;
  }
  *out = (int)v;
  return true;
}

static bool parse_args(int argc, char **argv, SimConfig *cfg) {
  const int max_steps_per_frame = 1024;

  for (int i = 1; i < argc; i++) {
    const char *arg = argv[i];
    if (strcmp(arg, "--mach") == 0 && i + 1 < argc) {
      if (!parse_double_flag(arg, argv[++i], &cfg->inflow_mach))
        return false;
    } else if (strcmp(arg, "--gamma") == 0 && i + 1 < argc) {
      if (!parse_double_flag(arg, argv[++i], &cfg->gamma))
        return false;
    } else if (strcmp(arg, "--cfl") == 0 && i + 1 < argc) {
      if (!parse_double_flag(arg, argv[++i], &cfg->cfl))
        return false;
    } else if (strcmp(arg, "--visc-nu") == 0 && i + 1 < argc) {
      if (!parse_double_flag(arg, argv[++i], &cfg->visc_nu))
        return false;
    } else if (strcmp(arg, "--visc-rho") == 0 && i + 1 < argc) {
      if (!parse_double_flag(arg, argv[++i], &cfg->visc_rho))
        return false;
    } else if (strcmp(arg, "--visc-e") == 0 && i + 1 < argc) {
      if (!parse_double_flag(arg, argv[++i], &cfg->visc_e))
        return false;
    } else if (strcmp(arg, "--steps-per-frame") == 0 && i + 1 < argc) {
      if (!parse_int_flag(arg, argv[++i], &cfg->steps_per_frame))
        return false;
    } else if (strcmp(arg, "--geom-x0") == 0 && i + 1 < argc) {
      if (!parse_double_flag(arg, argv[++i], &cfg->geom_x0))
        return false;
    } else if (strcmp(arg, "--geom-cy") == 0 && i + 1 < argc) {
      if (!parse_double_flag(arg, argv[++i], &cfg->geom_cy))
        return false;
    } else if (strcmp(arg, "--geom-rb") == 0 && i + 1 < argc) {
      if (!parse_double_flag(arg, argv[++i], &cfg->geom_Rb))
        return false;
    } else if (strcmp(arg, "--geom-rn") == 0 && i + 1 < argc) {
      if (!parse_double_flag(arg, argv[++i], &cfg->geom_Rn))
        return false;
    } else if (strcmp(arg, "--geom-theta") == 0 && i + 1 < argc) {
      if (!parse_double_flag(arg, argv[++i], &cfg->geom_theta))
        return false;
    } else {
      fprintf(stderr, "Unknown or incomplete argument: %s\n", arg);
      return false;
    }
  }

  if (cfg->gamma <= 1.0 || cfg->cfl <= 0.0 || cfg->visc_nu < 0.0 ||
      cfg->visc_rho < 0.0 || cfg->visc_e < 0.0 || cfg->inflow_mach <= 0.0 ||
      cfg->steps_per_frame <= 0 || cfg->steps_per_frame > max_steps_per_frame ||
      cfg->geom_Rb <= 0.0 || cfg->geom_Rn <= 0.0 ||
      cfg->geom_theta <= 0.0 || cfg->geom_theta >= 0.5 * kPi) {
    fprintf(stderr, "Invalid physical/geometry config values.\n");
    return false;
  }

  return true;
}

static void print_config(const SimConfig &cfg) {
  printf("SimConfig:\n");
  printf("  gamma=%.8g\n", cfg.gamma);
  printf("  cfl=%.8g\n", cfg.cfl);
  printf("  visc_nu=%.8g\n", cfg.visc_nu);
  printf("  visc_rho=%.8g\n", cfg.visc_rho);
  printf("  visc_e=%.8g\n", cfg.visc_e);
  printf("  inflow_mach=%.8g\n", cfg.inflow_mach);
  printf("  steps_per_frame=%d\n", cfg.steps_per_frame);
  printf("  geom_x0=%.8g geom_cy=%.8g geom_Rb=%.8g geom_Rn=%.8g geom_theta=%.8g\n",
         cfg.geom_x0, cfg.geom_cy, cfg.geom_Rb, cfg.geom_Rn, cfg.geom_theta);
}

#ifndef TAU_HYPERSONIC_CUDA_NO_MAIN
int main(int argc, char **argv) {
  SimConfig h_cfg = default_config();
  if (!parse_args(argc, argv, &h_cfg)) {
    print_usage(argv[0]);
    return 1;
  }
  print_config(h_cfg);
  CK(cudaMemcpyToSymbol(d_cfg, &h_cfg, sizeof(SimConfig)));

  InitWindow(W * SCALE, H * SCALE, "Hypersonic 2D Flow");
  SetTargetFPS(999);

  const int N = W * H;
  unsigned char *pixels = (unsigned char *)malloc((size_t)N * 4);

  Image img = {0};
  img.data = pixels;
  img.width = W;
  img.height = H;
  img.mipmaps = 1;
  img.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
  Texture2D tex = LoadTextureFromImage(img);

  Usoa dU{}, dUtmp{};
  alloc_Us(&dU, N);
  alloc_Us(&dUtmp, N);

  Csoa dXStateL{}, dXStateR{}, dYStateL{}, dYStateR{};
  Csoa dXFlux{}, dYFlux{};
  alloc_Cs(&dXStateL, N);
  alloc_Cs(&dXStateR, N);
  alloc_Cs(&dYStateL, N);
  alloc_Cs(&dYStateR, N);
  alloc_Cs(&dXFlux, (W + 1) * H);
  alloc_Cs(&dYFlux, W * (H + 1));

  uint8_t *dMask = nullptr;
  CK(cudaMalloc(&dMask, (size_t)N * sizeof(uint8_t)));

  double *dTmpVal = nullptr;
  CK(cudaMalloc(&dTmpVal, (size_t)N * sizeof(double)));

  uchar4 *dPixels = nullptr;
  CK(cudaMalloc(&dPixels, (size_t)N * sizeof(uchar4)));

  const int threads = 256;
  const size_t reduceSharedBytes = (size_t)threads * sizeof(double);
  const size_t reduceMinMaxSharedBytes = (size_t)threads * 2 * sizeof(double);
  validate_reduction_launch_config(threads, 2 * sizeof(double));

  const int blocksN = (N + threads - 1) / threads;
  const int xFaceCount = (W + 1) * H;
  const int yFaceCount = W * (H + 1);
  const int blocksXFaces = (xFaceCount + threads - 1) / threads;
  const int blocksYFaces = (yFaceCount + threads - 1) / threads;

  const dim3 tileBlock(32, 4);
  const dim3 blocksNTiled((W + tileBlock.x - 1) / tileBlock.x,
                          (H + tileBlock.y - 1) / tileBlock.y);
  const size_t tileCellsPredict =
      (size_t)(tileBlock.x + 2) * (size_t)(tileBlock.y + 2);
  const size_t tileCellsStep =
      (size_t)(tileBlock.x + 4) * (size_t)(tileBlock.y + 4);
  const size_t shmPredict = 4 * tileCellsPredict * sizeof(double) +
                            tileCellsPredict * sizeof(uint8_t);
  const size_t shmStep = 4 * tileCellsStep * sizeof(double) +
                         tileCellsStep * sizeof(uint8_t);

  double *dBlockMin = nullptr;
  double *dBlockMax = nullptr;
  CK(cudaMalloc(&dBlockMin, (size_t)blocksN * sizeof(double)));
  CK(cudaMalloc(&dBlockMax, (size_t)blocksN * sizeof(double)));

  double *dReduceMin = nullptr;
  double *dReduceMax = nullptr;
  CK(cudaMalloc(&dReduceMin, (size_t)blocksN * sizeof(double)));
  CK(cudaMalloc(&dReduceMax, (size_t)blocksN * sizeof(double)));

  double *dInvRange = nullptr;
  CK(cudaMalloc(&dInvRange, sizeof(double)));

  double *dMaxSpeed = nullptr;
  CK(cudaMalloc(&dMaxSpeed, sizeof(double)));

  double *dBlockSpeedMax = nullptr;
  CK(cudaMalloc(&dBlockSpeedMax, (size_t)blocksN * sizeof(double)));

  auto gpu_init = [&]() {
    k_init<<<blocksN, threads>>>(dU, dMask);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
  };

  gpu_init();

  double sim_t = 0.0;
  int view_mode = 3;

  while (!WindowShouldClose()) {
    if (IsKeyPressed(KEY_R)) {
      sim_t = 0.0;
      gpu_init();
    }
    if (IsKeyPressed(KEY_M))
      view_mode = (view_mode + 1) % 7;

    if (!IsKeyDown(KEY_SPACE)) {
      for (int k = 0; k < h_cfg.steps_per_frame; k++) {
        k_apply_inflow_left<<<(H + threads - 1) / threads, threads>>>(dU,
                                                                      dMask);
        CK(cudaGetLastError());

        // Two-stage GPU reduction to a single max wavespeed.
        k_max_wavespeed_blocks<<<blocksN, threads, reduceSharedBytes>>>(
            dU, dMask, dBlockSpeedMax);
        CK(cudaGetLastError());
        k_reduce_block_max<<<1, threads, reduceSharedBytes>>>(dBlockSpeedMax,
                                                        blocksN, dMaxSpeed);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());

        double maxs = 1e-12;
        CK(cudaMemcpy(&maxs, dMaxSpeed, sizeof(double),
                      cudaMemcpyDeviceToHost));
        if (!isfinite(maxs) || maxs < 1e-12)
          maxs = 1e-12;

        double dt_convective = h_cfg.cfl * 1.0 / maxs; // dx=dy=1

        double nu_max = fmax(h_cfg.visc_nu, fmax(h_cfg.visc_rho, h_cfg.visc_e));
        double dt_diff = dt_convective;
        if (isfinite(nu_max) && nu_max > 1e-12) {
          // Explicit 2D diffusion stability limit for dx=dy=1.
          dt_diff = 0.25 / nu_max;
        }

        double dt = fmin(dt_convective, dt_diff);
        double dt_dx = dt;
        double dt_dy = dt;
        double half_dt_dx = 0.5 * dt_dx;
        double half_dt_dy = 0.5 * dt_dy;

        k_predict_face_states<<<blocksNTiled, tileBlock, shmPredict>>>(
            dU, dMask, dXStateL, dXStateR, dYStateL, dYStateR, half_dt_dx,
            half_dt_dy);
        CK(cudaGetLastError());
        k_compute_xface_flux<<<blocksXFaces, threads>>>(dU, dMask, dXStateL,
                                                        dXStateR, dXFlux);
        CK(cudaGetLastError());
        k_compute_yface_flux<<<blocksYFaces, threads>>>(dU, dMask, dYStateL,
                                                        dYStateR, dYFlux);
        CK(cudaGetLastError());
        k_step<<<blocksNTiled, tileBlock, shmStep>>>(dU, dUtmp, dMask,
                                                dXFlux, dYFlux, dt, dt_dx,
                                                dt_dy);
        CK(cudaGetLastError());

        // Ping-pong swap (removes k_copy full-grid copy)
        swap_Us(&dU, &dUtmp);

        sim_t += dt;
      }
    }

    // render pass A: vals + per-block min/max
    k_render_vals<<<blocksN, threads, reduceMinMaxSharedBytes>>>(
        dU, dMask, view_mode, dTmpVal, dBlockMin, dBlockMax);
    CK(cudaGetLastError());

    const double *curMin = dBlockMin;
    const double *curMax = dBlockMax;
    double *outMin = dReduceMin;
    double *outMax = dReduceMax;
    int curN = blocksN;
    while (curN > 1) {
      int outN = (curN + (2 * threads - 1)) / (2 * threads);
      k_reduce_minmax<<<outN, threads, reduceMinMaxSharedBytes>>>(
          curMin, curMax, outMin, outMax, curN);
      CK(cudaGetLastError());
      curN = outN;
      const double *nextMin = outMin;
      const double *nextMax = outMax;
      outMin = (nextMin == dBlockMin) ? dReduceMin : dBlockMin;
      outMax = (nextMax == dBlockMax) ? dReduceMax : dBlockMax;
      curMin = nextMin;
      curMax = nextMax;
    }
    k_compute_inv_range<<<1, 1>>>(curMin, curMax, dInvRange);
    CK(cudaGetLastError());

    // render pass B: pixels
    k_render_pixels<<<blocksN, threads>>>(dMask, dTmpVal, curMin, dInvRange,
                                          dPixels);
    CK(cudaGetLastError());

    CK(cudaDeviceSynchronize());

    CK(cudaMemcpy(pixels, dPixels, (size_t)N * sizeof(uchar4),
                  cudaMemcpyDeviceToHost));
    UpdateTexture(tex, pixels);

    BeginDrawing();
    ClearBackground(BLACK);
    DrawTexturePro(tex, (Rectangle){0, 0, (float)W, (float)H},
                   (Rectangle){0, 0, (float)(W * SCALE), (float)(H * SCALE)},
                   (Vector2){0, 0}, 0.0f, WHITE);

    DrawText(TextFormat("%.6f", sim_t), 10, 10, 20, WHITE);

    const char *modestr = (view_mode == 0)   ? "log(rho)"
                          : (view_mode == 1) ? "log(p)"
                          : (view_mode == 2) ? "speed"
                          : (view_mode == 3) ? "schlieren"
                          : (view_mode == 4) ? "vorticity (asinh)"
                          : (view_mode == 5) ? "Mach"
                                             : "log(p/rho)";
    DrawText(modestr, 10, 34, 20, GREEN);

    EndDrawing();
  }

  // Deterministic cleanup in reverse allocation order.
  free_cuda_ptr((void **)&dBlockSpeedMax);
  free_cuda_ptr((void **)&dMaxSpeed);
  free_cuda_ptr((void **)&dInvRange);
  free_cuda_ptr((void **)&dReduceMax);
  free_cuda_ptr((void **)&dReduceMin);
  free_cuda_ptr((void **)&dBlockMax);
  free_cuda_ptr((void **)&dBlockMin);
  free_cuda_ptr((void **)&dPixels);
  free_cuda_ptr((void **)&dTmpVal);
  free_cuda_ptr((void **)&dMask);

  free_Cs(&dYFlux);
  free_Cs(&dXFlux);
  free_Cs(&dYStateR);
  free_Cs(&dYStateL);
  free_Cs(&dXStateR);
  free_Cs(&dXStateL);

  free_Us(&dUtmp);
  free_Us(&dU);

  UnloadTexture(tex);
  CloseWindow();
  free(pixels);

  cudaError_t reset_err = cudaDeviceReset();
  if (reset_err != cudaSuccess) {
    fprintf(stderr, "CUDA warning during cleanup: cudaDeviceReset failed: %s\n",
            cudaGetErrorString(reset_err));
  }

  return 0;
}
#endif
