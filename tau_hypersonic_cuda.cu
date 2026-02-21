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

static inline bool ck_host(cudaError_t e, const char *msg) {
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(e));
    return false;
  }
  return true;
}

typedef struct {
  double *rho, *mx, *my, *E;
} Usoa;

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

__device__ __forceinline__ Cons flux_x(Cons c) {
  Prim p = cons_to_prim(c);
  Cons f;
  f.rho = c.mx;
  f.mx = c.mx * p.u + p.p;
  f.my = c.my * p.u;
  f.E = (c.E + p.p) * p.u;
  return f;
}

__device__ __forceinline__ Cons flux_x(Prim p) {
  Cons c = prim_to_cons(p);
  Cons f;
  f.rho = c.mx;
  f.mx = c.mx * p.u + p.p;
  f.my = c.my * p.u;
  f.E = (c.E + p.p) * p.u;
  return f;
}

__device__ __forceinline__ Cons flux_y(Cons c) {
  Prim p = cons_to_prim(c);
  Cons f;
  f.rho = c.my;
  f.mx = c.mx * p.v;
  f.my = c.my * p.v + p.p;
  f.E = (c.E + p.p) * p.v;
  return f;
}

__device__ __forceinline__ Cons flux_y(Prim p) {
  Cons c = prim_to_cons(p);
  Cons f;
  f.rho = c.my;
  f.mx = c.mx * p.v;
  f.my = c.my * p.v + p.p;
  f.E = (c.E + p.p) * p.v;
  return f;
}

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

__device__ __forceinline__ FacePrim reconstruct_x(const Usoa U,
                                                  const uint8_t *mask, int x,
                                                  int y) {
  Cons Uc = load_cons(U, d_idx(x, y));
  Cons Um = neighbor_or_wall(U, mask, x, y, -1, 0);
  Cons Up = neighbor_or_wall(U, mask, x, y, +1, 0);

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

__device__ __forceinline__ FacePrim reconstruct_y(const Usoa U,
                                                  const uint8_t *mask, int x,
                                                  int y) {
  Cons Uc = load_cons(U, d_idx(x, y));
  Cons Um = neighbor_or_wall(U, mask, x, y, 0, -1);
  Cons Up = neighbor_or_wall(U, mask, x, y, 0, +1);

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

__device__ __forceinline__ Prim half_step_predict_x(Prim q, double dF_rho,
                                                    double dF_mx, double dF_my,
                                                    double dF_E,
                                                    double half_dt_dx) {
  Cons c = prim_to_cons(q);
  c.rho -= half_dt_dx * dF_rho;
  c.mx -= half_dt_dx * dF_mx;
  c.my -= half_dt_dx * dF_my;
  c.E -= half_dt_dx * dF_E;
  Prim out = cons_to_prim(c);
  out.rho = d_fmax(out.rho, EPS_RHO);
  out.p = d_fmax(out.p, EPS_P);
  return out;
}

__device__ __forceinline__ Prim half_step_predict_y(Prim q, double dG_rho,
                                                    double dG_mx, double dG_my,
                                                    double dG_E,
                                                    double half_dt_dy) {
  Cons c = prim_to_cons(q);
  c.rho -= half_dt_dy * dG_rho;
  c.mx -= half_dt_dy * dG_mx;
  c.my -= half_dt_dy * dG_my;
  c.E -= half_dt_dy * dG_E;
  Prim out = cons_to_prim(c);
  out.rho = d_fmax(out.rho, EPS_RHO);
  out.p = d_fmax(out.p, EPS_P);
  return out;
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

__device__ __forceinline__ Cons hlle_x(Cons UL, Cons UR) {
  Prim L = cons_to_prim(UL);
  Prim R = cons_to_prim(UR);
  double aL = sound_speed(L);
  double aR = sound_speed(R);
  double SL = d_fmin(L.u - aL, R.u - aR);
  double SR = d_fmax(L.u + aL, R.u + aR);

  Cons FL = flux_x(UL);
  Cons FR = flux_x(UR);

  if (SL >= 0.0)
    return FL;
  if (SR <= 0.0)
    return FR;

  double denom = SR - SL;
  if (d_fabs(denom) < 1e-14)
    return cons_mul(0.5, cons_add(FL, FR));

  // F = (SR*FL - SL*FR + SL*SR*(UR-UL)) / (SR-SL)
  Cons term1 = cons_mul(SR, FL);
  Cons term2 = cons_mul(-SL, FR);
  Cons term3 = cons_mul(SL * SR, cons_sub(UR, UL));
  return cons_mul(1.0 / denom, cons_add(cons_add(term1, term2), term3));
}

__device__ __forceinline__ Cons hlle_y(Cons UL, Cons UR) {
  Prim L = cons_to_prim(UL);
  Prim R = cons_to_prim(UR);
  double aL = sound_speed(L);
  double aR = sound_speed(R);
  double SL = d_fmin(L.v - aL, R.v - aR);
  double SR = d_fmax(L.v + aL, R.v + aR);

  Cons FL = flux_y(UL);
  Cons FR = flux_y(UR);

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

__device__ __forceinline__ Cons hllc_x(Cons UL, Cons UR) {
  Prim L = cons_to_prim(UL);
  Prim R = cons_to_prim(UR);

  double aL = sound_speed(L);
  double aR = sound_speed(R);

  double SL = d_fmin(L.u - aL, R.u - aR);
  double SR = d_fmax(L.u + aL, R.u + aR);

  Cons FL = flux_x(UL);
  Cons FR = flux_x(UR);

  if (SL >= 0.0)
    return FL;
  if (SR <= 0.0)
    return FR;

  double rhoL = L.rho, rhoR = R.rho;
  double uL = L.u, uR = R.u;
  double pL = L.p, pR = R.p;

  double num = pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR);
  double den = rhoL * (SL - uL) - rhoR * (SR - uR);

  if (d_fabs(den) < 1e-14 || !isfinite(num) || !isfinite(den)) {
    return hlle_x(UL, UR);
  }

  double SM = num / den;
  if (!isfinite(SM))
    return hlle_x(UL, UR);

  double pStar = pL + rhoL * (SL - uL) * (SM - uL);
  pStar = d_fmax(pStar, EPS_P);

  double dLS = (SL - SM);
  double dRS = (SR - SM);
  if (d_fabs(dLS) < 1e-14 || d_fabs(dRS) < 1e-14) {
    return hlle_x(UL, UR);
  }

  double rhoStarL = rhoL * (SL - uL) / dLS;
  double rhoStarR = rhoR * (SR - uR) / dRS;

  if (!(rhoStarL > 0.0) || !(rhoStarR > 0.0) || !isfinite(rhoStarL) ||
      !isfinite(rhoStarR)) {
    return hlle_x(UL, UR);
  }

  double mxStarL = rhoStarL * SM;
  double myStarL = rhoStarL * L.v;
  double EL = UL.E;
  double EStarL = ((SL - uL) * EL - pL * uL + pStar * SM) / dLS;
  if (!isfinite(EStarL))
    return hlle_x(UL, UR);
  Cons UStarL{rhoStarL, mxStarL, myStarL, EStarL};

  double mxStarR = rhoStarR * SM;
  double myStarR = rhoStarR * R.v;
  double ER = UR.E;
  double EStarR = ((SR - uR) * ER - pR * uR + pStar * SM) / dRS;
  if (!isfinite(EStarR))
    return hlle_x(UL, UR);
  Cons UStarR{rhoStarR, mxStarR, myStarR, EStarR};

  if (SM >= 0.0) {
    Cons F;
    F.rho = FL.rho + SL * (UStarL.rho - UL.rho);
    F.mx = FL.mx + SL * (UStarL.mx - UL.mx);
    F.my = FL.my + SL * (UStarL.my - UL.my);
    F.E = FL.E + SL * (UStarL.E - UL.E);
    return F;
  } else {
    Cons F;
    F.rho = FR.rho + SR * (UStarR.rho - UR.rho);
    F.mx = FR.mx + SR * (UStarR.mx - UR.mx);
    F.my = FR.my + SR * (UStarR.my - UR.my);
    F.E = FR.E + SR * (UStarR.E - UR.E);
    return F;
  }
}

__device__ __forceinline__ Cons hllc_x(Prim qL, Prim qR) {
  Cons UL = prim_to_cons(qL);
  Cons UR = prim_to_cons(qR);
  return hllc_x(UL, UR);
}

__device__ __forceinline__ Cons hllc_y(Cons UL, Cons UR) {
  Prim L = cons_to_prim(UL);
  Prim R = cons_to_prim(UR);

  double aL = sound_speed(L);
  double aR = sound_speed(R);

  double SL = d_fmin(L.v - aL, R.v - aR);
  double SR = d_fmax(L.v + aL, R.v + aR);

  Cons FL = flux_y(UL);
  Cons FR = flux_y(UR);

  if (SL >= 0.0)
    return FL;
  if (SR <= 0.0)
    return FR;

  double rhoL = L.rho, rhoR = R.rho;
  double vL = L.v, vR = R.v;
  double pL = L.p, pR = R.p;

  double num = pR - pL + rhoL * vL * (SL - vL) - rhoR * vR * (SR - vR);
  double den = rhoL * (SL - vL) - rhoR * (SR - vR);

  if (d_fabs(den) < 1e-14 || !isfinite(num) || !isfinite(den)) {
    return hlle_y(UL, UR);
  }

  double SM = num / den;
  if (!isfinite(SM))
    return hlle_y(UL, UR);

  double pStar = pL + rhoL * (SL - vL) * (SM - vL);
  pStar = d_fmax(pStar, EPS_P);

  double dLS = (SL - SM);
  double dRS = (SR - SM);
  if (d_fabs(dLS) < 1e-14 || d_fabs(dRS) < 1e-14) {
    return hlle_y(UL, UR);
  }

  double rhoStarL = rhoL * (SL - vL) / dLS;
  double rhoStarR = rhoR * (SR - vR) / dRS;

  if (!(rhoStarL > 0.0) || !(rhoStarR > 0.0) || !isfinite(rhoStarL) ||
      !isfinite(rhoStarR)) {
    return hlle_y(UL, UR);
  }

  double mxStarL = rhoStarL * L.u;
  double myStarL = rhoStarL * SM;
  double EL = UL.E;
  double EStarL = ((SL - vL) * EL - pL * vL + pStar * SM) / dLS;
  if (!isfinite(EStarL))
    return hlle_y(UL, UR);
  Cons UStarL{rhoStarL, mxStarL, myStarL, EStarL};

  double mxStarR = rhoStarR * R.u;
  double myStarR = rhoStarR * SM;
  double ER = UR.E;
  double EStarR = ((SR - vR) * ER - pR * vR + pStar * SM) / dRS;
  if (!isfinite(EStarR))
    return hlle_y(UL, UR);
  Cons UStarR{rhoStarR, mxStarR, myStarR, EStarR};

  if (SM >= 0.0) {
    Cons F;
    F.rho = FL.rho + SL * (UStarL.rho - UL.rho);
    F.mx = FL.mx + SL * (UStarL.mx - UL.mx);
    F.my = FL.my + SL * (UStarL.my - UL.my);
    F.E = FL.E + SL * (UStarL.E - UL.E);
    return F;
  } else {
    Cons F;
    F.rho = FR.rho + SR * (UStarR.rho - UR.rho);
    F.mx = FR.mx + SR * (UStarR.mx - UR.mx);
    F.my = FR.my + SR * (UStarR.my - UR.my);
    F.E = FR.E + SR * (UStarR.E - UR.E);
    return F;
  }
}

__device__ __forceinline__ Cons hllc_y(Prim qB, Prim qT) {
  Cons UL = prim_to_cons(qB);
  Cons UR = prim_to_cons(qT);
  return hllc_y(UL, UR);
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
  __shared__ double smax[256];
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
  __shared__ double smax[256];
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

__global__ void k_step(Usoa U, Usoa Uout, const uint8_t *mask, double dt,
                       double dt_dx, double dt_dy, double half_dt_dx,
                       double half_dt_dy) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = W * H;
  if (i >= N)
    return;

  if (mask[i]) {
    store_cons(Uout, i, load_cons(U, i));
    return;
  }

  int x = i % W;
  int y = i / W;

  // Reuse center reconstructions for both adjacent faces.
  FacePrim fpCx = reconstruct_x(U, mask, x, y);
  Cons fpCx_L = prim_to_cons(fpCx.L);
  Cons fpCx_R = prim_to_cons(fpCx.R);
  Cons fpCx_FL = flux_x(fpCx_L);
  Cons fpCx_FR = flux_x(fpCx_R);

  FacePrim fpCy = reconstruct_y(U, mask, x, y);
  Cons fpCy_L = prim_to_cons(fpCy.L);
  Cons fpCy_R = prim_to_cons(fpCy.R);
  Cons fpCy_GL = flux_y(fpCy_L);
  Cons fpCy_GR = flux_y(fpCy_R);

  // X faces
  Prim qL_left, qR_left;
  {
    if (x - 1 >= 0 && !mask[d_idx(x - 1, y)]) {
      FacePrim fp = reconstruct_x(U, mask, x - 1, y);
      Cons fpL = prim_to_cons(fp.L);
      Cons fpR = prim_to_cons(fp.R);
      Cons FLf = flux_x(fpR);
      Cons FLb = flux_x(fpL);
      qL_left =
          half_step_predict_x(fp.R, (FLf.rho - FLb.rho), (FLf.mx - FLb.mx),
                              (FLf.my - FLb.my), (FLf.E - FLb.E), half_dt_dx);
    } else {
      qL_left = cons_to_prim(neighbor_or_wall(U, mask, x, y, -1, 0));
    }

    qR_left = half_step_predict_x(
        fpCx.L, (fpCx_FR.rho - fpCx_FL.rho), (fpCx_FR.mx - fpCx_FL.mx),
        (fpCx_FR.my - fpCx_FL.my), (fpCx_FR.E - fpCx_FL.E), half_dt_dx);

    qL_left.rho = d_fmax(qL_left.rho, EPS_RHO);
    qL_left.p = d_fmax(qL_left.p, EPS_P);
    qR_left.rho = d_fmax(qR_left.rho, EPS_RHO);
    qR_left.p = d_fmax(qR_left.p, EPS_P);
  }

  Prim qL_right, qR_right;
  {
    qL_right = half_step_predict_x(
        fpCx.R, (fpCx_FR.rho - fpCx_FL.rho), (fpCx_FR.mx - fpCx_FL.mx),
        (fpCx_FR.my - fpCx_FL.my), (fpCx_FR.E - fpCx_FL.E), half_dt_dx);

    if (x + 1 < W && !mask[d_idx(x + 1, y)]) {
      FacePrim fpN = reconstruct_x(U, mask, x + 1, y);
      Cons fpNL = prim_to_cons(fpN.L);
      Cons fpNR = prim_to_cons(fpN.R);
      Cons NFf = flux_x(fpNR);
      Cons NFb = flux_x(fpNL);
      qR_right =
          half_step_predict_x(fpN.L, (NFf.rho - NFb.rho), (NFf.mx - NFb.mx),
                              (NFf.my - NFb.my), (NFf.E - NFb.E), half_dt_dx);
    } else {
      qR_right = cons_to_prim(neighbor_or_wall(U, mask, x, y, +1, 0));
    }

    qL_right.rho = d_fmax(qL_right.rho, EPS_RHO);
    qL_right.p = d_fmax(qL_right.p, EPS_P);
    qR_right.rho = d_fmax(qR_right.rho, EPS_RHO);
    qR_right.p = d_fmax(qR_right.p, EPS_P);
  }

  Cons qL_left_cons = prim_to_cons(qL_left);
  Cons qR_left_cons = prim_to_cons(qR_left);
  Cons qL_right_cons = prim_to_cons(qL_right);
  Cons qR_right_cons = prim_to_cons(qR_right);

  Cons FxL = hllc_x(qL_left_cons, qR_left_cons);
  Cons FxR = hllc_x(qL_right_cons, qR_right_cons);

  // Y faces
  Prim qB_bot, qT_bot;
  {
    if (y - 1 >= 0 && !mask[d_idx(x, y - 1)]) {
      FacePrim fp = reconstruct_y(U, mask, x, y - 1);
      Cons fpL = prim_to_cons(fp.L);
      Cons fpR = prim_to_cons(fp.R);
      Cons GBf = flux_y(fpR);
      Cons GBb = flux_y(fpL);
      qB_bot =
          half_step_predict_y(fp.R, (GBf.rho - GBb.rho), (GBf.mx - GBb.mx),
                              (GBf.my - GBb.my), (GBf.E - GBb.E), half_dt_dy);
    } else {
      qB_bot = cons_to_prim(neighbor_or_wall(U, mask, x, y, 0, -1));
    }

    qT_bot = half_step_predict_y(
        fpCy.L, (fpCy_GR.rho - fpCy_GL.rho), (fpCy_GR.mx - fpCy_GL.mx),
        (fpCy_GR.my - fpCy_GL.my), (fpCy_GR.E - fpCy_GL.E), half_dt_dy);

    qB_bot.rho = d_fmax(qB_bot.rho, EPS_RHO);
    qB_bot.p = d_fmax(qB_bot.p, EPS_P);
    qT_bot.rho = d_fmax(qT_bot.rho, EPS_RHO);
    qT_bot.p = d_fmax(qT_bot.p, EPS_P);
  }

  Prim qB_top, qT_top;
  {
    qB_top = half_step_predict_y(
        fpCy.R, (fpCy_GR.rho - fpCy_GL.rho), (fpCy_GR.mx - fpCy_GL.mx),
        (fpCy_GR.my - fpCy_GL.my), (fpCy_GR.E - fpCy_GL.E), half_dt_dy);

    if (y + 1 < H && !mask[d_idx(x, y + 1)]) {
      FacePrim fpN = reconstruct_y(U, mask, x, y + 1);
      Cons fpNL = prim_to_cons(fpN.L);
      Cons fpNR = prim_to_cons(fpN.R);
      Cons NFf = flux_y(fpNR);
      Cons NFb = flux_y(fpNL);
      qT_top =
          half_step_predict_y(fpN.L, (NFf.rho - NFb.rho), (NFf.mx - NFb.mx),
                              (NFf.my - NFb.my), (NFf.E - NFb.E), half_dt_dy);
    } else {
      qT_top = cons_to_prim(neighbor_or_wall(U, mask, x, y, 0, +1));
    }

    qB_top.rho = d_fmax(qB_top.rho, EPS_RHO);
    qB_top.p = d_fmax(qB_top.p, EPS_P);
    qT_top.rho = d_fmax(qT_top.rho, EPS_RHO);
    qT_top.p = d_fmax(qT_top.p, EPS_P);
  }

  Cons qB_bot_cons = prim_to_cons(qB_bot);
  Cons qT_bot_cons = prim_to_cons(qT_bot);
  Cons qB_top_cons = prim_to_cons(qB_top);
  Cons qT_top_cons = prim_to_cons(qT_top);

  Cons GyB = hllc_y(qB_bot_cons, qT_bot_cons);
  Cons GyT = hllc_y(qB_top_cons, qT_top_cons);

  // Hyperbolic update
  Cons Uc = load_cons(U, i);
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

    Cons xm2 = neighbor_for_diff(U, mask, x, y, x - 2, y);
    Cons xm1 = neighbor_for_diff(U, mask, x, y, x - 1, y);
    Cons xp1 = neighbor_for_diff(U, mask, x, y, x + 1, y);
    Cons xp2 = neighbor_for_diff(U, mask, x, y, x + 2, y);

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

    Cons ym2 = neighbor_for_diff(U, mask, x, y, x, y - 2);
    Cons ym1 = neighbor_for_diff(U, mask, x, y, x, y - 1);
    Cons yp1 = neighbor_for_diff(U, mask, x, y, x, y + 1);
    Cons yp2 = neighbor_for_diff(U, mask, x, y, x, y + 2);

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
  __shared__ double smin[256];
  __shared__ double smax[256];

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
  __shared__ double smin[256];
  __shared__ double smax[256];

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

static bool alloc_Us_host(Usoa *U, int N) {
  if (!ck_host(cudaMalloc(&U->rho, N * sizeof(double)), "cudaMalloc(&U->rho)"))
    return false;
  if (!ck_host(cudaMalloc(&U->mx, N * sizeof(double)), "cudaMalloc(&U->mx)")) {
    cudaFree(U->rho);
    U->rho = nullptr;
    return false;
  }
  if (!ck_host(cudaMalloc(&U->my, N * sizeof(double)), "cudaMalloc(&U->my)")) {
    cudaFree(U->rho);
    cudaFree(U->mx);
    U->rho = U->mx = nullptr;
    return false;
  }
  if (!ck_host(cudaMalloc(&U->E, N * sizeof(double)), "cudaMalloc(&U->E)")) {
    cudaFree(U->rho);
    cudaFree(U->mx);
    cudaFree(U->my);
    U->rho = U->mx = U->my = nullptr;
    return false;
  }
  return true;
}

static void free_Us(Usoa *U) {
  cudaFree(U->rho);
  cudaFree(U->mx);
  cudaFree(U->my);
  cudaFree(U->E);
  U->rho = U->mx = U->my = U->E = nullptr;
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
  cfg.geom_theta = PI / 4.0;
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
  long v = strtol(value, &end, 10);
  if (!end || *end != '\0') {
    fprintf(stderr, "Invalid value for %s: %s\n", name, value);
    return false;
  }
  *out = (int)v;
  return true;
}

static bool parse_args(int argc, char **argv, SimConfig *cfg) {
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
      cfg->steps_per_frame <= 0 || cfg->geom_Rb <= 0.0 || cfg->geom_Rn <= 0.0 ||
      cfg->geom_theta <= 0.0 || cfg->geom_theta >= 0.5 * PI) {
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
#define CK_MAIN(stmt)                                                            \
  do {                                                                           \
    if (!ck_host((stmt), #stmt)) {                                               \
      ret = 1;                                                                   \
      goto cleanup;                                                              \
    }                                                                            \
  } while (0)

  int ret = 0;
  SimConfig h_cfg = default_config();
  if (!parse_args(argc, argv, &h_cfg)) {
    print_usage(argv[0]);
    return 1;
  }
  print_config(h_cfg);
  CK_MAIN(cudaMemcpyToSymbol(d_cfg, &h_cfg, sizeof(SimConfig)));

  bool window_inited = false;
  bool texture_loaded = false;
  Texture2D tex = {0};
  unsigned char *pixels = nullptr;
  Usoa dU{}, dUtmp{};
  uint8_t *dMask = nullptr;
  double *dTmpVal = nullptr;
  uchar4 *dPixels = nullptr;
  double *dBlockMin = nullptr;
  double *dBlockMax = nullptr;
  double *dReduceMin = nullptr;
  double *dReduceMax = nullptr;
  double *dInvRange = nullptr;
  double *dMaxSpeed = nullptr;
  double *dBlockSpeedMax = nullptr;

  InitWindow(W * SCALE, H * SCALE, "Hypersonic 2D Flow");
  window_inited = true;
  SetTargetFPS(999);

  const int N = W * H;
  pixels = (unsigned char *)malloc((size_t)N * 4);
  if (!pixels) {
    fprintf(stderr, "malloc failed for pixels\n");
    ret = 1;
    goto cleanup;
  }

  Image img = {0};
  img.data = pixels;
  img.width = W;
  img.height = H;
  img.mipmaps = 1;
  img.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
  tex = LoadTextureFromImage(img);
  texture_loaded = tex.id != 0;

  if (!alloc_Us_host(&dU, N)) {
    ret = 1;
    goto cleanup;
  }
  if (!alloc_Us_host(&dUtmp, N)) {
    ret = 1;
    goto cleanup;
  }

  CK_MAIN(cudaMalloc(&dMask, (size_t)N * sizeof(uint8_t)));

  CK_MAIN(cudaMalloc(&dTmpVal, (size_t)N * sizeof(double)));

  CK_MAIN(cudaMalloc(&dPixels, (size_t)N * sizeof(uchar4)));

  const int threads = 256;
  const int blocksN = (N + threads - 1) / threads;

  CK_MAIN(cudaMalloc(&dBlockMin, (size_t)blocksN * sizeof(double)));
  CK_MAIN(cudaMalloc(&dBlockMax, (size_t)blocksN * sizeof(double)));

  CK_MAIN(cudaMalloc(&dReduceMin, (size_t)blocksN * sizeof(double)));
  CK_MAIN(cudaMalloc(&dReduceMax, (size_t)blocksN * sizeof(double)));

  CK_MAIN(cudaMalloc(&dInvRange, sizeof(double)));

  CK_MAIN(cudaMalloc(&dMaxSpeed, sizeof(double)));

  CK_MAIN(cudaMalloc(&dBlockSpeedMax, (size_t)blocksN * sizeof(double)));

  auto gpu_init = [&]() -> bool {
    k_init<<<blocksN, threads>>>(dU, dMask);
    if (!ck_host(cudaGetLastError(), "cudaGetLastError()"))
      return false;
    if (!ck_host(cudaDeviceSynchronize(), "cudaDeviceSynchronize()"))
      return false;
    return true;
  };

  if (!gpu_init()) {
    ret = 1;
    goto cleanup;
  }

  double sim_t = 0.0;
  int view_mode = 3;

  while (!WindowShouldClose()) {
    if (IsKeyPressed(KEY_R)) {
      sim_t = 0.0;
      if (!gpu_init()) {
        ret = 1;
        goto cleanup;
      }
    }
    if (IsKeyPressed(KEY_M))
      view_mode = (view_mode + 1) % 7;

    if (!IsKeyDown(KEY_SPACE)) {
      for (int k = 0; k < h_cfg.steps_per_frame; k++) {
        k_apply_inflow_left<<<(H + threads - 1) / threads, threads>>>(dU,
                                                                      dMask);
        CK_MAIN(cudaGetLastError());

        // Two-stage GPU reduction to a single max wavespeed.
        k_max_wavespeed_blocks<<<blocksN, threads>>>(dU, dMask,
                                                     dBlockSpeedMax);
        CK_MAIN(cudaGetLastError());
        k_reduce_block_max<<<1, threads>>>(dBlockSpeedMax, blocksN, dMaxSpeed);
        CK_MAIN(cudaGetLastError());
        CK_MAIN(cudaDeviceSynchronize());

        double maxs = 1e-12;
        CK_MAIN(cudaMemcpy(&maxs, dMaxSpeed, sizeof(double),
                           cudaMemcpyDeviceToHost));
        if (!isfinite(maxs) || maxs < 1e-12)
          maxs = 1e-12;

        double dt = h_cfg.cfl * 1.0 / maxs; // dx=dy=1
        double dt_dx = dt;
        double dt_dy = dt;
        double half_dt_dx = 0.5 * dt_dx;
        double half_dt_dy = 0.5 * dt_dy;

        k_step<<<blocksN, threads>>>(dU, dUtmp, dMask, dt, dt_dx, dt_dy,
                                     half_dt_dx, half_dt_dy);
        CK_MAIN(cudaGetLastError());
        CK_MAIN(cudaDeviceSynchronize());

        // Ping-pong swap (removes k_copy full-grid copy)
        swap_Us(&dU, &dUtmp);

        sim_t += dt;
      }
    }

    // render pass A: vals + per-block min/max
    k_render_vals<<<blocksN, threads>>>(dU, dMask, view_mode, dTmpVal,
                                        dBlockMin, dBlockMax);
    CK_MAIN(cudaGetLastError());
    CK_MAIN(cudaDeviceSynchronize());

    const double *curMin = dBlockMin;
    const double *curMax = dBlockMax;
    double *outMin = dReduceMin;
    double *outMax = dReduceMax;
    int curN = blocksN;
    while (curN > 1) {
      int outN = (curN + (2 * threads - 1)) / (2 * threads);
      k_reduce_minmax<<<outN, threads>>>(curMin, curMax, outMin, outMax, curN);
      CK_MAIN(cudaGetLastError());
      curN = outN;
      const double *nextMin = outMin;
      const double *nextMax = outMax;
      outMin = (nextMin == dBlockMin) ? dReduceMin : dBlockMin;
      outMax = (nextMax == dBlockMax) ? dReduceMax : dBlockMax;
      curMin = nextMin;
      curMax = nextMax;
    }
    k_compute_inv_range<<<1, 1>>>(curMin, curMax, dInvRange);
    CK_MAIN(cudaGetLastError());

    // render pass B: pixels
    k_render_pixels<<<blocksN, threads>>>(dMask, dTmpVal, curMin, dInvRange,
                                          dPixels);
    CK_MAIN(cudaGetLastError());
    CK_MAIN(cudaDeviceSynchronize());

    CK_MAIN(cudaMemcpy(pixels, dPixels, (size_t)N * sizeof(uchar4),
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

cleanup:
  if (texture_loaded)
    UnloadTexture(tex);
  if (window_inited)
    CloseWindow();

  free(pixels);

  cudaFree(dMask);
  cudaFree(dTmpVal);
  cudaFree(dPixels);
  cudaFree(dBlockMin);
  cudaFree(dBlockMax);
  cudaFree(dReduceMin);
  cudaFree(dReduceMax);
  cudaFree(dInvRange);
  cudaFree(dMaxSpeed);
  cudaFree(dBlockSpeedMax);

  free_Us(&dU);
  free_Us(&dUtmp);

  return ret;

#undef CK_MAIN
}
#endif
