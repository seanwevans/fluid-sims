// tau_hypersonic_cuda.cu
// Linux build (raylib installed system-wide):
//   nvcc -O3 -std=c++17 -Xcompiler "-O3" tau_hypersonic_cuda.cu \
//        -lraylib -lm -lX11 -o tau_2d_cuda
//
// SPACE pause, R reset, M mode

#include "raylib.h"
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define W 300
#define H 300
#define SCALE 2

#define GAMMA 1.4
#define CFL 0.3

#define STEPS_PER_FRAME 2

#define EPS_RHO 1e-10
#define EPS_P 1e-10

static inline void ck(cudaError_t e, const char *msg) {
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(e));
    exit(1);
  }
}
#define CK(x) ck((x), #x)

typedef struct {
  double *rho, *mx, *my, *E;
} Usoa;

static inline int h_idx(int x, int y) { return y * W + x; }

__device__ __forceinline__ int d_idx(int x, int y) { return y * W + x; }
__device__ __forceinline__ double d_fmax(double a, double b) {
  return a > b ? a : b;
}
__device__ __forceinline__ double d_fmin(double a, double b) {
  return a < b ? a : b;
}
__device__ __forceinline__ double d_fabs(double a) { return a < 0 ? -a : a; }

struct Cons {
  double rho, mx, my, E;
};
struct Prim {
  double rho, u, v, p;
};

__device__ __forceinline__ Prim cons_to_prim(Cons c) {
  Prim p;
  double rho = d_fmax(c.rho, EPS_RHO);
  double inv = 1.0 / rho;
  double u = c.mx * inv;
  double v = c.my * inv;
  double kin = 0.5 * rho * (u * u + v * v);
  double eint = c.E - kin;
  double pr = (GAMMA - 1.0) * d_fmax(eint, EPS_P);
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
  c.E = pr / (GAMMA - 1.0) + 0.5 * rho * (p.u * p.u + p.v * p.v);
  return c;
}

__device__ __forceinline__ double sound_speed(Prim p) {
  return sqrt(GAMMA * d_fmax(p.p, EPS_P) / d_fmax(p.rho, EPS_RHO));
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

__device__ __forceinline__ Cons flux_y(Cons c) {
  Prim p = cons_to_prim(c);
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
  const double mach = 15.0;
  const double rho = 1.0;
  const double p = 1.0;
  double a = sqrt(GAMMA * p / rho);
  double u = mach * a;
  Prim s{rho, u, 0.0, p};
  return s;
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

__device__ __forceinline__ Cons reflect_slip(Cons inside, double nx,
                                             double ny) {
  Prim p = cons_to_prim(inside);
  double vn = p.u * nx + p.v * ny;
  double ut = -p.u * ny + p.v * nx;
  vn = -vn;
  double u = vn * nx - ut * ny;
  double v = vn * ny + ut * nx;
  Prim g{p.rho, u, v, p.p};
  return prim_to_cons(g);
}

__device__ __forceinline__ Cons neighbor_or_wall(const Usoa U,
                                                 const uint8_t *mask, int x,
                                                 int y, int dx, int dy,
                                                 double nx, double ny) {
  int xn = x + dx;
  int yn = y + dy;

  if (yn < 0)
    yn = 0;
  if (yn >= H)
    yn = H - 1;

  // left inflow
  if (xn < 0) {
    return prim_to_cons(inflow_state());
  }
  // right outflow
  if (xn >= W) {
    int i = d_idx(W - 1, yn);
    return load_cons(U, i);
  }

  int j = d_idx(xn, yn);
  if (mask[j]) {
    int i = d_idx(x, y);
    return reflect_slip(load_cons(U, i), nx, ny);
  }
  return load_cons(U, j);
}

struct FacePrim {
  Prim L, R;
};

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
  int ic = d_idx(x, y);
  Cons Uc = load_cons(U, ic);
  Cons Um = neighbor_or_wall(U, mask, x, y, -1, 0, 1, 0);
  Cons Up = neighbor_or_wall(U, mask, x, y, +1, 0, 1, 0);

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
  int ic = d_idx(x, y);
  Cons Uc = load_cons(U, ic);
  Cons Um = neighbor_or_wall(U, mask, x, y, 0, -1, 0, 1);
  Cons Up = neighbor_or_wall(U, mask, x, y, 0, +1, 0, 1);

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
  double SM = num / den;

  double pStar = pL + rhoL * (SL - uL) * (SM - uL);
  pStar = d_fmax(pStar, EPS_P);

  double rhoStarL = rhoL * (SL - uL) / (SL - SM);
  double mxStarL = rhoStarL * SM;
  double myStarL = rhoStarL * L.v;
  double EL = UL.E;
  double EStarL = ((SL - uL) * EL - pL * uL + pStar * SM) / (SL - SM);
  Cons UStarL{rhoStarL, mxStarL, myStarL, EStarL};

  double rhoStarR = rhoR * (SR - uR) / (SR - SM);
  double mxStarR = rhoStarR * SM;
  double myStarR = rhoStarR * R.v;
  double ER = UR.E;
  double EStarR = ((SR - uR) * ER - pR * uR + pStar * SM) / (SR - SM);
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
  double SM = num / den;

  double pStar = pL + rhoL * (SL - vL) * (SM - vL);
  pStar = d_fmax(pStar, EPS_P);

  double rhoStarL = rhoL * (SL - vL) / (SL - SM);
  double mxStarL = rhoStarL * L.u;
  double myStarL = rhoStarL * SM;
  double EL = UL.E;
  double EStarL = ((SL - vL) * EL - pL * vL + pStar * SM) / (SL - SM);
  Cons UStarL{rhoStarL, mxStarL, myStarL, EStarL};

  double rhoStarR = rhoR * (SR - vR) / (SR - SM);
  double mxStarR = rhoStarR * R.u;
  double myStarR = rhoStarR * SM;
  double ER = UR.E;
  double EStarR = ((SR - vR) * ER - pR * vR + pStar * SM) / (SR - SM);
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

__global__ void k_init(Usoa U, uint8_t *mask) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = W * H;
  if (i >= N)
    return;

  int x = i % W;
  int y = i / W;

  int cx = W / 3;
  int cy = H / 2;
  int r = H / 6;

  int dx = x - cx;
  int dy = y - cy;
  uint8_t m = (dx * dx + dy * dy < r * r) ? 1 : 0;
  mask[i] = m;

  Prim inflow = inflow_state();
  Prim s = m ? Prim{inflow.rho, 0.0, 0.0, inflow.p} : inflow;
  Cons c = prim_to_cons(s);
  store_cons(U, i, c);
}

__global__ void k_apply_inflow_left(Usoa U, const uint8_t *mask) {
  int y = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (y >= H)
    return;
  int i0 = d_idx(0, y);
  if (!mask[i0]) {
    store_cons(U, i0, prim_to_cons(inflow_state()));
  }
}

__global__ void k_max_wavespeed(const Usoa U, const uint8_t *mask,
                                double *blockMax) {
  __shared__ double smax[256];
  int tid = threadIdx.x;
  int i = (int)(blockIdx.x * blockDim.x + tid);
  int N = W * H;

  double v = 1e-12;
  if (i < N && !mask[i]) {
    Cons c = load_cons(U, i);
    Prim p = cons_to_prim(c);
    double a = sound_speed(p);
    double sx = d_fabs(p.u) + a;
    double sy = d_fabs(p.v) + a;
    v = (sx > sy) ? sx : sy;
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

// Unew = U - dt/dx*(FxR-FxL) - dt/dy*(GyT-GyB)
__global__ void k_step(Usoa U, Usoa Unew, const uint8_t *mask, double dt_dx,
                       double dt_dy, double half_dt_dx, double half_dt_dy) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = W * H;
  if (i >= N)
    return;
  if (mask[i]) { // keep as-is (solid)
    store_cons(Unew, i, load_cons(U, i));
    return;
  }

  int x = i % W;
  int y = i / W;

  Prim qL_left, qR_left;
  {
    if (x - 1 >= 0 && !mask[d_idx(x - 1, y)]) {
      FacePrim fp = reconstruct_x(U, mask, x - 1, y);
      qL_left = fp.R;
      Cons FLf = flux_x(prim_to_cons(fp.R));
      Cons FLb = flux_x(prim_to_cons(fp.L));
      qL_left =
          half_step_predict_x(fp.R, (FLf.rho - FLb.rho), (FLf.mx - FLb.mx),
                              (FLf.my - FLb.my), (FLf.E - FLb.E), half_dt_dx);
    } else {
      Cons ULc = neighbor_or_wall(U, mask, x, y, -1, 0, 1, 0);
      qL_left = cons_to_prim(ULc);
    }

    if (!mask[d_idx(x, y)]) {
      FacePrim fp = reconstruct_x(U, mask, x, y);
      qR_left = fp.L;
      Cons FRf = flux_x(prim_to_cons(fp.R));
      Cons FRb = flux_x(prim_to_cons(fp.L));
      qR_left =
          half_step_predict_x(fp.L, (FRf.rho - FRb.rho), (FRf.mx - FRb.mx),
                              (FRf.my - FRb.my), (FRf.E - FRb.E), half_dt_dx);
    } else {
      qR_left = cons_to_prim(load_cons(U, i));
    }

    qL_left.rho = d_fmax(qL_left.rho, EPS_RHO);
    qL_left.p = d_fmax(qL_left.p, EPS_P);
    qR_left.rho = d_fmax(qR_left.rho, EPS_RHO);
    qR_left.p = d_fmax(qR_left.p, EPS_P);
  }

  Prim qL_right, qR_right;
  {
    // this cell right face
    FacePrim fpC = reconstruct_x(U, mask, x, y);
    qL_right = fpC.R;
    Cons CFf = flux_x(prim_to_cons(fpC.R));
    Cons CFb = flux_x(prim_to_cons(fpC.L));
    qL_right =
        half_step_predict_x(fpC.R, (CFf.rho - CFb.rho), (CFf.mx - CFb.mx),
                            (CFf.my - CFb.my), (CFf.E - CFb.E), half_dt_dx);

    // right neighbor left face
    if (x + 1 < W && !mask[d_idx(x + 1, y)]) {
      FacePrim fpN = reconstruct_x(U, mask, x + 1, y);
      qR_right = fpN.L;
      Cons NFf = flux_x(prim_to_cons(fpN.R));
      Cons NFb = flux_x(prim_to_cons(fpN.L));
      qR_right =
          half_step_predict_x(fpN.L, (NFf.rho - NFb.rho), (NFf.mx - NFb.mx),
                              (NFf.my - NFb.my), (NFf.E - NFb.E), half_dt_dx);
    } else {
      Cons URc = neighbor_or_wall(U, mask, x, y, +1, 0, 1, 0);
      qR_right = cons_to_prim(URc);
    }

    qL_right.rho = d_fmax(qL_right.rho, EPS_RHO);
    qL_right.p = d_fmax(qL_right.p, EPS_P);
    qR_right.rho = d_fmax(qR_right.rho, EPS_RHO);
    qR_right.p = d_fmax(qR_right.p, EPS_P);
  }

  Cons FxL = hllc_x(prim_to_cons(qL_left), prim_to_cons(qR_left));
  Cons FxR = hllc_x(prim_to_cons(qL_right), prim_to_cons(qR_right));

  Prim qB_bot, qT_bot;
  {
    if (y - 1 >= 0 && !mask[d_idx(x, y - 1)]) {
      FacePrim fp = reconstruct_y(U, mask, x, y - 1);
      qB_bot = fp.R;
      Cons GBf = flux_y(prim_to_cons(fp.R));
      Cons GBb = flux_y(prim_to_cons(fp.L));
      qB_bot =
          half_step_predict_y(fp.R, (GBf.rho - GBb.rho), (GBf.mx - GBb.mx),
                              (GBf.my - GBb.my), (GBf.E - GBb.E), half_dt_dy);
    } else {
      Cons UBc = neighbor_or_wall(U, mask, x, y, 0, -1, 0, 1);
      qB_bot = cons_to_prim(UBc);
    }

    FacePrim fpC = reconstruct_y(U, mask, x, y);
    qT_bot = fpC.L;
    Cons GTf = flux_y(prim_to_cons(fpC.R));
    Cons GTb = flux_y(prim_to_cons(fpC.L));
    qT_bot =
        half_step_predict_y(fpC.L, (GTf.rho - GTb.rho), (GTf.mx - GTb.mx),
                            (GTf.my - GTb.my), (GTf.E - GTb.E), half_dt_dy);

    qB_bot.rho = d_fmax(qB_bot.rho, EPS_RHO);
    qB_bot.p = d_fmax(qB_bot.p, EPS_P);
    qT_bot.rho = d_fmax(qT_bot.rho, EPS_RHO);
    qT_bot.p = d_fmax(qT_bot.p, EPS_P);
  }

  Prim qB_top, qT_top;
  {
    FacePrim fpC = reconstruct_y(U, mask, x, y);
    qB_top = fpC.R;
    Cons CFf = flux_y(prim_to_cons(fpC.R));
    Cons CFb = flux_y(prim_to_cons(fpC.L));
    qB_top =
        half_step_predict_y(fpC.R, (CFf.rho - CFb.rho), (CFf.mx - CFb.mx),
                            (CFf.my - CFb.my), (CFf.E - CFb.E), half_dt_dy);

    if (y + 1 < H && !mask[d_idx(x, y + 1)]) {
      FacePrim fpN = reconstruct_y(U, mask, x, y + 1);
      qT_top = fpN.L;
      Cons NFf = flux_y(prim_to_cons(fpN.R));
      Cons NFb = flux_y(prim_to_cons(fpN.L));
      qT_top =
          half_step_predict_y(fpN.L, (NFf.rho - NFb.rho), (NFf.mx - NFb.mx),
                              (NFf.my - NFb.my), (NFf.E - NFb.E), half_dt_dy);
    } else {
      Cons UTc = neighbor_or_wall(U, mask, x, y, 0, +1, 0, 1);
      qT_top = cons_to_prim(UTc);
    }

    qB_top.rho = d_fmax(qB_top.rho, EPS_RHO);
    qB_top.p = d_fmax(qB_top.p, EPS_P);
    qT_top.rho = d_fmax(qT_top.rho, EPS_RHO);
    qT_top.p = d_fmax(qT_top.p, EPS_P);
  }

  Cons GyB = hllc_y(prim_to_cons(qB_bot), prim_to_cons(qT_bot));
  Cons GyT = hllc_y(prim_to_cons(qB_top), prim_to_cons(qT_top));

  // Update
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

  // positivity
  Un.rho = d_fmax(Un.rho, EPS_RHO);
  Prim pp = cons_to_prim(Un);
  if (pp.p <= EPS_P) {
    pp.p = EPS_P;
    Un = prim_to_cons(pp);
  }

  store_cons(Unew, i, Un);
}

__global__ void k_swap(Usoa U, Usoa Unew) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = W * H;
  if (i >= N)
    return;
  store_cons(U, i, load_cons(Unew, i));
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
    Cons c = load_cons(U, i);
    Prim p = cons_to_prim(c);

    if (view_mode == 0)
      v = log(p.rho);
    else if (view_mode == 1)
      v = log(p.p);
    else if (view_mode == 2)
      v = sqrt(p.u * p.u + p.v * p.v);
    else {
      // schlieren
      int x = i % W;
      int y = i / W;

      auto sample_rho = [&](int sx, int sy) -> double {
        if (sy < 0)
          sy = 0;
        if (sy >= H)
          sy = H - 1;

        if (sx < 0) {
          return inflow_state().rho;
        }
        if (sx >= W) {
          int j = d_idx(W - 1, sy);
          return cons_to_prim(load_cons(U, j)).rho;
        }
        int j = d_idx(sx, sy);
        return cons_to_prim(load_cons(U, j)).rho;
      };

      double rhoL = sample_rho(x - 1, y);
      double rhoR = sample_rho(x + 1, y);
      double rhoB = sample_rho(x, y - 1);
      double rhoT = sample_rho(x, y + 1);
      double gx = 0.5 * (rhoR - rhoL);
      double gy = 0.5 * (rhoT - rhoB);
      v = log(1e-12 + sqrt(gx * gx + gy * gy));
    }

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
                                double minv, double invRange, uchar4 *out) {
  int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  int N = W * H;
  if (i >= N)
    return;

  if (mask[i]) {
    out[i] = pack_rgba(110, 110, 110);
    return;
  }

  double t = (tmpVal[i] - minv) * invRange;
  uint8_t r, g, b;
  get_color(t, r, g, b);
  out[i] = pack_rgba(r, g, b);
}

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

static double reduce_max(double *d_block, int blocks) {
  double *h = (double *)malloc(blocks * sizeof(double));
  CK(cudaMemcpy(h, d_block, blocks * sizeof(double), cudaMemcpyDeviceToHost));
  double m = 1e-12;
  for (int i = 0; i < blocks; i++)
    if (h[i] > m)
      m = h[i];
  free(h);
  return m;
}

static void reduce_minmax(double *d_min, double *d_max, int blocks,
                          double *outMin, double *outMax) {
  double *hmin = (double *)malloc(blocks * sizeof(double));
  double *hmax = (double *)malloc(blocks * sizeof(double));
  CK(cudaMemcpy(hmin, d_min, blocks * sizeof(double), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hmax, d_max, blocks * sizeof(double), cudaMemcpyDeviceToHost));
  double mn = 1e300, mx = -1e300;
  for (int i = 0; i < blocks; i++) {
    if (hmin[i] < mn)
      mn = hmin[i];
    if (hmax[i] > mx)
      mx = hmax[i];
  }
  free(hmin);
  free(hmax);
  *outMin = mn;
  *outMax = mx;
}

int main(void) {
  InitWindow(W * SCALE, H * SCALE,
             "Hypersonic 2D Flow (CUDA MUSCL-Hancock + HLLC)");
  SetTargetFPS(60);

  const int N = W * H;
  unsigned char *pixels = (unsigned char *)malloc(N * 4);

  Image img = {0};
  img.data = pixels;
  img.width = W;
  img.height = H;
  img.mipmaps = 1;
  img.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
  Texture2D tex = LoadTextureFromImage(img);

  Usoa dU{}, dUnew{};
  alloc_Us(&dU, N);
  alloc_Us(&dUnew, N);

  uint8_t *dMask = nullptr;
  CK(cudaMalloc(&dMask, N * sizeof(uint8_t)));

  double *dTmpVal = nullptr;
  CK(cudaMalloc(&dTmpVal, N * sizeof(double)));

  uchar4 *dPixels = nullptr;
  CK(cudaMalloc(&dPixels, N * sizeof(uchar4)));

  const int threads = 256;
  const int blocksN = (N + threads - 1) / threads;

  double *dBlockMax = nullptr;
  CK(cudaMalloc(&dBlockMax, blocksN * sizeof(double)));

  double *dBlockMin = nullptr;
  double *dBlockMax2 = nullptr;
  CK(cudaMalloc(&dBlockMin, blocksN * sizeof(double)));
  CK(cudaMalloc(&dBlockMax2, blocksN * sizeof(double)));

  auto gpu_init = [&]() {
    k_init<<<blocksN, threads>>>(dU, dMask);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());
  };

  gpu_init();

  double sim_t = 0.0;
  int view_mode = 0;

  while (!WindowShouldClose()) {
    if (IsKeyPressed(KEY_R)) {
      sim_t = 0.0;
      gpu_init();
    }
    if (IsKeyPressed(KEY_M))
      view_mode = (view_mode + 1) % 4;

    if (!IsKeyDown(KEY_SPACE)) {
      for (int k = 0; k < STEPS_PER_FRAME; k++) {
        k_apply_inflow_left<<<(H + threads - 1) / threads, threads>>>(dU,
                                                                      dMask);
        CK(cudaGetLastError());

        k_max_wavespeed<<<blocksN, threads>>>(dU, dMask, dBlockMax);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());

        double maxs = reduce_max(dBlockMax, blocksN);
        double dt = CFL * 1.0 / maxs; // dx=dy=1
        double dt_dx = dt;
        double dt_dy = dt;
        double half_dt_dx = 0.5 * dt_dx;
        double half_dt_dy = 0.5 * dt_dy;

        // step
        k_step<<<blocksN, threads>>>(dU, dUnew, dMask, dt_dx, dt_dy, half_dt_dx,
                                     half_dt_dy);
        CK(cudaGetLastError());

        // swap (copy new -> old)
        k_swap<<<blocksN, threads>>>(dU, dUnew);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());

        sim_t += dt;
      }
    }

    // render pass A: vals + per-block min/max
    k_render_vals<<<blocksN, threads>>>(dU, dMask, view_mode, dTmpVal,
                                        dBlockMin, dBlockMax2);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    double minv, maxv;
    reduce_minmax(dBlockMin, dBlockMax2, blocksN, &minv, &maxv);

    double invRange = 1.0 / fmax(maxv - minv, 1e-30);

    // render pass B: pixels
    k_render_pixels<<<blocksN, threads>>>(dMask, dTmpVal, minv, invRange,
                                          dPixels);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    // copy pixels to host RGBA
    CK(cudaMemcpy(pixels, dPixels, N * sizeof(uchar4), cudaMemcpyDeviceToHost));
    UpdateTexture(tex, pixels);

    BeginDrawing();
    ClearBackground(BLACK);
    DrawTexturePro(tex, (Rectangle){0, 0, (float)W, (float)H},
                   (Rectangle){0, 0, (float)(W * SCALE), (float)(H * SCALE)},
                   (Vector2){0, 0}, 0.0f, WHITE);

    DrawText(TextFormat("t = %.4f", sim_t), 10, 10, 20, WHITE);
    const char *modestr = (view_mode == 0)   ? "log(rho)"
                          : (view_mode == 1) ? "log(p)"
                          : (view_mode == 2) ? "speed"
                                             : "schlieren-ish";
    DrawText(modestr, 10, 34, 20, GREEN);
    DrawText("SPACE pause | R reset | M mode", 10, 58, 18,
             (Color){200, 200, 200, 255});
    EndDrawing();
  }

  CloseWindow();
  free(pixels);

  cudaFree(dMask);
  cudaFree(dTmpVal);
  cudaFree(dPixels);
  cudaFree(dBlockMax);
  cudaFree(dBlockMin);
  cudaFree(dBlockMax2);

  free_Us(&dU);
  free_Us(&dUnew);

  return 0;
}
