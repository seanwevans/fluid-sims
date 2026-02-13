/* tau_hypersonic.c
   gcc -O3 tau_hypersonic.c -lraylib -lm -o tau_2d
   SPACE pause, R reset, M toggle view
 */

#include "raylib.h"
#include <math.h>
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

typedef struct {
  double rho;
  double mx;
  double my;
  double E;
} Cons;

typedef struct {
  double rho;
  double u;
  double v;
  double p;
} Prim;

static Cons U[W * H];
static Cons Unew[W * H];
static unsigned char mask[W * H];
static unsigned char *pixels;

static double sim_t = 0.0;
static int view_mode = 0; // 0 rho, 1 p, 2 speed, 3 schlieren

static inline int idx(int x, int y) { return y * W + x; }
static inline double sqr(double a) { return a * a; }

static inline double minmod(double a, double b) {
  if (a * b <= 0.0)
    return 0.0;
  return (fabs(a) < fabs(b)) ? a : b;
}

static inline double mc_limiter(double dl, double dc, double dr) {
  // dl = q_i - q_{i-1}, dr = q_{i+1} - q_i, dc = 0.5*(q_{i+1}-q_{i-1})
  double mm1 = minmod(dl, dr);
  double mm2 = minmod(dc, 2.0 * dl);
  double mm3 = minmod(dc, 2.0 * dr);
  return minmod(mm1, minmod(mm2, mm3));
}

static inline Prim cons_to_prim(Cons c) {
  Prim p;
  double rho = fmax(c.rho, EPS_RHO);
  double inv = 1.0 / rho;
  double u = c.mx * inv;
  double v = c.my * inv;

  double kin = 0.5 * rho * (u * u + v * v);
  double eint = c.E - kin;
  double pr = (GAMMA - 1.0) * fmax(eint, EPS_P);

  p.rho = rho;
  p.u = u;
  p.v = v;
  p.p = pr;
  return p;
}

static inline Cons prim_to_cons(Prim p) {
  Cons c;
  double rho = fmax(p.rho, EPS_RHO);
  double pr = fmax(p.p, EPS_P);

  c.rho = rho;
  c.mx = rho * p.u;
  c.my = rho * p.v;
  c.E = pr / (GAMMA - 1.0) + 0.5 * rho * (p.u * p.u + p.v * p.v);
  return c;
}

static inline double sound_speed(Prim p) {
  return sqrt(GAMMA * fmax(p.p, EPS_P) / fmax(p.rho, EPS_RHO));
}

static inline Cons flux_x(Cons c) {
  Prim p = cons_to_prim(c);
  Cons f;
  f.rho = c.mx;
  f.mx = c.mx * p.u + p.p;
  f.my = c.my * p.u;
  f.E = (c.E + p.p) * p.u;
  return f;
}

static inline Cons flux_y(Cons c) {
  Prim p = cons_to_prim(c);
  Cons f;
  f.rho = c.my;
  f.mx = c.mx * p.v;
  f.my = c.my * p.v + p.p;
  f.E = (c.E + p.p) * p.v;
  return f;
}

static inline Cons hllc_x(Cons UL, Cons UR) {
  Prim L = cons_to_prim(UL);
  Prim R = cons_to_prim(UR);

  double aL = sound_speed(L);
  double aR = sound_speed(R);

  // wavespeed estimates
  double SL = fmin(L.u - aL, R.u - aR);
  double SR = fmax(L.u + aL, R.u + aR);

  Cons FL = flux_x(UL);
  Cons FR = flux_x(UR);

  if (SL >= 0.0)
    return FL;
  if (SR <= 0.0)
    return FR;

  // Contact speed SM (Toro)
  double rhoL = L.rho, rhoR = R.rho;
  double uL = L.u, uR = R.u;
  double pL = L.p, pR = R.p;

  double num = pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR);
  double den = rhoL * (SL - uL) - rhoR * (SR - uR);
  double SM = num / den;

  // Star pressure
  double pStar = pL + rhoL * (SL - uL) * (SM - uL);
  pStar = fmax(pStar, EPS_P);

  // Left star state
  double rhoStarL = rhoL * (SL - uL) / (SL - SM);
  double mxStarL = rhoStarL * SM;
  double myStarL = rhoStarL * L.v;
  double EL = UL.E;
  double EStarL = ((SL - uL) * EL - pL * uL + pStar * SM) / (SL - SM);
  Cons UStarL = {rhoStarL, mxStarL, myStarL, EStarL};

  // Right star state
  double rhoStarR = rhoR * (SR - uR) / (SR - SM);
  double mxStarR = rhoStarR * SM;
  double myStarR = rhoStarR * R.v;
  double ER = UR.E;
  double EStarR = ((SR - uR) * ER - pR * uR + pStar * SM) / (SR - SM);
  Cons UStarR = {rhoStarR, mxStarR, myStarR, EStarR};

  // HLLC flux
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

static inline Cons hllc_y(Cons UL, Cons UR) {
  Prim L = cons_to_prim(UL);
  Prim R = cons_to_prim(UR);

  double aL = sound_speed(L);
  double aR = sound_speed(R);

  double SL = fmin(L.v - aL, R.v - aR);
  double SR = fmax(L.v + aL, R.v + aR);

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
  pStar = fmax(pStar, EPS_P);

  double rhoStarL = rhoL * (SL - vL) / (SL - SM);
  double mxStarL = rhoStarL * L.u;
  double myStarL = rhoStarL * SM;

  double EL = UL.E;
  double EStarL = ((SL - vL) * EL - pL * vL + pStar * SM) / (SL - SM);
  Cons UStarL = {rhoStarL, mxStarL, myStarL, EStarL};

  double rhoStarR = rhoR * (SR - vR) / (SR - SM);
  double mxStarR = rhoStarR * R.u;
  double myStarR = rhoStarR * SM;

  double ER = UR.E;
  double EStarR = ((SR - vR) * ER - pR * vR + pStar * SM) / (SR - SM);
  Cons UStarR = {rhoStarR, mxStarR, myStarR, EStarR};

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

static inline Prim inflow_state() {
  // Mach 15 inflow, ideal gas, nondimensional
  const double mach = 15.0;
  const double rho = 1.0;
  const double p = 1.0;
  double a = sqrt(GAMMA * p / rho);
  double u = mach * a;
  Prim s = {rho, u, 0.0, p};
  return s;
}

static inline Cons get_cell_with_bc(int x, int y) {
  if (y < 0)
    y = 0;
  if (y >= H)
    y = H - 1;

  // Left boundary inflow
  if (x < 0) {
    return prim_to_cons(inflow_state());
  }

  // Right boundary outflow
  if (x >= W) {
    return U[idx(W - 1, y)];
  }

  // Inside domain
  int i = idx(x, y);

  // If solid
  return U[i];
}

static inline Cons reflect_slip(Cons inside, double nx, double ny) {
  // Reflect normal velocity at a slip wall: v_n -> -v_n, v_t unchanged
  Prim p = cons_to_prim(inside);
  double vn = p.u * nx + p.v * ny;
  double ut = -p.u * ny + p.v * nx;

  vn = -vn;

  // back to (u,v)
  double u = vn * nx - ut * ny;
  double v = vn * ny + ut * nx;

  Prim g = {p.rho, u, v, p.p};
  return prim_to_cons(g);
}

static inline Cons neighbor_or_wall(int x, int y, int nx_cell, int ny_cell,
                                    double nx, double ny) {
  int xn = x + nx_cell;
  int yn = y + ny_cell;

  if (xn < 0)
    return prim_to_cons(inflow_state());
  if (xn >= W)
    return U[idx(W - 1, y)];
  if (yn < 0)
    yn = 0;
  if (yn >= H)
    yn = H - 1;

  int j = idx(xn, yn);
  if (mask[j]) {
    return reflect_slip(U[idx(x, y)], nx, ny);
  }
  return U[j];
}

typedef struct {
  Prim L, R;
} FacePrim;

static inline void enforce_positive_faces(Prim *qm, Prim qc, Prim *qp) {
  for (int it = 0; it < 8; it++) {
    int bad = 0;
    if (qm->rho <= EPS_RHO || qp->rho <= EPS_RHO)
      bad = 1;
    if (qm->p <= EPS_P || qp->p <= EPS_P)
      bad = 1;
    if (!bad)
      return;

    // contract
    qm->rho = 0.5 * (qm->rho + qc.rho);
    qm->u = 0.5 * (qm->u + qc.u);
    qm->v = 0.5 * (qm->v + qc.v);
    qm->p = 0.5 * (qm->p + qc.p);

    qp->rho = 0.5 * (qp->rho + qc.rho);
    qp->u = 0.5 * (qp->u + qc.u);
    qp->v = 0.5 * (qp->v + qc.v);
    qp->p = 0.5 * (qp->p + qc.p);
  }

  qm->rho = fmax(qm->rho, EPS_RHO);
  qp->rho = fmax(qp->rho, EPS_RHO);
  qm->p = fmax(qm->p, EPS_P);
  qp->p = fmax(qp->p, EPS_P);
}

static inline FacePrim reconstruct_x(int x, int y) {
  Cons Uc = U[idx(x, y)];
  Cons Um = neighbor_or_wall(x, y, -1, 0, 1, 0);
  Cons Up = neighbor_or_wall(x, y, +1, 0, 1, 0);

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

  Prim qL = {qc.rho - 0.5 * s_rho, qc.u - 0.5 * s_u, qc.v - 0.5 * s_v,
             qc.p - 0.5 * s_p};
  Prim qR = {qc.rho + 0.5 * s_rho, qc.u + 0.5 * s_u, qc.v + 0.5 * s_v,
             qc.p + 0.5 * s_p};

  enforce_positive_faces(&qL, qc, &qR);

  FacePrim fp = {qL, qR};
  return fp;
}

static inline FacePrim reconstruct_y(int x, int y) {
  Cons Uc = U[idx(x, y)];
  Cons Um = neighbor_or_wall(x, y, 0, -1, 0, 1);
  Cons Up = neighbor_or_wall(x, y, 0, +1, 0, 1);

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

  Prim qL = {qc.rho - 0.5 * s_rho, qc.u - 0.5 * s_u, qc.v - 0.5 * s_v,
             qc.p - 0.5 * s_p};
  Prim qR = {qc.rho + 0.5 * s_rho, qc.u + 0.5 * s_u, qc.v + 0.5 * s_v,
             qc.p + 0.5 * s_p};

  enforce_positive_faces(&qL, qc, &qR);

  FacePrim fp = {qL, qR};
  return fp;
}

static inline Prim half_step_predict_x(Prim q, double dF_rho, double dF_mx,
                                       double dF_my, double dF_E,
                                       double dt_dx_half) {
  Cons c = prim_to_cons(q);
  c.rho -= dt_dx_half * dF_rho;
  c.mx -= dt_dx_half * dF_mx;
  c.my -= dt_dx_half * dF_my;
  c.E -= dt_dx_half * dF_E;

  Prim out = cons_to_prim(c);
  out.rho = fmax(out.rho, EPS_RHO);
  out.p = fmax(out.p, EPS_P);
  return out;
}

static inline Prim half_step_predict_y(Prim q, double dG_rho, double dG_mx,
                                       double dG_my, double dG_E,
                                       double dt_dy_half) {
  Cons c = prim_to_cons(q);
  c.rho -= dt_dy_half * dG_rho;
  c.mx -= dt_dy_half * dG_mx;
  c.my -= dt_dy_half * dG_my;
  c.E -= dt_dy_half * dG_E;

  Prim out = cons_to_prim(c);
  out.rho = fmax(out.rho, EPS_RHO);
  out.p = fmax(out.p, EPS_P);
  return out;
}

static void init_sim(void) {
  sim_t = 0.0;

  // obstacle
  int cx = W / 3;
  int cy = H / 2;
  int r = H / 6;

  Prim inflow = inflow_state();

  for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
      int i = idx(x, y);
      int dx = x - cx;
      int dy = y - cy;
      mask[i] = (dx * dx + dy * dy < r * r) ? 1 : 0;

      if (mask[i]) {
        Prim s = {inflow.rho, 0.0, 0.0, inflow.p};
        U[i] = prim_to_cons(s);
      } else {
        U[i] = prim_to_cons(inflow);
      }
    }
  }
}

static double compute_dt(void) {
  double maxs = 1e-12;
  for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
      int i = idx(x, y);
      if (mask[i])
        continue;
      Prim p = cons_to_prim(U[i]);
      double a = sound_speed(p);
      double sx = fabs(p.u) + a;
      double sy = fabs(p.v) + a;
      if (sx > maxs)
        maxs = sx;
      if (sy > maxs)
        maxs = sy;
    }
  }

  double dx = 1.0, dy = 1.0;
  double dt = CFL * fmin(dx, dy) / maxs;
  return dt;
}

static void step_physics(void) {
  double dx = 1.0, dy = 1.0;
  double dt = compute_dt();
  double dt_dx = dt / dx;
  double dt_dy = dt / dy;
  double half_dt_dx = 0.5 * dt_dx;
  double half_dt_dy = 0.5 * dt_dy;

  // explicit inflow boundary
  Cons inflowC = prim_to_cons(inflow_state());
  for (int y = 0; y < H; y++) {
    int i0 = idx(0, y);
    if (!mask[i0])
      U[i0] = inflowC;
  }

  memcpy(Unew, U, sizeof(U));

  for (int y = 0; y < H; y++) {
    for (int x = 1; x < W; x++) {
      int iL = idx(x - 1, y);
      int iR = idx(x, y);

      if (mask[iL] && mask[iR])
        continue;

      Cons UL = mask[iL] ? reflect_slip(U[iR], 1, 0) : U[iL];
      Cons UR = mask[iR] ? reflect_slip(U[iL], 1, 0) : U[iR];

      Prim qL, qR;

      if (!mask[iL]) {
        FacePrim fp = reconstruct_x(x - 1, y);
        qL = fp.R; // right face of left cell
      } else {
        qL = cons_to_prim(UL);
      }

      if (!mask[iR]) {
        FacePrim fp = reconstruct_x(x, y);
        qR = fp.L; // left face of right cell
      } else {
        qR = cons_to_prim(UR);
      }

      // Left cell predictor:
      if (!mask[iL]) {
        FacePrim fp = reconstruct_x(x - 1, y);
        Cons ULc = prim_to_cons(cons_to_prim(U[iL]));
        Cons FLf = flux_x(prim_to_cons(fp.R));
        Cons FLb = flux_x(prim_to_cons(fp.L));
        // dF = F_{i+1/2} - F_{i-1/2}
        qL =
            half_step_predict_x(fp.R, (FLf.rho - FLb.rho), (FLf.mx - FLb.mx),
                                (FLf.my - FLb.my), (FLf.E - FLb.E), half_dt_dx);
        (void)ULc;
      }

      // Right cell predictor:
      if (!mask[iR]) {
        FacePrim fp = reconstruct_x(x, y);
        Cons URc = prim_to_cons(cons_to_prim(U[iR]));
        Cons FRf = flux_x(prim_to_cons(fp.R));
        Cons FRb = flux_x(prim_to_cons(fp.L));
        qR =
            half_step_predict_x(fp.L, (FRf.rho - FRb.rho), (FRf.mx - FRb.mx),
                                (FRf.my - FRb.my), (FRf.E - FRb.E), half_dt_dx);
        (void)URc;
      }

      qL.rho = fmax(qL.rho, EPS_RHO);
      qL.p = fmax(qL.p, EPS_P);
      qR.rho = fmax(qR.rho, EPS_RHO);
      qR.p = fmax(qR.p, EPS_P);

      Cons FLUX = hllc_x(prim_to_cons(qL), prim_to_cons(qR));

      if (!mask[iL]) {
        Unew[iL].rho -= dt_dx * FLUX.rho;
        Unew[iL].mx -= dt_dx * FLUX.mx;
        Unew[iL].my -= dt_dx * FLUX.my;
        Unew[iL].E -= dt_dx * FLUX.E;
      }
      if (!mask[iR]) {
        Unew[iR].rho += dt_dx * FLUX.rho;
        Unew[iR].mx += dt_dx * FLUX.mx;
        Unew[iR].my += dt_dx * FLUX.my;
        Unew[iR].E += dt_dx * FLUX.E;
      }
    }
  }

  for (int y = 1; y < H; y++) {
    for (int x = 0; x < W; x++) {
      int iB = idx(x, y - 1);
      int iT = idx(x, y);

      if (mask[iB] && mask[iT])
        continue;

      Cons UB = mask[iB] ? reflect_slip(U[iT], 0, 1) : U[iB];
      Cons UT = mask[iT] ? reflect_slip(U[iB], 0, 1) : U[iT];

      Prim qB, qT;

      if (!mask[iB]) {
        FacePrim fp = reconstruct_y(x, y - 1);
        qB = fp.R; // top face of bottom cell
      } else {
        qB = cons_to_prim(UB);
      }

      if (!mask[iT]) {
        FacePrim fp = reconstruct_y(x, y);
        qT = fp.L; // bottom face of top cell
      } else {
        qT = cons_to_prim(UT);
      }

      // Predictor half-step for y
      if (!mask[iB]) {
        FacePrim fp = reconstruct_y(x, y - 1);
        Cons GBf = flux_y(prim_to_cons(fp.R));
        Cons GBb = flux_y(prim_to_cons(fp.L));
        qB =
            half_step_predict_y(fp.R, (GBf.rho - GBb.rho), (GBf.mx - GBb.mx),
                                (GBf.my - GBb.my), (GBf.E - GBb.E), half_dt_dy);
      }
      if (!mask[iT]) {
        FacePrim fp = reconstruct_y(x, y);
        Cons GTf = flux_y(prim_to_cons(fp.R));
        Cons GTb = flux_y(prim_to_cons(fp.L));
        qT =
            half_step_predict_y(fp.L, (GTf.rho - GTb.rho), (GTf.mx - GTb.mx),
                                (GTf.my - GTb.my), (GTf.E - GTb.E), half_dt_dy);
      }

      qB.rho = fmax(qB.rho, EPS_RHO);
      qB.p = fmax(qB.p, EPS_P);
      qT.rho = fmax(qT.rho, EPS_RHO);
      qT.p = fmax(qT.p, EPS_P);

      Cons FLUX = hllc_y(prim_to_cons(qB), prim_to_cons(qT));

      if (!mask[iB]) {
        Unew[iB].rho -= dt_dy * FLUX.rho;
        Unew[iB].mx -= dt_dy * FLUX.mx;
        Unew[iB].my -= dt_dy * FLUX.my;
        Unew[iB].E -= dt_dy * FLUX.E;
      }
      if (!mask[iT]) {
        Unew[iT].rho += dt_dy * FLUX.rho;
        Unew[iT].mx += dt_dy * FLUX.mx;
        Unew[iT].my += dt_dy * FLUX.my;
        Unew[iT].E += dt_dy * FLUX.E;
      }
    }
  }

  for (int i = 0; i < W * H; i++) {
    if (mask[i])
      continue;

    Unew[i].rho = fmax(Unew[i].rho, EPS_RHO);

    Prim p = cons_to_prim(Unew[i]);
    if (p.p <= EPS_P) {
      p.p = EPS_P;
      Unew[i] = prim_to_cons(p);
    }
    U[i] = Unew[i];
  }

  sim_t += dt;
}

static Color get_color(double t) {
  if (t < 0)
    t = 0;
  if (t > 1)
    t = 1;
  unsigned char r = (unsigned char)(255 * fmin(1, fmax(0, 3 * t - 1)));
  unsigned char g =
      (unsigned char)(255 * fmin(1, fmax(0, 2 - 4 * fabs(t - 0.5))));
  unsigned char b = (unsigned char)(255 * fmin(1, fmax(0, 2 - 3 * t)));
  return (Color){r, g, b, 255};
}

int main(void) {
  InitWindow(W * SCALE, H * SCALE, "Hypersonic 2D Flow (MUSCL-Hancock + HLLC)");
  SetTargetFPS(60);

  pixels = (unsigned char *)malloc(W * H * 4);
  Image img = {0};
  img.data = pixels;
  img.width = W;
  img.height = H;
  img.mipmaps = 1;
  img.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;

  Texture2D tex = LoadTextureFromImage(img);

  init_sim();

  while (!WindowShouldClose()) {
    if (IsKeyPressed(KEY_R))
      init_sim();
    if (IsKeyPressed(KEY_M))
      view_mode = (view_mode + 1) % 4;

    if (!IsKeyDown(KEY_SPACE)) {
      for (int k = 0; k < STEPS_PER_FRAME; k++)
        step_physics();
    }

    double minv = 1e300, maxv = -1e300;

    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
        int i = idx(x, y);
        if (mask[i])
          continue;

        Prim p = cons_to_prim(U[i]);
        double val = 0.0;

        if (view_mode == 0)
          val = log(p.rho);
        if (view_mode == 1)
          val = log(p.p);
        if (view_mode == 2)
          val = sqrt(p.u * p.u + p.v * p.v);
        if (view_mode == 3) {
          double rhoC = p.rho;
          double rhoL = cons_to_prim(get_cell_with_bc(x - 1, y)).rho;
          double rhoR = cons_to_prim(get_cell_with_bc(x + 1, y)).rho;
          double rhoB = cons_to_prim(get_cell_with_bc(x, y - 1)).rho;
          double rhoT = cons_to_prim(get_cell_with_bc(x, y + 1)).rho;
          double gx = 0.5 * (rhoR - rhoL);
          double gy = 0.5 * (rhoT - rhoB);
          val = log(1e-12 + sqrt(gx * gx + gy * gy));
        }

        if (val < minv)
          minv = val;
        if (val > maxv)
          maxv = val;
      }
    }

    // Render
    double inv = 1.0 / fmax(maxv - minv, 1e-30);
    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
        int i = idx(x, y);
        int pidx = 4 * i;

        if (mask[i]) {
          pixels[pidx + 0] = 110;
          pixels[pidx + 1] = 110;
          pixels[pidx + 2] = 110;
          pixels[pidx + 3] = 255;
          continue;
        }

        Prim p = cons_to_prim(U[i]);
        double val = 0.0;
        if (view_mode == 0)
          val = log(p.rho);
        if (view_mode == 1)
          val = log(p.p);
        if (view_mode == 2)
          val = sqrt(p.u * p.u + p.v * p.v);
        if (view_mode == 3) {
          double rhoL = cons_to_prim(get_cell_with_bc(x - 1, y)).rho;
          double rhoR = cons_to_prim(get_cell_with_bc(x + 1, y)).rho;
          double rhoB = cons_to_prim(get_cell_with_bc(x, y - 1)).rho;
          double rhoT = cons_to_prim(get_cell_with_bc(x, y + 1)).rho;
          double gx = 0.5 * (rhoR - rhoL);
          double gy = 0.5 * (rhoT - rhoB);
          val = log(1e-12 + sqrt(gx * gx + gy * gy));
        }

        double t = (val - minv) * inv;
        Color c = get_color(t);
        pixels[pidx + 0] = c.r;
        pixels[pidx + 1] = c.g;
        pixels[pidx + 2] = c.b;
        pixels[pidx + 3] = 255;
      }
    }

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
                                             : "schlieren";
    DrawText(modestr, 10, 34, 20, GREEN);
    DrawText("SPACE pause | R reset | M mode", 10, 58, 18,
             (Color){200, 200, 200, 255});
    EndDrawing();
  }

  CloseWindow();
  free(pixels);
  return 0;
}
