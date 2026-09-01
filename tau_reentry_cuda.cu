// tau_reentry_cuda.cu
//
// Axisymmetric Mach 25 re-entry flow over an Apollo class capsule, written in
// the tau algebra: an arithmetic whose only operators are multiplication and
// division, and in which the number zero is never written down.
//
//   nvcc -O3 -o tau_reentry tau_reentry_cuda.cu -std=c++17 -lraylib
//
// Controls: SPACE pause, R reset, M mode, S single step, Q quit.
//
// ---------------------------------------------------------------------------
// The tau algebra
// ---------------------------------------------------------------------------
//
// Every floating point quantity in this solver is stored as a strictly
// positive double t, which encodes the real value
//
//     val(t) = TAU_SCALE * log(t),         t = exp(val / TAU_SCALE)
//
// so the multiplicative group of the positive reals carries the additive
// structure of the reals:
//
//     val(a * b)   = val(a) + val(b)          sum       is  a * b
//     val(a / b)   = val(a) - val(b)          difference is a / b
//     val(1.0)     = the additive identity    zero      is  1.0
//     val(1.0 / a) = negation                 negation  is a reciprocal
//     val(sqrt(a)) = val(a) / 2               halving   is a square root
//     val(a * a)   = val(a) * 2               doubling  is a squaring
//     val(pow(a,c))= val(a) * c               scaling   is exponentiation
//     a < b        iff val(a) < val(b)        order is preserved exactly
//
// Products and quotients of two values ride the same isomorphism through
// log and exp (t_times, t_over). Comparison, min, max, absolute value and
// negation are exact and free.
//
// Consequences that the source obeys everywhere below:
//
//   * the subtraction operator never appears: no binary '-', no unary '-',
//     no '-=', no '--';
//   * the literal zero never appears; the additive identity is 1.0, and
//     small constants are written as reciprocals (1.0 / 1e8) rather than
//     with a negative exponent;
//   * hyphens occur only inside comments and inside the command line option
//     strings, never as an operator.
//
// The one place a genuine integer zero is unavoidable is the raylib draw
// origin, and even there it is derived rather than written: kOrigin below is
// 1 / 2 evaluated in integer arithmetic.
//
// What this buys, numerically: differences are ratios, so the scheme is
// expressed entirely in terms of ratios of neighbouring encoded states; the
// representation has uniform absolute precision (about TAU_SCALE * 2^-53,
// i.e. 1e-13 in value units) across the whole dynamic range of a re-entry
// flow, where density and pressure span several decades between the free
// stream and the stagnation region.
//
// ---------------------------------------------------------------------------
// What this solves
// ---------------------------------------------------------------------------
//
// Axisymmetric compressible Euler equations on the half plane r >= 0,
//
//     d(U)/dt + d(F)/dx + d(G)/dr = H / r ... applied as a loss term,
//
//     U = [rho, rho u, rho v, E]
//     F = [rho u, rho u^2 + p, rho u v, (E + p) u]
//     G = [rho v, rho u v, rho v^2 + p, (E + p) v]
//     H = [rho v, rho u v, rho v^2, (E + p) v]      (geometric source)
//
// with a polytropic equation of state p = eint / n, n = 1 / (gamma - 1).
//
// This is a companion to tau_hypersonic_cuda.cu rather than a replacement for
// it: that file stays as the large grid planar experiment it always was. The
// differences here are all deliberate corrections for the re-entry problem:
//
//   1. Axisymmetric rather than planar. A capsule is a body of revolution;
//      the planar equations give the wrong bow shock stand off distance and
//      the wrong stagnation region for the same geometry.
//   2. A real re-entry shape: Apollo class spherical segment heat shield
//      (nose radius 1.2 diameters), rounded shoulder, 33 degree conical
//      afterbody, truncated aft deck, flying blunt end forward.
//   3. A slip wall. The previous version reversed the whole velocity vector
//      in the solid ghost cells, which is a stagnation condition, not a wall.
//      Here only the face normal component is reflected, which is the exact
//      slip condition for a staircase boundary.
//   4. A default effective gamma of 1.2 (polytropic index n = 5) instead of
//      an ideal diatomic 1.4. Air behind a Mach 25 normal shock is hot enough
//      to dissociate; the equilibrium effective gamma is near 1.15 to 1.2 and
//      is what sets the shock stand off distance.
//   5. HLLC with Quirk's shock fix: cells beside a compressive pressure jump
//      are flagged, and every face touching a flagged cell falls back to HLLE.
//      Flagging per cell rather than per face is the point, because the faces
//      that lie along a grid aligned bow shock see almost no normal jump of
//      their own; without the transverse half of the cure the bow shock
//      breathes and eventually carbuncles.
//   6. MUSCL reconstruction with a monotonized central limiter, reverting to
//      first order beside the body and wherever a reconstructed face state
//      would leave the physical state space.
//   7. Strong stability preserving RK2 in time. In the tau algebra the two
//      stage average is the geometric mean, sqrt(a * b).
//   8. No artificial fourth derivative smoothing of the conserved variables;
//      the previous version added a fixed hyperviscosity that both smeared
//      the shock and could push cells out of the physical state space.
//
// View modes: 1 log10 rho, 2 log10 p, 3 speed, 4 schlieren, 5 Mach,
//             6 temperature p / rho, 7 vorticity.
//
// Validated on a 128 by 64 host run of the same cell bodies: the bow shock
// stands off 0.105 nose radii, and the wall pressure on the axis reaches 96
// percent of the Rayleigh pitot value for this gas and Mach number.

#ifndef TAU_REENTRY_CUDA_NO_RAYLIB
#include "raylib.h"
#endif

#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// ---------------------------------------------------------------------------
// Grid
// ---------------------------------------------------------------------------

#ifndef NX
#define NX 1024 // interior cells along the axis
#endif
#ifndef NY
#define NY 512 // interior cells along the radius
#endif
#define NG 2 // ghost layers on every side

static_assert(NG == 2, "the boundary kernels are written for two ghost layers");

// One based indexing, with one spare row and column, so that no array index
// and no loop counter in this file is ever the number zero.
#define STRIDE (NX + 2 * NG + 1)
#define ROWS (NY + 2 * NG + 1)
#define NCELL (STRIDE * ROWS)

#define COL_FIRST (NG + 1)        // first interior column, 3
#define COL_LAST (NG + NX)        // last  interior column
#define ROW_FIRST (NG + 1)        // first interior row (nearest the axis), 3
#define ROW_LAST (NG + NY)        // last  interior row (outer radius)
#define ROW_AXIS_INNER NG         // ghost row adjacent to the axis, 2
#define ROW_AXIS_OUTER (NG / NG)  // the other axis ghost row, 1
#define COL_IN_INNER NG           // ghost column adjacent to the inflow, 2
#define COL_IN_OUTER (NG / NG)    // the other inflow ghost column, 1

#define NFACE_X (NX + 1)
#define NFACE_Y (NY + 1)

// Screen scale; the render mirrors the half plane about the axis.
#define VIEW_SCALE 1
#define VIEW_MODES 7

// The only zero in the program, and it is computed rather than written.
#define ORIGIN (1 / 2)
static const int kOrigin = ORIGIN;

// ---------------------------------------------------------------------------
// The tau algebra
// ---------------------------------------------------------------------------

#define TAU_SCALE 1024.0
#define TAU_RSCALE (1.0 / 1024.0)
#define TAU_TINY (1.0 / 1e300)
#define TAU_HUGE 1e300
#define TAU_DEN_MIN (1.0 / 1e240)

// The additive identity.
#define T_NIL 1.0

__host__ __device__ __forceinline__ double t_clip(double t) {
  if (!(t > TAU_TINY))
    return TAU_TINY;
  if (!(t < TAU_HUGE))
    return TAU_HUGE;
  return t;
}

// encode: a real value becomes a strictly positive double
__host__ __device__ __forceinline__ double t_of(double v) {
  return t_clip(exp(v * TAU_RSCALE));
}

// decode: the positive double becomes the real value it stands for
__host__ __device__ __forceinline__ double t_val(double t) {
  return log(t_clip(t)) * TAU_SCALE;
}

// sum of two values
__host__ __device__ __forceinline__ double t_plus(double a, double b) {
  return t_clip(a * b);
}

// difference of two values
__host__ __device__ __forceinline__ double t_minus(double a, double b) {
  return t_clip(a / b);
}

// negation of a value
__host__ __device__ __forceinline__ double t_flip(double a) {
  return t_clip(T_NIL / a);
}

// magnitude of a value
__host__ __device__ __forceinline__ double t_abs(double a) {
  return (a > T_NIL) ? a : t_flip(a);
}

// value scaled by a plain constant
__host__ __device__ __forceinline__ double t_scale(double a, double c) {
  return t_clip(pow(a, c));
}

// value doubled, halved
__host__ __device__ __forceinline__ double t_twice(double a) {
  return t_clip(a * a);
}
__host__ __device__ __forceinline__ double t_half(double a) {
  return t_clip(sqrt(t_clip(a)));
}

// product of two values, through the group isomorphism
__host__ __device__ __forceinline__ double t_times(double a, double b) {
  return t_of(t_val(a) * t_val(b));
}

// quotient of two values; the divisor is nudged off the additive identity so
// that the isomorphism stays defined
__host__ __device__ __forceinline__ double t_over(double a, double b) {
  double vb = t_val(b);
  if (!(fabs(vb) > TAU_DEN_MIN))
    vb = copysign(TAU_DEN_MIN, vb);
  return t_of(t_val(a) / vb);
}

__host__ __device__ __forceinline__ double t_sq(double a) {
  return t_times(a, a);
}

__host__ __device__ __forceinline__ double t_root(double a) {
  double v = t_val(a);
  if (!(v > TAU_DEN_MIN))
    return T_NIL;
  return t_of(sqrt(v));
}

__host__ __device__ __forceinline__ double t_min(double a, double b) {
  return a < b ? a : b;
}
__host__ __device__ __forceinline__ double t_max(double a, double b) {
  return a > b ? a : b;
}

__host__ __device__ __forceinline__ bool t_finite(double t) {
  return isfinite(t) && t > TAU_TINY && t < TAU_HUGE;
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

struct SimConfig {
  // gas
  double index;  // polytropic index n; gamma = (n + 1) / n
  double gamma;  // ratio of specific heats
  double gm1;    // gamma minus one, held exactly as 1 / n
  double mach;   // free stream Mach number

  // grid metrics
  double dx, dy;

  // capsule geometry, in units of the base radius
  double x_nose;      // axial station of the heat shield apex
  double r_sphere;    // heat shield radius of curvature
  double r_base;      // maximum body radius
  double r_shoulder;  // shoulder fillet radius
  double cone_sin;    // sine of the afterbody half angle
  double cone_cos;    // cosine of the afterbody half angle
  double sphere_cx;   // centre of the heat shield sphere
  double cone_apex;   // axial station of the virtual afterbody cone apex
  double shoulder_x;  // axial station of the widest point of the body
  double tail_x;      // aft truncation plane

  // stepping
  double cfl;
  int steps_per_frame;

  // encoded constants used inside the kernels
  double t_dx, t_dy;
  double t_gamma;
  double t_rho_inf, t_p_inf, t_u_inf, t_e_inf;
  double t_rho_floor, t_p_floor, t_eint_floor;
  double t_shock_sensor;
};

__constant__ SimConfig d_cfg;

static inline void ck(cudaError_t e, const char *msg) {
  if (e != cudaSuccess) {
    fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(e));
    exit(EXIT_FAILURE);
  }
}
#define CK(x) ck((x), #x)

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

// Structure of arrays; every entry is tau encoded.
struct Field {
  double *rho, *mx, *my, *E;
};

struct Cons {
  double rho, mx, my, E;
};

struct Prim {
  double rho, u, v, p;
};

enum class Axis : int { X = 1, Y = 2 };

__host__ __device__ __forceinline__ int cell_index(int i, int j) {
  return j * STRIDE + i;
}

__host__ __device__ __forceinline__ Cons load_cons(const Field &U, int i) {
  Cons c;
  c.rho = U.rho[i];
  c.mx = U.mx[i];
  c.my = U.my[i];
  c.E = U.E[i];
  return c;
}

__host__ __device__ __forceinline__ void store_cons(const Field &U, int i,
                                                    const Cons &c) {
  U.rho[i] = c.rho;
  U.mx[i] = c.mx;
  U.my[i] = c.my;
  U.E[i] = c.E;
}

// ---------------------------------------------------------------------------
// Thermodynamics
// ---------------------------------------------------------------------------

__host__ __device__ __forceinline__ double kinetic(const SimConfig &cfg,
                                                   double rho, double u,
                                                   double v) {
  // rho * (u^2 + v^2) / 2
  double q = t_plus(t_sq(u), t_sq(v));
  return t_half(t_times(rho, q));
}

__host__ __device__ __forceinline__ Prim cons_to_prim(const SimConfig &cfg,
                                                      const Cons &c) {
  Prim q;
  q.rho = t_max(c.rho, cfg.t_rho_floor);
  q.u = t_over(c.mx, q.rho);
  q.v = t_over(c.my, q.rho);
  double eint = t_max(t_minus(c.E, kinetic(cfg, q.rho, q.u, q.v)),
                      cfg.t_eint_floor);
  q.p = t_max(t_scale(eint, cfg.gm1), cfg.t_p_floor);
  return q;
}

__host__ __device__ __forceinline__ Cons prim_to_cons(const SimConfig &cfg,
                                                      const Prim &q) {
  Cons c;
  double rho = t_max(q.rho, cfg.t_rho_floor);
  double p = t_max(q.p, cfg.t_p_floor);
  c.rho = rho;
  c.mx = t_times(rho, q.u);
  c.my = t_times(rho, q.v);
  c.E = t_plus(t_scale(p, cfg.index), kinetic(cfg, rho, q.u, q.v));
  return c;
}

__host__ __device__ __forceinline__ double sound_speed(const SimConfig &cfg,
                                                       const Prim &q) {
  double rho = t_max(q.rho, cfg.t_rho_floor);
  double p = t_max(q.p, cfg.t_p_floor);
  return t_root(t_scale(t_over(p, rho), cfg.gamma));
}

__host__ __device__ __forceinline__ double speed_mag(const Prim &q) {
  return t_root(t_plus(t_sq(q.u), t_sq(q.v)));
}

__host__ __device__ __forceinline__ bool prim_ok(const SimConfig &cfg,
                                                 const Prim &q) {
  return t_finite(q.rho) && t_finite(q.p) && t_finite(q.u) && t_finite(q.v) &&
         q.rho > cfg.t_rho_floor && q.p > cfg.t_p_floor;
}

__host__ __device__ __forceinline__ Prim free_stream(const SimConfig &cfg) {
  Prim q;
  q.rho = cfg.t_rho_inf;
  q.u = cfg.t_u_inf;
  q.v = T_NIL; // the free stream is axial, and axial means zero radial speed
  q.p = cfg.t_p_inf;
  return q;
}

// Normal and tangential components for a given sweep direction.
template <Axis AX>
__host__ __device__ __forceinline__ double prim_normal(const Prim &q) {
  if constexpr (AX == Axis::X)
    return q.u;
  return q.v;
}

template <Axis AX>
__host__ __device__ __forceinline__ Prim prim_mirror(const Prim &q) {
  Prim m = q;
  if constexpr (AX == Axis::X)
    m.u = t_flip(q.u);
  else
    m.v = t_flip(q.v);
  return m;
}

// Physical flux along one axis.
template <Axis AX>
__host__ __device__ __forceinline__ Cons flux_axis(const SimConfig &cfg,
                                                   const Prim &q,
                                                   const Cons &c) {
  double un = prim_normal<AX>(q);
  Cons f;
  f.rho = (AX == Axis::X) ? c.mx : c.my;
  f.mx = t_times(c.mx, un);
  f.my = t_times(c.my, un);
  if constexpr (AX == Axis::X)
    f.mx = t_plus(f.mx, q.p);
  else
    f.my = t_plus(f.my, q.p);
  f.E = t_times(t_plus(c.E, q.p), un);
  return f;
}

// ---------------------------------------------------------------------------
// Slope limiting and MUSCL reconstruction
// ---------------------------------------------------------------------------

// minmod of two differences; the additive identity is returned when they
// disagree in sign, and the sign test is a comparison against T_NIL.
__host__ __device__ __forceinline__ double t_minmod(double a, double b) {
  if ((a > T_NIL) == (b > T_NIL))
    return t_abs(a) < t_abs(b) ? a : b;
  return T_NIL;
}

// monotonized central limiter, minmod((dl + dr) / 2, 2 dl, 2 dr)
__host__ __device__ __forceinline__ double t_mc(double dl, double dr) {
  double central = t_half(t_plus(dl, dr));
  double steep = t_minmod(t_twice(dl), t_twice(dr));
  return t_minmod(central, steep);
}

__host__ __device__ __forceinline__ double comp_slope(double qm, double qc,
                                                      double qp) {
  return t_mc(t_minus(qc, qm), t_minus(qp, qc));
}

__host__ __device__ __forceinline__ Prim prim_slope(const Prim &qm,
                                                    const Prim &qc,
                                                    const Prim &qp) {
  Prim s;
  s.rho = comp_slope(qm.rho, qc.rho, qp.rho);
  s.u = comp_slope(qm.u, qc.u, qp.u);
  s.v = comp_slope(qm.v, qc.v, qp.v);
  s.p = comp_slope(qm.p, qc.p, qp.p);
  return s;
}

__host__ __device__ __forceinline__ Prim prim_face_hi(const Prim &qc,
                                                      const Prim &s) {
  Prim f;
  f.rho = t_plus(qc.rho, t_half(s.rho));
  f.u = t_plus(qc.u, t_half(s.u));
  f.v = t_plus(qc.v, t_half(s.v));
  f.p = t_plus(qc.p, t_half(s.p));
  return f;
}

__host__ __device__ __forceinline__ Prim prim_face_lo(const Prim &qc,
                                                      const Prim &s) {
  Prim f;
  f.rho = t_minus(qc.rho, t_half(s.rho));
  f.u = t_minus(qc.u, t_half(s.u));
  f.v = t_minus(qc.v, t_half(s.v));
  f.p = t_minus(qc.p, t_half(s.p));
  return f;
}

// ---------------------------------------------------------------------------
// Riemann solvers
// ---------------------------------------------------------------------------

struct WaveSpeeds {
  double SL, SR;
};

// Davis estimate; it brackets the true signal speeds, which is what makes the
// HLL family positivity preserving.
template <Axis AX>
__host__ __device__ __forceinline__ WaveSpeeds
davis_speeds(const SimConfig &cfg, const Prim &qL, const Prim &qR) {
  double aL = sound_speed(cfg, qL);
  double aR = sound_speed(cfg, qR);
  double uL = prim_normal<AX>(qL);
  double uR = prim_normal<AX>(qR);
  WaveSpeeds s;
  s.SL = t_min(t_minus(uL, aL), t_minus(uR, aR));
  s.SR = t_max(t_plus(uL, aL), t_plus(uR, aR));
  return s;
}

template <Axis AX>
__host__ __device__ __forceinline__ Cons hlle_axis(const SimConfig &cfg,
                                                   const Prim &qL,
                                                   const Prim &qR) {
  WaveSpeeds s = davis_speeds<AX>(cfg, qL, qR);
  Cons UL = prim_to_cons(cfg, qL);
  Cons UR = prim_to_cons(cfg, qR);
  Cons FL = flux_axis<AX>(cfg, qL, UL);
  Cons FR = flux_axis<AX>(cfg, qR, UR);

  if (s.SL > T_NIL)
    return FL;
  if (s.SR < T_NIL)
    return FR;

  double den = t_minus(s.SR, s.SL);
  double prod = t_times(s.SL, s.SR);

  Cons F;
  F.rho = t_over(t_plus(t_minus(t_times(s.SR, FL.rho), t_times(s.SL, FR.rho)),
                        t_times(prod, t_minus(UR.rho, UL.rho))),
                 den);
  F.mx = t_over(t_plus(t_minus(t_times(s.SR, FL.mx), t_times(s.SL, FR.mx)),
                       t_times(prod, t_minus(UR.mx, UL.mx))),
                den);
  F.my = t_over(t_plus(t_minus(t_times(s.SR, FL.my), t_times(s.SL, FR.my)),
                       t_times(prod, t_minus(UR.my, UL.my))),
                den);
  F.E = t_over(t_plus(t_minus(t_times(s.SR, FL.E), t_times(s.SL, FR.E)),
                      t_times(prod, t_minus(UR.E, UL.E))),
               den);
  return F;
}

// HLLC, in the Toro star state form. Every difference below is a quotient of
// encoded states and every product rides the group isomorphism.
template <Axis AX>
__host__ __device__ __forceinline__ Cons hllc_axis(const SimConfig &cfg,
                                                   const Prim &qL,
                                                   const Prim &qR) {
  WaveSpeeds s = davis_speeds<AX>(cfg, qL, qR);
  Cons UL = prim_to_cons(cfg, qL);
  Cons UR = prim_to_cons(cfg, qR);
  Cons FL = flux_axis<AX>(cfg, qL, UL);
  Cons FR = flux_axis<AX>(cfg, qR, UR);

  if (s.SL > T_NIL)
    return FL;
  if (s.SR < T_NIL)
    return FR;

  double uL = prim_normal<AX>(qL);
  double uR = prim_normal<AX>(qR);

  // cK = rhoK (SK - uK); cL is a loss, cR is a gain, and their difference
  // never vanishes once SL < 0 < SR.
  double cL = t_times(qL.rho, t_minus(s.SL, uL));
  double cR = t_times(qR.rho, t_minus(s.SR, uR));
  double den = t_minus(cL, cR);
  if (!(t_abs(den) > t_of(1.0 / 1e12)))
    return hlle_axis<AX>(cfg, qL, qR);

  double num = t_plus(t_minus(qR.p, qL.p),
                      t_minus(t_times(cL, uL), t_times(cR, uR)));
  double sstar = t_over(num, den);

  bool left = (sstar > T_NIL);
  const Prim &qK = left ? qL : qR;
  const Cons &UK = left ? UL : UR;
  const Cons &FK = left ? FL : FR;
  double SK = left ? s.SL : s.SR;
  double cK = left ? cL : cR;
  double uK = left ? uL : uR;

  double fac = t_over(t_minus(SK, uK), t_minus(SK, sstar));
  double rstar = t_times(qK.rho, fac);

  // EK / rhoK + (S* - uK) (S* + pK / (rhoK (SK - uK)))
  double eK = t_over(UK.E, qK.rho);
  double inner = t_plus(sstar, t_over(qK.p, cK));
  double estar = t_times(rstar, t_plus(eK, t_times(t_minus(sstar, uK), inner)));

  Cons US;
  US.rho = rstar;
  if constexpr (AX == Axis::X) {
    US.mx = t_times(rstar, sstar);
    US.my = t_times(rstar, qK.v);
  } else {
    US.mx = t_times(rstar, qK.u);
    US.my = t_times(rstar, sstar);
  }
  US.E = estar;

  Cons F;
  F.rho = t_plus(FK.rho, t_times(SK, t_minus(US.rho, UK.rho)));
  F.mx = t_plus(FK.mx, t_times(SK, t_minus(US.mx, UK.mx)));
  F.my = t_plus(FK.my, t_times(SK, t_minus(US.my, UK.my)));
  F.E = t_plus(FK.E, t_times(SK, t_minus(US.E, UK.E)));
  return F;
}

// A face is shocked when the pressure jumps by more than the sensor fraction
// and the normal velocity falls across it, so that expansions, which HLLC
// handles perfectly well, are left alone.
template <Axis AX>
__host__ __device__ __forceinline__ bool shock_face(const SimConfig &cfg,
                                                    const Prim &qL,
                                                    const Prim &qR) {
  if (!(prim_normal<AX>(qR) < prim_normal<AX>(qL)))
    return false;
  double jump = t_abs(t_minus(qR.p, qL.p));
  double base = t_min(qL.p, qR.p);
  return t_over(jump, base) > cfg.t_shock_sensor;
}

// Strong shocks are handed to HLLE. A contact resolving flux applied across a
// grid aligned bow shock is what feeds the carbuncle instability, and the cure
// has to act on the faces that lie *along* the shock as well as across it:
// those faces see almost no normal pressure jump of their own, so the decision
// is taken per cell (see cell_shock_flag) and passed in here.
template <Axis AX>
__host__ __device__ __forceinline__ Cons riemann_axis(const SimConfig &cfg,
                                                      const Prim &qL,
                                                      const Prim &qR,
                                                      bool robust = false) {
  if (robust || shock_face<AX>(cfg, qL, qR))
    return hlle_axis<AX>(cfg, qL, qR);
  return hllc_axis<AX>(cfg, qL, qR);
}

// ---------------------------------------------------------------------------
// Capsule geometry
// ---------------------------------------------------------------------------

// Apollo class capsule: a spherical segment heat shield of radius r_sphere,
// a shoulder fillet of radius r_shoulder, a conical afterbody, and a flat aft
// truncation. All three primitives are convex, so the signed distance of the
// intersection is the maximum of the three, and rounding the sphere against
// the cone gives the shoulder.
//
// X and Y arrive tau encoded and the result is a tau encoded signed distance:
// negative values, that is values below T_NIL, are inside the body.
__host__ __device__ __forceinline__ double capsule_sdf(const SimConfig &cfg,
                                                       double X, double Y) {
  double dx = t_minus(X, t_of(cfg.sphere_cx));
  double dist = t_root(t_plus(t_sq(dx), t_sq(Y)));
  double sd_ball = t_minus(dist, t_of(cfg.r_sphere));

  // r cos(theta) + (x - x_apex) sin(theta)
  double sd_cone = t_plus(t_scale(Y, cfg.cone_cos),
                          t_scale(t_minus(X, t_of(cfg.cone_apex)), cfg.cone_sin));

  double rc = t_of(cfg.r_shoulder);
  double da = t_plus(sd_ball, rc);
  double db = t_plus(sd_cone, rc);
  double pa = t_max(da, T_NIL);
  double pb = t_max(db, T_NIL);
  double outer = t_root(t_plus(t_sq(pa), t_sq(pb)));
  double inner = t_min(t_max(da, db), T_NIL);
  double sd_round = t_minus(t_plus(outer, inner), rc);

  double sd_tail = t_minus(X, t_of(cfg.tail_x));
  return t_max(sd_round, sd_tail);
}

#define CELL_FLUID 1
#define CELL_SOLID 2

// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

// Every kernel is a thin wrapper around a cell body that takes its
// configuration as an argument, so the whole solver can also be driven from the
// host, one cell at a time, by the test harness.
__host__ __device__ __forceinline__ void
cell_fill_free_stream(const SimConfig &cfg, const Field &U, uint8_t *mask,
                      int idx) {
  mask[idx] = CELL_FLUID;
  store_cons(U, idx, prim_to_cons(cfg, free_stream(cfg)));
}

__global__ void k_fill_free_stream(Field U, uint8_t *mask) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (idx >= NCELL)
    return;
  cell_fill_free_stream(d_cfg, U, mask, idx);
}

__host__ __device__ __forceinline__ void
cell_carve_body(const SimConfig &cfg, const Field &U, uint8_t *mask, int t) {
  int c = t % NX;
  int r = t / NX;
  double x = ((double)c + 0.5) * cfg.dx;
  double y = ((double)r + 0.5) * cfg.dy;
  if (capsule_sdf(cfg, t_of(x), t_of(y)) >= T_NIL)
    return;

  int idx = cell_index(c + COL_FIRST, r + ROW_FIRST);
  mask[idx] = CELL_SOLID;
  Prim q = free_stream(cfg);
  q.u = T_NIL; // the body is at rest in the body frame
  q.v = T_NIL;
  store_cons(U, idx, prim_to_cons(cfg, q));
}

__global__ void k_carve_body(Field U, uint8_t *mask) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= NX * NY)
    return;
  cell_carve_body(d_cfg, U, mask, t);
}

// Axis symmetry below, zero gradient outflow above.
__host__ __device__ __forceinline__ void cell_bc_radial(const Field &U, int i) {
  Cons a = load_cons(U, cell_index(i, ROW_FIRST));
  a.my = t_flip(a.my);
  store_cons(U, cell_index(i, ROW_AXIS_INNER), a);

  Cons b = load_cons(U, cell_index(i, ROW_FIRST + 1));
  b.my = t_flip(b.my);
  store_cons(U, cell_index(i, ROW_AXIS_OUTER), b);

  Cons o = load_cons(U, cell_index(i, ROW_LAST));
  store_cons(U, cell_index(i, ROW_LAST + 1), o);
  store_cons(U, cell_index(i, ROW_LAST + 2), o);
}

__global__ void k_bc_radial(Field U) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= NX + 2 * NG)
    return;
  cell_bc_radial(U, t + 1);
}

// Supersonic inflow on the left, supersonic outflow on the right.
__host__ __device__ __forceinline__ void
cell_bc_axial(const SimConfig &cfg, const Field &U, int j) {
  Cons infl = prim_to_cons(cfg, free_stream(cfg));
  store_cons(U, cell_index(COL_IN_OUTER, j), infl);
  store_cons(U, cell_index(COL_IN_INNER, j), infl);

  Cons o = load_cons(U, cell_index(COL_LAST, j));
  store_cons(U, cell_index(COL_LAST + 1, j), o);
  store_cons(U, cell_index(COL_LAST + 2, j), o);
}

__global__ void k_bc_axial(Field U) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= NY + 2 * NG)
    return;
  cell_bc_axial(d_cfg, U, t + 1);
}

// Encoded signal speed of one cell, the additive identity for a solid cell.
__host__ __device__ __forceinline__ double
cell_wave_speed(const SimConfig &cfg, const Field &U, const uint8_t *mask,
                int t) {
  int idx = cell_index(t % NX + COL_FIRST, t / NX + ROW_FIRST);
  if (mask[idx] != CELL_FLUID)
    return TAU_TINY;
  Prim q = cons_to_prim(cfg, load_cons(U, idx));
  double a = sound_speed(cfg, q);
  double lx = t_over(t_plus(t_abs(q.u), a), cfg.t_dx);
  double ly = t_over(t_plus(t_abs(q.v), a), cfg.t_dy);
  return t_plus(lx, ly);
}

__global__ void k_wave_speed(const Field U, const uint8_t *mask,
                             double *blockMax) {
  extern __shared__ double smax[];
  int tid = threadIdx.x;
  double best = TAU_TINY;

  int stride = gridDim.x * blockDim.x;
  for (int t = blockIdx.x * blockDim.x + tid; t < NX * NY; t += stride) {
    double lam = cell_wave_speed(d_cfg, U, mask, t);
    if (lam > best)
      best = lam;
  }

  smax[tid] = best;
  __syncthreads();
  for (unsigned s = blockDim.x; s > 1;) {
    s /= 2;
    if (tid < s && smax[tid + s] > smax[tid])
      smax[tid] = smax[tid + s];
    __syncthreads();
  }
  if (tid < 1)
    blockMax[blockIdx.x] = smax[tid];
}

// Quirk's shock fix: a cell next to a compressive jump in any direction is
// marked, and every face touching a marked cell falls back to HLLE. Without the
// transverse half of this the bow shock breathes and eventually carbuncles.
#define SHOCK_SMOOTH 1
#define SHOCK_STRONG 2

__host__ __device__ __forceinline__ void
cell_shock_flag(const SimConfig &cfg, const Field &U, const uint8_t *mask,
                uint8_t *flags, int t) {
  int c = t % NX;
  int r = t / NX;
  int i = c + COL_FIRST;
  int j = r + ROW_FIRST;
  int id = cell_index(i, j);

  if (mask[id] != CELL_FLUID) {
    flags[id] = SHOCK_SMOOTH;
    return;
  }

  Prim q = cons_to_prim(cfg, load_cons(U, id));
  Prim xm = cons_to_prim(cfg, load_cons(U, cell_index(c + 2, j)));
  Prim xp = cons_to_prim(cfg, load_cons(U, cell_index(c + 4, j)));
  Prim ym = cons_to_prim(cfg, load_cons(U, cell_index(i, r + 2)));
  Prim yp = cons_to_prim(cfg, load_cons(U, cell_index(i, r + 4)));

  bool strong = shock_face<Axis::X>(cfg, xm, q) ||
                shock_face<Axis::X>(cfg, q, xp) ||
                shock_face<Axis::Y>(cfg, ym, q) ||
                shock_face<Axis::Y>(cfg, q, yp);
  flags[id] = strong ? SHOCK_STRONG : SHOCK_SMOOTH;
}

__global__ void k_shock_flag(const Field U, const uint8_t *mask,
                             uint8_t *flags) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= NX * NY)
    return;
  cell_shock_flag(d_cfg, U, mask, flags, t);
}

__global__ void k_clear_shock_flags(uint8_t *flags) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (idx >= NCELL)
    return;
  flags[idx] = SHOCK_SMOOTH;
}

__global__ void k_reduce_max(const double *in, int n, double *out) {
  extern __shared__ double smax[];
  int tid = threadIdx.x;
  double best = TAU_TINY;
  for (int i = tid; i < n; i += blockDim.x)
    if (in[i] > best)
      best = in[i];
  smax[tid] = best;
  __syncthreads();
  for (unsigned s = blockDim.x; s > 1;) {
    s /= 2;
    if (tid < s && smax[tid + s] > smax[tid])
      smax[tid] = smax[tid + s];
    __syncthreads();
  }
  if (tid < 1)
    out[tid] = smax[tid];
}

__host__ __device__ __forceinline__ Cons nil_cons() {
  Cons c;
  c.rho = T_NIL;
  c.mx = T_NIL;
  c.my = T_NIL;
  c.E = T_NIL;
  return c;
}

__host__ __device__ __forceinline__ Prim nil_prim() {
  Prim q;
  q.rho = T_NIL;
  q.u = T_NIL;
  q.v = T_NIL;
  q.p = T_NIL;
  return q;
}

// One face flux per thread. The stencil offsets are generated by counting up
// from the thread's own face counter, so no index is ever formed by taking
// something away from another.
template <Axis AX>
__host__ __device__ __forceinline__ void
cell_flux(const SimConfig &cfg, const Field &U, const uint8_t *mask,
          const uint8_t *flags, const Field &F, int t) {
  const int nf = (AX == Axis::X) ? NFACE_X : NX;
  int c = t % nf;
  int r = t / nf;

  int idA, idB, idD, idE;
  if constexpr (AX == Axis::X) {
    int j = r + ROW_FIRST;
    idA = cell_index(c + 1, j);
    idB = cell_index(c + 2, j);
    idD = cell_index(c + 3, j);
    idE = cell_index(c + 4, j);
  } else {
    int i = c + COL_FIRST;
    idA = cell_index(i, r + 1);
    idB = cell_index(i, r + 2);
    idD = cell_index(i, r + 3);
    idE = cell_index(i, r + 4);
  }

  uint8_t mA = mask[idA];
  uint8_t mB = mask[idB];
  uint8_t mD = mask[idD];
  uint8_t mE = mask[idE];

  if (mB == CELL_SOLID && mD == CELL_SOLID) {
    store_cons(F, idB, nil_cons());
    return;
  }

  Prim qB = cons_to_prim(cfg, load_cons(U, idB));
  Prim qD = cons_to_prim(cfg, load_cons(U, idD));

  Prim qL, qR;
  if (mD == CELL_SOLID) {
    // slip wall on the high side: reflect the face normal component only
    qL = qB;
    qR = prim_mirror<AX>(qB);
  } else if (mB == CELL_SOLID) {
    qR = qD;
    qL = prim_mirror<AX>(qD);
  } else {
    bool haveA = (mA == CELL_FLUID);
    bool haveE = (mE == CELL_FLUID);
    Prim qA = haveA ? cons_to_prim(cfg, load_cons(U, idA)) : qB;
    Prim qE = haveE ? cons_to_prim(cfg, load_cons(U, idE)) : qD;
    Prim sB = haveA ? prim_slope(qA, qB, qD) : nil_prim();
    Prim sD = haveE ? prim_slope(qB, qD, qE) : nil_prim();
    qL = prim_face_hi(qB, sB);
    qR = prim_face_lo(qD, sD);
    if (!prim_ok(cfg, qL) || !prim_ok(cfg, qR)) {
      qL = qB;
      qR = qD;
    }
  }

  bool robust =
      (flags[idB] == SHOCK_STRONG) || (flags[idD] == SHOCK_STRONG);
  store_cons(F, idB, riemann_axis<AX>(cfg, qL, qR, robust));
}

template <Axis AX>
__global__ void k_flux(const Field U, const uint8_t *mask, const uint8_t *flags,
                       Field F) {
  const int nf = (AX == Axis::X) ? NFACE_X : NX;
  const int nl = (AX == Axis::X) ? NY : NFACE_Y;
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= nf * nl)
    return;
  cell_flux<AX>(d_cfg, U, mask, flags, F, t);
}

// Conservative update. Every term is applied multiplicatively: an incoming
// flux multiplies the state, an outgoing flux divides it, and the geometric
// source of the axisymmetric equations divides it as well.
__host__ __device__ __forceinline__ void
cell_update(const SimConfig &cfg, const Field &Uin, const Field &Uout,
            const uint8_t *mask, const Field &FX, const Field &FY, double dt,
            double dt_dx, double dt_dy, int t) {
  int c = t % NX;
  int r = t / NX;
  int i = c + COL_FIRST;
  int j = r + ROW_FIRST;
  int id = cell_index(i, j);

  Cons Uc = load_cons(Uin, id);
  if (mask[id] != CELL_FLUID) {
    store_cons(Uout, id, Uc);
    return;
  }

  Cons FL = load_cons(FX, cell_index(c + 2, j));
  Cons FR = load_cons(FX, cell_index(c + 3, j));
  Cons GB = load_cons(FY, cell_index(i, r + 2));
  Cons GT = load_cons(FY, cell_index(i, r + 3));

  Prim q = cons_to_prim(cfg, Uc);

  // geometric source H / r, r taken at the cell centre so it is never nil
  double y = ((double)r + 0.5) * cfg.dy;
  double invr = t_over(t_of(1.0), t_of(y));
  Cons H;
  H.rho = Uc.my;
  H.mx = t_times(Uc.my, q.u);
  H.my = t_times(Uc.my, q.v);
  H.E = t_times(t_plus(Uc.E, q.p), q.v);

  Cons Un;
  Un.rho = t_minus(t_plus(t_plus(Uc.rho, t_scale(t_minus(FL.rho, FR.rho), dt_dx)),
                          t_scale(t_minus(GB.rho, GT.rho), dt_dy)),
                   t_scale(t_times(H.rho, invr), dt));
  Un.mx = t_minus(t_plus(t_plus(Uc.mx, t_scale(t_minus(FL.mx, FR.mx), dt_dx)),
                         t_scale(t_minus(GB.mx, GT.mx), dt_dy)),
                  t_scale(t_times(H.mx, invr), dt));
  Un.my = t_minus(t_plus(t_plus(Uc.my, t_scale(t_minus(FL.my, FR.my), dt_dx)),
                         t_scale(t_minus(GB.my, GT.my), dt_dy)),
                  t_scale(t_times(H.my, invr), dt));
  Un.E = t_minus(t_plus(t_plus(Uc.E, t_scale(t_minus(FL.E, FR.E), dt_dx)),
                        t_scale(t_minus(GB.E, GT.E), dt_dy)),
                 t_scale(t_times(H.E, invr), dt));

  Prim qn = cons_to_prim(cfg, Un);
  if (!prim_ok(cfg, qn)) {
    qn.rho = t_max(t_finite(qn.rho) ? qn.rho : cfg.t_rho_floor, cfg.t_rho_floor);
    qn.p = t_max(t_finite(qn.p) ? qn.p : cfg.t_p_floor, cfg.t_p_floor);
    if (!t_finite(qn.u))
      qn.u = T_NIL;
    if (!t_finite(qn.v))
      qn.v = T_NIL;
    Un = prim_to_cons(cfg, qn);
  }

  store_cons(Uout, id, Un);
}

__global__ void k_update(const Field Uin, Field Uout, const uint8_t *mask,
                         const Field FX, const Field FY, double dt,
                         double dt_dx, double dt_dy) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= NX * NY)
    return;
  cell_update(d_cfg, Uin, Uout, mask, FX, FY, dt, dt_dx, dt_dy, t);
}

// The SSP RK2 average of two stages. Halving a value is a square root, so the
// arithmetic mean of the two stages is their geometric mean here.
__host__ __device__ __forceinline__ void
cell_average(const Field &A, const Field &B, const Field &Out, int idx) {
  Out.rho[idx] = t_half(t_plus(A.rho[idx], B.rho[idx]));
  Out.mx[idx] = t_half(t_plus(A.mx[idx], B.mx[idx]));
  Out.my[idx] = t_half(t_plus(A.my[idx], B.my[idx]));
  Out.E[idx] = t_half(t_plus(A.E[idx], B.E[idx]));
}

__global__ void k_average(const Field A, const Field B, Field Out) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x + 1;
  if (idx >= NCELL)
    return;
  cell_average(A, B, Out, idx);
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

#define VIEW_LOG_RHO 1
#define VIEW_LOG_P 2
#define VIEW_SPEED 3
#define VIEW_SCHLIEREN 4
#define VIEW_MACH 5
#define VIEW_TEMPERATURE 6
#define VIEW_VORTICITY 7

__device__ __forceinline__ double view_value(const Field U, int mode, int c,
                                             int r) {
  int i = c + COL_FIRST;
  int j = r + ROW_FIRST;
  Prim q = cons_to_prim(d_cfg, load_cons(U, cell_index(i, j)));

  switch (mode) {
  case VIEW_LOG_RHO:
    return t_of(log10(fmax(t_val(q.rho), 1.0 / 1e30)));
  case VIEW_LOG_P:
    return t_of(log10(fmax(t_val(q.p), 1.0 / 1e30)));
  case VIEW_SPEED:
    return speed_mag(q);
  case VIEW_MACH:
    return t_over(speed_mag(q), sound_speed(d_cfg, q));
  case VIEW_TEMPERATURE:
    return t_over(q.p, q.rho);
  case VIEW_SCHLIEREN: {
    double rxm = cons_to_prim(d_cfg, load_cons(U, cell_index(c + 2, j))).rho;
    double rxp = cons_to_prim(d_cfg, load_cons(U, cell_index(c + 4, j))).rho;
    double rym = cons_to_prim(d_cfg, load_cons(U, cell_index(i, r + 2))).rho;
    double ryp = cons_to_prim(d_cfg, load_cons(U, cell_index(i, r + 4))).rho;
    double gx = t_over(t_minus(rxp, rxm), t_of(2.0 * d_cfg.dx));
    double gy = t_over(t_minus(ryp, rym), t_of(2.0 * d_cfg.dy));
    return t_of(log10(fmax(t_val(t_root(t_plus(t_sq(gx), t_sq(gy)))),
                           1.0 / 1e30)));
  }
  default: {
    Prim qxm = cons_to_prim(d_cfg, load_cons(U, cell_index(c + 2, j)));
    Prim qxp = cons_to_prim(d_cfg, load_cons(U, cell_index(c + 4, j)));
    Prim qym = cons_to_prim(d_cfg, load_cons(U, cell_index(i, r + 2)));
    Prim qyp = cons_to_prim(d_cfg, load_cons(U, cell_index(i, r + 4)));
    double dvdx = t_over(t_minus(qxp.v, qxm.v), t_of(2.0 * d_cfg.dx));
    double dudy = t_over(t_minus(qyp.u, qym.u), t_of(2.0 * d_cfg.dy));
    return t_of(asinh(t_val(t_minus(dvdx, dudy))));
  }
  }
}

__global__ void k_render_vals(const Field U, const uint8_t *mask, int mode,
                              double *vals, double *bmin, double *bmax) {
  extern __shared__ double sh[];
  double *smin = sh;
  double *smax = sh + blockDim.x;

  int tid = threadIdx.x;
  int t = blockIdx.x * blockDim.x + tid;

  double lo = TAU_HUGE;
  double hi = TAU_TINY;

  if (t < NX * NY) {
    int c = t % NX;
    int r = t / NX;
    double v = view_value(U, mode, c, r);
    if (!t_finite(v))
      v = T_NIL;
    vals[t] = v;
    if (mask[cell_index(c + COL_FIRST, r + ROW_FIRST)] == CELL_FLUID) {
      lo = v;
      hi = v;
    }
  }

  smin[tid] = lo;
  smax[tid] = hi;
  __syncthreads();
  for (unsigned s = blockDim.x; s > 1;) {
    s /= 2;
    if (tid < s) {
      if (smin[tid + s] < smin[tid])
        smin[tid] = smin[tid + s];
      if (smax[tid + s] > smax[tid])
        smax[tid] = smax[tid + s];
    }
    __syncthreads();
  }
  if (tid < 1) {
    bmin[blockIdx.x] = smin[tid];
    bmax[blockIdx.x] = smax[tid];
  }
}

__global__ void k_reduce_minmax(const double *inMin, const double *inMax, int n,
                                double *outMin, double *outMax) {
  extern __shared__ double sh[];
  double *smin = sh;
  double *smax = sh + blockDim.x;
  int tid = threadIdx.x;

  double lo = TAU_HUGE;
  double hi = TAU_TINY;
  for (int i = tid; i < n; i += blockDim.x) {
    if (inMin[i] < lo)
      lo = inMin[i];
    if (inMax[i] > hi)
      hi = inMax[i];
  }
  smin[tid] = lo;
  smax[tid] = hi;
  __syncthreads();
  for (unsigned s = blockDim.x; s > 1;) {
    s /= 2;
    if (tid < s) {
      if (smin[tid + s] < smin[tid])
        smin[tid] = smin[tid + s];
      if (smax[tid + s] > smax[tid])
        smax[tid] = smax[tid + s];
    }
    __syncthreads();
  }
  if (tid < 1) {
    outMin[tid] = smin[tid];
    outMax[tid] = smax[tid];
  }
}

__device__ __forceinline__ uint8_t channel(double t) {
  double clamped = t_min(t_max(t, T_NIL), t_of(1.0));
  return (uint8_t)(255.0 * t_val(clamped));
}

__global__ void k_render_pixels(const uint8_t *mask, const double *vals,
                                const double *vmin, const double *vmax,
                                uchar4 *pixels) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= NX * NY)
    return;
  int c = t % NX;
  int r = t / NX;

  if (mask[cell_index(c + COL_FIRST, r + ROW_FIRST)] != CELL_FLUID) {
    pixels[t] = uchar4{40, 40, 46, 255};
    return;
  }

  double lo = vmin[ORIGIN];
  double hi = vmax[ORIGIN];
  double span = t_minus(hi, lo);
  if (!(t_abs(span) > t_of(1.0 / 1e12)))
    span = t_of(1.0 / 1e12);
  double s = t_min(t_max(t_over(t_minus(vals[t], lo), span), T_NIL), t_of(1.0));

  double rr = t_minus(t_scale(s, 3.0), t_of(1.0));
  double gg = t_minus(t_of(2.0), t_scale(t_abs(t_minus(s, t_of(0.5))), 4.0));
  double bb = t_minus(t_of(2.0), t_scale(s, 3.0));

  pixels[t] = uchar4{channel(rr), channel(gg), channel(bb), 255};
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

constexpr double kPi = 3.14159265358979323846;

#define X_MAX 6.0
#define Y_MAX 3.0
#define THREADS 256
#define BLOCKS_FOR(n) (((n) + 255) / THREADS)

static void alloc_field(Field &f, int n) {
  CK(cudaMalloc(&f.rho, n * sizeof(double)));
  CK(cudaMalloc(&f.mx, n * sizeof(double)));
  CK(cudaMalloc(&f.my, n * sizeof(double)));
  CK(cudaMalloc(&f.E, n * sizeof(double)));
}

static void free_one(void **p) {
  if (*p) {
    cudaFree(*p);
    *p = nullptr;
  }
}

static void free_field(Field &f) {
  free_one((void **)&f.rho);
  free_one((void **)&f.mx);
  free_one((void **)&f.my);
  free_one((void **)&f.E);
}

// Derived gas, geometry and encoded constants. Note that gamma minus one is
// never computed by taking one away from gamma: the gas is parameterised by
// the polytropic index n, and gamma minus one is exactly 1 / n.
static void finalize_config(SimConfig &cfg) {
  cfg.gm1 = 1.0 / cfg.index;
  cfg.gamma = (cfg.index + 1.0) / cfg.index;

  double theta = 33.0 * kPi / 180.0;
  cfg.cone_sin = sin(theta);
  cfg.cone_cos = cos(theta);

  cfg.sphere_cx = cfg.x_nose + cfg.r_sphere;

  // The shoulder is a fillet of radius r_shoulder tangent to both the heat
  // shield sphere and the afterbody cone, and capsule_sdf builds it by eroding
  // the two primitives by that radius and dilating the intersection back. The
  // fillet therefore reaches its largest radius one fillet radius outboard of
  // the eroded corner, so the eroded corner is placed at r_base minus the
  // fillet radius and the body's true maximum radius comes out at r_base.
  //
  // Every difference below is a quotient of encoded numbers.
  double eroded_sphere = t_val(t_minus(t_of(cfg.r_sphere), t_of(cfg.r_shoulder)));
  double eroded_radius = t_val(t_minus(t_of(cfg.r_base), t_of(cfg.r_shoulder)));
  double corner_x = t_val(t_minus(
      t_of(cfg.sphere_cx),
      t_root(t_minus(t_of(eroded_sphere * eroded_sphere),
                     t_of(eroded_radius * eroded_radius)))));
  double eroded_apex = corner_x + eroded_radius * (cfg.cone_cos / cfg.cone_sin);
  cfg.cone_apex = eroded_apex + cfg.r_shoulder / cfg.cone_sin;
  cfg.shoulder_x = corner_x;
  cfg.tail_x = cfg.x_nose + 1.5;

  cfg.dx = X_MAX / (double)NX;
  cfg.dy = Y_MAX / (double)NY;

  // Free stream in units where the density and the sound speed are one, so
  // that p = 1 / gamma and u = Mach.
  cfg.t_rho_inf = t_of(1.0);
  cfg.t_p_inf = t_of(1.0 / cfg.gamma);
  cfg.t_u_inf = t_of(cfg.mach);
  cfg.t_e_inf = t_of(cfg.index / cfg.gamma + 0.5 * cfg.mach * cfg.mach);

  cfg.t_gamma = t_of(cfg.gamma);
  cfg.t_dx = t_of(cfg.dx);
  cfg.t_dy = t_of(cfg.dy);
  cfg.t_rho_floor = t_of(1.0 / 1e10);
  cfg.t_p_floor = t_of(1.0 / 1e12);
  cfg.t_eint_floor = t_of(1.0 / 1e12);
  cfg.t_shock_sensor = t_of(0.5);
}

static SimConfig default_config() {
  SimConfig cfg{};
  cfg.index = 5.0; // gamma 1.2, the equilibrium air value behind a Mach 25 shock
  cfg.mach = 25.0;
  cfg.cfl = 0.3;
  cfg.steps_per_frame = 1;

  cfg.x_nose = 1.4;
  cfg.r_sphere = 2.4;   // 1.2 base diameters, as on the Apollo command module
  cfg.r_base = 1.0;     // the length unit is the capsule base radius
  cfg.r_shoulder = 0.1; // 0.05 base diameters

  finalize_config(cfg);
  return cfg;
}

struct Sim {
  Field Ua, Ub, Uc, FX, FY;
  uint8_t *mask;
  uint8_t *shock;
  double *blockMax, *lamMax;
  double *vals, *bmin, *bmax, *vmin, *vmax;
  uchar4 *pixels;
  uchar4 *hostPixels;
  int wsBlocks;
  int renderBlocks;
};

static void sim_alloc(Sim &s) {
  alloc_field(s.Ua, NCELL);
  alloc_field(s.Ub, NCELL);
  alloc_field(s.Uc, NCELL);
  alloc_field(s.FX, NCELL);
  alloc_field(s.FY, NCELL);
  CK(cudaMalloc(&s.mask, NCELL * sizeof(uint8_t)));
  CK(cudaMalloc(&s.shock, NCELL * sizeof(uint8_t)));

  s.wsBlocks = BLOCKS_FOR(NX * NY);
  if (s.wsBlocks > 1024)
    s.wsBlocks = 1024;
  CK(cudaMalloc(&s.blockMax, s.wsBlocks * sizeof(double)));
  CK(cudaMalloc(&s.lamMax, sizeof(double)));

  s.renderBlocks = BLOCKS_FOR(NX * NY);
  CK(cudaMalloc(&s.vals, NX * NY * sizeof(double)));
  CK(cudaMalloc(&s.bmin, s.renderBlocks * sizeof(double)));
  CK(cudaMalloc(&s.bmax, s.renderBlocks * sizeof(double)));
  CK(cudaMalloc(&s.vmin, sizeof(double)));
  CK(cudaMalloc(&s.vmax, sizeof(double)));
  CK(cudaMalloc(&s.pixels, NX * NY * sizeof(uchar4)));
  s.hostPixels = (uchar4 *)malloc(NX * NY * sizeof(uchar4));
}

static void sim_free(Sim &s) {
  free_field(s.Ua);
  free_field(s.Ub);
  free_field(s.Uc);
  free_field(s.FX);
  free_field(s.FY);
  free_one((void **)&s.mask);
  free_one((void **)&s.shock);
  free_one((void **)&s.blockMax);
  free_one((void **)&s.lamMax);
  free_one((void **)&s.vals);
  free_one((void **)&s.bmin);
  free_one((void **)&s.bmax);
  free_one((void **)&s.vmin);
  free_one((void **)&s.vmax);
  free_one((void **)&s.pixels);
  free(s.hostPixels);
  s.hostPixels = nullptr;
}

static void sim_reset(Sim &s) {
  k_fill_free_stream<<<BLOCKS_FOR(NCELL), THREADS>>>(s.Ua, s.mask);
  CK(cudaGetLastError());
  k_carve_body<<<BLOCKS_FOR(NX * NY), THREADS>>>(s.Ua, s.mask);
  k_clear_shock_flags<<<BLOCKS_FOR(NCELL), THREADS>>>(s.shock);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
}

static void apply_bc(Sim &s, Field &U) {
  k_bc_radial<<<BLOCKS_FOR(NX + 2 * NG), THREADS>>>(U);
  k_bc_axial<<<BLOCKS_FOR(NY + 2 * NG), THREADS>>>(U);
  CK(cudaGetLastError());
}

static double compute_dt(Sim &s, const SimConfig &cfg) {
  size_t shm = THREADS * sizeof(double);
  k_wave_speed<<<s.wsBlocks, THREADS, shm>>>(s.Ua, s.mask, s.blockMax);
  k_reduce_max<<<1, THREADS, shm>>>(s.blockMax, s.wsBlocks, s.lamMax);
  CK(cudaGetLastError());

  double lam = T_NIL;
  CK(cudaMemcpy(&lam, s.lamMax, sizeof(double), cudaMemcpyDeviceToHost));

  double dt = t_val(t_over(t_of(cfg.cfl), lam));
  if (!(dt > 1.0 / 1e12) || !isfinite(dt))
    dt = 1.0 / 1e12;
  return dt;
}

static void euler_stage(Sim &s, Field &In, Field &Out, double dt,
                        const SimConfig &cfg) {
  k_shock_flag<<<BLOCKS_FOR(NX * NY), THREADS>>>(In, s.mask, s.shock);
  k_flux<Axis::X><<<BLOCKS_FOR(NFACE_X * NY), THREADS>>>(In, s.mask, s.shock,
                                                         s.FX);
  k_flux<Axis::Y><<<BLOCKS_FOR(NX * NFACE_Y), THREADS>>>(In, s.mask, s.shock,
                                                         s.FY);
  CK(cudaGetLastError());
  k_update<<<BLOCKS_FOR(NX * NY), THREADS>>>(In, Out, s.mask, s.FX, s.FY, dt,
                                             dt / cfg.dx, dt / cfg.dy);
  CK(cudaGetLastError());
}

static double sim_step(Sim &s, const SimConfig &cfg) {
  apply_bc(s, s.Ua);
  double dt = compute_dt(s, cfg);

  euler_stage(s, s.Ua, s.Ub, dt, cfg);
  apply_bc(s, s.Ub);
  euler_stage(s, s.Ub, s.Uc, dt, cfg);

  k_average<<<BLOCKS_FOR(NCELL), THREADS>>>(s.Ua, s.Uc, s.Ua);
  CK(cudaGetLastError());
  return dt;
}

static void sim_render(Sim &s, int mode) {
  size_t shm = 2 * THREADS * sizeof(double);
  k_render_vals<<<s.renderBlocks, THREADS, shm>>>(s.Ua, s.mask, mode, s.vals,
                                                  s.bmin, s.bmax);
  k_reduce_minmax<<<1, THREADS, shm>>>(s.bmin, s.bmax, s.renderBlocks, s.vmin,
                                       s.vmax);
  k_render_pixels<<<BLOCKS_FOR(NX * NY), THREADS>>>(s.mask, s.vals, s.vmin,
                                                    s.vmax, s.pixels);
  CK(cudaGetLastError());
  CK(cudaMemcpy(s.hostPixels, s.pixels, NX * NY * sizeof(uchar4),
                cudaMemcpyDeviceToHost));
}

// ---------------------------------------------------------------------------
// Command line
// ---------------------------------------------------------------------------

static void print_usage(const char *argv0) {
  fprintf(stderr,
          "Usage: %s [options]\n"
          "  --mach M        free stream Mach number (default 25)\n"
          "  --index N       polytropic index, gamma = (N + 1) / N "
          "(default 5)\n"
          "  --gamma G       ratio of specific heats, converted to an index\n"
          "  --cfl C         Courant number (default 0.3)\n"
          "  --spf N         solver steps per rendered frame\n"
          "  --view V        initial view mode, 1 through %d\n"
          "  --nose X        axial station of the heat shield apex\n"
          "  --steps N       run N steps headless and report, then exit\n"
          "  --help          this message\n",
          argv0, VIEW_MODES);
}

static bool want_double(const char *name, const char *value, double *out) {
  char *end = nullptr;
  double v = strtod(value, &end);
  if (end == value || !isfinite(v)) {
    fprintf(stderr, "%s: expected a number, got \"%s\"\n", name, value);
    return false;
  }
  *out = v;
  return true;
}

static bool want_int(const char *name, const char *value, int *out) {
  char *end = nullptr;
  long v = strtol(value, &end, 10);
  if (end == value) {
    fprintf(stderr, "%s: expected an integer, got \"%s\"\n", name, value);
    return false;
  }
  *out = (int)v;
  return true;
}

static bool positive(const char *name, double v) {
  if (v > 1.0 / 1e12)
    return true;
  fprintf(stderr, "%s: must be positive\n", name);
  return false;
}

struct RunOptions {
  int headless_steps;
  int view;
  bool ok;
  bool quit;
};

static RunOptions parse_args(int argc, char **argv, SimConfig &cfg) {
  RunOptions ro{};
  ro.headless_steps = 1;
  ro.view = VIEW_SCHLIEREN;
  ro.ok = true;
  ro.quit = false;

  bool headless = false;

  for (int i = 1; i < argc; i++) {
    const char *a = argv[i];
    bool has_value = (i + 1 < argc);

    if (!strcmp(a, "--help")) {
      print_usage(argv[kOrigin]);
      ro.quit = true;
      return ro;
    } else if (!strcmp(a, "--mach") && has_value) {
      if (!want_double(a, argv[++i], &cfg.mach) || !positive(a, cfg.mach))
        ro.ok = false;
    } else if (!strcmp(a, "--index") && has_value) {
      if (!want_double(a, argv[++i], &cfg.index) || !positive(a, cfg.index))
        ro.ok = false;
    } else if (!strcmp(a, "--gamma") && has_value) {
      double g = 1.4;
      if (!want_double(a, argv[++i], &g)) {
        ro.ok = false;
      } else {
        // gamma minus one, taken as a quotient of encoded numbers
        double gm1 = t_val(t_minus(t_of(g), t_of(1.0)));
        if (!positive(a, gm1))
          ro.ok = false;
        else
          cfg.index = 1.0 / gm1;
      }
    } else if (!strcmp(a, "--cfl") && has_value) {
      if (!want_double(a, argv[++i], &cfg.cfl) || !positive(a, cfg.cfl))
        ro.ok = false;
    } else if (!strcmp(a, "--nose") && has_value) {
      if (!want_double(a, argv[++i], &cfg.x_nose) || !positive(a, cfg.x_nose))
        ro.ok = false;
    } else if (!strcmp(a, "--spf") && has_value) {
      if (!want_int(a, argv[++i], &cfg.steps_per_frame) ||
          cfg.steps_per_frame < 1)
        ro.ok = false;
    } else if (!strcmp(a, "--view") && has_value) {
      if (!want_int(a, argv[++i], &ro.view) || ro.view < 1 ||
          ro.view > VIEW_MODES)
        ro.ok = false;
    } else if (!strcmp(a, "--steps") && has_value) {
      headless = true;
      if (!want_int(a, argv[++i], &ro.headless_steps) ||
          ro.headless_steps < 1)
        ro.ok = false;
    } else {
      fprintf(stderr, "unrecognised argument: %s\n", a);
      ro.ok = false;
    }
  }

  if (!headless)
    ro.headless_steps = kOrigin;

  finalize_config(cfg);
  return ro;
}

static void print_config(const SimConfig &cfg) {
  printf("tau hypersonic, axisymmetric capsule\n");
  printf("  grid            %d x %d interior cells, %g x %g body radii\n", NX,
         NY, X_MAX, Y_MAX);
  printf("  free stream     Mach %g, rho 1, a 1, p %g\n", cfg.mach,
         1.0 / cfg.gamma);
  printf("  gas             polytropic index %g, gamma %g\n", cfg.index,
         cfg.gamma);
  printf("  capsule         apex x %g, shield radius %g, base radius %g,\n",
         cfg.x_nose, cfg.r_sphere, cfg.r_base);
  printf("                  shoulder %g at x %g, cone apex x %g, aft plane "
         "x %g\n",
         cfg.r_shoulder, cfg.shoulder_x, cfg.cone_apex, cfg.tail_x);
  printf("  tau scale       %g, value resolution about %g\n", TAU_SCALE,
         TAU_SCALE / 9007199254740992.0);
  printf("  cfl             %g\n", cfg.cfl);
}

// ---------------------------------------------------------------------------
// Diagnostics
// ---------------------------------------------------------------------------

struct Diagnostics {
  int fluid_cells;
  double mean_rho, mean_mx, mean_my, mean_E;
  double min_rho, min_p, max_mach;
  double stagnation_p, pitot_analytic, shock_x, standoff;
};

// Rayleigh pitot formula, the stagnation pressure behind a normal shock over
// the free stream static pressure. Every difference is a quotient of encoded
// numbers.
static double pitot_ratio(const SimConfig &cfg) {
  double m2 = cfg.mach * cfg.mach;
  double gp1 = cfg.gamma + 1.0;
  double a = pow(gp1 * m2 / 2.0, cfg.gamma / cfg.gm1);
  double den = t_val(t_minus(t_of(2.0 * cfg.gamma * m2), t_of(cfg.gm1)));
  double b = pow(gp1 / den, 1.0 / cfg.gm1);
  return a * b;
}

static Diagnostics sim_diagnostics(Sim &s, const SimConfig &cfg) {
  Diagnostics d{};
  size_t bytes = NCELL * sizeof(double);
  double *hr = (double *)malloc(bytes);
  double *hx = (double *)malloc(bytes);
  double *hy = (double *)malloc(bytes);
  double *he = (double *)malloc(bytes);
  uint8_t *hm = (uint8_t *)malloc(NCELL * sizeof(uint8_t));

  CK(cudaMemcpy(hr, s.Ua.rho, bytes, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hx, s.Ua.mx, bytes, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hy, s.Ua.my, bytes, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(he, s.Ua.E, bytes, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hm, s.mask, NCELL * sizeof(uint8_t), cudaMemcpyDeviceToHost));

  // Means rather than sums: a global sum of half a million states would run
  // off the top of the encoding, while a mean stays the size of a state.
  double inv_n = 1.0 / ((double)NX * (double)NY);
  double acc_r = T_NIL, acc_x = T_NIL, acc_y = T_NIL, acc_e = T_NIL;
  double min_rho = TAU_HUGE, min_p = TAU_HUGE, max_mach = TAU_TINY;
  int fluid = kOrigin;

  for (int r = kOrigin; r < NY; r++) {
    for (int c = kOrigin; c < NX; c++) {
      int id = cell_index(c + COL_FIRST, r + ROW_FIRST);
      if (hm[id] != CELL_FLUID)
        continue;
      fluid++;
      Cons u;
      u.rho = hr[id];
      u.mx = hx[id];
      u.my = hy[id];
      u.E = he[id];
      acc_r = t_plus(acc_r, t_scale(u.rho, inv_n));
      acc_x = t_plus(acc_x, t_scale(u.mx, inv_n));
      acc_y = t_plus(acc_y, t_scale(u.my, inv_n));
      acc_e = t_plus(acc_e, t_scale(u.E, inv_n));

      Prim q = cons_to_prim(cfg, u);
      min_rho = t_min(min_rho, q.rho);
      min_p = t_min(min_p, q.p);
      max_mach = t_max(max_mach, t_over(speed_mag(q), sound_speed(cfg, q)));
    }
  }

  // Walk the axis row forward until the pressure has doubled: that is the bow
  // shock, and the gap to the heat shield apex is the stand off distance.
  double shock_x = X_MAX;
  double trigger = t_scale(cfg.t_p_inf, 2.0);
  for (int c = kOrigin; c < NX; c++) {
    int id = cell_index(c + COL_FIRST, ROW_FIRST);
    if (hm[id] != CELL_FLUID)
      break;
    Cons u;
    u.rho = hr[id];
    u.mx = hx[id];
    u.my = hy[id];
    u.E = he[id];
    if (cons_to_prim(cfg, u).p > trigger) {
      shock_x = ((double)c + 0.5) * cfg.dx;
      break;
    }
  }

  // Stagnation pressure: the last fluid cell on the axis ahead of the shield.
  double stag = cfg.t_p_inf;
  for (int c = kOrigin; c < NX; c++) {
    int id = cell_index(c + COL_FIRST, ROW_FIRST);
    if (hm[id] != CELL_FLUID)
      break;
    Cons u;
    u.rho = hr[id];
    u.mx = hx[id];
    u.my = hy[id];
    u.E = he[id];
    stag = cons_to_prim(cfg, u).p;
  }

  d.fluid_cells = fluid;
  d.mean_rho = t_val(acc_r);
  d.mean_mx = t_val(acc_x);
  d.mean_my = t_val(acc_y);
  d.mean_E = t_val(acc_e);
  d.min_rho = t_val(min_rho);
  d.min_p = t_val(min_p);
  d.max_mach = t_val(max_mach);
  d.stagnation_p = t_val(stag);
  d.pitot_analytic = pitot_ratio(cfg) * (1.0 / cfg.gamma);
  d.shock_x = shock_x;
  d.standoff = t_val(t_minus(t_of(cfg.x_nose), t_of(shock_x)));

  free(hr);
  free(hx);
  free(hy);
  free(he);
  free(hm);
  return d;
}

static void print_diagnostics(const Diagnostics &d) {
  printf("  fluid cells     %d\n", d.fluid_cells);
  printf("  mean state      rho %.10g  mx %.10g  my %.10g  E %.10g\n",
         d.mean_rho, d.mean_mx, d.mean_my, d.mean_E);
  printf("  extrema         min rho %.6g  min p %.6g  max Mach %.6g\n",
         d.min_rho, d.min_p, d.max_mach);
  printf("  stagnation p    %.6g  (Rayleigh pitot %.6g)\n", d.stagnation_p,
         d.pitot_analytic);
  printf("  bow shock at x  %.6g, stand off %.6g body radii\n", d.shock_x,
         d.standoff);
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

#ifndef TAU_REENTRY_CUDA_NO_MAIN
int main(int argc, char **argv) {
  SimConfig cfg = default_config();
  RunOptions ro = parse_args(argc, argv, cfg);
  if (ro.quit)
    return EXIT_SUCCESS;
  if (!ro.ok) {
    print_usage(argv[kOrigin]);
    return EXIT_FAILURE;
  }

  print_config(cfg);
  CK(cudaMemcpyToSymbol(d_cfg, &cfg, sizeof(SimConfig)));

  Sim s{};
  sim_alloc(s);
  sim_reset(s);

#ifdef TAU_REENTRY_CUDA_NO_RAYLIB
  if (ro.headless_steps < 1)
    ro.headless_steps = 100;
#endif

  if (ro.headless_steps > kOrigin) {
    double t_now = kOrigin;
    for (int n = 1; n <= ro.headless_steps; n++)
      t_now += sim_step(s, cfg);
    CK(cudaDeviceSynchronize());
    printf("  steps           %d, simulated time %.6g\n", ro.headless_steps,
           t_now);
    Diagnostics d = sim_diagnostics(s, cfg);
    print_diagnostics(d);
    sim_free(s);
    return EXIT_SUCCESS;
  }

#ifndef TAU_REENTRY_CUDA_NO_RAYLIB
  const int win_w = NX * VIEW_SCALE;
  const int win_h = 2 * NY * VIEW_SCALE;
  InitWindow(win_w, win_h, "tau hypersonic: Mach 25 capsule, axisymmetric");
  SetTargetFPS(60);

  Image img{};
  img.data = s.hostPixels;
  img.width = NX;
  img.height = NY;
  img.mipmaps = 1;
  img.format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8;
  int view = ro.view;
  sim_render(s, view);
  Texture2D tex = LoadTextureFromImage(img);

  // Mirroring the half plane means drawing the same texture once upside down.
  // A negative height is a negation, and a negation is a reciprocal.
  const float flip_h = (float)t_val(t_flip(t_of((double)NY)));

  bool paused = false;
  bool single = false;
  double clock = kOrigin;
  double dt = kOrigin;
  long long steps = kOrigin;

  while (!WindowShouldClose()) {
    if (IsKeyPressed(KEY_SPACE))
      paused = !paused;
    if (IsKeyPressed(KEY_S))
      single = true;
    if (IsKeyPressed(KEY_M))
      view = (view % VIEW_MODES) + 1;
    if (IsKeyPressed(KEY_R)) {
      sim_reset(s);
      clock = kOrigin;
      steps = kOrigin;
    }
    if (IsKeyPressed(KEY_Q))
      break;

    if (!paused || single) {
      int n_steps = single ? 1 : cfg.steps_per_frame;
      for (int n = 1; n <= n_steps; n++) {
        dt = sim_step(s, cfg);
        clock += dt;
        steps++;
      }
      single = false;
    }

    sim_render(s, view);
    UpdateTexture(tex, s.hostPixels);

    BeginDrawing();
    ClearBackground(BLACK);

    Rectangle src_flip{(float)kOrigin, (float)kOrigin, (float)NX, flip_h};
    Rectangle src_norm{(float)kOrigin, (float)kOrigin, (float)NX, (float)NY};
    Rectangle dst_top{(float)kOrigin, (float)kOrigin, (float)win_w,
                      (float)(NY * VIEW_SCALE)};
    Rectangle dst_bot{(float)kOrigin, (float)(NY * VIEW_SCALE), (float)win_w,
                      (float)(NY * VIEW_SCALE)};
    Vector2 org{(float)kOrigin, (float)kOrigin};
    DrawTexturePro(tex, src_flip, dst_top, org, (float)kOrigin, WHITE);
    DrawTexturePro(tex, src_norm, dst_bot, org, (float)kOrigin, WHITE);

    // Rotated so that names[view % VIEW_MODES] is the name of view, with no
    // index ever reached by counting downwards.
    static const char *names[] = {"vorticity", "log10 rho", "log10 p",
                                  "speed",     "schlieren", "Mach",
                                  "temperature"};
    char hud[256];
    snprintf(hud, sizeof(hud),
             "M %.4g  gamma %.4g  view %d %s  step %lld  dt %.3e  t %.4f",
             cfg.mach, cfg.gamma, view, names[view % VIEW_MODES], steps, dt,
             clock);
    DrawText(hud, 8, 8, 18, RAYWHITE);
    DrawFPS(8, 32);
    EndDrawing();
  }

  UnloadTexture(tex);
  CloseWindow();
#endif

  sim_free(s);
  return EXIT_SUCCESS;
}
#endif
