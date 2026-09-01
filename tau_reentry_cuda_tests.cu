// tau_reentry_cuda_tests.cu
//
//   nvcc -O2 -std=c++17 -o tau_reentry_cuda_tests tau_reentry_cuda_tests.cu
//
// Unit and regression tests for tau_reentry_cuda.cu.
//
// The solver under test never subtracts and never writes a zero. This harness
// deliberately does both: ordinary arithmetic is the independent oracle the
// tau algebra is checked against, so the two sides of every assertion are
// computed by genuinely different means.
//
// The host tests need no GPU. The device tests and the regression baseline do;
// when no device is present they are skipped and the harness still reports on
// everything it could run.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define TAU_REENTRY_CUDA_NO_RAYLIB
#define TAU_REENTRY_CUDA_NO_MAIN
#include "tau_reentry_cuda.cu"

// ---------------------------------------------------------------------------
// Harness
// ---------------------------------------------------------------------------

struct Stats {
  int passed;
  int failed;
};

static Stats g_stats{0, 0};

static void check(bool cond, const char *msg, const char *file, int line) {
  if (cond) {
    g_stats.passed++;
  } else {
    g_stats.failed++;
    fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, file, line);
  }
}

static void check_near(double a, double b, double tol, const char *msg,
                       const char *file, int line) {
  if (std::isfinite(a) && std::isfinite(b) && std::fabs(a - b) <= tol) {
    g_stats.passed++;
  } else {
    g_stats.failed++;
    fprintf(stderr, "FAIL: %s (%s:%d): %.17g vs %.17g, difference %.3g\n", msg,
            file, line, a, b, std::fabs(a - b));
  }
}

#define CHECK(cond, msg) check((cond), (msg), __FILE__, __LINE__)
#define CHECK_NEAR(a, b, tol, msg)                                             \
  check_near((a), (b), (tol), (msg), __FILE__, __LINE__)

// The absolute resolution of the encoding, in value units.
static const double kTauEps = TAU_SCALE / 9007199254740992.0;

// ---------------------------------------------------------------------------
// The tau algebra
// ---------------------------------------------------------------------------

static void test_algebra() {
  const double vals[] = {1.0 / 1e3, 0.25, 1.0, 3.5, 25.0, 316.0, 700.0};

  for (double v : vals) {
    CHECK_NEAR(t_val(t_of(v)), v, 4 * kTauEps, "encode then decode is identity");
    CHECK_NEAR(t_val(t_flip(t_of(v))), -v, 4 * kTauEps,
               "negation is a reciprocal");
    CHECK_NEAR(t_val(t_abs(t_flip(t_of(v)))), v, 4 * kTauEps,
               "magnitude of a negated value");
    CHECK_NEAR(t_val(t_twice(t_of(v))), 2 * v, 8 * kTauEps,
               "doubling is squaring");
    CHECK_NEAR(t_val(t_half(t_of(v))), v / 2, 4 * kTauEps,
               "halving is a square root");
    CHECK(t_of(v) > 0.0, "every encoded value is strictly positive");
  }

  for (double a : vals) {
    for (double b : vals) {
      CHECK_NEAR(t_val(t_plus(t_of(a), t_of(b))), a + b, 8 * kTauEps,
                 "a sum is a product");
      CHECK_NEAR(t_val(t_minus(t_of(a), t_of(b))), a - b, 8 * kTauEps,
                 "a difference is a quotient");
      double tol_mul = (std::fabs(a) + std::fabs(b)) * 4 * kTauEps + 1e-9;
      CHECK_NEAR(t_val(t_times(t_of(a), t_of(b))), a * b, tol_mul,
                 "a product rides the isomorphism");
      double tol_div =
          (4 * kTauEps + std::fabs(a / b) * 4 * kTauEps) / std::fabs(b) + 1e-9;
      CHECK_NEAR(t_val(t_over(t_of(a), t_of(b))), a / b, tol_div,
                 "a quotient rides the isomorphism");
    }
  }

  CHECK_NEAR(t_val(T_NIL), 0.0, 1e-15, "T_NIL decodes to the additive identity");
  CHECK_NEAR(t_val(t_plus(t_of(7.0), T_NIL)), 7.0, 4 * kTauEps,
             "T_NIL is neutral for sums");
  CHECK_NEAR(t_val(t_minus(t_of(7.0), t_of(7.0))), 0.0, 1e-15,
             "a value divided by itself is nil");
  CHECK_NEAR(t_val(t_scale(t_of(3.0), 4.0)), 12.0, 1e-10,
             "scaling by a constant is exponentiation");
  CHECK_NEAR(t_val(t_root(t_of(9.0))), 3.0, 1e-10, "square root of a value");
  CHECK_NEAR(t_val(t_sq(t_of(9.0))), 81.0, 1e-8, "square of a value");

  CHECK(t_of(2.0) > t_of(1.0) && t_of(1.0) > T_NIL && T_NIL > t_of(-1.0) &&
            t_of(-1.0) > t_of(-2.0),
        "the encoding is order preserving");
  CHECK(t_min(t_of(3.0), t_of(-4.0)) == t_of(-4.0), "min compares values");
  CHECK(t_max(t_of(3.0), t_of(-4.0)) == t_of(3.0), "max compares values");
  CHECK(t_finite(t_of(1.0)) && !t_finite(0.0) && !t_finite(INFINITY),
        "finiteness rejects the degenerate encodings");
}

static void test_limiter() {
  CHECK_NEAR(t_val(t_minmod(t_of(1.0), t_of(2.0))), 1.0, 8 * kTauEps,
             "minmod takes the smaller of two rises");
  CHECK_NEAR(t_val(t_minmod(t_of(-2.0), t_of(-1.0))), -1.0, 8 * kTauEps,
             "minmod takes the smaller of two falls");
  CHECK_NEAR(t_val(t_minmod(t_of(-1.0), t_of(2.0))), 0.0, 1e-15,
             "minmod is nil across an extremum");
  CHECK_NEAR(t_val(t_mc(t_of(1.0), t_of(1.0))), 1.0, 8 * kTauEps,
             "the mc limiter passes a smooth ramp");
  CHECK_NEAR(t_val(t_mc(t_of(1.0), t_of(-1.0))), 0.0, 1e-15,
             "the mc limiter clips an extremum");
  CHECK_NEAR(t_val(t_mc(t_of(1.0), t_of(100.0))), 2.0, 1e-9,
             "the mc limiter is bounded by twice the smaller slope");

  Prim q = nil_prim();
  Prim s = prim_slope(q, q, q);
  CHECK(s.rho == T_NIL && s.u == T_NIL && s.v == T_NIL && s.p == T_NIL,
        "a uniform state has a nil slope");
  Prim hi = prim_face_hi(q, s);
  Prim lo = prim_face_lo(q, s);
  CHECK(hi.rho == q.rho && lo.rho == q.rho,
        "a nil slope leaves both faces at the cell value");
}

// ---------------------------------------------------------------------------
// Thermodynamics and fluxes
// ---------------------------------------------------------------------------

static void test_thermo(const SimConfig &cfg) {
  CHECK_NEAR(cfg.gamma, 1.2, 1e-15, "the default gas is gamma 1.2");
  CHECK_NEAR(cfg.gm1, 0.2, 1e-15, "gamma minus one is exactly one over n");
  CHECK_NEAR(cfg.gamma - 1.0, cfg.gm1, 1e-15,
             "the index parameterisation agrees with plain subtraction");

  Prim inf = free_stream(cfg);
  CHECK_NEAR(t_val(inf.rho), 1.0, 4 * kTauEps, "free stream density is one");
  CHECK_NEAR(t_val(inf.p), 1.0 / cfg.gamma, 4 * kTauEps,
             "free stream pressure makes the sound speed one");
  CHECK_NEAR(t_val(inf.u), cfg.mach, 4 * kTauEps,
             "free stream speed is the Mach number");
  CHECK_NEAR(t_val(inf.v), 0.0, 1e-15, "the free stream is axial");
  CHECK_NEAR(t_val(sound_speed(cfg, inf)), 1.0, 1e-10,
             "free stream sound speed is one");

  Cons c = prim_to_cons(cfg, inf);
  double e_exact = (1.0 / cfg.gamma) / (cfg.gamma - 1.0) +
                   0.5 * 1.0 * cfg.mach * cfg.mach;
  CHECK_NEAR(t_val(c.E), e_exact, 1e-9, "free stream total energy");
  CHECK_NEAR(t_val(c.mx), cfg.mach, 1e-10, "free stream axial momentum");

  Prim back = cons_to_prim(cfg, c);
  CHECK_NEAR(t_val(back.rho), 1.0, 1e-10, "conserved to primitive keeps rho");
  CHECK_NEAR(t_val(back.u), cfg.mach, 1e-9, "conserved to primitive keeps u");
  CHECK_NEAR(t_val(back.p), 1.0 / cfg.gamma, 1e-8,
             "conserved to primitive keeps p");

  // The floors keep a badly damaged state inside the physical set.
  Cons bad;
  bad.rho = t_of(-5.0);
  bad.mx = t_of(3.0);
  bad.my = t_of(4.0);
  bad.E = t_of(-1.0);
  Prim fixed = cons_to_prim(cfg, bad);
  CHECK(fixed.rho >= cfg.t_rho_floor, "density is floored, never negative");
  CHECK(fixed.p >= cfg.t_p_floor, "pressure is floored, never negative");
}

static void test_flux(const SimConfig &cfg) {
  Prim q;
  q.rho = t_of(0.7);
  q.u = t_of(3.0);
  q.v = t_of(1.5);
  q.p = t_of(2.0);
  Cons U = prim_to_cons(cfg, q);

  Cons FX = flux_axis<Axis::X>(cfg, q, U);
  double rho = 0.7, u = 3.0, v = 1.5, p = 2.0;
  double E = p / (cfg.gamma - 1.0) + 0.5 * rho * (u * u + v * v);
  CHECK_NEAR(t_val(FX.rho), rho * u, 1e-10, "axial mass flux");
  CHECK_NEAR(t_val(FX.mx), rho * u * u + p, 1e-9, "axial momentum flux");
  CHECK_NEAR(t_val(FX.my), rho * u * v, 1e-9, "axial transport of radial momentum");
  CHECK_NEAR(t_val(FX.E), (E + p) * u, 1e-7, "axial energy flux");

  Cons FY = flux_axis<Axis::Y>(cfg, q, U);
  CHECK_NEAR(t_val(FY.rho), rho * v, 1e-10, "radial mass flux");
  CHECK_NEAR(t_val(FY.my), rho * v * v + p, 1e-9, "radial momentum flux");

  // Consistency: a Riemann solver handed one state must return its own flux.
  Cons RX = riemann_axis<Axis::X>(cfg, q, q);
  Cons RY = riemann_axis<Axis::Y>(cfg, q, q);
  Cons HX = hlle_axis<Axis::X>(cfg, q, q);
  Cons CX = hllc_axis<Axis::X>(cfg, q, q);
  CHECK_NEAR(t_val(RX.rho), t_val(FX.rho), 1e-7, "hllc x consistency, mass");
  CHECK_NEAR(t_val(RX.mx), t_val(FX.mx), 1e-6, "hllc x consistency, momentum");
  CHECK_NEAR(t_val(RX.E), t_val(FX.E), 1e-4, "hllc x consistency, energy");
  CHECK_NEAR(t_val(RY.my), t_val(FY.my), 1e-6, "hllc y consistency, momentum");
  CHECK_NEAR(t_val(HX.mx), t_val(FX.mx), 1e-6, "hlle x consistency, momentum");
  CHECK_NEAR(t_val(CX.E), t_val(FX.E), 1e-4, "hllc star state consistency");

  // Supersonic upwinding: everything moves right, so the flux is the left one.
  Prim fast = free_stream(cfg);
  Prim fast2 = fast;
  fast2.rho = t_of(1.1);
  Cons up = hllc_axis<Axis::X>(cfg, fast, fast2);
  Cons FL = flux_axis<Axis::X>(cfg, fast, prim_to_cons(cfg, fast));
  CHECK_NEAR(t_val(up.rho), t_val(FL.rho), 1e-9,
             "a fully supersonic face is pure upwind");
}

static void test_rankine_hugoniot(const SimConfig &cfg) {
  const double g = cfg.gamma, m = cfg.mach, m2 = m * m;
  const double rr = ((g + 1.0) * m2) / ((g - 1.0) * m2 + 2.0);
  const double pr = (2.0 * g * m2 - (g - 1.0)) / (g + 1.0);

  Prim L = free_stream(cfg);
  Prim R;
  R.rho = t_of(rr);
  R.u = t_of(m / rr);
  R.v = T_NIL;
  R.p = t_of(pr / g);

  Cons FL = flux_axis<Axis::X>(cfg, L, prim_to_cons(cfg, L));
  Cons FR = flux_axis<Axis::X>(cfg, R, prim_to_cons(cfg, R));
  CHECK_NEAR(t_val(FL.rho), t_val(FR.rho), 1e-8, "mass flux across the shock");
  CHECK_NEAR(t_val(FL.mx), t_val(FR.mx), 1e-6, "momentum flux across the shock");
  CHECK_NEAR(t_val(FL.E), t_val(FR.E), 1e-4, "energy flux across the shock");

  // Such a face must be recognised as a shock and handed to HLLE.
  CHECK(shock_face<Axis::X>(cfg, L, R), "a Mach 25 jump trips the shock sensor");
  CHECK(!shock_face<Axis::X>(cfg, L, L),
        "a uniform face does not trip the sensor");
  // The reverse ordering is an expansion, not a shock, and HLLC keeps it.
  CHECK(!shock_face<Axis::X>(cfg, R, L),
        "a decompression does not trip the shock sensor");
  CHECK_NEAR(t_val(riemann_axis<Axis::X>(cfg, L, R, true).rho),
             t_val(hlle_axis<Axis::X>(cfg, L, R).rho), 1e-12,
             "the robust flag forces the dissipative flux");
}

static void test_slip_wall(const SimConfig &cfg) {
  Prim q = free_stream(cfg);
  q.u = t_of(4.0);
  q.v = t_of(1.0);
  Prim mirrored = prim_mirror<Axis::X>(q);
  CHECK_NEAR(t_val(mirrored.u), -4.0, 1e-10,
             "the wall reflects the face normal component");
  CHECK_NEAR(t_val(mirrored.v), 1.0, 1e-10,
             "the wall leaves the tangential component alone");
  CHECK(mirrored.rho == q.rho && mirrored.p == q.p,
        "the wall ghost keeps the thermodynamic state");

  // The reflected pair is what the flux kernel feeds a solid face, and it must
  // produce a pure pressure force: no mass and no energy cross a slip wall.
  Cons F = riemann_axis<Axis::X>(cfg, q, mirrored);
  CHECK_NEAR(t_val(F.rho), 0.0, 1e-9, "no mass crosses a slip wall");
  CHECK_NEAR(t_val(F.E), 0.0, 1e-6, "no energy crosses a slip wall");
  CHECK(t_val(F.mx) > 0.0, "the wall carries a positive pressure force");

  Prim mirroredY = prim_mirror<Axis::Y>(q);
  Cons G = riemann_axis<Axis::Y>(cfg, q, mirroredY);
  CHECK_NEAR(t_val(G.rho), 0.0, 1e-9, "no mass crosses a radial slip wall");
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

static void test_geometry(const SimConfig &cfg) {
  const double eps = 1e-3;
  CHECK(capsule_sdf(cfg, t_of(cfg.x_nose + eps), t_of(eps)) < T_NIL,
        "the heat shield apex is inside the body");
  CHECK(capsule_sdf(cfg, t_of(cfg.x_nose - 0.05), t_of(eps)) > T_NIL,
        "a point ahead of the apex is outside");
  CHECK(capsule_sdf(cfg, t_of(cfg.tail_x + 0.05), t_of(eps)) > T_NIL,
        "a point behind the aft plane is outside");
  CHECK(capsule_sdf(cfg, t_of(X_MAX), t_of(Y_MAX)) > T_NIL,
        "the far field is outside");
  CHECK(capsule_sdf(cfg, t_of(cfg.x_nose + 0.2), t_of(Y_MAX)) > T_NIL,
        "a point far off the axis is outside");

  double rmax = 0.0, xlo = X_MAX, xhi = 0.0;
  int solid = 0;
  for (int i = 0; i < NX; i++) {
    for (int j = 0; j < NY; j++) {
      double x = (i + 0.5) * cfg.dx;
      double y = (j + 0.5) * cfg.dy;
      if (capsule_sdf(cfg, t_of(x), t_of(y)) < T_NIL) {
        solid++;
        rmax = std::fmax(rmax, y);
        xlo = std::fmin(xlo, x);
        xhi = std::fmax(xhi, x);
      }
    }
  }
  // The fillet is flat at its crown, so many neighbouring columns tie for the
  // widest cell; take the middle of the tie as the shoulder station.
  double xr_lo = X_MAX, xr_hi = 0.0;
  for (int i = 0; i < NX; i++) {
    double x = (i + 0.5) * cfg.dx;
    if (capsule_sdf(cfg, t_of(x), t_of(rmax)) < T_NIL) {
      xr_lo = std::fmin(xr_lo, x);
      xr_hi = std::fmax(xr_hi, x);
    }
  }
  double xr = 0.5 * (xr_lo + xr_hi);
  printf("  body occupies %d cells, x in [%.4f, %.4f], max radius %.4f at "
         "x %.4f\n",
         solid, xlo, xhi, rmax, xr);
  // the capsule covers roughly one part in two hundred of the tunnel
  CHECK(solid > (NX * NY) / 200,
        "the body is resolved by a useful number of cells");
  CHECK_NEAR(xlo, cfg.x_nose, 2.0 * cfg.dx, "the nose sits where it was asked to");
  CHECK_NEAR(xhi, cfg.tail_x, 2.0 * cfg.dx, "the aft plane truncates the cone");
  // The fillet is built by eroding and dilating the two primitives, which is an
  // exact circular arc only where they meet at a right angle; here it is good
  // to about half a percent of the base radius.
  CHECK_NEAR(rmax, cfg.r_base, 2.0 * cfg.dy + 0.01 * cfg.r_base,
             "the filleted shoulder reaches the base radius");
  CHECK_NEAR(xr, cfg.shoulder_x, 0.25 * cfg.r_shoulder + cfg.dx,
             "the widest station is the predicted shoulder station");

  // The heat shield really is a sphere of the requested radius: sample the
  // surface on the axis and check the distance to the centre of curvature.
  double along = cfg.x_nose + 0.05;
  double lo = 0.0, hi = cfg.r_base;
  for (int k = 0; k < 60; k++) {
    double mid = 0.5 * (lo + hi);
    if (capsule_sdf(cfg, t_of(along), t_of(mid)) < T_NIL)
      lo = mid;
    else
      hi = mid;
  }
  double dxc = along - cfg.sphere_cx;
  CHECK_NEAR(std::sqrt(dxc * dxc + lo * lo), cfg.r_sphere, 1e-3,
             "the heat shield is a spherical segment of the stated radius");
}

// ---------------------------------------------------------------------------
// A one dimensional driver over the same kernels' arithmetic
// ---------------------------------------------------------------------------

struct Line {
  std::vector<Cons> U;
  double dx;
};

static void advance_line(const SimConfig &cfg, Line &L, double tend,
                         double cfl) {
  const int M = (int)L.U.size() - 1;
  std::vector<Cons> F(M + 1, nil_cons()), U1(L.U), U2(L.U);
  const double dx = L.dx;

  auto stage = [&](std::vector<Cons> &in, std::vector<Cons> &out, double dt) {
    for (int f = 2; f + 2 <= M; f++) {
      Prim a = cons_to_prim(cfg, in[f - 1]);
      Prim b = cons_to_prim(cfg, in[f]);
      Prim c = cons_to_prim(cfg, in[f + 1]);
      Prim d = cons_to_prim(cfg, in[f + 2]);
      Prim l = prim_face_hi(b, prim_slope(a, b, c));
      Prim r = prim_face_lo(c, prim_slope(b, c, d));
      if (!prim_ok(cfg, l) || !prim_ok(cfg, r)) {
        l = b;
        r = c;
      }
      F[f] = riemann_axis<Axis::X>(cfg, l, r);
    }
    for (int i = 3; i + 2 <= M; i++) {
      out[i].rho =
          t_plus(in[i].rho, t_scale(t_minus(F[i - 1].rho, F[i].rho), dt / dx));
      out[i].mx =
          t_plus(in[i].mx, t_scale(t_minus(F[i - 1].mx, F[i].mx), dt / dx));
      out[i].my =
          t_plus(in[i].my, t_scale(t_minus(F[i - 1].my, F[i].my), dt / dx));
      out[i].E =
          t_plus(in[i].E, t_scale(t_minus(F[i - 1].E, F[i].E), dt / dx));
    }
    for (int i = 1; i <= 3; i++)
      out[i] = out[4];
    for (int i = M - 2; i <= M; i++)
      out[i] = out[M - 3];
  };

  double t = 0.0;
  while (t < tend) {
    double lam = 0.0;
    for (int i = 3; i + 2 <= M; i++) {
      Prim q = cons_to_prim(cfg, L.U[i]);
      double s = std::fabs(t_val(q.u)) + t_val(sound_speed(cfg, q));
      if (s > lam)
        lam = s;
    }
    double dt = cfl * dx / lam;
    if (t + dt > tend)
      dt = tend - t;
    stage(L.U, U1, dt);
    stage(U1, U2, dt);
    for (int i = 1; i <= M; i++) {
      L.U[i].rho = t_half(t_plus(L.U[i].rho, U2[i].rho));
      L.U[i].mx = t_half(t_plus(L.U[i].mx, U2[i].mx));
      L.U[i].my = t_half(t_plus(L.U[i].my, U2[i].my));
      L.U[i].E = t_half(t_plus(L.U[i].E, U2[i].E));
    }
    t += dt;
  }
}

static Line make_line(const SimConfig &cfg, int n, int ghost, const Prim &left,
                      const Prim &right) {
  Line L;
  L.dx = 1.0 / n;
  L.U.assign(n + 2 * ghost + 1, nil_cons());
  for (int i = 1; i < (int)L.U.size(); i++) {
    double x = (i - ghost - 0.5) * L.dx;
    L.U[i] = prim_to_cons(cfg, x < 0.5 ? left : right);
  }
  return L;
}

static void test_shock_tube(const SimConfig &cfg) {
  const int n = 800, ghost = 2;
  Prim l, r;
  l.rho = t_of(1.0);
  l.u = T_NIL;
  l.v = T_NIL;
  l.p = t_of(1.0);
  r.rho = t_of(0.125);
  r.u = T_NIL;
  r.v = T_NIL;
  r.p = t_of(0.1);

  Line L = make_line(cfg, n, ghost, l, r);
  advance_line(cfg, L, 0.15, 0.4);

  double mass = 0.0, energy = 0.0, min_rho = 1e30, min_p = 1e30;
  for (int i = ghost + 1; i <= n + ghost; i++) {
    Prim q = cons_to_prim(cfg, L.U[i]);
    mass += t_val(L.U[i].rho) * L.dx;
    energy += t_val(L.U[i].E) * L.dx;
    min_rho = std::fmin(min_rho, t_val(q.rho));
    min_p = std::fmin(min_p, t_val(q.p));
  }
  const double mass0 = 0.5 * (1.0 + 0.125);
  const double energy0 = 0.5 * (1.0 + 0.1) / (cfg.gamma - 1.0);
  printf("  shock tube: mass %.10f (exact %.10f), energy %.10f (exact %.10f)\n",
         mass, mass0, energy, energy0);
  CHECK_NEAR(mass, mass0, 2e-4, "the multiplicative update conserves mass");
  CHECK_NEAR(energy, energy0, 2e-4, "the multiplicative update conserves energy");
  CHECK(min_rho > 0.1, "no density undershoot");
  CHECK(min_p > 0.05, "no pressure undershoot");
}

static void test_standing_shock(const SimConfig &cfg) {
  const int n = 800, ghost = 2;
  const double g = cfg.gamma, m = cfg.mach, m2 = m * m;
  const double rr = ((g + 1.0) * m2) / ((g - 1.0) * m2 + 2.0);
  const double pr = (2.0 * g * m2 - (g - 1.0)) / (g + 1.0);

  Prim l = free_stream(cfg);
  Prim r;
  r.rho = t_of(rr);
  r.u = t_of(m / rr);
  r.v = T_NIL;
  r.p = t_of(pr / g);

  Line L = make_line(cfg, n, ghost, l, r);
  advance_line(cfg, L, 0.02, 0.4);

  double half = 0.5 * (1.0 + rr), x_shock = 0.0;
  double rho_max = 0.0, p_max = 0.0, rho_min = 1e30, p_min = 1e30;
  for (int i = ghost + 1; i <= n + ghost; i++) {
    Prim q = cons_to_prim(cfg, L.U[i]);
    double rho = t_val(q.rho), p = t_val(q.p);
    if (x_shock == 0.0 && rho > half)
      x_shock = (i - ghost - 0.5) * L.dx;
    rho_max = std::fmax(rho_max, rho);
    p_max = std::fmax(p_max, p);
    rho_min = std::fmin(rho_min, rho);
    p_min = std::fmin(p_min, p);
  }
  printf("  standing Mach %.4g shock: x %.5f (exact 0.5), rho %.4f (exact "
         "%.4f), p %.4f (exact %.4f)\n",
         cfg.mach, x_shock, rho_max, rr, p_max, pr / g);
  CHECK_NEAR(x_shock, 0.5, 4.0 * L.dx, "a stationary shock does not drift");
  CHECK_NEAR(rho_max, rr, 0.05 * rr, "post shock density matches Rankine Hugoniot");
  CHECK_NEAR(p_max, pr / g, 0.05 * pr / g,
             "post shock pressure matches Rankine Hugoniot");
  CHECK(rho_min > 0.9, "no undershoot ahead of a Mach 25 shock");
  CHECK(p_min > 0.5 / g, "no pressure undershoot ahead of a Mach 25 shock");
}

static void test_uniform_flow_is_steady(const SimConfig &cfg) {
  const int n = 200, ghost = 2;
  Prim f = free_stream(cfg);
  Line L = make_line(cfg, n, ghost, f, f);
  Line before = L;
  advance_line(cfg, L, 0.01, 0.4);
  double worst = 0.0;
  for (int i = ghost + 1; i <= n + ghost; i++) {
    worst = std::fmax(worst, std::fabs(t_val(L.U[i].rho) - t_val(before.U[i].rho)));
    worst = std::fmax(worst, std::fabs(t_val(L.U[i].E) - t_val(before.U[i].E)));
  }
  printf("  uniform Mach %.4g stream drifts by %.3g over many steps\n", cfg.mach,
         worst);
  CHECK(worst < 1e-9, "a uniform stream is a fixed point of the scheme");
}

// ---------------------------------------------------------------------------
// The full axisymmetric solver, driven on the host
// ---------------------------------------------------------------------------
//
// The kernels are thin wrappers around host callable cell bodies, so the same
// solver that runs on the GPU can be stepped here one cell at a time. At the
// default grid this is slow; build with, for example,
//   nvcc -O3 -std=c++17 -DNX=192 -DNY=96 ... and pass --host-2d 2000
// to watch a bow shock form without a GPU.

struct HostField {
  std::vector<double> rho, mx, my, E;
  Field view() {
    Field f;
    f.rho = rho.data();
    f.mx = mx.data();
    f.my = my.data();
    f.E = E.data();
    return f;
  }
  HostField() : rho(NCELL, T_NIL), mx(NCELL, T_NIL), my(NCELL, T_NIL),
                E(NCELL, T_NIL) {}
};

// The shock flag has to fire in both directions, because the faces lying along
// a bow shock are exactly the ones a per face sensor would miss.
static void test_shock_flag(const SimConfig &cfg) {
  HostField hf;
  Field U = hf.view();
  std::vector<uint8_t> mask(NCELL, CELL_FLUID);
  std::vector<uint8_t> flags(NCELL, SHOCK_SMOOTH);

  Prim ahead = free_stream(cfg);
  Prim behind = ahead;
  behind.p = t_of(200.0);
  behind.u = t_of(2.0);
  behind.v = t_of(-2.0);

  const int cut_c = NX / 2, cut_r = NY / 2;

  // A compressive jump across one column: an axial shock.
  for (int j = 1; j <= NY + 2 * NG; j++)
    for (int i = 1; i <= NX + 2 * NG; i++)
      store_cons(U, cell_index(i, j),
                 prim_to_cons(cfg, i <= cut_c + NG ? ahead : behind));
  for (int t = 0; t < NX * NY; t++)
    cell_shock_flag(cfg, U, mask.data(), flags.data(), t);
  CHECK(flags[cell_index(cut_c + NG, ROW_FIRST + 1)] == SHOCK_STRONG,
        "the cell ahead of an axial jump is flagged");
  CHECK(flags[cell_index(cut_c + NG + 1, ROW_FIRST + 1)] == SHOCK_STRONG,
        "the cell behind an axial jump is flagged");
  CHECK(flags[cell_index(COL_FIRST + 1, ROW_FIRST + 1)] == SHOCK_SMOOTH,
        "smooth free stream is not flagged");

  // A compressive jump across one row: a radial shock. A per face sensor
  // sweeping in x would see nothing here.
  Prim below = free_stream(cfg);
  Prim above = below;
  above.p = t_of(200.0);
  above.v = t_of(-2.0);
  below.v = t_of(2.0);
  for (int j = 1; j <= NY + 2 * NG; j++)
    for (int i = 1; i <= NX + 2 * NG; i++)
      store_cons(U, cell_index(i, j),
                 prim_to_cons(cfg, j <= cut_r + NG ? below : above));
  for (int t = 0; t < NX * NY; t++)
    cell_shock_flag(cfg, U, mask.data(), flags.data(), t);
  int probe = cell_index(COL_FIRST + 3, cut_r + NG);
  CHECK(flags[probe] == SHOCK_STRONG,
        "a radial jump flags cells whose axial neighbours are smooth");
  CHECK(flags[cell_index(COL_FIRST + 3, ROW_FIRST + 1)] == SHOCK_SMOOTH,
        "cells far from the radial jump stay smooth");
}

static void test_host_two_d(const SimConfig &cfg, int steps) {
  HostField ha, hb, hc, hfx, hfy;
  Field Ua = ha.view(), Ub = hb.view(), Uc = hc.view();
  Field FX = hfx.view(), FY = hfy.view();
  std::vector<uint8_t> mask(NCELL, CELL_FLUID);
  std::vector<uint8_t> shock(NCELL, SHOCK_SMOOTH);

  for (int idx = 1; idx < NCELL; idx++)
    cell_fill_free_stream(cfg, Ua, mask.data(), idx);
  for (int t = 0; t < NX * NY; t++)
    cell_carve_body(cfg, Ua, mask.data(), t);

  Cons solid_before = load_cons(Ua, cell_index(NX / 2, ROW_FIRST));
  int solid_probe = -1;
  for (int c = 0; c < NX && solid_probe < 0; c++) {
    int id = cell_index(c + COL_FIRST, ROW_FIRST);
    if (mask[id] == CELL_SOLID)
      solid_probe = id;
  }
  if (solid_probe >= 0)
    solid_before = load_cons(Ua, solid_probe);

  auto boundaries = [&](Field &U) {
    for (int i = 1; i <= NX + 2 * NG; i++)
      cell_bc_radial(U, i);
    for (int j = 1; j <= NY + 2 * NG; j++)
      cell_bc_axial(cfg, U, j);
  };
  auto stage = [&](Field &in, Field &out, double dt) {
    for (int t = 0; t < NX * NY; t++)
      cell_shock_flag(cfg, in, mask.data(), shock.data(), t);
    for (int t = 0; t < NFACE_X * NY; t++)
      cell_flux<Axis::X>(cfg, in, mask.data(), shock.data(), FX, t);
    for (int t = 0; t < NX * NFACE_Y; t++)
      cell_flux<Axis::Y>(cfg, in, mask.data(), shock.data(), FY, t);
    for (int t = 0; t < NX * NY; t++)
      cell_update(cfg, in, out, mask.data(), FX, FY, dt, dt / cfg.dx,
                  dt / cfg.dy, t);
  };

  double elapsed = 0.0;
  for (int n = 0; n < steps; n++) {
    boundaries(Ua);
    double lam = TAU_TINY;
    for (int t = 0; t < NX * NY; t++)
      lam = t_max(lam, cell_wave_speed(cfg, Ua, mask.data(), t));
    double dt = t_val(t_over(t_of(cfg.cfl), lam));
    CHECK(dt > 0.0 && std::isfinite(dt), "the time step stays usable");
    stage(Ua, Ub, dt);
    boundaries(Ub);
    stage(Ub, Uc, dt);
    for (int idx = 1; idx < NCELL; idx++)
      cell_average(Ua, Uc, Ua, idx);
    elapsed += dt;
  }

  int bad = 0;
  double min_rho = 1e30, min_p = 1e30, max_mach = 0.0;
  for (int r = 0; r < NY; r++) {
    for (int c = 0; c < NX; c++) {
      int id = cell_index(c + COL_FIRST, r + ROW_FIRST);
      if (mask[id] != CELL_FLUID)
        continue;
      Cons u = load_cons(Ua, id);
      if (!t_finite(u.rho) || !t_finite(u.E))
        bad++;
      Prim q = cons_to_prim(cfg, u);
      min_rho = std::fmin(min_rho, t_val(q.rho));
      min_p = std::fmin(min_p, t_val(q.p));
      max_mach = std::fmax(max_mach, t_val(t_over(speed_mag(q),
                                                  sound_speed(cfg, q))));
    }
  }
  CHECK(bad == 0, "no cell leaves the representable range");
  CHECK(min_rho > 0.0 && min_p > 0.0, "the run stays in the physical state "
                                      "space");

  if (solid_probe >= 0) {
    Cons after = load_cons(Ua, solid_probe);
    CHECK(after.rho == solid_before.rho && after.E == solid_before.E,
          "solid cells are never updated");
  }

  // The upstream corner, far from the shock, must still be the free stream.
  Cons want = prim_to_cons(cfg, free_stream(cfg));
  int corner = cell_index(COL_FIRST, ROW_LAST);
  CHECK_NEAR(t_val(load_cons(Ua, corner).rho), t_val(want.rho), 1e-9,
             "the undisturbed corner is still the free stream");

  double p_inf = 1.0 / cfg.gamma, shock_x = X_MAX, stagnation = p_inf;
  for (int c = 0; c < NX; c++) {
    int id = cell_index(c + COL_FIRST, ROW_FIRST);
    if (mask[id] != CELL_FLUID)
      break;
    double p = t_val(cons_to_prim(cfg, load_cons(Ua, id)).p);
    if (shock_x == X_MAX && p > 2.0 * p_inf)
      shock_x = (c + 0.5) * cfg.dx;
    stagnation = p;
  }
  printf("  host 2d: %d steps, t %.5f, min rho %.4g, min p %.4g, max Mach "
         "%.4f\n",
         steps, elapsed, min_rho, min_p, max_mach);
  printf("  host 2d: bow shock x %.4f, stand off %.4f, stagnation p %.3f "
         "(Rayleigh pitot %.3f)\n",
         shock_x, cfg.x_nose - shock_x, stagnation, pitot_ratio(cfg) * p_inf);
  CHECK(max_mach > 0.9 * cfg.mach, "the tunnel still runs at the design Mach "
                                   "number");
  if (steps >= 500) {
    CHECK(shock_x < cfg.x_nose, "a bow shock stands ahead of the heat shield");
    CHECK_NEAR(stagnation, pitot_ratio(cfg) * p_inf,
               0.25 * pitot_ratio(cfg) * p_inf,
               "the stagnation pressure approaches the Rayleigh pitot value");
  }
}

// ---------------------------------------------------------------------------
// Device tests
// ---------------------------------------------------------------------------

static void test_device_setup(const SimConfig &cfg, Sim &s) {
  std::vector<uint8_t> mask(NCELL);
  std::vector<double> rho(NCELL), mx(NCELL), my(NCELL), E(NCELL);
  CK(cudaMemcpy(mask.data(), s.mask, NCELL, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(rho.data(), s.Ua.rho, NCELL * sizeof(double),
                cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(mx.data(), s.Ua.mx, NCELL * sizeof(double),
                cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(E.data(), s.Ua.E, NCELL * sizeof(double),
                cudaMemcpyDeviceToHost));

  int solid = 0, fluid = 0, host_solid = 0;
  for (int r = 0; r < NY; r++) {
    for (int c = 0; c < NX; c++) {
      int id = cell_index(c + COL_FIRST, r + ROW_FIRST);
      if (mask[id] == CELL_SOLID)
        solid++;
      else
        fluid++;
      double x = (c + 0.5) * cfg.dx, y = (r + 0.5) * cfg.dy;
      if (capsule_sdf(cfg, t_of(x), t_of(y)) < T_NIL)
        host_solid++;
    }
  }
  CHECK(solid == host_solid, "the carving kernel agrees with the host geometry");
  CHECK(solid > 1000 && fluid > solid, "the body is carved but does not fill "
                                       "the tunnel");

  // Free stream everywhere that is not solid.
  Cons want = prim_to_cons(cfg, free_stream(cfg));
  int id = cell_index(COL_FIRST, ROW_FIRST);
  CHECK_NEAR(rho[id], want.rho, 1e-15, "the initial state is the free stream");
  CHECK_NEAR(mx[id], want.mx, 1e-15, "the initial momentum is the free stream");
  CHECK_NEAR(E[id], want.E, 1e-15, "the initial energy is the free stream");
}

static void test_device_boundaries(const SimConfig &cfg, Sim &s) {
  apply_bc(s, s.Ua);
  CK(cudaDeviceSynchronize());

  std::vector<double> my(NCELL), rho(NCELL);
  CK(cudaMemcpy(my.data(), s.Ua.my, NCELL * sizeof(double),
                cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(rho.data(), s.Ua.rho, NCELL * sizeof(double),
                cudaMemcpyDeviceToHost));

  int col = COL_FIRST + NX / 4;
  CHECK_NEAR(my[cell_index(col, ROW_AXIS_INNER)],
             t_flip(my[cell_index(col, ROW_FIRST)]), 1e-15,
             "the axis ghost reverses the radial momentum");
  CHECK_NEAR(my[cell_index(col, ROW_AXIS_OUTER)],
             t_flip(my[cell_index(col, ROW_FIRST + 1)]), 1e-15,
             "the second axis ghost mirrors the second interior row");
  CHECK_NEAR(rho[cell_index(col, ROW_LAST + 1)], rho[cell_index(col, ROW_LAST)],
             1e-15, "the outer boundary is zero gradient");

  int row = ROW_FIRST + NY / 4;
  Cons want = prim_to_cons(cfg, free_stream(cfg));
  CHECK_NEAR(rho[cell_index(COL_IN_OUTER, row)], want.rho, 1e-15,
             "the inflow ghost carries the free stream");
  CHECK_NEAR(rho[cell_index(COL_LAST + 2, row)],
             rho[cell_index(COL_LAST, row)], 1e-15,
             "the outflow ghost extrapolates");
}

static void test_device_uniform_steady(const SimConfig &cfg, Sim &s) {
  // Refill without carving the body: an unobstructed tunnel must stay uniform.
  k_fill_free_stream<<<BLOCKS_FOR(NCELL), THREADS>>>(s.Ua, s.mask);
  CK(cudaDeviceSynchronize());

  Cons want = prim_to_cons(cfg, free_stream(cfg));
  for (int n = 0; n < 8; n++)
    sim_step(s, cfg);
  CK(cudaDeviceSynchronize());

  std::vector<double> rho(NCELL), E(NCELL), my(NCELL);
  CK(cudaMemcpy(rho.data(), s.Ua.rho, NCELL * sizeof(double),
                cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(E.data(), s.Ua.E, NCELL * sizeof(double),
                cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(my.data(), s.Ua.my, NCELL * sizeof(double),
                cudaMemcpyDeviceToHost));

  double worst_rho = 0.0, worst_E = 0.0, worst_my = 0.0;
  for (int r = 0; r < NY; r++) {
    for (int c = 0; c < NX; c++) {
      int id = cell_index(c + COL_FIRST, r + ROW_FIRST);
      worst_rho = std::fmax(worst_rho, std::fabs(t_val(rho[id]) - t_val(want.rho)));
      worst_E = std::fmax(worst_E, std::fabs(t_val(E[id]) - t_val(want.E)));
      worst_my = std::fmax(worst_my, std::fabs(t_val(my[id])));
    }
  }
  printf("  device uniform stream drift: rho %.3g, E %.3g, radial momentum "
         "%.3g\n",
         worst_rho, worst_E, worst_my);
  CHECK(worst_rho < 1e-9, "the axisymmetric update leaves a uniform rho alone");
  CHECK(worst_E < 1e-6, "the axisymmetric update leaves a uniform energy alone");
  CHECK(worst_my < 1e-9, "the geometric source creates no radial momentum");

  sim_reset(s);
}

// ---------------------------------------------------------------------------
// Regression baseline
// ---------------------------------------------------------------------------

struct Options {
  int steps;
  int host_2d_steps;
  const char *baseline_path;
  bool write_baseline;
  bool verify_baseline;
};

static bool parse_options(int argc, char **argv, Options *opt) {
  opt->steps = 24;
  opt->host_2d_steps = 0;
  opt->baseline_path = "tau_reentry_cuda_baseline.txt";
  opt->write_baseline = false;
  opt->verify_baseline = true;

  for (int i = 1; i < argc; i++) {
    const char *arg = argv[i];
    if (strcmp(arg, "--steps") == 0 && i + 1 < argc) {
      opt->steps = atoi(argv[++i]);
    } else if (strcmp(arg, "--host-2d") == 0 && i + 1 < argc) {
      opt->host_2d_steps = atoi(argv[++i]);
    } else if (strcmp(arg, "--baseline") == 0 && i + 1 < argc) {
      opt->baseline_path = argv[++i];
    } else if (strcmp(arg, "--write-baseline") == 0) {
      opt->write_baseline = true;
      opt->verify_baseline = false;
    } else if (strcmp(arg, "--verify-baseline") == 0) {
      opt->verify_baseline = true;
      opt->write_baseline = false;
    } else {
      fprintf(stderr,
              "Usage: %s [--steps N] [--host-2d N] [--baseline PATH] "
              "[--write-baseline|--verify-baseline]\n",
              argv[0]);
      return false;
    }
  }

  if (opt->steps <= 0) {
    fprintf(stderr, "--steps must be positive\n");
    return false;
  }
  return true;
}

static bool write_baseline(const char *path, int steps, const Diagnostics &d) {
  FILE *f = fopen(path, "w");
  if (!f) {
    fprintf(stderr, "Failed to open baseline for write: %s\n", path);
    return false;
  }
  fprintf(f, "steps %d\n", steps);
  fprintf(f, "fluid_cells %d\n", d.fluid_cells);
  fprintf(f, "mean_rho %.17g\n", d.mean_rho);
  fprintf(f, "mean_mx %.17g\n", d.mean_mx);
  fprintf(f, "mean_my %.17g\n", d.mean_my);
  fprintf(f, "mean_E %.17g\n", d.mean_E);
  fprintf(f, "min_rho %.17g\n", d.min_rho);
  fprintf(f, "min_p %.17g\n", d.min_p);
  fprintf(f, "max_mach %.17g\n", d.max_mach);
  fprintf(f, "stagnation_p %.17g\n", d.stagnation_p);
  fprintf(f, "shock_x %.17g\n", d.shock_x);
  fclose(f);
  return true;
}

static bool read_baseline(const char *path, int *steps, Diagnostics *d) {
  FILE *f = fopen(path, "r");
  if (!f) {
    fprintf(stderr, "Failed to open baseline for read: %s\n", path);
    return false;
  }
  int fields = fscanf(f,
                      "steps %d\nfluid_cells %d\nmean_rho %lf\nmean_mx %lf\n"
                      "mean_my %lf\nmean_E %lf\nmin_rho %lf\nmin_p %lf\n"
                      "max_mach %lf\nstagnation_p %lf\nshock_x %lf\n",
                      steps, &d->fluid_cells, &d->mean_rho, &d->mean_mx,
                      &d->mean_my, &d->mean_E, &d->min_rho, &d->min_p,
                      &d->max_mach, &d->stagnation_p, &d->shock_x);
  fclose(f);
  return fields == 11;
}

static void test_regression(const SimConfig &cfg, Sim &s, const Options &opt) {
  for (int n = 0; n < opt.steps; n++)
    sim_step(s, cfg);
  CK(cudaDeviceSynchronize());

  Diagnostics d = sim_diagnostics(s, cfg);
  print_diagnostics(d);

  CHECK(d.min_rho > 0.0, "density stays positive after the run");
  CHECK(d.min_p > 0.0, "pressure stays positive after the run");
  CHECK(std::isfinite(d.mean_rho) && std::isfinite(d.mean_E),
        "the mean state stays finite");
  CHECK(d.max_mach > 1.0, "the tunnel is still supersonic somewhere");
  CHECK_NEAR(d.max_mach, cfg.mach, 0.5,
             "the free stream Mach number survives the run");

  if (opt.write_baseline) {
    CHECK(write_baseline(opt.baseline_path, opt.steps, d),
          "baseline written");
    return;
  }
  if (!opt.verify_baseline)
    return;

  int steps = 0;
  Diagnostics ref{};
  if (!read_baseline(opt.baseline_path, &steps, &ref)) {
    g_stats.failed++;
    fprintf(stderr, "FAIL: could not read the baseline\n");
    return;
  }
  CHECK(steps == opt.steps, "the baseline was taken at the same step count");
  CHECK(ref.fluid_cells == d.fluid_cells, "the fluid cell count is stable");
  CHECK_NEAR(d.mean_rho, ref.mean_rho, 1e-12, "mean density matches baseline");
  CHECK_NEAR(d.mean_mx, ref.mean_mx, 1e-10, "mean axial momentum matches "
                                            "baseline");
  CHECK_NEAR(d.mean_my, ref.mean_my, 1e-10, "mean radial momentum matches "
                                            "baseline");
  CHECK_NEAR(d.mean_E, ref.mean_E, 1e-9, "mean energy matches baseline");
  CHECK_NEAR(d.min_rho, ref.min_rho, 1e-10, "minimum density matches baseline");
  CHECK_NEAR(d.min_p, ref.min_p, 1e-8, "minimum pressure matches baseline");
  CHECK_NEAR(d.max_mach, ref.max_mach, 1e-9, "peak Mach matches baseline");
  CHECK_NEAR(d.shock_x, ref.shock_x, 1e-12, "the shock sits where it did");
}

// ---------------------------------------------------------------------------

int main(int argc, char **argv) {
  Options opt{};
  if (!parse_options(argc, argv, &opt))
    return EXIT_FAILURE;

  SimConfig cfg = default_config();

  printf("host tests\n");
  test_algebra();
  test_limiter();
  test_thermo(cfg);
  test_flux(cfg);
  test_rankine_hugoniot(cfg);
  test_slip_wall(cfg);
  test_geometry(cfg);
  test_uniform_flow_is_steady(cfg);
  test_shock_tube(cfg);
  test_standing_shock(cfg);
  test_shock_flag(cfg);
  if (opt.host_2d_steps > 0)
    test_host_two_d(cfg, opt.host_2d_steps);

  int devices = 0;
  cudaError_t err = cudaGetDeviceCount(&devices);
  if (err != cudaSuccess || devices < 1) {
    printf("\nno CUDA device available: device tests and the regression "
           "baseline were skipped\n");
    printf("Passed: %d\nFailed: %d\n", g_stats.passed, g_stats.failed);
    return g_stats.failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
  }

  printf("\ndevice tests\n");
  CK(cudaMemcpyToSymbol(d_cfg, &cfg, sizeof(SimConfig)));
  Sim s{};
  sim_alloc(s);
  sim_reset(s);

  test_device_setup(cfg, s);
  test_device_boundaries(cfg, s);
  test_device_uniform_steady(cfg, s);
  test_regression(cfg, s, opt);

  sim_free(s);

  printf("Passed: %d\nFailed: %d\n", g_stats.passed, g_stats.failed);
  return g_stats.failed == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
