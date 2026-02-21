#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define TAU_HYPERSONIC_CUDA_NO_RAYLIB
#define TAU_HYPERSONIC_CUDA_NO_MAIN
#include "tau_hypersonic_cuda.cu"

struct TestStats {
  int passed;
  int failed;
};

struct RegressionOptions {
  int steps;
  const char *baseline_path;
  bool write_baseline;
  bool verify_baseline;
};

struct RegressionSnapshot {
  int steps;
  int fluid_cells;
  double sum_rho;
  double sum_mx;
  double sum_my;
  double sum_E;
  double min_rho;
  double min_p;
  double max_mach;
  double checksum_rho;
  double checksum_mx;
  double checksum_E;
};

#define CHECK_TRUE(stats, cond, msg)                                             \
  do {                                                                            \
    if (cond) {                                                                   \
      (stats).passed++;                                                           \
    } else {                                                                      \
      (stats).failed++;                                                           \
      fprintf(stderr, "FAIL: %s (%s:%d)\n", msg, __FILE__, __LINE__);           \
    }                                                                             \
  } while (0)

#define CHECK_NEAR(stats, a, b, tol, msg)                                         \
  CHECK_TRUE(stats, std::fabs((a) - (b)) <= (tol), msg)

static bool parse_regression_options(int argc, char **argv,
                                     RegressionOptions *opt) {
  opt->steps = 24;
  opt->baseline_path = "tau_hypersonic_cuda_baseline.txt";
  opt->write_baseline = false;
  opt->verify_baseline = true;

  for (int i = 1; i < argc; i++) {
    const char *arg = argv[i];
    if (strcmp(arg, "--steps") == 0 && i + 1 < argc) {
      opt->steps = atoi(argv[++i]);
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
              "Usage: %s [--steps N] [--baseline PATH] [--write-baseline|--verify-baseline]\n",
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

static bool write_snapshot(const char *path, const RegressionSnapshot &s) {
  FILE *f = fopen(path, "w");
  if (!f) {
    fprintf(stderr, "Failed to open baseline for write: %s\n", path);
    return false;
  }

  fprintf(f, "steps %d\n", s.steps);
  fprintf(f, "fluid_cells %d\n", s.fluid_cells);
  fprintf(f, "sum_rho %.17g\n", s.sum_rho);
  fprintf(f, "sum_mx %.17g\n", s.sum_mx);
  fprintf(f, "sum_my %.17g\n", s.sum_my);
  fprintf(f, "sum_E %.17g\n", s.sum_E);
  fprintf(f, "min_rho %.17g\n", s.min_rho);
  fprintf(f, "min_p %.17g\n", s.min_p);
  fprintf(f, "max_mach %.17g\n", s.max_mach);
  fprintf(f, "checksum_rho %.17g\n", s.checksum_rho);
  fprintf(f, "checksum_mx %.17g\n", s.checksum_mx);
  fprintf(f, "checksum_E %.17g\n", s.checksum_E);

  fclose(f);
  return true;
}

static bool read_snapshot(const char *path, RegressionSnapshot *s) {
  FILE *f = fopen(path, "r");
  if (!f) {
    fprintf(stderr, "Failed to open baseline for read: %s\n", path);
    return false;
  }

  const int fields = fscanf(
      f,
      "steps %d\nfluid_cells %d\nsum_rho %lf\nsum_mx %lf\nsum_my %lf\nsum_E %lf\n"
      "min_rho %lf\nmin_p %lf\nmax_mach %lf\nchecksum_rho %lf\nchecksum_mx %lf\nchecksum_E %lf\n",
      &s->steps, &s->fluid_cells, &s->sum_rho, &s->sum_mx, &s->sum_my,
      &s->sum_E, &s->min_rho, &s->min_p, &s->max_mach, &s->checksum_rho,
      &s->checksum_mx, &s->checksum_E);

  fclose(f);
  return fields == 12;
}

static Prim host_cons_to_prim(const SimConfig &cfg, const Cons &c) {
  Prim p{};
  const double rho = fmax(c.rho, EPS_RHO);
  const double inv = 1.0 / rho;
  const double u = c.mx * inv;
  const double v = c.my * inv;
  const double kin = 0.5 * rho * (u * u + v * v);
  const double eint = c.E - kin;
  const double pr = (cfg.gamma - 1.0) * fmax(eint, EPS_P);
  p.rho = rho;
  p.u = u;
  p.v = v;
  p.p = pr;
  return p;
}

static void compute_snapshot(const SimConfig &cfg, int steps, const double *rho,
                             const double *mx, const double *my,
                             const double *E, const uint8_t *mask,
                             RegressionSnapshot *out) {
  RegressionSnapshot s{};
  s.steps = steps;
  s.min_rho = 1e300;
  s.min_p = 1e300;

  for (int i = 0; i < W * H; i++) {
    if (mask[i])
      continue;

    Cons c{rho[i], mx[i], my[i], E[i]};
    Prim p = host_cons_to_prim(cfg, c);
    double a = sqrt(cfg.gamma * fmax(p.p, EPS_P) / fmax(p.rho, EPS_RHO));
    double mach = sqrt(p.u * p.u + p.v * p.v) / fmax(a, 1e-30);
    double w = (double)((i % 8191) + 1);

    s.fluid_cells++;
    s.sum_rho += p.rho;
    s.sum_mx += c.mx;
    s.sum_my += c.my;
    s.sum_E += c.E;
    s.min_rho = fmin(s.min_rho, p.rho);
    s.min_p = fmin(s.min_p, p.p);
    s.max_mach = fmax(s.max_mach, mach);
    s.checksum_rho += w * p.rho;
    s.checksum_mx += w * c.mx;
    s.checksum_E += w * c.E;
  }

  *out = s;
}

static void run_hypersonic_steps(const SimConfig &cfg, Usoa dU, Usoa dUtmp,
                                 uint8_t *dMask, Csoa dXStateL, Csoa dXStateR,
                                 Csoa dYStateL, Csoa dYStateR, Csoa dXFlux,
                                 Csoa dYFlux, double *dMaxSpeed,
                                 double *dBlockSpeedMax, int steps) {
  const int N = W * H;
  const int threads = 256;
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

  for (int s = 0; s < steps; s++) {
    k_apply_inflow_left<<<(H + threads - 1) / threads, threads>>>(dU, dMask);
    CK(cudaGetLastError());

    k_max_wavespeed_blocks<<<blocksN, threads>>>(dU, dMask, dBlockSpeedMax);
    CK(cudaGetLastError());
    k_reduce_block_max<<<1, threads>>>(dBlockSpeedMax, blocksN, dMaxSpeed);
    CK(cudaGetLastError());
    CK(cudaDeviceSynchronize());

    double maxs = 1e-12;
    CK(cudaMemcpy(&maxs, dMaxSpeed, sizeof(double), cudaMemcpyDeviceToHost));
    if (!isfinite(maxs) || maxs < 1e-12)
      maxs = 1e-12;

    const double dt_convective = cfg.cfl / maxs;
    const double nu_max = fmax(cfg.visc_nu, fmax(cfg.visc_rho, cfg.visc_e));
    double dt_diff = dt_convective;
    if (isfinite(nu_max) && nu_max > 1e-12)
      dt_diff = 0.25 / nu_max;

    const double dt = fmin(dt_convective, dt_diff);
    const double half_dt = 0.5 * dt;

    k_predict_face_states<<<blocksNTiled, tileBlock, shmPredict>>>(
        dU, dMask, dXStateL, dXStateR, dYStateL, dYStateR, half_dt, half_dt);
    CK(cudaGetLastError());
    k_compute_xface_flux<<<blocksXFaces, threads>>>(dU, dMask, dXStateL,
                                                    dXStateR, dXFlux);
    CK(cudaGetLastError());
    k_compute_yface_flux<<<blocksYFaces, threads>>>(dU, dMask, dYStateL,
                                                    dYStateR, dYFlux);
    CK(cudaGetLastError());
    k_step<<<blocksNTiled, tileBlock, shmStep>>>(dU, dUtmp, dMask, dXFlux,
                                                  dYFlux, dt, dt, dt);
    CK(cudaGetLastError());
    swap_Us(&dU, &dUtmp);
  }

  CK(cudaDeviceSynchronize());
}

__global__ void k_test_roundtrip(double *out) {
  Prim p{1.4, 2.2, -0.7, 3.6};
  Cons c = prim_to_cons(p);
  Prim q = cons_to_prim(c);
  out[0] = q.rho;
  out[1] = q.u;
  out[2] = q.v;
  out[3] = q.p;
}

__global__ void k_test_hllc_consistency(double *out) {
  Prim p{1.0, 3.0, -0.5, 2.0};
  Cons U = prim_to_cons(p);
  Cons Fx = hllc_x(U, U);
  Cons Fy = hllc_y(U, U);
  Cons Fx_ref = flux_x(U);
  Cons Fy_ref = flux_y(U);

  out[0] = Fx.rho - Fx_ref.rho;
  out[1] = Fx.mx - Fx_ref.mx;
  out[2] = Fx.my - Fx_ref.my;
  out[3] = Fx.E - Fx_ref.E;
  out[4] = Fy.rho - Fy_ref.rho;
  out[5] = Fy.mx - Fy_ref.mx;
  out[6] = Fy.my - Fy_ref.my;
  out[7] = Fy.E - Fy_ref.E;
}

int main(int argc, char **argv) {
  RegressionOptions options{};
  if (!parse_regression_options(argc, argv, &options))
    return 2;

  TestStats stats{0, 0};
  SimConfig cfg = default_config();
  CK(cudaMemcpyToSymbol(d_cfg, &cfg, sizeof(SimConfig)));

  double *d = nullptr;
  CK(cudaMalloc(&d, 16 * sizeof(double)));
  double h[16];

  k_test_roundtrip<<<1, 1>>>(d);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, d, 4 * sizeof(double), cudaMemcpyDeviceToHost));
  CHECK_NEAR(stats, h[0], 1.4, 1e-12, "cons/prim roundtrip preserves rho");
  CHECK_NEAR(stats, h[1], 2.2, 1e-12, "cons/prim roundtrip preserves u");
  CHECK_NEAR(stats, h[2], -0.7, 1e-12, "cons/prim roundtrip preserves v");
  CHECK_NEAR(stats, h[3], 3.6, 1e-12, "cons/prim roundtrip preserves p");

  k_test_hllc_consistency<<<1, 1>>>(d);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, d, 8 * sizeof(double), cudaMemcpyDeviceToHost));
  for (int i = 0; i < 8; i++) {
    CHECK_NEAR(stats, h[i], 0.0, 1e-11,
               "hllc flux matches physical flux for equal states");
  }

  const int N = W * H;
  const int threads = 256;
  const int blocksN = (N + threads - 1) / threads;

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

  double *dMaxSpeed = nullptr;
  double *dBlockSpeedMax = nullptr;
  CK(cudaMalloc(&dMaxSpeed, sizeof(double)));
  CK(cudaMalloc(&dBlockSpeedMax, (size_t)blocksN * sizeof(double)));

  k_init<<<blocksN, threads>>>(dU, dMask);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());

  run_hypersonic_steps(cfg, dU, dUtmp, dMask, dXStateL, dXStateR, dYStateL,
                       dYStateR, dXFlux, dYFlux, dMaxSpeed, dBlockSpeedMax,
                       options.steps);

  double *hRho = (double *)malloc((size_t)N * sizeof(double));
  double *hMx = (double *)malloc((size_t)N * sizeof(double));
  double *hMy = (double *)malloc((size_t)N * sizeof(double));
  double *hE = (double *)malloc((size_t)N * sizeof(double));
  uint8_t *hMask = (uint8_t *)malloc((size_t)N * sizeof(uint8_t));

  CK(cudaMemcpy(hRho, dU.rho, (size_t)N * sizeof(double), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hMx, dU.mx, (size_t)N * sizeof(double), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hMy, dU.my, (size_t)N * sizeof(double), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hE, dU.E, (size_t)N * sizeof(double), cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hMask, dMask, (size_t)N * sizeof(uint8_t), cudaMemcpyDeviceToHost));

  RegressionSnapshot current{};
  compute_snapshot(cfg, options.steps, hRho, hMx, hMy, hE, hMask, &current);
  CHECK_TRUE(stats, current.fluid_cells > 0, "snapshot includes fluid cells");
  CHECK_TRUE(stats, current.min_rho >= EPS_RHO,
             "regression run keeps density positive");
  CHECK_TRUE(stats, current.min_p >= EPS_P,
             "regression run keeps pressure positive");

  if (options.write_baseline) {
    CHECK_TRUE(stats, write_snapshot(options.baseline_path, current),
               "write regression baseline");
  } else if (options.verify_baseline) {
    RegressionSnapshot expected{};
    CHECK_TRUE(stats, read_snapshot(options.baseline_path, &expected),
               "read regression baseline");
    if (stats.failed == 0) {
      CHECK_TRUE(stats, expected.steps == current.steps,
                 "steps match baseline");
      CHECK_TRUE(stats, expected.fluid_cells == current.fluid_cells,
                 "fluid cell count matches baseline");
      CHECK_NEAR(stats, current.sum_rho, expected.sum_rho, 5e-8 * fabs(expected.sum_rho) + 1e-8,
                 "sum_rho matches baseline");
      CHECK_NEAR(stats, current.sum_mx, expected.sum_mx, 5e-8 * fabs(expected.sum_mx) + 1e-8,
                 "sum_mx matches baseline");
      CHECK_NEAR(stats, current.sum_my, expected.sum_my, 5e-8 * fabs(expected.sum_my) + 1e-8,
                 "sum_my matches baseline");
      CHECK_NEAR(stats, current.sum_E, expected.sum_E, 5e-8 * fabs(expected.sum_E) + 1e-8,
                 "sum_E matches baseline");
      CHECK_NEAR(stats, current.min_rho, expected.min_rho, 1e-9,
                 "min_rho matches baseline");
      CHECK_NEAR(stats, current.min_p, expected.min_p, 1e-9,
                 "min_p matches baseline");
      CHECK_NEAR(stats, current.max_mach, expected.max_mach,
                 5e-8 * fabs(expected.max_mach) + 1e-8,
                 "max_mach matches baseline");
      CHECK_NEAR(stats, current.checksum_rho, expected.checksum_rho,
                 5e-8 * fabs(expected.checksum_rho) + 1e-8,
                 "checksum_rho matches baseline");
      CHECK_NEAR(stats, current.checksum_mx, expected.checksum_mx,
                 5e-8 * fabs(expected.checksum_mx) + 1e-8,
                 "checksum_mx matches baseline");
      CHECK_NEAR(stats, current.checksum_E, expected.checksum_E,
                 5e-8 * fabs(expected.checksum_E) + 1e-8,
                 "checksum_E matches baseline");
    }
  }

  free(hRho);
  free(hMx);
  free(hMy);
  free(hE);
  free(hMask);

  cudaFree(dMaxSpeed);
  cudaFree(dBlockSpeedMax);
  cudaFree(dMask);
  free_Cs(&dXStateL);
  free_Cs(&dXStateR);
  free_Cs(&dYStateL);
  free_Cs(&dYStateR);
  free_Cs(&dXFlux);
  free_Cs(&dYFlux);
  free_Us(&dU);
  free_Us(&dUtmp);
  cudaFree(d);

  printf("Passed: %d\nFailed: %d\n", stats.passed, stats.failed);
  return stats.failed == 0 ? 0 : 1;
}
