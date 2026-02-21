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

__global__ void k_test_roundtrip(double *out) {
  Prim p{1.4, 2.2, -0.7, 3.6};
  Cons c = prim_to_cons(p);
  Prim q = cons_to_prim(c);
  out[0] = q.rho;
  out[1] = q.u;
  out[2] = q.v;
  out[3] = q.p;
}

__global__ void k_test_clamps(double *out) {
  Prim badp{-2.0, 1.5, -0.5, -7.0};
  Cons c = prim_to_cons(badp);
  Prim q = cons_to_prim(Cons{1.0, 3.0, 4.0, 1e-20});

  out[0] = c.rho;
  out[1] = c.E;
  out[2] = q.rho;
  out[3] = q.p;
}

__global__ void k_test_limiters(double *out) {
  out[0] = minmod(1.0, 2.0);
  out[1] = minmod(-1.0, 2.0);
  out[2] = mc_limiter(1.0, 1.2, 1.5);
  out[3] = mc_limiter(-1.0, 0.2, 1.0);
}

__global__ void k_test_fluxes_and_sound(double *out) {
  Prim p{2.0, 3.0, -4.0, 5.0};
  Cons U = prim_to_cons(p);
  Cons Fx = flux_x(U);
  Cons Fy = flux_y(U);

  out[0] = Fx.rho;
  out[1] = Fx.mx;
  out[2] = Fx.my;
  out[3] = Fx.E;
  out[4] = Fy.rho;
  out[5] = Fy.mx;
  out[6] = Fy.my;
  out[7] = Fy.E;
  out[8] = sound_speed(p);
}

__global__ void k_test_inflow_state(double *out) {
  Prim infl = inflow_state();
  out[0] = infl.rho;
  out[1] = infl.u;
  out[2] = infl.v;
  out[3] = infl.p;
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

__global__ void k_test_enforce_positive(double *out) {
  Prim qc{1.0, 4.0, -2.0, 1.0};
  Prim qm{-1.0, 8.0, -4.0, -3.0};
  Prim qp{-2.0, -8.0, 4.0, -2.0};
  enforce_positive_faces(qm, qc, qp);

  out[0] = qm.rho;
  out[1] = qm.p;
  out[2] = qp.rho;
  out[3] = qp.p;
}

__global__ void k_test_enforce_positive_no_change(double *out) {
  Prim qc{1.0, 2.0, -1.0, 1.0};
  Prim qm{0.8, 2.2, -0.9, 1.1};
  Prim qp{1.2, 1.8, -1.2, 0.9};
  enforce_positive_faces(qm, qc, qp);

  out[0] = qm.rho;
  out[1] = qm.p;
  out[2] = qp.rho;
  out[3] = qp.p;
}

__global__ void k_test_sdf(double *out) {
  double Rb = 5.0;
  double Rn = 2.0;
  double theta = 0.6;
  out[0] = sdSphereConeCapsule(1.0, 0.0, Rb, Rn, theta);
  out[1] = sdSphereConeCapsule(40.0, 0.0, Rb, Rn, theta);
}

__global__ void k_test_neighbors(Usoa U, const uint8_t *mask, double *out,
                                 int x, int y) {
  Cons left = neighbor_or_wall(U, mask, x, y, -1, 0);
  Cons right = neighbor_or_wall(U, mask, x, y, +1, 0);
  Cons up = neighbor_or_wall(U, mask, x, y, 0, +1);

  out[0] = left.rho;
  out[1] = left.mx;
  out[2] = right.rho;
  out[3] = right.mx;
  out[4] = up.mx;
}

__global__ void k_test_neighbor_for_diff(Usoa U, const uint8_t *mask,
                                         double *out, int x, int y) {
  Cons left = neighbor_for_diff(U, mask, x, y, x - 1, y);
  Cons wall = neighbor_for_diff(U, mask, x, y, x, y + 1);
  Cons top_clamped = neighbor_for_diff(U, mask, x, y, x, H + 20);

  out[0] = left.rho;
  out[1] = left.mx;
  out[2] = wall.mx;
  out[3] = top_clamped.rho;
}

int main() {
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

  k_test_clamps<<<1, 1>>>(d);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, d, 4 * sizeof(double), cudaMemcpyDeviceToHost));
  CHECK_NEAR(stats, h[0], EPS_RHO, 1e-30, "prim_to_cons clamps rho floor");
  CHECK_TRUE(stats, h[1] >= EPS_P / (cfg.gamma - 1.0),
             "prim_to_cons keeps positive internal energy");
  CHECK_NEAR(stats, h[2], 1.0, 1e-12, "cons_to_prim keeps positive rho");
  CHECK_TRUE(stats, h[3] >= EPS_P, "cons_to_prim clamps pressure floor");

  k_test_limiters<<<1, 1>>>(d);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, d, 4 * sizeof(double), cudaMemcpyDeviceToHost));
  CHECK_NEAR(stats, h[0], 1.0, 1e-15, "minmod picks smaller same-sign value");
  CHECK_NEAR(stats, h[1], 0.0, 1e-15, "minmod returns zero opposite sign");
  CHECK_TRUE(stats, h[2] > 0.0 && h[2] <= 1.0,
             "mc limiter bounded for monotone stencil");
  CHECK_NEAR(stats, h[3], 0.0, 1e-15,
             "mc limiter returns zero across sign change");

  k_test_fluxes_and_sound<<<1, 1>>>(d);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, d, 9 * sizeof(double), cudaMemcpyDeviceToHost));
  CHECK_NEAR(stats, h[0], 6.0, 1e-12, "flux_x rho equals rho*u");
  CHECK_NEAR(stats, h[1], 23.0, 1e-12, "flux_x mx equals rho*u^2+p");
  CHECK_NEAR(stats, h[2], -24.0, 1e-12, "flux_x my equals rho*u*v");
  CHECK_NEAR(stats, h[3], 102.0, 1e-12, "flux_x E equals (E+p)u");
  CHECK_NEAR(stats, h[4], -8.0, 1e-12, "flux_y rho equals rho*v");
  CHECK_NEAR(stats, h[5], -24.0, 1e-12, "flux_y mx equals rho*u*v");
  CHECK_NEAR(stats, h[6], 37.0, 1e-12, "flux_y my equals rho*v^2+p");
  CHECK_NEAR(stats, h[7], -136.0, 1e-12, "flux_y E equals (E+p)v");
  CHECK_NEAR(stats, h[8], std::sqrt(cfg.gamma * 5.0 / 2.0), 1e-12,
             "sound speed matches ideal-gas formula");

  k_test_inflow_state<<<1, 1>>>(d);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, d, 4 * sizeof(double), cudaMemcpyDeviceToHost));
  CHECK_NEAR(stats, h[0], 1.0, 1e-12, "inflow rho matches default");
  CHECK_NEAR(stats, h[1], cfg.inflow_mach * std::sqrt(cfg.gamma), 1e-12,
             "inflow u equals mach * sound speed");
  CHECK_NEAR(stats, h[2], 0.0, 1e-12, "inflow v is zero");
  CHECK_NEAR(stats, h[3], 1.0, 1e-12, "inflow p matches default");

  k_test_hllc_consistency<<<1, 1>>>(d);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, d, 8 * sizeof(double), cudaMemcpyDeviceToHost));
  for (int i = 0; i < 8; i++) {
    CHECK_NEAR(stats, h[i], 0.0, 1e-11,
               "hllc flux matches physical flux for equal states");
  }

  k_test_enforce_positive<<<1, 1>>>(d);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, d, 4 * sizeof(double), cudaMemcpyDeviceToHost));
  CHECK_TRUE(stats, h[0] >= EPS_RHO, "enforce_positive_faces keeps qm rho > 0");
  CHECK_TRUE(stats, h[1] >= EPS_P, "enforce_positive_faces keeps qm p > 0");
  CHECK_TRUE(stats, h[2] >= EPS_RHO, "enforce_positive_faces keeps qp rho > 0");
  CHECK_TRUE(stats, h[3] >= EPS_P, "enforce_positive_faces keeps qp p > 0");

  k_test_enforce_positive_no_change<<<1, 1>>>(d);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, d, 4 * sizeof(double), cudaMemcpyDeviceToHost));
  CHECK_NEAR(stats, h[0], 0.8, 1e-12,
             "enforce_positive_faces keeps valid qm rho unchanged");
  CHECK_NEAR(stats, h[1], 1.1, 1e-12,
             "enforce_positive_faces keeps valid qm p unchanged");
  CHECK_NEAR(stats, h[2], 1.2, 1e-12,
             "enforce_positive_faces keeps valid qp rho unchanged");
  CHECK_NEAR(stats, h[3], 0.9, 1e-12,
             "enforce_positive_faces keeps valid qp p unchanged");

  k_test_sdf<<<1, 1>>>(d);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, d, 2 * sizeof(double), cudaMemcpyDeviceToHost));
  CHECK_TRUE(stats, h[0] < 0.0, "sdf is negative inside body");
  CHECK_TRUE(stats, h[1] > 0.0, "sdf is positive outside body");

  const int N = W * H;
  Usoa U{};
  alloc_Us(&U, N);
  uint8_t *dMask = nullptr;
  CK(cudaMalloc(&dMask, N * sizeof(uint8_t)));

  double *hrho = (double *)malloc(N * sizeof(double));
  double *hmx = (double *)malloc(N * sizeof(double));
  double *hmy = (double *)malloc(N * sizeof(double));
  double *hE = (double *)malloc(N * sizeof(double));
  uint8_t *hmask = (uint8_t *)malloc(N * sizeof(uint8_t));

  for (int i = 0; i < N; i++) {
    hrho[i] = 1.0;
    hmx[i] = 0.0;
    hmy[i] = 0.0;
    hE[i] = prim_to_cons(Prim{1.0, 0.0, 0.0, 1.0}).E;
    hmask[i] = 0;
  }

  int x = 0, y = 10;
  int i_center = y * W + x;
  int i_right = y * W + (x + 1);
  int i_up = (y + 1) * W + x;

  hmx[i_center] = 3.0;
  hmx[i_right] = 7.0;
  hmask[i_up] = 1;

  CK(cudaMemcpy(U.rho, hrho, N * sizeof(double), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(U.mx, hmx, N * sizeof(double), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(U.my, hmy, N * sizeof(double), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(U.E, hE, N * sizeof(double), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dMask, hmask, N * sizeof(uint8_t), cudaMemcpyHostToDevice));

  k_test_neighbors<<<1, 1>>>(U, dMask, d, x, y);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, d, 5 * sizeof(double), cudaMemcpyDeviceToHost));
  Prim infl = inflow_state();
  CHECK_NEAR(stats, h[0], infl.rho, 1e-12, "left boundary uses inflow rho");
  CHECK_NEAR(stats, h[1], prim_to_cons(infl).mx, 1e-10,
             "left boundary uses inflow momentum");
  CHECK_NEAR(stats, h[2], 1.0, 1e-12, "right neighbor uses cell rho");
  CHECK_NEAR(stats, h[3], 7.0, 1e-12, "right neighbor uses fluid state");
  CHECK_NEAR(stats, h[4], -3.0, 1e-12,
             "masked neighbor reflects no-slip momentum");

  k_test_neighbor_for_diff<<<1, 1>>>(U, dMask, d, x, y);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, d, 4 * sizeof(double), cudaMemcpyDeviceToHost));
  CHECK_NEAR(stats, h[0], infl.rho, 1e-12,
             "neighbor_for_diff left boundary uses inflow rho");
  CHECK_NEAR(stats, h[1], prim_to_cons(infl).mx, 1e-10,
             "neighbor_for_diff left boundary uses inflow mx");
  CHECK_NEAR(stats, h[2], -3.0, 1e-12,
             "neighbor_for_diff masked cell reflects momentum");
  CHECK_NEAR(stats, h[3], 1.0, 1e-12,
             "neighbor_for_diff clamps y index before lookup");

  free(hrho);
  free(hmx);
  free(hmy);
  free(hE);
  free(hmask);
  cudaFree(dMask);
  free_Us(&U);

  cudaFree(d);

  printf("Passed: %d\nFailed: %d\n", stats.passed, stats.failed);
  return stats.failed == 0 ? 0 : 1;
}
