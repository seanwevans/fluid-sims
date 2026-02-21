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

__global__ void k_test_limiters(double *out) {
  out[0] = minmod(1.0, 2.0);
  out[1] = minmod(-1.0, 2.0);
  out[2] = mc_limiter(1.0, 1.2, 1.5);
  out[3] = mc_limiter(-1.0, 0.2, 1.0);
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

int main() {
  TestStats stats{0, 0};

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

  k_test_limiters<<<1, 1>>>(d);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, d, 4 * sizeof(double), cudaMemcpyDeviceToHost));
  CHECK_NEAR(stats, h[0], 1.0, 1e-15, "minmod picks smaller same-sign value");
  CHECK_NEAR(stats, h[1], 0.0, 1e-15, "minmod returns zero opposite sign");
  CHECK_TRUE(stats, h[2] > 0.0 && h[2] <= 1.0,
             "mc limiter bounded for monotone stencil");
  CHECK_NEAR(stats, h[3], 0.0, 1e-15,
             "mc limiter returns zero across sign change");

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
