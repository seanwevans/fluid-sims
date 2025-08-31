// tau_burgers_ncurses.cu  — Burgers (2-D), log-time clock, optional asinh
// "log-space". Upgrades: --halfblocks (renderer), --muscl (MUSCL/minmod),
// --visc_substeps K,
//           --colehopf (1-D validation with exact solution; shows rel L2
//           error).
//
// Build:
//   nvcc -std=c++17 -O3 -use_fast_math -arch=sm_86 -lineinfo -o tau_burgers \
//        tau_burgers_ncurses.cu -lncursesw
//
// Notes:
// - State stores phi_u, phi_v where u = u0*sinh(phi_u), v = u0*sinh(phi_v).
// - Log time: t = t0*exp(tau), dt_eff = min(t*dtau, dt_CFL).
// - Convection: Rusanov (LLF). MUSCL optional (minmod).
// - Viscosity: explicit Laplacian on (u,v). "Semi-implicit" via K substeps.
// - Cole–Hopf mode: exact 1-D solution with theta(x,t) = 1 + a*e^{-νk^2 t}
// cos(kx)
//   u_exact = 2ν a k e^{-νk^2 t} sin(kx) / (1 + a e^{-νk^2 t} cos(kx))
//   Use --colehopf, --ck (integer mode), --ca (|A|<1).
//
// Controls in UI: 'q' to quit.

#include <algorithm>
#include <chrono>
#include <cstdint>

#include <getopt.h>
#include <locale.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vector>

#include <cuda_runtime.h>
#include <ncursesw/curses.h>

static bool curses_active = false;

#define CUDA_CHECK(ans)                                                        \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    if (curses_active) {
      endwin();
      curses_active = false;
    }
    fprintf(stderr, "CUDA Error: %s %s:%d\n", cudaGetErrorString(code), file,
            line);
    exit(code);
  }
}

struct Params {
  // grid
  int nx = 512, ny = 512;
  float dx = 1.0f, dy = 1.0f;

  // physics
  float nu = 0.1f;
  float u0 = 1.0f;

  // initial field (2-D)
  float amp = 1.0f;
  float bsig = 16.0f;
  float swirl = 10.0f;
  float rc = 40.0f;
  float offx = 0.0f, offy = 0.0f;
  float asym = 0.0f;

  // time / CFL
  float CFL = 0.45f;
  int steps = 0; // 0 = run forever
  float tau0 = 0.0f, t0 = 1.0f, dtau = 1.0f;

  // UI/bench
  bool headless = false;
  int stride = 5;
  int fps_limit = 0;

  // upgrades/toggles
  bool halfblocks = false;
  bool muscl = false;
  int visc_substeps = 1;

  // Cole–Hopf harness (1-D)
  bool colehopf = false;
  int ck = 4;      // mode number
  float ca = 0.5f; // amplitude |ca|<1
};

__host__ __device__ inline int wrap(int i, int n) {
  i %= n;
  if (i < 0)
    i += n;
  return i;
}
__host__ __device__ inline int ID(int i, int j, int nx, int ny) {
  return wrap(j, ny) * nx + wrap(i, nx);
}
__host__ __device__ inline int IDXF(int i, int j, int nx, int ny) {
  return wrap(j, ny) * nx + wrap(i, nx);
}
__host__ __device__ inline int IDYF(int i, int j, int nx, int ny) {
  return wrap(j, ny) * nx + wrap(i, nx);
}

void usage(const char *prog) {
  printf("Usage: %s [options]\n", prog);
  puts("  --nx N           grid x (512)");
  puts("  --ny N           grid y (512)");
  puts("  --dx M           dx (1)");
  puts("  --dy M           dy (1)");
  puts("  --nu NU          viscosity (0.01)");
  puts("  --u0 U0          scale for u = u0*sinh(phi) (1)");
  puts("  --amp A          init amplitude (1)");
  puts("  --bsig S         init sigma (cells) (16)");
  puts("  --swirl O        init swirl rate (1)");
  puts("  --rc R           core radius (cells) (40)");
  puts("  --offx X         center x shift (0)");
  puts("  --offy Y         center y shift (0)");
  puts("  --asym A         dipole modulation (0)");
  puts("  --CFL C          CFL (0.45)");
  puts("  --steps K        steps (0 forever)");
  puts("  --tau0 T         initial tau (0)");
  puts("  --t0 T0          t at tau0 (1)");
  puts("  --dtau D         log-time step (1e-3)");
  puts("  --headless       benchmark mode");
  puts("  --stride N       render every Nth step (5)");
  puts("  --fps N          FPS cap (0 uncapped)");
  puts("  --halfblocks     high-res terminal renderer");
  puts("  --muscl          MUSCL/minmod reconstruction");
  puts("  --visc_substeps K  viscosity sub-iterations (1)");
  puts("  --colehopf       enable 1-D Cole–Hopf validation");
  puts("  --ck M           Cole–Hopf mode number (4)");
  puts("  --ca A           Cole–Hopf amplitude |A|<1 (0.5)");
  puts("  -h, --help");
}

void parse_args(int argc, char **argv, Params &P) {
  static const struct option long_opts[] = {
      {"nx", required_argument, 0, 0},
      {"ny", required_argument, 0, 0},
      {"dx", required_argument, 0, 0},
      {"dy", required_argument, 0, 0},
      {"nu", required_argument, 0, 0},
      {"u0", required_argument, 0, 0},
      {"amp", required_argument, 0, 0},
      {"bsig", required_argument, 0, 0},
      {"swirl", required_argument, 0, 0},
      {"rc", required_argument, 0, 0},
      {"offx", required_argument, 0, 0},
      {"offy", required_argument, 0, 0},
      {"asym", required_argument, 0, 0},
      {"CFL", required_argument, 0, 0},
      {"steps", required_argument, 0, 0},
      {"tau0", required_argument, 0, 0},
      {"t0", required_argument, 0, 0},
      {"dtau", required_argument, 0, 0},
      {"headless", no_argument, 0, 'H'},
      {"stride", required_argument, 0, 'r'},
      {"fps", required_argument, 0, 'f'},
      {"halfblocks", no_argument, 0, 0},
      {"muscl", no_argument, 0, 0},
      {"visc_substeps", required_argument, 0, 0},
      {"colehopf", no_argument, 0, 0},
      {"ck", required_argument, 0, 0},
      {"ca", required_argument, 0, 0},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};
  int idx = 0, c;
  while ((c = getopt_long(argc, argv, "H:r:f:h", long_opts, &idx)) != -1) {
    if (c == 'h') {
      usage(argv[0]);
      exit(0);
    }
    if (c == 'H') {
      P.headless = true;
      continue;
    }
    if (c == 'r') {
      P.stride = atoi(optarg);
      continue;
    }
    if (c == 'f') {
      P.fps_limit = atoi(optarg);
      continue;
    }
    if (c == 0) {
      const char *n = long_opts[idx].name;
      if (!strcmp(n, "nx"))
        P.nx = atoi(optarg);
      else if (!strcmp(n, "ny"))
        P.ny = atoi(optarg);
      else if (!strcmp(n, "dx"))
        P.dx = atof(optarg);
      else if (!strcmp(n, "dy"))
        P.dy = atof(optarg);
      else if (!strcmp(n, "nu"))
        P.nu = atof(optarg);
      else if (!strcmp(n, "u0"))
        P.u0 = atof(optarg);
      else if (!strcmp(n, "amp"))
        P.amp = atof(optarg);
      else if (!strcmp(n, "bsig"))
        P.bsig = atof(optarg);
      else if (!strcmp(n, "swirl"))
        P.swirl = atof(optarg);
      else if (!strcmp(n, "rc"))
        P.rc = atof(optarg);
      else if (!strcmp(n, "offx"))
        P.offx = atof(optarg);
      else if (!strcmp(n, "offy"))
        P.offy = atof(optarg);
      else if (!strcmp(n, "asym"))
        P.asym = atof(optarg);
      else if (!strcmp(n, "CFL"))
        P.CFL = atof(optarg);
      else if (!strcmp(n, "steps"))
        P.steps = atoi(optarg);
      else if (!strcmp(n, "tau0"))
        P.tau0 = atof(optarg);
      else if (!strcmp(n, "t0"))
        P.t0 = atof(optarg);
      else if (!strcmp(n, "dtau"))
        P.dtau = atof(optarg);
      else if (!strcmp(n, "halfblocks"))
        P.halfblocks = true;
      else if (!strcmp(n, "muscl"))
        P.muscl = true;
      else if (!strcmp(n, "visc_substeps"))
        P.visc_substeps = atoi(optarg);
      else if (!strcmp(n, "colehopf"))
        P.colehopf = true;
      else if (!strcmp(n, "ck"))
        P.ck = atoi(optarg);
      else if (!strcmp(n, "ca"))
        P.ca = atof(optarg);
    }
  }
}

// Host
struct HostState {
  std::vector<float> h_phi_u, h_phi_v;
  float *hp_phi_u = nullptr, *hp_phi_v = nullptr; // pinned for draw
};

void initialize_host(const Params &P, HostState &H) {
  int nx = P.nx, ny = P.ny, N = nx * ny;
  H.h_phi_u.assign(N, 0.0f);
  H.h_phi_v.assign(N, 0.0f);

  if (P.colehopf) {
    // 1-D exact-driven init: u0(x) = 2ν a k sin(kx) / (1 + a cos(kx))
    float Lx = P.dx * nx;
    float k = 2.0f * (float)M_PI * P.ck / Lx;
    for (int i = 0; i < nx; ++i) {
      float x = (i + 0.5f) * P.dx;
      float denom = 1.0f + P.ca * cosf(k * x);
      float u = (denom != 0.0f) ? (2.0f * P.nu * P.ca * k * sinf(k * x) / denom)
                                : 0.0f;
      float phi = asinhf(u / P.u0);
      for (int j = 0; j < ny; ++j) {
        int id = ID(i, j, nx, ny);
        H.h_phi_u[id] = phi;
        H.h_phi_v[id] = 0.0f; // pure 1-D
      }
    }
    return;
  }

  // 2-D swirl + Gaussian
  float cx = 0.5f * nx + P.offx, cy = 0.5f * ny + P.offy;
  float sig2 = P.bsig * P.bsig;
  float rc = P.rc * fminf(P.dx, P.dy);
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      float dx = i - cx, dy = j - cy;
      float r2 = (dx * dx + dy * dy) / fmaxf(sig2, 1e-6f);
      float theta = atan2f(dy, dx);
      float mod = 1.0f + P.asym * cosf(theta);

      float rx = dx * P.dx, ry = dy * P.dy;
      float r = sqrtf(rx * rx + ry * ry);
      float u_theta =
          (r > 0.0f) ? (P.swirl * r * expf(-0.5f * (r / rc) * (r / rc))) : 0.0f;
      float u = (r > 0.0f) ? (-u_theta * (ry / r)) : 0.0f;
      float v = (r > 0.0f) ? (u_theta * (rx / r)) : 0.0f;

      float g = P.amp * mod * expf(-0.5f * r2);
      u += 0.5f * g;
      v += -0.5f * g;

      int id = ID(i, j, nx, ny);
      H.h_phi_u[id] = asinhf(u / P.u0);
      H.h_phi_v[id] = asinhf(v / P.u0);
    }
  }
}

// Device
struct DeviceState {
  float *d_phi_u = nullptr, *d_phi_v = nullptr;
  float *d_Fu_x = nullptr, *d_Fv_x = nullptr;
  float *d_Gu_y = nullptr, *d_Gv_y = nullptr;
  float *d_block_max = nullptr;
};

void device_alloc(DeviceState &D, int N) {
  CUDA_CHECK(cudaMalloc(&D.d_phi_u, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&D.d_phi_v, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&D.d_Fu_x, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&D.d_Fv_x, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&D.d_Gu_y, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&D.d_Gv_y, N * sizeof(float)));
}

void device_free(DeviceState &D) {
  cudaFree(D.d_phi_u);
  cudaFree(D.d_phi_v);
  cudaFree(D.d_Fu_x);
  cudaFree(D.d_Fv_x);
  cudaFree(D.d_Gu_y);
  cudaFree(D.d_Gv_y);
  cudaFree(D.d_block_max);
}

// Kernels
__device__ inline float minmod(float a, float b) {
  return (a * b <= 0.0f) ? 0.0f : copysignf(fminf(fabsf(a), fabsf(b)), a);
}

// CFL driver: max(|u|/dx + |v|/dy)
__global__ void wavespeed_block_max(const float *__restrict__ phi_u,
                                    const float *__restrict__ phi_v, float u0,
                                    int nx, int ny, float invdx, float invdy,
                                    float *__restrict__ block_max) {
  extern __shared__ float sdata[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  float val = 0.0f;
  if (i < nx && j < ny) {
    int id = ID(i, j, nx, ny);
    float u = u0 * sinhf(phi_u[id]), v = u0 * sinhf(phi_v[id]);
    val = fabsf(u) * invdx + fabsf(v) * invdy;
  }
  sdata[tid] = val;
  __syncthreads();
  int sz = blockDim.x * blockDim.y;
  for (int s = sz >> 1; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
  if (tid == 0)
    block_max[blockIdx.y * gridDim.x + blockIdx.x] = sdata[0];
}

// Rusanov x-face with optional MUSCL on phi_u, phi_v
__global__ void flux_x_kernel(const float *__restrict__ phi_u,
                              const float *__restrict__ phi_v,
                              float *__restrict__ Fu_x,
                              float *__restrict__ Fv_x, int nx, int ny,
                              float u0, int use_muscl) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= nx || j >= ny)
    return;
  int iL = i, iR = wrap(i + 1, nx);
  int idL = ID(iL, j, nx, ny), idR = ID(iR, j, nx, ny);

  float pUL = phi_u[idL], pUR = phi_u[idR];
  float pVL = phi_v[idL], pVR = phi_v[idR];

  if (use_muscl) {
    int iLm = wrap(i - 1, nx), iRp = wrap(i + 2, nx);
    float pU_Lm = phi_u[ID(iLm, j, nx, ny)];
    float pU_Rp = phi_u[ID(iRp, j, nx, ny)];
    float pV_Lm = phi_v[ID(iLm, j, nx, ny)];
    float pV_Rp = phi_v[ID(iRp, j, nx, ny)];

    float sUL = 0.5f * minmod(pUL - pU_Lm, pUR - pUL);
    float sUR = 0.5f * minmod(pU_Rp - pUR, pUR - pUL);
    float sVL = 0.5f * minmod(pVL - pV_Lm, pVR - pVL);
    float sVR = 0.5f * minmod(pV_Rp - pVR, pVR - pVL);

    pUL = pUL + sUL; // left cell right face
    pUR = pUR - sUR; // right cell left face
    pVL = pVL + sVL;
    pVR = pVR - sVR;
  }

  float uL = u0 * sinhf(pUL), vL = u0 * sinhf(pVL);
  float uR = u0 * sinhf(pUR), vR = u0 * sinhf(pVR);

  float FL_u = 0.5f * uL * uL;
  float FL_v = uL * vL;
  float FR_u = 0.5f * uR * uR;
  float FR_v = uR * vR;
  float a = fmaxf(fabsf(uL), fabsf(uR));

  Fu_x[IDXF(i, j, nx, ny)] = 0.5f * (FL_u + FR_u) - 0.5f * a * (uR - uL);
  Fv_x[IDXF(i, j, nx, ny)] = 0.5f * (FL_v + FR_v) - 0.5f * a * (vR - vL);
}

// Rusanov y-face with optional MUSCL
__global__ void flux_y_kernel(const float *__restrict__ phi_u,
                              const float *__restrict__ phi_v,
                              float *__restrict__ Gu_y,
                              float *__restrict__ Gv_y, int nx, int ny,
                              float u0, int use_muscl) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= nx || j >= ny)
    return;
  int jB = j, jT = wrap(j + 1, ny);
  int idB = ID(i, jB, nx, ny), idT = ID(i, jT, nx, ny);

  float pUB = phi_u[idB], pUT = phi_u[idT];
  float pVB = phi_v[idB], pVT = phi_v[idT];

  if (use_muscl) {
    int jBm = wrap(j - 1, ny), jTp = wrap(j + 2, ny);
    float pU_Bm = phi_u[ID(i, jBm, nx, ny)];
    float pU_Tp = phi_u[ID(i, jTp, nx, ny)];
    float pV_Bm = phi_v[ID(i, jBm, nx, ny)];
    float pV_Tp = phi_v[ID(i, jTp, nx, ny)];

    float sUB = 0.5f * minmod(pUB - pU_Bm, pUT - pUB);
    float sUT = 0.5f * minmod(pU_Tp - pUT, pUT - pUB);
    float sVB = 0.5f * minmod(pVB - pV_Bm, pVT - pVB);
    float sVT = 0.5f * minmod(pV_Tp - pVT, pVT - pVB);

    pUB = pUB + sUB;
    pUT = pUT - sUT;
    pVB = pVB + sVB;
    pVT = pVT - sVT;
  }

  float uB = u0 * sinhf(pUB), vB = u0 * sinhf(pVB);
  float uT = u0 * sinhf(pUT), vT = u0 * sinhf(pVT);

  float GL_u = uB * vB;
  float GL_v = 0.5f * vB * vB;
  float GR_u = uT * vT;
  float GR_v = 0.5f * vT * vT;
  float a = fmaxf(fabsf(vB), fabsf(vT));

  Gu_y[IDYF(i, j, nx, ny)] = 0.5f * (GL_u + GR_u) - 0.5f * a * (uT - uB);
  Gv_y[IDYF(i, j, nx, ny)] = 0.5f * (GL_v + GR_v) - 0.5f * a * (vT - vB);
}

// Convection
__global__ void update_convective(
    float *__restrict__ phi_u, float *__restrict__ phi_v,
    const float *__restrict__ Fu_x, const float *__restrict__ Fv_x,
    const float *__restrict__ Gu_y, const float *__restrict__ Gv_y, int nx,
    int ny, float dx, float dy, float dt, float u0, int oneD) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= nx || j >= ny)
    return;

  int id = ID(i, j, nx, ny);
  int im = wrap(i - 1, nx), jm = wrap(j - 1, ny);

  float u = u0 * sinhf(phi_u[id]);
  float v = u0 * sinhf(phi_v[id]);

  float invdx = 1.0f / dx, invdy = (oneD ? 0.0f : (1.0f / dy));
  float dFx_u = Fu_x[IDXF(i, j, nx, ny)] - Fu_x[IDXF(im, j, nx, ny)];
  float dFx_v = Fv_x[IDXF(i, j, nx, ny)] - Fv_x[IDXF(im, j, nx, ny)];
  float dGy_u =
      oneD ? 0.0f : (Gu_y[IDYF(i, j, nx, ny)] - Gu_y[IDYF(i, jm, nx, ny)]);
  float dGy_v =
      oneD ? 0.0f : (Gv_y[IDYF(i, j, nx, ny)] - Gv_y[IDYF(i, jm, nx, ny)]);

  u -= dt * (dFx_u * invdx + dGy_u * invdy);
  v -= dt * (dFx_v * invdx + dGy_v * invdy);

  phi_u[id] = asinhf(u / u0);
  phi_v[id] = asinhf(v / u0);
}

// Viscosity
__global__ void viscosity_step(float *__restrict__ phi_u,
                               float *__restrict__ phi_v, int nx, int ny,
                               float dx, float dy, float nu, float dt, float u0,
                               int oneD) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= nx || j >= ny)
    return;

  int id = ID(i, j, nx, ny);
  float invdx2 = 1.0f / (dx * dx);
  float invdy2 = oneD ? 0.0f : (1.0f / (dy * dy));

  auto U = [&](int ii, int jj) {
    int k = ID(wrap(ii, nx), wrap(jj, ny), nx, ny);
    return make_float2(u0 * sinhf(phi_u[k]), u0 * sinhf(phi_v[k]));
  };

  float2 c = U(i, j);
  float2 xp = U(i + 1, j);
  float2 xm = U(i - 1, j);
  float2 yp = U(i, j + 1);
  float2 ym = U(i, j - 1);

  float2 lap;
  lap.x =
      (xp.x - 2.0f * c.x + xm.x) * invdx2 + (yp.x - 2.0f * c.x + ym.x) * invdy2;
  lap.y =
      (xp.y - 2.0f * c.y + xm.y) * invdx2 + (yp.y - 2.0f * c.y + ym.y) * invdy2;

  float u = c.x + nu * dt * lap.x;
  float v = c.y + nu * dt * lap.y;

  phi_u[id] = asinhf(u / u0);
  phi_v[id] = asinhf(v / u0);
}

// Display
static const wchar_t WRAMP[] = {L' ', L'▁', L'▂', L'▃', L'▄',
                                L'▅', L'▆', L'▇', L'█'};
static const int WRAMP_L = (int)(sizeof(WRAMP) / sizeof(WRAMP[0])) - 1;

struct RendererCtx {
  bool halfblocks = false;
};

static inline uint32_t hashpx(int i, int j) {
  uint32_t h = (uint32_t)(1469598103u ^ (j * 1315423911u + i * 2654435761u));
  h ^= h >> 13;
  h *= 0x5bd1e995u;
  h ^= h >> 15;
  return h;
}

void show_ncurses(int step, double fps, double avgfps, const float *phi_u,
                  const float *phi_v, const Params &P, float t, float tau,
                  double relL2 = -1.0) {
  auto speedAt = [&](double x, double y) {
    auto sample_bilinear_phi = [&](const float *A, double x, double y) {
      int nx = P.nx, ny = P.ny;
      int i0 = (int)floor(x), j0 = (int)floor(y);
      int i1 = wrap(i0 + 1, nx), j1 = wrap(j0 + 1, ny);
      double tx = x - i0, ty = y - j0;
      i0 = wrap(i0, nx);
      j0 = wrap(j0, ny);
      double a00 = A[ID(i0, j0, nx, ny)], a10 = A[ID(i1, j0, nx, ny)];
      double a01 = A[ID(i0, j1, nx, ny)], a11 = A[ID(i1, j1, nx, ny)];
      double ax0 = a00 * (1.0 - tx) + a10 * tx,
             ax1 = a01 * (1.0 - tx) + a11 * tx;
      return (float)(ax0 * (1.0 - ty) + ax1 * ty);
    };
    float phiu = sample_bilinear_phi(phi_u, x, y);
    float phiv = sample_bilinear_phi(phi_v, x, y);
    double u = P.u0 * sinh(phiu), v = P.u0 * sinh(phiv);
    return hypot(u, v);
  };

  int rows, cols;
  getmaxyx(stdscr, rows, cols);
  if (P.colehopf && relL2 >= 0.0) {
    mvprintw(0, 0,
             "step %d  FPS: %.1f (avg %.1f)  t=%.3g tau=%.6f dtau=%.2e  "
             "nu=%.3g u0=%.2g  L2rel=%.3e  [q]",
             step, fps, avgfps, t, tau, P.dtau, P.nu, P.u0, relL2);
  } else {
    mvprintw(0, 0,
             "step %d  FPS: %.1f (avg %.1f)  t=%.3g tau=%.6f dtau=%.2e  "
             "nu=%.3g u0=%.2g  [q]",
             step, fps, avgfps, t, tau, P.dtau, P.nu, P.u0);
  }

  int disp_rows = rows - 1, disp_cols = cols;
  if (disp_rows <= 0 || disp_cols <= 0) {
    refresh();
    return;
  }

  int N = P.nx * P.ny;
  double sum = 0.0, sum2 = 0.0;
  for (int k = 0; k < N; ++k) {
    double u = P.u0 * sinh(phi_u[k]), v = P.u0 * sinh(phi_v[k]);
    double s = hypot(u, v);
    sum += s;
    sum2 += s * s;
  }
  double mean = sum / N;
  double var = fmax(0.0, sum2 / N - mean * mean);
  double sd = sqrt(var);
  double lo = mean - 2.0 * sd, hi = mean + 2.0 * sd;
  double inv = (hi > lo) ? 1.0 / (hi - lo) : 1.0;

  if (!P.halfblocks) {
    for (int j = 0; j < disp_rows; ++j) {
      double y = ((j + 0.5) * P.ny) / (double)disp_rows;
      move(j + 1, 0);
      for (int i = 0; i < disp_cols; ++i) {
        double x = ((i + 0.5) * P.nx) / (double)disp_cols;
        double s = speedAt(x, y);
        uint32_t h = hashpx(i, j);
        double dither =
            ((h & 0xFFFF) / 65535.0 - 0.5) * (0.5 / (double)WRAMP_L);
        double val = (s - lo) * inv + dither;
        if (val < 0)
          val = 0;
        if (val > 1)
          val = 1;
        int idx = (int)(val * WRAMP_L + 0.5);
        if (idx < 0)
          idx = 0;
        if (idx > WRAMP_L)
          idx = WRAMP_L;
        addnwstr(&WRAMP[idx], 1);
      }
      clrtoeol();
    }
    refresh();
    return;
  }

  for (int j = 0; j < disp_rows; ++j) {
    double y0 = ((2 * j + 0.25) * P.ny) / (double)(2 * disp_rows);
    double y1 = ((2 * j + 1.25) * P.ny) / (double)(2 * disp_rows);
    move(j + 1, 0);
    for (int i = 0; i < disp_cols; ++i) {
      double x = ((i + 0.5) * P.nx) / (double)disp_cols;
      auto tone = [&](double s) {
        double val = (s - lo) * inv;
        if (val < 0)
          val = 0;
        if (val > 1)
          val = 1;
        return (int)(val * WRAMP_L + 0.5);
      };
      double s0 = speedAt(x, y0), s1 = speedAt(x, y1);
      int t0 = tone(s0), t1 = tone(s1);
      wchar_t ch = (t0 >= t1) ? L'▀' : L'▄';
      addnwstr(&ch, 1);
    }
    clrtoeol();
  }
  refresh();
}

int main(int argc, char **argv) {
  Params P;
  parse_args(argc, argv, P);
  if (P.colehopf)
    P.ny = 1;

  HostState H;
  initialize_host(P, H);
  int nx = P.nx, ny = P.ny, N = nx * ny;

  DeviceState D;
  device_alloc(D, N);
  CUDA_CHECK(cudaMemcpy(D.d_phi_u, H.h_phi_u.data(), N * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(D.d_phi_v, H.h_phi_v.data(), N * sizeof(float),
                        cudaMemcpyHostToDevice));

  dim3 bs(16, 16), gs((nx + bs.x - 1) / bs.x, (ny + bs.y - 1) / bs.y);
  CUDA_CHECK(cudaMalloc(&D.d_block_max, gs.x * gs.y * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&H.hp_phi_u, N * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&H.hp_phi_v, N * sizeof(float)));

  float tau = P.tau0, t = P.t0, dtau = P.dtau;

  auto do_step = [&](float &dt_eff) {
    // CFL
    size_t shmem = bs.x * bs.y * sizeof(float);
    wavespeed_block_max<<<gs, bs, shmem>>>(
        D.d_phi_u, D.d_phi_v, P.u0, nx, ny, 1.0f / P.dx,
        (ny > 1 ? 1.0f / P.dy : 0.0f), D.d_block_max);
    CUDA_CHECK(cudaPeekAtLastError());
    std::vector<float> h_blk(gs.x * gs.y);
    CUDA_CHECK(cudaMemcpy(h_blk.data(), D.d_block_max,
                          h_blk.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float smax = 1e-12f;
    for (float v : h_blk)
      smax = fmaxf(smax, v);
    float dt_cfl = P.CFL / smax;
    dt_eff = fminf(t * dtau, dt_cfl);

    // Fluxes
    flux_x_kernel<<<gs, bs>>>(D.d_phi_u, D.d_phi_v, D.d_Fu_x, D.d_Fv_x, nx, ny,
                              P.u0, P.muscl ? 1 : 0);
    CUDA_CHECK(cudaPeekAtLastError());
    if (!P.colehopf) { // skip y-flux in 1-D
      flux_y_kernel<<<gs, bs>>>(D.d_phi_u, D.d_phi_v, D.d_Gu_y, D.d_Gv_y, nx,
                                ny, P.u0, P.muscl ? 1 : 0);
      CUDA_CHECK(cudaPeekAtLastError());
    }

    // Convection
    update_convective<<<gs, bs>>>(D.d_phi_u, D.d_phi_v, D.d_Fu_x, D.d_Fv_x,
                                  D.d_Gu_y, D.d_Gv_y, nx, ny, P.dx, P.dy,
                                  dt_eff, P.u0, P.colehopf ? 1 : 0);
    CUDA_CHECK(cudaPeekAtLastError());

    // Viscosity
    int K = (P.visc_substeps > 0 ? P.visc_substeps : 1);
    float sub = dt_eff / K;
    for (int k = 0; k < K; ++k) {
      viscosity_step<<<gs, bs>>>(D.d_phi_u, D.d_phi_v, nx, ny, P.dx, P.dy, P.nu,
                                 sub, P.u0, P.colehopf ? 1 : 0);
      CUDA_CHECK(cudaPeekAtLastError());
    }
  };

  auto colehopf_relL2 = [&](const float *host_phi_u, float t_now) -> double {
    // Compute relative L2 error vs exact 1-D solution
    float Lx = P.dx * nx;
    float k = 2.0f * (float)M_PI * P.ck / Lx;
    float decay = expf(-P.nu * k * k * t_now);
    double num = 0.0, den = 0.0;
    for (int i = 0; i < nx; ++i) {
      float x = (i + 0.5f) * P.dx;
      float u_ex = (2.0f * P.nu * P.ca * k * decay * sinf(k * x)) /
                   (1.0f + P.ca * decay * cosf(k * x));
      double u_num = P.u0 * sinh(host_phi_u[ID(i, 0, nx, ny)]);
      double diff = u_num - u_ex;
      num += diff * diff;
      den += u_ex * u_ex;
    }
    return (den > 0.0) ? sqrt(num / den) : sqrt(num);
  };

  if (!P.headless) {
    curses_active = true;
    setlocale(LC_ALL, "");
    initscr();
    cbreak();
    noecho();
    curs_set(0);
    nodelay(stdscr, TRUE);
    keypad(stdscr, TRUE);

    int step = 0;
    double avgfps = 0.0, fps = 0.0;
    auto last = std::chrono::high_resolution_clock::now();
    while (P.steps == 0 || step < P.steps) {
      if (getch() == 'q')
        break;
      float dt_eff = 0.0f;
      do_step(dt_eff);
      tau += dtau;
      t *= expf(dtau);

      if (step % P.stride == 0) {
        auto now = std::chrono::high_resolution_clock::now();
        double dtw = std::chrono::duration<double>(now - last).count();
        fps = (dtw > 0) ? 1.0 / dtw : 0.0;
        avgfps = 0.95 * avgfps + 0.05 * fps;
        last = now;

        CUDA_CHECK(cudaMemcpy(H.hp_phi_u, D.d_phi_u, N * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(H.hp_phi_v, D.d_phi_v, N * sizeof(float),
                              cudaMemcpyDeviceToHost));
        double relL2 = -1.0;
        if (P.colehopf)
          relL2 = colehopf_relL2(H.hp_phi_u, t);

        show_ncurses(step, fps, avgfps, H.hp_phi_u, H.hp_phi_v, P, t, tau,
                     relL2);

        if (P.fps_limit > 0) {
          double target = 1.0 / P.fps_limit;
          if (dtw < target) {
            int us = (int)((target - dtw) * 1e6);
            if (us > 0)
              usleep(us);
          }
        }
      }
      step++;
    }
    endwin();
    curses_active = false;
  } else {
    using clock = std::chrono::high_resolution_clock;
    auto start = clock::now();
    cudaEvent_t evs, eve;
    CUDA_CHECK(cudaEventCreate(&evs));
    CUDA_CHECK(cudaEventCreate(&eve));
    CUDA_CHECK(cudaEventRecord(evs));
    int frames = 0;
    for (int step = 0; step < (P.steps ? P.steps : 2000); ++step) {
      float dt_eff = 0.0f;
      do_step(dt_eff);
      tau += dtau;
      t *= expf(dtau);
      if (step % P.stride == 0)
        frames++;
    }
    CUDA_CHECK(cudaEventRecord(eve));
    CUDA_CHECK(cudaEventSynchronize(eve));
    auto end = clock::now();
    float ms_gpu = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_gpu, evs, eve));
    double secs = std::chrono::duration<double>(end - start).count();
    printf("Headless (stride=%d):\n", P.stride);
    printf("  Steps: %d\n", (P.steps ? P.steps : 2000));
    printf("  Wall:  %d frames in %.3f s -> %.1f FPS\n", frames, secs,
           (frames > 0 ? frames / secs : 0.0));
    printf("  GPU:   %d frames in %.3f s -> %.1f FPS\n", frames,
           ms_gpu / 1000.0, (ms_gpu > 0 ? 1000.0 * frames / ms_gpu : 0.0));
    CUDA_CHECK(cudaEventDestroy(evs));
    CUDA_CHECK(cudaEventDestroy(eve));
  }

  device_free(D);
  if (H.hp_phi_u)
    cudaFreeHost(H.hp_phi_u);
  if (H.hp_phi_v)
    cudaFreeHost(H.hp_phi_v);
  return 0;
}
