// tau_shallow_water_ncurses.cu
// 2-D shallow-water in log-time (tau) with log-depth (sigma = ln h)
// Flux-form finite volume with HLL Riemann solver (positivity-preserving),
// adaptive dt via CFL on the τ-clock (dt_eff = min(t*dtau, CFL*dx/cmax)),
// optional explicit viscosity on momentum, and ncurses rendering ala
// js_cuda.cu.
//
//  - State stores sigma = ln h (positivity guaranteed)
//  - Time advances in τ: t = t0 * exp(τ); dt_eff = t * dτ but clamped by CFL
//  - Fluxes computed conservatively (h, hu, hv) from exp(sigma) and u,v
//  - After update, we map back to logs (sigma = ln h_new) and real u,v
//
// Build:
//   nvcc -std=c++17 -O3 -use_fast_math -arch=sm_86 -lineinfo -o tau_sw
//   tau_shallow_water_ncurses.cu -lncursesw
//
// Run:
//   ./tau_sw --nx 256 --ny 256 --dx 1000 --dy 1000 --steps 0 --dtau 1e-3 --f0
//   1e-4 --nu 50 --stride 2 --fps 60
//   ./tau_sw --headless --steps 2000 --stride 4

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
  int nx = 512;
  int ny = 512;

  float dx = 1.0f; // meters
  float dy = 1.0f; // meters

  float g = 9.81f;    // m/s^2
  float f0 = 1.0f;    // Coriolis (f-plane)
  float nu = 0.001f;  // optional explicit viscosity on u,v [m^2/s]
  float H0 = 1000.0f; // mean depth [m]

  // initial impulse
  float bumpAmp = 1.0f;   // initial gaussian bump amplitude [m]
  float bumpSigma = 1.0f; // stdev in grid cells
  float CFL = 0.5f;       // CFL number for HLL update

  float offx = 100.0f; // center shift in x (cells)
  float offy = 100.0f; // center shift in y (cells)
  float asym = 10.0f;  // dipole modulation amplitude

  // swirl parameters
  float swirl = 1.0f;     // angular speed Ω [1/s]
  float swirlRc = 100.0f; // core radius in CELLS (Gaussian peak)

  // time (log-time clock)
  int steps = 0; // 0 = run forever
  float tau0 = 0.0f;
  float t0 = 1.0f;   // seconds at tau0
  float dtau = 1.0f; // log-time step

  // UI/benchmark
  bool headless = false;
  int stride = 5;    // render every Nth step
  int fps_limit = 0; // 0 = uncapped
};

// indexing
__host__ __device__ inline int wrap(int i, int n) {
  i %= n;
  if (i < 0)
    i += n;
  return i;
}

__host__ __device__ inline int ID(int i, int j, int nx, int ny) {
  return wrap(j, ny) * nx + wrap(i, nx);
}

// face indexers
__host__ __device__ inline int IDXF(int i, int j, int nx, int ny) {
  return wrap(j, ny) * nx + wrap(i, nx);
} // x-faces: between i and i+1

__host__ __device__ inline int IDYF(int i, int j, int nx, int ny) {
  return wrap(j, ny) * nx + wrap(i, nx);
} // y-faces: between j and j+1

// state
struct HostState {
  std::vector<float> h_sigma, h_u, h_v; // stored in log for h, real for u,v
  float *hp_sigma_pinned = nullptr;     // for drawing
};

void usage(const char *prog) {
  printf("Usage: %s [options]\n", prog);
  puts("  --nx N        grid cells in x (256)");
  puts("  --ny N        grid cells in y (256)");
  puts("  --dx M        cell size x meters (1000)");
  puts("  --dy M        cell size y meters (1000)");
  puts("  --g G         gravity (9.81)");
  puts("  --f0 F        coriolis f (0 or 1e-4)");
  puts("  --nu NU       eddy viscosity on u,v m^2/s (0)");
  puts("  --H0 H        mean depth (1000)");
  puts("  --amp A       initial Gaussian bump amplitude (1)");
  puts("  --bsig S      bump sigma in cells (2)");
  puts("  --CFL C       CFL number (0.45)");
  puts("  --steps K     number of steps (0 = forever)");
  puts("  --tau0 T      initial log-time (0)");
  puts("  --t0 T0       physical seconds at tau0 (1)");
  puts("  --dtau D      log-time step (1e-3)");
  puts("  --headless    run without UI (benchmark)");
  puts("  --stride N    render every Nth step (1)");
  puts("  --fps N       limit FPS to N (0 = uncapped)");
  puts("  --offx X      bump center x shift in cells (0)");
  puts("  --offy Y      bump center y shift in cells (0)");
  puts("  --asym A      small dipole modulation (0.0)");
  puts("  --swirl O     angular speed [1/s] for initial vortex (0)");
  puts("  --rc R        core radius in cells for vortex (5)");
  puts("  -h, --help    show this help");
}

void parse_args(int argc, char **argv, Params &P) {
  static const struct option long_opts[] = {
      {"nx", required_argument, 0, 0},
      {"ny", required_argument, 0, 0},
      {"dx", required_argument, 0, 0},
      {"dy", required_argument, 0, 0},
      {"g", required_argument, 0, 0},
      {"f0", required_argument, 0, 0},
      {"nu", required_argument, 0, 0},
      {"H0", required_argument, 0, 0},
      {"amp", required_argument, 0, 0},
      {"bsig", required_argument, 0, 0},
      {"CFL", required_argument, 0, 0},
      {"steps", required_argument, 0, 0},
      {"tau0", required_argument, 0, 0},
      {"t0", required_argument, 0, 0},
      {"dtau", required_argument, 0, 0},
      {"headless", no_argument, 0, 'H'},
      {"stride", required_argument, 0, 'r'},
      {"fps", required_argument, 0, 'f'},
      {"offx", required_argument, 0, 0},
      {"offy", required_argument, 0, 0},
      {"asym", required_argument, 0, 0},
      {"swirl", required_argument, 0, 0},
      {"rc", required_argument, 0, 0},
      {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};
  int idx = 0;
  int c;
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
      const char *name = long_opts[idx].name;
      if (!strcmp(name, "nx"))
        P.nx = atoi(optarg);
      else if (!strcmp(name, "ny"))
        P.ny = atoi(optarg);
      else if (!strcmp(name, "dx"))
        P.dx = atof(optarg);
      else if (!strcmp(name, "dy"))
        P.dy = atof(optarg);
      else if (!strcmp(name, "g"))
        P.g = atof(optarg);
      else if (!strcmp(name, "f0"))
        P.f0 = atof(optarg);
      else if (!strcmp(name, "nu"))
        P.nu = atof(optarg);
      else if (!strcmp(name, "H0"))
        P.H0 = atof(optarg);
      else if (!strcmp(name, "amp"))
        P.bumpAmp = atof(optarg);
      else if (!strcmp(name, "bsig"))
        P.bumpSigma = atof(optarg);
      else if (!strcmp(name, "CFL"))
        P.CFL = atof(optarg);
      else if (!strcmp(name, "steps"))
        P.steps = atoi(optarg);
      else if (!strcmp(name, "tau0"))
        P.tau0 = atof(optarg);
      else if (!strcmp(name, "t0"))
        P.t0 = atof(optarg);
      else if (!strcmp(name, "dtau"))
        P.dtau = atof(optarg);
      else if (!strcmp(name, "offx"))
        P.offx = atof(optarg);
      else if (!strcmp(name, "offy"))
        P.offy = atof(optarg);
      else if (!strcmp(name, "asym"))
        P.asym = atof(optarg);
      else if (!strcmp(name, "swirl"))
        P.swirl = atof(optarg);
      else if (!strcmp(name, "rc"))
        P.swirlRc = atof(optarg);
    }
  }
}

void initialize_host(const Params &P, HostState &H) {
  int nx = P.nx, ny = P.ny, N = nx * ny;
  H.h_sigma.assign(N, 0);
  H.h_u.assign(N, 0);
  H.h_v.assign(N, 0);
  float cx = 0.5f * nx + P.offx;
  float cy = 0.5f * ny + P.offy;
  float sig2 = P.bumpSigma * P.bumpSigma;
  for (int j = 0; j < ny; ++j) {
    for (int i = 0; i < nx; ++i) {
      float dx = i - cx, dy = j - cy;
      float r2 = (dx * dx + dy * dy) / sig2;

      // small dipole (m=1) modulation: stronger bump on one side
      float theta = atan2f(dy, dx); // [-pi, pi]
      float mod = 1.0f + P.asym * cosf(theta);

      float h = P.H0 + (P.bumpAmp * mod) * expf(-0.5f * r2);

      H.h_sigma[ID(i, j, nx, ny)] = logf(fmaxf(h, 1e-6f));

      // swirl
      float rx = dx * P.dx, ry = dy * P.dy;
      float r = sqrtf(rx * rx + ry * ry);
      float rc = P.swirlRc * fminf(P.dx, P.dy);

      float u_theta = (r > 0.0f && P.swirl != 0.0f)
                          ? (P.swirl * r * expf(-0.5f * (r / rc) * (r / rc)))
                          : 0.0f;

      // (-sinθ, cosθ) = (-ry/r, rx/r)
      float u = (r > 0.0f) ? (-u_theta * (ry / r)) : 0.0f;
      float v = (r > 0.0f) ? (u_theta * (rx / r)) : 0.0f;

      H.h_u[ID(i, j, nx, ny)] = u;
      H.h_v[ID(i, j, nx, ny)] = v;
    }
  }
}

// device
struct DeviceState {
  float *d_sigma = nullptr, *d_u = nullptr,
        *d_v = nullptr; // cell vars (log h, u, v)
  float *d_Fh_x = nullptr, *d_Fmx_x = nullptr,
        *d_Fmy_x = nullptr; // x-face fluxes
  float *d_Gh_y = nullptr, *d_Gmx_y = nullptr,
        *d_Gmy_y = nullptr;      // y-face fluxes
  float *d_block_cmax = nullptr; // per-block max wavespeed
};

void device_alloc(DeviceState &D, int N) {
  CUDA_CHECK(cudaMalloc(&D.d_sigma, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&D.d_u, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&D.d_v, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&D.d_Fh_x, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&D.d_Fmx_x, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&D.d_Fmy_x, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&D.d_Gh_y, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&D.d_Gmx_y, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&D.d_Gmy_y, N * sizeof(float)));
}

void device_free(DeviceState &D) {
  cudaFree(D.d_sigma);
  cudaFree(D.d_u);
  cudaFree(D.d_v);
  cudaFree(D.d_Fh_x);
  cudaFree(D.d_Fmx_x);
  cudaFree(D.d_Fmy_x);
  cudaFree(D.d_Gh_y);
  cudaFree(D.d_Gmx_y);
  cudaFree(D.d_Gmy_y);
  cudaFree(D.d_block_cmax);
}

// physics
__device__ inline void cons_from_logs(float sig, float u, float v, float g,
                                      float &h, float &mx, float &my, float &c,
                                      float &c_y) {
  h = expf(sig);
  float rtgh = sqrtf(g * h);
  mx = h * u;
  my = h * v;
  c = fabsf(u) + rtgh;
  c_y = fabsf(v) + rtgh;
}

// HLL flux for shallow water in x-direction
__device__ inline void hll_x(float hL, float uL, float vL, float hR, float uR,
                             float vR, float g, float &Fh, float &Fmx,
                             float &Fmy) {
  float cL = sqrtf(g * hL), cR = sqrtf(g * hR);
  float sL = fminf(uL - cL, uR - cR);
  float sR = fmaxf(uL + cL, uR + cR);
  float mL = hL * uL, mR = hR * uR;
  float nL = hL * vL, nR = hR * vR;
  float FL_h = mL;
  float FL_mx = mL * uL + 0.5f * g * hL * hL;
  float FL_my = mL * vL;
  float FR_h = mR;
  float FR_mx = mR * uR + 0.5f * g * hR * hR;
  float FR_my = mR * vR;
  if (sL >= 0.0f) {
    Fh = FL_h;
    Fmx = FL_mx;
    Fmy = FL_my;
    return;
  }
  if (sR <= 0.0f) {
    Fh = FR_h;
    Fmx = FR_mx;
    Fmy = FR_my;
    return;
  }
  float inv = 1.0f / (sR - sL);
  Fh = (sR * FL_h - sL * FR_h + sR * sL * (hR - hL)) * inv;
  Fmx = (sR * FL_mx - sL * FR_mx + sR * sL * (mR - mL)) * inv;
  Fmy = (sR * FL_my - sL * FR_my + sR * sL * (nR - nL)) * inv;
}

// HLL flux for shallow water in y-direction
__device__ inline void hll_y(float hB, float uB, float vB, float hT, float uT,
                             float vT, float g, float &Gh, float &Gmx,
                             float &Gmy) {
  float cB = sqrtf(g * hB), cT = sqrtf(g * hT);
  float sB = fminf(vB - cB, vT - cT);
  float sT = fmaxf(vB + cB, vT + cT);
  float mB = hB * uB, mT = hT * uT;
  float nB = hB * vB, nT = hT * vT;
  float GL_h = nB;
  float GL_mx = mB * vB;
  float GL_my = nB * vB + 0.5f * g * hB * hB;
  float GR_h = nT;
  float GR_mx = mT * vT;
  float GR_my = nT * vT + 0.5f * g * hT * hT;
  if (sB >= 0.0f) {
    Gh = GL_h;
    Gmx = GL_mx;
    Gmy = GL_my;
    return;
  }
  if (sT <= 0.0f) {
    Gh = GR_h;
    Gmx = GR_mx;
    Gmy = GR_my;
    return;
  }
  float inv = 1.0f / (sT - sB);
  Gh = (sT * GL_h - sB * GR_h + sT * sB * (hT - hB)) * inv;
  Gmx = (sT * GL_mx - sB * GR_mx + sT * sB * (mT - mB)) * inv;
  Gmy = (sT * GL_my - sB * GR_my + sT * sB * (nT - nB)) * inv;
}

// kernels
// Compute per-block max wavespeed (for CFL)
__global__ void wavespeed_block_max(const float *__restrict__ sigma,
                                    const float *__restrict__ u,
                                    const float *__restrict__ v, float g,
                                    int nx, int ny,
                                    float *__restrict__ block_max) {
  extern __shared__ float sdata[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  int tid = threadIdx.y * blockDim.x + threadIdx.x;
  float val = 0.0f;
  if (i < nx && j < ny) {
    float h = expf(sigma[ID(i, j, nx, ny)]);
    float c = sqrtf(g * h);
    float uu = fabsf(u[ID(i, j, nx, ny)]), vv = fabsf(v[ID(i, j, nx, ny)]);
    val = fmaxf(uu + c, vv + c);
  }
  sdata[tid] = val;
  __syncthreads();
  // reduce to max
  int sz = blockDim.x * blockDim.y;
  for (int s = sz >> 1; s > 0; s >>= 1) {
    if (tid < s)
      sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
  if (tid == 0) {
    block_max[blockIdx.y * gridDim.x + blockIdx.x] = sdata[0];
  }
}

// Compute HLL fluxes on x faces (i+1/2,j)
__global__ void
flux_x_kernel(const float *__restrict__ sigma, const float *__restrict__ u,
              const float *__restrict__ v, float *__restrict__ Fh_x,
              float *__restrict__ Fmx_x, float *__restrict__ Fmy_x, int nx,
              int ny, float g) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= nx || j >= ny)
    return;
  int iR = wrap(i + 1, nx);
  // reconstruct states
  float hL = expf(sigma[ID(i, j, nx, ny)]);
  float uL = u[ID(i, j, nx, ny)];
  float vL = v[ID(i, j, nx, ny)];
  float hR = expf(sigma[ID(iR, j, nx, ny)]);
  float uR = u[ID(iR, j, nx, ny)];
  float vR = v[ID(iR, j, nx, ny)];
  float Fh, Fmx, Fmy;
  hll_x(hL, uL, vL, hR, uR, vR, g, Fh, Fmx, Fmy);
  Fh_x[IDXF(i, j, nx, ny)] = Fh;
  Fmx_x[IDXF(i, j, nx, ny)] = Fmx;
  Fmy_x[IDXF(i, j, nx, ny)] = Fmy;
}

// Compute HLL fluxes on y faces (i,j+1/2)
__global__ void
flux_y_kernel(const float *__restrict__ sigma, const float *__restrict__ u,
              const float *__restrict__ v, float *__restrict__ Gh_y,
              float *__restrict__ Gmx_y, float *__restrict__ Gmy_y, int nx,
              int ny, float g) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= nx || j >= ny)
    return;
  int jT = wrap(j + 1, ny);
  float hB = expf(sigma[ID(i, j, nx, ny)]);
  float uB = u[ID(i, j, nx, ny)];
  float vB = v[ID(i, j, nx, ny)];
  float hT = expf(sigma[ID(i, jT, nx, ny)]);
  float uT = u[ID(i, jT, nx, ny)];
  float vT = v[ID(i, jT, nx, ny)];
  float Gh, Gmx, Gmy;
  hll_y(hB, uB, vB, hT, uT, vT, g, Gh, Gmx, Gmy);
  Gh_y[IDYF(i, j, nx, ny)] = Gh;
  Gmx_y[IDYF(i, j, nx, ny)] = Gmx;
  Gmy_y[IDYF(i, j, nx, ny)] = Gmy;
}

// Update conserved vars then map back to (sigma,u,v)
__global__ void
update_kernel(float *__restrict__ sigma, float *__restrict__ u,
              float *__restrict__ v, const float *__restrict__ Fh_x,
              const float *__restrict__ Fmx_x, const float *__restrict__ Fmy_x,
              const float *__restrict__ Gh_y, const float *__restrict__ Gmx_y,
              const float *__restrict__ Gmy_y, int nx, int ny, float dx,
              float dy, float dt_eff, float g) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= nx || j >= ny)
    return;
  int id = ID(i, j, nx, ny);
  int im = wrap(i - 1, nx), jm = wrap(j - 1, ny);

  // reconstruct conserved
  float h = expf(sigma[id]);
  float mx = h * u[id];
  float my = h * v[id];

  // flux differences
  float dFx_h = Fh_x[IDXF(i, j, nx, ny)] - Fh_x[IDXF(im, j, nx, ny)];
  float dFx_mx = Fmx_x[IDXF(i, j, nx, ny)] - Fmx_x[IDXF(im, j, nx, ny)];
  float dFx_my = Fmy_x[IDXF(i, j, nx, ny)] - Fmy_x[IDXF(im, j, nx, ny)];
  float dGy_h = Gh_y[IDYF(i, j, nx, ny)] - Gh_y[IDYF(i, jm, nx, ny)];
  float dGy_mx = Gmx_y[IDYF(i, j, nx, ny)] - Gmx_y[IDYF(i, jm, nx, ny)];
  float dGy_my = Gmy_y[IDYF(i, j, nx, ny)] - Gmy_y[IDYF(i, jm, nx, ny)];

  // conservative update
  float invdx = 1.0f / dx, invdy = 1.0f / dy;
  h -= dt_eff * (dFx_h * invdx + dGy_h * invdy);
  mx -= dt_eff * (dFx_mx * invdx + dGy_mx * invdy);
  my -= dt_eff * (dFx_my * invdx + dGy_my * invdy);

  // positivity floor & map back
  const float eps = 1e-6f;
  h = fmaxf(h, eps);
  sigma[id] = logf(h);
  u[id] = mx / h;
  v[id] = my / h;
}

// Laplacian viscosity on u,v only (no device lambdas)
__global__ void viscosity_uv(float *__restrict__ u, float *__restrict__ v,
                             int nx, int ny, float dx, float dy, float nu,
                             float dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= nx || j >= ny)
    return;

  float invdx2 = 1.0f / (dx * dx), invdy2 = 1.0f / (dy * dy);
  int id = ID(i, j, nx, ny);

  // Laplacian on u
  float u_c = u[id];
  float u_xp = u[ID(wrap(i + 1, nx), j, nx, ny)];
  float u_xm = u[ID(wrap(i - 1, nx), j, nx, ny)];
  float u_yp = u[ID(i, wrap(j + 1, ny), nx, ny)];
  float u_ym = u[ID(i, wrap(j - 1, ny), nx, ny)];
  float du =
      (u_xp - 2.0f * u_c + u_xm) * invdx2 + (u_yp - 2.0f * u_c + u_ym) * invdy2;

  // Laplacian on v
  float v_c = v[id];
  float v_xp = v[ID(wrap(i + 1, nx), j, nx, ny)];
  float v_xm = v[ID(wrap(i - 1, nx), j, nx, ny)];
  float v_yp = v[ID(i, wrap(j + 1, ny), nx, ny)];
  float v_ym = v[ID(i, wrap(j - 1, ny), nx, ny)];
  float dv =
      (v_xp - 2.0f * v_c + v_xm) * invdx2 + (v_yp - 2.0f * v_c + v_ym) * invdy2;

  u[id] += nu * dt * du;
  v[id] += nu * dt * dv;
}

// display
void show_ncurses(int step, double fps, double avgfps, const float *host_sigma,
                  int nx, int ny, float t, float tau, float dtau, float f0,
                  float nu) {

  static const wchar_t wramp[] = {L' ', L'▁', L'▂', L'▃', L'▄',
                                  L'▅', L'▆', L'▇', L'█'};
  const int L = (int)(sizeof(wramp) / sizeof(wramp[0])) - 1;

  auto sample_bilinear = [&](double x, double y) {
    // x,y in [0, nx) × [0, ny)
    int i0 = (int)floor(x), j0 = (int)floor(y);
    int i1 = wrap(i0 + 1, nx), j1 = wrap(j0 + 1, ny);
    double tx = x - i0, ty = y - j0;
    i0 = wrap(i0, nx);
    j0 = wrap(j0, ny);
    double s00 = host_sigma[ID(i0, j0, nx, ny)];
    double s10 = host_sigma[ID(i1, j0, nx, ny)];
    double s01 = host_sigma[ID(i0, j1, nx, ny)];
    double s11 = host_sigma[ID(i1, j1, nx, ny)];
    double sx0 = s00 * (1.0 - tx) + s10 * tx;
    double sx1 = s01 * (1.0 - tx) + s11 * tx;
    return (float)(sx0 * (1.0 - ty) + sx1 * ty);
  };

  // Header line
  int rows, cols;
  getmaxyx(stdscr, rows, cols);
  mvprintw(0, 0,
           "step %d  FPS: %.1f (avg %.1f)  t=%.3gs tau=%.6f dtau=%.2e  f0=%.2e "
           "nu=%.2f  [q quit]",
           step, fps, avgfps, t, tau, dtau, f0, nu);

  // Viewport excluding the header row
  int disp_rows = rows - 1, disp_cols = cols;
  if (disp_rows <= 0 || disp_cols <= 0) {
    refresh();
    return;
  }

  // Mean/Std contrast window
  const int N = nx * ny;
  double sum = 0.0, sum2 = 0.0;
  for (int k = 0; k < N; ++k) {
    double s = host_sigma[k];
    sum += s;
    sum2 += s * s;
  }
  double mean = sum / N;
  double var = fmax(0.0, sum2 / N - mean * mean);
  double sd = sqrt(var);

  // window; tweak the multiplier (e.g., 1.5–3.0) to taste
  double lo = mean - 2.0 * sd;
  double hi = mean + 2.0 * sd;
  double inv = (hi > lo) ? 1.0 / (hi - lo) : 1.0;

  for (int j = 0; j < disp_rows; ++j) {
    // sample at the center of each terminal row/col
    double y = ((j + 0.5) * ny) / (double)disp_rows;
    move(j + 1, 0);
    for (int i = 0; i < disp_cols; ++i) {
      double x = ((i + 0.5) * nx) / (double)disp_cols;
      float s = sample_bilinear(x, y);

      // Blue-noise-ish hash-based dither in ±(0.5/L) bins
      uint32_t h =
          (uint32_t)(1469598103u ^ (j * 1315423911u + i * 2654435761u));
      h ^= h >> 13;
      h *= 0x5bd1e995u;
      h ^= h >> 15;
      double dither = ((h & 0xFFFF) / 65535.0 - 0.5) * (0.5 / (double)L);

      double val = (s - lo) * inv + dither;
      if (val < 0.0)
        val = 0.0;
      if (val > 1.0)
        val = 1.0;

      int idx = (int)(val * L + 0.5);
      if (idx < 0)
        idx = 0;
      if (idx > L)
        idx = L;

      addnwstr(&wramp[idx], 1);
    }
    clrtoeol();
  }
  refresh();
}

int main(int argc, char **argv) {
  Params P;
  parse_args(argc, argv, P);

  // host & device
  HostState H;
  initialize_host(P, H);
  int nx = P.nx, ny = P.ny, N = nx * ny;
  DeviceState D;
  device_alloc(D, N);
  CUDA_CHECK(cudaMemcpy(D.d_sigma, H.h_sigma.data(), N * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(D.d_u, H.h_u.data(), N * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(D.d_v, H.h_v.data(), N * sizeof(float),
                        cudaMemcpyHostToDevice));

  // per-block max buffer for CFL
  dim3 bs(16, 16);
  dim3 gs((nx + bs.x - 1) / bs.x, (ny + bs.y - 1) / bs.y);
  CUDA_CHECK(cudaMalloc(&D.d_block_cmax, gs.x * gs.y * sizeof(float)));

  // pinned buffer for draw
  CUDA_CHECK(cudaMallocHost(&H.hp_sigma_pinned, N * sizeof(float)));

  // time vars
  float tau = P.tau0;
  float t = P.t0;
  float dtau = P.dtau;

  auto do_step = [&](float &dt_eff) {
    // compute cmax on device
    size_t shmem = bs.x * bs.y * sizeof(float);
    wavespeed_block_max<<<gs, bs, shmem>>>(D.d_sigma, D.d_u, D.d_v, P.g, nx, ny,
                                           D.d_block_cmax);
    CUDA_CHECK(cudaPeekAtLastError());
    // copy per-block maxima and reduce on host
    std::vector<float> h_blk(gs.x * gs.y);
    CUDA_CHECK(cudaMemcpy(h_blk.data(), D.d_block_cmax,
                          h_blk.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    float cmax = 0.0f;
    for (float v : h_blk)
      cmax = std::max(cmax, v);
    if (cmax < 1e-12f)
      cmax = 1e-12f;
    float dt_cfl = P.CFL * fminf(P.dx, P.dy) / cmax;
    dt_eff = fminf(t * dtau, dt_cfl);

    // fluxes and update
    flux_x_kernel<<<gs, bs>>>(D.d_sigma, D.d_u, D.d_v, D.d_Fh_x, D.d_Fmx_x,
                              D.d_Fmy_x, nx, ny, P.g);
    CUDA_CHECK(cudaPeekAtLastError());
    flux_y_kernel<<<gs, bs>>>(D.d_sigma, D.d_u, D.d_v, D.d_Gh_y, D.d_Gmx_y,
                              D.d_Gmy_y, nx, ny, P.g);
    CUDA_CHECK(cudaPeekAtLastError());
    update_kernel<<<gs, bs>>>(D.d_sigma, D.d_u, D.d_v, D.d_Fh_x, D.d_Fmx_x,
                              D.d_Fmy_x, D.d_Gh_y, D.d_Gmx_y, D.d_Gmy_y, nx, ny,
                              P.dx, P.dy, dt_eff, P.g);
    CUDA_CHECK(cudaPeekAtLastError());
    if (P.nu > 0.0f) {
      viscosity_uv<<<gs, bs>>>(D.d_u, D.d_v, nx, ny, P.dx, P.dy, P.nu, dt_eff);
      CUDA_CHECK(cudaPeekAtLastError());
    }
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

    int step = 0, frames = 0;
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
        frames++;
        CUDA_CHECK(cudaMemcpy(H.hp_sigma_pinned, D.d_sigma, N * sizeof(float),
                              cudaMemcpyDeviceToHost));
        show_ncurses(step, fps, avgfps, H.hp_sigma_pinned, nx, ny, t, tau, dtau,
                     P.f0, P.nu);
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
    // headless benchmark
    using clock = std::chrono::high_resolution_clock;
    auto start = clock::now();
    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));
    CUDA_CHECK(cudaEventRecord(ev_start));
    int frames = 0;
    for (int step = 0; step < (P.steps ? P.steps : 2000); ++step) {
      float dt_eff = 0.0f;
      do_step(dt_eff);
      tau += dtau;
      t *= expf(dtau);
      if (step % P.stride == 0)
        frames++;
    }
    CUDA_CHECK(cudaEventRecord(ev_end));
    CUDA_CHECK(cudaEventSynchronize(ev_end));
    auto end = clock::now();
    float ms_gpu = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_gpu, ev_start, ev_end));
    double secs = std::chrono::duration<double>(end - start).count();
    printf("Headless benchmark (stride=%d):\n", P.stride);
    printf("  Simulated steps: %d\n", (P.steps ? P.steps : 2000));
    printf("  Wall-clock: %d frames in %.3f s -> %.1f FPS\n", frames, secs,
           (frames > 0 ? frames / secs : 0.0));
    printf("  GPU only:   %d frames in %.3f s -> %.1f FPS\n", frames,
           ms_gpu / 1000.0, (ms_gpu > 0 ? 1000.0 * frames / ms_gpu : 0.0));
    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_end));
  }

  // cleanup
  device_free(D);
  if (H.hp_sigma_pinned)
    cudaFreeHost(H.hp_sigma_pinned);
  return 0;
}
