// js_cuda3d.cu : CUDA 3D fluid sim
// nvcc -std=c++14 -O3 -use_fast_math -arch=sm_86 js_cuda3d.cu -lncursesw -o jsc3d

#include <algorithm>
#include <chrono>
#include <cmath>
#include <getopt.h>
#include <locale.h>
#include <math.h>
#include <ncursesw/curses.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <wchar.h>

#include <cuda_runtime.h>

typedef float Real;
#define R(x) (Real)(x)

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static const Real TWO_PI = (Real)(2.0 * M_PI);

#define N 192
#define IX(i, j, k) ((i) + (N + 2) * ((j) + (N + 2) * (k)))

#define CUDA_CHECK(ans)                                                        \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s:%d\n", cudaGetErrorString(code), file,
            line);
    exit(code);
  }
}

__device__ inline float rand01(uint32_t s) {
  s ^= s << 13;
  s ^= s >> 17;
  s ^= s << 5;
  return (float)s * 2.3283064365386963e-10f; // / 2^32
}

static bool curses_active = false;

static int g_levels = 256;     // intensity bands [0..g_levels]
static Real g_gain = R(0.2f);  // exposure/opacity for screen finalize
static Real g_gamma = R(1.2f); // display gamma (>1 darkens highlights)

static Real g_dt = R(1.0f);
static Real g_visc = R(1e-5f);
static Real g_diff = R(1e-6f);
static Real g_decay = R(0.9f);
static Real g_src_gain = R(0.25f); // density injection multiplier
static Real g_src_freq = R(0.02f); // source orbital frequency
static Real g_seed_amp = R(1.2f);
static Real g_seed_noise = R(0.25f);
static Real g_seed_dens_amp = R(0.8f);
static Real g_seed_sigma = R(0.12f);

static const Real NO4 = (Real)N / R(4.0f);

// colors
static int g_num_pairs = 0;
static bool g_have_colors = false;

// fields
Real *d_u, *d_v, *d_w;    // velocity
Real *d_u0, *d_v0, *d_w0; // temp velocity
Real *d_d, *d_d0, *d_tmp; // density and scratch
Real *d_p, *d_div;        // pressure, divergence

// accumulators
Real *d_acc = nullptr;      // W*H floats
uint16_t *d_cbuf = nullptr; // W*H band indices [0..g_levels]
uint16_t *h_cbuf = nullptr; // host mirror for cbuf
int scrW = 0, scrH = 0;

// geometry
dim3 bs(8, 8, 4);
dim3 gs((N + bs.x - 1) / bs.x, (N + bs.y - 1) / bs.y, (N + bs.z - 1) / bs.z);

// kernels
__global__ void k_decay(Real *d, Real decay) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
  if (i <= N && j <= N && k <= N)
    d[IX(i, j, k)] *= decay;
}

__global__ void k_add_source3d(Real *u, Real *v, Real *w, Real *d, int step,
                               Real src_gain, Real src_freq) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

  Real t = src_freq * (Real)step;
  Real dx = (Real)i - NO4 * (R(1.0f) + cosf(t));
  Real dy = (Real)j - NO4 * (R(1.0f) + sinf(t));
  Real dz = (Real)k - NO4 * (R(1.0f) + sinf(t));
  Real r2 = dx * dx + dy * dy + dz * dz;

  if (r2 < N) {
    Real r = sqrtf(r2) + R(1e-7f);
    d[IX(i, j, k)] += src_gain * expf(-r2 / N);
    u[IX(i, j, k)] += dz / r;
    v[IX(i, j, k)] += dy / r;
    w[IX(i, j, k)] += dx / r;
  }
}

__global__ void k_set_bnd(Real *u, Real *v, Real *w, Real *d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
  if (i > N || j > N || k > N)
    return;

  // X
  u[IX(0, j, k)] = -u[IX(1, j, k)];
  u[IX(N + 1, j, k)] = -u[IX(N, j, k)];
  v[IX(0, j, k)] = v[IX(1, j, k)];
  v[IX(N + 1, j, k)] = v[IX(N, j, k)];
  w[IX(0, j, k)] = w[IX(1, j, k)];
  w[IX(N + 1, j, k)] = w[IX(N, j, k)];

  // Y
  v[IX(i, 0, k)] = -v[IX(i, 1, k)];
  v[IX(i, N + 1, k)] = -v[IX(i, N, k)];
  u[IX(i, 0, k)] = u[IX(i, 1, k)];
  u[IX(i, N + 1, k)] = u[IX(i, N, k)];
  w[IX(i, 0, k)] = w[IX(i, 1, k)];
  w[IX(i, N + 1, k)] = w[IX(i, N, k)];

  // Z
  w[IX(i, j, 0)] = -w[IX(i, j, 1)];
  w[IX(i, j, N + 1)] = -w[IX(i, j, N)];
  u[IX(i, j, 0)] = u[IX(i, j, 1)];
  u[IX(i, j, N + 1)] = u[IX(i, j, N)];
  v[IX(i, j, 0)] = v[IX(i, j, 1)];
  v[IX(i, j, N + 1)] = v[IX(i, j, N)];

  // density
  d[IX(0, j, k)] = d[IX(1, j, k)];
  d[IX(N + 1, j, k)] = d[IX(N, j, k)];
  d[IX(i, 0, k)] = d[IX(i, 1, k)];
  d[IX(i, N + 1, k)] = d[IX(i, N, k)];
  d[IX(i, j, 0)] = d[IX(i, j, 1)];
  d[IX(i, j, N + 1)] = d[IX(i, j, N)];
}

__global__ void k_lin(Real *x_new, const Real *x_old, const Real *x0, Real a,
                      Real c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
  x_new[IX(i, j, k)] = (x0[IX(i, j, k)] +
                        a * (x_old[IX(i - 1, j, k)] + x_old[IX(i + 1, j, k)] +
                             x_old[IX(i, j - 1, k)] + x_old[IX(i, j + 1, k)] +
                             x_old[IX(i, j, k - 1)] + x_old[IX(i, j, k + 1)])) /
                       c;
}

__global__ void k_div(const Real *u, const Real *v, const Real *w, Real *div,
                      Real *p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
  div[IX(i, j, k)] = -R(0.5f) * ((u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)]) +
                                 (v[IX(i, j + 1, k)] - v[IX(i, j - 1, k)]) +
                                 (w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]));
  p[IX(i, j, k)] = R(0.0f);
}

__global__ void k_proj(Real *u, Real *v, Real *w, const Real *p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
  u[IX(i, j, k)] -= R(0.5f) * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
  v[IX(i, j, k)] -= R(0.5f) * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
  w[IX(i, j, k)] -= R(0.5f) * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
}

// Semi-Lagrangian advection (trilinear)
__global__ void k_adv3d(Real *q, const Real *q0, const Real *u, const Real *v,
                        const Real *w, Real dt) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
  if (i > N || j > N || k > N)
    return;

  Real x = (Real)i - dt * u[IX(i, j, k)];
  Real y = (Real)j - dt * v[IX(i, j, k)];
  Real z = (Real)k - dt * w[IX(i, j, k)];

  x = fminf(fmaxf(x, R(0.5f)), R(N) + R(0.5f));
  y = fminf(fmaxf(y, R(0.5f)), R(N) + R(0.5f));
  z = fminf(fmaxf(z, R(0.5f)), R(N) + R(0.5f));

  int i0 = (int)floorf(x), i1 = i0 + 1;
  int j0 = (int)floorf(y), j1 = j0 + 1;
  int k0 = (int)floorf(z), k1 = k0 + 1;

  Real sx = x - (Real)i0, tx = R(1.0f) - sx;
  Real sy = y - (Real)j0, ty = R(1.0f) - sy;
  Real sz = z - (Real)k0, tz = R(1.0f) - sz;

  Real c000 = q0[IX(i0, j0, k0)], c100 = q0[IX(i1, j0, k0)];
  Real c010 = q0[IX(i0, j1, k0)], c110 = q0[IX(i1, j1, k0)];
  Real c001 = q0[IX(i0, j0, k1)], c101 = q0[IX(i1, j0, k1)];
  Real c011 = q0[IX(i0, j1, k1)], c111 = q0[IX(i1, j1, k1)];

  Real c00 = tx * c000 + sx * c100;
  Real c10 = tx * c010 + sx * c110;
  Real c01 = tx * c001 + sx * c101;
  Real c11 = tx * c011 + sx * c111;

  Real c0 = ty * c00 + sy * c10;
  Real c1 = ty * c01 + sy * c11;

  q[IX(i, j, k)] = tz * c0 + sz * c1;
}

__global__ void k_clear_acc(int W, int H, Real *acc) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int n = W * H;
  if (idx < n)
    acc[idx] = R(0.0f);
}

__global__ void k_iso_accumulate(int Ngrid, int W, int H, Real sproj, Real cx,
                                 Real cy, const Real *__restrict__ dens,
                                 Real *__restrict__ acc) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
  if (i > Ngrid || j > Ngrid || k > Ngrid)
    return;

  Real val = sqrtf(fmaxf(dens[IX(i, j, k)], R(0.0f)));
  if (val <= R(0.0f))
    return;

  // iso projection
  Real X = ((Real)i - (Real)j) * sproj + cx;
  Real Y = (((Real)i + (Real)j) * R(0.5f) - (Real)k) * sproj + cy;

  int x0 = (int)floorf(X), y0 = (int)floorf(Y);
  Real fx = X - (Real)x0, fy = Y - (Real)y0;
  int x1 = x0 + 1, y1 = y0 + 1;

  Real w00 = (R(1.0f) - fx) * (R(1.0f) - fy);
  Real w10 = (fx) * (R(1.0f) - fy);
  Real w01 = (R(1.0f) - fx) * (fy);
  Real w11 = (fx) * (fy);

  if (x0 >= 0 && x0 < W && y0 >= 0 && y0 < H)
    atomicAdd(&acc[y0 * W + x0], val * w00);
  if (x1 >= 0 && x1 < W && y0 >= 0 && y0 < H)
    atomicAdd(&acc[y0 * W + x1], val * w10);
  if (x0 >= 0 && x0 < W && y1 >= 0 && y1 < H)
    atomicAdd(&acc[y1 * W + x0], val * w01);
  if (x1 >= 0 && x1 < W && y1 >= 0 && y1 < H)
    atomicAdd(&acc[y1 * W + x1], val * w11);
}

__global__ void k_finalize_screen(int W, int H, Real gain, int levels,
                                  const Real *acc, uint16_t *cbuf, Real gamma) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int n = W * H;
  if (idx >= n)
    return;

  Real a = acc[idx];
  Real y = R(1.0f) - expf(-gain * a);
  y = powf(y, gamma);
  y = fminf(fmaxf(y, R(0.0f)), R(1.0f));

  Real L = (Real)levels;
  int q = (int)floorf(y * L + R(0.5f));
  if (q < 0)
    q = 0;
  if (q > levels)
    q = levels;

  cbuf[idx] = (uint16_t)q;
}

// host
void lin_solve(Real *x, Real *x0, Real a, Real c, int iters) {
  Real *read = x;
  Real *write = d_tmp;
  for (int it = 0; it < iters; ++it) {
    k_lin<<<gs, bs>>>(write, read, x0, a, c);
    CUDA_CHECK(cudaGetLastError());
    Real *tmp = read;
    read = write;
    write = tmp;
  }
  if (read != x) {
    CUDA_CHECK(cudaMemcpy(x, read,
                          (size_t)(N + 2) * (N + 2) * (N + 2) * sizeof(Real),
                          cudaMemcpyDeviceToDevice));
  }
}

void diffuse(Real *x, Real *x0, Real diffc) {
  Real a = g_dt * diffc * (Real)N * (Real)N;
  lin_solve(x, x0, a, R(1.0f) + R(6.0f) * a, 12);
}

void project(Real *u, Real *v, Real *w, Real *p, Real *div) {
  k_div<<<gs, bs>>>(u, v, w, div, p);
  CUDA_CHECK(cudaGetLastError());
  lin_solve(p, div, R(1.0f), R(6.0f), 12);
  k_proj<<<gs, bs>>>(u, v, w, p);
  CUDA_CHECK(cudaGetLastError());
}

void advect(Real *q, Real *q0, Real *u, Real *v, Real *w) {
  k_adv3d<<<gs, bs>>>(q, q0, u, v, w, g_dt);
  CUDA_CHECK(cudaGetLastError());
}

void vel_step() {
  diffuse(d_u0, d_u, g_visc);
  diffuse(d_v0, d_v, g_visc);
  diffuse(d_w0, d_w, g_visc);
  k_set_bnd<<<gs, bs>>>(d_u0, d_v0, d_w0, d_d);
  CUDA_CHECK(cudaGetLastError());

  project(d_u0, d_v0, d_w0, d_p, d_div);
  k_set_bnd<<<gs, bs>>>(d_u0, d_v0, d_w0, d_d);
  CUDA_CHECK(cudaGetLastError());

  advect(d_u, d_u0, d_u0, d_v0, d_w0);
  advect(d_v, d_v0, d_u0, d_v0, d_w0);
  advect(d_w, d_w0, d_u0, d_v0, d_w0);
  k_set_bnd<<<gs, bs>>>(d_u, d_v, d_w, d_d);
  CUDA_CHECK(cudaGetLastError());

  project(d_u, d_v, d_w, d_p, d_div);
  k_set_bnd<<<gs, bs>>>(d_u, d_v, d_w, d_d);
  CUDA_CHECK(cudaGetLastError());
}

void dens_step() {
  diffuse(d_d0, d_d, g_diff);
  k_set_bnd<<<gs, bs>>>(d_u, d_v, d_w, d_d0);
  CUDA_CHECK(cudaGetLastError());

  advect(d_d, d_d0, d_u, d_v, d_w);
  k_set_bnd<<<gs, bs>>>(d_u, d_v, d_w, d_d);
  CUDA_CHECK(cudaGetLastError());
}

__global__ void k_seed_turbulence(Real *u, Real *v, Real *w, Real *d, Real amp,
                                  Real noise, Real dens_amp, Real sigma,
                                  uint32_t seed) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
  if (i > N || j > N || k > N)
    return;

  Real xn = ((Real)i - R(0.5f)) / (Real)N;
  Real yn = ((Real)j - R(0.5f)) / (Real)N;
  Real zn = ((Real)k - R(0.5f)) / (Real)N;

  Real X = TWO_PI * xn;
  Real Y = TWO_PI * yn;
  Real Z = TWO_PI * zn;

  Real A = amp, B = amp, C = amp;
  Real uu = A * sinf(Z) + C * cosf(Y);
  Real vv = B * sinf(X) + A * cosf(Z);
  Real ww = C * sinf(Y) + B * cosf(X);

  uint32_t base = seed ^ (uint32_t)(i * 73856093u) ^ (uint32_t)(j * 19349663u) ^
                  (uint32_t)(k * 83492791u);
  uu += noise * (rand01(base + 0) - 0.5f);
  vv += noise * (rand01(base + 1) - 0.5f);
  ww += noise * (rand01(base + 2) - 0.5f);

  u[IX(i, j, k)] = uu;
  v[IX(i, j, k)] = vv;
  w[IX(i, j, k)] = ww;

  Real dx = xn - R(0.5f), dy = yn - R(0.5f), dz = zn - R(0.5f);
  Real r2 = dx * dx + dy * dy + dz * dz;
  Real g = expf(-r2 / (R(2.0f) * sigma * sigma));
  Real tex =
      R(0.5f) *
      (sinf(R(2.0f) * X) * sinf(R(2.0f) * Y) * sinf(R(2.0f) * Z) + R(1.0f));
  d[IX(i, j, k)] = dens_amp * (g + R(0.35f) * tex);
}

void seed_initial_turbulence() {
  k_seed_turbulence<<<gs, bs>>>(d_u, d_v, d_w, d_d, g_seed_amp, g_seed_noise,
                                g_seed_dens_amp, g_seed_sigma, 1337u);
  CUDA_CHECK(cudaGetLastError());
  k_set_bnd<<<gs, bs>>>(d_u, d_v, d_w, d_d);
  CUDA_CHECK(cudaGetLastError());
  project(d_u, d_v, d_w, d_p, d_div);
  k_set_bnd<<<gs, bs>>>(d_u, d_v, d_w, d_d);
  CUDA_CHECK(cudaGetLastError());
}

// gpu
void gpu_alloc() {
  size_t s = (size_t)(N + 2) * (N + 2) * (N + 2) * sizeof(Real);
  CUDA_CHECK(cudaMalloc(&d_u, s));
  CUDA_CHECK(cudaMalloc(&d_v, s));
  CUDA_CHECK(cudaMalloc(&d_w, s));
  CUDA_CHECK(cudaMalloc(&d_u0, s));
  CUDA_CHECK(cudaMalloc(&d_v0, s));
  CUDA_CHECK(cudaMalloc(&d_w0, s));
  CUDA_CHECK(cudaMalloc(&d_d, s));
  CUDA_CHECK(cudaMalloc(&d_d0, s));
  CUDA_CHECK(cudaMalloc(&d_tmp, s));
  CUDA_CHECK(cudaMalloc(&d_p, s));
  CUDA_CHECK(cudaMalloc(&d_div, s));

  CUDA_CHECK(cudaMemset(d_u, 0, s));
  CUDA_CHECK(cudaMemset(d_v, 0, s));
  CUDA_CHECK(cudaMemset(d_w, 0, s));
  CUDA_CHECK(cudaMemset(d_u0, 0, s));
  CUDA_CHECK(cudaMemset(d_v0, 0, s));
  CUDA_CHECK(cudaMemset(d_w0, 0, s));
  CUDA_CHECK(cudaMemset(d_d, 0, s));
  CUDA_CHECK(cudaMemset(d_d0, 0, s));
  CUDA_CHECK(cudaMemset(d_tmp, 0, s));
  CUDA_CHECK(cudaMemset(d_p, 0, s));
  CUDA_CHECK(cudaMemset(d_div, 0, s));

  d_acc = nullptr;
  d_cbuf = nullptr;
  h_cbuf = nullptr;
  scrW = scrH = 0;
}

void gpu_free() {
  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_w);
  cudaFree(d_u0);
  cudaFree(d_v0);
  cudaFree(d_w0);
  cudaFree(d_d);
  cudaFree(d_d0);
  cudaFree(d_tmp);
  cudaFree(d_p);
  cudaFree(d_div);
  if (d_acc)
    cudaFree(d_acc), d_acc = nullptr;
  if (d_cbuf)
    cudaFree(d_cbuf), d_cbuf = nullptr;
  if (h_cbuf)
    free(h_cbuf), h_cbuf = nullptr;
}

// render
void init_colors_dynamic() {
  g_have_colors = has_colors();
  if (!g_have_colors)
    return;

  start_color();
  use_default_colors();

  int max_pairs = COLOR_PAIRS - 1;
  int max_colors = COLORS - 1;
  g_num_pairs = std::min({max_pairs, max_colors, 240});
  if (g_num_pairs <= 0) {
    g_num_pairs = 0;
    return;
  }

  if (COLORS >= 256) {
    for (int i = 1; i <= g_num_pairs; ++i) {
      int idx = 16 + (int)llround((double)((i - 1) * (231 - 16) /
                                           std::max(1, g_num_pairs - 1)));
      init_pair(i, idx, -1);
    }
  } else {
    static const short base[] = {COLOR_BLUE,   COLOR_CYAN,    COLOR_GREEN,
                                 COLOR_YELLOW, COLOR_MAGENTA, COLOR_RED,
                                 COLOR_WHITE};
    int baseN = (int)(sizeof(base) / sizeof(base[0]));
    for (int i = 1; i <= g_num_pairs; ++i) {
      short col = base[(i - 1) % baseN];
      init_pair(i, col, -1);
    }
  }
}

inline int band_to_pair(uint16_t band) {
  if (!g_have_colors || g_num_pairs <= 0)
    return 0;
  if (band == 0)
    return 0;
  int pair = 1 + (int)((uint64_t)(band - 1) * (uint64_t)(g_num_pairs - 1) /
                       (uint64_t)std::max(1, g_levels - 1));
  if (pair < 1)
    pair = 1;
  if (pair > g_num_pairs)
    pair = g_num_pairs;
  return pair;
}

void show_iso(int step, double fps) {
  static const wchar_t wramp[] = {L' ', L'░', L'▒', L'▓', L'█'};

  int rows, cols;
  getmaxyx(stdscr, rows, cols);
  if (rows < 5 || cols < 10)
    return;

  mvprintw(0, 0, "step %d  FPS %.1f  (q=quit)  N=%d  L=%d  gain=%.3f", step,
           fps, N, g_levels, (double)g_gain);
  mvprintw(1, 0,
           "dt=%.3f visc=%.2e diff=%.2e decay=%.3f gamma=%.2f  srcGain=%.3f "
           "srcFreq=%.3f",
           (double)g_dt, (double)g_visc, (double)g_diff, (double)g_decay,
           (double)g_gamma, (double)g_src_gain, (double)g_src_freq);

  int W = cols, H = rows - 2; // two rows for HUD
  if (W != scrW || H != scrH || !d_acc || !d_cbuf || !h_cbuf) {
    if (d_acc)
      cudaFree(d_acc);
    if (d_cbuf)
      cudaFree(d_cbuf);
    if (h_cbuf)
      free(h_cbuf);
    CUDA_CHECK(cudaMalloc(&d_acc, W * H * sizeof(Real)));
    CUDA_CHECK(cudaMalloc(&d_cbuf, W * H * sizeof(uint16_t)));
    h_cbuf = (uint16_t *)malloc(W * H * sizeof(uint16_t));
    scrW = W;
    scrH = H;
  }

  Real sx = (Real)W / (R(2.0f) * (Real)N + R(2.0f));
  Real sy = (Real)H / (R(2.0f) * (Real)N + R(2.0f));
  Real sproj = (sx < sy ? sx : sy);
  if (sproj <= R(0.0f))
    sproj = R(1.0f);
  Real cx = (Real)W * R(0.5f);
  Real cy = (Real)H * R(0.5f);

  int pixels = W * H;
  int tpb = 256;
  int blocks = (pixels + tpb - 1) / tpb;

  k_clear_acc<<<blocks, tpb>>>(W, H, d_acc);
  CUDA_CHECK(cudaGetLastError());
  k_iso_accumulate<<<gs, bs>>>(N, W, H, sproj, cx, cy, d_d, d_acc);
  CUDA_CHECK(cudaGetLastError());
  k_finalize_screen<<<blocks, tpb>>>(W, H, g_gain, g_levels, d_acc, d_cbuf,
                                     g_gamma);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy(h_cbuf, d_cbuf, W * H * sizeof(uint16_t),
                        cudaMemcpyDeviceToHost));

  int last_pair = -1;
  for (int y = 0; y < H; ++y) {
    move(y + 2, 0);
    const uint16_t *row = h_cbuf + y * W;
    for (int x = 0; x < W; ++x) {
      uint16_t band = row[x];
      int glyph_idx = (int)((uint64_t)band * 4 / std::max(1, g_levels));
      if (glyph_idx < 0)
        glyph_idx = 0;
      if (glyph_idx > 4)
        glyph_idx = 4;

      int pair = band_to_pair(band);
      if (pair != last_pair) {
        attrset(COLOR_PAIR(pair));
        last_pair = pair;
      }

      addnwstr(&wramp[glyph_idx], 1);
    }
  }
  wnoutrefresh(stdscr);
  doupdate();
}

// admin
void handle_exit(int sig) {
  if (curses_active) {
    endwin();
    curses_active = false;
  }
  gpu_free();
  fprintf(stderr, "\n[trap] caught sig %d\n", sig);
  exit(1);
}

void usage(const char *prog) {
  fprintf(stderr,
          "Usage: %s [options]\n"
          "  -L, --levels <int>         : intensity bands [8]\n"
          "  -g, --gain <float>         : exposure gain [0.02]\n"
          "  -G, --gamma <float>        : display gamma [1.8]\n"
          "      --dt <float>           : timestep [1.0]\n"
          "      --visc <float>         : viscosity [1e-5]\n"
          "      --diff <float>         : diffusion [1e-6]\n"
          "      --decay <float>        : density decay per step [0.9]\n"
          "      --amp <float>          : seed velocity amplitude [1.2]\n"
          "      --noise <float>        : seed velocity noise [0.25]\n"
          "      --dens-amp <float>     : seed density amplitude [0.8]\n"
          "      --sigma <float>        : seed Gaussian radius [0.12]\n"
          "      --src-gain <float>     : source density gain [0.25]\n"
          "      --src-freq <float>     : source orbital freq [0.05]\n"
          "  -h, --help\n",
          prog);
}

int main(int argc, char **argv) {
  static struct option long_opts[] = {{"levels", required_argument, 0, 'L'},
                                      {"gain", required_argument, 0, 'g'},
                                      {"gamma", required_argument, 0, 'G'},
                                      {"dt", required_argument, 0, 1000},
                                      {"visc", required_argument, 0, 1001},
                                      {"diff", required_argument, 0, 1002},
                                      {"decay", required_argument, 0, 1003},
                                      {"amp", required_argument, 0, 1004},
                                      {"noise", required_argument, 0, 1005},
                                      {"dens-amp", required_argument, 0, 1006},
                                      {"sigma", required_argument, 0, 1007},
                                      {"src-gain", required_argument, 0, 1008},
                                      {"src-freq", required_argument, 0, 1009},
                                      {"help", no_argument, 0, 'h'},
                                      {0, 0, 0, 0}};
  int opt, idx;
  while ((opt = getopt_long(argc, argv, "L:g:G:h", long_opts, &idx)) != -1) {
    switch (opt) {
    case 'L': {
      int L = atoi(optarg);
      if (L < 1)
        L = 1;
      g_levels = L;
    } break;
    case 'g': {
      g_gain = (Real)atof(optarg);
      if (g_gain <= R(0))
        g_gain = R(0.001f);
    } break;
    case 'G': {
      g_gamma = (Real)atof(optarg);
      if (g_gamma < R(0.3f))
        g_gamma = R(0.3f);
    } break;
    case 1000:
      g_dt = (Real)atof(optarg);
      break;
    case 1001:
      g_visc = (Real)atof(optarg);
      break;
    case 1002:
      g_diff = (Real)atof(optarg);
      break;
    case 1003:
      g_decay = (Real)atof(optarg);
      break;
    case 1004:
      g_seed_amp = (Real)atof(optarg);
      break;
    case 1005:
      g_seed_noise = (Real)atof(optarg);
      break;
    case 1006:
      g_seed_dens_amp = (Real)atof(optarg);
      break;
    case 1007:
      g_seed_sigma = (Real)atof(optarg);
      break;
    case 1008:
      g_src_gain = (Real)atof(optarg);
      break;
    case 1009:
      g_src_freq = (Real)atof(optarg);
      break;
    case 'h':
    default:
      usage(argv[0]);
      return 0;
    }
  }

  signal(SIGINT, handle_exit);
  signal(SIGTERM, handle_exit);

  gpu_alloc();
  seed_initial_turbulence();

  curses_active = true;
  setlocale(LC_ALL, "");
  initscr();
  cbreak();
  noecho();
  curs_set(0);
  nodelay(stdscr, TRUE);
  keypad(stdscr, TRUE);
  init_colors_dynamic();

  int step = 0;
  auto last = std::chrono::high_resolution_clock::now();

  while (true) {
    int ch = getch();
    if (ch == 'q' || ch == 'Q')
      break;

    k_decay<<<gs, bs>>>(d_d, g_decay);
    CUDA_CHECK(cudaGetLastError());
    k_add_source3d<<<gs, bs>>>(d_u, d_v, d_w, d_d, step, g_src_gain,
                               g_src_freq);
    CUDA_CHECK(cudaGetLastError());

    vel_step();
    dens_step();

    auto now = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(now - last).count();
    double fps = (dt > 0) ? 1.0 / dt : 0.0;
    show_iso(step, fps);
    last = now;
    ++step;
  }

  endwin();
  curses_active = false;
  gpu_free();
  return 0;
}
