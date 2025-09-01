// tau_sph.cu
// 2‑D Smoothed‑Particle Hydrodynamics (SPH)
//
// Build
//   nvcc -std=c++17 -O3 -use_fast_math -arch=sm_86 -lineinfo tau_sph.cu \
//   -o tau_sph -lncursesw
//
// Run
//   ./tau_sph --n 32768 --CFL 0.25 --dTau 1e-3 --visc 0.1 --visc_substeps 2 \
//   --stride 2 --fps 60
//   ./tau_sph --n 65536 --headless --stride 10 --visc_substeps 3

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <chrono>
#include <getopt.h>
#include <locale.h>
#include <math.h>
#include <ncursesw/curses.h>
#include <random>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <wchar.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
  // domain
  int N = 1 << 16;   // particle count
  float boxX = 1.0f; // domain width (m)
  float boxY = 1.0f; // domain height (m)

  // time
  float dTau = 1.0f; // τ step
  float t0 = 1.0f;   // reference time at τ=0
  float CFL = 1.0f;  // CFL factor

  // fluid
  float rho0 = 1.0f;       // reference density (kg/m^3)
  float c0 = 1.0f;         // artificial sound speed (m/s)
  float gammaEOS = 1.0f;   // Tait EOS exponent
  float hMul = 2.0f;       // smoothing length multiplier over particle spacing
  float viscAlpha = 0.25f; // Monaghan viscosity α
  float gravity = 9.81f;   // m/s^2 downward

  // UI
  bool headless = false;
  int stride = 1;         // render every Nth step
  int fps = 120;          // UI target FPS
  int fps_cap = 0;        // hard cap (0 uncapped)
  bool halfblocks = true; // high‑res renderer (two vertical samples per cell)

  // toggles
  bool rain = true; // GPU rain inflow
  bool useVisc = true;
  bool useGrav = true;

  int viscSub = 1;       // sub‑steps per frame (splits dt_eff)
  bool useXSPH = false;  // enabled by --muscl or xsph_eps>0
  float xsphEps = 0.25f; // XSPH smoothing strength ε

  int seed = 69420;
};

enum { GL_NONE = 0, GL_TOP = 1, GL_BOTTOM = 2, GL_FULL = 3, GL_COUNT };
static const wchar_t *HALF_BLOCKS[GL_COUNT] = {L" ", L"▀", L"▄", L"█"};

struct DevState {
  float2 *pos;
  float2 *vel;
  float2 *acc;
  float *s;
  float *press;
  int *cellHead;
  int *next;
  int Gx, Gy;
  float cell;
  int *grid2;
  size_t grid2Size;
  int W, H;
};

__device__ __forceinline__ float W_cubic(float r, float h) {
  float q = r / h;
  const float alpha = 10.0f / (7.0f * M_PI * h * h);
  if (q < 1.0f) {
    float q2 = q * q, q3 = q2 * q;
    return alpha * (1.f - 1.5f * q2 + 0.75f * q3);
  } else if (q < 2.0f) {
    float t = 2.f - q;
    return alpha * 0.25f * t * t * t;
  } else
    return 0.f;
}

__device__ __forceinline__ float2 gradW_cubic(float2 rij, float r, float h) {
  if (r <= 1e-8f || r >= 2.0f * h)
    return make_float2(0.f, 0.f);
  float q = r / h;
  const float alpha = 10.0f / (7.0f * M_PI * h * h);
  float dWdq;
  if (q < 1.0f)
    dWdq = alpha * (-3.0f * q + 2.25f * q * q);
  else {
    float t = 2.0f - q;
    dWdq = alpha * (-0.75f * t * t);
  }
  float invr = 1.0f / r;
  float dWdr = dWdq / h;
  return make_float2(dWdr * rij.x * invr, dWdr * rij.y * invr);
}

__device__ __forceinline__ int cell_index(int gx, int gy, int Gx, int Gy) {
  if ((unsigned)gx >= (unsigned)Gx || (unsigned)gy >= (unsigned)Gy)
    return -1;
  return gy * Gx + gx;
}

__device__ __forceinline__ int grid_x(float x, float cell, int Gx) {
  int gx = (int)floorf(x / cell);
  if (gx < 0)
    gx = 0;
  if (gx >= Gx)
    gx = Gx - 1;
  return gx;
}

__device__ __forceinline__ int grid_y(float y, float cell, int Gy) {
  int gy = (int)floorf(y / cell);
  if (gy < 0)
    gy = 0;
  if (gy >= Gy)
    gy = Gy - 1;
  return gy;
}

__global__ void k_clear_heads(int *__restrict__ heads, int M) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < M)
    heads[i] = -1;
}

__global__ void k_build_cells(const float2 *__restrict__ pos, int N,
                              int *__restrict__ cellHead,
                              int *__restrict__ next, int Gx, int Gy,
                              float cell) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;
  int gx = grid_x(pos[i].x, cell, Gx);
  int gy = grid_y(pos[i].y, cell, Gy);
  int c = gy * Gx + gx;
  next[i] = atomicExch(cellHead + c, i);
}

__global__ void k_density_pressure_cell(
    const float2 *__restrict__ pos, float *__restrict__ s,
    float *__restrict__ press, const int *__restrict__ cellHead,
    const int *__restrict__ next, int N, float mass, float h, float rho0,
    float c0, float gammaEOS, int Gx, int Gy, float cell) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;
  float2 xi = pos[i];
  int gx = grid_x(xi.x, cell, Gx);
  int gy = grid_y(xi.y, cell, Gy);
  float rho = 0.f;
  float twoh = 2.f * h;
  float twoh2 = twoh * twoh;
  for (int oy = -1; oy <= 1; ++oy)
    for (int ox = -1; ox <= 1; ++ox) {
      int cx = gx + ox, cy = gy + oy;
      if ((unsigned)cx >= (unsigned)Gx || (unsigned)cy >= (unsigned)Gy)
        continue;
      int head = cellHead[cy * Gx + cx];
      for (int j = head; j != -1; j = next[j]) {
        float2 rij = make_float2(xi.x - pos[j].x, xi.y - pos[j].y);
        float r2 = rij.x * rij.x + rij.y * rij.y;
        if (r2 >= twoh2)
          continue;
        float r = sqrtf(r2);
        rho += mass * W_cubic(r, h);
      }
    }
  float si = logf(fmaxf(rho, 1e-6f));
  s[i] = si;
  rho = expf(si);
  float ratio = rho / rho0;
  float p = (c0 * c0) * rho0 * (powf(ratio, gammaEOS) - 1.0f) / gammaEOS;
  press[i] = fmaxf(p, 0.0f);
}

__global__ void
k_forces_cell(const float2 *__restrict__ pos, const float2 *__restrict__ vel,
              const float *__restrict__ s, const float *__restrict__ press,
              float2 *__restrict__ acc, const int *__restrict__ cellHead,
              const int *__restrict__ next, int N, float mass, float h,
              float viscAlpha, float c0, float gx, float gy, bool useVisc,
              bool useGrav, int Gx, int Gy, float cell) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;
  float2 xi = pos[i];
  float2 vi = vel[i];
  float rhoi = expf(s[i]);
  float pi = press[i];
  float2 ai = make_float2(0.f, 0.f);
  int gx_i = grid_x(xi.x, cell, Gx);
  int gy_i = grid_y(xi.y, cell, Gy);
  float twoh = 2.f * h;
  float twoh2 = twoh * twoh;
  for (int oy = -1; oy <= 1; ++oy)
    for (int ox = -1; ox <= 1; ++ox) {
      int cx = gx_i + ox, cy = gy_i + oy;
      if ((unsigned)cx >= (unsigned)Gx || (unsigned)cy >= (unsigned)Gy)
        continue;
      int head = cellHead[cy * Gx + cx];
      for (int j = head; j != -1; j = next[j])
        if (j != i) {
          float2 rij = make_float2(xi.x - pos[j].x, xi.y - pos[j].y);
          float r2 = rij.x * rij.x + rij.y * rij.y;
          if (r2 >= twoh2 || r2 <= 1e-16f)
            continue;
          float r = sqrtf(r2);
          float2 gradW = gradW_cubic(rij, r, h);
          float rhoj = expf(s[j]);
          float pj = press[j];
          float common = -mass * (pi / (rhoi * rhoi) + pj / (rhoj * rhoj));
          ai.x += common * gradW.x;
          ai.y += common * gradW.y;
          if (useVisc) {
            float2 vj = vel[j];
            float2 vij = make_float2(vi.x - vj.x, vi.y - vj.y);
            float dot = vij.x * rij.x + vij.y * rij.y;
            if (dot < 0.f) {
              float mu = (h * dot) / (r2 + 0.01f * h * h);
              float rhoBar = 0.5f * (rhoi + rhoj);
              float Pi_ij = (-viscAlpha * c0 * mu) / rhoBar; // β term omitted
              ai.x += -mass * Pi_ij * gradW.x;
              ai.y += -mass * Pi_ij * gradW.y;
            }
          }
        }
    }
  if (useGrav) {
    ai.x += gx;
    ai.y += gy;
  }
  acc[i] = ai;
}

__global__ void
k_xsph_cell(const float2 *__restrict__ pos, const float2 *__restrict__ vel,
            const float *__restrict__ s, float2 *__restrict__ dvel,
            const int *__restrict__ cellHead, const int *__restrict__ next,
            int N, float mass, float h, float eps, int Gx, int Gy, float cell) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;
  float2 xi = pos[i];
  float2 vi = vel[i];
  float rhoi = expf(s[i]);
  float2 dv = make_float2(0.f, 0.f);
  int gx_i = grid_x(xi.x, cell, Gx);
  int gy_i = grid_y(xi.y, cell, Gy);
  float twoh = 2.f * h;
  float twoh2 = twoh * twoh;
  for (int oy = -1; oy <= 1; ++oy)
    for (int ox = -1; ox <= 1; ++ox) {
      int cx = gx_i + ox, cy = gy_i + oy;
      if ((unsigned)cx >= (unsigned)Gx || (unsigned)cy >= (unsigned)Gy)
        continue;
      int head = cellHead[cy * Gx + cx];
      for (int j = head; j != -1; j = next[j])
        if (j != i) {
          float2 rij = make_float2(xi.x - pos[j].x, xi.y - pos[j].y);
          float r2 = rij.x * rij.x + rij.y * rij.y;
          if (r2 >= twoh2)
            continue;
          float r = sqrtf(r2);
          float w = W_cubic(r, h);
          float rhoj = expf(s[j]);
          float rhoBar = 0.5f * (rhoi + rhoj);
          float2 vij = make_float2(vel[j].x - vi.x, vel[j].y - vi.y);
          dv.x += (mass / rhoBar) * vij.x * w;
          dv.y += (mass / rhoBar) * vij.y * w;
        }
    }
  dvel[i].x = eps * dv.x;
  dvel[i].y = eps * dv.y;
}

__global__ void k_apply_xsph(float2 *__restrict__ vel,
                             const float2 *__restrict__ dvel, int N) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    vel[i].x += dvel[i].x;
    vel[i].y += dvel[i].y;
  }
}

__global__ void k_integrate(float2 *__restrict__ pos, float2 *__restrict__ vel,
                            const float2 *__restrict__ acc, int N, float dt,
                            float boxX, float boxY) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;
  float2 v = vel[i];
  float2 x = pos[i];
  v.x += acc[i].x * dt;
  v.y += acc[i].y * dt;
  x.x += v.x * dt;
  x.y += v.y * dt;
  const float e = 0.2f;
  if (x.x < 0.f) {
    x.x = 0.f;
    v.x = -e * v.x;
  }
  if (x.x > boxX) {
    x.x = boxX;
    v.x = -e * v.x;
  }
  if (x.y < 0.f) {
    x.y = 0.f;
    v.y = -e * v.y;
  }
  if (x.y > boxY) {
    x.y = boxY;
    v.y = -e * v.y;
  }
  pos[i] = x;
  vel[i] = v;
}

__global__ void k_clear_grid(int *__restrict__ grid, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
    grid[i] = 0;
}

__global__ void k_rasterize(const float2 *__restrict__ pos, int N,
                            int *__restrict__ grid2, int W, int H, float boxX,
                            float boxY) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;
  float2 p = pos[i];
  int cx = (int)(p.x / boxX * (W - 1));
  int sy = (int)((boxY - p.y) / boxY * (2 * H - 1)); // flip y
  if ((unsigned)cx < (unsigned)W && (unsigned)sy < (unsigned)(2 * H))
    atomicAdd(&grid2[sy * W + cx], 1);
}

// rain
__global__ void k_rain(float2 *pos, float2 *vel, int N, int nspawn, float boxX,
                       float boxY, float c0, unsigned seed) {
  int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k >= nspawn)
    return;
  unsigned s = seed ^ (k * 1664525u + 1013904223u);
  s = s * 1664525u + 1013904223u;
  float rx = (s & 0x00FFFFFF) / 16777216.f;
  s = s * 1664525u + 1013904223u;
  float x = rx * (boxX * 0.8f) + 0.1f * boxX;
  float ry = (s & 0x00FFFFFF) / 16777216.f;
  float y = boxY * (0.9f + 0.08f * ry);
  int i = (s % N);
  pos[i] = make_float2(x, y);
  vel[i] = make_float2(0.f, -0.5f * c0);
}

static void parse_args(int argc, char **argv, Params &P) {
  const struct option longopts[] = {
      {"n", required_argument, 0, 'n'},
      {"box", required_argument, 0, 'b'},
      {"dTau", required_argument, 0, 't'},
      {"rho0", required_argument, 0, 'r'},
      {"c0", required_argument, 0, 'c'},
      {"gamma", required_argument, 0, 'g'},
      {"CFL", required_argument, 0, 'f'},
      {"hMul", required_argument, 0, 'h'},
      {"visc", required_argument, 0, 'v'},
      {"gravity", required_argument, 0, 'y'},
      {"fps", required_argument, 0, 'p'},
      {"fpscap", required_argument, 0, 'F'},
      {"stride", required_argument, 0, 'S'},
      {"seed", required_argument, 0, 's'},
      {"rain", no_argument, 0, 'R'},
      {"headless", no_argument, 0, 'H'},
      {"halfblocks", no_argument, 0, 'B'},
      {"visc_substeps", required_argument, 0, 'k'},
      {"muscl", no_argument, 0, 'm'},
      {"xsph_eps", required_argument, 0, 'x'},
      {0, 0, 0, 0}};
  int c;
  while ((c = getopt_long(argc, argv, "n:b:t:r:c:g:f:h:v:y:p:F:S:s:RHBk:mx:",
                          longopts, nullptr)) != -1) {
    switch (c) {
    case 'n':
      P.N = atoi(optarg);
      break;
    case 'b':
      sscanf(optarg, "%fx%f", &P.boxX, &P.boxY);
      break;
    case 't':
      P.dTau = atof(optarg);
      break;
    case 'r':
      P.rho0 = atof(optarg);
      break;
    case 'c':
      P.c0 = atof(optarg);
      break;
    case 'g':
      P.gammaEOS = atof(optarg);
      break;
    case 'f':
      P.CFL = atof(optarg);
      break;
    case 'h':
      P.hMul = atof(optarg);
      break;
    case 'v':
      P.viscAlpha = atof(optarg);
      break;
    case 'y':
      P.gravity = atof(optarg);
      P.useGrav = (P.gravity != 0.f);
      break;
    case 'p':
      P.fps = atoi(optarg);
      break;
    case 'F':
      P.fps_cap = atoi(optarg);
      break;
    case 'S':
      P.stride = atoi(optarg);
      if (P.stride < 1)
        P.stride = 1;
      break;
    case 's':
      P.seed = atoi(optarg);
      break;
    case 'R':
      P.rain = true;
      break;
    case 'H':
      P.headless = true;
      break;
    case 'B':
      P.halfblocks = true;
      break;
    case 'k':
      P.viscSub = atoi(optarg);
      if (P.viscSub < 1)
        P.viscSub = 1;
      break;
    case 'm':
      P.useXSPH = true;
      break;
    case 'x':
      P.xsphEps = atof(optarg);
      P.useXSPH = (P.xsphEps > 0.f) || P.useXSPH;
      break;
    default:
      break;
    }
  }
}

static void reset_particles(const Params &P, std::vector<float2> &h_pos,
                            std::vector<float2> &h_vel) {
  std::mt19937 rng(P.seed);
  std::uniform_real_distribution<float> U(0.f, 1.f);
  int nSide = (int)sqrtf((float)P.N);
  int nx = nSide, ny = (P.N + nSide - 1) / nSide;
  float padX = 0.05f * P.boxX, padY = 0.05f * P.boxY;
  float width = P.boxX - 2 * padX, height = 0.6f * P.boxY - padY;
  for (int i = 0; i < P.N; ++i) {
    int ix = i % nx, iy = i / nx;
    float fx = (ix + 0.5f) / nx, fy = (iy + 0.5f) / ny;
    float x = padX + fx * width, y = padY + fy * height;
    x += (U(rng) - 0.5f) * 0.2f * width / nx;
    y += (U(rng) - 0.5f) * 0.2f * height / ny;
    h_pos[i] = make_float2(x, y);
    h_vel[i] = make_float2(0.f, 0.f);
  }
}

static void ensure_cell_buffers(DevState &d, int N, float boxX, float boxY,
                                float h) {
  float cell = 2.0f * h;
  int Gx = (int)ceilf(boxX / cell);
  int Gy = (int)ceilf(boxY / cell);
  if (Gx < 1)
    Gx = 1;
  if (Gy < 1)
    Gy = 1;
  int M = Gx * Gy;
  bool needAlloc = (d.cellHead == nullptr) || (d.Gx != Gx) || (d.Gy != Gy);
  if (needAlloc) {
    if (d.cellHead) {
      cudaFree(d.cellHead);
      d.cellHead = nullptr;
    }
    if (d.next) {
      cudaFree(d.next);
      d.next = nullptr;
    }
    CUDA_CHECK(cudaMalloc(&d.cellHead, M * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d.next, N * sizeof(int)));
    d.Gx = Gx;
    d.Gy = Gy;
    d.cell = cell;
  } else {
    d.cell = cell;
  }
}

static void ensure_grid2_buffers(DevState &d, int W, int H) {
  size_t need = (size_t)(2 * H) * (size_t)W;
  if (need == d.grid2Size && d.W == W && d.H == H && d.grid2)
    return;
  if (d.grid2)
    cudaFree(d.grid2);
  CUDA_CHECK(cudaMalloc(&d.grid2, need * sizeof(int)));
  d.grid2Size = need;
  d.W = W;
  d.H = H;
}

int main(int argc, char **argv) {
  Params P;
  parse_args(argc, argv, P);

  std::vector<float2> h_pos(P.N), h_vel(P.N);
  DevState d{};
  memset(&d, 0, sizeof(d));
  CUDA_CHECK(cudaMalloc(&d.pos, P.N * sizeof(float2)));
  CUDA_CHECK(cudaMalloc(&d.vel, P.N * sizeof(float2)));
  CUDA_CHECK(cudaMalloc(&d.acc, P.N * sizeof(float2)));
  CUDA_CHECK(cudaMalloc(&d.s, P.N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d.press, P.N * sizeof(float)));

  reset_particles(P, h_pos, h_vel);
  CUDA_CHECK(cudaMemcpy(d.pos, h_pos.data(), P.N * sizeof(float2),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.vel, h_vel.data(), P.N * sizeof(float2),
                        cudaMemcpyHostToDevice));

  const float area = P.boxX * P.boxY;
  const float mass = (P.rho0 * area) / P.N;
  const float spacing = sqrtf(area / P.N);
  float h = P.hMul * spacing; // smoothing length
  float tau = 0.f;
  float t = P.t0 * expf(tau);
  int W = 0, H = 0;
  int *h_grid2 = nullptr;
  size_t h_grid2_bytes = 0;
  if (!P.headless) {
    curses_active = true;
    setlocale(LC_ALL, "");
    initscr();
    cbreak();
    noecho();
    curs_set(0);
    nodelay(stdscr, TRUE);
    keypad(stdscr, TRUE);
    if (has_colors()) {
      start_color();
      use_default_colors();
    }
    getmaxyx(stdscr, H, W);
    ensure_grid2_buffers(d, W, H);
    h_grid2_bytes = d.grid2Size * sizeof(int);
    CUDA_CHECK(
        cudaHostAlloc((void **)&h_grid2, h_grid2_bytes, cudaHostAllocDefault));
  }

  ensure_cell_buffers(d, P.N, P.boxX, P.boxY, h);

  auto frame_sleep = [&](double dtw) {
    if (P.fps_cap > 0) {
      double target = 1.0 / P.fps_cap;
      if (dtw < target) {
        int us = (int)((target - dtw) * 1e6);
        if (us > 0)
          usleep(us);
      }
    }
  };

  bool running = true, paused = false, stepOnce = false;
  auto last = std::chrono::high_resolution_clock::now();
  double avgfps = 0.0, fps = 0.0;
  int step = 0;
  float rain_carry = 0.f;

  while (running) {
    if (!P.headless) {
      int ch;
      while ((ch = getch()) != ERR) {
        if (ch == 'q') {
          running = false;
          break;
        } else if (ch == 'p')
          paused = !paused;
        else if (ch == ' ')
          stepOnce = true;
        else if (ch == 'r') {
          reset_particles(P, h_pos, h_vel);
          CUDA_CHECK(cudaMemcpy(d.pos, h_pos.data(), P.N * sizeof(float2),
                                cudaMemcpyHostToDevice));
          CUDA_CHECK(cudaMemcpy(d.vel, h_vel.data(), P.N * sizeof(float2),
                                cudaMemcpyHostToDevice));
        } else if (ch == 'g')
          P.useGrav = !P.useGrav;
        else if (ch == 'v')
          P.useVisc = !P.useVisc;
        else if (ch == '=') {
          h *= 1.05f;
          ensure_cell_buffers(d, P.N, P.boxX, P.boxY, h);
        } else if (ch == '-') {
          h *= 0.95f;
          ensure_cell_buffers(d, P.N, P.boxX, P.boxY, h);
        } else if (ch == ']')
          P.c0 *= 1.05f;
        else if (ch == '[')
          P.c0 *= 0.95f;
        else if (ch == '>')
          P.dTau *= 1.2f;
        else if (ch == '<')
          P.dTau *= (1 / 1.2f);
      }
    }

    bool doStep = paused ? stepOnce : true;
    stepOnce = false;
    float dt_eff = 0.f;
    float dTau_accum = 0.f;
    int K = (P.viscSub > 0 ? P.viscSub : 1);

    if (doStep) {
      float dt_try = t * P.dTau;
      float dt_cfl = P.CFL * h / (P.c0 * (1.0f + 2.0f * P.viscAlpha));
      dt_eff = fminf(dt_try, dt_cfl);
      float dt_sub = dt_eff / K;

      int BS = 256;
      int GS = (P.N + BS - 1) / BS;
      int M = d.Gx * d.Gy;
      int GSm = (M + BS - 1) / BS;

      for (int k = 0; k < K; ++k) {
        k_clear_heads<<<GSm, BS>>>(d.cellHead, M);
        CUDA_CHECK(cudaPeekAtLastError());
        k_build_cells<<<GS, BS>>>(d.pos, P.N, d.cellHead, d.next, d.Gx, d.Gy,
                                  d.cell);
        CUDA_CHECK(cudaPeekAtLastError());

        k_density_pressure_cell<<<GS, BS>>>(d.pos, d.s, d.press, d.cellHead,
                                            d.next, P.N, mass, h, P.rho0, P.c0,
                                            P.gammaEOS, d.Gx, d.Gy, d.cell);
        CUDA_CHECK(cudaPeekAtLastError());

        k_forces_cell<<<GS, BS>>>(d.pos, d.vel, d.s, d.press, d.acc, d.cellHead,
                                  d.next, P.N, mass, h, P.viscAlpha, P.c0, 0.f,
                                  -(P.useGrav ? P.gravity : 0.f), P.useVisc,
                                  P.useGrav, d.Gx, d.Gy, d.cell);
        CUDA_CHECK(cudaPeekAtLastError());

        k_integrate<<<GS, BS>>>(d.pos, d.vel, d.acc, P.N, dt_sub, P.boxX,
                                P.boxY);
        CUDA_CHECK(cudaPeekAtLastError());

        if (P.useXSPH && P.xsphEps > 0.f) {
          k_xsph_cell<<<GS, BS>>>(d.pos, d.vel, d.s, d.acc, d.cellHead, d.next,
                                  P.N, mass, h, P.xsphEps, d.Gx, d.Gy, d.cell);
          CUDA_CHECK(cudaPeekAtLastError());
          k_apply_xsph<<<GS, BS>>>(d.vel, d.acc, P.N);
          CUDA_CHECK(cudaPeekAtLastError());
        }

        if (P.rain) {
          rain_carry += 0.02f * P.N * dt_sub;
          int nspawn = (int)rain_carry;
          rain_carry -= nspawn;
          if (nspawn > 0) {
            int BSr = 128, GSr = (nspawn + BSr - 1) / BSr;
            k_rain<<<GSr, BSr>>>(d.pos, d.vel, P.N, nspawn, P.boxX, P.boxY,
                                 P.c0, (unsigned)(P.seed + step));
            CUDA_CHECK(cudaPeekAtLastError());
          }
        }

        float dTau_actual = dt_sub / fmaxf(t, 1e-9f);
        dTau_accum += dTau_actual;
        t = P.t0 * expf(tau + dTau_accum);
      }
      tau += dTau_accum;
    }

    if (!P.headless && (step % P.stride == 0)) {
      auto now = std::chrono::high_resolution_clock::now();
      double dtw = std::chrono::duration<double>(now - last).count();
      fps = (dtw > 0) ? (1.0 / dtw) : 0.0;
      avgfps = 0.95 * avgfps + 0.05 * fps;
      last = now;

      int hH, hW;
      getmaxyx(stdscr, hH, hW);
      if (hH != H || hW != W) {
        H = hH;
        W = hW;
        ensure_grid2_buffers(d, W, H);
        if (h_grid2) {
          cudaFreeHost(h_grid2);
          h_grid2 = nullptr;
        }
        h_grid2_bytes = d.grid2Size * sizeof(int);
        CUDA_CHECK(cudaHostAlloc((void **)&h_grid2, h_grid2_bytes,
                                 cudaHostAllocDefault));
      }

      int BSg = 256;
      int GSg = ((int)d.grid2Size + BSg - 1) / BSg;
      k_clear_grid<<<GSg, BSg>>>(d.grid2, (int)d.grid2Size);
      CUDA_CHECK(cudaPeekAtLastError());
      int BSp = 256, GSp = (P.N + BSp - 1) / BSp;
      k_rasterize<<<GSp, BSp>>>(d.pos, P.N, d.grid2, W, H, P.boxX, P.boxY);
      CUDA_CHECK(cudaPeekAtLastError());

      CUDA_CHECK(
          cudaMemcpy(h_grid2, d.grid2, h_grid2_bytes, cudaMemcpyDeviceToHost));

      erase();
      mvprintw(0, 2,
               "SPH2D | N=%d h=%.3g c0=%.2g α=%.2g γ=%.1f ρ0=%.0f g=%s visc=%s "
               " τ=%.4g t=%.4g  dt=%.3g  FPS:%.1f(%.1f) stride=%d K=%d",
               P.N, h, P.c0, P.viscAlpha, P.gammaEOS, P.rho0,
               P.useGrav ? "on" : "off", P.useVisc ? "on" : "off", tau, t,
               dt_eff, fps, avgfps, P.stride, P.viscSub);
      mvprintw(1, 2,
               "Keys: q quit | p pause | space step | r reset | g gravity | v "
               "viscosity | +/- h | ]/[ c0 | </> dτ  xsph=%s ε=%.2f",
               P.useXSPH ? "on" : "off", P.xsphEps);

      for (int y = 0; y < H; ++y) {
        move(y + 2, 0);
        int sy0 = 2 * y, sy1 = 2 * y + 1;
        for (int x = 0; x < W; ++x) {
          int top = h_grid2[sy0 * W + x];
          int bot = h_grid2[sy1 * W + x];
          int g = (top > 0 ? (bot > 0 ? GL_FULL : GL_TOP)
                           : (bot > 0 ? GL_BOTTOM : GL_NONE));
          addwstr(HALF_BLOCKS[g]);
        }
      }
      refresh();
      frame_sleep(dtw);
    }

    if (P.headless && (step % P.stride == 0)) {
      if (step % (100 * P.stride) == 0) {
        printf("step %d  t=%.3g τ=%.3g\n", step, t, tau);
        fflush(stdout);
      }
    }

    ++step;
  }

  if (!P.headless && curses_active) {
    endwin();
    curses_active = false;
  }
  if (h_grid2)
    cudaFreeHost(h_grid2);

  cudaFree(d.pos);
  cudaFree(d.vel);
  cudaFree(d.acc);
  cudaFree(d.s);
  cudaFree(d.press);
  if (d.cellHead)
    cudaFree(d.cellHead);
  if (d.next)
    cudaFree(d.next);
  if (d.grid2)
    cudaFree(d.grid2);
  return 0;
}
