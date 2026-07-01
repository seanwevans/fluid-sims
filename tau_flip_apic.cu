// tau_flip_apic.cu
// 2-D hybrid FLIP/APIC incompressible fluid simulator.
//
// Build:
//   nvcc -std=c++17 -O3 -use_fast_math -arch=sm_86 tau_flip_apic.cu -o tau_flip_apic -lncursesw
//
// Run:
//   ./tau_flip_apic --particles 65536 --grid 128 --apic 0.85 --flip 0.97
//   ./tau_flip_apic --headless --steps 600 --stride 20

#include <cuda_runtime.h>

#include <chrono>
#include <getopt.h>
#include <locale.h>
#include <math.h>
#include <ncursesw/curses.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <vector>
#include <wchar.h>

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
  int particles = 1 << 16;
  int grid = 128;
  int jacobi = 48;
  int steps = 0;
  int stride = 1;
  int fps = 60;
  bool headless = false;
  float dt = 0.004f;
  float gravity = 7.5f;
  float flip = 0.97f;
  float apic = 0.85f;
  float jitter = 0.22f;
  int seed = 1337;
};

struct DevState {
  float2 *pos, *vel, *vel_pic, *affine_x, *affine_y;
  float *mass, *u, *v, *u_prev, *v_prev, *u_proj, *v_proj, *p, *p_next, *div;
  int *density;
};

__device__ __forceinline__ int gix(int i, int j, int n) { return i + n * j; }
__device__ __forceinline__ float clampf(float x, float lo, float hi) {
  return fminf(fmaxf(x, lo), hi);
}
__device__ __forceinline__ float w1(float x) {
  x = fabsf(x);
  return x < 1.0f ? 1.0f - x : 0.0f;
}

__global__ void k_seed(float2 *pos, float2 *vel, float2 *ax, float2 *ay, int np,
                       int seed, float jitter) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= np)
    return;
  int side = (int)ceilf(sqrtf((float)np));
  int ix = id % side;
  int iy = id / side;
  unsigned h = (unsigned)(id * 747796405u + seed * 2891336453u);
  h = (h ^ (h >> 16)) * 2246822519u;
  float rx = ((h & 1023u) / 1023.0f - 0.5f) * jitter;
  float ry = (((h >> 10) & 1023u) / 1023.0f - 0.5f) * jitter;
  float x = 0.12f + 0.45f * ((ix + 0.5f + rx) / side);
  float y = 0.12f + 0.74f * ((iy + 0.5f + ry) / side);
  pos[id] = make_float2(clampf(x, 0.02f, 0.98f), clampf(y, 0.02f, 0.98f));
  float cx = x - 0.38f, cy = y - 0.55f;
  vel[id] = make_float2(-1.8f * cy, 1.8f * cx);
  ax[id] = make_float2(0, 0);
  ay[id] = make_float2(0, 0);
}

__global__ void k_clear_grid(float *mass, float *u, float *v, float *u_prev,
                             float *v_prev, float *u_proj, float *v_proj,
                             float *p, float *p_next, float *div, int *density,
                             int cells) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= cells)
    return;
  mass[id] = u[id] = v[id] = u_prev[id] = v_prev[id] = u_proj[id] = v_proj[id] =
      p[id] = p_next[id] = div[id] = 0.0f;
  density[id] = 0;
}

__global__ void k_p2g(const float2 *pos, const float2 *vel, const float2 *ax,
                      const float2 *ay, float *mass, float *u, float *v,
                      int np, int n, float apic) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= np)
    return;
  float gx = pos[id].x * (n - 1), gy = pos[id].y * (n - 1);
  int base_x = (int)floorf(gx), base_y = (int)floorf(gy);
  for (int oy = -1; oy <= 1; ++oy) {
    int j = min(max(base_y + oy, 0), n - 1);
    float wy = w1(gy - j);
    for (int ox = -1; ox <= 1; ++ox) {
      int i = min(max(base_x + ox, 0), n - 1);
      float wx = w1(gx - i), wt = wx * wy;
      if (wt <= 0.0f)
        continue;
      float2 r = make_float2((i - gx) / (n - 1), (j - gy) / (n - 1));
      float2 vv = vel[id];
      vv.x += apic * (ax[id].x * r.x + ay[id].x * r.y);
      vv.y += apic * (ax[id].y * r.x + ay[id].y * r.y);
      int gi = gix(i, j, n);
      atomicAdd(&mass[gi], wt);
      atomicAdd(&u[gi], wt * vv.x);
      atomicAdd(&v[gi], wt * vv.y);
    }
  }
}

__global__ void k_normalize_forces(float *mass, float *u, float *v, float *u_prev,
                                   float *v_prev, int n, float dt,
                                   float gravity) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= n * n)
    return;
  if (mass[id] > 1e-8f) {
    u[id] /= mass[id];
    v[id] = v[id] / mass[id] - gravity * dt;
  }
  int i = id % n, j = id / n;
  if (i == 0 || i == n - 1)
    u[id] = 0.0f;
  if (j == 0 || j == n - 1)
    v[id] = 0.0f;
  u_prev[id] = u[id];
  v_prev[id] = v[id];
}

__global__ void k_divergence(const float *u, const float *v, float *div, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i >= n - 1 || j >= n - 1)
    return;
  div[gix(i, j, n)] = -0.5f * (n - 1) *
                      (u[gix(i + 1, j, n)] - u[gix(i - 1, j, n)] +
                       v[gix(i, j + 1, n)] - v[gix(i, j - 1, n)]);
}

__global__ void k_jacobi(const float *p, float *pn, const float *div, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i >= n - 1 || j >= n - 1)
    return;
  pn[gix(i, j, n)] = (div[gix(i, j, n)] + p[gix(i - 1, j, n)] +
                      p[gix(i + 1, j, n)] + p[gix(i, j - 1, n)] +
                      p[gix(i, j + 1, n)]) *
                     0.25f;
}

__global__ void k_project(float *u, float *v, float *u_proj, float *v_proj,
                          const float *p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i >= n - 1 || j >= n - 1)
    return;
  int id = gix(i, j, n);
  u_proj[id] = u[id] - 0.5f * (p[gix(i + 1, j, n)] - p[gix(i - 1, j, n)]) /
                           (n - 1);
  v_proj[id] = v[id] - 0.5f * (p[gix(i, j + 1, n)] - p[gix(i, j - 1, n)]) /
                           (n - 1);
}

__device__ float2 sample_grid(const float *u, const float *v, float2 p, int n) {
  float gx = clampf(p.x * (n - 1), 0.0f, n - 1.001f);
  float gy = clampf(p.y * (n - 1), 0.0f, n - 1.001f);
  int i0 = (int)floorf(gx), j0 = (int)floorf(gy);
  int i1 = min(i0 + 1, n - 1), j1 = min(j0 + 1, n - 1);
  float tx = gx - i0, ty = gy - j0;
  float u00 = u[gix(i0, j0, n)], u10 = u[gix(i1, j0, n)];
  float u01 = u[gix(i0, j1, n)], u11 = u[gix(i1, j1, n)];
  float v00 = v[gix(i0, j0, n)], v10 = v[gix(i1, j0, n)];
  float v01 = v[gix(i0, j1, n)], v11 = v[gix(i1, j1, n)];
  return make_float2((1 - tx) * ((1 - ty) * u00 + ty * u01) +
                         tx * ((1 - ty) * u10 + ty * u11),
                     (1 - tx) * ((1 - ty) * v00 + ty * v01) +
                         tx * ((1 - ty) * v10 + ty * v11));
}

__global__ void k_g2p(float2 *pos, float2 *vel, float2 *pic, float2 *ax,
                      float2 *ay, const float *u_prev, const float *v_prev,
                      const float *u_proj, const float *v_proj, int *density,
                      int np, int n, float dt, float flip_blend) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id >= np)
    return;
  float2 p = pos[id];
  float2 newv = sample_grid(u_proj, v_proj, p, n);
  float2 oldv = sample_grid(u_prev, v_prev, p, n);
  float2 flipv = make_float2(vel[id].x + newv.x - oldv.x,
                             vel[id].y + newv.y - oldv.y);
  vel[id] = make_float2((1 - flip_blend) * newv.x + flip_blend * flipv.x,
                        (1 - flip_blend) * newv.y + flip_blend * flipv.y);
  pic[id] = newv;

  float h = 1.0f / (n - 1);
  float2 vx1 = sample_grid(u_proj, v_proj, make_float2(p.x + h, p.y), n);
  float2 vx0 = sample_grid(u_proj, v_proj, make_float2(p.x - h, p.y), n);
  float2 vy1 = sample_grid(u_proj, v_proj, make_float2(p.x, p.y + h), n);
  float2 vy0 = sample_grid(u_proj, v_proj, make_float2(p.x, p.y - h), n);
  ax[id] = make_float2(0.5f * (vx1.x - vx0.x) / h, 0.5f * (vx1.y - vx0.y) / h);
  ay[id] = make_float2(0.5f * (vy1.x - vy0.x) / h, 0.5f * (vy1.y - vy0.y) / h);

  p.x += vel[id].x * dt;
  p.y += vel[id].y * dt;
  if (p.x < 0.01f || p.x > 0.99f) {
    vel[id].x *= -0.35f;
    p.x = clampf(p.x, 0.01f, 0.99f);
  }
  if (p.y < 0.01f || p.y > 0.99f) {
    vel[id].y *= -0.35f;
    p.y = clampf(p.y, 0.01f, 0.99f);
  }
  pos[id] = p;
  int rx = min(max((int)(p.x * n), 0), n - 1);
  int ry = min(max((int)(p.y * n), 0), n - 1);
  atomicAdd(&density[gix(rx, ry, n)], 1);
}

static void alloc_state(DevState &s, const Params &p) {
  size_t ps = p.particles * sizeof(float2);
  size_t gs = p.grid * p.grid * sizeof(float);
  size_t is = p.grid * p.grid * sizeof(int);
  CUDA_CHECK(cudaMalloc(&s.pos, ps));
  CUDA_CHECK(cudaMalloc(&s.vel, ps));
  CUDA_CHECK(cudaMalloc(&s.vel_pic, ps));
  CUDA_CHECK(cudaMalloc(&s.affine_x, ps));
  CUDA_CHECK(cudaMalloc(&s.affine_y, ps));
  CUDA_CHECK(cudaMalloc(&s.mass, gs));
  CUDA_CHECK(cudaMalloc(&s.u, gs));
  CUDA_CHECK(cudaMalloc(&s.v, gs));
  CUDA_CHECK(cudaMalloc(&s.u_prev, gs));
  CUDA_CHECK(cudaMalloc(&s.v_prev, gs));
  CUDA_CHECK(cudaMalloc(&s.u_proj, gs));
  CUDA_CHECK(cudaMalloc(&s.v_proj, gs));
  CUDA_CHECK(cudaMalloc(&s.p, gs));
  CUDA_CHECK(cudaMalloc(&s.p_next, gs));
  CUDA_CHECK(cudaMalloc(&s.div, gs));
  CUDA_CHECK(cudaMalloc(&s.density, is));
}

static void step(DevState &s, const Params &p) {
  int cells = p.grid * p.grid;
  int tb = 256, pb = (p.particles + tb - 1) / tb, gb = (cells + tb - 1) / tb;
  dim3 bs(16, 16), gr((p.grid + bs.x - 1) / bs.x, (p.grid + bs.y - 1) / bs.y);
  k_clear_grid<<<gb, tb>>>(s.mass, s.u, s.v, s.u_prev, s.v_prev, s.u_proj,
                           s.v_proj, s.p, s.p_next, s.div, s.density, cells);
  k_p2g<<<pb, tb>>>(s.pos, s.vel, s.affine_x, s.affine_y, s.mass, s.u, s.v,
                    p.particles, p.grid, p.apic);
  k_normalize_forces<<<gb, tb>>>(s.mass, s.u, s.v, s.u_prev, s.v_prev, p.grid,
                                 p.dt, p.gravity);
  k_divergence<<<gr, bs>>>(s.u, s.v, s.div, p.grid);
  for (int i = 0; i < p.jacobi; ++i) {
    k_jacobi<<<gr, bs>>>(s.p, s.p_next, s.div, p.grid);
    float *tmp = s.p;
    s.p = s.p_next;
    s.p_next = tmp;
  }
  k_project<<<gr, bs>>>(s.u, s.v, s.u_proj, s.v_proj, s.p, p.grid);
  k_g2p<<<pb, tb>>>(s.pos, s.vel, s.vel_pic, s.affine_x, s.affine_y, s.u_prev,
                    s.v_prev, s.u_proj, s.v_proj, s.density, p.particles,
                    p.grid, p.dt, p.flip);
  CUDA_CHECK(cudaGetLastError());
}

static void render(const std::vector<int> &density, int n, int step_no) {
  static const wchar_t ramp[] = {L' ', L'░', L'▒', L'▓', L'█'};
  int rows, cols;
  getmaxyx(stdscr, rows, cols);
  mvprintw(0, 0, "FLIP/APIC step %d   (press 'q' to quit)", step_no);
  for (int y = 0; y < rows - 1; ++y) {
    int gy = n - 1 - (int)((float)y * n / (rows - 1));
    move(y + 1, 0);
    for (int x = 0; x < cols; ++x) {
      int gx = (int)((float)x * n / cols);
      int d = density[gix(gx, gy, n)];
      int idx = d > 12 ? 4 : d > 6 ? 3 : d > 2 ? 2 : d > 0 ? 1 : 0;
      addnwstr(&ramp[idx], 1);
    }
    clrtoeol();
  }
  refresh();
}

static void usage(const char *argv0) {
  printf("Usage: %s [--particles N] [--grid N] [--steps N] [--headless]\n", argv0);
  printf("       [--dt S] [--gravity G] [--flip 0..1] [--apic 0..1] [--jacobi N]\n");
}

int main(int argc, char **argv) {
  Params p;
  static option opts[] = {{"particles", required_argument, 0, 'n'},
                          {"grid", required_argument, 0, 'g'},
                          {"steps", required_argument, 0, 's'},
                          {"stride", required_argument, 0, 'r'},
                          {"headless", no_argument, 0, 'H'},
                          {"dt", required_argument, 0, 'd'},
                          {"gravity", required_argument, 0, 'G'},
                          {"flip", required_argument, 0, 'f'},
                          {"apic", required_argument, 0, 'a'},
                          {"jacobi", required_argument, 0, 'j'},
                          {"help", no_argument, 0, '?'},
                          {0, 0, 0, 0}};
  for (;;) {
    int c = getopt_long(argc, argv, "n:g:s:r:Hd:G:f:a:j:?", opts, nullptr);
    if (c == -1)
      break;
    switch (c) {
    case 'n': p.particles = atoi(optarg); break;
    case 'g': p.grid = atoi(optarg); break;
    case 's': p.steps = atoi(optarg); break;
    case 'r': p.stride = atoi(optarg); break;
    case 'H': p.headless = true; break;
    case 'd': p.dt = strtof(optarg, nullptr); break;
    case 'G': p.gravity = strtof(optarg, nullptr); break;
    case 'f': p.flip = strtof(optarg, nullptr); break;
    case 'a': p.apic = strtof(optarg, nullptr); break;
    case 'j': p.jacobi = atoi(optarg); break;
    default: usage(argv[0]); return 0;
    }
  }
  p.grid = p.grid < 16 ? 16 : p.grid;
  p.flip = fminf(fmaxf(p.flip, 0.0f), 1.0f);
  p.apic = fminf(fmaxf(p.apic, 0.0f), 1.0f);

  DevState s{};
  alloc_state(s, p);
  int tb = 256, pb = (p.particles + tb - 1) / tb;
  k_seed<<<pb, tb>>>(s.pos, s.vel, s.affine_x, s.affine_y, p.particles,
                     p.seed, p.jitter);
  CUDA_CHECK(cudaGetLastError());

  std::vector<int> h_density(p.grid * p.grid);
  if (!p.headless) {
    setlocale(LC_ALL, "");
    initscr();
    cbreak();
    noecho();
    nodelay(stdscr, TRUE);
    curs_set(0);
    curses_active = true;
  }

  auto last = std::chrono::steady_clock::now();
  for (int frame = 0; p.steps == 0 || frame < p.steps; ++frame) {
    step(s, p);
    if (frame % p.stride == 0) {
      CUDA_CHECK(cudaMemcpy(h_density.data(), s.density,
                            h_density.size() * sizeof(int),
                            cudaMemcpyDeviceToHost));
      if (p.headless) {
        long long occupied = 0, peak = 0;
        for (int d : h_density) {
          occupied += d > 0;
          peak = peak > d ? peak : d;
        }
        printf("step=%d occupied=%lld peak_cell=%lld\n", frame, occupied, peak);
      } else {
        render(h_density, p.grid, frame);
        int ch = getch();
        if (ch == 'q' || ch == 'Q')
          break;
        auto now = std::chrono::steady_clock::now();
        int target = 1000000 / (p.fps > 0 ? p.fps : 60);
        int elapsed = (int)std::chrono::duration_cast<std::chrono::microseconds>(
                          now - last)
                          .count();
        if (elapsed < target)
          usleep(target - elapsed);
        last = std::chrono::steady_clock::now();
      }
    }
  }
  if (curses_active)
    endwin();
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
