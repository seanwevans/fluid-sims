// tau_mpm.cu
// 2-D Material Point Method (MPM) elastoplastic material simulator
//
// Build
//   nvcc -std=c++17 -O3 -use_fast_math -arch=sm_86 -lineinfo tau_mpm.cu \
//   -o tau_mpm -lncursesw
//
// Run
//   ./tau_mpm --n 32768 --grid 96x96 --dt 8e-5 --steps 20000
//   ./tau_mpm --n 65536 --headless --steps 1200 --material snow

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
  int N = 1 << 15;
  int Gx = 96, Gy = 96;
  float boxX = 1.0f, boxY = 1.0f;
  float dt = 8.0e-5f;
  int steps = 0;
  int stride = 2;
  int fps = 120;
  int seed = 2026;
  bool headless = false;
  bool halfblocks = true;
  float gravity = 9.81f;
  float particleMass = 1.0f;
  float volume = 1.0f;
  float hardening = 10.0f;
  float mu0 = 18.0f;
  float lambda0 = 40.0f;
  float criticalCompression = 2.5e-2f;
  float criticalStretch = 7.5e-3f;
  enum Material { MUD, SNOW, SAND } material = SNOW;
};

enum { GL_NONE = 0, GL_TOP = 1, GL_BOTTOM = 2, GL_FULL = 3, GL_COUNT };
static const wchar_t *HALF_BLOCKS[GL_COUNT] = {L" ", L"▀", L"▄", L"█"};

struct Mat2 {
  float a, b, c, d;
};

struct DevState {
  float2 *pos = nullptr;
  float2 *vel = nullptr;
  Mat2 *F = nullptr;
  float *Jp = nullptr;
  float *gridMass = nullptr;
  float2 *gridVel = nullptr;
  int *grid2 = nullptr;
  size_t grid2Size = 0;
  int W = 0, H = 0;
};

__host__ __device__ __forceinline__ Mat2 m2(float a, float b, float c,
                                            float d) {
  Mat2 M{a, b, c, d};
  return M;
}
__host__ __device__ __forceinline__ Mat2 madd(Mat2 A, Mat2 B) {
  return m2(A.a + B.a, A.b + B.b, A.c + B.c, A.d + B.d);
}
__host__ __device__ __forceinline__ Mat2 msub(Mat2 A, Mat2 B) {
  return m2(A.a - B.a, A.b - B.b, A.c - B.c, A.d - B.d);
}
__host__ __device__ __forceinline__ Mat2 mmul(Mat2 A, Mat2 B) {
  return m2(A.a * B.a + A.b * B.c, A.a * B.b + A.b * B.d,
            A.c * B.a + A.d * B.c, A.c * B.b + A.d * B.d);
}
__host__ __device__ __forceinline__ Mat2 mscale(Mat2 A, float s) {
  return m2(A.a * s, A.b * s, A.c * s, A.d * s);
}
__host__ __device__ __forceinline__ Mat2 mtranspose(Mat2 A) {
  return m2(A.a, A.c, A.b, A.d);
}
__host__ __device__ __forceinline__ float mdet(Mat2 A) { return A.a * A.d - A.b * A.c; }
__host__ __device__ __forceinline__ float2 mv(Mat2 A, float2 x) {
  return make_float2(A.a * x.x + A.b * x.y, A.c * x.x + A.d * x.y);
}

__global__ void k_clear_grid(float *mass, float2 *vel, int M) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < M) {
    mass[i] = 0.0f;
    vel[i] = make_float2(0.0f, 0.0f);
  }
}

__device__ __forceinline__ float clampf(float x, float lo, float hi) {
  return fminf(fmaxf(x, lo), hi);
}

__global__ void k_p2g(const float2 *__restrict__ pos, const float2 *__restrict__ vel,
                      Mat2 *__restrict__ F, float *__restrict__ Jp,
                      float *__restrict__ gridMass, float2 *__restrict__ gridVel,
                      int N, int Gx, int Gy, float dx, float invDx, float dt,
                      float mass, float volume, float mu0, float lambda0,
                      float hardening, float thetaC, float thetaS, int material) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= N)
    return;

  float2 Xp = make_float2(pos[p].x * invDx, pos[p].y * invDx);
  int baseX = (int)floorf(Xp.x - 0.5f);
  int baseY = (int)floorf(Xp.y - 0.5f);
  float fx = Xp.x - baseX;
  float fy = Xp.y - baseY;
  float wx[3] = {0.5f * (1.5f - fx) * (1.5f - fx),
                 0.75f - (fx - 1.0f) * (fx - 1.0f),
                 0.5f * (fx - 0.5f) * (fx - 0.5f)};
  float wy[3] = {0.5f * (1.5f - fy) * (1.5f - fy),
                 0.75f - (fy - 1.0f) * (fy - 1.0f),
                 0.5f * (fy - 0.5f) * (fy - 0.5f)};

  Mat2 Fe = F[p];
  if (material == 1) {
    Fe.a = clampf(Fe.a, 1.0f - thetaC, 1.0f + thetaS);
    Fe.d = clampf(Fe.d, 1.0f - thetaC, 1.0f + thetaS);
    Fe.b *= 0.98f;
    Fe.c *= 0.98f;
  }
  float J = fmaxf(mdet(Fe), 0.2f);
  float e = expf(hardening * (1.0f - Jp[p]));
  float mu = mu0 * e;
  float lambda = lambda0 * e;
  if (material == 0) // mud: weak shear with strong volume preservation
    mu *= 0.25f;
  if (material == 2) { // sand: pressure-hardening frictional grains
    mu *= 1.8f;
    lambda *= 0.75f;
  }
  Mat2 I = m2(1, 0, 0, 1);
  Mat2 PFt = madd(mscale(msub(mmul(Fe, mtranspose(Fe)), I), mu),
                  mscale(I, lambda * logf(J) * J));
  Mat2 stress = mscale(PFt, -4.0f * invDx * invDx * dt * volume);

  for (int gx = 0; gx < 3; ++gx) {
    for (int gy = 0; gy < 3; ++gy) {
      int ix = baseX + gx, iy = baseY + gy;
      if ((unsigned)ix >= (unsigned)Gx || (unsigned)iy >= (unsigned)Gy)
        continue;
      float w = wx[gx] * wy[gy];
      float2 dpos = make_float2((gx - fx) * dx, (gy - fy) * dx);
      float2 momentum = make_float2(mass * vel[p].x, mass * vel[p].y);
      float2 force = mv(stress, dpos);
      int id = iy * Gx + ix;
      atomicAdd(&gridMass[id], w * mass);
      atomicAdd(&gridVel[id].x, w * (momentum.x + force.x));
      atomicAdd(&gridVel[id].y, w * (momentum.y + force.y));
    }
  }
  F[p] = Fe;
}

__global__ void k_grid_update(float *mass, float2 *vel, int Gx, int Gy, float dt,
                              float gravity) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  int M = Gx * Gy;
  if (id >= M || mass[id] <= 0.0f)
    return;
  vel[id].x /= mass[id];
  vel[id].y = vel[id].y / mass[id] - gravity * dt;
  int x = id % Gx, y = id / Gx;
  if ((x < 3 && vel[id].x < 0.0f) || (x > Gx - 4 && vel[id].x > 0.0f))
    vel[id].x = 0.0f;
  if ((y < 3 && vel[id].y < 0.0f) || (y > Gy - 4 && vel[id].y > 0.0f))
    vel[id].y = 0.0f;
}

__global__ void k_g2p(float2 *__restrict__ pos, float2 *__restrict__ vel,
                      Mat2 *__restrict__ F, float *__restrict__ Jp,
                      const float2 *__restrict__ gridVel, int N, int Gx, int Gy,
                      float dx, float invDx, float dt, int material) {
  int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= N)
    return;
  float2 Xp = make_float2(pos[p].x * invDx, pos[p].y * invDx);
  int baseX = (int)floorf(Xp.x - 0.5f);
  int baseY = (int)floorf(Xp.y - 0.5f);
  float fx = Xp.x - baseX;
  float fy = Xp.y - baseY;
  float wx[3] = {0.5f * (1.5f - fx) * (1.5f - fx),
                 0.75f - (fx - 1.0f) * (fx - 1.0f),
                 0.5f * (fx - 0.5f) * (fx - 0.5f)};
  float wy[3] = {0.5f * (1.5f - fy) * (1.5f - fy),
                 0.75f - (fy - 1.0f) * (fy - 1.0f),
                 0.5f * (fy - 0.5f) * (fy - 0.5f)};
  float2 newV = make_float2(0, 0);
  Mat2 C = m2(0, 0, 0, 0);
  for (int gx = 0; gx < 3; ++gx) {
    for (int gy = 0; gy < 3; ++gy) {
      int ix = baseX + gx, iy = baseY + gy;
      if ((unsigned)ix >= (unsigned)Gx || (unsigned)iy >= (unsigned)Gy)
        continue;
      float w = wx[gx] * wy[gy];
      float2 gv = gridVel[iy * Gx + ix];
      float2 dpos = make_float2((gx - fx) * dx, (gy - fy) * dx);
      newV.x += w * gv.x;
      newV.y += w * gv.y;
      C.a += 4.0f * invDx * w * gv.x * dpos.x;
      C.b += 4.0f * invDx * w * gv.x * dpos.y;
      C.c += 4.0f * invDx * w * gv.y * dpos.x;
      C.d += 4.0f * invDx * w * gv.y * dpos.y;
    }
  }
  Mat2 oldF = F[p];
  Mat2 newF = mmul(madd(m2(1, 0, 0, 1), mscale(C, dt)), oldF);
  float oldJ = fmaxf(mdet(oldF), 1.0e-6f);
  float newJ = fmaxf(mdet(newF), 1.0e-6f);
  if (material == 0) { // mud relaxes shear while retaining compressibility
    newF.b *= 0.96f;
    newF.c *= 0.96f;
  }
  Jp[p] = clampf(Jp[p] * oldJ / newJ, 0.05f, 20.0f);
  F[p] = newF;
  float2 x = make_float2(pos[p].x + dt * newV.x, pos[p].y + dt * newV.y);
  x.x = clampf(x.x, 2.0f * dx, (Gx - 3.0f) * dx);
  x.y = clampf(x.y, 2.0f * dx, (Gy - 3.0f) * dx);
  pos[p] = x;
  vel[p] = newV;
}

__global__ void k_clear_pixels(int *grid, int size) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < size)
    grid[i] = 0;
}

__global__ void k_rasterize(const float2 *pos, int N, int *grid2, int W, int H,
                            float boxX, float boxY) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= N)
    return;
  int cx = (int)(pos[i].x / boxX * (W - 1));
  int sy = (int)((boxY - pos[i].y) / boxY * (2 * H - 1));
  if ((unsigned)cx < (unsigned)W && (unsigned)sy < (unsigned)(2 * H))
    atomicAdd(&grid2[sy * W + cx], 1);
}

static void parse_args(int argc, char **argv, Params &P) {
  const struct option longopts[] = {{"n", required_argument, 0, 'n'},
                                    {"grid", required_argument, 0, 'g'},
                                    {"dt", required_argument, 0, 't'},
                                    {"steps", required_argument, 0, 's'},
                                    {"material", required_argument, 0, 'm'},
                                    {"gravity", required_argument, 0, 'y'},
                                    {"stride", required_argument, 0, 'S'},
                                    {"fps", required_argument, 0, 'p'},
                                    {"headless", no_argument, 0, 'H'},
                                    {"seed", required_argument, 0, 'e'},
                                    {0, 0, 0, 0}};
  int c;
  while ((c = getopt_long(argc, argv, "n:g:t:s:m:y:S:p:He:", longopts, nullptr)) != -1) {
    switch (c) {
    case 'n': P.N = atoi(optarg); break;
    case 'g': sscanf(optarg, "%dx%d", &P.Gx, &P.Gy); break;
    case 't': P.dt = atof(optarg); break;
    case 's': P.steps = atoi(optarg); break;
    case 'm':
      if (strcmp(optarg, "mud") == 0) P.material = Params::MUD;
      else if (strcmp(optarg, "sand") == 0) P.material = Params::SAND;
      else P.material = Params::SNOW;
      break;
    case 'y': P.gravity = atof(optarg); break;
    case 'S': P.stride = fmaxf(1, atoi(optarg)); break;
    case 'p': P.fps = atoi(optarg); break;
    case 'H': P.headless = true; break;
    case 'e': P.seed = atoi(optarg); break;
    default: break;
    }
  }
}

static void reset_particles(const Params &P, std::vector<float2> &pos,
                            std::vector<float2> &vel, std::vector<Mat2> &F,
                            std::vector<float> &Jp) {
  std::mt19937 rng(P.seed);
  std::uniform_real_distribution<float> U(-0.5f, 0.5f);
  int nx = (int)sqrtf((float)P.N);
  int ny = (P.N + nx - 1) / nx;
  for (int i = 0; i < P.N; ++i) {
    int ix = i % nx, iy = i / nx;
    float x = 0.22f + 0.42f * (ix + 0.5f) / nx;
    float y = 0.28f + 0.45f * (iy + 0.5f) / ny;
    pos[i] = make_float2(x + U(rng) * 0.12f / nx, y + U(rng) * 0.12f / ny);
    vel[i] = make_float2(1.0f * (0.5f - y), 0.0f);
    F[i] = m2(1, 0, 0, 1);
    Jp[i] = 1.0f;
  }
}

static void step_mpm(const Params &P, DevState &d) {
  int block = 256;
  int gridNodes = P.Gx * P.Gy;
  int particleBlocks = (P.N + block - 1) / block;
  int gridBlocks = (gridNodes + block - 1) / block;
  float dx = P.boxX / (P.Gx - 1);
  float invDx = 1.0f / dx;
  k_clear_grid<<<gridBlocks, block>>>(d.gridMass, d.gridVel, gridNodes);
  k_p2g<<<particleBlocks, block>>>(d.pos, d.vel, d.F, d.Jp, d.gridMass, d.gridVel,
                                   P.N, P.Gx, P.Gy, dx, invDx, P.dt, P.particleMass,
                                   P.volume, P.mu0, P.lambda0, P.hardening,
                                   P.criticalCompression, P.criticalStretch,
                                   (int)P.material);
  k_grid_update<<<gridBlocks, block>>>(d.gridMass, d.gridVel, P.Gx, P.Gy, P.dt, P.gravity);
  k_g2p<<<particleBlocks, block>>>(d.pos, d.vel, d.F, d.Jp, d.gridVel, P.N, P.Gx,
                                   P.Gy, dx, invDx, P.dt, (int)P.material);
  CUDA_CHECK(cudaGetLastError());
}

static void render(const Params &P, DevState &d, int step) {
  int H, W;
  getmaxyx(stdscr, H, W);
  if (W <= 0 || H <= 0)
    return;
  if (W != d.W || H != d.H) {
    if (d.grid2)
      CUDA_CHECK(cudaFree(d.grid2));
    d.W = W; d.H = H; d.grid2Size = (size_t)W * 2 * H;
    CUDA_CHECK(cudaMalloc(&d.grid2, d.grid2Size * sizeof(int)));
  }
  int block = 256;
  k_clear_pixels<<<(d.grid2Size + block - 1) / block, block>>>(d.grid2, d.grid2Size);
  k_rasterize<<<(P.N + block - 1) / block, block>>>(d.pos, P.N, d.grid2, W, H, P.boxX, P.boxY);
  std::vector<int> hgrid(d.grid2Size);
  CUDA_CHECK(cudaMemcpy(hgrid.data(), d.grid2, d.grid2Size * sizeof(int), cudaMemcpyDeviceToHost));
  erase();
  mvprintw(0, 0, "CUDA MPM %s  N=%d grid=%dx%d dt=%.1e step=%d  q quits",
           P.material == Params::MUD ? "mud" : (P.material == Params::SAND ? "sand" : "snow"),
           P.N, P.Gx, P.Gy, P.dt, step);
  for (int y = 1; y < H; ++y) {
    for (int x = 0; x < W; ++x) {
      int top = hgrid[(2 * y - 2) * W + x] > 0;
      int bot = hgrid[(2 * y - 1) * W + x] > 0;
      mvaddwstr(y, x, HALF_BLOCKS[(top ? GL_TOP : 0) | (bot ? GL_BOTTOM : 0)]);
    }
  }
  refresh();
}

int main(int argc, char **argv) {
  Params P;
  parse_args(argc, argv, P);
  P.volume = (P.boxX * P.boxY) / (P.Gx * P.Gy) * 0.45f;
  std::vector<float2> hpos(P.N), hvel(P.N);
  std::vector<Mat2> hF(P.N);
  std::vector<float> hJp(P.N);
  reset_particles(P, hpos, hvel, hF, hJp);

  DevState d;
  CUDA_CHECK(cudaMalloc(&d.pos, P.N * sizeof(float2)));
  CUDA_CHECK(cudaMalloc(&d.vel, P.N * sizeof(float2)));
  CUDA_CHECK(cudaMalloc(&d.F, P.N * sizeof(Mat2)));
  CUDA_CHECK(cudaMalloc(&d.Jp, P.N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d.gridMass, P.Gx * P.Gy * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d.gridVel, P.Gx * P.Gy * sizeof(float2)));
  CUDA_CHECK(cudaMemcpy(d.pos, hpos.data(), P.N * sizeof(float2), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.vel, hvel.data(), P.N * sizeof(float2), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.F, hF.data(), P.N * sizeof(Mat2), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d.Jp, hJp.data(), P.N * sizeof(float), cudaMemcpyHostToDevice));

  if (!P.headless) {
    setlocale(LC_ALL, "");
    initscr(); cbreak(); noecho(); nodelay(stdscr, TRUE); curs_set(0);
    curses_active = true;
  }

  int step = 0;
  auto last = std::chrono::steady_clock::now();
  while (P.steps <= 0 || step < P.steps) {
    step_mpm(P, d);
    ++step;
    if (!P.headless && step % P.stride == 0) {
      render(P, d, step);
      int ch = getch();
      if (ch == 'q' || ch == 'Q')
        break;
      if (P.fps > 0) {
        auto now = std::chrono::steady_clock::now();
        auto target = std::chrono::milliseconds(1000 / P.fps);
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last);
        if (elapsed < target)
          usleep((target - elapsed).count() * 1000);
        last = std::chrono::steady_clock::now();
      }
    }
  }

  if (curses_active) { endwin(); curses_active = false; }
  CUDA_CHECK(cudaFree(d.pos)); CUDA_CHECK(cudaFree(d.vel)); CUDA_CHECK(cudaFree(d.F));
  CUDA_CHECK(cudaFree(d.Jp)); CUDA_CHECK(cudaFree(d.gridMass)); CUDA_CHECK(cudaFree(d.gridVel));
  if (d.grid2) CUDA_CHECK(cudaFree(d.grid2));
  return 0;
}
