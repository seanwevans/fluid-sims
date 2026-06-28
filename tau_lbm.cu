// tau_lbm.cu — CUDA D2Q9 Lattice Boltzmann fluid simulator
//
// Build:
//   nvcc -std=c++17 -O3 -use_fast_math -arch=sm_86 -lineinfo -o tau_lbm \
//        tau_lbm.cu -lncursesw
//
// This is a mesoscopic BGK LBM solver: each cell stores nine particle
// distribution functions, collides them toward equilibrium, then streams them
// to neighboring lattice nodes. Solid cells use on-link bounce-back, so the
// demo can place obstacles without a pressure Poisson solve.
//
// Controls: 'q' quits, 'o' toggles the cylinder obstacle, '+'/'-' adjusts drive.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <locale.h>
#include <ncursesw/curses.h>
#include <unistd.h>

#include <cuda_runtime.h>

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
  int nx = 512, ny = 256;
  int steps = 0;
  int stride = 4;
  int fps_limit = 0;
  bool headless = false;
  bool obstacle = true;
  float tau = 0.56f;   // viscosity = cs^2 * (tau - 1/2)
  float drive = 1.0e-6f;
  float rho0 = 1.0f;
  float obstacle_radius = 32.0f;
};

__constant__ int c_ex[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
__constant__ int c_ey[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
__constant__ int c_opp[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
__constant__ float c_w[9] = {4.0f / 9.0f,  1.0f / 9.0f,  1.0f / 9.0f,
                             1.0f / 9.0f,  1.0f / 9.0f,  1.0f / 36.0f,
                             1.0f / 36.0f, 1.0f / 36.0f, 1.0f / 36.0f};

__host__ __device__ inline int idx(int i, int j, int nx) { return j * nx + i; }
__host__ __device__ inline int fidx(int q, int i, int j, int nx, int ny) {
  return q * nx * ny + idx(i, j, nx);
}

__device__ inline float feq(int q, float rho, float ux, float uy) {
  const float cu = 3.0f * (c_ex[q] * ux + c_ey[q] * uy);
  const float u2 = ux * ux + uy * uy;
  return c_w[q] * rho * (1.0f + cu + 0.5f * cu * cu - 1.5f * u2);
}

__global__ void init_kernel(float *f, unsigned char *solid, Params P) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= P.nx || j >= P.ny)
    return;

  const int p = idx(i, j, P.nx);
  const float cx = 0.28f * P.nx;
  const float cy = 0.5f * P.ny;
  const float dx = i - cx;
  const float dy = j - cy;
  const bool wall = (j == 0 || j == P.ny - 1);
  const bool cyl = P.obstacle && (dx * dx + dy * dy < P.obstacle_radius * P.obstacle_radius);
  solid[p] = (wall || cyl) ? 1 : 0;

  const float shear = 0.015f * sinf(2.0f * 3.14159265f * j / (P.ny > 1 ? P.ny - 1 : 1));
  for (int q = 0; q < 9; ++q)
    f[fidx(q, i, j, P.nx, P.ny)] = feq(q, P.rho0, shear, 0.0f);
}

__global__ void collide_stream_kernel(const float *fin, float *fout,
                                      const unsigned char *solid, Params P) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= P.nx || j >= P.ny)
    return;

  const int p = idx(i, j, P.nx);
  float local[9];
  for (int q = 0; q < 9; ++q)
    local[q] = fin[fidx(q, i, j, P.nx, P.ny)];

  if (solid[p]) {
    for (int q = 0; q < 9; ++q)
      fout[fidx(c_opp[q], i, j, P.nx, P.ny)] = local[q];
    return;
  }

  float rho = 0.0f, ux = 0.0f, uy = 0.0f;
  for (int q = 0; q < 9; ++q) {
    rho += local[q];
    ux += local[q] * c_ex[q];
    uy += local[q] * c_ey[q];
  }
  rho = fmaxf(rho, 1.0e-6f);
  ux = ux / rho + P.drive;
  uy /= rho;

  const float omega = 1.0f / P.tau;
  for (int q = 0; q < 9; ++q) {
    const float post = local[q] - omega * (local[q] - feq(q, rho, ux, uy));
    const int ni = (i + c_ex[q] + P.nx) % P.nx;
    const int nj = j + c_ey[q];
    if (nj < 0 || nj >= P.ny || solid[idx(ni, nj, P.nx)])
      fout[fidx(c_opp[q], i, j, P.nx, P.ny)] = post;
    else
      fout[fidx(q, ni, nj, P.nx, P.ny)] = post;
  }
}

__global__ void render_kernel(const float *f, const unsigned char *solid,
                              float *speed, Params P) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= P.nx || j >= P.ny)
    return;
  const int p = idx(i, j, P.nx);
  if (solid[p]) {
    speed[p] = -1.0f;
    return;
  }
  float rho = 0.0f, ux = 0.0f, uy = 0.0f;
  for (int q = 0; q < 9; ++q) {
    const float fq = f[fidx(q, i, j, P.nx, P.ny)];
    rho += fq;
    ux += fq * c_ex[q];
    uy += fq * c_ey[q];
  }
  speed[p] = hypotf(ux / rho, uy / rho);
}

void usage(const char *prog) {
  printf("Usage: %s [options]\n", prog);
  puts("  --nx N              grid x (512)");
  puts("  --ny N              grid y (256)");
  puts("  --tau T             BGK relaxation time > 0.5 (0.56)");
  puts("  --drive A           body-force-like x acceleration (1e-6)");
  puts("  --radius R          cylinder radius in cells (32)");
  puts("  --no-obstacle       disable cylinder; keep channel walls");
  puts("  --steps K           steps (0 forever)");
  puts("  --stride N          render every N steps (4)");
  puts("  --fps N             FPS cap (0 uncapped)");
  puts("  --headless          benchmark mode");
  puts("  -h, --help");
}

void parse_args(int argc, char **argv, Params &P) {
  static const option opts[] = {{"nx", required_argument, 0, 0},
                                {"ny", required_argument, 0, 0},
                                {"tau", required_argument, 0, 0},
                                {"drive", required_argument, 0, 0},
                                {"radius", required_argument, 0, 0},
                                {"no-obstacle", no_argument, 0, 0},
                                {"steps", required_argument, 0, 0},
                                {"stride", required_argument, 0, 0},
                                {"fps", required_argument, 0, 0},
                                {"headless", no_argument, 0, 0},
                                {"help", no_argument, 0, 'h'},
                                {0, 0, 0, 0}};
  for (;;) {
    int idx = 0;
    const int c = getopt_long(argc, argv, "h", opts, &idx);
    if (c == -1)
      break;
    if (c == 'h') {
      usage(argv[0]);
      exit(0);
    }
    const char *name = opts[idx].name;
    if (!strcmp(name, "nx")) P.nx = atoi(optarg);
    else if (!strcmp(name, "ny")) P.ny = atoi(optarg);
    else if (!strcmp(name, "tau")) P.tau = atof(optarg);
    else if (!strcmp(name, "drive")) P.drive = atof(optarg);
    else if (!strcmp(name, "radius")) P.obstacle_radius = atof(optarg);
    else if (!strcmp(name, "no-obstacle")) P.obstacle = false;
    else if (!strcmp(name, "steps")) P.steps = atoi(optarg);
    else if (!strcmp(name, "stride")) P.stride = atoi(optarg);
    else if (!strcmp(name, "fps")) P.fps_limit = atoi(optarg);
    else if (!strcmp(name, "headless")) P.headless = true;
  }
  P.nx = std::max(P.nx, 16);
  P.ny = std::max(P.ny, 16);
  P.tau = std::max(P.tau, 0.501f);
  P.stride = std::max(P.stride, 1);
}

void draw(const float *speed, const Params &P, int step, double mlups) {
  static const wchar_t ramp[] = {L' ', L'·', L'░', L'▒', L'▓', L'█'};
  int rows, cols;
  getmaxyx(stdscr, rows, cols);
  mvprintw(0, 0, "D2Q9 BGK LBM step %d  %.1f MLUPS  tau %.3f drive %.2g  q quit +/- drive o obstacle",
           step, mlups, P.tau, P.drive);
  const int disp_rows = rows - 1;
  const int disp_cols = cols;
  const float scale = 35.0f;
  for (int y = 0; y < disp_rows; ++y) {
    move(y + 1, 0);
    const int j = std::min(P.ny - 1, y * P.ny / std::max(disp_rows, 1));
    for (int x = 0; x < disp_cols; ++x) {
      const int i = std::min(P.nx - 1, x * P.nx / std::max(disp_cols, 1));
      const float v = speed[idx(i, j, P.nx)];
      if (v < 0.0f) {
        addch('#');
      } else {
        int k = std::min(5, std::max(0, (int)(v * scale)));
        addnwstr(&ramp[k], 1);
      }
    }
    clrtoeol();
  }
  refresh();
}

int main(int argc, char **argv) {
  Params P;
  parse_args(argc, argv, P);

  const size_t cells = (size_t)P.nx * P.ny;
  float *d_f0, *d_f1, *d_speed;
  unsigned char *d_solid;
  CUDA_CHECK(cudaMalloc(&d_f0, 9 * cells * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_f1, 9 * cells * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_speed, cells * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_solid, cells * sizeof(unsigned char)));

  dim3 block(16, 16);
  dim3 grid((P.nx + block.x - 1) / block.x, (P.ny + block.y - 1) / block.y);
  init_kernel<<<grid, block>>>(d_f0, d_solid, P);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaMemcpy(d_f1, d_f0, 9 * cells * sizeof(float), cudaMemcpyDeviceToDevice));

  float *h_speed = nullptr;
  if (!P.headless) {
    h_speed = (float *)malloc(cells * sizeof(float));
    setlocale(LC_ALL, "");
    initscr(); cbreak(); noecho(); nodelay(stdscr, TRUE); curs_set(0);
    curses_active = true;
  }

  auto start = std::chrono::steady_clock::now();
  auto last = start;
  double mlups = 0.0;
  int step = 0;
  for (; P.steps == 0 || step < P.steps; ++step) {
    collide_stream_kernel<<<grid, block>>>(d_f0, d_f1, d_solid, P);
    CUDA_CHECK(cudaGetLastError());
    std::swap(d_f0, d_f1);

    if (!P.headless && step % P.stride == 0) {
      render_kernel<<<grid, block>>>(d_f0, d_solid, d_speed, P);
      CUDA_CHECK(cudaMemcpy(h_speed, d_speed, cells * sizeof(float), cudaMemcpyDeviceToHost));
      auto now = std::chrono::steady_clock::now();
      const double sec = std::chrono::duration<double>(now - last).count();
      if (sec > 0.0)
        mlups = (double)cells * P.stride / (sec * 1.0e6);
      last = now;
      draw(h_speed, P, step, mlups);
      const int ch = getch();
      if (ch == 'q') break;
      if (ch == '+') P.drive *= 1.2f;
      if (ch == '-') P.drive /= 1.2f;
      if (ch == 'o') { P.obstacle = !P.obstacle; init_kernel<<<grid, block>>>(d_f0, d_solid, P); }
      if (P.fps_limit > 0) usleep(1000000 / P.fps_limit);
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  const double total = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  if (P.headless)
    printf("LBM D2Q9: %d steps, %zu cells, %.2f MLUPS\n", step, cells,
           total > 0.0 ? (double)cells * step / (total * 1.0e6) : 0.0);

  if (curses_active) { endwin(); curses_active = false; }
  free(h_speed);
  cudaFree(d_f0); cudaFree(d_f1); cudaFree(d_speed); cudaFree(d_solid);
  return 0;
}
