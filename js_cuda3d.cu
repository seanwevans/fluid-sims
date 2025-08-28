// js_cuda3d.cu : CUDA 3D fluid sim with isometric ncurses rendering
// nvcc -O3 -arch=sm_86 js_cuda3d.cu -o jsc3d -lncursesw

#include <chrono>
#include <getopt.h>
#include <locale.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <cuda_runtime.h>
#include <ncursesw/curses.h>

#define N 128

#define DT 1
#define VISC 1e-5
#define DIFF 1e-6
#define DENS_DECAY (1 - 1e-2)

#define IX(i, j, k) ((i) + (N + 2) * ((j) + (N + 2) * (k)))

#define CUDA_CHECK(ans)                                                        \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    exit(code);
  }
}

static bool curses_active = false;
static const double N2S = N * N / 16;
static const double NO4 = N / 4;

double *d_u, *d_v, *d_w;    // vel
double *d_u0, *d_v0, *d_w0; // temp vel
double *d_d, *d_d0, *d_tmp; // density + temp
double *d_p, *d_div;        // pressure, divergence
double *h_d;                // host density

dim3 bs(8, 8, 8);
dim3 gs((N + bs.x - 1) / bs.x, (N + bs.y - 1) / bs.y, (N + bs.z - 1) / bs.z);

// kernels
__global__ void k_decay(double *d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
  if (i <= N && j <= N && k <= N)
    d[IX(i, j, k)] *= DENS_DECAY;
}

__global__ void k_add_source3d(double *u, double *v, double *w, double *d,
                               int step) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

  double dx = i - NO4 * (1 + cos(step));
  double dy = j - NO4 * (1 + sin(step));
  double dz = k - NO4 * (1 + sin(step));
  double r2 = dx * dx + dy * dy + dz * dz;

  if (r2 < N2S) {
    double r = sqrt(r2) + 1e-7;
    d[IX(i, j, k)] += exp(-r2 / N2S);
    u[IX(i, j, k)] += dz / r;
    v[IX(i, j, k)] += dy / r;
    w[IX(i, j, k)] += dx / r;
  }
}

__global__ void k_lin(double *x_new, double *x_old, double *x0, double a,
                      double c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

  x_new[IX(i, j, k)] = (x0[IX(i, j, k)] +
                        a * (x_old[IX(i - 1, j, k)] + x_old[IX(i + 1, j, k)] +
                             x_old[IX(i, j - 1, k)] + x_old[IX(i, j + 1, k)] +
                             x_old[IX(i, j, k - 1)] + x_old[IX(i, j, k + 1)])) /
                       c;
}

__global__ void k_div(double *u, double *v, double *w, double *div, double *p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

  div[IX(i, j, k)] = -0.5 * ((u[IX(i + 1, j, k)] - u[IX(i - 1, j, k)]) +
                             (v[IX(i, j + 1, k)] - v[IX(i, j - 1, k)]) +
                             (w[IX(i, j, k + 1)] - w[IX(i, j, k - 1)]));
  p[IX(i, j, k)] = 0;
}

__global__ void k_proj(double *u, double *v, double *w, double *p) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;

  u[IX(i, j, k)] -= 0.5 * (p[IX(i + 1, j, k)] - p[IX(i - 1, j, k)]);
  v[IX(i, j, k)] -= 0.5 * (p[IX(i, j + 1, k)] - p[IX(i, j - 1, k)]);
  w[IX(i, j, k)] -= 0.5 * (p[IX(i, j, k + 1)] - p[IX(i, j, k - 1)]);
}

// 3D semi-Lagrangian advection with trilinear interpolation
__global__ void k_adv3d(double *q, const double *q0, const double *u,
                        const double *v, const double *w) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int k = blockIdx.z * blockDim.z + threadIdx.z + 1;
  if (i > N || j > N || k > N)
    return;

  // backtrace in index-space
  double x = i - DT * u[IX(i, j, k)];
  double y = j - DT * v[IX(i, j, k)];
  double z = k - DT * w[IX(i, j, k)];

  // clamp to cell-centered domain
  if (x < 0.5)
    x = 0.5;
  if (x > N + 0.5)
    x = N + 0.5;
  if (y < 0.5)
    y = 0.5;
  if (y > N + 0.5)
    y = N + 0.5;
  if (z < 0.5)
    z = 0.5;
  if (z > N + 0.5)
    z = N + 0.5;

  int i0 = (int)floor(x), i1 = i0 + 1;
  int j0 = (int)floor(y), j1 = j0 + 1;
  int k0 = (int)floor(z), k1 = k0 + 1;

  double sx = x - i0, tx = 1.0 - sx;
  double sy = y - j0, ty = 1.0 - sy;
  double sz = z - k0, tz = 1.0 - sz;

  // trilinear interpolation
  double c000 = q0[IX(i0, j0, k0)], c100 = q0[IX(i1, j0, k0)];
  double c010 = q0[IX(i0, j1, k0)], c110 = q0[IX(i1, j1, k0)];
  double c001 = q0[IX(i0, j0, k1)], c101 = q0[IX(i1, j0, k1)];
  double c011 = q0[IX(i0, j1, k1)], c111 = q0[IX(i1, j1, k1)];

  double c00 = tx * c000 + sx * c100;
  double c10 = tx * c010 + sx * c110;
  double c01 = tx * c001 + sx * c101;
  double c11 = tx * c011 + sx * c111;

  double c0 = ty * c00 + sy * c10;
  double c1 = ty * c01 + sy * c11;

  q[IX(i, j, k)] = tz * c0 + sz * c1;
}

// helpers
void lin_solve(double *x, double *x0, double a, double c, int iters) {
  double *read = x, *write = d_tmp;
  for (int it = 0; it < iters; ++it) {
    k_lin<<<gs, bs>>>(write, read, x0, a, c);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    double *tmp = read;
    read = write;
    write = tmp;
  }
  if (read != x)
    CUDA_CHECK(cudaMemcpy(x, read,
                          (size_t)(N + 2) * (N + 2) * (N + 2) * sizeof(double),
                          cudaMemcpyDeviceToDevice));
}

void diffuse(double *x, double *x0, double diffc) {
  double a = DT * diffc * N * N;
  lin_solve(x, x0, a, 1 + 6 * a, 20);
}

void project(double *u, double *v, double *w, double *p, double *div) {
  k_div<<<gs, bs>>>(u, v, w, div, p);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  lin_solve(p, div, 1.0, 6.0, 20);
  k_proj<<<gs, bs>>>(u, v, w, p);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

void advect(double *q, double *q0, double *u, double *v, double *w) {
  k_adv3d<<<gs, bs>>>(q, q0, u, v, w);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

void vel_step() {
  // diffuse velocities
  diffuse(d_u0, d_u, VISC);
  diffuse(d_v0, d_v, VISC);
  diffuse(d_w0, d_w, VISC);

  // make them divergence-free
  project(d_u0, d_v0, d_w0, d_p, d_div);

  // advect velocities by themselves
  advect(d_u, d_u0, d_u0, d_v0, d_w0);
  advect(d_v, d_v0, d_u0, d_v0, d_w0);
  advect(d_w, d_w0, d_u0, d_v0, d_w0);

  // project again
  project(d_u, d_v, d_w, d_p, d_div);
}

void dens_step() {
  diffuse(d_d0, d_d, DIFF);
  advect(d_d, d_d0, d_u, d_v, d_w);
}

// gpu
void gpu_alloc() {
  size_t s = (size_t)(N + 2) * (N + 2) * (N + 2) * sizeof(double);
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
  CUDA_CHECK(cudaMallocHost(&h_d, s));
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
  cudaFreeHost(h_d);
}

// render
void show_iso(int step, double fps) {
  static const wchar_t wramp[] = {L' ', L'░', L'▒', L'▓', L'█'};
  const int L = 4;

  int rows, cols;
  getmaxyx(stdscr, rows, cols);
  if (rows < 5 || cols < 10)
    return;

  mvprintw(0, 0, "step %d  FPS %.1f  (q=quit)  N=%d", step, fps, N);

  size_t s = (size_t)(N + 2) * (N + 2) * (N + 2) * sizeof(double);
  CUDA_CHECK(cudaMemcpy(h_d, d_d, s, cudaMemcpyDeviceToHost));

  int W = cols, H = rows - 1; // keep row 0 for HUD
  static int bufW = 0, bufH = 0;
  static int *zbuf = nullptr, *cbuf = nullptr;
  if (W != bufW || H != bufH || !zbuf) {
    free(zbuf);
    free(cbuf);
    zbuf = (int *)malloc(W * H * sizeof(int));
    cbuf = (int *)malloc(W * H * sizeof(int));
    bufW = W;
    bufH = H;
  }
  for (int i = 0; i < W * H; i++) {
    zbuf[i] = -1;
    cbuf[i] = 0;
  }

  // scale iso footprint to window
  double sx = (double)W / (2.0 * N + 2.0);
  double sy = (double)H / (2.0 * N + 2.0);
  double sproj = fmin(sx, sy);
  if (sproj <= 0)
    sproj = 1.0;

  double cx = W * 0.5;
  double cy = H * 0.5;

  // front-to-back (increasing k) with z-buffer on k
  for (int k = 1; k <= N; k++) {
    for (int j = 1; j <= N; j++) {
      for (int i = 1; i <= N; i++) {
        double val = h_d[IX(i, j, k)];
        if (val <= 0.0)
          continue;
        if (val > 1.0)
          val = 1.0;

        // classic iso: x = i - j; y = (i + j)/2 - k
        double X = (i - j) * sproj;
        double Y = ((i + j) * 0.5 - k) * sproj;

        int x = (int)llround(X + cx);
        int y = (int)llround(Y + cy);

        if (x >= 0 && x < W && y >= 0 && y < H) {
          int zidx = y * W + x;
          if (k > zbuf[zidx]) { // nearest to viewer
            zbuf[zidx] = k;
            int idx = (int)llround(val * L);
            if (idx < 0)
              idx = 0;
            if (idx > L)
              idx = L;
            cbuf[zidx] = idx;
          }
        }
      }
    }
  }

  for (int y = 0; y < H; y++) {
    move(y + 1, 0);
    for (int x = 0; x < W; x++) {
      int idx = cbuf[y * W + x];
      addnwstr(&wramp[idx], 1);
    }
    clrtoeol();
  }
  refresh();
}

// trap
void handle_exit(int sig) {
  if (curses_active) {
    endwin();
    curses_active = false;
  }
  gpu_free();
  fprintf(stderr, "\n[trap] caught sig %d\n", sig);
  exit(1);
}

int main(int argc, char **argv) {
  signal(SIGINT, handle_exit);
  signal(SIGTERM, handle_exit);

  gpu_alloc();
  curses_active = true;
  setlocale(LC_ALL, "");
  initscr();
  cbreak();
  noecho();
  curs_set(0);
  nodelay(stdscr, TRUE);
  keypad(stdscr, TRUE);

  int step = 0;
  auto last = std::chrono::high_resolution_clock::now();

  while (true) {
    int ch = getch();
    if (ch == 'q' || ch == 'Q')
      break;

    k_decay<<<gs, bs>>>(d_d);
    CUDA_CHECK(cudaGetLastError());

    k_add_source3d<<<gs, bs>>>(d_u, d_v, d_w, d_d, step);
    CUDA_CHECK(cudaGetLastError());

    vel_step();
    dens_step();

    auto now = std::chrono::high_resolution_clock::now();
    double dt = std::chrono::duration<double>(now - last).count();
    double fps = (dt > 0) ? 1.0 / dt : 0.0;
    show_iso(step, fps);
    last = now;
    step++;
  }

  endwin();
  curses_active = false;
  gpu_free();
  return 0;
}
