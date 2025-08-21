// js_cuda.cu : CUDA fluid sim with ncurses rendering
// nvcc -O3 -arch=sm_86 js_cuda.cu -o jsc -lncursesw

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

#define N 512
#define DT 1.0
#define VISC 1e-6
#define DIFF 1e-7
#define DENS_DECAY (1 - 1e-6)

#define X0 1.0
#define Y0 1.0
#define ETA_MIN -1.5
#define ETA_MAX 1.5

#define IX(i, j) ((i) + (N + 2) * (j))

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
double *d_u, *d_v, *d_u0, *d_v0, *d_d, *d_d0, *d_dx, *d_dy, *d_tmp;
double *d_p, *d_div;
double *h_d;

dim3 bs(16, 16);
dim3 gs((N + bs.x - 1) / bs.x, (N + bs.y - 1) / bs.y);

// kernels
__global__ void k_decay(double *d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i <= N && j <= N)
    d[IX(i, j)] *= DENS_DECAY;
}

__global__ void k_seed(double *u, double *v, double *d) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  int cx = N / 2, cy = N / 2;
  double R = N / 2.5, sw = 0.5;
  double dx = i - cx, dy = j - cy, r2 = dx * dx + dy * dy;
  if (r2 < R * R) {
    double r = sqrt(r2) + 1e-6;
    d[IX(i, j)] += 0.4 * exp(-r2 / (R * R));
    u[IX(i, j)] = -sw * dy / r;
    v[IX(i, j)] = sw * dx / r;
  }
}

__global__ void k_lin(double *x_new, double *x_old, double *x0, double a,
                      double c) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i <= N && j <= N) {
    x_new[IX(i, j)] =
        (x0[IX(i, j)] + a * (x_old[IX(i - 1, j)] + x_old[IX(i + 1, j)] +
                             x_old[IX(i, j - 1)] + x_old[IX(i, j + 1)])) /
        c;
  }
}

__global__ void k_adv(double *q, double *q0, double *uu, double *vv) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i <= N && j <= N) {
    double dη = (ETA_MAX - ETA_MIN) / N;
    double ηx = ETA_MIN + (i - 0.5) * dη;
    double ηy = ETA_MIN + (j - 0.5) * dη;
    double xp = X0 * exp(ηx), yp = Y0 * exp(ηy);
    double bx = ηx - DT * uu[IX(i, j)] / xp;
    double by = ηy - DT * vv[IX(i, j)] / yp;
    double s = (bx - ETA_MIN) / dη + 0.5;
    double t = (by - ETA_MIN) / dη + 0.5;
    s = fmin(fmax(s, 0.5), N + 0.5);
    t = fmin(fmax(t, 0.5), N + 0.5);
    int i0 = (int)s, i1 = i0 + 1;
    int j0 = (int)t, j1 = j0 + 1;
    double s1 = s - i0, s0 = 1 - s1;
    double t1 = t - j0, t0 = 1 - t1;
    q[IX(i, j)] = s0 * (t0 * q0[IX(i0, j0)] + t1 * q0[IX(i0, j1)]) +
                  s1 * (t0 * q0[IX(i1, j0)] + t1 * q0[IX(i1, j1)]);
  }
}

__global__ void k_div(double *uu, double *vv, double *div, double *p,
                      double *dx, double *dy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i <= N && j <= N) {
    div[IX(i, j)] = -0.5 * ((uu[IX(i + 1, j)] - uu[IX(i - 1, j)]) / dx[i] +
                            (vv[IX(i, j + 1)] - vv[IX(i, j - 1)]) / dy[j]);
    p[IX(i, j)] = 0;
  }
}

__global__ void k_proj(double *uu, double *vv, double *p, double *dx,
                       double *dy) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  if (i <= N && j <= N) {
    uu[IX(i, j)] -= 0.5 * dx[i] * (p[IX(i + 1, j)] - p[IX(i - 1, j)]);
    vv[IX(i, j)] -= 0.5 * dy[j] * (p[IX(i, j + 1)] - p[IX(i, j - 1)]);
  }
}

__global__ void k_add_source(double *u, double *v, double *d, int step) {
  int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
  int j = blockIdx.y * blockDim.y + threadIdx.y + 1;
  double ang = step * 0.015;
  int cx = N / 2 + (int)((N / 4) * cos(ang));
  int cy = N / 2 + (int)((N / 4) * sin(ang));
  double R = 3.0, swirl = 0.6, amp = 0.5 + 0.4 * sin(step * 0.02);
  double dx = i - cx, dy = j - cy, r2 = dx * dx + dy * dy;
  if (r2 < R * R) {
    double r = sqrt(r2) + 1e-6;
    d[IX(i, j)] += amp * exp(-r2 / (R * R));
    u[IX(i, j)] += -swirl * dy / r;
    v[IX(i, j)] += swirl * dx / r;
  }
}

// --- sim helpers ---
void lin_solve(double *x, double *x0, double a, double c, int iters) {
  double *read = x;
  double *write = d_tmp;
  for (int k = 0; k < iters; k++) {
    k_lin<<<gs, bs>>>(write, read, x0, a, c);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    double *tmp = read;
    read = write;
    write = tmp;
  }
  if (read != x) {
    CUDA_CHECK(cudaMemcpy(x, read, (N + 2) * (N + 2) * sizeof(double),
                          cudaMemcpyDeviceToDevice));
  }
}

void diffuse(double *x, double *x0, double diffc) {
  double a = DT * diffc * N * N;
  lin_solve(x, x0, a, 1 + 4 * a, 40);
}

void vel_step() {
  diffuse(d_u0, d_u, VISC);
  diffuse(d_v0, d_v, VISC);
  k_div<<<gs, bs>>>(d_u0, d_v0, d_div, d_p, d_dx, d_dy);
  CUDA_CHECK(cudaGetLastError());
  lin_solve(d_p, d_div, 1.0, 4.0, 40);
  k_proj<<<gs, bs>>>(d_u0, d_v0, d_p, d_dx, d_dy);
  CUDA_CHECK(cudaGetLastError());
  k_adv<<<gs, bs>>>(d_u, d_u0, d_u0, d_v0);
  CUDA_CHECK(cudaGetLastError());
  k_adv<<<gs, bs>>>(d_v, d_v0, d_u0, d_v0);
  CUDA_CHECK(cudaGetLastError());
  k_div<<<gs, bs>>>(d_u, d_v, d_div, d_p, d_dx, d_dy);
  CUDA_CHECK(cudaGetLastError());
  lin_solve(d_p, d_div, 1.0, 4.0, 40);
  k_proj<<<gs, bs>>>(d_u, d_v, d_p, d_dx, d_dy);
  CUDA_CHECK(cudaGetLastError());
}

void dens_step() {
  diffuse(d_d0, d_d, DIFF);
  k_adv<<<gs, bs>>>(d_tmp, d_d0, d_u, d_v);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaMemcpy(d_d, d_tmp, (N + 2) * (N + 2) * sizeof(double),
                        cudaMemcpyDeviceToDevice));
}

// --- init + alloc ---
void seed_initial() { k_seed<<<gs, bs>>>(d_u, d_v, d_d); }

void init_grid() {
  double *hx = (double *)malloc((N + 2) * sizeof(double));
  double *hy = (double *)malloc((N + 2) * sizeof(double));
  double dη = (ETA_MAX - ETA_MIN) / N;
  for (int i = 1; i <= N; i++) {
    double η = ETA_MIN + (i - 0.5) * dη;
    hx[i] = X0 * exp(η + dη / 2) - X0 * exp(η - dη / 2);
  }
  for (int j = 1; j <= N; j++) {
    double η = ETA_MIN + (j - 0.5) * dη;
    hy[j] = Y0 * exp(η + dη / 2) - Y0 * exp(η - dη / 2);
  }
  CUDA_CHECK(
      cudaMemcpy(d_dx, hx, (N + 2) * sizeof(double), cudaMemcpyHostToDevice));
  CUDA_CHECK(
      cudaMemcpy(d_dy, hy, (N + 2) * sizeof(double), cudaMemcpyHostToDevice));
  free(hx);
  free(hy);
}

void gpu_alloc() {
  size_t s = (N + 2) * (N + 2) * sizeof(double);
  CUDA_CHECK(cudaMalloc(&d_dx, (N + 2) * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_dy, (N + 2) * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_u, s));
  CUDA_CHECK(cudaMalloc(&d_v, s));
  CUDA_CHECK(cudaMalloc(&d_u0, s));
  CUDA_CHECK(cudaMalloc(&d_v0, s));
  CUDA_CHECK(cudaMalloc(&d_d, s));
  CUDA_CHECK(cudaMalloc(&d_d0, s));
  CUDA_CHECK(cudaMalloc(&d_tmp, s));
  CUDA_CHECK(cudaMalloc(&d_p, s));
  CUDA_CHECK(cudaMalloc(&d_div, s));
  CUDA_CHECK(cudaMemset(d_u, 0, s));
  CUDA_CHECK(cudaMemset(d_v, 0, s));
  CUDA_CHECK(cudaMemset(d_u0, 0, s));
  CUDA_CHECK(cudaMemset(d_v0, 0, s));
  CUDA_CHECK(cudaMemset(d_d, 0, s));
  CUDA_CHECK(cudaMemset(d_d0, 0, s));
  CUDA_CHECK(cudaMemset(d_p, 0, s));
  CUDA_CHECK(cudaMemset(d_div, 0, s));
  CUDA_CHECK(cudaMallocHost(&h_d, s));
}

void gpu_free() {
  cudaFree(d_dx);
  cudaFree(d_dy);
  cudaFree(d_u);
  cudaFree(d_v);
  cudaFree(d_u0);
  cudaFree(d_v0);
  cudaFree(d_d);
  cudaFree(d_d0);
  cudaFree(d_tmp);
  cudaFree(d_p);
  cudaFree(d_div);
  cudaFreeHost(h_d);
}

// display
void show_ncurses(int step, double fps, double avgfps) {
  static const wchar_t wramp[] = {L' ', L'░', L'▒', L'▓', L'█'};
  const int L = 4;
  int rows, cols;
  getmaxyx(stdscr, rows, cols);
  mvprintw(0, 0, "step %d   FPS: %.1f (avg %.1f)  (press 'q' to quit)", step,
           fps, avgfps);
  CUDA_CHECK(cudaMemcpy(h_d, d_d, (N + 2) * (N + 2) * sizeof(double),
                        cudaMemcpyDeviceToHost));
  int disp_rows = rows - 1, disp_cols = cols;
  for (int j = 0; j < disp_rows; j++) {
    int sim_j = 1 + (int)((double)N * j / disp_rows);
    move(j + 1, 0);
    for (int i = 0; i < disp_cols; i++) {
      int sim_i = 1 + (int)((double)N * i / disp_cols);
      double val = h_d[IX(sim_i, sim_j)];
      val = (val < 0) ? 0 : (val > 1) ? 1 : val;
      int idx = (int)(val * L + 0.5);
      if (idx > L)
        idx = L;
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
  fprintf(stderr, "\n[trap] Caught signal %d, exiting...\n", sig);
  exit(1);
}

int main(int argc, char **argv) {
  bool headless = false;
  int steps = 0;
  int stride = 1;    // render every frame by default
  int fps_limit = 0; // uncapped

  const struct option long_opts[] = {{"headless", no_argument, 0, 'H'},
                                     {"steps", required_argument, 0, 's'},
                                     {"stride", required_argument, 0, 'r'},
                                     {"fps", required_argument, 0, 'f'},
                                     {"help", no_argument, 0, '?'},
                                     {0, 0, 0, 0}};

  int opt, idx;
  while ((opt = getopt_long(argc, argv, "Hs:r:f:?", long_opts, &idx)) != -1) {
    switch (opt) {
    case 'H':
      headless = true;
      break;
    case 's':
      steps = atoi(optarg);
      break;
    case 'r':
      stride = atoi(optarg);
      break;
    case 'f':
      fps_limit = atoi(optarg);
      break;
    case '?':
    default:
      printf("Usage: %s [options]\n", argv[0]);
      puts("  -H, --headless  Run headless benchmark");
      puts("  -s, --steps N   Number of simulation steps (0 = forever)");
      puts("  -r, --stride N  Render every Nth frame (default 1)");
      puts("  -f, --fps N     Limit FPS to N (0 = uncapped)");
      puts("  -?, --help      Show this help message");
      exit(0);
    }
  }

  signal(SIGINT, handle_exit);
  signal(SIGTERM, handle_exit);

  gpu_alloc();
  init_grid();
  seed_initial();
  CUDA_CHECK(cudaDeviceSynchronize());

  if (!headless) {
    curses_active = true;
    setlocale(LC_ALL, "");
    initscr();
    cbreak();
    noecho();
    curs_set(0);
    nodelay(stdscr, TRUE);
    keypad(stdscr, TRUE);

    int step = 0, frames = 0;
    auto start_all = std::chrono::high_resolution_clock::now();
    auto last = std::chrono::high_resolution_clock::now();
    double avgfps = 0.0, fps = 0.0, dt = 0.0;

    while (step < steps || steps == 0) {
      if (getch() == 'q')
        break;

      k_decay<<<gs, bs>>>(d_d);
      CUDA_CHECK(cudaGetLastError());

      k_add_source<<<gs, bs>>>(d_u, d_v, d_d, step);
      CUDA_CHECK(cudaGetLastError());

      vel_step();
      dens_step();

      if (step % stride == 0) {
        auto now = std::chrono::high_resolution_clock::now();
        dt = std::chrono::duration<double>(now - last).count();
        fps = (dt > 0) ? 1.0 / dt : 0.0;
        avgfps = 0.95 * avgfps + 0.05 * fps;
        show_ncurses(step, fps, avgfps);
        last = now;
        frames++;

        if (fps_limit > 0) {
          double target = 1.0 / fps_limit;
          if (dt < target) {
            int us = (int)((target - dt) * 1e6);
            if (us > 0)
              usleep(us);
          }
        }
      }
      step++;
    }

    endwin();
    curses_active = false;
    auto end_all = std::chrono::high_resolution_clock::now();
    double secs = std::chrono::duration<double>(end_all - start_all).count();
    printf("Interactive session summary (stride=%d, fps_limit=%d):\n", stride,
           fps_limit);
    printf("  Simulated steps: %d\n", step);
    printf("  Rendered frames: %d in %.3f s → avg %.1f FPS\n", frames, secs,
           (frames > 0 ? frames / secs : 0.0));

  } else {
    using clock = std::chrono::high_resolution_clock;
    auto start = clock::now();
    cudaEvent_t ev_start, ev_end;
    CUDA_CHECK(cudaEventCreate(&ev_start));
    CUDA_CHECK(cudaEventCreate(&ev_end));
    CUDA_CHECK(cudaEventRecord(ev_start));

    int frames = 0;
    for (int step = 0; step < steps; step++) {
      k_decay<<<gs, bs>>>(d_d);
      CUDA_CHECK(cudaGetLastError());

      k_add_source<<<gs, bs>>>(d_u, d_v, d_d, step);
      CUDA_CHECK(cudaGetLastError());

      vel_step();
      dens_step();

      if (step % stride == 0)
        frames++;
    }

    CUDA_CHECK(cudaEventRecord(ev_end));
    CUDA_CHECK(cudaEventSynchronize(ev_end));
    auto end = clock::now();

    float ms_gpu = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_gpu, ev_start, ev_end));
    double secs = std::chrono::duration<double>(end - start).count();

    printf("Headless benchmark (stride=%d):\n", stride);
    printf("  Simulated steps: %d\n", steps);
    printf("  Wall-clock: %d frames in %.3f s → %.1f FPS\n", frames, secs,
           frames / secs);
    printf("  GPU only:   %d frames in %.3f s → %.1f FPS\n", frames,
           ms_gpu / 1000.0, 1000.0 * frames / ms_gpu);

    CUDA_CHECK(cudaEventDestroy(ev_start));
    CUDA_CHECK(cudaEventDestroy(ev_end));
  }

  gpu_free();
  return 0;
}
