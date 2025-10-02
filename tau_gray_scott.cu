// tau_gray_scott.cu — CUDA Gray-Scott reaction-diffusion simulator with ncurses
//
// Build:
//   nvcc -std=c++17 -O3 -use_fast_math -arch=sm_86 -lineinfo -o tau_gray_scott \
//        tau_gray_scott.cu -lncursesw
//
// Run:
//   ./tau_gray_scott --nx 256 --ny 256 --Du 0.2 --Dv 0.1 --F 0.0367 --k 0.0649 \
//       --dt 1.0 --stride 4 --fps 30
//   ./tau_gray_scott --headless --steps 10000
//
// The simulation integrates the Gray-Scott reaction-diffusion equations on a
// periodic 2-D domain:
//   du/dt = Du ∇²u - uv² + F (1 - u)
//   dv/dt = Dv ∇²v + uv² - (F + k) v
// using an explicit five-point Laplacian. Rendering is handled via ncurses using
// a simple ASCII ramp based on the inhibitor (v) concentration.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#include <getopt.h>
#include <locale.h>
#include <ncursesw/curses.h>

#include <cuda_runtime.h>

static bool curses_active = false;

#define CUDA_CHECK(ans)                                                         \
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
  int nx = 256;
  int ny = 256;
  float dx = 1.0f;
  float dt = 1.0f;

  float Du = 0.2f;
  float Dv = 0.1f;
  float feed = 0.0367f;
  float kill = 0.0649f;

  int steps = 0; // 0 = run forever
  bool headless = false;
  int stride = 4;
  int fps_limit = 30;
  unsigned seed = 1337;
};

void usage(const char *prog) {
  printf("Usage: %s [options]\n", prog);
  puts("  --nx N        grid cells in x (256)");
  puts("  --ny N        grid cells in y (256)");
  puts("  --dx DX       cell size (1)");
  puts("  --dt DT       time step (1)");
  puts("  --Du D        diffusion coefficient for U (0.2)");
  puts("  --Dv D        diffusion coefficient for V (0.1)");
  puts("  --F F         feed rate (0.0367)");
  puts("  --k K         kill rate (0.0649)");
  puts("  --steps K     number of steps (0 = infinite)");
  puts("  --headless    disable ncurses rendering");
  puts("  --stride N    render every N steps (4)");
  puts("  --fps N       cap display FPS (30, 0 = uncapped)");
  puts("  --seed S      RNG seed for initial pattern (1337)");
  puts("  -h, --help    show this help message");
}

void parse_args(int argc, char **argv, Params &P) {
  static const struct option long_opts[] = {
      {"nx", required_argument, 0, 0},     {"ny", required_argument, 0, 0},
      {"dx", required_argument, 0, 0},     {"dt", required_argument, 0, 0},
      {"Du", required_argument, 0, 0},     {"Dv", required_argument, 0, 0},
      {"F", required_argument, 0, 0},      {"k", required_argument, 0, 0},
      {"steps", required_argument, 0, 0},  {"headless", no_argument, 0, 0},
      {"stride", required_argument, 0, 0}, {"fps", required_argument, 0, 0},
      {"seed", required_argument, 0, 0},   {"help", no_argument, 0, 'h'},
      {0, 0, 0, 0}};

  while (1) {
    int idx = 0;
    int c = getopt_long(argc, argv, "h", long_opts, &idx);
    if (c == -1)
      break;
    if (c == 'h') {
      usage(argv[0]);
      exit(EXIT_SUCCESS);
    }
    if (c)
      continue;
    const char *opt = long_opts[idx].name;
    if (strcmp(opt, "nx") == 0)
      P.nx = atoi(optarg);
    else if (strcmp(opt, "ny") == 0)
      P.ny = atoi(optarg);
    else if (strcmp(opt, "dx") == 0)
      P.dx = atof(optarg);
    else if (strcmp(opt, "dt") == 0)
      P.dt = atof(optarg);
    else if (strcmp(opt, "Du") == 0)
      P.Du = atof(optarg);
    else if (strcmp(opt, "Dv") == 0)
      P.Dv = atof(optarg);
    else if (strcmp(opt, "F") == 0)
      P.feed = atof(optarg);
    else if (strcmp(opt, "k") == 0)
      P.kill = atof(optarg);
    else if (strcmp(opt, "steps") == 0)
      P.steps = atoi(optarg);
    else if (strcmp(opt, "headless") == 0)
      P.headless = true;
    else if (strcmp(opt, "stride") == 0)
      P.stride = std::max(1, atoi(optarg));
    else if (strcmp(opt, "fps") == 0)
      P.fps_limit = atoi(optarg);
    else if (strcmp(opt, "seed") == 0)
      P.seed = static_cast<unsigned>(strtoul(optarg, nullptr, 10));
  }
}

__device__ inline int wrap(int i, int n) {
  return (i % n + n) % n;
}

__global__ void step_kernel(float *u_next, float *v_next, const float *u_cur,
                            const float *v_cur, int nx, int ny, float Du,
                            float Dv, float dt, float dx, float feed,
                            float kill) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= nx || j >= ny)
    return;

  int idx = j * nx + i;
  int ip = wrap(i + 1, nx);
  int im = wrap(i - 1, nx);
  int jp = wrap(j + 1, ny);
  int jm = wrap(j - 1, ny);

  float u = u_cur[idx];
  float v = v_cur[idx];
  float lap_u = (u_cur[j * nx + ip] + u_cur[j * nx + im] +
                 u_cur[jp * nx + i] + u_cur[jm * nx + i] - 4.0f * u) /
                (dx * dx);
  float lap_v = (v_cur[j * nx + ip] + v_cur[j * nx + im] +
                 v_cur[jp * nx + i] + v_cur[jm * nx + i] - 4.0f * v) /
                (dx * dx);

  float uvv = u * v * v;
  float du = Du * lap_u - uvv + feed * (1.0f - u);
  float dv = Dv * lap_v + uvv - (feed + kill) * v;

  u_next[idx] = u + dt * du;
  v_next[idx] = v + dt * dv;
}

void init_pattern(std::vector<float> &u, std::vector<float> &v, int nx, int ny,
                  unsigned seed) {
  std::fill(u.begin(), u.end(), 1.0f);
  std::fill(v.begin(), v.end(), 0.0f);

  int cx = nx / 2;
  int cy = ny / 2;
  int r = std::min(nx, ny) / 12;
  for (int j = -r; j <= r; ++j) {
    for (int i = -r; i <= r; ++i) {
      int x = (cx + i + nx) % nx;
      int y = (cy + j + ny) % ny;
      u[y * nx + x] = 0.50f;
      v[y * nx + x] = 0.25f;
    }
  }

  // sprinkle random seeds for variety
  uint32_t state = seed ? seed : 1u;
  auto rng = [&]() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
  };

  for (int n = 0; n < 64; ++n) {
    int x = rng() % nx;
    int y = rng() % ny;
    u[y * nx + x] = 0.35f;
    v[y * nx + x] = 0.65f;
  }
}

void render_field(const float *v_field, int nx, int ny, int step, const Params &P,
                  float elapsed_ms) {
  static const char ramp[] = " .:-=+*#%@";
  const int ramp_len = sizeof(ramp) - 1;

  int rows = std::min(ny, LINES - 2);
  int cols = std::min(nx, COLS);

  float vmin = 1e9f, vmax = -1e9f;
  for (int j = 0; j < ny; ++j)
    for (int i = 0; i < nx; ++i) {
      float v = v_field[j * nx + i];
      vmin = std::min(vmin, v);
      vmax = std::max(vmax, v);
    }
  float inv = (vmax > vmin) ? 1.0f / (vmax - vmin) : 1.0f;

  for (int j = 0; j < rows; ++j) {
    for (int i = 0; i < cols; ++i) {
      float v = v_field[j * nx + i];
      float t = (v - vmin) * inv;
      int idx = std::clamp(int(t * ramp_len), 0, ramp_len - 1);
      mvaddch(j, i, ramp[idx]);
    }
  }
  for (int j = rows; j < LINES - 1; ++j)
    mvhline(j, 0, ' ', COLS);

  mvprintw(LINES - 1, 0,
           "step=%d dt=%.3f F=%.4f k=%.4f Du=%.3f Dv=%.3f frame=%.2fms",
           step, P.dt, P.feed, P.kill, P.Du, P.Dv, elapsed_ms);
  refresh();
}

int main(int argc, char **argv) {
  setlocale(LC_ALL, "");
  Params P;
  parse_args(argc, argv, P);
  if (P.nx <= 0 || P.ny <= 0) {
    fprintf(stderr, "Grid dimensions must be positive\n");
    return EXIT_FAILURE;
  }

  size_t N = static_cast<size_t>(P.nx) * static_cast<size_t>(P.ny);
  std::vector<float> h_u(N), h_v(N);
  init_pattern(h_u, h_v, P.nx, P.ny, P.seed);

  float *d_u0 = nullptr, *d_u1 = nullptr, *d_v0 = nullptr, *d_v1 = nullptr;
  CUDA_CHECK(cudaMalloc(&d_u0, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_u1, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v0, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_v1, N * sizeof(float)));

  CUDA_CHECK(cudaMemcpy(d_u0, h_u.data(), N * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_v0, h_v.data(), N * sizeof(float),
                        cudaMemcpyHostToDevice));

  float *h_v_pinned = nullptr;
  if (!P.headless) {
    CUDA_CHECK(cudaHostAlloc(&h_v_pinned, N * sizeof(float), cudaHostAllocDefault));
    initscr();
    curses_active = true;
    noecho();
    curs_set(0);
    nodelay(stdscr, TRUE);
  }

  dim3 block(16, 16);
  dim3 grid((P.nx + block.x - 1) / block.x,
            (P.ny + block.y - 1) / block.y);

  int step = 0;
  bool running = true;
  while (running && (P.steps == 0 || step < P.steps)) {
    auto frame_start = std::chrono::steady_clock::now();
    step_kernel<<<grid, block>>>(d_u1, d_v1, d_u0, d_v0, P.nx, P.ny, P.Du,
                                 P.Dv, P.dt, P.dx, P.feed, P.kill);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::swap(d_u0, d_u1);
    std::swap(d_v0, d_v1);
    ++step;

    if (!P.headless && (step % P.stride == 0)) {
      CUDA_CHECK(cudaMemcpy(h_v_pinned, d_v0, N * sizeof(float),
                            cudaMemcpyDeviceToHost));
      auto now = std::chrono::steady_clock::now();
      float ms = std::chrono::duration<float, std::milli>(now - frame_start).count();
      render_field(h_v_pinned, P.nx, P.ny, step, P, ms);
      int ch = getch();
      if (ch == 'q' || ch == 'Q')
        running = false;
      if (P.fps_limit > 0) {
        auto target = std::chrono::milliseconds(1000 / P.fps_limit);
        auto elapsed = std::chrono::steady_clock::now() - frame_start;
        if (elapsed < target)
          std::this_thread::sleep_for(target - elapsed);
      }
    }
  }

  if (curses_active) {
    endwin();
    curses_active = false;
  }

  if (!P.headless)
    CUDA_CHECK(cudaFreeHost(h_v_pinned));

  CUDA_CHECK(cudaFree(d_u0));
  CUDA_CHECK(cudaFree(d_u1));
  CUDA_CHECK(cudaFree(d_v0));
  CUDA_CHECK(cudaFree(d_v1));
  return EXIT_SUCCESS;
}
