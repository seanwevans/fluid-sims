// Jos Stam–style 'Stable Fluids' solver in double precision
// gcc -O3 -march=native -ffast-math -funroll-loops sim.c -lncursesw -lm -o sim && ./sim

#define _GNU_SOURCE

#include <locale.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <ncursesw/curses.h>

#define N 512                   // grid resolution
#define DT 1                    // time-step (τ)
#define VISC 1e-6               // viscosity
#define DIFF 1e-7               // density diffusion
#define DENS_DECAY (1-1e-6)     // global exponential fade

#define X0 1.0
#define Y0 1.0
#define ETA_MIN -1.5
#define ETA_MAX 1.5

#define DELAY_US 0

#define IX(i, j) ((i) + (N + 2) * (j))

static double *u, *v, *u0, *v0, *d, *d0; // fluid & density fields
static double dx[N + 2], dy[N + 2];      // non-uniform spacing

static void init_grid(void) {
  double dη = (ETA_MAX - ETA_MIN) / N;
  for (int i = 1; i <= N; i++) {
    double η = ETA_MIN + (i - 0.5) * dη;
    dx[i] = X0 * exp(η + dη / 2) - X0 * exp(η - dη / 2);
  }
  for (int j = 1; j <= N; j++) {
    double η = ETA_MIN + (j - 0.5) * dη;
    dy[j] = Y0 * exp(η + dη / 2) - Y0 * exp(η - dη / 2);
  }
}

static void alloc_fields(void) {
  size_t s = (N + 2) * (N + 2) * sizeof(double);
  u = calloc(1, s);
  v = calloc(1, s);
  u0 = calloc(1, s);
  v0 = calloc(1, s);
  d = calloc(1, s);
  d0 = calloc(1, s);
  if (!u || !v || !u0 || !v0 || !d || !d0) {
    endwin();
    perror("calloc");
    exit(EXIT_FAILURE);
  }
}

static void seed_initial(void) {
  int cx = N / 2, cy = N / 2;
  double R = N / 2.5, sw = 0.5;
  for (int j = 1; j <= N; j++)
    for (int i = 1; i <= N; i++) {
      double dx_ = i - cx, dy_ = j - cy, r2 = dx_ * dx_ + dy_ * dy_;
      if (r2 < R * R) {
        double r = sqrt(r2) + 1e-6;
        d[IX(i, j)] += 0.4 * exp(-r2 / (R * R));
        u[IX(i, j)] = -sw * dy_ / r;
        v[IX(i, j)] = sw * dx_ / r;
      }
    }
}

static void add_source(int step) {
  double ang = step * 0.015;
  int cx = N / 2 + (int)((N / 4) * cos(ang));
  int cy = N / 2 + (int)((N / 4) * sin(ang));
  double R = 3.0, swirl = 0.6;
  double amp = 0.5 + 0.4 * sin(step * 0.02);

  for (int j = cy - 2; j <= cy + 2; j++)
    for (int i = cx - 2; i <= cx + 2; i++) {
      if (i < 1 || i > N || j < 1 || j > N)
        continue;
      double dx_ = i - cx, dy_ = j - cy, r2 = dx_ * dx_ + dy_ * dy_;
      if (r2 > R * R)
        continue;
      double r = sqrt(r2) + 1e-6;
      d[IX(i, j)] += amp * exp(-r2 / (R * R));
      u[IX(i, j)] += -swirl * dy_ / r;
      v[IX(i, j)] += swirl * dx_ / r;
    }
}

static void bnd(int b, double *x) {
  for (int i = 1; i <= N; i++) {
    x[IX(0, i)] = (b == 1 ? -x[IX(1, i)] : x[IX(1, i)]);
    x[IX(N + 1, i)] = (b == 1 ? -x[IX(N, i)] : x[IX(N, i)]);
    x[IX(i, 0)] = (b == 2 ? -x[IX(i, 1)] : x[IX(i, 1)]);
    x[IX(i, N + 1)] = (b == 2 ? -x[IX(i, N)] : x[IX(i, N)]);
  }
  x[IX(0, 0)] = 0.5 * (x[IX(1, 0)] + x[IX(0, 1)]);
  x[IX(0, N + 1)] = 0.5 * (x[IX(1, N + 1)] + x[IX(0, N)]);
  x[IX(N + 1, 0)] = 0.5 * (x[IX(N, 0)] + x[IX(N + 1, 1)]);
  x[IX(N + 1, N + 1)] = 0.5 * (x[IX(N, N + 1)] + x[IX(N + 1, N)]);
}

static void lin(int b, double *x, double *x0, double a, double c) {
  for (int k = 0; k < 15; k++) {
    for (int j = 1; j <= N; j++)
      for (int i = 1; i <= N; i++)
        x[IX(i, j)] = (x0[IX(i, j)] + a * (x[IX(i - 1, j)] + x[IX(i + 1, j)] +
                                           x[IX(i, j - 1)] + x[IX(i, j + 1)])) /
                      c;
    bnd(b, x);
  }
}

static void diff(int b, double *x, double *x0, double diffc) {
  lin(b, x, x0, DT * diffc * N * N, 1 + 4 * DT * diffc * N * N);
}

static void adv(int b, double *q, double *q0, double *uu, double *vv) {
  double dη = (ETA_MAX - ETA_MIN) / N;
  for (int j = 1; j <= N; j++)
    for (int i = 1; i <= N; i++) {
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
  bnd(b, q);
}

static void proj(double *uu, double *vv, double *p, double *div) {
  for (int j = 1; j <= N; j++)
    for (int i = 1; i <= N; i++) {
      div[IX(i, j)] = -0.5 * ((uu[IX(i + 1, j)] - uu[IX(i - 1, j)]) / dx[i] +
                              (vv[IX(i, j + 1)] - vv[IX(i, j - 1)]) / dy[j]);
      p[IX(i, j)] = 0;
    }
  bnd(0, div);
  bnd(0, p);
  lin(0, p, div, 1, 4);
  for (int j = 1; j <= N; j++)
    for (int i = 1; i <= N; i++) {
      uu[IX(i, j)] -= 0.5 * dx[i] * (p[IX(i + 1, j)] - p[IX(i - 1, j)]);
      vv[IX(i, j)] -= 0.5 * dy[j] * (p[IX(i, j + 1)] - p[IX(i, j - 1)]);
    }
  bnd(1, uu);
  bnd(2, vv);
}

static void vel_step(void) {
  diff(1, u0, u, VISC);
  diff(2, v0, v, VISC);
  proj(u0, v0, u, v);
  adv(1, u, u0, u0, v0);
  adv(2, v, v0, u0, v0);
  proj(u, v, u0, v0);
}

static void dens_step(void) {
  diff(0, d0, d, DIFF);
  adv(0, d, d0, u, v);
}

static void decay_density(void) {
  for (int j = 1; j <= N; j++)
    for (int i = 1; i <= N; i++)
      d[IX(i, j)] *= DENS_DECAY;
}

static void show(int step) {
  static const wchar_t wramp[] = {L' ', L'░', L'▒', L'▓', L'█'};
  const int L = 4;

  int rows, cols;
  getmaxyx(stdscr, rows, cols);

  mvprintw(0, 0, "step %d   (press 'q' to quit)", step);

  int disp_rows = rows - 1;
  int disp_cols = cols;

  for (int j = 0; j < disp_rows; j++) {
    int sim_j = 1 + (int)((double)N * j / disp_rows);
    move(j + 1, 0);
    for (int i = 0; i < disp_cols; i++) {
      int sim_i = 1 + (int)((double)N * i / disp_cols);
      double val = d[IX(sim_i, sim_j)];
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

int main(void) {
  setlocale(LC_ALL, "");
  initscr();
  cbreak();
  noecho();
  curs_set(0);
  nodelay(stdscr, TRUE);
  keypad(stdscr, TRUE);

  alloc_fields();
  init_grid();
  seed_initial();

  int step = 0;
  while (1) {
    if (getch() == 'q')
      break;

    decay_density();
    add_source(step);

    vel_step();
    dens_step();

    show(step);

    if (DELAY_US)
      usleep(DELAY_US);

    step++;
  }

  endwin();
  return 0;
}