/* tau_hypersonic_2d.c
 * 2D Log-Space Hypersonic Flow over a Cylinder.
 *   gcc -O3 tau_hypersonic_2d.c -lraylib -lm -o tau_2d
 * Controls: SPACE (Pause), R (Reset), M (Toggle Mode: Density/Pressure/Speed)
 */

#include "raylib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define W 250
#define H 250
#define SCALE 2
#define GAMMA 1.4
#define CFL 0.9
#define DTAU 0.001
#define U0 100.0
#define STEPS_PER_FRAME 10

typedef struct {
  double chi;  // ln(rho)
  double phix; // asinh(u/U0)
  double phiy; // asinh(v/U0)
  double pi;   // ln(P)
} LogState;

typedef struct {
  double rho;
  double mx; // rho * u
  double my; // rho * v
  double E;  // Total Energy
} ConsState;

LogState cells[W * H];
LogState cells_new[W * H];
unsigned char *pixels;

unsigned char mask[W * H];
double t = 1.0;
int view_mode = 0; // 0=Density, 1=Pressure, 2=Speed

ConsState to_cons(LogState s) {
  double rho = exp(s.chi);
  double u = U0 * sinh(s.phix);
  double v = U0 * sinh(s.phiy);
  double P = exp(s.pi);

  ConsState c;
  c.rho = rho;
  c.mx = rho * u;
  c.my = rho * v;
  c.E = P / (GAMMA - 1.0) + 0.5 * rho * (u * u + v * v);
  return c;
}

LogState to_log(ConsState c) {
  const double EPS = 1e-12;
  double rho = fmax(c.rho, EPS);
  double inv_rho = 1.0 / rho;
  double u = c.mx * inv_rho;
  double v = c.my * inv_rho;

  double e_int = c.E - 0.5 * rho * (u * u + v * v);
  double P = (GAMMA - 1.0) * fmax(e_int, EPS);

  LogState s;
  s.chi = log(rho);
  s.phix = asinh(u / U0);
  s.phiy = asinh(v / U0);
  s.pi = log(P);
  return s;
}

double get_c(ConsState s) {
  double P = (GAMMA - 1.0) * (s.E - 0.5 * (s.mx * s.mx + s.my * s.my) / s.rho);
  return sqrt(GAMMA * fmax(P, 1e-9) / s.rho);
}

ConsState flux(ConsState c, double nx, double ny) {
  double u = c.mx / c.rho;
  double v = c.my / c.rho;
  double P = (GAMMA - 1.0) * (c.E - 0.5 * c.rho * (u * u + v * v));

  double vn = u * nx + v * ny;

  ConsState f;
  f.rho = c.rho * vn;
  f.mx = c.mx * vn + P * nx;
  f.my = c.my * vn + P * ny;
  f.E = vn * (c.E + P);
  return f;
}

// Directional HLL Flux Solver
ConsState hll_flux(LogState L_log, LogState R_log, double nx, double ny) {
  ConsState L = to_cons(L_log);
  ConsState R = to_cons(R_log);

  double uL = L.mx / L.rho, vL = L.my / L.rho;
  double uR = R.mx / R.rho, vR = R.my / R.rho;

  double vnL = uL * nx + vL * ny;
  double vnR = uR * nx + vR * ny;

  double cL = get_c(L);
  double cR = get_c(R);

  double SL = fmin(vnL - cL, vnR - cR);
  double SR = fmax(vnL + cL, vnR + cR);

  ConsState FL = flux(L, nx, ny);
  ConsState FR = flux(R, nx, ny);

  if (SL >= 0.0)
    return FL;
  if (SR <= 0.0)
    return FR;

  double inv = 1.0 / (SR - SL);
  ConsState F;
  F.rho = (SR * FL.rho - SL * FR.rho + SL * SR * (R.rho - L.rho)) * inv;
  F.mx = (SR * FL.mx - SL * FR.mx + SL * SR * (R.mx - L.mx)) * inv;
  F.my = (SR * FL.my - SL * FR.my + SL * SR * (R.my - L.my)) * inv;
  F.E = (SR * FL.E - SL * FR.E + SL * SR * (R.E - L.E)) * inv;
  return F;
}

void init_sim() {
  t = 1.0;

  int cx = W / 3, cy = H / 2, r = H / 5;

  for (int y = 0; y < H; y++) {
    for (int x = 0; x < W; x++) {
      int idx = y * W + x;
      int dx = x - cx;
      int dy = y - cy;
      mask[idx] = (dx * dx + dy * dy < r * r) ? 1 : 0;

      double rho = 1.0;
      double P = 1.0;
      double u = 0.0;

      if (mask[idx]) {
        rho = 1.0;
        P = 1.0;
        u = 0;
      }

      cells[idx].chi = log(rho);
      cells[idx].pi = log(P);
      cells[idx].phix = asinh(u / U0);
      cells[idx].phiy = asinh(0.0);
    }
  }
}

void step_physics() {
  // Boundary Conditions (Inflow Left)
  double mach_in = 15.0;
  double rho_in = 1.0;
  double P_in = 1.0;
  double c_in = sqrt(GAMMA * P_in / rho_in);
  double u_in = mach_in * c_in;

  for (int y = 0; y < H; y++) {
    LogState in;
    in.chi = log(rho_in);
    in.pi = log(P_in);
    in.phix = asinh(u_in / U0);
    in.phiy = 0.0;
    cells[y * W + 0] = in;
  }

  double max_wave = 1e-6;
  for (int i = 0; i < W * H; i++) {
    if (mask[i])
      continue;
    ConsState c = to_cons(cells[i]);
    double vel = sqrt(c.mx * c.mx + c.my * c.my) / c.rho;
    double cs = get_c(c);
    if (vel + cs > max_wave)
      max_wave = vel + cs;
  }
  double dt = fmin(CFL * 1.0 / max_wave, t * DTAU);

  for (int y = 1; y < H - 1; y++) {
    for (int x = 1; x < W - 1; x++) {
      int i = y * W + x;
      if (mask[i])
        continue;

      LogState L = cells[i - 1], C = cells[i], R = cells[i + 1];

      if (mask[i - 1]) {
        L = C;
        L.phix *= -1.0;
      }
      if (mask[i + 1]) {
        R = C;
        R.phix *= -1.0;
      }

      ConsState Fx_L = hll_flux(L, C, 1, 0);
      ConsState Fx_R = hll_flux(C, R, 1, 0);

      LogState B = cells[i - W], T = cells[i + W];
      if (mask[i - W]) {
        B = C;
        B.phiy *= -1.0;
      }
      if (mask[i + W]) {
        T = C;
        T.phiy *= -1.0;
      }

      ConsState Fy_B = hll_flux(B, C, 0, 1);
      ConsState Fy_T = hll_flux(C, T, 0, 1);

      ConsState U = to_cons(C);
      U.rho -= dt * (Fx_R.rho - Fx_L.rho + Fy_T.rho - Fy_B.rho);
      U.mx -= dt * (Fx_R.mx - Fx_L.mx + Fy_T.mx - Fy_B.mx);
      U.my -= dt * (Fx_R.my - Fx_L.my + Fy_T.my - Fy_B.my);
      U.E -= dt * (Fx_R.E - Fx_L.E + Fy_T.E - Fy_B.E);

      cells_new[i] = to_log(U);
    }
  }

  for (int i = 0; i < W * H; i++)
    if (!mask[i])
      cells[i] = cells_new[i];

  t += dt;
}

Color get_color(double val, double min, double max) {
  double t = (val - min) / (max - min);
  if (t < 0)
    t = 0;
  if (t > 1)
    t = 1;

  unsigned char r =
      (unsigned char)(255 * fmin(1, fmax(0, 3 * t - 1))); // Red late
  unsigned char g =
      (unsigned char)(255 * fmin(1, fmax(0, 2 - 4 * fabs(t - 0.5))));
  unsigned char b =
      (unsigned char)(255 * fmin(1, fmax(0, 2 - 3 * t))); // Blue early
  return (Color){r, g, b, 255};
}

int main() {
  InitWindow(W * SCALE, H * SCALE, "Hypersonic 2D Flow");
  SetTargetFPS(60);

  pixels = (unsigned char *)malloc(W * H * 4);
  Image img = {.data = pixels,
               .width = W,
               .height = H,
               .mipmaps = 1,
               .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8};
  Texture2D tex = LoadTextureFromImage(img);

  init_sim();

  while (!WindowShouldClose()) {
    if (IsKeyPressed(KEY_R))
      init_sim();
    if (IsKeyPressed(KEY_M))
      view_mode = (view_mode + 1) % 3;
    if (!IsKeyDown(KEY_SPACE)) {
      for (int k = 0; k < STEPS_PER_FRAME; k++)
        step_physics();
    }

    double min_v = 1e9, max_v = -1e9;
    for (int i = 0; i < W * H; i++) {
      if (mask[i])
        continue;
      double val = 0;
      if (view_mode == 0)
        val = cells[i].chi; // Density
      if (view_mode == 1)
        val = cells[i].pi;  // Pressure
      if (view_mode == 2) { // Speed
        double u = sinh(cells[i].phix) * U0;
        double v = sinh(cells[i].phiy) * U0;
        val = sqrt(u * u + v * v);
      }
      if (val < min_v)
        min_v = val;
      if (val > max_v)
        max_v = val;
    }

    for (int y = 0; y < H; y++) {
      for (int x = 0; x < W; x++) {
        int i = y * W + x;
        int p_idx = (y * W + x) * 4;
        if (mask[i]) {
          pixels[p_idx] = 100;
          pixels[p_idx + 1] = 100;
          pixels[p_idx + 2] = 100;
          pixels[p_idx + 3] = 255;
        } else {
          double val = 0;
          if (view_mode == 0)
            val = cells[i].chi;
          if (view_mode == 1)
            val = cells[i].pi;
          if (view_mode == 2) {
            double u = sinh(cells[i].phix) * U0;
            double v = sinh(cells[i].phiy) * U0;
            val = sqrt(u * u + v * v);
          }
          Color c = get_color(val, min_v, max_v);
          pixels[p_idx] = c.r;
          pixels[p_idx + 1] = c.g;
          pixels[p_idx + 2] = c.b;
          pixels[p_idx + 3] = 255;
        }
      }
    }

    UpdateTexture(tex, pixels);

    BeginDrawing();
    ClearBackground(BLACK);
    DrawTexturePro(tex, (Rectangle){0, 0, W, H},
                   (Rectangle){0, 0, W * SCALE, H * SCALE}, (Vector2){0, 0},
                   0.0f, WHITE);
    DrawText(TextFormat("Time: %.3f", t - 1.0), 10, 10, 20, WHITE);
    const char *mode_str = (view_mode == 0)
                               ? "Log-Density"
                               : (view_mode == 1 ? "Log-Pressure" : "Speed");
    DrawText(mode_str, 10, 30, 20, GREEN);
    EndDrawing();
  }

  CloseWindow();
  free(pixels);
  return 0;
}
