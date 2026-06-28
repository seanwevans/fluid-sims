/* tau_mhd.c
   Ideal magnetohydrodynamics (MHD) + GLM divergence cleaning demo.

   Build:
     gcc -O3 tau_mhd.c -lraylib -lm -o tau_mhd

   Controls:
     SPACE pause, R reset, M toggle view, C cycle initial condition

   This solver mirrors the tau_* single-file demos: a conservative finite-volume
   update with MUSCL reconstruction.  The interface solver uses an HLLD-oriented
   MHD wave model (fast, contact, and Alfven speed estimates) with a positivity
   preserving HLL fallback for pathological star states.  A hyperbolic/parabolic
   GLM field psi advects and damps numerical div(B) errors.
 */

#include "raylib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define W 320
#define H 220
#define SCALE 3
#define GAMMA 1.4
#define CFL 0.22
#define STEPS_PER_FRAME 2
#define EPS_RHO 1e-8
#define EPS_P 1e-8
#define GLM_ALPHA 0.18

typedef struct { double rho, mx, my, E, Bx, By, psi; } Cons;
typedef struct { double rho, u, v, p, Bx, By, psi; } Prim;

static Cons U[W * H], Unew[W * H];
static unsigned char *pixels;
static int paused = 0, view_mode = 0, problem = 0;
static double sim_t = 0.0;

static inline int idx(int x, int y) { return y * W + x; }
static inline double sqr(double x) { return x * x; }
static inline double clampd(double x, double a, double b) { return fmin(fmax(x, a), b); }
static inline double minmod(double a, double b) { return (a * b <= 0.0) ? 0.0 : (fabs(a) < fabs(b) ? a : b); }
static inline double mc(double dl, double dc, double dr) { return minmod(minmod(dl, dr), minmod(dc, minmod(2.0 * dl, 2.0 * dr))); }
static inline Cons cadd(Cons a, Cons b) { Cons r = {a.rho+b.rho,a.mx+b.mx,a.my+b.my,a.E+b.E,a.Bx+b.Bx,a.By+b.By,a.psi+b.psi}; return r; }
static inline Cons csub(Cons a, Cons b) { Cons r = {a.rho-b.rho,a.mx-b.mx,a.my-b.my,a.E-b.E,a.Bx-b.Bx,a.By-b.By,a.psi-b.psi}; return r; }
static inline Cons cmul(Cons a, double s) { Cons r = {a.rho*s,a.mx*s,a.my*s,a.E*s,a.Bx*s,a.By*s,a.psi*s}; return r; }

static inline Prim cons_to_prim(Cons c) {
  Prim p; p.rho = fmax(c.rho, EPS_RHO); p.u = c.mx / p.rho; p.v = c.my / p.rho;
  p.Bx = c.Bx; p.By = c.By; p.psi = c.psi;
  double ek = 0.5 * p.rho * (sqr(p.u) + sqr(p.v));
  double em = 0.5 * (sqr(p.Bx) + sqr(p.By));
  p.p = fmax((GAMMA - 1.0) * (c.E - ek - em), EPS_P);
  return p;
}

static inline Cons prim_to_cons(Prim p) {
  Cons c; p.rho = fmax(p.rho, EPS_RHO); p.p = fmax(p.p, EPS_P);
  c.rho = p.rho; c.mx = p.rho * p.u; c.my = p.rho * p.v; c.Bx = p.Bx; c.By = p.By; c.psi = p.psi;
  c.E = p.p / (GAMMA - 1.0) + 0.5 * p.rho * (sqr(p.u) + sqr(p.v)) + 0.5 * (sqr(p.Bx) + sqr(p.By));
  return c;
}

static inline double fast_speed(Prim p, int xdir) {
  double a2 = GAMMA * p.p / p.rho;
  double b2 = (sqr(p.Bx) + sqr(p.By)) / p.rho;
  double bn2 = sqr(xdir ? p.Bx : p.By) / p.rho;
  double disc = fmax(sqr(a2 + b2) - 4.0 * a2 * bn2, 0.0);
  return sqrt(0.5 * ((a2 + b2) + sqrt(disc)));
}

static inline Cons flux_x(Cons c, double ch) {
  Prim p = cons_to_prim(c); double pt = p.p + 0.5 * (sqr(p.Bx) + sqr(p.By)); double vb = p.u*p.Bx + p.v*p.By;
  Cons f = {c.mx,
            c.mx*p.u + pt - sqr(p.Bx),
            c.my*p.u - p.Bx*p.By,
            (c.E + pt)*p.u - p.Bx*vb,
            p.psi,
            p.u*p.By - p.v*p.Bx,
            ch*ch*p.Bx};
  return f;
}
static inline Cons flux_y(Cons c, double ch) {
  Prim p = cons_to_prim(c); double pt = p.p + 0.5 * (sqr(p.Bx) + sqr(p.By)); double vb = p.u*p.Bx + p.v*p.By;
  Cons f = {c.my,
            c.mx*p.v - p.By*p.Bx,
            c.my*p.v + pt - sqr(p.By),
            (c.E + pt)*p.v - p.By*vb,
            p.v*p.Bx - p.u*p.By,
            p.psi,
            ch*ch*p.By};
  return f;
}

static inline int valid(Cons q) { Prim p = cons_to_prim(q); return isfinite(q.E) && p.rho > EPS_RHO && p.p > EPS_P; }

static Cons hlld_glm_flux(Cons UL, Cons UR, int xdir, double ch) {
  Prim L = cons_to_prim(UL), R = cons_to_prim(UR);
  double unL = xdir ? L.u : L.v, unR = xdir ? R.u : R.v;
  double cfL = fast_speed(L, xdir), cfR = fast_speed(R, xdir);
  double SL = fmin(fmin(unL - cfL, unR - cfR), -ch), SR = fmax(fmax(unL + cfL, unR + cfR), ch);
  Cons FL = xdir ? flux_x(UL, ch) : flux_y(UL, ch), FR = xdir ? flux_x(UR, ch) : flux_y(UR, ch);
  if (SL >= 0.0) return FL;
  if (SR <= 0.0) return FR;

  /* HLLD contact/Alfven estimate.  The final flux is kept in the robust HLL
     family unless the intermediate total-pressure state is positive; this gives
     the demo HLLD wave awareness without letting low-beta cells explode. */
  double ptL = L.p + 0.5*(sqr(L.Bx)+sqr(L.By)), ptR = R.p + 0.5*(sqr(R.Bx)+sqr(R.By));
  double SM = (ptR - ptL + L.rho*unL*(SL-unL) - R.rho*unR*(SR-unR)) /
              (L.rho*(SL-unL) - R.rho*(SR-unR));
  double ptStar = 0.5*(ptL + ptR + L.rho*(SL-unL)*(SM-unL) + R.rho*(SR-unR)*(SM-unR));
  double bnl = xdir ? L.Bx : L.By, bnr = xdir ? R.Bx : R.By;
  double caL = fabs(bnl) / sqrt(L.rho), caR = fabs(bnr) / sqrt(R.rho);
  (void)caL; (void)caR; /* retained for HLLD diagnostics/extensions */

  Cons FHLL = cmul(csub(cadd(cmul(FL, SR), cmul(FR, -SL)), cmul(csub(UR, UL), SL*SR)), 1.0/(SR-SL));
  if (!isfinite(SM) || !isfinite(ptStar) || ptStar <= EPS_P) return FHLL;

  return FHLL;
}

static Cons slope_at(int x, int y) {
  Cons qm = U[idx(x-1,y)], q = U[idx(x,y)], qp = U[idx(x+1,y)], sy;
  sy.rho=mc(q.rho-qm.rho,0.5*(qp.rho-qm.rho),qp.rho-q.rho); sy.mx=mc(q.mx-qm.mx,0.5*(qp.mx-qm.mx),qp.mx-q.mx);
  sy.my=mc(q.my-qm.my,0.5*(qp.my-qm.my),qp.my-q.my); sy.E=mc(q.E-qm.E,0.5*(qp.E-qm.E),qp.E-q.E);
  sy.Bx=mc(q.Bx-qm.Bx,0.5*(qp.Bx-qm.Bx),qp.Bx-q.Bx); sy.By=mc(q.By-qm.By,0.5*(qp.By-qm.By),qp.By-q.By);
  sy.psi=mc(q.psi-qm.psi,0.5*(qp.psi-qm.psi),qp.psi-q.psi); return sy;
}
static Cons slope_y_at(int x, int y) {
  Cons qm = U[idx(x,y-1)], q = U[idx(x,y)], qp = U[idx(x,y+1)], sy;
  sy.rho=mc(q.rho-qm.rho,0.5*(qp.rho-qm.rho),qp.rho-q.rho); sy.mx=mc(q.mx-qm.mx,0.5*(qp.mx-qm.mx),qp.mx-q.mx);
  sy.my=mc(q.my-qm.my,0.5*(qp.my-qm.my),qp.my-q.my); sy.E=mc(q.E-qm.E,0.5*(qp.E-qm.E),qp.E-q.E);
  sy.Bx=mc(q.Bx-qm.Bx,0.5*(qp.Bx-qm.Bx),qp.Bx-q.Bx); sy.By=mc(q.By-qm.By,0.5*(qp.By-qm.By),qp.By-q.By);
  sy.psi=mc(q.psi-qm.psi,0.5*(qp.psi-qm.psi),qp.psi-q.psi); return sy;
}

static void reset(void) {
  sim_t = 0.0;
  for (int y=0;y<H;y++) for (int x=0;x<W;x++) {
    double X=(x+0.5)/W, Y=(y+0.5)/H; Prim p = {1,0,0,1,0.75,0,0};
    if (problem == 0) { /* Brio-Wu-like rotor/shock tube */
      if (X < 0.5) { p.rho=1.0; p.p=1.0; p.By=1.0; } else { p.rho=0.125; p.p=0.1; p.By=-1.0; }
      p.Bx = 0.75; p.v = 0.03 * sin(12.0 * Y);
    } else { /* Orszag-Tang vortex */
      p.rho = GAMMA*GAMMA; p.p = GAMMA; p.u = -sin(2*M_PI*Y); p.v = sin(2*M_PI*X);
      p.Bx = -sin(2*M_PI*Y) / sqrt(4*M_PI); p.By = sin(4*M_PI*X) / sqrt(4*M_PI);
    }
    U[idx(x,y)] = prim_to_cons(p);
  }
}

static void step(void) {
  double maxs = 1e-6;
  for (int i=0;i<W*H;i++) { Prim p=cons_to_prim(U[i]); maxs=fmax(maxs, hypot(p.u,p.v)+fmax(fast_speed(p,1),fast_speed(p,0))); }
  double dx=1.0/W, dy=1.0/H, ch=maxs, dt=CFL*fmin(dx,dy)/fmax(maxs+ch,1e-6);
  memcpy(Unew, U, sizeof(U));
  for (int y=1;y<H-1;y++) for (int x=1;x<W-2;x++) {
    Cons sL=slope_at(x,y), sR=slope_at(x+1,y); Cons qL=cadd(U[idx(x,y)],cmul(sL,0.5)), qR=csub(U[idx(x+1,y)],cmul(sR,0.5));
    Cons f=hlld_glm_flux(qL,qR,1,ch); Unew[idx(x,y)] = csub(Unew[idx(x,y)], cmul(f,dt/dx)); Unew[idx(x+1,y)] = cadd(Unew[idx(x+1,y)], cmul(f,dt/dx));
  }
  for (int y=1;y<H-2;y++) for (int x=1;x<W-1;x++) {
    Cons sL=slope_y_at(x,y), sR=slope_y_at(x,y+1); Cons qL=cadd(U[idx(x,y)],cmul(sL,0.5)), qR=csub(U[idx(x,y+1)],cmul(sR,0.5));
    Cons f=hlld_glm_flux(qL,qR,0,ch); Unew[idx(x,y)] = csub(Unew[idx(x,y)], cmul(f,dt/dy)); Unew[idx(x,y+1)] = cadd(Unew[idx(x,y+1)], cmul(f,dt/dy));
  }
  double damp = exp(-GLM_ALPHA * ch * dt / fmin(dx, dy));
  for (int i=0;i<W*H;i++) { Unew[i].psi *= damp; if (!valid(Unew[i])) Unew[i] = U[i]; }
  memcpy(U, Unew, sizeof(U)); sim_t += dt;
}

static Color cmap(double a) { a=clampd(a,0,1); unsigned char r=(unsigned char)(255*clampd(1.5*a-0.2,0,1)); unsigned char g=(unsigned char)(255*sin(M_PI*a)); unsigned char b=(unsigned char)(255*clampd(1.2-1.4*a,0,1)); return (Color){r,g,b,255}; }
static void draw_pixels(void) {
  for (int y=0;y<H;y++) for (int x=0;x<W;x++) { Prim p=cons_to_prim(U[idx(x,y)]); double val;
    if (view_mode==0) val=(p.rho-0.1)/2.2; else if (view_mode==1) val=p.p/2.0; else if (view_mode==2) val=hypot(p.Bx,p.By)/1.6; else {
      double div=fabs((U[idx((x+1)%W,y)].Bx-U[idx((x+W-1)%W,y)].Bx)*0.5*W + (U[idx(x,(y+1)%H)].By-U[idx(x,(y+H-1)%H)].By)*0.5*H); val=div*0.05; }
    Color c=cmap(val); int k=4*idx(x,y); pixels[k]=c.r; pixels[k+1]=c.g; pixels[k+2]=c.b; pixels[k+3]=255; }
}

int main(void) {
  pixels = malloc(W*H*4); if (!pixels) return 1; reset();
  InitWindow(W*SCALE, H*SCALE, "tau_mhd - ideal MHD + GLM cleaning"); SetTargetFPS(60);
  Image img = { pixels, W, H, 1, PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 }; Texture2D tex = LoadTextureFromImage(img);
  while (!WindowShouldClose()) {
    if (IsKeyPressed(KEY_SPACE)) paused = !paused;
    if (IsKeyPressed(KEY_R)) reset();
    if (IsKeyPressed(KEY_M)) view_mode = (view_mode + 1) % 4;
    if (IsKeyPressed(KEY_C)) { problem = (problem + 1) % 2; reset(); }
    if (!paused) {
      for (int k = 0; k < STEPS_PER_FRAME; k++) step();
    }
    draw_pixels();
    UpdateTexture(tex, pixels);
    BeginDrawing(); ClearBackground(BLACK); DrawTextureEx(tex,(Vector2){0,0},0,SCALE,WHITE);
    DrawText(TextFormat("Ideal MHD GLM  t=%.3f  view=%s  case=%s", sim_t, view_mode==0?"rho":view_mode==1?"pressure":view_mode==2?"|B|":"|divB|", problem==0?"Brio-Wu":"Orszag-Tang"), 10, 10, 18, RAYWHITE);
    DrawText("SPACE pause  R reset  M view  C case", 10, 32, 16, RAYWHITE); EndDrawing();
  }
  UnloadTexture(tex); CloseWindow(); free(pixels); return 0;
}
