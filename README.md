# fluid-sims

[![CI](https://github.com/seanwevans/fluid-sims/actions/workflows/ci.yml/badge.svg)](https://github.com/seanwevans/fluid-sims/actions/workflows/ci.yml)

**A numerical-computing laboratory for fluid dynamics, continuum mechanics, reaction–diffusion, and GPU simulation.**

This repository is a collection of mostly standalone C and CUDA experiments.

The implementations range from finite-volume compressible flow to particle methods, particle/grid hybrids, lattice methods, shallow water, magnetohydrodynamics, and reaction–diffusion.

This is intentionally a laboratory rather than a single CFD framework. Different files test different numerical representations, data layouts, integration strategies, hardware targets, and visualization approaches.

---

## Highlights

The repository currently includes:

* 2-D and 3-D hypersonic compressible-flow solvers
* an axisymmetric Mach 25 re-entry capsule written without subtraction or zero
* CPU, SIMD, and CUDA implementations of related flow problems
* Smoothed Particle Hydrodynamics (SPH)
* hybrid FLIP/APIC incompressible flow
* Material Point Method (MPM) elastoplastic simulation
* D2Q9 Lattice Boltzmann flow
* shallow-water finite-volume simulation
* viscous Burgers flow
* ideal magnetohydrodynamics with divergence cleaning
* Gray–Scott reaction–diffusion
* CUDA grid-fluid experiments
* interactive and headless execution paths
* regression testing for the CUDA hypersonic and re-entry solvers

The goal is to compare representations.

---

# Gallery

## 2-D CUDA hypersonic flow

Screen capture of `tau_hypersonic_cuda.cu` running in speed view.

https://github.com/user-attachments/assets/fea24b88-89ef-4f4a-b7ff-2ba0645da416

## Gray–Scott reaction–diffusion

CUDA Gray–Scott simulation.

https://github.com/user-attachments/assets/e8f9c28d-f60ca-4f95-825d-3046b817139b

## 3-D CUDA fluid simulation

https://github.com/user-attachments/assets/b6ebef66-554e-477c-8204-cc5b7d855403

https://github.com/user-attachments/assets/c80b3505-f605-4a49-815f-2f64e1824464

## Smoothed Particle Hydrodynamics

2-D CUDA SPH.

https://github.com/user-attachments/assets/a18f274d-f7ff-45e5-adf1-a8898b76c65f

## Viscous Burgers flow

https://github.com/user-attachments/assets/19409252-181f-4162-bbd9-970239427b1f

## Shallow water

https://github.com/user-attachments/assets/2d664af1-662b-49a4-9d7e-dda5710ed6e2

## CUDA grid fluid

https://github.com/user-attachments/assets/b7bbda96-7fec-4abb-9a80-bc461a0edaa6

---

# Solver map

| Source                                                   | Model / method                                                       | Backend        | Display |
| -------------------------------------------------------- | -------------------------------------------------------------------- | -------------- | ------- |
| [`tau_hypersonic.c`](tau_hypersonic.c)                   | 2-D compressible Euler, MUSCL reconstruction, MC limiting, HLLC flux | C / CPU        | raylib  |
| [`tau_hypersonic_simd.c`](tau_hypersonic_simd.c)         | SIMD-oriented hypersonic-flow implementation                         | C / AVX2 + FMA | raylib  |
| [`tau_hypersonic_cuda.cu`](tau_hypersonic_cuda.cu)       | large-grid 2-D hypersonic compressible flow                          | CUDA           | raylib  |
| [`tau_reentry_cuda.cu`](tau_reentry_cuda.cu)             | axisymmetric Mach 25 re-entry capsule, subtraction-free arithmetic   | CUDA           | raylib  |
| [`tau_hypersonic_3d_cuda.cu`](tau_hypersonic_3d_cuda.cu) | 3-D hypersonic flow                                                  | CUDA           | raylib  |
| [`tau_sph.cu`](tau_sph.cu)                               | 2-D Smoothed Particle Hydrodynamics                                  | CUDA           | ncurses |
| [`tau_flip_apic.cu`](tau_flip_apic.cu)                   | hybrid FLIP/APIC incompressible flow                                 | CUDA           | ncurses |
| [`tau_mpm.cu`](tau_mpm.cu)                               | elastoplastic Material Point Method                                  | CUDA           | ncurses |
| [`tau_lbm.cu`](tau_lbm.cu)                               | D2Q9 BGK Lattice Boltzmann                                           | CUDA           | ncurses |
| [`tau_shallow_water.cu`](tau_shallow_water.cu)           | conservative shallow-water finite volume with HLL flux               | CUDA           | ncurses |
| [`tau_burgers.cu`](tau_burgers.cu)                       | 2-D viscous Burgers flow                                             | CUDA           | ncurses |
| [`tau_gray_scott.cu`](tau_gray_scott.cu)                 | Gray–Scott reaction–diffusion                                        | CUDA           | ncurses |
| [`tau_mhd.c`](tau_mhd.c)                                 | ideal MHD with GLM divergence cleaning                               | C / CPU        | raylib  |
| [`js_cuda.cu`](js_cuda.cu)                               | grid-based advection, diffusion, and projection experiment           | CUDA           | ncurses |
| [`js_cuda3d.cu`](js_cuda3d.cu)                           | 3-D CUDA grid-fluid experiment                                       | CUDA           | ncurses |


---

# Numerical experiments

## Hypersonic compressible flow

The hypersonic family is the most developed line of experiments in the repository.

The CPU solver stores the conservative state U = [𝜌, 𝜌u, 𝜌v, E] and evolves the compressible Euler equations using finite-volume fluxes.

Primitive variables are reconstructed from the conservative state, p = (𝛾-1)(E-½𝜌⁢(u²+v²)) with local sound speed a = √(𝛾p/𝜌⁢)

The CPU implementation uses:

* MUSCL-style reconstruction
* monotonized-central limiting
* HLLC interface fluxes
* CFL-controlled stepping
* solid obstacle geometry
* multiple physical diagnostic views

The CUDA branch explores the same general problem at much larger grid sizes and exposes views for:

* density
* pressure
* speed
* schlieren-like density gradient
* vorticity
* Mach number
* pressure/density structure

The point of keeping CPU, SIMD, 2-D CUDA, and 3-D CUDA versions beside one another is that the governing physics can remain recognizable while the computational representation changes substantially.

---

## Mach 25 re-entry, in an arithmetic without subtraction

[`tau_reentry_cuda.cu`](tau_reentry_cuda.cu) is the most explicit representation experiment in the repository. It sits beside the hypersonic family rather than replacing any of it, and it changes two things at once: the problem it targets and the arithmetic it is written in.

### The physics

It solves the **axisymmetric** compressible Euler equations around an **Apollo-class re-entry capsule at Mach 25**, blunt heat shield forward:

* spherical-segment heat shield with a radius of curvature of 1.2 base diameters, a filleted shoulder, a 33° conical afterbody, and a truncated aft deck
* a geometric source term for the axisymmetric equations, with a symmetry condition on the axis; the planar equations give the wrong shock stand-off for a body of revolution
* an effective ratio of specific heats of 1.2, parameterized as a polytropic index; air behind a Mach 25 normal shock dissociates, and the equilibrium effective gamma is what sets the stand-off distance
* MUSCL reconstruction with monotonized-central limiting, dropping to first order beside the body and wherever a reconstructed face would leave the physical state space
* HLLC interface fluxes with Quirk's shock fix: cells beside a compressive pressure jump are flagged and every face touching one falls back to HLLE. Flagging per cell rather than per face is the point, since the faces lying *along* a grid-aligned bow shock see almost no normal jump of their own; without the transverse half of the cure the bow shock breathes and eventually carbuncles
* an exact slip wall: only the face-normal velocity component is reflected in the solid ghost cells
* SSP-RK2 in time, and no artificial hyperviscosity anywhere

Views: log density, log pressure, speed, schlieren, Mach number, temperature, and vorticity, drawn as a half plane mirrored about the axis.

On a 128 x 64 host run of the same cell bodies the bow shock stands off 0.105 nose radii and the wall pressure on the axis reaches 96% of the Rayleigh pitot value for this gas and Mach number.

### The arithmetic

Every floating-point quantity in the solver is a strictly positive `double` `t` standing for the real value `TAU_SCALE * log(t)`. The multiplicative group of the positive reals then carries the additive structure of the reals:

| in the value | in the code |
| ------------ | ----------- |
| `a + b`      | `a * b`     |
| `a - b`      | `a / b`     |
| `0`          | `1.0`       |
| `-a`         | `1.0 / a`   |
| `a / 2`      | `sqrt(a)`   |
| `a * 2`      | `a * a`     |
| `a * c`      | `pow(a, c)` |
| `a < b`      | `a < b`     |

Products and quotients of two values ride the same isomorphism through `log` and `exp`; comparison, `min`, `max`, magnitude, and negation are exact and free.

The consequence is that **the source contains no subtraction operator and no literal zero**. There is no binary `-`, no unary `-`, no `-=`, no `--`, and no `0`: small constants are written as reciprocals such as `1.0 / 1e8`, hyphens appear only in comments and in command-line option strings, and the one integer zero the raylib draw origin needs is derived as `1 / 2` rather than written down.

Consequences worth knowing: the encoding has *uniform absolute* precision of about `TAU_SCALE * 2^-53`, roughly `1e-13` in value units, across the whole dynamic range of a re-entry flow, and its representable range is about `±7.1e5`. Global sums must therefore be taken as means, which is what the diagnostics do.

Its kernels are thin wrappers around host-callable cell bodies, and `NX`/`NY` can be overridden at build time, so the whole solver can also be driven on the CPU at a coarse grid without a GPU.

---

## Smoothed Particle Hydrodynamics

[`tau_sph.cu`](tau_sph.cu) implements a 2-D particle fluid using SPH.

Particles carry position and velocity directly rather than storing the fluid state on a fixed Eulerian grid.

The solver includes:

* cubic smoothing kernels
* kernel gradients
* density estimation
* pressure forces
* Tait-style equation-of-state behavior
* Monaghan artificial viscosity
* gravity
* optional XSPH smoothing
* spatial binning for local neighbor searches
* configurable particle counts
* headless execution

Example:

```bash
./tau_sph \
    --n 32768 \
    --CFL 0.25 \
    --dTau 1e-3 \
    --visc 0.1 \
    --visc_substeps 2 \
    --stride 2 \
    --fps 60
```

A larger headless run:

```bash
./tau_sph \
    --n 65536 \
    --headless \
    --stride 10 \
    --visc_substeps 3
```

SPH provides a useful contrast with the grid-based solvers because the discretization follows the material itself.

---

## FLIP / APIC

[`tau_flip_apic.cu`](tau_flip_apic.cu) explores the opposite compromise: keep particles for material motion, but repeatedly transfer information to a grid for the pressure solve.

The implementation combines:

* particle-to-grid transfer
* grid velocity fields
* pressure projection
* Jacobi iterations
* grid-to-particle transfer
* FLIP velocity updates
* APIC affine velocity information

The blend can be controlled at runtime:

```bash
./tau_flip_apic \
    --particles 65536 \
    --grid 128 \
    --apic 0.85 \
    --flip 0.97
```

or executed without rendering:

```bash
./tau_flip_apic --headless --steps 600 --stride 20
```

This makes FLIP/APIC particularly interesting as a representation experiment: the state alternates between Lagrangian particles and an Eulerian grid every step.

---

## Material Point Method

[`tau_mpm.cu`](tau_mpm.cu) uses another particle/grid decomposition, but for deformable elastoplastic material rather than an ordinary incompressible liquid.

Particles carry state including deformation information while a background grid is used to evaluate forces and momentum transfer.

Available material presets include:

* `mud`
* `snow`
* `sand`

Example:

```bash
./tau_mpm \
    --n 32768 \
    --grid 96x96 \
    --dt 8e-5 \
    --steps 20000
```

Headless snow simulation:

```bash
./tau_mpm \
    --n 65536 \
    --headless \
    --steps 1200 \
    --material snow
```

The implementation exposes deformation gradients, hardening parameters, elastic coefficients, and compression/stretch limits directly rather than hiding them behind a general simulation framework.

---

## Lattice Boltzmann

[`tau_lbm.cu`](tau_lbm.cu) approaches fluid simulation from a mesoscopic direction.

Each grid cell stores nine D2Q9 distribution populations. A time step consists primarily of:

1. recovering macroscopic density and velocity,
2. relaxing distributions toward local equilibrium,
3. streaming populations to neighboring lattice cells.

The solver uses a BGK collision model.

Solid boundaries use on-link bounce-back, making obstacles easy to introduce without solving a pressure Poisson equation.

The CUDA implementation includes an optional cylindrical obstacle and adjustable driving force.

Controls:

```text
q       quit
o       toggle obstacle
+ / -   adjust drive
```

This makes LBM a particularly useful comparison with the finite-volume and pressure-projection solvers: similar visible fluid behavior emerges from a radically different computational state.

---

## Shallow water

[`tau_shallow_water.cu`](tau_shallow_water.cu) implements the 2-D shallow-water equations in conservative finite-volume form.

The stored water depth uses 𝜎 = ln h so that h = e^𝜎 > 0.

The solver includes:

* HLL interface fluxes
* positive-depth preservation
* CFL-limited stepping
* optional momentum viscosity
* Coriolis forcing
* initial surface perturbations
* initial rotational flow
* headless execution

It also experiments with a logarithmic time coordinate t = t₀ e^𝜏,

while still clamping the effective physical timestep to the local CFL condition.

Example:

```bash
./tau_sw \
    --nx 256 \
    --ny 256 \
    --dx 1000 \
    --dy 1000 \
    --dtau 1e-3 \
    --f0 1e-4 \
    --nu 50 \
    --stride 2 \
    --fps 60
```

Benchmark/headless mode:

```bash
./tau_sw --headless --steps 2000 --stride 4
```

---

## Magnetohydrodynamics

[`tau_mhd.c`](tau_mhd.c) extends the conservative finite-volume experiments to ideal magnetohydrodynamics.

The state contains fluid and magnetic variables, 𝑈 =(𝜌,𝜌⁢𝑢,𝜌⁢𝑣,𝐸,𝐵𝑥,𝐵𝑦,𝜓), where 𝜓 is the Generalized Lagrange Multiplier field used for divergence cleaning.

The implementation includes:

* conservative finite-volume evolution
* MUSCL reconstruction
* fast magnetosonic wave-speed estimates
* HLLD-oriented intermediate-state estimates
* robust HLL fallback
* hyperbolic/parabolic GLM divergence cleaning
* several initial-condition families

Controls:

```text
SPACE   pause
R       reset
M       cycle view
C       cycle initial condition
```

This solver is deliberately compact enough that the numerical method can be read directly from a single C source file.

---

## Gray–Scott reaction–diffusion

[`tau_gray_scott.cu`](tau_gray_scott.cu) moves away from fluid dynamics entirely and solves the Gray–Scott two-species reaction–diffusion system.

The state consists of two interacting fields (U) and (V), controlled by:

* diffusion coefficients (D_u) and (D_v)
* feed rate (F)
* kill rate (k)

Runtime parameters can be varied directly:

```bash
./tgs --nx 200 --ny 200
```

or:

```bash
./tgs \
    --Du 0.2 \
    --Dv 0.1 \
    --F 0.03 \
    --k 0.06
```

The same massively parallel stencil structure that appears in fluid solvers also produces self-organizing chemical patterns here.

That computational similarity is one of the reasons the model belongs in this repository.

---

# Building

The repository is primarily developed for Linux and uses:

* GCC / G++
* NVIDIA CUDA Toolkit / `nvcc`
* raylib
* ncursesw
* GNU Make

The CI configuration currently builds against raylib 5.5.

## Build everything wired into the Makefile

```bash
make
```

## CPU programs only

```bash
make cpu
```

This builds:

```text
number_fluid2d
number_fluid3d
sim
tau_hypersonic
tau_hypersonic_simd
tau_mhd
```

## CUDA programs only

```bash
make cuda
```

This currently builds:

```text
jsc
jsc3d
tau_burgers
tgs
tau3d
tau_2d_hypersonic_cuda
tau_hypersonic_cuda_tests
tau_reentry
tau_reentry_cuda_tests
tau_sw
tau_sph
```

The Makefile currently targets `sm_86` for several CUDA programs. Adjust the architecture flag if appropriate for your toolchain or target GPU.

The host compiler passed through `nvcc` defaults to:

```make
CCBIN ?= g++-10
```

and can be overridden:

```bash
make cuda CCBIN=g++
```

---

# Standalone CUDA experiments

Some newer experiments are present in the repository but are not currently part of the default `make cuda` target.

## Lattice Boltzmann

```bash
nvcc \
    -std=c++17 \
    -O3 \
    -use_fast_math \
    -arch=sm_86 \
    -lineinfo \
    tau_lbm.cu \
    -o tau_lbm \
    -lncursesw
```

## FLIP/APIC

```bash
nvcc \
    -std=c++17 \
    -O3 \
    -use_fast_math \
    -arch=sm_86 \
    tau_flip_apic.cu \
    -o tau_flip_apic \
    -lncursesw
```

## MPM

```bash
nvcc \
    -std=c++17 \
    -O3 \
    -use_fast_math \
    -arch=sm_86 \
    -lineinfo \
    tau_mpm.cu \
    -o tau_mpm \
    -lncursesw
```

---

# Testing

The CUDA hypersonic and re-entry implementations both have unit/regression machinery and a reproducible snapshot format.

Build the test executables with:

```bash
make tau_hypersonic_cuda_tests
make tau_reentry_cuda_tests
```

Run the standard regression round trip:

```bash
make test
```

For each solver the Makefile performs two runs:

1. generate a fresh baseline,
2. rerun the simulation and verify against that baseline.

`tau_reentry_cuda_tests` adds three layers:

* **host tests**, which need no GPU: the tau algebra checked against ordinary arithmetic as an independent oracle, thermodynamic round trips, flux consistency, Rankine-Hugoniot jump conditions at Mach 25, the slip-wall flux, the shock flag in both directions, the capsule geometry, and a 1-D driver that runs the solver's own reconstruction, Riemann solver, and multiplicative update over a shock tube and a standing Mach 25 shock;
* **device tests**, which check the carving, boundary, and stepping kernels, including that a uniform free stream is an exact fixed point of the axisymmetric update;
* the **regression baseline** round trip.

When no CUDA device is present its host tests still run and the rest are skipped. Passing `--host-2d N` additionally drives the full 2-D solver on the CPU for `N` steps, which is practical when the binary is built with a coarse grid, for example `-DNX=192 -DNY=96`.

The regression snapshot records quantities including:

```text
fluid cell count
sum(rho)
sum(mx)
sum(my)
sum(E)
minimum density
minimum pressure
maximum Mach number
weighted state checksums
```

The default regression length is 24 steps and can be changed:

```bash
make test TEST_STEPS=100
```

A custom baseline path can also be supplied:

```bash
make test BASELINE=my_baseline.txt
```

---

# Continuous integration

GitHub Actions builds both sides of the repository.

The CPU job installs the required graphics/system dependencies and runs:

```bash
make cpu
```

The CUDA job installs the CUDA compiler/runtime and runs:

```bash
make cuda CCBIN=g++
```

CUDA tests are compiled on ordinary hosted runners. They execute only when the runner actually exposes a CUDA-capable GPU.

This distinction is intentional: compilation should remain testable independently of GPU availability.

---

# Headless execution

Several CUDA experiments support a `--headless` mode.

This is useful for:

* performance measurements
* numerical experiments
* automated runs
* parameter sweeps
* avoiding terminal rendering overhead

Examples:

```bash
./tau_sph --n 65536 --headless
```

```bash
./tau_sw --headless --steps 2000
```

```bash
./tau_flip_apic --headless --steps 600
```

```bash
./tau_mpm --headless --steps 1200 --material snow
```

```bash
./tgs --headless --steps 10000
```

For performance work, prefer headless execution
