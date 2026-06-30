# Makefile for fluid-sims
#
# Build recipes are taken from the comment header of each source file.
# Targets are grouped so you can build just the CPU (gcc) demos, just the
# CUDA (nvcc) demos, or everything at once:
#
#   make            # build everything
#   make cpu        # gcc / raylib / ncurses demos
#   make cuda       # nvcc CUDA demos
#   make sim        # build a single target by name
#   make clean      # remove all built binaries

CC      = gcc
NVCC    = nvcc

# Host compiler nvcc hands C++ off to (overridable for CI).
CCBIN   ?= g++-10

# ---------------------------------------------------------------------------
# Binaries
# ---------------------------------------------------------------------------
CPU_BINS  := number_fluid2d number_fluid3d sim tau_hypersonic \
             tau_hypersonic_simd tau_mhd

CUDA_BINS := jsc jsc3d tau_burgers tgs tau3d tau_2d_hypersonic_cuda \
             tau_hypersonic_cuda_tests tau_sw tau_sph

# th3cs also needs 4splat.c, so it is kept out of the default group and built
# explicitly with `make th3cs`.

.PHONY: all cpu cuda test clean
all: cpu cuda
cpu: $(CPU_BINS)
cuda: $(CUDA_BINS)

# Build and run the hypersonic CUDA regression + unit test suite.
# Requires a CUDA-capable GPU at runtime; writes a fresh baseline and then
# verifies the same run against it (round-trip self-check).
BASELINE ?= tau_hypersonic_cuda_baseline.txt
TEST_STEPS ?= 24
test: tau_hypersonic_cuda_tests
	./tau_hypersonic_cuda_tests --steps $(TEST_STEPS) --write-baseline  --baseline $(BASELINE)
	./tau_hypersonic_cuda_tests --steps $(TEST_STEPS) --verify-baseline --baseline $(BASELINE)

# ---------------------------------------------------------------------------
# CPU targets (gcc)
# ---------------------------------------------------------------------------
number_fluid2d: number_fluid2d.c
	$(CC) -Ofast -march=native -flto -std=c99 $< -lraylib -lm -ldl -lpthread -lGL -lrt -lX11 -o $@

number_fluid3d: number_fluid3d.c
	$(CC) -Ofast -march=native -flto -std=c99 $< -lraylib -lm -ldl -lpthread -lGL -lrt -lX11 -o $@

sim: sim.c
	$(CC) -O3 -march=native -ffast-math -funroll-loops $< -lncursesw -lm -o $@

tau_hypersonic: tau_hypersonic.c
	$(CC) -O3 $< -lraylib -lm -o $@

tau_hypersonic_simd: tau_hypersonic_simd.c
	$(CC) -O3 -mavx2 -mfma $< -lraylib -lm -o $@

tau_mhd: tau_mhd.c
	$(CC) -O3 $< -lraylib -lm -o $@

# ---------------------------------------------------------------------------
# CUDA targets (nvcc)
# ---------------------------------------------------------------------------
jsc: js_cuda.cu
	$(NVCC) -O3 -arch=sm_86 $< -o $@ -lncursesw

jsc3d: js_cuda3d.cu
	$(NVCC) -std=c++14 -O3 -use_fast_math -arch=sm_86 $< -lncursesw -o $@

tau_burgers: tau_burgers.cu
	$(NVCC) -std=c++17 -O3 -use_fast_math -arch=sm_86 -lineinfo -o $@ $< -lncursesw

tgs: tau_gray_scott.cu
	$(NVCC) -std=c++17 -ccbin $(CCBIN) -O3 -use_fast_math -arch=sm_86 -lineinfo $< -lncursesw -o $@

tau3d: tau_hypersonic_3d_cuda.cu
	$(NVCC) -O3 -std=c++17 $< -o $@ -lineinfo

tau_2d_hypersonic_cuda: tau_hypersonic_cuda.cu
	$(NVCC) -O3 -o $@ $< -std=c++17 -lraylib

tau_hypersonic_cuda_tests: tau_hypersonic_cuda_tests.cu
	$(NVCC) -O2 -std=c++17 -o $@ $<

tau_sw: tau_shallow_water.cu
	$(NVCC) -std=c++17 -O3 -use_fast_math -arch=sm_86 -lineinfo -o $@ $< -lncursesw

tau_sph: tau_sph.cu
	$(NVCC) -std=c++17 -O3 -use_fast_math -arch=sm_86 -lineinfo $< -o $@ -lncursesw

th3cs: th3cs.cu 4splat.c
	$(NVCC) -O3 -std=c++17 $< 4splat.c -o $@ -lineinfo

# ---------------------------------------------------------------------------
clean:
	$(RM) $(CPU_BINS) $(CUDA_BINS) th3cs $(BASELINE)
