Video: CUDA Gray-Scott reaction-diffusion simulator (build `tau_gray_scott`)

https://github.com/user-attachments/assets/e8f9c28d-60ca-4f95-825d-3046b817139b

Video: A CUDA-based 3D fluid dynamics simulator

https://github.com/user-attachments/assets/b6ebef66-554e-477c-8204-cc5b7d855403

https://github.com/user-attachments/assets/c80b3505-f605-4a49-815f-2f64e1824464

Video: 2-D CUDA-accelerated SPH fluid simulator

https://github.com/user-attachments/assets/a18f274d-f7ff-45e5-adf1-a8898b76c65f

Video: A 2-D CUDA solver for viscous Burgers’ flow

https://github.com/user-attachments/assets/19409252-181f-4162-bbd9-970239427b1f

Video: 2-D shallow-water simulation

https://github.com/user-attachments/assets/2d664af1-662b-49a4-9d7e-dda5710ed6e2

Video: A CUDA-based 2-D fluid dynamics simulator

https://github.com/user-attachments/assets/b7bbda96-7fec-4abb-9a80-bc461a0edaa6

## Hypersonic CUDA regression test bed

`tau_hypersonic_cuda_tests.cu` now supports a small deterministic regression harness for `tau_hypersonic_cuda.cu`:

- Runs core solver kernels for a configurable number of steps (`--steps N`).
- Computes a physics snapshot (mass/momentum/energy totals, positivity floors, max Mach, weighted checksums).
- Can **record** a baseline (`--write-baseline`) or **verify** against one (`--verify-baseline`, default).

Example workflow on a CUDA machine:

```bash
nvcc -O2 -std=c++17 -o tau_hypersonic_cuda_tests tau_hypersonic_cuda_tests.cu
./tau_hypersonic_cuda_tests --steps 24 --write-baseline --baseline tau_hypersonic_cuda_baseline.txt
./tau_hypersonic_cuda_tests --steps 24 --verify-baseline --baseline tau_hypersonic_cuda_baseline.txt
```
