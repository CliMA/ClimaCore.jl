# cuTile.jl vs ClimaCore spectral Gradient — 1v1 GPU benchmark

Head-to-head speed test of ClimaCore's production spectral-element `Gradient`
(the generic `apply_operator` fused-slab CUDA path) against a
[cuTile.jl](https://github.com/JuliaGPU/cuTile.jl) implementation of the same
strong-form contraction, plus a hand-written CUDA.jl reference and a
traffic-matched copy kernel as the bandwidth floor. Correctness of every
contender is gated against the ClimaCore CPU result before any timing.

The cuTile kernel ("KronGEMM") computes both covariant components in one
batched GEMM per element: the VIJFH parents reshape contiguously to
`(Nv, Nq², Nh)` / `(Nv, 2Nq², Nh)` and the gradient becomes
`O[:, :, h] = F[:, :, h] * Wt` with a precomputed `Nq² × 2Nq²` Kronecker
weight built from the GLL differentiation matrix.

## Requirements

- Julia ≥ 1.11 (`julia +1.11` via juliaup)
- NVIDIA GPU, compute capability ≥ 8.0 (A100/H100)
- Driver supporting CUDA 13 (≥ 580); otherwise try
  `export JULIA_CUDA_USE_COMPAT=true`
- No CUDA toolkit install needed — CUDA.jl downloads artifacts

The benchmark has its own environment so ClimaCore's compat floors
(Julia 1.10, CUDA 5.5+) are untouched.

## Setup (once, from the ClimaCore.jl repo root)

```bash
export CLIMACOMMS_DEVICE=CUDA
julia +1.11 --project=benchmarks/cutile -e '
    using Pkg; Pkg.develop(path="."); Pkg.instantiate(); Pkg.precompile()'
```

## Run

No-GPU sanity check (validates the KronGEMM weight ordering and VIJFH layout
assumptions against ClimaCore's Gradient on CPU; runs on any machine):

```bash
julia +1.11 --project=benchmarks/cutile benchmarks/cutile/test_weight_cpu.jl
```

Tiny-config GPU correctness check (cheap, do this before full-size runs):

```bash
julia +1.11 --project=benchmarks/cutile benchmarks/cutile/benchmark_gradient.jl \
    --float-type Float64 --helem 2 --zelem 4 --n-reads-writes 3
```

Full sweep (Float32 and Float64, `h_elem` 30/60/90, copy-floor traffic
counts 3/6/9). One process, then a table of cuTile speedup against ClimaCore:

```bash
julia +1.11 --project=benchmarks/cutile benchmarks/cutile/benchmark_gradient.jl
```

The gradient kernels always move one read and two writes. Counts 6 and 9 are
copy floors only: one read and 5 or 8 writes of the same value, so the
bandwidth reference can be store-heavy. Narrow a single axis with a comma
list, for example `--float-type Float64 --helem 30,60`.

Other options: `--zelem 63 --nq 4 --tv 64`
(`--tv` is the cuTile tile size along the vertical; power of two),
`--contenders climacore,cutile,cuda_ref,copy_floor` (default `all`;
drop `cutile` to run on nodes without CUDA-13 drivers).

Fused production expressions (pressure gradient, scalar hyperdiffusion).
Compares the ClimaAtmos broadcast with a split that materializes each
strong gradient, and with that split when the gradient is the cuTile
KronGEMM. The summary speedup is fused time over cuTile time. Default is
both precisions at `h_elem = 30`:

```bash
julia +1.11 --project=benchmarks/cutile benchmarks/cutile/benchmark_fused.jl
julia +1.11 --project=benchmarks/cutile benchmarks/cutile/benchmark_fused.jl \
    --float-type Float64 --helem 30,60,90
```

## Slurm (Caltech cluster)

```bash
#!/bin/bash
#SBATCH --job-name=cutile-grad-bench
#SBATCH --time=01:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
export CLIMACOMMS_DEVICE=CUDA
export JULIA_CUDA_USE_COMPAT=true   # forward-compat libcuda if node driver < 580
cd $SLURM_SUBMIT_DIR                # submit from the ClimaCore.jl root
julia +1.11 --project=benchmarks/cutile benchmarks/cutile/benchmark_gradient.jl
```

Interactive: `srun --partition=gpu --gres=gpu:a100:1 --mem=32G --time=1:00:00 --pty bash`,
then the same commands.

## Reading the results

- The headline metric is achieved bandwidth (GB/s and % of device peak) in the
  final table — this op is memory-bound (~0.7 flop/byte); the ideal traffic is
  1 read + 2 writes per point (130.6 MiB at the canonical Float64 size, so the
  A100-40GB roofline is ~95–110 µs).
- Nothing should beat the 1R+2W `copy_floor`; if it does, the script warns
  (timing artifact / clock boost). The 1R+5W and 1R+8W rows are heavier
  copies, not a floor for the gradient.
- The last table is cuTile speedup against ClimaCore,
  `t_climacore / t_cutile`, at each float type and `h_elem`. A value above 1
  means cuTile is faster.
- The two timing methods (BenchmarkTools min, CUDA-event loop min) should
  agree within ~10%; the loop variant amortizes launch overhead, which can
  differ between cuTile launches and ClimaCore's `auto_launch!`.
- Do not compare against `test/Operators/spectralelement/benchmark_times.jl`
  (that config is much smaller and latency-bound).

## Known API caveats (cuTile 1.0)

- The store of the padded v-tile relies on cuTile masking out-of-bounds tile
  rows; the correctness gate catches it if that assumption fails (workaround:
  pick `--tv` dividing `--zelem`, e.g. `--tv 63` is invalid but `--zelem 64`
  works, or pad the arrays).
- Float32 tile `muladd` must not silently demote to TF32. The gate allows
  ~2e-4 of the peak derivative (the Float32 CPU-vs-GPU cancellation floor)
  and fails for a TF32-scale error, about 1e-3 of the peak.
