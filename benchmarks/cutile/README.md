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
julia +1.11 --project=benchmarks/cutile benchmarks/cutile/test_fused_cpu.jl
```

The second script checks the arithmetic of the single-kernel operators
(`fused_kernels.jl`, see below) against ClimaCore's `wdiv(grad(χ))`,
weighted/accumulated Laplacians, the enthalpy prologue and the pressure
gradient, on a shallow sphere, a deep sphere and terrain-following
coordinates.

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
strong gradient, with that split when the gradient is the cuTile KronGEMM,
and with the whole expression as one cuTile kernel (`1kernel`, see
"Single-kernel operators" below). The summary speedups are fused time over
cuTile time for both cuTile variants. Default is both precisions at
`h_elem = 30`:

```bash
julia --project=benchmarks/cutile benchmarks/cutile/benchmark_fused.jl
julia --project=benchmarks/cutile benchmarks/cutile/benchmark_fused.jl \
    --float-type Float64 --helem 30,60,90
```

`examples/hybrid/sphere/baroclinic_wave_rhoe_cutile.jl` runs the standard
baroclinic wave with two tendencies switchable between the ClimaCore
broadcast and a single cuTile kernel: the pressure-gradient force
(`PGRAD`: `Yₜ.c.uₕ -= gradₕ(p)/ρ + gradₕ(K + Φ)`) and the two scalar
Laplacian passes of the energy hyperdiffusion (`HYPERDIFF`:
`χ = ∇²((ρe + p)/ρ)`, then `ρeₜ -= κ₄ ∇·(ρ ∇χ)`). `HYPERDIFF` defaults to
`PGRAD`, so `PGRAD=fused` is the pure baseline; everything else (momentum
hyperdiffusion, vertical operators, DSS) is ClimaCore in every
configuration, so the walltimes printed by `driver.jl` are directly
comparable. Each kernel configuration is checked against ClimaCore on the
GPU when the cache is built.

```bash
export CLIMACOMMS_DEVICE=CUDA
export TEST_NAME=sphere/baroclinic_wave_rhoe_cutile
PGRAD=fused  julia +1.11 --project=benchmarks/cutile examples/hybrid/driver.jl
PGRAD=cutile julia +1.11 --project=benchmarks/cutile examples/hybrid/driver.jl
PGRAD=cutile HYPERDIFF=fused julia +1.11 --project=benchmarks/cutile examples/hybrid/driver.jl
```

The pressure gradient alone is far below a percent of the step, so a
`PGRAD`-only A/B is dominated by compilation and node noise (a first pair of
runs at the default configuration gave 166 s fused vs 163 s cuTile). Profile
a step (`CUDA.@profile`) to see the share of the swapped kernels before
reading a walltime difference as a kernel speedup.

Defaults: `H_ELEM=30`, `Z_ELEM=63`, Float32, `npoly = 3` (fixed: cuTile tile
extents must be powers of two, so Nq = 4, matching the microbenchmarks above),
six simulated hours (`T_END=21600`), and `dt`/`κ₄` scaled from the `h_elem = 4`
case (override with `DT`/`KAPPA_4` if the defaults misbehave at a new
resolution). Compilation is paid equally by both runs; keep `T_END` large
enough that it amortizes, or compare a pair of restarts. The final `ρe` norms
and max meridional wind are printed for cross-checking the two runs.

## Single-kernel operators (`fused_kernels*.jl`)

The split contenders showed where cuTile pays off and where it does not: a
faster gradient GEMM wins inside the pressure gradient (1.4–2.0×) but loses
inside the Laplacian (0.6× at Float64), because the split has to
materialize the gradient between the two contractions. `fused_kernels_cutile.jl`
therefore runs each whole expression as one kernel:

- `scalar_laplacian_kernel!`: `OUT = [OUT +] scale · wdiv([ρ] grad χ)` with
  `χ = X` or `(X + p)/ρ`. Per (v-tile, element): one `(tv, Nq²)` slab load,
  two GEMMs against `kron(I, D)'` and `kron(D, I)'` for the covariant
  gradient, the metric conversion `uⁱ = gⁱʲ gⱼ` and the `WJ` weighting as
  pointwise tile ops, two GEMMs against `-kron(I, D)` and `-kron(D, I)` for
  the weak divergence, divide by `WJ`, optional accumulate, one store.
  Nothing touches memory in between.
- `pressure_gradient_kernel!`: `uₜ = [uₜ +] scale · (grad(p)/ρ + grad(K + Φ))`
  with `K + Φ` as prologue and the momentum update as epilogue, writing the
  two covariant components of the tendency in place.

Arrays enter as `(Nv, Nq², Nh)` strided views of the VIJFH parents
(`tile3`), including components of `FieldVector` blocks, so there are no
scratch fields or copies. Metrics are read once per level, or once per
element when they are level-uniform (`metrics_are_level_uniform`: shallow
sphere, with or without linear terrain-following coordinates), which drops
four field reads per Laplacian. The `v` extent is zero-padded to `tv` on
load and clipped on store; every operation is per-row or a contraction over
the node index, so padded rows never mix with valid ones.

Validation: `test_fused_cpu.jl` checks plain-array mirrors of the kernel
arithmetic against ClimaCore on three geometries (CPU, no GPU needed);
`benchmark_fused.jl --check-only` gates the kernels on the GPU;
`ct.code_tiled` lowers every kernel variant to Tile IR on any machine
(`sm_arch = v"8.0"`, `bytecode_version = v"13.1"`).

Not covered: the momentum hyperdiffusion `wgrad(div(u)) − wcurl(curl(u))`,
which needs the vector-operator contractions (`J`-weighted divergence,
Levi-Civita curl) as a further kernel, and `npoly ≠ 3` (tile extents must
be powers of two).

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
