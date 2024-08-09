#=
julia --project=.buildkite
using Revise; include(joinpath("benchmarks", "scripts", "thermo_bench_bw.jl"))

# Info

 - This is a benchmark for ClimaCore pointwise kernels, with
   multiple field variables. We locally define a toy version
   of Thermodynamics, and avoid all flops and only measure
   the bandwidth performance achieved on the hardware.

# Benchmark results:

Clima A100:
```
[ Info: device = ClimaComms.CUDADevice()
Problem size: (63, 4, 4, 1, 5400), float_type = Float32, device_bandwidth_GBs=2039
┌────────────────────────────────────────────────────┬───────────────────────────────────┬─────────┬─────────────┬────────────────┬────────┐
│ funcs                                              │ time per call                     │ bw %    │ achieved bw │ n-reads/writes │ n-reps │
├────────────────────────────────────────────────────┼───────────────────────────────────┼─────────┼─────────────┼────────────────┼────────┤
│     TBB.thermo_func_bc!(x, us; nreps=100, bm)      │ 797 microseconds, 92 nanoseconds  │ 12.4764 │ 254.394     │ 10             │ 100    │
│     TBB.thermo_func_sol!(x_vec, us; nreps=100, bm) │ 131 microseconds, 851 nanoseconds │ 75.4252 │ 1537.92     │ 10             │ 100    │
│     TBB.thermo_func_bc!(x, us; nreps=100, bm)      │ 797 microseconds, 164 nanoseconds │ 12.4753 │ 254.371     │ 10             │ 100    │
│     TBB.thermo_func_sol!(x_vec, us; nreps=100, bm) │ 131 microseconds, 943 nanoseconds │ 75.3725 │ 1536.84     │ 10             │ 100    │
└────────────────────────────────────────────────────┴───────────────────────────────────┴─────────┴─────────────┴────────────────┴────────┘

[ Info: device = ClimaComms.CUDADevice()
Problem size: (63, 4, 4, 1, 5400), float_type = Float64, device_bandwidth_GBs=2039
┌────────────────────────────────────────────────────┬───────────────────────────────────┬─────────┬─────────────┬────────────────┬────────┐
│ funcs                                              │ time per call                     │ bw %    │ achieved bw │ n-reads/writes │ n-reps │
├────────────────────────────────────────────────────┼───────────────────────────────────┼─────────┼─────────────┼────────────────┼────────┤
│     TBB.thermo_func_bc!(x, us; nreps=100, bm)      │ 1 millisecond, 45 microseconds    │ 19.0163 │ 387.743     │ 10             │ 100    │
│     TBB.thermo_func_sol!(x_vec, us; nreps=100, bm) │ 258 microseconds, 120 nanoseconds │ 77.0559 │ 1571.17     │ 10             │ 100    │
│     TBB.thermo_func_bc!(x, us; nreps=100, bm)      │ 1 millisecond, 46 microseconds    │ 19.0147 │ 387.709     │ 10             │ 100    │
│     TBB.thermo_func_sol!(x_vec, us; nreps=100, bm) │ 257 microseconds, 915 nanoseconds │ 77.1171 │ 1572.42     │ 10             │ 100    │
└────────────────────────────────────────────────────┴───────────────────────────────────┴─────────┴─────────────┴────────────────┴────────┘
```
=#

#! format: off
module ThermoBenchBandwidth

include("benchmark_utils.jl")

import ClimaCore
import CUDA
using ClimaComms
using Test
using StaticArrays, IntervalSets, LinearAlgebra
using JET

import ClimaCore: Spaces, Fields
import ClimaCore.Domains: Geometry

struct PhaseEquil{FT}
    ρ::FT
    p::FT
    e_int::FT
    q_tot::FT
    T::FT
end

@inline Base.zero(::Type{PhaseEquil{FT}}) where {FT} =
    PhaseEquil{FT}(0, 0, 0, 0, 0)

function thermo_func_bc!(x, us; nreps = 1, bm=nothing, n_trials = 30)
    e = Inf
    for t in 1:n_trials
        et = CUDA.@elapsed begin
            for _ in 1:nreps
                (; ts, ρ,p,e_int,q_tot,T) = x
                @. ts = PhaseEquil(ρ,p,e_int,q_tot,T) # 5 reads, 5 writes, 0 flops
            end
        end
        e = min(e, et)
    end
    push_info(bm; e, nreps, caller = @caller_name(@__FILE__),n_reads_writes=10)
    return nothing
end

function thermo_func_sol!(x, us::UniversalSizesStatic; nreps = 1, bm=nothing, n_trials = 30)
    e = Inf
    for t in 1:n_trials
        et = CUDA.@elapsed begin
            (; ts, ρ,p,e_int,q_tot,T) = x
            kernel = CUDA.@cuda always_inline = true launch = false thermo_func_sol_kernel!(ts,ρ,p,e_int,q_tot,T,us)
            N = get_N(us)
            config = CUDA.launch_configuration(kernel.fun)
            threads = min(N, config.threads)
            blocks = cld(N, threads)
            for _ in 1:nreps
                kernel(ts,ρ,p,e_int,q_tot,T,us; threads, blocks)
            end
        end
        e = min(e, et)
    end
    push_info(bm; e, nreps, caller = @caller_name(@__FILE__),n_reads_writes=10)
    return nothing
end

# Mimics how indexing works in generalized pointwise kernels
function thermo_func_sol_kernel!(ts, ρ,p,e_int,q_tot,T, us)
    @inbounds begin
        FT = eltype(ts.ρ)
        I = (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x
        if I ≤ get_N(us)
            # 5 reads, 5 writes, 0 flops
            ts_i = PhaseEquil(ρ[I],p[I],e_int[I],q_tot[I],T[I])
            ts.ρ[I] = ts_i.ρ
            ts.p[I] = ts_i.p
            ts.T[I] = ts_i.T
            ts.e_int[I] = ts_i.e_int
            ts.q_tot[I] = ts_i.q_tot
        end
    end
    return nothing
end

end # module

import .ThermoBenchBandwidth as TBB

import CUDA
using ClimaComms
using ClimaCore
import ClimaCore: Spaces, Fields
import ClimaCore.Domains: Geometry

ENV["CLIMACOMMS_DEVICE"] = "CUDA";
ClimaComms.@import_required_backends
using BenchmarkTools
@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU;

using Test
@testset "Thermo state" begin
    FT = Float32
    bm = TBB.Benchmark(;problem_size=(63,4,4,1,5400), float_type=FT)
    device = ClimaComms.device()
    context = ClimaComms.context(device)
    cspace = TU.CenterExtrudedFiniteDifferenceSpace(
        FT;
        zelem = 63,
        context,
        helem = 30,
        Nq = 4,
    )
    fspace = Spaces.FaceExtrudedFiniteDifferenceSpace(cspace)
    @info "device = $device"
    # TODO: fill with non-trivial values (e.g., use Thermodynamics TestedProfiles) to verify correctness.
    nt_core = (; ρ = FT(1), p = FT(2),e_int = FT(3),q_tot = FT(4),T = FT(5))
    nt_ts = (;
        ρ = FT(0),
        p = FT(0),
        e_int = FT(0),
        q_tot = FT(0),
        T = FT(0),
    )
    x = fill((; ts = zero(TBB.PhaseEquil{FT}), nt_core...), cspace)
    xv = fill((; ts = nt_ts, nt_core...), cspace)
    (_, Nij, _, Nv, Nh) = size(Fields.field_values(x.ts))
    us = TBB.UniversalSizesStatic(Nv, Nij, Nh)
    function to_vec(ξ)
        pns = propertynames(ξ)
        dl_vals = map(pns) do pn
            val = getproperty(ξ, pn)
            pn == :ts ? to_vec(val) :
            CUDA.CuArray(collect(vec(parent(Fields.field_values(val)))))
        end
        return (; zip(propertynames(ξ), dl_vals)...)
    end
    x_vec = to_vec(xv)

    TBB.thermo_func_bc!(x, us; nreps=1, n_trials = 1)
    TBB.thermo_func_sol!(x_vec, us; nreps=1, n_trials = 1)

    rc = Fields.rcompare(x_vec, to_vec(x))
    rc || Fields.rprint_diff(x_vec, to_vec(x)) # test correctness (should print nothing)
    @test rc # test correctness

    TBB.thermo_func_bc!(x, us; nreps=100, bm)
    TBB.thermo_func_sol!(x_vec, us; nreps=100, bm)

    TBB.thermo_func_bc!(x, us; nreps=100, bm)
    TBB.thermo_func_sol!(x_vec, us; nreps=100, bm)

    TBB.tabulate_benchmark(bm)

end
#! format: on
