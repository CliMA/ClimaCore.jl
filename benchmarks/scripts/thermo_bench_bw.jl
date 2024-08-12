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
│     TBB.singlefield_bc!(x_soa, us; nreps=100, bm)  │ 67 microseconds, 554 nanoseconds  │ 29.4429 │ 600.341     │ 2              │ 100    │
│     TBB.singlefield_bc!(x_aos, us; nreps=100, bm)  │ 69 microseconds, 653 nanoseconds  │ 28.5556 │ 582.248     │ 2              │ 100    │
│     TBB.thermo_func_bc!(x, us; nreps=100, bm)      │ 796 microseconds, 877 nanoseconds │ 12.4798 │ 254.462     │ 10             │ 100    │
│     TBB.thermo_func_sol!(x_vec, us; nreps=100, bm) │ 131 microseconds, 72 nanoseconds  │ 75.873  │ 1547.05     │ 10             │ 100    │
└────────────────────────────────────────────────────┴───────────────────────────────────┴─────────┴─────────────┴────────────────┴────────┘

[ Info: device = ClimaComms.CUDADevice()
Problem size: (63, 4, 4, 1, 5400), float_type = Float64, device_bandwidth_GBs=2039
┌────────────────────────────────────────────────────┬───────────────────────────────────┬─────────┬─────────────┬────────────────┬────────┐
│ funcs                                              │ time per call                     │ bw %    │ achieved bw │ n-reads/writes │ n-reps │
├────────────────────────────────────────────────────┼───────────────────────────────────┼─────────┼─────────────┼────────────────┼────────┤
│     TBB.singlefield_bc!(x_soa, us; nreps=100, bm)  │ 108 microseconds, 790 nanoseconds │ 36.5653 │ 745.567     │ 2              │ 100    │
│     TBB.singlefield_bc!(x_aos, us; nreps=100, bm)  │ 123 microseconds, 730 nanoseconds │ 32.1501 │ 655.541     │ 2              │ 100    │
│     TBB.thermo_func_bc!(x, us; nreps=100, bm)      │ 1 millisecond, 43 microseconds    │ 19.0568 │ 388.569     │ 10             │ 100    │
│     TBB.thermo_func_sol!(x_vec, us; nreps=100, bm) │ 256 microseconds, 717 nanoseconds │ 77.477  │ 1579.76     │ 10             │ 100    │
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

function singlefield_bc!(x, us; nreps = 1, bm=nothing, n_trials = 30)
    e = Inf
    for t in 1:n_trials
        et = CUDA.@elapsed begin
            for _ in 1:nreps
                (; ρ_read, ρ_write) = x
                @. ρ_write = ρ_read
            end
        end
        e = min(e, et)
    end
    push_info(bm; e, nreps, caller = @caller_name(@__FILE__),n_reads_writes=2)
    return nothing
end

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

    x_aos = fill((; ρ_read = FT(0), ρ_write = FT(0)), cspace)
    x_soa = (;
        ρ_read = Fields.Field(FT, cspace),
        ρ_write = Fields.Field(FT, cspace),
    )
    @. x_soa.ρ_read = 6
    @. x_soa.ρ_write = 7
    @. x_aos.ρ_read = 6
    @. x_aos.ρ_write = 7
    TBB.singlefield_bc!(x_soa, us; nreps=1, n_trials = 1)
    TBB.singlefield_bc!(x_aos, us; nreps=1, n_trials = 1)

    TBB.thermo_func_bc!(x, us; nreps=1, n_trials = 1)
    TBB.thermo_func_sol!(x_vec, us; nreps=1, n_trials = 1)

    rc = Fields.rcompare(x_vec, to_vec(x))
    rc || Fields.@rprint_diff(x_vec, to_vec(x)) # test correctness (should print nothing)
    @test rc # test correctness

    TBB.singlefield_bc!(x_soa, us; nreps=100, bm)
    TBB.singlefield_bc!(x_aos, us; nreps=100, bm)
    TBB.thermo_func_bc!(x, us; nreps=100, bm)
    TBB.thermo_func_sol!(x_vec, us; nreps=100, bm)

    TBB.tabulate_benchmark(bm)

end
#! format: on
