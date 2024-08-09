#=
julia --project=.buildkite
using Revise; include(joinpath("benchmarks", "scripts", "thermo_bench.jl"))

This benchmark requires Thermodynamics and ClimaParams
to be in your local environment to run.

# Benchmark results:

Clima A100:
```
[ Info: device = ClimaComms.CUDADevice()
Problem size: (63, 4, 4, 1, 5400), float_type = Float32, device_bandwidth_GBs=2039
┌──────────────────────────────────────────────────────────────────┬───────────────────────────────────┬─────────┬─────────────┬────────────────┬────────┐
│ funcs                                                            │ time per call                     │ bw %    │ achieved bw │ n-reads/writes │ n-reps │
├──────────────────────────────────────────────────────────────────┼───────────────────────────────────┼─────────┼─────────────┼────────────────┼────────┤
│     TB.thermo_func_bc!(x, thermo_params, us; nreps=100, bm)      │ 586 microseconds, 517 nanoseconds │ 15.2602 │ 311.155     │ 9              │ 100    │
│     TB.thermo_func_sol!(x_vec, thermo_params, us; nreps=100, bm) │ 292 microseconds, 178 nanoseconds │ 30.6332 │ 624.611     │ 9              │ 100    │
│     TB.thermo_func_bc!(x, thermo_params, us; nreps=100, bm)      │ 586 microseconds, 988 nanoseconds │ 15.2479 │ 310.905     │ 9              │ 100    │
│     TB.thermo_func_sol!(x_vec, thermo_params, us; nreps=100, bm) │ 292 microseconds, 178 nanoseconds │ 30.6332 │ 624.611     │ 9              │ 100    │
└──────────────────────────────────────────────────────────────────┴───────────────────────────────────┴─────────┴─────────────┴────────────────┴────────┘
```
=#

#! format: off
module ThermoBench

include("benchmark_utils.jl")

import ClimaCore
import CUDA
using ClimaComms
using Test
using StaticArrays, IntervalSets, LinearAlgebra
using JET

import ClimaCore: Spaces, Fields
import ClimaCore.Domains: Geometry

@inline ts_gs(thermo_params, e_tot, q_tot, K, Φ, ρ) =
    thermo_state(thermo_params, e_tot - K - Φ, q_tot, ρ)
@inline thermo_state(thermo_params, ρ, e_int, q_tot) =
    TD.PhaseEquil_ρeq(thermo_params,ρ,e_int,q_tot, 3, eltype(thermo_params)(0.003))

import Thermodynamics as TD

function thermo_func_bc!(x, thermo_params, us; nreps = 1, bm=nothing, n_trials = 30)
    e = Inf
    for t in 1:n_trials
        et = CUDA.@elapsed begin
            for _ in 1:nreps
                (; ts, e_tot, q_tot, K, Φ, ρ) = x
                @. ts = ts_gs(thermo_params, e_tot, q_tot, K, Φ, ρ) # 5 reads, 5 writes, many flops
            end
        end
        e = min(e, et)
    end
    push_info(bm; e, nreps, caller = @caller_name(@__FILE__),n_reads_writes=5+4) # TODO: verify this
    return nothing
end

function thermo_func_sol!(x, thermo_params, us::UniversalSizesStatic; nreps = 1, bm=nothing, n_trials = 30)
    e = Inf
    for t in 1:n_trials
        et = CUDA.@elapsed begin
            (; ts, e_tot, q_tot, K, Φ, ρ) = x
            kernel = CUDA.@cuda always_inline = true launch = false thermo_func_sol_kernel!(ts,e_tot,q_tot,K,Φ,ρ,thermo_params,us)
            N = get_N(us)
            config = CUDA.launch_configuration(kernel.fun)
            threads = min(N, config.threads)
            blocks = cld(N, threads)
            for _ in 1:nreps
                kernel(ts,e_tot,q_tot,K,Φ,ρ,thermo_params,us; threads, blocks)
            end
        end
        e = min(e, et)
    end
    push_info(bm; e, nreps, caller = @caller_name(@__FILE__),n_reads_writes=5+4) # TODO: verify this
    return nothing
end

# Mimics how indexing works in generalized pointwise kernels
function thermo_func_sol_kernel!(ts, e_tot, q_tot, K, Φ, ρ, thermo_params, us)
    @inbounds begin
        FT = eltype(e_tot)
        I = (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x
        if I ≤ get_N(us)
            # Data is not read into the correct fields because this is only used
            # to compare with the case when the number of flops goes to zero.

            # 5 reads, 5 writes, potentially many flops (see thermodynamics for estimate)
            ts_i = ts_gs(thermo_params, e_tot[I], q_tot[I], K[I], Φ[I], ρ[I])
            ts.ρ[I] = ts_i.ρ
            ts.p[I] = ts_i.p
            ts.T[I] = ts_i.T
            ts.e_int[I] = ts_i.e_int
            ts.q_tot[I] = ts_i.q_tot
        end
    end
    return nothing
end

end

import ClimaParams # trigger Thermo extension
import .ThermoBench as TB

import Thermodynamics as TD
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
    bm = TB.Benchmark(;problem_size=(63,4,4,1,5400), float_type=FT)
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
    thermo_params = TD.Parameters.ThermodynamicsParameters(FT)
    # TODO: fill with non-trivial values (e.g., use Thermodynamics TestedProfiles) to verify correctness.
    nt_core = (; K = FT(0), Φ = FT(1), ρ = FT(0), e_tot = FT(1), q_tot = FT(0.001))
    nt_ts = (;
        ρ = FT(0),
        p = FT(0),
        e_int = FT(0),
        q_tot = FT(0),
        T = FT(0),
    )
    x = fill((; ts = zero(TD.PhaseEquil{FT}), nt_core...), cspace)
    xv = fill((; ts = nt_ts, nt_core...), cspace)
    (_, Nij, _, Nv, Nh) = size(Fields.field_values(x.ts))
    us = TB.UniversalSizesStatic(Nv, Nij, Nh)
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

    TB.thermo_func_bc!(x, thermo_params, us; nreps=1, n_trials = 1)
    TB.thermo_func_sol!(x_vec, thermo_params, us; nreps=1, n_trials = 1)

    rc = Fields.rcompare(x_vec, to_vec(x))
    rc || Fields.rprint_diff(x_vec, to_vec(x)) # test correctness (should print nothing)
    @test rc # test correctness

    TB.thermo_func_bc!(x, thermo_params, us; nreps=100, bm)
    TB.thermo_func_sol!(x_vec, thermo_params, us; nreps=100, bm)

    TB.thermo_func_bc!(x, thermo_params, us; nreps=100, bm)
    TB.thermo_func_sol!(x_vec, thermo_params, us; nreps=100, bm)

    TB.tabulate_benchmark(bm)

end
#! format: on
