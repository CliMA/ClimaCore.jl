#=
julia --project=.buildkite
using Revise; include(joinpath("benchmarks", "scripts", "thermo_bench.jl"))

This benchmark requires Thermodynamics and ClimaParams
to be in your local environment to run.
=#

#! format: off
import ClimaCore
import Thermodynamics as TD
import CUDA
using ClimaComms
import ClimaCore: Spaces, Fields
import ClimaCore.Domains: Geometry

ENV["CLIMACOMMS_DEVICE"] = "CUDA";
ClimaComms.@import_required_backends
using BenchmarkTools
@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
import .TestUtilities as TU;

module ThermoBench
import ClimaCore
import CUDA
using ClimaComms
using Test
using StaticArrays, IntervalSets, LinearAlgebra
using JET

import ClimaCore: Spaces, Fields
import ClimaCore.Domains: Geometry
import Dates
print_time_and_units(x) = println(time_and_units_str(x))
time_and_units_str(x::Real) =
    trunc_time(string(compound_period(x, Dates.Second)))
function compound_period(x::Real, ::Type{T}) where {T <: Dates.Period}
    nf = Dates.value(convert(Dates.Nanosecond, T(1)))
    ns = Dates.Nanosecond(ceil(x * nf))
    return Dates.canonicalize(Dates.CompoundPeriod(ns))
end
trunc_time(s::String) = count(',', s) > 1 ? join(split(s, ",")[1:2], ",") : s

@inline ts_gs(thermo_params, e_tot, q_tot, K, Φ, ρ) =
    thermo_state(thermo_params, e_tot - K - Φ, q_tot, ρ)
@inline thermo_state(thermo_params, ρ, e_int, q_tot) =
    TD.PhaseEquil_ρeq(thermo_params,ρ,e_int,q_tot, 3, eltype(thermo_params)(0.003))

struct UniversalSizesStatic{Nv, Nij, Nh} end
get_Nv(::UniversalSizesStatic{Nv}) where {Nv} = Nv
get_Nij(::UniversalSizesStatic{Nv, Nij}) where {Nv, Nij} = Nij
get_Nh(::UniversalSizesStatic{Nv, Nij, Nh}) where {Nv, Nij, Nh} = Nh
get_N(us::UniversalSizesStatic{Nv, Nij}) where {Nv, Nij} =
    prod((Nv, Nij, Nij, 1, get_Nh(us)))
UniversalSizesStatic(Nv, Nij, Nh) = UniversalSizesStatic{Nv, Nij, Nh}()
import Thermodynamics as TD

function thermo_func_bc!(x, thermo_params, us, niter = 1)
    e = CUDA.@elapsed begin
        for i in 1:niter # reduce variance / impact of launch latency
            (; ts, e_tot, q_tot, K, Φ, ρ) = x
            @. ts = ts_gs(thermo_params, e_tot, q_tot, K, Φ, ρ)
        end
    end
    print_time_and_units(e / niter)
    return nothing
end

function thermo_func_sol!(x, thermo_params, us::UniversalSizesStatic, niter = 1)
    e = CUDA.@elapsed begin
        for i in 1:niter # reduce variance / impact of launch latency
            (; ts, e_tot, q_tot, K, Φ, ρ) = x
            kernel = CUDA.@cuda always_inline = true launch = false thermo_func_sol_kernel!(ts,e_tot,q_tot,K,Φ,ρ,thermo_params,us)
            N = get_N(us)
            config = CUDA.launch_configuration(kernel.fun)
            threads = min(N, config.threads)
            blocks = cld(N, threads)
            kernel(ts,e_tot,q_tot,K,Φ,ρ,thermo_params,us; threads, blocks)
        end
    end
    print_time_and_units(e / niter)
    return nothing
end

# Mimics how indexing works in generalized pointwise kernels
function thermo_func_sol_kernel!(ts, e_tot, q_tot, K, Φ, ρ, thermo_params, us)
    @inbounds begin
        (; ts_ρ, ts_p, ts_e_int, ts_q_tot, ts_T) = ts
        FT = eltype(e_tot)
        I = (CUDA.blockIdx().x - Int32(1)) * CUDA.blockDim().x + CUDA.threadIdx().x
        if I ≤ get_N(us)
            ts_i = ts_gs(thermo_params, e_tot[I], q_tot[I], K[I], Φ[I], ρ[I]) # 5 reads, potentially many flops (see thermodynamics for estimate)

            # Data is not read into the correct fields because this is only used
            # to compare with the case when the number of flops goes to zero.
            # ts_i = TD.PhaseEquil{FT}(ρ[I], K[I], e_tot[I], q_tot[I], Φ[I]) # 5 reads, 0 flops
            ts_ρ[I] = ts_i.ρ
            ts_p[I] = ts_i.p
            ts_T[I] = ts_i.T
            ts_e_int[I] = ts_i.e_int
            ts_q_tot[I] = ts_i.q_tot
        end
    end
    return nothing
end

end

import ClimaParams # trigger Thermo extension
import .ThermoBench
using Test
@testset "Thermo state" begin
    FT = Float32
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
    nt_core = (; K = FT(0), Φ = FT(1), ρ = FT(0), e_tot = FT(1), q_tot = FT(0.001))
    nt_ts = (;
        ts_ρ = FT(0),
        ts_p = FT(0),
        ts_e_int = FT(0),
        ts_q_tot = FT(0),
        ts_T = FT(0),
    )
    x = fill((; ts = zero(TD.PhaseEquil{FT}), nt_core...), cspace)
    xv = fill((; ts = nt_ts, nt_core...), cspace)
    (_, Nij, _, Nv, Nh) = size(Fields.field_values(x.ts))
    us = ThermoBench.UniversalSizesStatic(Nv, Nij, Nh)
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

    ThermoBench.thermo_func_bc!(x, thermo_params, us)
    ThermoBench.thermo_func_sol!(x_vec, thermo_params, us)

    ThermoBench.thermo_func_bc!(x, thermo_params, us, 100)
    ThermoBench.thermo_func_sol!(x_vec, thermo_params, us, 100)
end
#! format: on
