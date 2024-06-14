#=
julia --project=.buildkite
using Revise; include(joinpath("test", "Operators", "finitedifference", "benchmark_examples.jl"))
=#
import ClimaCore
using ClimaComms
ClimaComms.@import_required_backends
using BenchmarkTools
@isdefined(TU) || include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
);
;

using Test
using StaticArrays, IntervalSets, LinearAlgebra
using JET

import ClimaCore: Spaces, Fields, Operators
import ClimaCore.Domains: Geometry

# https://github.com/CliMA/ClimaCore.jl/issues/1602
const CT3 = Geometry.Contravariant3Vector
const C12 = ClimaCore.Geometry.Covariant12Vector
const ᶠwinterp = Operators.WeightedInterpolateC2F(
    bottom = Operators.Extrapolate(),
    top = Operators.Extrapolate(),
)
function set_ᶠuₕ³!(ᶜx, ᶠx)
    ᶜJ = Fields.local_geometry_field(ᶜx).J
    @. ᶠx.ᶠuₕ³ = ᶠwinterp(ᶜx.ρ * ᶜJ, CT3(ᶜx.uₕ))
    return nothing
end
@testset "Inference/allocations when broadcasting types" begin
    FT = Float64
    cspace = CenterExtrudedFiniteDifferenceSpace(FT; zelem = 25, helem = 10)
    fspace = Spaces.FaceExtrudedFiniteDifferenceSpace(cspace)
    device = ClimaComms.device(cspace)
    @info "device = $device"
    ᶜx = fill((; uₕ = zero(C12{FT}), ρ = FT(0)), cspace)
    ᶠx = fill((; ᶠuₕ³ = zero(CT3{FT})), fspace)
    set_ᶠuₕ³!(ᶜx, ᶠx) # compile
    if device isa ClimaComms.CUDADevice
        @test_broken 0 == @allocated set_ᶠuₕ³!(ᶜx, ᶠx)
    else
        @test 0 == @allocated set_ᶠuₕ³!(ᶜx, ᶠx)
    end

    trial = @benchmark ClimaComms.@cuda_sync $device set_ᶠuₕ³!($ ᶜx, $ᶠx)
    show(stdout, MIME("text/plain"), trial)
end
