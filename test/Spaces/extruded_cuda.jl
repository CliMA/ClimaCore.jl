#=
julia --project
using Revise; include(joinpath("test", "Spaces", "extruded_cuda.jl"))
=#
using LinearAlgebra, IntervalSets, UnPack
using ClimaComms
using CUDA
using ClimaComms: SingletonCommsContext
import ClimaCore: Domains, Topologies, Meshes, Spaces, Geometry, column, Fields
using Test

include(joinpath(@__DIR__, "..", "Fields", "util_spaces.jl"))

function FieldFromNamedTuple(space, nt::NamedTuple)
    cmv(z) = nt
    return cmv.(Fields.coordinate_field(space))
end

@testset "CuArray-backed extruded spaces" begin
    context = SingletonCommsContext(
        CUDA.functional() ? ClimaComms.CUDA() : ClimaComms.CPU(),
    )
    @info context
    FT = Float64
    for space in all_spaces(FT; zelem = 10, context)
        Y = FieldFromNamedTuple(space, (; v = FT(0)))
        X = FieldFromNamedTuple(space, (; v = FT(0)))
    end
end

@testset "copyto! with CuArray-backed extruded spaces" begin
    context = SingletonCommsContext(
        CUDA.functional() ? ClimaComms.CUDA() : ClimaComms.CPU(),
    )
    @info context
    FT = Float64
    for space in all_spaces(FT; zelem = 10, context)
        Y = FieldFromNamedTuple(space, (; v = FT(0)))
        X = FieldFromNamedTuple(space, (; v = FT(0)))
        @. X.v = Y.v
    end
end
