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
const CFT = Float64
function FieldFromNamedTuple(space, nt::NamedTuple)
    cmv(z) = (; v = CFT(0))
    return cmv.(Fields.coordinate_field(space))
end

@testset "CuArray-backed extruded spaces" begin
    context = SingletonCommsContext(
        CUDA.functional() ? ClimaComms.CUDA() : ClimaComms.CPU(),
    )
    collect(all_spaces(Float64; zelem = 10, context)) # make sure we can construct spaces
    as = collect(all_spaces(Float64; zelem = 10, context))
    @test length(as) == 8
end

@testset "copyto! with CuArray-backed extruded spaces" begin
    context = SingletonCommsContext(
        CUDA.functional() ? ClimaComms.CUDA() : ClimaComms.CPU(),
    )
    @info context
    FT = Float64
    # CUDA.allowscalar(false)
    for space in all_spaces(FT; zelem = 10, context)
        Y = Fields.Field(typeof((; v = FT(0))), space)
        X = Fields.Field(typeof((; v = FT(0))), space)
        @. Y.v = 0
        @. X.v = 2
        @test all(parent(Y.v) .== 0)
        @test all(parent(X.v) .== 2)
        @. X.v = Y.v
        function data_layout_name(space)
            s = string(typeof(space))
            first(split(last(split(s, "DataLayouts.")), "{"))
        end
        if all(parent(Y.v) .== parent(X.v))
            @info "space $(data_layout_name(space)) succeeded! 🎉"
        else
            @warn "space $(data_layout_name(space)) failed"
            @show maximum(abs.(parent(Y.v) .- parent(X.v)))
            @show minimum(abs.(parent(Y.v) .- parent(X.v)))
            @show maximum(abs.(parent(Y.v)))
            @show minimum(abs.(parent(X.v)))
        end
        # @test all(parent(Y.v) .== parent(X.v))
    end
end
