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
    FT = Spaces.undertype(space)
    cmv(z) = (; v = FT(0))
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
    cpu_context = SingletonCommsContext(ClimaComms.CPU())
    gpu_context = SingletonCommsContext(ClimaComms.CUDA())

    FT = Float64
    CUDA.allowscalar(true)
    # TODO: add support and test for all spaces
    cpuspace = last(all_spaces(FT; zelem = 10, context = cpu_context))
    gpuspace = last(all_spaces(FT; zelem = 10, context = gpu_context))

    # Test that all geometries match with CPU version:
    @test parent(cpuspace.center_local_geometry) ≈
          Array(parent(gpuspace.center_local_geometry)),
    @test parent(cpuspace.face_local_geometry) ≈
          Array(parent(gpuspace.face_local_geometry))

    @test all(
        parent(cpuspace.center_ghost_geometry) .==
        Array(parent(gpuspace.center_ghost_geometry)),
    )
    @test all(
        parent(cpuspace.face_ghost_geometry) .==
        Array(parent(gpuspace.face_ghost_geometry)),
    )

    space = gpuspace
    Y = Fields.Field(typeof((; v = FT(0))), space)
    X = Fields.Field(typeof((; v = FT(0))), space)
    @. Y.v = 0
    @. X.v = 2
    @test all(parent(Y.v) .== 0)
    @test all(parent(X.v) .== 2)

    gpuz = Fields.coordinate_field(gpuspace).z
    cpuz = Fields.coordinate_field(cpuspace).z
    pgpuz = parent(gpuz)
    pcpuz = parent(cpuz)
    CUDA.copyto!(pgpuz, pcpuz)
    @. Y.v = gpuz
    @test all(Array(parent(Y.v)) .== pcpuz)

    CUDA.allowscalar(false)
    @. X.v = Y.v
    CUDA.allowscalar(true)
    @test all(parent(Y.v) .== parent(X.v))
end
