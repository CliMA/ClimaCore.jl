#=
julia --project
using Revise; include(joinpath("test", "Spaces", "extruded_cuda.jl"))
=#
using LinearAlgebra, IntervalSets
using ClimaComms
ClimaComms.@import_required_backends

using ClimaComms: SingletonCommsContext
import ClimaCore
import ClimaCore:
    Domains, Topologies, Meshes, Spaces, Geometry, column, Fields, Grids
using Test

include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
)


compare(cpu, gpu) = all(parent(cpu) .≈ Array(parent(gpu)))
compare(cpu, gpu, f) = all(parent(f(cpu)) .≈ Array(parent(f(gpu))))

@testset "CuArray-backed extruded spaces" begin
    device = ClimaComms.device()
    context = SingletonCommsContext(device)
    collect(all_spaces(Float64; zelem = 10, context)) # make sure we can construct spaces
    as = collect(all_spaces(Float64; zelem = 10, context))
    @test length(as) == 8
end

@testset "copyto! with CuArray-backed extruded spaces" begin
    cpu_context = SingletonCommsContext(ClimaComms.CPUSingleThreaded())
    gpu_context = SingletonCommsContext(ClimaComms.CUDADevice())
    device = ClimaComms.device(gpu_context)

    FT = Float64
    device = ClimaComms.device(gpu_context)
    local X, Y
    ClimaComms.allowscalar(device) do
        # TODO: add support and test for all spaces
        cpuspace =
            CenterExtrudedFiniteDifferenceSpace(FT; context = cpu_context)
        gpuspace =
            CenterExtrudedFiniteDifferenceSpace(FT; context = gpu_context)

        # Test that all geometries match with CPU version:
        @test compare(
            cpuspace,
            gpuspace,
            x -> Spaces.local_geometry_data(Spaces.grid(x), Grids.CellCenter()),
        )
        @test compare(
            cpuspace,
            gpuspace,
            x -> Spaces.local_geometry_data(Spaces.grid(x), Grids.CellFace()),
        )

        space = gpuspace
        Y = Fields.Field(typeof((; v = FT(0))), space)
        X = Fields.Field(typeof((; v = FT(0))), space)
        @. Y.v = 0
        @. X.v = 2
        @test all(parent(Y.v) .== 0)
        @test all(parent(X.v) .== 2)
    end

    @. X.v = Y.v
    ClimaComms.allowscalar(device) do
        @test all(parent(Y.v) .== parent(X.v))
        # TODO: add support and test for all spaces
        cpuspace = SpectralElementSpace2D(FT; context = cpu_context)
        gpuspace = SpectralElementSpace2D(FT; context = gpu_context)

        # Test that all geometries match with CPU version:
        @test compare(cpuspace, gpuspace, x -> Spaces.local_geometry_data(x))

        space = gpuspace
        Y = Fields.Field(typeof((; v = FT(0))), space)
        X = Fields.Field(typeof((; v = FT(0))), space)
        @. Y.v = 0
        @. X.v = 2
        @test all(parent(Y.v) .== 0)
        @test all(parent(X.v) .== 2)
    end

    @. X.v = Y.v
    ClimaComms.allowscalar(device) do
        @test all(parent(Y.v) .== parent(X.v))
    end
end
