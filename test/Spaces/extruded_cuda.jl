#=
julia --project
using Revise; include(joinpath("test", "Spaces", "extruded_cuda.jl"))
=#
using LinearAlgebra, IntervalSets
using ClimaComms
if pkgversion(ClimaComms) >= v"0.6"
    ClimaComms.@import_required_backends
end
using CUDA
using ClimaComms: SingletonCommsContext
import ClimaCore
import ClimaCore:
    Domains, Topologies, Meshes, Spaces, Geometry, column, Fields, Grids
using Test

include(
    joinpath(pkgdir(ClimaCore), "test", "TestUtilities", "TestUtilities.jl"),
)
import .TestUtilities as TU

compare(cpu, gpu) = all(parent(cpu) .≈ Array(parent(gpu)))
compare(cpu, gpu, f) = all(parent(f(cpu)) .≈ Array(parent(f(gpu))))

@testset "CuArray-backed extruded spaces" begin
    context = SingletonCommsContext(
        CUDA.functional() ? ClimaComms.CUDADevice() :
        ClimaComms.CPUSingleThreaded(),
    )
    collect(TU.all_spaces(Float64; zelem = 10, context)) # make sure we can construct spaces
    as = collect(TU.all_spaces(Float64; zelem = 10, context))
    @test length(as) == 8
end

@testset "copyto! with CuArray-backed extruded spaces" begin
    cpu_context = SingletonCommsContext(ClimaComms.CPUSingleThreaded())
    gpu_context = SingletonCommsContext(ClimaComms.CUDADevice())

    FT = Float64
    CUDA.allowscalar(true)
    # TODO: add support and test for all spaces
    cpuspace = TU.CenterExtrudedFiniteDifferenceSpace(FT; context = cpu_context)
    gpuspace = TU.CenterExtrudedFiniteDifferenceSpace(FT; context = gpu_context)

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
    CUDA.allowscalar(false)
    @. X.v = Y.v
    CUDA.allowscalar(true)
    @test all(parent(Y.v) .== parent(X.v))


    CUDA.allowscalar(true)
    # TODO: add support and test for all spaces
    cpuspace = TU.SpectralElementSpace2D(FT; context = cpu_context)
    gpuspace = TU.SpectralElementSpace2D(FT; context = gpu_context)

    # Test that all geometries match with CPU version:
    @test compare(cpuspace, gpuspace, x -> Spaces.local_geometry_data(x))

    space = gpuspace
    Y = Fields.Field(typeof((; v = FT(0))), space)
    X = Fields.Field(typeof((; v = FT(0))), space)
    @. Y.v = 0
    @. X.v = 2
    @test all(parent(Y.v) .== 0)
    @test all(parent(X.v) .== 2)
    CUDA.allowscalar(false)
    @. X.v = Y.v
    CUDA.allowscalar(true)
    @test all(parent(Y.v) .== parent(X.v))
end
