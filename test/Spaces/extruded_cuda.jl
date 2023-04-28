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

compare(cpu, gpu) = all(parent(cpu) .≈ Array(parent(gpu)))
compare(cpu, gpu, sym) =
    all(parent(getproperty(cpu, sym)) .≈ Array(parent(getproperty(gpu, sym))))

@testset "CuArray-backed extruded spaces" begin
    context = SingletonCommsContext(
        CUDA.functional() ? ClimaComms.CUDADevice() : ClimaComms.CPUDevice(),
    )
    collect(all_spaces(Float64; zelem = 10, context)) # make sure we can construct spaces
    as = collect(all_spaces(Float64; zelem = 10, context))
    @test length(as) == 8
end

@testset "copyto! with CuArray-backed extruded spaces" begin
    cpu_context = SingletonCommsContext(ClimaComms.CPUDevice())
    gpu_context = SingletonCommsContext(ClimaComms.CUDADevice())

    FT = Float64
    CUDA.allowscalar(true)
    # TODO: add support and test for all spaces
    cpuspaces = all_spaces(FT; zelem = 10, context = cpu_context)
    gpuspaces = all_spaces(FT; zelem = 10, context = gpu_context)

    cpuspace = cpuspaces[end] # ExtrudedFiniteDifferenceSpace
    gpuspace = gpuspaces[end] # ExtrudedFiniteDifferenceSpace

    # Test that all geometries match with CPU version:
    @test compare(cpuspace, gpuspace, :center_local_geometry)
    @test compare(cpuspace, gpuspace, :face_local_geometry)
    @test compare(cpuspace, gpuspace, :center_ghost_geometry)
    @test compare(cpuspace, gpuspace, :face_ghost_geometry)

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
    cpuspace = cpuspaces[4] # SpectralElementSpace2D
    gpuspace = gpuspaces[4] # SpectralElementSpace2D

    # Test that all geometries match with CPU version:
    @test compare(cpuspace, gpuspace, :local_geometry)
    @test compare(cpuspace, gpuspace, :ghost_geometry)

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
